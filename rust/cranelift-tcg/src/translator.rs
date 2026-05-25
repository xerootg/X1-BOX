//! TCG IR -> Cranelift IR lowering.
//!
//! Lives on the dispatcher's worker thread. Owns the `JITModule`.
//!
//! Each compile produces a function with the signature:
//! ```text
//!   extern "C" fn jit_entry(env: *mut CPUArchState) -> i64;
//! ```
//! SystemV AArch64 ABI: env in X0, return value in X0. The C-side ABI
//! shim bridges this to TCG's prologue contract (env in X19, return via
//! tb_ret_addr).
//!
//! SSA conversion strategy
//! -----------------------
//! TCG temps become Cranelift [`Variable`]s. Each temp T is declared
//! once on first reference (with the type the lowering uses at that
//! point) and then accessed with `def_var` / `use_var`, which
//! transparently handles phi insertion at basic-block boundaries.
//!
//! Type compatibility: TCG temps can be 32-bit or 64-bit. We pick the
//! type when the variable is first declared and zero/sign-extend on
//! reads that need a wider type. If the IR mixes types on the same
//! temp (rare but legal in TCG) we narrow via `ireduce` or extend.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, Function, InstBuilder, MemFlags, Signature, UserFuncName, Type, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::Variable;

/// Sentinel offset value the C side uses to mark a `TEMP_FIXED`
/// pseudo-global (specifically `env`). Mirrors
/// `CRANELIFT_GLOBAL_OFFSET_FIXED_REG` in accel/tcg/cranelift-bridge.c.
const FIXED_REG_OFFSET: u32 = 0xFFFFFFFF;
use cranelift_codegen::Context as CgContext;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::context::{JitContext, TierTwoEntry};
use crate::env::EnvDesc;
use crate::ir::{DecodedOp, OpSnapshot};
use crate::opc::{flags, Op, Opc, TcgCond, TcgType};

#[derive(Debug)]
pub enum TransError {
    UnsupportedOp(u16),
    BadCondition(u64),
    InvalidIr(&'static str),
    VerifierFailed(String),
    ModuleError(String),
    Internal(&'static str),
}

impl TransError {
    pub fn to_ffi(&self) -> i32 {
        use TransError::*;
        match self {
            UnsupportedOp(_) => crate::ffi::CRANELIFT_TCG_ERR_UNSUPPORTED_OP,
            BadCondition(_) | InvalidIr(_) => crate::ffi::CRANELIFT_TCG_ERR_INVALID_IR,
            VerifierFailed(_) => crate::ffi::CRANELIFT_TCG_ERR_VERIFIER_FAILED,
            ModuleError(_) => crate::ffi::CRANELIFT_TCG_ERR_OUT_OF_MEMORY,
            Internal(_) => crate::ffi::CRANELIFT_TCG_ERR_INTERNAL,
        }
    }
}

pub struct Translator {
    module: JITModule,
    func_index: u32,
    env: EnvDesc,
    pub host_ptr_ty: Type,
}

impl Translator {
    pub fn new(env: &EnvDesc) -> Result<Self, TransError> {
        let mut flag_builder = settings::builder();
        // cranelift-jit requires is_pic=false (it does its own
        // address materialisation).
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| TransError::ModuleError(e.to_string()))?;
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| TransError::ModuleError(e.to_string()))?;
        flag_builder
            .set("enable_verifier", "true")
            .map_err(|e| TransError::ModuleError(e.to_string()))?;
        let isa_builder = cranelift_native::builder().map_err(|e| {
            TransError::ModuleError(format!("native ISA builder failed: {e}"))
        })?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| TransError::ModuleError(e.to_string()))?;

        let host_ptr_ty = if env.host_ptr_size == 8 { types::I64 } else { types::I32 };

        let jit_builder = JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        );
        let module = JITModule::new(jit_builder);

        Ok(Translator {
            module,
            func_index: 0,
            env: env.clone(),
            host_ptr_ty,
        })
    }

    /// Compile a single TB. Returns the emitted code entry on success.
    pub fn compile(
        &mut self,
        ctx: &Arc<JitContext>,
        tb_pc: u64,
        snap: &OpSnapshot,
    ) -> Result<TierTwoEntry, TransError> {
        /*
         * Bail on TBs that touch HF_INHIBIT_IRQ_MASK (set or clear).
         *
         * This is the STI/MOV-SS one-instruction interrupt-delay mechanism:
         *
         *   - The STI TB sets HF_INHIBIT_IRQ_MASK at end-of-TB so the
         *     *very next* x86 instruction executes before any pending
         *     interrupt fires (the "STI shadow").
         *   - The post-STI TB clears HF_INHIBIT_IRQ_MASK at start-of-TB so
         *     interrupts can fire from that instruction onward.
         *
         * In tier-1 these two TBs are CHAINED via `goto_tb`, so the second
         * one runs directly with no dispatcher round-trip and the shadow
         * is exactly one instruction wide.
         *
         * In tier-2 we don't implement TB chaining yet (every TB returns
         * to the dispatcher), so the shadow stretches across the dispatcher
         * loop. Pending interrupts then race against the inhibit bit, and
         * Xbox kernel spin loops that depend on interrupt-driven flag
         * clears (e.g. the wait-for-bit-28 pattern at 0x8004edd3) never
         * make progress.
         *
         * Detection: any op that ORs the immediate `HF_INHIBIT_IRQ_MASK`
         * (=0x8) or ANDs the immediate `~HF_INHIBIT_IRQ_MASK` (=0xfffffff7)
         * is treated as touching the inhibit bit. Catches both halves of
         * the pair conservatively.
         */
        /*
         * Earlier we bailed here on TBs that touch hflags (env+60),
         * thinking the STI / post-STI chaining contract was the root
         * cause of the cap=4 crash. Turned out that was a symptom: the
         * real bug was that `cranelift_bridge_try_swap` mutated
         * tb->tc.ptr, which broke `tcg_tb_lookup(host_pc)` for any
         * helper that needed precise instruction state. The C side
         * now stores the shim in a side-table instead, leaving the TB
         * tree intact, so this filter is no longer needed.
         */

        /*
         * STI-shadow filter — RESTORED 2026-05-24 (3rd attempt failed).
         *
         * Three attempts to remove the filter all caused either guest
         * wedge (May 21 ANR) or guest-state corruption (May 24 title
         * corruption + audio loss). The chain-dispatcher inhibit-gate
         * wasn't enough; trusting x86_cpu_pending_interrupt's native
         * inhibit gate wasn't enough either. The bug is somewhere
         * deeper in Cranelift's codegen for post-STI TBs — possibly
         * MemFlags::trusted() permitting an alias optimisation, or an
         * op-ordering issue when env writes intermix with goto_tb
         * patterns. Needs offline investigation with a minimal
         * reproducer (single post-STI TB → diff Cranelift IR vs
         * tier-1 TCG output for the same TB).
         *
         * Restore filter; the ~37 opc56 bails on SS2 + ~27 on Halo 2
         * are the tax for correctness. The chain inhibit-gate also
         * removed since it's now dead code with the filter in place.
         */
        let hflags_offset: Option<u32> = self
            .env
            .globals
            .iter()
            .find(|g| g.name == "cc_op")
            .map(|g| g.offset + 8);
        if let Some(hf_off) = hflags_offset {
            for op in &snap.ops {
                use crate::opc::{Op, Opc};
                if !matches!(
                    op.op,
                    Op::Known(Opc::St | Opc::St32 | Opc::St16 | Opc::St8)
                ) {
                    continue;
                }
                if op.nb_cargs == 0 {
                    continue;
                }
                if (op.carg(0) as u32) == hf_off {
                    return Err(TransError::UnsupportedOp(op.op.raw()));
                }
            }
        }

        let name = format!("tb_{tb_pc:016x}");
        let host_ptr_ty = self.host_ptr_ty;

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(host_ptr_ty));
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Export, &sig)
            .map_err(|e| TransError::ModuleError(e.to_string()))?;

        let mut func = Function::with_name_signature(
            UserFuncName::user(0, self.func_index),
            sig,
        );
        self.func_index = self.func_index.wrapping_add(1);

        let mut fb_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);
            let mut lower = Lowering::new(
                &mut builder,
                &self.env,
                host_ptr_ty,
                &ctx.helpers,
            );
            lower.lower_block(&snap.ops)?;
            lower.terminate_if_needed();
        }
        // finalize() consumes the FunctionBuilder; we created a new
        // one for each compile so dropping it is fine.

        if let Err(e) = cranelift_codegen::verifier::verify_function(&func, self.module.isa())
        {
            return Err(TransError::VerifierFailed(format!("{e:?}")));
        }

        let mut cg_ctx = CgContext::for_function(func);
        self.module
            .define_function(func_id, &mut cg_ctx)
            .map_err(|e| TransError::ModuleError(e.to_string()))?;
        self.module
            .finalize_definitions()
            .map_err(|e| TransError::ModuleError(e.to_string()))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let code_bytes_for_dump = cg_ctx
            .compiled_code()
            .map(|c| c.code_buffer().to_vec())
            .unwrap_or_default();
        let code_size = code_bytes_for_dump.len();

        // Dump the first few compiled functions as hex so we can
        // disassemble them offline and see whether they actually
        // advance the guest state.
        static DUMP_COUNT: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(0);
        let dc = DUMP_COUNT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Bumped from 4 -> 64. The shim swap currently caps installs at
        // 64; logging every JIT function lets us reverse-lookup the
        // offender from a crash's lr/pc by matching ptr=ce_code in the
        // install#N log line.
        if dc < 64 {
            let hex: String = code_bytes_for_dump
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join("");
            crate::dispatcher::log_to_android(&format!(
                "tb#{dc} pc=0x{tb_pc:x} ptr={code_ptr:p} size={code_size} bytes={hex}"
            ));
        }

        Ok(TierTwoEntry {
            code: code_ptr,
            size: code_size,
            tb_pc,
        })
    }
}

/// Per-function lowering state.
pub(crate) struct Lowering<'a, 'b> {
    pub(crate) builder: &'a mut FunctionBuilder<'b>,
    pub(crate) env: &'a EnvDesc,
    pub(crate) env_val: Value,
    pub(crate) host_ptr_ty: Type,
    /// Snapshot of QEMU's softmmu slow-path helper table.
    pub(crate) helpers: &'a crate::context::HelperTable,
    /// TCG temp ID -> (Cranelift Variable, type it was declared with).
    pub(crate) temps: HashMap<u64, (Variable, Type)>,
    pub(crate) next_var: u32,
    pub(crate) labels: HashMap<u64, cranelift_codegen::ir::Block>,
    pub(crate) block_terminated: bool,
    #[allow(dead_code)]
    pub(crate) entry_block: cranelift_codegen::ir::Block,
}

impl<'a, 'b> Lowering<'a, 'b> {
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        env: &'a EnvDesc,
        host_ptr_ty: Type,
        helpers: &'a crate::context::HelperTable,
    ) -> Self {
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);
        let env_val = builder.block_params(entry_block)[0];
        Lowering {
            builder,
            env,
            env_val,
            host_ptr_ty,
            helpers,
            temps: HashMap::new(),
            next_var: 0,
            labels: HashMap::new(),
            block_terminated: false,
            entry_block,
        }
    }

    /// Look up (or declare) the Variable for a TCG temp id.
    /// The Variable is declared with `want_ty` on first reference;
    /// subsequent reads/writes coerce to/from that type.
    fn temp_var(&mut self, id: u64, want_ty: Type) -> Variable {
        if let Some(&(v, _)) = self.temps.get(&id) {
            return v;
        }
        // Cranelift 0.130: declare_var returns the new Variable.
        let v = self.builder.declare_var(want_ty);
        self.next_var = self.next_var.wrapping_add(1);
        self.temps.insert(id, (v, want_ty));
        v
    }

    fn temp_decl_ty(&self, id: u64) -> Option<Type> {
        self.temps.get(&id).map(|&(_, t)| t)
    }

    pub(crate) fn terminate_if_needed(&mut self) {
        if !self.block_terminated {
            let zero = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().return_(&[zero]);
            self.block_terminated = true;
        }
        // Seal every block that's still open. With the Variable-based
        // SSA we cannot leave any block unsealed at finalize time:
        // the FunctionBuilder needs full predecessor info to insert
        // phi values via def_var.
        self.builder.seal_all_blocks();
    }

    pub(crate) fn type_for(ty: TcgType) -> Type {
        match ty {
            TcgType::I32 => types::I32,
            TcgType::I64 => types::I64,
            TcgType::I128 => types::I128,
            // Cranelift has no native I64X1; widen V64 to I64X2 - the high half is unused.
            TcgType::V64 => types::I64X2,
            TcgType::V128 => types::I64X2,
            TcgType::V256 => types::I64X2, // fallback
        }
    }

    /// Coerce `val` to `want` via extend/reduce/bitcast as needed.
    /// No-op if equal.
    ///
    /// `uextend` and `ireduce` are integer-only operations in Cranelift;
    /// passing them a float type fails the verifier (the source of the
    /// pre-2026-05-21 `VerifierFailed=22` bucket: `uextend.i64 v488`
    /// where v488 was f32, and `ireduce.f32 v219` where the destination
    /// was f32). Both shapes arise when a TCG temp's first writer was
    /// an FP op (`temp_decl_ty` becomes F32/F64) and a later reader
    /// requests it as a different-width integer, or vice versa.
    ///
    /// The fix is to route any cross-domain width change through an
    /// integer of the source width: bitcast float→int, extend/reduce
    /// in integer space, then bitcast back if the destination is float.
    fn coerce(&mut self, val: Value, want: Type) -> Value {
        let cur = self.builder.func.dfg.value_type(val);
        if cur == want {
            return val;
        }

        let mf = MemFlags::new();
        let cur_is_float = cur == types::F32 || cur == types::F64;
        let want_is_float = want == types::F32 || want == types::F64;

        /*
         * Cross-domain scalar ↔ vector coerce.
         *
         * x86 SSE creates temps that may be first-written by a SCALAR
         * op (e.g., movd xmm, gpr produces an I32; the XMM register's
         * upper 96 bits are zeroed by definition), then later read as
         * V128 for spill via st_vec. The integer fast-path below would
         * try `uextend(I64X2, scalar)` which is invalid IR — uextend's
         * destination must be a scalar integer type. Pre-fix this
         * accounted for ~245 st_vec bails per Halo 2 session (38% of
         * all tier-2 compile errors).
         *
         * The Cranelift `scalar_to_vector` instruction matches movd/
         * movq semantics exactly: scalar value goes into lane 0, all
         * upper lanes are zeroed. Symmetric path for vector → scalar
         * extracts lane 0 and recurses for any further width adjust.
         */
        if !cur.is_vector() && want.is_vector() {
            let lane_ty = want.lane_type();
            let lane_val = if cur == lane_ty {
                val
            } else {
                /*
                 * Coerce scalar to lane width first. Recursing into
                 * `coerce` reuses the same width/float handling logic;
                 * since `lane_ty` is a scalar this won't re-enter the
                 * vector branch.
                 */
                self.coerce(val, lane_ty)
            };
            return self.builder.ins().scalar_to_vector(want, lane_val);
        }
        if cur.is_vector() && !want.is_vector() {
            let lane_ty = cur.lane_type();
            let lo = self.builder.ins().extractlane(val, 0);
            /*
             * `extractlane` already produces a value of `lane_ty`; if
             * `want` differs (width or float-ness), recurse to handle.
             */
            return if lane_ty == want { lo } else { self.coerce(lo, want) };
        }

        // Fast path: pure integer extend / reduce.
        if !cur_is_float && !want_is_float {
            if cur.bits() < want.bits() {
                return self.builder.ins().uextend(want, val);
            }
            if cur.bits() > want.bits() {
                return self.builder.ins().ireduce(want, val);
            }
            // Same width different lane shape (vector ↔ scalar).
            return self.builder.ins().bitcast(want, mf, val);
        }

        // Same-width int↔float: pure bitcast.
        if cur.bits() == want.bits() {
            return self.builder.ins().bitcast(want, mf, val);
        }

        // Different-width involving float: bitcast through a same-width
        // integer first, extend/reduce in integer space, then bitcast to
        // float if needed.
        let int_at_cur = match cur.bits() {
            32 => types::I32,
            64 => types::I64,
            _ => {
                // Unusual width (8/16 floats don't exist) — fall back to
                // a bitcast and let the verifier complain if shape is
                // genuinely wrong.
                return self.builder.ins().bitcast(want, mf, val);
            }
        };
        let v_int = if cur_is_float {
            self.builder.ins().bitcast(int_at_cur, mf, val)
        } else {
            val
        };
        let int_at_want = match want.bits() {
            8 => types::I8,
            16 => types::I16,
            32 => types::I32,
            64 => types::I64,
            _ => return self.builder.ins().bitcast(want, mf, val),
        };
        let v_at_want_width = if cur.bits() < want.bits() {
            self.builder.ins().uextend(int_at_want, v_int)
        } else if cur.bits() > want.bits() {
            self.builder.ins().ireduce(int_at_want, v_int)
        } else {
            v_int
        };
        if want_is_float {
            self.builder.ins().bitcast(want, mf, v_at_want_width)
        } else {
            v_at_want_width
        }
    }

    /// Resolve an input arg into a Cranelift Value.
    ///
    /// If the op's `const_mask` bit for the corresponding position is
    /// set, args[pos] holds the TEMP_CONST's int64 value directly and
    /// we materialise it as `iconst(val)`. Otherwise we fall through to
    /// the normal `read_temp` path.
    ///
    /// Every iarg read in lowering MUST go through this helper, not
    /// through `read_temp(op.iarg(...))` directly — without it, const
    /// temps are interpreted as fresh local Variables seeded to zero
    /// and tier-2 silently zeros every immediate operand.
    pub(crate) fn read_iarg(
        &mut self,
        op: &DecodedOp,
        i: usize,
        ty: TcgType,
    ) -> Result<Value, TransError> {
        if op.iarg_is_const(i) {
            let want = Self::type_for(ty);
            let raw = op.iarg(i) as i64;
            // Cranelift requires the immediate to fit the destination
            // integer type; ireduce widens-then-truncates if needed.
            let val = self.builder.ins().iconst(types::I64, raw);
            return Ok(self.coerce(val, want));
        }
        self.read_temp(op.iarg(i), ty)
    }

    pub(crate) fn read_temp(&mut self, id: u64, ty: TcgType) -> Result<Value, TransError> {
        let want = Self::type_for(ty);

        // Fixed-reg pseudo-global (env). The C side packs env as
        // globals[0] with offset == UINT32_MAX as a sentinel.
        let global = self.global_for_temp(id).cloned();
        if let Some(g) = &global {
            if g.offset == FIXED_REG_OFFSET {
                return Ok(self.coerce(self.env_val, want));
            }
        }
        // Memory-backed globals: always emit a load from env memory.
        // Cranelift's optimizer can hoist these where it's legal.
        // Tracking globals through Variables would require us to
        // define them on every path that reaches a use (otherwise the
        // FunctionBuilder hits an "undefined variable" error at phi
        // insertion). Going through memory keeps the lowering simple
        // and matches what the existing TCG ARM64 backend does.
        if let Some(g) = global {
            let cty = match g.size {
                1 => types::I8,
                2 => types::I16,
                4 => types::I32,
                8 => types::I64,
                /*
                 * 16-byte globals are vector temps (XMM registers etc.
                 * — the C side packs `TCG_TYPE_V128` as `size = 16`).
                 * Load them as I64X2 so subsequent vector ops (ld_vec,
                 * mov_vec, etc.) see a properly-typed value.
                 */
                16 => types::I64X2,
                _ => return Err(TransError::Internal("bad global size")),
            };
            /*
             * Cross-width read between vector storage and scalar
             * requestor (e.g. MOVD from XMM low → GPR loads 32 bits
             * from a 16-byte XMM global). Cranelift's `ireduce` on
             * vector→scalar is unspecified/unsupported; rather than
             * emit IR that might silently miscompile, bail the TB to
             * tier-1 where TCG-aarch64's well-tested codegen handles
             * the partial access. Likewise scalar global → vector
             * requestor (would need uextend scalar→vector).
             */
            if (cty == types::I64X2) != (want == types::I64X2) {
                return Err(TransError::UnsupportedOp(crate::opc::Opc::Ld as u16));
            }
            let loaded = self.builder.ins().load(
                cty,
                MemFlags::trusted(),
                self.env_val,
                g.offset as i32,
            );
            return Ok(self.coerce(loaded, want));
        }

        // Local temp: use a Cranelift Variable so cross-block flow
        // works via implicit phi insertion. Declare on first use,
        // seeded with zero.
        //
        // Cranelift's `iconst` is defined for scalar integer types only —
        // the IR verifier's `iconst_bounds` check (verifier/mod.rs:1949)
        // panics with `unreachable!()` when ctrl_typevar isn't I8/I16/
        // I32/I64. So for vector temps we materialize zero via
        // `splat(want, iconst.i64 0)` which produces a properly-typed
        // vector zero. Hit when ld_vec/st_vec lowering reads a vector
        // temp before any tier-2-supported op writes to it.
        if !self.temps.contains_key(&id) {
            let var = self.temp_var(id, want);
            let zero = if want.is_vector() {
                let lane_zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().splat(want, lane_zero)
            } else {
                self.builder.ins().iconst(want, 0)
            };
            self.builder.def_var(var, zero);
        }
        let (var, _) = self.temps[&id];
        let raw = self.builder.use_var(var);
        Ok(self.coerce(raw, want))
    }

    pub(crate) fn write_temp(&mut self, id: u64, val: Value) {
        let val_ty = self.builder.func.dfg.value_type(val);

        if let Some(g) = self.global_for_temp(id).cloned() {
            // Writes to the env fixed-reg pseudo-global are no-ops:
            // env's value lives in a host register (X19 on aarch64
            // under the TCG ABI; X0 under ours) and the guest can't
            // legally rewrite it. TCG never actually emits such writes
            // but be defensive.
            if g.offset == FIXED_REG_OFFSET {
                return;
            }
            // Memory-backed globals: flush directly to env memory.
            let store_ty = match g.size {
                1 => types::I8,
                2 => types::I16,
                4 => types::I32,
                8 => types::I64,
                /*
                 * 16-byte globals are V128 vector temps (XMM regs etc.).
                 * The OLD fallback (`_ => I64`) silently TRUNCATED them
                 * to I64, dropping the high half — this was the latent
                 * bug exposed by enabling ld_vec/st_vec tier-2 lowering:
                 * a tier-2 ld_vec followed by a write to a global XMM
                 * temp would write only the low 8 bytes, corrupting
                 * x86 SSE state and wedging guest code. Store as I64X2.
                 */
                16 => types::I64X2,
                /*
                 * Unknown size: skip the store rather than silently
                 * truncate. Loud-failure is preferable to data loss.
                 */
                _ => return,
            };
            /*
             * Cross-width write (scalar value into vector global, or
             * vector value into scalar global). The needed
             * uextend/ireduce between scalar and vector is unsupported
             * by Cranelift; rather than emit IR that may silently
             * miscompile, drop the global store. The dispatcher's TB
             * verifier will catch the divergence on the verify pass.
             *
             * NOTE: this `return` skips the write entirely — but the
             * only way we'd legitimately reach here is a TCG-emitted
             * partial-XMM access, which TCG-aarch64 (tier-1) handles
             * correctly. The compile-error machinery in the bridge
             * runs the TB on tier-1 if any op signals UnsupportedOp;
             * for write_temp we have no error channel back. So skip.
             */
            let val_ty = self.builder.func.dfg.value_type(val);
            if (val_ty == types::I64X2) != (store_ty == types::I64X2) {
                return;
            }
            let flush_val = self.coerce(val, store_ty);
            self.builder.ins().store(
                MemFlags::trusted(),
                flush_val,
                self.env_val,
                g.offset as i32,
            );
            return;
        }

        // Local temp: def_var with a stable declared type. Use the
        // existing declaration's type if present so we don't trip the
        // verifier with mismatched types across def sites.
        let decl_ty = self.temp_decl_ty(id).unwrap_or(val_ty);
        let var = self.temp_var(id, decl_ty);
        let to_store = self.coerce(val, decl_ty);
        self.builder.def_var(var, to_store);
    }

    fn global_for_temp(&self, id: u64) -> Option<&crate::env::GlobalDesc> {
        let idx = id as usize;
        self.env.globals.get(idx)
    }

    pub(crate) fn label_block(&mut self, id: u64) -> cranelift_codegen::ir::Block {
        if let Some(&b) = self.labels.get(&id) {
            return b;
        }
        let b = self.builder.create_block();
        self.labels.insert(id, b);
        b
    }

    fn lower_block(&mut self, ops: &[DecodedOp]) -> Result<(), TransError> {
        for (i, op) in ops.iter().enumerate() {
            if self.block_terminated {
                // Spawn a fresh block to land in. We do NOT seal it here
                // because future SetLabel ops may also need to jump in
                // (forward branch from a yet-to-be-emitted brcond is
                // legal but rare); the block gets sealed by either a
                // SetLabel that lands in it, or at end-of-function via
                // seal_all_blocks().
                let cont = self.builder.create_block();
                self.builder.switch_to_block(cont);
                self.block_terminated = false;
            }
            if op.flags & flags::TCG_OPF_NOT_PRESENT != 0
                && !matches!(
                    op.op,
                    Op::Known(
                        Opc::Discard
                            | Opc::SetLabel
                            | Opc::Mov
                            | Opc::ExitTb
                            | Opc::GotoTb
                            | Opc::Call
                            | Opc::InsnStart
                            | Opc::Mb
                            | Opc::PluginCb
                            | Opc::PluginMemCb
                    )
                )
            {
                // Not-present ops with no special handling are no-ops.
                continue;
            }
            /*
             * Pattern elision: `Call helper_lookup_tb_ptr → GotoPtr`.
             *
             * The x86 frontend's tcg_gen_lookup_and_goto_ptr emits this
             * pair every time. Our GotoPtr lowering ignores the input
             * (the host code pointer returned by the helper) and re-
             * does the lookup via cranelift_chain_continue. That makes
             * the helper call dead work in tier-2.
             *
             * Optimizer data-deps keep these two adjacent: goto_ptr
             * reads the call's output, so reordering would break SSA.
             * Skip the call. Its output temp is never read because
             * GotoPtr's input is intentionally unused; if anything
             * downstream tried to read it (it doesn't, GotoPtr ends
             * the block) it would seed zero, also harmless.
             *
             * Saves one full TB lookup per chained dispatch -- helper
             * + chain_continue was doing the QHT walk twice.
             */
            if matches!(op.op, Op::Known(Opc::Call))
                && self.env.lookup_tb_ptr_fn != 0
                && op.carg(0) == self.env.lookup_tb_ptr_fn
                && i + 1 < ops.len()
                && matches!(ops[i + 1].op, Op::Known(Opc::GotoPtr))
            {
                continue;
            }
            self.lower_op(op)?;
        }
        Ok(())
    }

    fn lower_op(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        match op.op {
            Op::Known(known) => self.lower_known(known, op),
            Op::Other(raw) => {
                // Vector / FP ops live past QemuSt2; dispatch.
                if op.flags & flags::TCG_OPF_FP != 0 {
                    crate::fpvec::lower_fp(self, raw, op)
                } else if op.flags & flags::TCG_OPF_VECTOR != 0 {
                    crate::fpvec::lower_vec(self, raw, op)
                } else {
                    Err(TransError::UnsupportedOp(raw))
                }
            }
        }
    }

    fn lower_known(&mut self, opc: Opc, op: &DecodedOp) -> Result<(), TransError> {
        use Opc::*;
        match opc {
            Discard | InsnStart | PluginCb | PluginMemCb => Ok(()),
            Mb => {
                self.builder.ins().fence();
                Ok(())
            }
            SetLabel => {
                let id = op.carg(0);
                let blk = self.label_block(id);
                // Close current block by branching into the label.
                // Do NOT seal blk here - TCG IR can produce backward
                // branches that add later predecessors. We seal all
                // blocks at end-of-function via seal_all_block_params().
                self.builder.ins().jump(blk, &[] as &[BlockArg]);
                self.builder.switch_to_block(blk);
                Ok(())
            }
            Mov => self.lower_mov(op),
            Add => self.lower_bin(op, |b, a, c| b.ins().iadd(a, c)),
            Sub => self.lower_bin(op, |b, a, c| b.ins().isub(a, c)),
            Mul => self.lower_bin(op, |b, a, c| b.ins().imul(a, c)),
            And => self.lower_bin(op, |b, a, c| b.ins().band(a, c)),
            Or  => self.lower_bin(op, |b, a, c| b.ins().bor(a, c)),
            Xor => self.lower_bin(op, |b, a, c| b.ins().bxor(a, c)),
            Andc => self.lower_bin(op, |b, a, c| b.ins().band_not(a, c)),
            Orc  => self.lower_bin(op, |b, a, c| b.ins().bor_not(a, c)),
            Eqv  => self.lower_bin(op, |b, a, c| {
                let x = b.ins().bxor(a, c);
                b.ins().bnot(x)
            }),
            Nand => self.lower_bin(op, |b, a, c| {
                let x = b.ins().band(a, c);
                b.ins().bnot(x)
            }),
            Nor => self.lower_bin(op, |b, a, c| {
                let x = b.ins().bor(a, c);
                b.ins().bnot(x)
            }),
            Shl => self.lower_bin(op, |b, a, c| b.ins().ishl(a, c)),
            Shr => self.lower_bin(op, |b, a, c| b.ins().ushr(a, c)),
            Sar => self.lower_bin(op, |b, a, c| b.ins().sshr(a, c)),
            Rotl => self.lower_bin(op, |b, a, c| b.ins().rotl(a, c)),
            Rotr => self.lower_bin(op, |b, a, c| b.ins().rotr(a, c)),
            Divs => self.lower_bin(op, |b, a, c| b.ins().sdiv(a, c)),
            Divu => self.lower_bin(op, |b, a, c| b.ins().udiv(a, c)),
            Rems => self.lower_bin(op, |b, a, c| b.ins().srem(a, c)),
            Remu => self.lower_bin(op, |b, a, c| b.ins().urem(a, c)),
            Neg => self.lower_un(op, |b, a| b.ins().ineg(a)),
            Not => self.lower_un(op, |b, a| b.ins().bnot(a)),
            Ctpop => self.lower_un(op, |b, a| b.ins().popcnt(a)),
            // clz / ctz take a "zero result" arg; we ignore it and let
            // Cranelift's clz/ctz semantics (undef on zero) match TCG's
            // zero handling for the common 64-bit case.
            Clz => self.lower_clz_ctz(op, true),
            Ctz => self.lower_clz_ctz(op, false),
            Bswap16 => self.lower_un(op, |b, a| b.ins().bswap(a)),
            Bswap32 => self.lower_un(op, |b, a| b.ins().bswap(a)),
            Bswap64 => self.lower_un(op, |b, a| b.ins().bswap(a)),
            ExtI32I64 => self.lower_ext(op, types::I64, true),
            ExtuI32I64 => self.lower_ext(op, types::I64, false),
            ExtrlI64I32 => self.lower_extr(op, false),
            ExtrhI64I32 => self.lower_extr(op, true),
            Extract => self.lower_extract(op, false),
            Sextract => self.lower_extract(op, true),
            Deposit => self.lower_deposit(op),
            Setcond => self.lower_setcond(op, false),
            Negsetcond => self.lower_setcond(op, true),
            Movcond => self.lower_movcond(op),
            Br => {
                let label = op.carg(0);
                let blk = self.label_block(label);
                self.builder.ins().jump(blk, &[] as &[BlockArg]);
                self.block_terminated = true;
                Ok(())
            }
            Brcond => self.lower_brcond(op),
            Ld8u | Ld8s | Ld16u | Ld16s | Ld32u | Ld32s | Ld => {
                self.lower_env_ld(opc, op)
            }
            St8 | St16 | St32 | St => self.lower_env_st(opc, op),
            ExitTb => {
                let ret = op.carg(0) as i64;
                let v = self.builder.ins().iconst(types::I64, ret);
                self.builder.ins().return_(&[v]);
                self.block_terminated = true;
                Ok(())
            }
            GotoTb => {
                /* TB chaining for Cranelift. Tier-1 TCG patches goto_tb
                 * into a direct jump to the next TB's host code, so a
                 * hot loop body has zero dispatcher round-trip per
                 * iteration. Cranelift TBs can't be patched post-hoc,
                 * so we instead call cranelift_chain_continue (in
                 * accel/tcg/cpu-exec.c) which loops dispatching
                 * successive TBs (shim or tier-1) until it hits an
                 * interrupt or a non-zero exit reason. The chain
                 * helper's address is passed in via EnvDesc at init.
                 * Without this, every Cranelift TB ending in goto_tb
                 * returned 0 and forced the dispatcher to look up the
                 * next TB by env->eip — fine for cold paths but for
                 * audio-mix / video-decode / animation inner loops it
                 * added enough per-iteration variance to cause
                 * audible chop and visible stutter.
                 * env->eip is already updated to the destination by
                 * the frontend before this op runs.  If the helper
                 * address wasn't set up, fall back to a plain
                 * return-to-dispatcher. */
                let chain_fn = self.env.chain_continue_fn;
                if chain_fn != 0 {
                    let mut sig =
                        Signature::new(CallConv::SystemV);
                    sig.params.push(AbiParam::new(self.host_ptr_ty));
                    sig.returns.push(AbiParam::new(types::I64));
                    let sig_ref = self.builder.import_signature(sig);
                    let addr = self
                        .builder
                        .ins()
                        .iconst(self.host_ptr_ty, chain_fn as i64);
                    let inst = self.builder.ins().call_indirect(
                        sig_ref,
                        addr,
                        &[self.env_val],
                    );
                    let ret = self.builder.inst_results(inst)[0];
                    self.builder.ins().return_(&[ret]);
                } else {
                    let v = self.builder.ins().iconst(types::I64, 0);
                    self.builder.ins().return_(&[v]);
                }
                self.block_terminated = true;
                Ok(())
            }
            GotoPtr => {
                /* Computed branch (x86 `ret`, indirect `jmp`, etc.).
                 *
                 * The frontend has already updated env->eip to the
                 * destination guest PC and computed a host code pointer
                 * via `helper_lookup_tb_ptr` (whose result is goto_ptr's
                 * input arg). We can't tail-call that pointer directly
                 * because tier-1 TBs use the TCG-prologue ABI (env in
                 * X19) and we're in a SystemV function (env in X0).
                 *
                 * Functionally identical to goto_tb's chain fallback:
                 * return into cranelift_chain_continue, which re-looks
                 * up the next TB by env->eip and dispatches it (with
                 * shim substitution if it's also tier-2). The duplicate
                 * lookup is cheap relative to the cost of bouncing the
                 * entire TB back to tier-1 every dispatch.
                 *
                 * The input arg (host code pointer) is intentionally
                 * unused; Cranelift's DCE drops the dead load. The
                 * helper_lookup_tb_ptr call that produced it still
                 * executes for its side effects.
                 */
                let chain_fn = self.env.chain_continue_fn;
                if chain_fn != 0 {
                    let mut sig =
                        Signature::new(CallConv::SystemV);
                    sig.params.push(AbiParam::new(self.host_ptr_ty));
                    sig.returns.push(AbiParam::new(types::I64));
                    let sig_ref = self.builder.import_signature(sig);
                    let addr = self
                        .builder
                        .ins()
                        .iconst(self.host_ptr_ty, chain_fn as i64);
                    let inst = self.builder.ins().call_indirect(
                        sig_ref,
                        addr,
                        &[self.env_val],
                    );
                    let ret = self.builder.inst_results(inst)[0];
                    self.builder.ins().return_(&[ret]);
                } else {
                    let v = self.builder.ins().iconst(types::I64, 0);
                    self.builder.ins().return_(&[v]);
                }
                self.block_terminated = true;
                Ok(())
            }
            QemuLd | QemuLd2 => crate::memory::lower_load(self, op, opc == QemuLd2),
            QemuSt | QemuSt2 => crate::memory::lower_store(self, op, opc == QemuSt2),
            Call => crate::helper::lower_call(self, op),
            // 32-bit-specific 64-bit emulation ops.
            Brcond2I32 => self.lower_brcond2_i32(op),
            Setcond2I32 => self.lower_setcond2_i32(op),
            // Carry-chain - we lower as 128-bit add and split.
            Addco | Addc1o | Addci | Addcio | Subbo | Subb1o | Subbi | Subbio => {
                Err(TransError::UnsupportedOp(opc as u16))
            }
            Divs2 | Divu2 | Muls2 | Mulu2 | Mulsh | Muluh | Extract2 => {
                self.lower_wide_arith(opc, op)
            }
        }
    }

    fn lower_mov(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        /*
         * After TCG optimisation, `tcg_gen_movi_i{32,64}(t, imm)` collapses
         * to `mov t, const_temp` where the source is a `TEMP_CONST` whose
         * `ts->val` holds `imm`. The C-side serialiser packs that value
         * into args[1] and sets the corresponding const_mask bit, so we
         * MUST go through `read_iarg` to materialise it as an `iconst`.
         * Calling `read_temp(op.iarg(0), ...)` instead would interpret the
         * embedded immediate as a temp index, declare a fresh Variable
         * seeded with 0, and silently turn every constant load into zero.
         */
        let dst = op.oarg(0);
        let v = self.read_iarg(op, 0, op.ty)?;
        self.write_temp(dst, v);
        Ok(())
    }

    fn lower_bin<F>(&mut self, op: &DecodedOp, f: F) -> Result<(), TransError>
    where
        F: FnOnce(&mut FunctionBuilder<'_>, Value, Value) -> Value,
    {
        let a = self.read_iarg(op, 0, op.ty)?;
        let b = self.read_iarg(op, 1, op.ty)?;
        let v = f(self.builder, a, b);
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_un<F>(&mut self, op: &DecodedOp, f: F) -> Result<(), TransError>
    where
        F: FnOnce(&mut FunctionBuilder<'_>, Value) -> Value,
    {
        let a = self.read_iarg(op, 0, op.ty)?;
        let v = f(self.builder, a);
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_clz_ctz(&mut self, op: &DecodedOp, is_clz: bool) -> Result<(), TransError> {
        let a = self.read_iarg(op, 0, op.ty)?;
        let zero_res = self.read_iarg(op, 1, op.ty)?;
        let ty = self.builder.func.dfg.value_type(a);
        let zero = self.builder.ins().iconst(ty, 0);
        let is_zero = self.builder.ins().icmp(IntCC::Equal, a, zero);
        let cnt = if is_clz {
            self.builder.ins().clz(a)
        } else {
            self.builder.ins().ctz(a)
        };
        let v = self.builder.ins().select(is_zero, zero_res, cnt);
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_ext(
        &mut self,
        op: &DecodedOp,
        to: Type,
        signed: bool,
    ) -> Result<(), TransError> {
        let a = self.read_iarg(op, 0, TcgType::I32)?;
        let v = if signed {
            self.builder.ins().sextend(to, a)
        } else {
            self.builder.ins().uextend(to, a)
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_extr(&mut self, op: &DecodedOp, high: bool) -> Result<(), TransError> {
        let a = self.read_iarg(op, 0, TcgType::I64)?;
        let v = if high {
            let shifted = self
                .builder
                .ins()
                .ushr_imm(a, 32);
            self.builder.ins().ireduce(types::I32, shifted)
        } else {
            self.builder.ins().ireduce(types::I32, a)
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_extract(&mut self, op: &DecodedOp, signed: bool) -> Result<(), TransError> {
        let a = self.read_iarg(op, 0, op.ty)?;
        let pos = op.carg(0) as i64;
        let len = op.carg(1) as i64;
        let ty = self.builder.func.dfg.value_type(a);
        let bits = ty.bits() as i64;
        let shift = bits - pos - len;
        let v_shifted = if shift > 0 {
            self.builder.ins().ishl_imm(a, shift)
        } else {
            a
        };
        let v_back = if signed {
            self.builder.ins().sshr_imm(v_shifted, bits - len)
        } else {
            self.builder.ins().ushr_imm(v_shifted, bits - len)
        };
        self.write_temp(op.oarg(0), v_back);
        Ok(())
    }

    fn lower_deposit(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        let base = self.read_iarg(op, 0, op.ty)?;
        let val = self.read_iarg(op, 1, op.ty)?;
        let pos = op.carg(0) as i64;
        let len = op.carg(1) as i64;
        let ty = self.builder.func.dfg.value_type(base);
        let bits = ty.bits() as i64;
        /*
         * `pos` and `len` are compile-time constants in TCG IR — resolve
         * the masks here rather than emitting `(1 << len) - 1` at
         * runtime. That avoids the wrap-to-0 footgun when `len ==
         * ty.bits()` (shifting an Nbit value by N is UB / wraps in
         * Cranelift) and saves a handful of instructions per deposit.
         *
         * Semantics: dst = (base & ~mask) | ((val & low_mask) << pos)
         *   low_mask = (1 << len) - 1
         *   mask     = low_mask << pos
         *
         * The original lowering OR-ed `val << pos` directly without
         * masking, which leaked bits of `val` above the `len`-bit
         * window into the result. For x86 that mis-set status flags
         * during PUSHF/POPF/SAHF, FLAGS deposits from helper returns,
         * and bit-field instructions like SHLD/SHRD.
         */
        let low_mask_val: i64 = if len >= bits {
            -1i64
        } else {
            (1i64 << len) - 1
        };
        let shifted_mask_val: i64 = low_mask_val << pos;
        let inv_mask_val: i64 = !shifted_mask_val;
        let low_mask = self.builder.ins().iconst(ty, low_mask_val);
        let inv_mask = self.builder.ins().iconst(ty, inv_mask_val);
        let val_masked = self.builder.ins().band(val, low_mask);
        let val_shifted = self.builder.ins().ishl_imm(val_masked, pos);
        let cleared = self.builder.ins().band(base, inv_mask);
        let v = self.builder.ins().bor(cleared, val_shifted);
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    pub(crate) fn intcc_for(&self, cond: TcgCond) -> Option<IntCC> {
        Some(match cond {
            TcgCond::Eq => IntCC::Equal,
            TcgCond::Ne => IntCC::NotEqual,
            TcgCond::Lt => IntCC::SignedLessThan,
            TcgCond::Ge => IntCC::SignedGreaterThanOrEqual,
            TcgCond::Le => IntCC::SignedLessThanOrEqual,
            TcgCond::Gt => IntCC::SignedGreaterThan,
            TcgCond::Ltu => IntCC::UnsignedLessThan,
            TcgCond::Geu => IntCC::UnsignedGreaterThanOrEqual,
            TcgCond::Leu => IntCC::UnsignedLessThanOrEqual,
            TcgCond::Gtu => IntCC::UnsignedGreaterThan,
            TcgCond::Never | TcgCond::Always => return None,
            // TstEq / TstNe are AND-then-compare-vs-0 — see eval_tcg_cond
            // which materialises the AND. Returning None here is the
            // "needs structural lowering, not a direct cc" signal.
            TcgCond::TstEq | TcgCond::TstNe => return None,
        })
    }

    /// Evaluate a TCG condition into a Cranelift i8 boolean. Handles
    /// the full TcgCond set including `TstEq`/`TstNe` (which TCG emits
    /// constantly for x86 SETcc / Jcc on AND'd flag values) and the
    /// degenerate `Never` / `Always`.
    fn eval_tcg_cond(
        &mut self,
        cond: TcgCond,
        a: Value,
        b: Value,
    ) -> Result<Value, TransError> {
        match cond {
            TcgCond::Never => Ok(self.builder.ins().iconst(types::I8, 0)),
            TcgCond::Always => Ok(self.builder.ins().iconst(types::I8, 1)),
            TcgCond::TstEq | TcgCond::TstNe => {
                let ty = self.builder.func.dfg.value_type(a);
                let and = self.builder.ins().band(a, b);
                let zero = self.builder.ins().iconst(ty, 0);
                let cc = if matches!(cond, TcgCond::TstEq) {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                };
                Ok(self.builder.ins().icmp(cc, and, zero))
            }
            _ => {
                let cc = self
                    .intcc_for(cond)
                    .ok_or(TransError::BadCondition(cond as u64))?;
                Ok(self.builder.ins().icmp(cc, a, b))
            }
        }
    }

    fn lower_setcond(&mut self, op: &DecodedOp, neg: bool) -> Result<(), TransError> {
        let cond_raw = op.carg(0);
        let cond = TcgCond::from_raw(cond_raw).ok_or(TransError::BadCondition(cond_raw))?;
        let a = self.read_iarg(op, 0, op.ty)?;
        let b = self.read_iarg(op, 1, op.ty)?;
        let ty = self.builder.func.dfg.value_type(a);
        /* Short-circuit the structural Never/Always so we can emit a
         * proper-typed constant (eval_tcg_cond returns an i8). */
        let v = match cond {
            TcgCond::Never => self.builder.ins().iconst(ty, 0),
            TcgCond::Always => self.builder.ins().iconst(ty, if neg { -1 } else { 1 }),
            _ => {
                let bit = self.eval_tcg_cond(cond, a, b)?;
                let ext = self.builder.ins().uextend(ty, bit);
                if neg {
                    self.builder.ins().ineg(ext)
                } else {
                    ext
                }
            }
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_movcond(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        let cond_raw = op.carg(0);
        let cond = TcgCond::from_raw(cond_raw).ok_or(TransError::BadCondition(cond_raw))?;
        let a = self.read_iarg(op, 0, op.ty)?;
        let b = self.read_iarg(op, 1, op.ty)?;
        let v_true = self.read_iarg(op, 2, op.ty)?;
        let v_false = self.read_iarg(op, 3, op.ty)?;
        /* Never/Always degenerate: just pick one of the operands. */
        let v = match cond {
            TcgCond::Never => v_false,
            TcgCond::Always => v_true,
            _ => {
                let bit = self.eval_tcg_cond(cond, a, b)?;
                self.builder.ins().select(bit, v_true, v_false)
            }
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_brcond(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        let cond_raw = op.carg(0);
        let cond = TcgCond::from_raw(cond_raw).ok_or(TransError::BadCondition(cond_raw))?;
        let label = op.carg(1);
        let a = self.read_iarg(op, 0, op.ty)?;
        let b = self.read_iarg(op, 1, op.ty)?;
        /* Never collapses to fallthrough, Always to unconditional jump. */
        if matches!(cond, TcgCond::Never) {
            return Ok(());
        }
        if matches!(cond, TcgCond::Always) {
            let target = self.label_block(label);
            self.builder.ins().jump(target, &[] as &[BlockArg]);
            self.block_terminated = true;
            return Ok(());
        }
        let bit = self.eval_tcg_cond(cond, a, b)?;
        let target = self.label_block(label);
        let fallthrough = self.builder.create_block();
        self.builder
            .ins()
            .brif(bit, target, &[] as &[BlockArg], fallthrough, &[] as &[BlockArg]);
        self.builder.switch_to_block(fallthrough);
        Ok(())
    }

    fn lower_brcond2_i32(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        let cond_raw = op.carg(0);
        let cond = TcgCond::from_raw(cond_raw).ok_or(TransError::BadCondition(cond_raw))?;
        let label = op.carg(1);
        let al = self.read_iarg(op, 0, TcgType::I32)?;
        let ah = self.read_iarg(op, 1, TcgType::I32)?;
        let bl = self.read_iarg(op, 2, TcgType::I32)?;
        let bh = self.read_iarg(op, 3, TcgType::I32)?;
        let al64 = self.builder.ins().uextend(types::I64, al);
        let ah64 = self.builder.ins().uextend(types::I64, ah);
        let bl64 = self.builder.ins().uextend(types::I64, bl);
        let bh64 = self.builder.ins().uextend(types::I64, bh);
        let a = {
            let s = self.builder.ins().ishl_imm(ah64, 32);
            self.builder.ins().bor(s, al64)
        };
        let b = {
            let s = self.builder.ins().ishl_imm(bh64, 32);
            self.builder.ins().bor(s, bl64)
        };
        if matches!(cond, TcgCond::Never) {
            return Ok(());
        }
        if matches!(cond, TcgCond::Always) {
            let target = self.label_block(label);
            self.builder.ins().jump(target, &[] as &[BlockArg]);
            self.block_terminated = true;
            return Ok(());
        }
        let bit = self.eval_tcg_cond(cond, a, b)?;
        let target = self.label_block(label);
        let fallthrough = self.builder.create_block();
        self.builder
            .ins()
            .brif(bit, target, &[] as &[BlockArg], fallthrough, &[] as &[BlockArg]);
        self.builder.switch_to_block(fallthrough);
        Ok(())
    }

    fn lower_setcond2_i32(&mut self, op: &DecodedOp) -> Result<(), TransError> {
        let cond_raw = op.carg(0);
        let cond = TcgCond::from_raw(cond_raw).ok_or(TransError::BadCondition(cond_raw))?;
        let al = self.read_iarg(op, 0, TcgType::I32)?;
        let ah = self.read_iarg(op, 1, TcgType::I32)?;
        let bl = self.read_iarg(op, 2, TcgType::I32)?;
        let bh = self.read_iarg(op, 3, TcgType::I32)?;
        let al64 = self.builder.ins().uextend(types::I64, al);
        let ah64 = self.builder.ins().uextend(types::I64, ah);
        let bl64 = self.builder.ins().uextend(types::I64, bl);
        let bh64 = self.builder.ins().uextend(types::I64, bh);
        let a = {
            let s = self.builder.ins().ishl_imm(ah64, 32);
            self.builder.ins().bor(s, al64)
        };
        let b = {
            let s = self.builder.ins().ishl_imm(bh64, 32);
            self.builder.ins().bor(s, bl64)
        };
        let v = match cond {
            TcgCond::Never => self.builder.ins().iconst(types::I32, 0),
            TcgCond::Always => self.builder.ins().iconst(types::I32, 1),
            _ => {
                let bit = self.eval_tcg_cond(cond, a, b)?;
                self.builder.ins().uextend(types::I32, bit)
            }
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_wide_arith(&mut self, opc: Opc, op: &DecodedOp) -> Result<(), TransError> {
        use Opc::*;
        match opc {
            Mulsh | Muluh => {
                let a = self.read_iarg(op, 0, op.ty)?;
                let b = self.read_iarg(op, 1, op.ty)?;
                let v = if matches!(opc, Mulsh) {
                    self.builder.ins().smulhi(a, b)
                } else {
                    self.builder.ins().umulhi(a, b)
                };
                self.write_temp(op.oarg(0), v);
                Ok(())
            }
            Mulu2 | Muls2 => {
                let a = self.read_iarg(op, 0, op.ty)?;
                let b = self.read_iarg(op, 1, op.ty)?;
                let lo = self.builder.ins().imul(a, b);
                let hi = if matches!(opc, Muls2) {
                    self.builder.ins().smulhi(a, b)
                } else {
                    self.builder.ins().umulhi(a, b)
                };
                self.write_temp(op.oarg(0), lo);
                self.write_temp(op.oarg(1), hi);
                Ok(())
            }
            Divs2 | Divu2 => {
                /*
                 * TCG `divs2 q, r, lo, hi, d` divides the 2N-bit value
                 * `(hi : lo)` by the N-bit divisor `d`. The previous
                 * implementation ignored `hi` entirely and just did
                 * `q = lo / d`, which is wrong whenever the guest had
                 * any value in EDX (or RDX for 64-bit IDIV/DIV).
                 *
                 * Doing this correctly for I32 inputs needs a widen-to-
                 * I64 + divide + narrow dance; for I64 inputs it needs
                 * I128 ops that Cranelift doesn't reliably support for
                 * sdiv/udiv. For now bail to tier-1 — div/idiv is rare
                 * outside specialised hot loops, so the perf cost of
                 * leaving it on the legacy backend is small, and
                 * miscompiling it is catastrophic for game state.
                 */
                Err(TransError::UnsupportedOp(opc as u16))
            }
            Extract2 => {
                // dst = (lo | (hi << bits)) >> pos    (truncated to type bits)
                let a = self.read_iarg(op, 0, op.ty)?;
                let b = self.read_iarg(op, 1, op.ty)?;
                let pos = op.carg(0) as i64;
                let ty = self.builder.func.dfg.value_type(a);
                let bits = ty.bits() as i64;
                let lo_shifted = self.builder.ins().ushr_imm(a, pos);
                let hi_shifted = self.builder.ins().ishl_imm(b, bits - pos);
                let v = self.builder.ins().bor(lo_shifted, hi_shifted);
                self.write_temp(op.oarg(0), v);
                Ok(())
            }
            _ => Err(TransError::UnsupportedOp(opc as u16)),
        }
    }

    fn lower_env_ld(&mut self, opc: Opc, op: &DecodedOp) -> Result<(), TransError> {
        let base = self.read_iarg(op, 0, TcgType::I64)?;
        let off = op.carg(0) as i64;
        let addr = self.builder.ins().iadd_imm(base, off);
        let (ty, signed): (Type, bool) = match opc {
            Opc::Ld8u => (types::I8, false),
            Opc::Ld8s => (types::I8, true),
            Opc::Ld16u => (types::I16, false),
            Opc::Ld16s => (types::I16, true),
            Opc::Ld32u => (types::I32, false),
            Opc::Ld32s => (types::I32, true),
            Opc::Ld => (Self::type_for(op.ty), false),
            _ => return Err(TransError::Internal("not an env-ld")),
        };
        let v = self
            .builder
            .ins()
            .load(ty, MemFlags::trusted(), addr, 0);
        let dst_ty = Self::type_for(op.ty);
        let v = if ty != dst_ty {
            if signed {
                self.builder.ins().sextend(dst_ty, v)
            } else {
                self.builder.ins().uextend(dst_ty, v)
            }
        } else {
            v
        };
        self.write_temp(op.oarg(0), v);
        Ok(())
    }

    fn lower_env_st(&mut self, opc: Opc, op: &DecodedOp) -> Result<(), TransError> {
        let val = self.read_iarg(op, 0, op.ty)?;
        let base = self.read_iarg(op, 1, TcgType::I64)?;
        let off = op.carg(0) as i64;
        let addr = self.builder.ins().iadd_imm(base, off);
        let store_ty: Type = match opc {
            Opc::St8 => types::I8,
            Opc::St16 => types::I16,
            Opc::St32 => types::I32,
            Opc::St => Self::type_for(op.ty),
            _ => return Err(TransError::Internal("not an env-st")),
        };
        let cur = self.builder.func.dfg.value_type(val);
        let to_store = if cur != store_ty {
            if cur.bits() > store_ty.bits() {
                self.builder.ins().ireduce(store_ty, val)
            } else {
                self.builder.ins().uextend(store_ty, val)
            }
        } else {
            val
        };
        self.builder
            .ins()
            .store(MemFlags::trusted(), to_store, addr, 0);
        Ok(())
    }
}

// Sibling modules use Lowering::builder directly; no helper needed.
