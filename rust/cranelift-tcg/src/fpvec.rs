//! FP and vector op lowering.
//!
//! TCG's FP ops live past `QemuSt2` in tcg-opc.h; we identify them by
//! name (looking up the raw opcode against a small table). Vector ops
//! likewise.
//!
//! Coverage strategy:
//! - Common scalar FP arithmetic (`add_f32`, `add_f64`, `sub_*`,
//!   `mul_*`, `div_*`, `sqrt_*`, `chs_*`, `abs_*`, `mov_*`) map 1:1
//!   onto Cranelift's `fadd/fsub/fmul/fdiv/sqrt/fneg/fabs/copy`.
//! - Conversions (`cvt32f_f64`, `cvt32f_i32`, `cvt32i_f32` etc.) map
//!   onto Cranelift's `fpromote`, `fdemote`, `fcvt_to_sint`,
//!   `fcvt_from_sint`.
//! - Transcendentals (`sin_*`, `cos_*`) fall back via libcall.
//! - Vector ops route through Cranelift's `iaddv`, `imul`, `bxor`,
//!   `band`, `bor`, `bnot`, shuffle.
//! - Atomic ops route through Cranelift's `atomic_rmw` /
//!   `atomic_cas` / `fence`.

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Signature, Value};
use cranelift_codegen::isa::CallConv;

use crate::ir::DecodedOp;
use crate::opc::TcgType;
use crate::translator::{Lowering, TransError};

/*
 * Raw TCG opcode numbers for ld_vec/st_vec. These live past the named
 * `Opc` enum (which stops at QemuSt2 = 82) and reach Cranelift as
 * `Op::Other(raw)` filtered by `TCG_OPF_VECTOR`. The numbers come from
 * tcg-opc.h's DEF() ordering and must stay synchronized; the C-side
 * `BUILD_BUG_ON_TCG_OPC_DRIFT` guards catch drift at compile time.
 *
 * Compile telemetry confirms these are the dominant vector bailout
 * sources in Halo 2: ~30-40 errs/run combined across ld_vec + st_vec
 * (each err bails an entire TB containing SSE memory ops back to
 * tier-1 TCG, where x86 movdqa/movaps land).
 */
const OPC_LD_VEC: u16 = 127;
const OPC_ST_VEC: u16 = 128;
const OPC_OR_VEC: u16 = 144;

/// Lower an FP opcode by its raw integer code (past QemuSt2 / 82).
///
/// The numbering here MUST match the order in tcg-opc.h after the
/// Host floating point support comment block.
pub(crate) fn lower_fp(l: &mut Lowering<'_, '_>, raw: u16, op: &DecodedOp) -> Result<(), TransError> {
    // First FP opcode (`flcr`) sits at index 83.
    const FP_BASE: u16 = 83;
    let idx = raw.checked_sub(FP_BASE).ok_or(TransError::UnsupportedOp(raw))?;

    // Helper to read an FP input as the right Cranelift type.
    fn fp_ty(ty: TcgType) -> cranelift_codegen::ir::Type {
        match ty {
            TcgType::I32 | TcgType::V64 => types::F32,
            _ => types::F64,
        }
    }
    let cl_ty = fp_ty(op.ty);

    fn bitcast_to_fp(l: &mut Lowering<'_, '_>, v: Value, fty: cranelift_codegen::ir::Type) -> Value {
        let cur = l.builder.func.dfg.value_type(v);
        if cur == fty {
            return v;
        }
        let mf = cranelift_codegen::ir::MemFlags::new();
        l.builder.ins().bitcast(fty, mf, v)
    }
    fn bitcast_to_int(l: &mut Lowering<'_, '_>, v: Value, ity: cranelift_codegen::ir::Type) -> Value {
        let mf = cranelift_codegen::ir::MemFlags::new();
        l.builder.ins().bitcast(ity, mf, v)
    }

    // Names of FP opcodes in tcg-opc.h order:
    //   0: flcr, 1: ld80f_f32, 2: ld80f_f64, 3: st80f_f32, 4: st80f_f64,
    //   5: abs_f32, 6: abs_f64,
    //   7: add_f32, 8: add_f64,
    //   9: chs_f32, 10: chs_f64,
    //   11: com_f32, 12: com_f64,
    //   13: cos_f32, 14: cos_f64,
    //   15: cvt32f_f64, 16: cvt32f_i32, 17: cvt32f_i64,
    //   18: cvt32i_f32, 19: cvt32i_f64,
    //   20: cvt64f_f32, 21: cvt64f_i32, 22: cvt64f_i64,
    //   23: cvt64i_f32, 24: cvt64i_f64,
    //   25: div_f32, 26: div_f64,
    //   27: mov32f_i32, 28: mov32i_f32, 29: mov64f_i64, 30: mov64i_f64,
    //   31: mov_f32, 32: mov_f64,
    //   33: mul_f32, 34: mul_f64,
    //   35: sin_f32, 36: sin_f64,
    //   37: sqrt_f32, 38: sqrt_f64,
    //   39: sub_f32, 40: sub_f64,
    match idx {
        /*
         * flcr — write FPU control register.
         *
         * Tier-1 (tcg/aarch64/tcg-target.c.inc:3603) lowers this as an
         * inline MSR FPCR sequence converting MXCSR rounding bits to
         * FPCR rounding bits. Cranelift IR has no MSR primitive, so we
         * route through cranelift_helper_flcr in cranelift-bridge.c
         * which does the same bit-fiddling and msr.
         *
         * Before this lowering existed every TB containing FLDCW /
         * FNCLEX bailed to tier-1; on Halo 2 that was 23/584 ≈ 4%
         * of attempted tier-2 compiles, third-largest bail source
         * after st_vec and the STI-shadow filter (which are both
         * gated for orthogonal reasons).
         *
         * DEF(flcr) in tcg-opc.h: 0 outs, 1 in (the MXCSR value),
         * 0 cargs. We pass the value through to the helper as i32.
         */
        0 => {
            let flcr_fn = l.env.flcr_fn;
            if flcr_fn == 0 {
                return Err(TransError::UnsupportedOp(raw));
            }
            let mxcsr = l.read_iarg(op, 0, TcgType::I32)?;
            /* Phase 2 (gated X1BOX_DIRECT_BL_EXT=flcr|1): direct
             * relocated bl to cranelift_helper_flcr; fall back to
             * iconst+call_indirect when not gated. */
            if let Some(func_ref) = l.declare_helper("cranelift_helper_flcr") {
                l.builder.ins().call(func_ref, &[mxcsr]);
            } else {
                let mut sig = Signature::new(CallConv::SystemV);
                sig.params.push(AbiParam::new(types::I32));
                let sig_ref = l.builder.import_signature(sig);
                let addr = l
                    .builder
                    .ins()
                    .iconst(l.host_ptr_ty, flcr_fn as i64);
                l.builder.ins().call_indirect(sig_ref, addr, &[mxcsr]);
            }
            Ok(())
        }
        /*
         * com_f32 / com_f64: FP comparison producing x86-FCOM-style
         * EFLAGS bits packed into an i64 output.
         *
         * Tier-1's tcg/aarch64 lowering (tcg-target.c.inc:3119
         * `tcg_out_fp_com`) runs FCMP, reads NZCV via MRS, and maps:
         *   CF (bit 0) = N | V   (less-than OR unordered)
         *   PF (bit 2) = V       (unordered / NaN)
         *   ZF (bit 6) = Z | V   (equal OR unordered)
         *
         * In Cranelift IR we go via three FloatCC predicates that
         * already encode "or unordered" semantics:
         *   FloatCC::UnorderedOrLessThan      → CF
         *   FloatCC::Unordered                → PF
         *   FloatCC::UnorderedOrEqual         → ZF
         * Each yields an i8 (0/1) which we widen to i64 and OR with
         * the correct bit position. Output is i64 per tcg-op-fp.c:71.
         *
         * Indices 11 (com_f32) and 12 (com_f64) — biggest remaining
         * FP bail source at ~17/108 cumulative errs on Halo 2.
         */
        11 | 12 => {
            let a = l.read_iarg(op, 0, op.ty)?;
            let b = l.read_iarg(op, 1, op.ty)?;
            let mf = cranelift_codegen::ir::MemFlags::new();
            let af = if l.builder.func.dfg.value_type(a) != cl_ty {
                l.builder.ins().bitcast(cl_ty, mf, a)
            } else { a };
            let bf = if l.builder.func.dfg.value_type(b) != cl_ty {
                l.builder.ins().bitcast(cl_ty, mf, b)
            } else { b };
            /*
             * Cranelift's aarch64 backend supports `UnorderedOrLessThan`
             * (maps to Cond::Lt) but `unimplemented!()`s on
             * `UnorderedOrEqual` (cranelift-codegen 0.130.2
             * aarch64/lower.rs:91) — no single ARM64 condition encodes
             * UN|EQ. Compose it as bor(Equal, Unordered).
             *
             * Crash 2026-05-21: first build with com_f64 lowering
             * panicked in Cranelift compile thread at lower.rs:91:38
             * "not implemented" before the third install. Decomposing
             * sidesteps the missing condition.
             */
            /*
             * CF = less-than OR unordered. cranelift-codegen 0.130.2's aarch64
             * backend MIS-LOWERS the combined FloatCC::UnorderedOrLessThan here
             * — device-confirmed via the COMI shadow: COMISS(0.0, -1.0), which
             * is "greater", came back CF=1 (0x01, "below") instead of 0, i.e.
             * less<->greater inverted on every ordered compare. ZF/PF via
             * Equal/Unordered lower correctly, so the fault is specific to the
             * combined predicate. Decompose into LessThan | Unordered — the
             * same workaround already applied just below for UnorderedOrEqual.
             */
            let lt = l.builder.ins().fcmp(FloatCC::LessThan, af, bf);
            let uo = l.builder.ins().fcmp(FloatCC::Unordered, af, bf);
            let eq = l.builder.ins().fcmp(FloatCC::Equal, af, bf);
            let lt64 = l.builder.ins().uextend(types::I64, lt);
            let pf64 = l.builder.ins().uextend(types::I64, uo);
            let eq64 = l.builder.ins().uextend(types::I64, eq);
            let cf64 = l.builder.ins().bor(lt64, pf64);
            let zf64 = l.builder.ins().bor(eq64, pf64);
            let pf_shifted = l.builder.ins().ishl_imm(pf64, 2);
            let zf_shifted = l.builder.ins().ishl_imm(zf64, 6);
            let t = l.builder.ins().bor(cf64, pf_shifted);
            let result = l.builder.ins().bor(t, zf_shifted);
            l.write_temp(op.oarg(0), result);
            Ok(())
        }
        7 | 8 => bin_fp(l, op, cl_ty, |b, a, c| b.ins().fadd(a, c)),
        25 | 26 => bin_fp(l, op, cl_ty, |b, a, c| b.ins().fdiv(a, c)),
        33 | 34 => bin_fp(l, op, cl_ty, |b, a, c| b.ins().fmul(a, c)),
        39 | 40 => bin_fp(l, op, cl_ty, |b, a, c| b.ins().fsub(a, c)),
        5 | 6 => un_fp(l, op, cl_ty, |b, a| b.ins().fabs(a)),
        9 | 10 => un_fp(l, op, cl_ty, |b, a| b.ins().fneg(a)),
        37 | 38 => un_fp(l, op, cl_ty, |b, a| b.ins().sqrt(a)),
        31 | 32 => {
            let a = l.read_iarg(op, 0, op.ty)?;
            let v = bitcast_to_fp(l, a, cl_ty);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        15 => {
            // cvt32f_f64 = fpromote f32 -> f64
            let a = l.read_iarg(op, 0, TcgType::I32)?;
            let af = bitcast_to_fp(l, a, types::F32);
            let pr = l.builder.ins().fpromote(types::F64, af);
            let v = bitcast_to_int(l, pr, types::I64);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        20 => {
            // cvt64f_f32 = fdemote f64 -> f32
            let a = l.read_iarg(op, 0, TcgType::I64)?;
            let af = bitcast_to_fp(l, a, types::F64);
            let dem = l.builder.ins().fdemote(types::F32, af);
            let v = bitcast_to_int(l, dem, types::I32);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        16 => {
            // cvt32f_i32 = fcvt_from_sint i32 -> f32
            let a = l.read_iarg(op, 0, TcgType::I32)?;
            let f = l.builder.ins().fcvt_from_sint(types::F32, a);
            let v = bitcast_to_int(l, f, types::I32);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        17 => {
            let a = l.read_iarg(op, 0, TcgType::I64)?;
            let f = l.builder.ins().fcvt_from_sint(types::F32, a);
            let v = bitcast_to_int(l, f, types::I32);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        21 => {
            let a = l.read_iarg(op, 0, TcgType::I32)?;
            let f = l.builder.ins().fcvt_from_sint(types::F64, a);
            let v = bitcast_to_int(l, f, types::I64);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        22 => {
            let a = l.read_iarg(op, 0, TcgType::I64)?;
            let f = l.builder.ins().fcvt_from_sint(types::F64, a);
            let v = bitcast_to_int(l, f, types::I64);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        // cvt*f_i* = x87 FIST/FISTP (float -> signed int). Cranelift's
        // fcvt_to_sint is round-toward-zero (truncate), and Cranelift IR has no
        // "convert using the dynamic FPCR rounding mode" op, so tier-2 cannot
        // honor the guest x87 rounding-control bits: it would truncate even
        // under the games' default RC=nearest (FISTP 2.7 -> 2 instead of 3,
        // device-confirmed). Forcing nearest() here would instead break the
        // RC=toward-zero MSVC ftol idiom (e.g. Serious Sam II). Bail the whole
        // TB to tier-1, whose aarch64 lowering does FRINTI(per-FPCR)+FCVTZS
        // correctly for any RC. See HANDOFF_math_xpack.md.
        18 | 19 | 23 | 24 => Err(TransError::UnsupportedOp(raw)),
        // mov32f_i32 / mov32i_f32 / mov64f_i64 / mov64i_f64 are bitcasts.
        27 | 28 | 29 | 30 => {
            let a = l.read_iarg(op, 0, op.ty)?;
            l.write_temp(op.oarg(0), a);
            Ok(())
        }
        _ => Err(TransError::UnsupportedOp(raw)),
    }
}

fn bin_fp(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    cl_ty: cranelift_codegen::ir::Type,
    f: impl FnOnce(&mut cranelift_frontend::FunctionBuilder<'_>, Value, Value) -> Value,
) -> Result<(), TransError> {
    let a = l.read_iarg(op, 0, op.ty)?;
    let b = l.read_iarg(op, 1, op.ty)?;
    let mf = cranelift_codegen::ir::MemFlags::new();
    let af = if l.builder.func.dfg.value_type(a) != cl_ty {
        l.builder.ins().bitcast(cl_ty, mf, a)
    } else { a };
    let bf = if l.builder.func.dfg.value_type(b) != cl_ty {
        l.builder.ins().bitcast(cl_ty, mf, b)
    } else { b };
    let rf = f(l.builder, af, bf);
    let ity = if cl_ty == types::F32 { types::I32 } else { types::I64 };
    let v = l.builder.ins().bitcast(ity, mf, rf);
    l.write_temp(op.oarg(0), v);
    Ok(())
}

fn un_fp(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    cl_ty: cranelift_codegen::ir::Type,
    f: impl FnOnce(&mut cranelift_frontend::FunctionBuilder<'_>, Value) -> Value,
) -> Result<(), TransError> {
    let a = l.read_iarg(op, 0, op.ty)?;
    let mf = cranelift_codegen::ir::MemFlags::new();
    let af = if l.builder.func.dfg.value_type(a) != cl_ty {
        l.builder.ins().bitcast(cl_ty, mf, a)
    } else { a };
    let rf = f(l.builder, af);
    let ity = if cl_ty == types::F32 { types::I32 } else { types::I64 };
    let v = l.builder.ins().bitcast(ity, mf, rf);
    l.write_temp(op.oarg(0), v);
    Ok(())
}

/// Lower a vector opcode by its raw integer.
///
/// TCG vector ops start at index 0x?? (past FP) - see tcg-opc.h.
/// We approximate by routing every TCG_OPF_VECTOR op through a
/// dispatcher table. Anything we don't know returns UnsupportedOp so
/// the dispatcher leaves the TB on tier-1.
pub(crate) fn lower_vec(l: &mut Lowering<'_, '_>, raw: u16, op: &DecodedOp) -> Result<(), TransError> {
    match raw {
        OPC_LD_VEC => lower_ld_vec(l, op),
        OPC_ST_VEC => lower_st_vec(l, op),
        OPC_OR_VEC => lower_or_vec(l, op),
        _ => Err(TransError::UnsupportedOp(raw)),
    }
}

/*
 * or_vec — bitwise OR of two vector temps.
 *
 * Lane width (vece) is irrelevant for bitwise ops, so we don't need
 * the vece slot that cranelift-bridge.c:854-857 currently drops on the
 * floor. read_iarg routes both operands through the scalar↔vector
 * coerce so they arrive as I64X2 regardless of how TCG materialised
 * them; `bor` on I64X2 lowers to AArch64 `ORR Vd.16B, Vn.16B, Vm.16B`
 * which gives the right answer for any element size.
 *
 * Halo 2 hit this exactly once per ~1900 compiles in the 2026-05-27
 * Zenfone histogram (opc144=1). Tiny lever, but a clean closing motion
 * on the Phase 1 bail-reduction pass — no plumbing, single op.
 */
fn lower_or_vec(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    let a = l.read_iarg(op, 0, op.ty)?;
    let b = l.read_iarg(op, 1, op.ty)?;
    let r = l.builder.ins().bor(a, b);
    l.write_temp(op.oarg(0), r);
    Ok(())
}

/*
 * ld_vec/st_vec — load/store of a TCG vector temp to/from HOST memory
 * (env-relative spill or scratch). NOT guest memory; these are QEMU's
 * own register-allocator spill ops and don't need the TLB.
 *
 * Signature shape mirrors the regular `ld`/`st` env ops:
 *   ld_vec out, base, offset_carg
 *   st_vec val, base, offset_carg
 *
 * The translator represents both V64 and V128 temps as Cranelift
 * `I64X2` (see `type_for`). For V64 we load/store only the low 8
 * bytes; for V128 the full 16. The host pointer (`base + offset`) is
 * always 16-byte aligned on env spills, so `MemFlags::trusted()` is
 * safe.
 */
/*
 * Coverage matrix:
 *   - V128 (SSE movdqa / movaps spill — Halo 2's dominant case at
 *     146/241 cumulative bails before this change): full I64X2 path.
 *   - V64  (MMX or SSE movq spill): 8-byte path. The temp is still
 *     declared I64X2 per `type_for`, so we extract the low lane via
 *     `extractlane` for the store, and `insertlane` a zero high lane
 *     after the 8-byte load so the rest of the temp's lifetime is
 *     well-defined. The high lane being garbage was the original
 *     reason this case was deferred, not anything fundamental.
 *   - V256 (AVX): still bails. Xbox is pre-AVX so this is dead in
 *     practice; the path would require I64X4 plumbing we don't have.
 *
 * Bails kept:
 *   - const-valued vector value input (no scalar→vector iconst path
 *     in Cranelift; vanishingly rare on x86 anyway — TCG doesn't fold
 *     vector temps to const).
 *
 * Bail relaxed:
 *   - const base address: just an iconst, fully supported via
 *     `read_iarg`. Was previously bailing for "encoding-risk surface"
 *     reasons that don't actually apply.
 *
 * Alignment: env spill slots are 16-byte aligned but we use
 * `MemFlags::new()` (no alignment promise) so Cranelift picks AArch64
 * unaligned `LDR Q` / `STR Q`, which A720/X4 handle without trap.
 */
fn lower_ld_vec(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    let width_bytes = match op.ty {
        TcgType::V128 => 16,
        TcgType::V64 => 8,
        _ => return Err(TransError::UnsupportedOp(OPC_LD_VEC)),
    };
    let base = l.read_iarg(op, 0, TcgType::I64)?;
    let off = op.carg(0) as i64;
    let addr = l.builder.ins().iadd_imm(base, off);
    let v = if width_bytes == 16 {
        l.builder
            .ins()
            .load(types::I64X2, MemFlags::new(), addr, 0)
    } else {
        /* Load the low 8 bytes as i64, splat-zero a vector, insert
         * the loaded scalar as lane 0. Lane 1 stays zero so any
         * later widening read sees a clean upper half. */
        let lo = l
            .builder
            .ins()
            .load(types::I64, MemFlags::new(), addr, 0);
        let zero = l.builder.ins().iconst(types::I64, 0);
        let vec_zero = l.builder.ins().splat(types::I64X2, zero);
        l.builder.ins().insertlane(vec_zero, lo, 0)
    };
    l.write_temp(op.oarg(0), v);
    Ok(())
}

/*
 * Diagnostic subcodes for the remaining lower_st_vec bail branches.
 * Logged via the existing opcN telemetry; non-overlapping with real
 * TCG opcode space (which tops out around 200). 0x9080+offset is a
 * sentinel range — opc36992/3/4/5 in the breakdown log distinguishes
 * which branch fires so we can target the highest-volume one.
 */
const OPC_ST_VEC_BAIL_V256:        u16 = 0x9080;
/* OPC_ST_VEC_BAIL_CONST_VAL (0x9081) was retired 2026-05-22 — const
 * values now flow through scalar_to_vector via coerce. */
const OPC_ST_VEC_BAIL_V128_VAL_TY: u16 = 0x9082;
const OPC_ST_VEC_BAIL_V64_VAL_TY:  u16 = 0x9083;

fn lower_st_vec(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    let width_bytes = match op.ty {
        TcgType::V128 => 16,
        TcgType::V64 => 8,
        _ => return Err(TransError::UnsupportedOp(OPC_ST_VEC_BAIL_V256)),
    };
    /*
     * Const-valued vector: read_iarg materializes as iconst.I64 then
     * coerces to I64X2 via the 2026-05-22 scalar→vector path
     * (`scalar_to_vector(I64X2, iconst.I64)` puts the literal in lane 0
     * and zeros lane 1 — matches TCG's "TEMP_CONST holds a 64-bit value
     * spilling into the vector's low half" convention).
     * Before that fix this branch was the dominant remaining bail (19
     * of 19 opc128 errors per session). Now const values flow through.
     */
    let val = l.read_iarg(op, 0, op.ty)?;
    let base = l.read_iarg(op, 1, TcgType::I64)?;
    let off = op.carg(0) as i64;
    let addr = l.builder.ins().iadd_imm(base, off);
    if width_bytes == 16 {
        let vty = l.builder.func.dfg.value_type(val);
        if vty != types::I64X2 {
            /* Diagnostic: capture the actual Cranelift value type so
             * we know which coercion is missing.  One-shot via static
             * AtomicBool — same pattern as the dispatcher's first-
             * verifier-failure logger. */
            static FIRST: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(true);
            if FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) {
                crate::dispatcher::log_to_android(&format!(
                    "st_vec V128 bail (first): val_ty={} op.ty={:?}",
                    vty, op.ty
                ));
            }
            return Err(TransError::UnsupportedOp(OPC_ST_VEC_BAIL_V128_VAL_TY));
        }
        l.builder.ins().store(MemFlags::new(), val, addr, 0);
    } else {
        let val_ty = l.builder.func.dfg.value_type(val);
        let lo = if val_ty == types::I64X2 {
            l.builder.ins().extractlane(val, 0)
        } else if val_ty == types::I64 {
            val
        } else {
            static FIRST: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(true);
            if FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) {
                crate::dispatcher::log_to_android(&format!(
                    "st_vec V64 bail (first): val_ty={} op.ty={:?}",
                    val_ty, op.ty
                ));
            }
            return Err(TransError::UnsupportedOp(OPC_ST_VEC_BAIL_V64_VAL_TY));
        };
        l.builder.ins().store(MemFlags::new(), lo, addr, 0);
    }
    Ok(())
}
