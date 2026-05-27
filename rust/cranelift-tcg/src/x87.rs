//! Native fp80 (HARD_FPU) inline lowerings for the x87 helper calls
//! that still reach Cranelift as call_indirect.
//!
//! Background. On AArch64 with `g_use_fp_jit = true`, target/i386 dual-
//! compiles fpu_helper.c — `__hard` variants use native `double` storage
//! at `env->fpregs[i].native_d` and `env->ft0_native`. The tier-1 inline
//! TCG-fp path covers add/sub/mul/div/sqrt/abs/chs/cmp/fldz/fld1 already
//! (see ops_fpu.h via `gen_helper_fp_arith_*` and the BISECT_GRP_* gates
//! in translate.c). What still arrives in tier-2 as `call_indirect`:
//!
//!   * fucom / fcomi / fucomi    — quiet compares + EFLAGS materialise
//!   * fxam                       — fp classification → fpus
//!   * fldl2t / fldl2e / fldpi / fldlg2 / fldln2 — constant pushers
//!   * the arith / stack-mgmt / move helpers when the bisect groups are
//!     flipped off, or in any rebuilt frontend path that opted out
//!
//! All of these reduce to a handful of env loads/stores plus one f64
//! arith op — Cranelift emits a straight-line block per call site and
//! the call_indirect goes away.
//!
//! IEEE exception-flag side effects (`save_exception_flags` /
//! `merge_exception_flags`) are deliberately SKIPPED on the inline path:
//!   * In HARD_FPU mode merge_exception_flags is a no-op.
//!   * save_exception_flags just zeros env->fp_status.exception_flags;
//!     no Xbox title we've profiled observes those bits.
//! If a future workload depends on env->fp_status.exception_flags being
//! cleared per op — or on host FPU exceptions propagating to fpus / the
//! guest — that is the first thing to revisit here. The C-side gate is
//! the `X1BOX_X87_INLINE=0` env var (see cranelift-bridge.c).

use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, InstBuilder, MemFlags, Signature, types, Value,
};
use cranelift_codegen::isa::CallConv;

use crate::ir::DecodedOp;
use crate::translator::{Lowering, TransError};

/// x86 CC flag bits we touch. Mirrors `CC_*` macros in
/// target/i386/cpu.h — keep these in sync if the host changes.
const CC_C: i64 = 0x0001;
const CC_P: i64 = 0x0004;
const CC_Z: i64 = 0x0040;
const CC_OP_EFLAGS: i64 = 0;

/// fpus comparison bits (C3,C2,C0) packed at bits 14,10,8 of fpus.
/// Per `fcom_ccval[]` in target/i386/tcg/fpu_helper.c.
const FPUS_LT:    i64 = 0x0100;
const FPUS_EQ:    i64 = 0x4000;
const FPUS_GT:    i64 = 0x0000;
const FPUS_UNORD: i64 = 0x4500;

/// fxam classification bits.
const FXAM_NAN:    i64 = 0x0100;
const FXAM_INF:    i64 = 0x0500;
const FXAM_ZERO:   i64 = 0x4000;
const FXAM_SUBN:   i64 = 0x4400;
const FXAM_NORMAL: i64 = 0x0400;
const FXAM_EMPTY:  i64 = 0x4100;
const FXAM_SIGN:   i64 = 0x0200;

/// Top-level dispatch. Returns Ok(true) when the call was lowered
/// inline (caller must NOT fall through to call_indirect); Ok(false)
/// means no inline pattern matched and the caller should emit the
/// normal helper-call sequence. Err(...) bubbles a translation error.
pub(crate) fn try_lower(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    helper_ptr: u64,
) -> Result<bool, TransError> {
    let x = &l.env.x87;

    // No-arg-int helpers (env-only). 1 iarg = env, 0 oargs.
    if op.nb_iargs == 1 && op.nb_oargs == 0 {
        // Comparisons.
        if helper_ptr != 0 && helper_ptr == x.fucom_st0_ft0_fn {
            lower_fucom(l)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fcomi_st0_ft0_fn {
            lower_fcomi(l, /*signaling=*/true)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fucomi_st0_ft0_fn {
            lower_fcomi(l, /*signaling=*/false)?; return Ok(true);
        }

        // Classification.
        if helper_ptr != 0 && helper_ptr == x.fxam_st0_fn {
            lower_fxam(l)?; return Ok(true);
        }

        // Constant pushers — fpush + ST0 = K. The C side resolves the
        // helper-ptr against the matching `__hard`/`__soft` symbol, so
        // any of these arms only fires for the variant we have an
        // address for.
        if helper_ptr != 0 && helper_ptr == x.fldl2t_st0_fn {
            // log2(10)
            lower_const_push(l, 3.32192809488736234787)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldl2e_st0_fn {
            lower_const_push(l, std::f64::consts::LOG2_E)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldpi_st0_fn {
            lower_const_push(l, std::f64::consts::PI)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldlg2_st0_fn {
            lower_const_push(l, std::f64::consts::LN_2 / std::f64::consts::LN_10)?;
            return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldln2_st0_fn {
            lower_const_push(l, std::f64::consts::LN_2)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fld1_st0_fn {
            lower_const_push(l, 1.0)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldz_st0_fn {
            lower_const_push(l, 0.0)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fldz_ft0_fn {
            // FT0 = 0.0, no stack push. Used by gen_fcom_ST0 paths
            // that compare against zero.
            let zero = l.builder.ins().f64const(0.0);
            store_env_f64(l, x.ft0_native_offset as i32, zero);
            return Ok(true);
        }

        // Stack manipulation.
        if helper_ptr != 0 && helper_ptr == x.fpush_fn {
            lower_fpush(l)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fpop_fn {
            lower_fpop(l)?; return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fdecstp_fn {
            lower_fdecstp_fincstp(l, /*delta=*/-1)?;
            return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fincstp_fn {
            lower_fdecstp_fincstp(l, /*delta=*/1)?;
            return Ok(true);
        }

        // Move ST0 <- FT0.
        if helper_ptr != 0 && helper_ptr == x.fmov_st0_ft0_fn {
            let ft0 = read_env_f64(l, x.ft0_native_offset as i32);
            write_st_n_dyn(l, /*n_const=*/0, ft0);
            return Ok(true);
        }

        // Unary ops on ST0.
        if helper_ptr != 0 && helper_ptr == x.fchs_st0_fn {
            let v = read_st_n_dyn(l, 0);
            let neg = l.builder.ins().fneg(v);
            write_st_n_dyn(l, 0, neg);
            return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fabs_st0_fn {
            let v = read_st_n_dyn(l, 0);
            let abs = l.builder.ins().fabs(v);
            write_st_n_dyn(l, 0, abs);
            return Ok(true);
        }
        if helper_ptr != 0 && helper_ptr == x.fsqrt_fn {
            let v = read_st_n_dyn(l, 0);
            let s = l.builder.ins().sqrt(v);
            write_st_n_dyn(l, 0, s);
            return Ok(true);
        }

        // Arith ST0 op= FT0 (env-only variants).
        if helper_ptr != 0 && helper_ptr == x.fadd_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::Add).map(|_| true);
        }
        if helper_ptr != 0 && helper_ptr == x.fmul_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::Mul).map(|_| true);
        }
        if helper_ptr != 0 && helper_ptr == x.fsub_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::Sub).map(|_| true);
        }
        if helper_ptr != 0 && helper_ptr == x.fsubr_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::SubR).map(|_| true);
        }
        if helper_ptr != 0 && helper_ptr == x.fdiv_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::Div).map(|_| true);
        }
        if helper_ptr != 0 && helper_ptr == x.fdivr_st0_ft0_fn {
            return lower_arith_st0_ft0(l, BinOp::DivR).map(|_| true);
        }
    }

    // 2-iarg helpers: (env, int st_index).
    if op.nb_iargs == 2 && op.nb_oargs == 0 {
        // ffree_STN(env, st_index): fptags[(fpstt + n) & 7] = 1.
        if helper_ptr != 0 && helper_ptr == x.ffree_stn_fn {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            let idx = idx_for_stn(l, n);
            let one = l.builder.ins().iconst(types::I8, 1);
            store_fptag(l, idx, one);
            return Ok(true);
        }

        // fmov_FT0_STN(env, n): FT0 = ST(n).
        if helper_ptr != 0 && helper_ptr == x.fmov_ft0_stn_fn {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            let v = read_st_n_dynval(l, n);
            store_env_f64(l, l.env.x87.ft0_native_offset as i32, v);
            return Ok(true);
        }

        // fmov_ST0_STN(env, n): ST0 = ST(n).
        if helper_ptr != 0 && helper_ptr == x.fmov_st0_stn_fn {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            let v = read_st_n_dynval(l, n);
            write_st_n_dyn(l, 0, v);
            return Ok(true);
        }

        // fmov_STN_ST0(env, n): ST(n) = ST0.
        if helper_ptr != 0 && helper_ptr == x.fmov_stn_st0_fn {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            let v = read_st_n_dyn(l, 0);
            write_st_n_dynval(l, n, v);
            return Ok(true);
        }

        // fxchg_ST0_STN(env, n): swap ST0 and ST(n).
        if helper_ptr != 0 && helper_ptr == x.fxchg_st0_stn_fn {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            let st0 = read_st_n_dyn(l, 0);
            let stn = read_st_n_dynval(l, n);
            write_st_n_dyn(l, 0, stn);
            write_st_n_dynval(l, n, st0);
            return Ok(true);
        }

        // Arith STN op= ST0.
        let arith = if helper_ptr != 0 && helper_ptr == x.fadd_stn_st0_fn {
            Some(BinOp::Add)
        } else if helper_ptr != 0 && helper_ptr == x.fmul_stn_st0_fn {
            Some(BinOp::Mul)
        } else if helper_ptr != 0 && helper_ptr == x.fsub_stn_st0_fn {
            Some(BinOp::Sub)
        } else if helper_ptr != 0 && helper_ptr == x.fsubr_stn_st0_fn {
            Some(BinOp::SubR)
        } else if helper_ptr != 0 && helper_ptr == x.fdiv_stn_st0_fn {
            Some(BinOp::Div)
        } else if helper_ptr != 0 && helper_ptr == x.fdivr_stn_st0_fn {
            Some(BinOp::DivR)
        } else {
            None
        };
        if let Some(o) = arith {
            let n = l.read_iarg(op, 1, crate::opc::TcgType::I64)?;
            return lower_arith_stn_st0(l, n, o).map(|_| true);
        }
    }

    Ok(false)
}

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Mul,
    Sub,
    SubR, // ST0 = b - a
    Div,
    DivR, // ST0 = b / a
}

// --- env load/store primitives ----------------------------------------

fn read_env_u8(l: &mut Lowering<'_, '_>, off: i32) -> Value {
    l.builder.ins().load(types::I8, MemFlags::trusted(), l.env_val, off)
}
fn read_env_i32(l: &mut Lowering<'_, '_>, off: i32) -> Value {
    l.builder.ins().load(types::I32, MemFlags::trusted(), l.env_val, off)
}
fn read_env_u16_zx32(l: &mut Lowering<'_, '_>, off: i32) -> Value {
    let v = l.builder.ins().load(types::I16, MemFlags::trusted(), l.env_val, off);
    l.builder.ins().uextend(types::I32, v)
}
fn read_env_f64(l: &mut Lowering<'_, '_>, off: i32) -> Value {
    l.builder.ins().load(types::F64, MemFlags::trusted(), l.env_val, off)
}
fn read_env_i64(l: &mut Lowering<'_, '_>, off: i32) -> Value {
    l.builder.ins().load(types::I64, MemFlags::trusted(), l.env_val, off)
}

fn store_env_i32(l: &mut Lowering<'_, '_>, off: i32, v: Value) {
    l.builder.ins().store(MemFlags::trusted(), v, l.env_val, off);
}
fn store_env_u16_from_i32(l: &mut Lowering<'_, '_>, off: i32, v: Value) {
    // Truncate i32 → i16 and store.
    let t = l.builder.ins().ireduce(types::I16, v);
    l.builder.ins().store(MemFlags::trusted(), t, l.env_val, off);
}
fn store_env_f64(l: &mut Lowering<'_, '_>, off: i32, v: Value) {
    l.builder.ins().store(MemFlags::trusted(), v, l.env_val, off);
}
fn store_env_i64(l: &mut Lowering<'_, '_>, off: i32, v: Value) {
    l.builder.ins().store(MemFlags::trusted(), v, l.env_val, off);
}

/// Address of `env->fpregs[(fpstt + offset) & 7].native_d` for a
/// compile-time `offset` (0 = ST0). Computes the index at runtime from
/// the current env->fpstt; the slot stride and within-FPReg offset are
/// compile-time constants from `X87Layout`.
fn st_n_addr_const(l: &mut Lowering<'_, '_>, offset_const: i32) -> Value {
    let x = l.env.x87.clone();
    let fpstt = read_env_i32(l, x.fpstt_offset as i32);
    let idx_pre = l.builder.ins().iadd_imm(fpstt, offset_const as i64);
    let idx = l.builder.ins().band_imm(idx_pre, 7);
    // idx (i32) -> i64 byte offset
    let idx_i64 = l.builder.ins().uextend(types::I64, idx);
    let stride = x.fpreg_stride as i64;
    let byte_off = l.builder.ins().imul_imm(idx_i64, stride);
    // base = env + fpregs_offset + native_d_off + byte_off
    let env_plus = l.builder.ins().iadd_imm(
        l.env_val,
        (x.fpregs_offset as i64) + (x.fpreg_native_d_off as i64),
    );
    l.builder.ins().iadd(env_plus, byte_off)
}

/// Variant taking a dynamic `n` (i64 Value) for helpers that pass the
/// stack index via the helper-call arg list (fmov_*_STN, fxchg_STN).
fn st_n_addr_dyn(l: &mut Lowering<'_, '_>, n_i64: Value) -> Value {
    let x = l.env.x87.clone();
    let fpstt = read_env_i32(l, x.fpstt_offset as i32);
    let fpstt_i64 = l.builder.ins().uextend(types::I64, fpstt);
    let idx_pre = l.builder.ins().iadd(fpstt_i64, n_i64);
    let idx = l.builder.ins().band_imm(idx_pre, 7);
    let stride = x.fpreg_stride as i64;
    let byte_off = l.builder.ins().imul_imm(idx, stride);
    let env_plus = l.builder.ins().iadd_imm(
        l.env_val,
        (x.fpregs_offset as i64) + (x.fpreg_native_d_off as i64),
    );
    l.builder.ins().iadd(env_plus, byte_off)
}

fn read_st_n_dyn(l: &mut Lowering<'_, '_>, offset_const: i32) -> Value {
    let addr = st_n_addr_const(l, offset_const);
    l.builder.ins().load(types::F64, MemFlags::trusted(), addr, 0)
}
fn write_st_n_dyn(l: &mut Lowering<'_, '_>, offset_const: i32, v: Value) {
    let addr = st_n_addr_const(l, offset_const);
    l.builder.ins().store(MemFlags::trusted(), v, addr, 0);
}

fn read_st_n_dynval(l: &mut Lowering<'_, '_>, n_i64: Value) -> Value {
    let addr = st_n_addr_dyn(l, n_i64);
    l.builder.ins().load(types::F64, MemFlags::trusted(), addr, 0)
}
fn write_st_n_dynval(l: &mut Lowering<'_, '_>, n_i64: Value, v: Value) {
    let addr = st_n_addr_dyn(l, n_i64);
    l.builder.ins().store(MemFlags::trusted(), v, addr, 0);
}

/// Compute index for ST(n_i64): (fpstt + n) & 7, returned as i64.
fn idx_for_stn(l: &mut Lowering<'_, '_>, n_i64: Value) -> Value {
    let fpstt = read_env_i32(l, l.env.x87.fpstt_offset as i32);
    let fpstt_i64 = l.builder.ins().uextend(types::I64, fpstt);
    let s = l.builder.ins().iadd(fpstt_i64, n_i64);
    l.builder.ins().band_imm(s, 7)
}

/// fptags[idx_i64] = tag_i8.
fn store_fptag(l: &mut Lowering<'_, '_>, idx_i64: Value, tag_i8: Value) {
    let fptags_base = l.builder.ins().iadd_imm(
        l.env_val,
        l.env.x87.fptags_offset as i64,
    );
    let addr = l.builder.ins().iadd(fptags_base, idx_i64);
    l.builder.ins().store(MemFlags::trusted(), tag_i8, addr, 0);
}

// --- Comparisons ------------------------------------------------------

/// Compute a 3-way ladder of comparison results and produce per-case
/// values. Returns `Value` of type I64 holding one of the four passed
/// in i64 constants based on (st0 vs ft0).
fn fcmp_3way(
    l: &mut Lowering<'_, '_>,
    st0: Value,
    ft0: Value,
    lt_val: i64,
    eq_val: i64,
    gt_val: i64,
    unord_val: i64,
) -> Value {
    let unord = l.builder.ins().fcmp(FloatCC::Unordered, st0, ft0);
    let lt = l.builder.ins().fcmp(FloatCC::LessThan, st0, ft0);
    let eq = l.builder.ins().fcmp(FloatCC::Equal, st0, ft0);

    let gt_v   = l.builder.ins().iconst(types::I64, gt_val);
    let eq_v   = l.builder.ins().iconst(types::I64, eq_val);
    let lt_v   = l.builder.ins().iconst(types::I64, lt_val);
    let unord_v = l.builder.ins().iconst(types::I64, unord_val);

    let eq_or_gt = l.builder.ins().select(eq, eq_v, gt_v);
    let lt_or_else = l.builder.ins().select(lt, lt_v, eq_or_gt);
    l.builder.ins().select(unord, unord_v, lt_or_else)
}

fn lower_fucom(l: &mut Lowering<'_, '_>) -> Result<(), TransError> {
    let x = l.env.x87.clone();
    let st0 = read_st_n_dyn(l, 0);
    let ft0 = read_env_f64(l, x.ft0_native_offset as i32);
    let val = fcmp_3way(l, st0, ft0, FPUS_LT, FPUS_EQ, FPUS_GT, FPUS_UNORD);

    // fpus = (fpus & ~0x4500) | val
    let fpus = read_env_u16_zx32(l, x.fpus_offset as i32);
    let fpus_i64 = l.builder.ins().uextend(types::I64, fpus);
    let masked = l.builder.ins().band_imm(fpus_i64, !FPUS_UNORD);
    let merged = l.builder.ins().bor(masked, val);
    let merged_i32 = l.builder.ins().ireduce(types::I32, merged);
    store_env_u16_from_i32(l, x.fpus_offset as i32, merged_i32);
    Ok(())
}

fn lower_fcomi(
    l: &mut Lowering<'_, '_>,
    _signaling: bool,
) -> Result<(), TransError> {
    // signaling=true (fcomi) vs false (fucomi) differ only in whether
    // sNaN sets the invalid-operation exception. Since we skip
    // exception-flag side effects entirely, the two collapse to the
    // same lowering. Document this so a future debugger doesn't chase
    // the dead distinction.
    let x = l.env.x87.clone();
    let st0 = read_st_n_dyn(l, 0);
    let ft0 = read_env_f64(l, x.ft0_native_offset as i32);
    let val = fcmp_3way(l, st0, ft0, CC_C, CC_Z, 0, CC_Z | CC_P | CC_C);

    // Materialise current eflags via helper_cc_compute_all if we have
    // the addr, otherwise abort the inline path (fall through). With
    // the inline cc_compute_all specialization in helper.rs, this call
    // is cheap when CC_OP is already 0.
    let cc_ptr = l.env.cc_compute_all_fn;
    if cc_ptr == 0 {
        // No cc_compute_all → can't materialise eflags safely. Bail to
        // the caller's normal call_indirect path.
        return Err(TransError::UnsupportedOp(0xfc01));
    }

    let cc_dst  = read_env_i64(l, x.cc_dst_offset as i32);
    let cc_src  = read_env_i64(l, x.cc_src_offset as i32);
    let cc_src2 = read_env_i64(l, x.cc_src2_offset as i32);
    let cc_op_v = read_env_i32(l, x.cc_op_offset as i32);
    let cc_op_i64 = l.builder.ins().uextend(types::I64, cc_op_v);

    /* Phase 2 (gated X1BOX_DIRECT_BL_EXT=cc|1): direct relocated bl to
     * helper_cc_compute_all. Same shape as helper.rs::lower_cc_compute_all_inline
     * slow path; falls back to iconst+call_indirect when not gated. */
    let eflags = if let Some(func_ref) =
        l.declare_helper("helper_cc_compute_all")
    {
        let inst = l
            .builder
            .ins()
            .call(func_ref, &[cc_dst, cc_src, cc_src2, cc_op_i64]);
        l.builder.inst_results(inst)[0]
    } else {
        let mut sig = Signature::new(CallConv::SystemV);
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        let sig_ref = l.builder.import_signature(sig);
        let addr = l.builder.ins().iconst(l.host_ptr_ty, cc_ptr as i64);
        let call = l
            .builder
            .ins()
            .call_indirect(sig_ref, addr, &[cc_dst, cc_src, cc_src2, cc_op_i64]);
        l.builder.inst_results(call)[0]
    };

    let mask: i64 = !(CC_Z | CC_P | CC_C);
    let masked = l.builder.ins().band_imm(eflags, mask);
    let merged = l.builder.ins().bor(masked, val);
    store_env_i64(l, x.cc_src_offset as i32, merged);

    let cc_op_eflags = l.builder.ins().iconst(types::I32, CC_OP_EFLAGS);
    store_env_i32(l, x.cc_op_offset as i32, cc_op_eflags);
    Ok(())
}

// --- Classification ---------------------------------------------------

fn lower_fxam(l: &mut Lowering<'_, '_>) -> Result<(), TransError> {
    let x = l.env.x87.clone();

    // 1. Load ST0 bits (i64) and fpstt for fptag lookup.
    let st0_addr = st_n_addr_const(l, 0);
    let bits = l.builder.ins().load(types::I64, MemFlags::trusted(), st0_addr, 0);

    let fpstt = read_env_i32(l, x.fpstt_offset as i32);
    let fpstt_masked = l.builder.ins().band_imm(fpstt, 7);
    let fpstt_i64 = l.builder.ins().uextend(types::I64, fpstt_masked);

    // 2. signbit -> bit 9 (FXAM_SIGN = 0x200).
    let sign = l.builder.ins().ushr_imm(bits, 63);   // 0 or 1, i64
    let sign_bit = l.builder.ins().ishl_imm(sign, 9);

    // 3. exp_field = (bits >> 52) & 0x7FF; mant = bits & ((1<<52)-1).
    let exp_shifted = l.builder.ins().ushr_imm(bits, 52);
    let exp_field = l.builder.ins().band_imm(exp_shifted, 0x7FF);
    let mant_mask: i64 = (1i64 << 52) - 1;
    let mant_bits = l.builder.ins().band_imm(bits, mant_mask);

    let mant_nz = l.builder.ins().icmp_imm(
        cranelift_codegen::ir::condcodes::IntCC::NotEqual, mant_bits, 0,
    );
    let exp_zero = l.builder.ins().icmp_imm(
        cranelift_codegen::ir::condcodes::IntCC::Equal, exp_field, 0,
    );
    let exp_max = l.builder.ins().icmp_imm(
        cranelift_codegen::ir::condcodes::IntCC::Equal, exp_field, 0x7FF,
    );

    let zero_v   = l.builder.ins().iconst(types::I64, FXAM_ZERO);
    let subn_v   = l.builder.ins().iconst(types::I64, FXAM_SUBN);
    let normal_v = l.builder.ins().iconst(types::I64, FXAM_NORMAL);
    let inf_v    = l.builder.ins().iconst(types::I64, FXAM_INF);
    let nan_v    = l.builder.ins().iconst(types::I64, FXAM_NAN);

    // base = exp_zero ? (mant_nz ? subn : zero) : (exp_max ? (mant_nz ? nan : inf) : normal)
    let exp_zero_branch = l.builder.ins().select(mant_nz, subn_v, zero_v);
    let exp_max_branch  = l.builder.ins().select(mant_nz, nan_v, inf_v);
    let non_zero_exp   = l.builder.ins().select(exp_max, exp_max_branch, normal_v);
    let category       = l.builder.ins().select(exp_zero, exp_zero_branch, non_zero_exp);

    // 4. Empty check: fptags[fpstt] != 0  → category replaced by EMPTY.
    let fptags_addr = l.builder.ins().iadd_imm(
        l.env_val, x.fptags_offset as i64,
    );
    let tag_addr = l.builder.ins().iadd(fptags_addr, fpstt_i64);
    let tag = l.builder.ins().load(types::I8, MemFlags::trusted(), tag_addr, 0);
    let tag_i64 = l.builder.ins().uextend(types::I64, tag);
    let is_empty = l.builder.ins().icmp_imm(
        cranelift_codegen::ir::condcodes::IntCC::NotEqual, tag_i64, 0,
    );
    let empty_v = l.builder.ins().iconst(types::I64, FXAM_EMPTY);
    let final_cat = l.builder.ins().select(is_empty, empty_v, category);

    // 5. fpus = (fpus & ~0x4700) | sign_bit | final_cat.
    let fpus_i32 = read_env_u16_zx32(l, x.fpus_offset as i32);
    let fpus_i64 = l.builder.ins().uextend(types::I64, fpus_i32);
    let masked = l.builder.ins().band_imm(fpus_i64, !0x4700i64);
    let with_sign = l.builder.ins().bor(masked, sign_bit);
    let with_cat = l.builder.ins().bor(with_sign, final_cat);

    // Suppress the unused FXAM_SIGN binding warning (we use 0x200 via
    // sign_bit shift) — kept as a named constant for grep-ability.
    let _ = FXAM_SIGN;

    let merged_i32 = l.builder.ins().ireduce(types::I32, with_cat);
    store_env_u16_from_i32(l, x.fpus_offset as i32, merged_i32);
    Ok(())
}

// --- Stack manipulation ----------------------------------------------

fn lower_fpush(l: &mut Lowering<'_, '_>) -> Result<(), TransError> {
    // fpstt = (fpstt - 1) & 7; fptags[new fpstt] = 0.
    let x = l.env.x87.clone();
    let cur = read_env_i32(l, x.fpstt_offset as i32);
    let dec = l.builder.ins().iadd_imm(cur, -1);
    let new = l.builder.ins().band_imm(dec, 7);
    store_env_i32(l, x.fpstt_offset as i32, new);

    let new_i64 = l.builder.ins().uextend(types::I64, new);
    let zero = l.builder.ins().iconst(types::I8, 0);
    store_fptag(l, new_i64, zero);
    Ok(())
}

fn lower_fpop(l: &mut Lowering<'_, '_>) -> Result<(), TransError> {
    // fptags[fpstt] = 1; fpstt = (fpstt + 1) & 7.
    let x = l.env.x87.clone();
    let cur = read_env_i32(l, x.fpstt_offset as i32);
    let cur_masked = l.builder.ins().band_imm(cur, 7);
    let cur_i64 = l.builder.ins().uextend(types::I64, cur_masked);
    let one = l.builder.ins().iconst(types::I8, 1);
    store_fptag(l, cur_i64, one);

    let inc = l.builder.ins().iadd_imm(cur, 1);
    let new = l.builder.ins().band_imm(inc, 7);
    store_env_i32(l, x.fpstt_offset as i32, new);
    Ok(())
}

/// fdecstp(delta=-1) / fincstp(delta=+1).
///
/// Both adjust env->fpstt by delta and clear the same fpus bits
/// (0x4700 = C3|C2|C1|C0). Neither touches fptags — only fpush/fpop do.
fn lower_fdecstp_fincstp(
    l: &mut Lowering<'_, '_>,
    delta: i64,
) -> Result<(), TransError> {
    let x = l.env.x87.clone();
    let cur = read_env_i32(l, x.fpstt_offset as i32);
    let adj = l.builder.ins().iadd_imm(cur, delta);
    let new = l.builder.ins().band_imm(adj, 7);
    store_env_i32(l, x.fpstt_offset as i32, new);

    // fpus &= ~0x4700.
    let fpus = read_env_u16_zx32(l, x.fpus_offset as i32);
    let masked = l.builder.ins().band_imm(fpus, !0x4700i64);
    store_env_u16_from_i32(l, x.fpus_offset as i32, masked);
    Ok(())
}

// --- Constants -------------------------------------------------------

fn lower_const_push(l: &mut Lowering<'_, '_>, k: f64) -> Result<(), TransError> {
    lower_fpush(l)?;
    let kv = l.builder.ins().f64const(k);
    write_st_n_dyn(l, 0, kv);
    Ok(())
}

// --- Arithmetic ------------------------------------------------------

fn arith_emit(l: &mut Lowering<'_, '_>, a: Value, b: Value, op: BinOp) -> Value {
    match op {
        BinOp::Add  => l.builder.ins().fadd(a, b),
        BinOp::Mul  => l.builder.ins().fmul(a, b),
        BinOp::Sub  => l.builder.ins().fsub(a, b),
        BinOp::SubR => l.builder.ins().fsub(b, a),
        BinOp::Div  => l.builder.ins().fdiv(a, b),
        BinOp::DivR => l.builder.ins().fdiv(b, a),
    }
}

fn lower_arith_st0_ft0(
    l: &mut Lowering<'_, '_>,
    op: BinOp,
) -> Result<(), TransError> {
    let x = l.env.x87.clone();
    let st0 = read_st_n_dyn(l, 0);
    let ft0 = read_env_f64(l, x.ft0_native_offset as i32);
    let r = arith_emit(l, st0, ft0, op);
    write_st_n_dyn(l, 0, r);
    Ok(())
}

fn lower_arith_stn_st0(
    l: &mut Lowering<'_, '_>,
    n_i64: Value,
    op: BinOp,
) -> Result<(), TransError> {
    // Note operand ordering mirrors helper_f{add,sub,sub r,…}_STN_ST0 in
    // fpu_helper.c — see ST(st_index) = floatx80_X(ST(st_index), ST0).
    let stn = read_st_n_dynval(l, n_i64);
    let st0 = read_st_n_dyn(l, 0);
    let r = arith_emit(l, stn, st0, op);
    write_st_n_dynval(l, n_i64, r);
    Ok(())
}

// `read_env_u8` is kept for future fptag-batch readers — silence the
// unused-fn lint without rendering the helper unreachable.
#[allow(dead_code)]
const _READ_ENV_U8_KEEP: fn(&mut Lowering<'_, '_>, i32) -> Value = read_env_u8;
#[allow(dead_code)]
const _BLOCK_ARG_KEEP: fn() -> Option<BlockArg> = || None;
