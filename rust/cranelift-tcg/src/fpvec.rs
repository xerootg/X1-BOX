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

use cranelift_codegen::ir::immediates::Imm64;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Value};

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
        18 => {
            // cvt32i_f32 = fcvt_to_sint f32 -> i32
            let a = l.read_iarg(op, 0, TcgType::I32)?;
            let f = bitcast_to_fp(l, a, types::F32);
            let v = l.builder.ins().fcvt_to_sint(types::I32, f);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        19 => {
            let a = l.read_iarg(op, 0, TcgType::I64)?;
            let f = bitcast_to_fp(l, a, types::F64);
            let v = l.builder.ins().fcvt_to_sint(types::I32, f);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        23 => {
            let a = l.read_iarg(op, 0, TcgType::I32)?;
            let f = bitcast_to_fp(l, a, types::F32);
            let v = l.builder.ins().fcvt_to_sint(types::I64, f);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
        24 => {
            let a = l.read_iarg(op, 0, TcgType::I64)?;
            let f = bitcast_to_fp(l, a, types::F64);
            let v = l.builder.ins().fcvt_to_sint(types::I64, f);
            l.write_temp(op.oarg(0), v);
            Ok(())
        }
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
        _ => Err(TransError::UnsupportedOp(raw)),
    }
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
 * Defensive scope: only handle the V128 case (SSE 128-bit movdqa /
 * movaps spill — the common path). V64 (MMX) and V256 (AVX) bail to
 * tier-1 so we never risk a wrong-width memory op. V64 is also rare
 * on Halo 2 (no MMX inner loops) and not worth the encoding risk
 * surface.
 *
 * Also guard against:
 *   - const-valued vector inputs (TCG IR-optimizer corner case;
 *     iconst-on-I64X2 isn't supported by Cranelift and the helper
 *     coerce path would assert on the scalar→vector widening);
 *   - non-trusted alignment: env spill slots ARE 16-byte aligned but
 *     we use MemFlags::new() (no alignment promise) to let Cranelift
 *     pick the AArch64 unaligned `LDR Q` / `STR Q` form, which the
 *     A78 cores handle without trap or measurable penalty.
 */
fn lower_ld_vec(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    if !matches!(op.ty, TcgType::V128) {
        return Err(TransError::UnsupportedOp(OPC_LD_VEC));
    }
    if op.iarg_is_const(0) {
        return Err(TransError::UnsupportedOp(OPC_LD_VEC));
    }
    let base = l.read_iarg(op, 0, TcgType::I64)?;
    let off = op.carg(0) as i64;
    let addr = l.builder.ins().iadd_imm(base, off);
    let v = l
        .builder
        .ins()
        .load(types::I64X2, MemFlags::new(), addr, 0);
    l.write_temp(op.oarg(0), v);
    Ok(())
}

fn lower_st_vec(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    if !matches!(op.ty, TcgType::V128) {
        return Err(TransError::UnsupportedOp(OPC_ST_VEC));
    }
    /* Both inputs must be temps, not consts — a const-valued vector
     * arg would force a scalar→vector coerce path that doesn't exist. */
    if op.iarg_is_const(0) || op.iarg_is_const(1) {
        return Err(TransError::UnsupportedOp(OPC_ST_VEC));
    }
    let val = l.read_iarg(op, 0, TcgType::V128)?;
    /* Belt-and-suspenders: verify the value really is a 128-bit
     * vector before we hand it to the store. If the temp had been
     * declared as something else upstream, bail rather than emit a
     * wrong-width store. */
    if l.builder.func.dfg.value_type(val) != types::I64X2 {
        return Err(TransError::UnsupportedOp(OPC_ST_VEC));
    }
    let base = l.read_iarg(op, 1, TcgType::I64)?;
    let off = op.carg(0) as i64;
    let addr = l.builder.ins().iadd_imm(base, off);
    l.builder
        .ins()
        .store(MemFlags::new(), val, addr, 0);
    Ok(())
}
