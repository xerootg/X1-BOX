//! TCG helper-call lowering.
//!
//! `call` ops carry a function-pointer constant and a flags word.
//! TCG argument layout for a call op:
//!
//!   - oargs[0..nb_oargs] = output temps
//!   - iargs[0..nb_iargs] = input temps
//!   - cargs[0]           = TCGHelperInfo *, where `func` is the host pointer
//!   - cargs[1]           = TCG flags word
//!
//! In tier-2 we lower this as a Cranelift indirect-call against a
//! function signature derived from the helper info. For now we
//! synthesise an opaque signature with N i64 inputs and the recorded
//! output type; this matches TCG's reg-only marshalling on AArch64.

use cranelift_codegen::ir::{
    AbiParam, InstBuilder, Signature, types,
};
use cranelift_codegen::isa::CallConv;

use crate::ir::DecodedOp;
use crate::opc::TcgType;
use crate::translator::{Lowering, TransError};

/* TCG helper-call lowering. lower_call_impl works mechanically (carg(0)
 * is the func ptr per tcg_call_func, all args fit AAPCS64 integer
 * regs) and lifts Cranelift compile success from 45% to 55%, plus
 * Halo 2 title-screen FPS 7 -> 13 peak. BUT the generic helper-call
 * lowering introduces audio choppiness and cutscene video stutter —
 * timing-sensitive helpers (cc_compute_all, segment loads, MMX/SSE
 * math) likely don't tolerate being called from a Cranelift TB the
 * way they do from tier-1, even though the ABI marshalling itself is
 * correct. Possible culprits to investigate before re-enabling:
 *   - Helpers that mutate env state in ways Cranelift's globals
 *     plumbing doesn't reload on the next op (Cranelift loads/stores
 *     globals through env memory, so cross-call freshness should be
 *     fine, but verify)
 *   - Helpers expecting TCG_AREG0 (x19) to hold env via some inline
 *     asm or hidden contract
 *   - Helpers that longjmp via cpu_loop_exit_restore and skip our
 *     Cranelift epilogue, leaving guest state mid-update
 * For now bail to tier-1; the qemu_ld/qemu_st wins from the prior
 * commit are preserved. */
pub(crate) fn lower_call(_l: &mut Lowering<'_, '_>, _op: &DecodedOp) -> Result<(), TransError> {
    Err(TransError::UnsupportedOp(crate::opc::Opc::Call as u16))
}

#[allow(dead_code)]
fn lower_call_impl(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    // Helper function pointer is the first cargs entry. Some op
    // generators put a struct pointer here whose first field is the
    // function pointer; for now we treat carg(0) as the helper
    // address directly (TCG sets it that way after optimization).
    let helper_ptr = op.carg(0);
    if helper_ptr == 0 {
        return Err(TransError::InvalidIr("call with null helper ptr"));
    }

    // Build a signature: each input becomes an i64 by value, single
    // i64 return value if nb_oargs == 1, none otherwise.
    let mut sig = Signature::new(CallConv::SystemV);
    for _ in 0..op.nb_iargs {
        sig.params.push(AbiParam::new(types::I64));
    }
    if op.nb_oargs >= 1 {
        sig.returns.push(AbiParam::new(types::I64));
    }
    let sig_ref = l.builder.import_signature(sig);

    // Materialise the helper address as a constant pointer.
    let host_ptr_ty = l.host_ptr_ty;
    let addr = l
        .builder
        .ins()
        .iconst(host_ptr_ty, helper_ptr as i64);

    // Marshal inputs - read every iarg as an i64 (zero-extended where
    // narrower).
    let mut args = Vec::with_capacity(op.nb_iargs as usize);
    for i in 0..op.nb_iargs {
        let v = l.read_iarg(op, i as usize, TcgType::I64)?;
        let v_ty = l.builder.func.dfg.value_type(v);
        let v64 = if v_ty != types::I64 {
            l.builder.ins().uextend(types::I64, v)
        } else {
            v
        };
        args.push(v64);
    }

    let inst = l.builder.ins().call_indirect(sig_ref, addr, &args);
    if op.nb_oargs >= 1 {
        let rets = l.builder.inst_results(inst);
        let ret = rets[0];
        l.write_temp(op.oarg(0), ret);
    }
    Ok(())
}
