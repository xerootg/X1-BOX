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
    AbiParam, BlockArg, InstBuilder, Signature, types,
};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::isa::CallConv;

use crate::ir::DecodedOp;
use crate::opc::TcgType;
use crate::translator::{Lowering, TransError};

pub(crate) fn lower_call(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    lower_call_impl(l, op)
}

fn lower_call_impl(l: &mut Lowering<'_, '_>, op: &DecodedOp) -> Result<(), TransError> {
    // Helper function pointer is the first cargs entry. Some op
    // generators put a struct pointer here whose first field is the
    // function pointer; for now we treat carg(0) as the helper
    // address directly (TCG sets it that way after optimization).
    let helper_ptr = op.carg(0);
    if helper_ptr == 0 {
        return Err(TransError::InvalidIr("call with null helper ptr"));
    }

    // Specialisation: `helper_cc_compute_all(dst, src1, src2, op)`
    // is one of the hottest helpers in the x86 frontend (~1.25% of
    // Halo 2 title-screen CPU). Inline the trivial CC_OP_EFLAGS
    // (=0, returns src1) fast path; fall through to the real call
    // for the ~50 other CC_OP cases. CC_OP is dynamic at runtime
    // (the optimizer doesn't fold the call away when CC_OP is in
    // the live env), so the branch is decided at execution time.
    //
    // Layout from target/i386/helper.h:
    //   DEF_HELPER_FLAGS_4(cc_compute_all, NO_RWG_SE,
    //                      tl, tl, tl, tl, int)
    // -> 4 inputs, no env (NO_RWG_SE strips it), 1 output.
    if l.env.cc_compute_all_fn != 0
        && helper_ptr == l.env.cc_compute_all_fn
        && op.nb_iargs == 4
        && op.nb_oargs == 1
    {
        return lower_cc_compute_all_inline(l, op, helper_ptr);
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

    // Marshal inputs - read every iarg as an i64.
    //
    // read_iarg now routes through `coerce` which handles float types
    // correctly (bitcast → extend → bitcast), so requesting TcgType::I64
    // gives back an I64-typed value even when the underlying temp was
    // declared as F32/F64. The old manual `uextend(I64, v)` fallback
    // here was the second site of the `VerifierFailed=22` bucket: when
    // an SSE-helper arg temp was first written by an FP op, v came back
    // as F32 and uextend rejected it. Now coerce handles that path, and
    // we just trust the return type.
    let mut args = Vec::with_capacity(op.nb_iargs as usize);
    for i in 0..op.nb_iargs {
        let v = l.read_iarg(op, i as usize, TcgType::I64)?;
        args.push(v);
    }

    let inst = l.builder.ins().call_indirect(sig_ref, addr, &args);
    if op.nb_oargs >= 1 {
        let rets = l.builder.inst_results(inst);
        let ret = rets[0];
        l.write_temp(op.oarg(0), ret);
    }
    Ok(())
}

/// Inline-fastpath for `helper_cc_compute_all`. Branches on the
/// runtime `op` argument:
///   * `op == CC_OP_EFLAGS (0)` -> result = src1
///   * otherwise -> result = call helper(dst, src1, src2, op)
///
/// The if-ladder is intentionally cheap: one icmp + one brif. When
/// the guest is in a steady "lazy eflags already materialised"
/// state (post-sti/popf/iret/cpuid), this collapses to a single
/// register move per lazy-eflags consumer.
fn lower_cc_compute_all_inline(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    helper_ptr: u64,
) -> Result<(), TransError> {
    let dst  = l.read_iarg(op, 0, TcgType::I64)?;
    let src1 = l.read_iarg(op, 1, TcgType::I64)?;
    let src2 = l.read_iarg(op, 2, TcgType::I64)?;
    let opv  = l.read_iarg(op, 3, TcgType::I64)?;

    let helper_block = l.builder.create_block();
    let merge_block  = l.builder.create_block();
    let merged       = l.builder.append_block_param(merge_block, types::I64);

    // if op == 0 -> jump merge(src1), else fall to helper_block
    let is_eflags = l.builder.ins().icmp_imm(IntCC::Equal, opv, 0);
    l.builder.ins().brif(
        is_eflags,
        merge_block, &[BlockArg::Value(src1)],
        helper_block, &[] as &[BlockArg],
    );

    // Slow path: the real C helper.
    l.builder.switch_to_block(helper_block);
    l.builder.seal_block(helper_block);

    let mut sig = Signature::new(CallConv::SystemV);
    for _ in 0..4 {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));
    let sig_ref = l.builder.import_signature(sig);
    let addr = l
        .builder
        .ins()
        .iconst(l.host_ptr_ty, helper_ptr as i64);
    let call_inst = l
        .builder
        .ins()
        .call_indirect(sig_ref, addr, &[dst, src1, src2, opv]);
    let helper_ret = l.builder.inst_results(call_inst)[0];
    l.builder
        .ins()
        .jump(merge_block, &[BlockArg::Value(helper_ret)]);

    l.builder.switch_to_block(merge_block);
    l.builder.seal_block(merge_block);
    l.write_temp(op.oarg(0), merged);
    Ok(())
}
