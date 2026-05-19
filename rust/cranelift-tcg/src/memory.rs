//! Guest memory operations - `qemu_ld_*` / `qemu_st_*` lowering.
//!
//! The TLB fast-path mirrors what `tcg/aarch64/tcg-target.c.inc` emits:
//!
//! ```text
//!   ; TLB descriptor: CPUTLBDescFast {mask, table} at env - tlb_ofs
//!   ldp  mask, table, [env, #-tlb_ofs]
//!   ; idx = (addr & mask >> CPU_TLB_ENTRY_BITS_PRE_SHIFT) — but `mask`
//!   ; is already pre-shifted so `addr & mask >> shift` simplifies to:
//!   lsr  tmp, addr, #(TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS)
//!   and  tmp, tmp, mask           ; byte offset into table[]
//!   add  entry, table, tmp        ; CPUTLBEntry *
//!   ldr  tag,    [entry, #addr_read|addr_write]
//!   ldr  addend, [entry, #addend]
//!   ; Compare tag to (addr & (TARGET_PAGE_MASK | align_mask))
//!   and  tmp2, addr, #(TARGET_PAGE_MASK | a_mask)
//!   cmp  tag, tmp2
//!   b.ne slow_path
//!   ; Fast path: host_ptr = addend + addr
//!   add  host_ptr, addend, addr
//!   ldr  val, [host_ptr]
//! ```
//!
//! On miss we tail-call the QEMU helper:
//!   `extern uint64_t helper_ldul_mmu(env, addr, oi, retaddr)`
//! where `oi = MemOpIdx = memop | (mmu_idx << 4)`. retaddr is the
//! return address into the TB so the helper can find the TB on
//! exception unwinding; we pass `0` because we don't unwind into
//! Cranelift code (it would need the same per-insn search map TCG has).

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, Signature, Type, Value};
use cranelift_codegen::isa::CallConv;

use crate::ir::DecodedOp;
use crate::opc::TcgType;
use crate::tlb::{memop_flags, MemOpSize, TARGET_PAGE_MASK, TLB_ENTRY_ADDEND};
use crate::translator::{Lowering, TransError};

fn type_for_size(size: MemOpSize) -> Type {
    match size {
        MemOpSize::Mo8 => types::I8,
        MemOpSize::Mo16 => types::I16,
        MemOpSize::Mo32 => types::I32,
        MemOpSize::Mo64 => types::I64,
    }
}

struct Memop {
    size: MemOpSize,
    signed: bool,
    bswap: bool,
    /// The MemOp bits ONLY (raw_idx >> 5). What the helper-table
    /// lookup uses to find the right `helper_ld*_mmu` function.
    memop: u32,
    /// The original packed `MemOpIdx` from the TCG carg, ready to
    /// pass to the helper as its `oi` argument.
    raw_idx: u32,
    mmu_idx: u32,
}

/// Decode a MemOpIdx (the carg attached to qemu_ld / qemu_st ops).
/// Layout: `(memop << 5) | mmu_idx`, with mmu_idx <= 31. See
/// include/exec/memopidx.h for the authoritative packing.
fn decode_memop(raw: u32) -> Memop {
    let mmu_idx = raw & 0x1f;
    let memop = raw >> 5;
    Memop {
        size: MemOpSize::from_memop(memop),
        signed: (memop & memop_flags::MO_SIGN) != 0,
        bswap: (memop & memop_flags::MO_BSWAP) != 0,
        memop,
        raw_idx: raw,
        mmu_idx,
    }
}

/// Offsets within `CPUTLBEntry` (from include/exec/tlb-common.h).
///
/// Layout for 64-bit host:
///   0:  addr_read   (uintptr_t)
///   8:  addr_write  (uintptr_t)
///   16: addr_code   (uintptr_t)
///   24: addend      (uintptr_t)
const TLB_ENTRY_ADDR_READ: i32 = 0;
const TLB_ENTRY_ADDR_WRITE: i32 = 8;
#[allow(dead_code)]
const TLB_ENTRY_ADDR_CODE: i32 = 16;
const TLB_ENTRY_ADDEND_64: i32 = TLB_ENTRY_ADDEND as i32;

/// `CPUTLBDescFast` layout (include/exec/tlb-common.h):
///   0: mask  (uintptr_t)        - already pre-shifted by CPU_TLB_ENTRY_BITS
///   8: table (CPUTLBEntry *)
const TLB_DESC_FAST_MASK: i32 = 0;
const TLB_DESC_FAST_TABLE: i32 = 8;

/// Per-mmu-idx stride within `CPUTLB.f[NB_MMU_MODES]`.
/// `sizeof(CPUTLBDescFast)` is 16 on 64-bit hosts.
const TLB_DESC_FAST_STRIDE: i32 = 16;

/// Page-table shift on x86 guests (4 KiB pages).
const TARGET_PAGE_BITS: i64 = 12;

/// CPUTLBEntry alignment shift used to derive byte offset from index.
/// `CPU_TLB_ENTRY_BITS = 5` on 64-bit; the `mask` field is already
/// pre-shifted so we don't multiply by entry size here.
const CPU_TLB_ENTRY_BITS: i64 = 5;

struct FastPath {
    host_ptr: Value,
    slow_block: cranelift_codegen::ir::Block,
    rejoin: cranelift_codegen::ir::Block,
}

/// Emit the TLB fast-path test. Returns the (host_ptr, slow_block,
/// rejoin_block) tuple. On entry the builder is in the "pre-test"
/// block; on exit it's positioned in the fast-path branch with
/// `host_ptr` valid. Caller emits the actual load/store, jumps to
/// `rejoin`, then switches to `slow_block` to emit the miss handler.
fn emit_tlb_fastpath(
    l: &mut Lowering<'_, '_>,
    guest_addr: Value,
    mmu_idx: u32,
    size: MemOpSize,
    write: bool,
) -> Result<FastPath, TransError> {
    let env = l.env_val;
    let host_ptr_ty = l.host_ptr_ty;

    // Compute env-relative offset of CPUTLBDescFast.f[mmu_idx]. The
    // C side publishes `tlb_offset` as the absolute value of the
    // negative offset to .f[NB_MMU_MODES-1-0]. For mmu_idx 0 we
    // subtract; for higher idx we move forward by the stride.
    let base_neg = -(l.env.tlb_offset as i32);
    let f_off = base_neg + (mmu_idx as i32) * TLB_DESC_FAST_STRIDE;

    // Load mask and table. They live consecutively, so we issue two
    // loads from the same base; Cranelift's regalloc will coalesce.
    let mask = l.builder.ins().load(
        host_ptr_ty,
        MemFlags::trusted(),
        env,
        f_off + TLB_DESC_FAST_MASK,
    );
    let table = l.builder.ins().load(
        host_ptr_ty,
        MemFlags::trusted(),
        env,
        f_off + TLB_DESC_FAST_TABLE,
    );

    // idx_bytes = (addr >> (TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS)) & mask
    let pre_shift = TARGET_PAGE_BITS - CPU_TLB_ENTRY_BITS;
    let addr_ext = if l.builder.func.dfg.value_type(guest_addr) != host_ptr_ty {
        l.builder.ins().uextend(host_ptr_ty, guest_addr)
    } else {
        guest_addr
    };
    let shifted = l.builder.ins().ushr_imm(addr_ext, pre_shift);
    let idx_bytes = l.builder.ins().band(shifted, mask);
    let entry = l.builder.ins().iadd(table, idx_bytes);

    // Compare tag.
    let tag_off = if write { TLB_ENTRY_ADDR_WRITE } else { TLB_ENTRY_ADDR_READ };
    let tag = l
        .builder
        .ins()
        .load(host_ptr_ty, MemFlags::trusted(), entry, tag_off);
    let align_mask: i64 = (TARGET_PAGE_MASK as i64) | ((size.bytes() as i64) - 1);
    let cmp_addr = l.builder.ins().band_imm(addr_ext, align_mask);
    let is_hit = l.builder.ins().icmp(IntCC::Equal, tag, cmp_addr);

    let fast_block = l.builder.create_block();
    let slow_block = l.builder.create_block();
    let rejoin = l.builder.create_block();
    l.builder
        .ins()
        .brif(is_hit, fast_block, &[] as &[BlockArg], slow_block, &[] as &[BlockArg]);
    l.builder.switch_to_block(fast_block);

    // host_ptr = addr + addend
    let addend = l
        .builder
        .ins()
        .load(host_ptr_ty, MemFlags::trusted(), entry, TLB_ENTRY_ADDEND_64);
    let host_ptr = l.builder.ins().iadd(addend, addr_ext);

    Ok(FastPath {
        host_ptr,
        slow_block,
        rejoin,
    })
}

/// Build the SystemV signature for a slow-path load helper:
///   `extern uint64_t helper_ld{q,ul,uw,ub,sl,sw,sb}_mmu(env, addr,
///                                                       oi, retaddr)`
fn ld_helper_sig() -> Signature {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // env
    sig.params.push(AbiParam::new(types::I64)); // addr
    sig.params.push(AbiParam::new(types::I32)); // memop_idx
    sig.params.push(AbiParam::new(types::I64)); // retaddr
    sig.returns.push(AbiParam::new(types::I64));
    sig
}

/// Store helper signature:
///   `extern void helper_st{b,w,l,q}_mmu(env, addr, val, oi, retaddr)`
fn st_helper_sig(val_ty: Type) -> Signature {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64)); // env
    sig.params.push(AbiParam::new(types::I64)); // addr
    sig.params.push(AbiParam::new(val_ty));
    sig.params.push(AbiParam::new(types::I32)); // memop_idx
    sig.params.push(AbiParam::new(types::I64)); // retaddr
    sig
}

pub(crate) fn lower_load(
    _l: &mut Lowering<'_, '_>,
    _op: &DecodedOp,
    _is_pair: bool,
) -> Result<(), TransError> {
    // Bail on every guest memory load. The TLB fast-path + helper
    // slow-path produced subtly wrong addresses (retaddr=0 + missing
    // unwind info crashed the helper). Until we have a verified
    // slow-path we just refuse to compile TBs that touch guest mem.
    Err(TransError::UnsupportedOp(crate::opc::Opc::QemuLd as u16))
    /* old body kept below, dead code:
    let addr_temp = op.iarg(0);
    let guest_addr = l.read_temp(addr_temp, TcgType::I64)?;
    let memop_raw = op.carg(0) as u32;
    let memop = decode_memop(memop_raw);

    let helper_ptr = l
        .helpers
        .ld_helper(memop.memop)
        .ok_or(TransError::UnsupportedOp(crate::opc::Opc::QemuLd as u16))?;
    */
}

#[allow(dead_code)]
fn lower_load_with_helper(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    is_pair: bool,
) -> Result<(), TransError> {
    let addr_temp = op.iarg(0);
    let guest_addr = l.read_temp(addr_temp, TcgType::I64)?;
    let memop_raw = op.carg(0) as u32;
    let memop = decode_memop(memop_raw);

    let helper_ptr = l
        .helpers
        .ld_helper(memop.memop)
        .ok_or(TransError::UnsupportedOp(crate::opc::Opc::QemuLd as u16))?;

    let fast = emit_tlb_fastpath(l, guest_addr, memop.mmu_idx, memop.size, false)?;

    // Fast path: direct load from host_ptr.
    let load_ty = type_for_size(memop.size);
    let mut v = l
        .builder
        .ins()
        .load(load_ty, MemFlags::new(), fast.host_ptr, 0);
    if memop.bswap {
        v = l.builder.ins().bswap(v);
    }
    let dst_ty = Lowering::type_for(op.ty);
    let v = if load_ty != dst_ty {
        if memop.signed {
            l.builder.ins().sextend(dst_ty, v)
        } else {
            l.builder.ins().uextend(dst_ty, v)
        }
    } else {
        v
    };
    l.builder.ins().jump(fast.rejoin, &[BlockArg::Value(v)]);

    // Slow path: call the real QEMU helper.
    l.builder.switch_to_block(fast.slow_block);
    let sig = l.builder.import_signature(ld_helper_sig());
    let helper_addr = l
        .builder
        .ins()
        .iconst(l.host_ptr_ty, helper_ptr as i64);
    let addr_i64 = if l.builder.func.dfg.value_type(guest_addr) != types::I64 {
        l.builder.ins().uextend(types::I64, guest_addr)
    } else {
        guest_addr
    };
    let env_i64 = l.env_val;
    let oi = l.builder.ins().iconst(types::I32, memop.raw_idx as i64);
    let retaddr = l.builder.ins().iconst(types::I64, 0);
    let call_inst = l.builder.ins().call_indirect(
        sig,
        helper_addr,
        &[env_i64, addr_i64, oi, retaddr],
    );
    let slow_v_raw = l.builder.inst_results(call_inst)[0];
    let slow_v = if l.builder.func.dfg.value_type(slow_v_raw) != dst_ty {
        if memop.signed {
            // Helper returns sign-extended already for signed loads;
            // truncate or extend without re-sign as needed.
            if dst_ty.bits() < 64 {
                l.builder.ins().ireduce(dst_ty, slow_v_raw)
            } else {
                slow_v_raw
            }
        } else if dst_ty.bits() < 64 {
            l.builder.ins().ireduce(dst_ty, slow_v_raw)
        } else {
            slow_v_raw
        }
    } else {
        slow_v_raw
    };
    l.builder
        .ins()
        .jump(fast.rejoin, &[BlockArg::Value(slow_v)]);

    let merged = l.builder.append_block_param(fast.rejoin, dst_ty);
    l.builder.switch_to_block(fast.rejoin);
    l.write_temp(op.oarg(0), merged);

    if is_pair {
        // qemu_ld2 outputs a pair; the high half is read by the next
        // sequential entry. Emit a second helper call to read it -
        // since we don't have a tagged fast-path for paired ops,
        // always use the slow path here.
        let helper_q = l
            .helpers
            .ld_helper(memop.memop | memop_flags::MO_SIZE)
            .unwrap_or(helper_ptr);
        let sig2 = l.builder.import_signature(ld_helper_sig());
        let h2 = l.builder.ins().iconst(l.host_ptr_ty, helper_q as i64);
        let addr_hi = l.builder.ins().iadd_imm(addr_i64, 8);
        let oi_hi = l.builder.ins().iconst(types::I32, memop.raw_idx as i64);
        let ret_hi = l.builder.ins().iconst(types::I64, 0);
        let inst_hi = l.builder.ins().call_indirect(
            sig2,
            h2,
            &[env_i64, addr_hi, oi_hi, ret_hi],
        );
        let hi = l.builder.inst_results(inst_hi)[0];
        l.write_temp(op.oarg(1), hi);
    }
    Ok(())
}

pub(crate) fn lower_store(
    _l: &mut Lowering<'_, '_>,
    _op: &DecodedOp,
    _is_pair: bool,
) -> Result<(), TransError> {
    // Same rationale as lower_load: bail to tier-1 on any guest store
    // until the slow-path helper ABI is verified end-to-end.
    Err(TransError::UnsupportedOp(crate::opc::Opc::QemuSt as u16))
}

#[allow(dead_code)]
fn lower_store_with_helper(
    l: &mut Lowering<'_, '_>,
    op: &DecodedOp,
    is_pair: bool,
) -> Result<(), TransError> {
    let val_temp = op.iarg(0);
    let addr_temp = if is_pair { op.iarg(2) } else { op.iarg(1) };
    let val = l.read_temp(val_temp, op.ty)?;
    let guest_addr = l.read_temp(addr_temp, TcgType::I64)?;
    let memop_raw = op.carg(0) as u32;
    let memop = decode_memop(memop_raw);

    let helper_ptr = l
        .helpers
        .st_helper(memop.memop)
        .ok_or(TransError::UnsupportedOp(crate::opc::Opc::QemuSt as u16))?;

    let fast = emit_tlb_fastpath(l, guest_addr, memop.mmu_idx, memop.size, true)?;

    // Fast path: direct store.
    let store_ty = type_for_size(memop.size);
    let cur = l.builder.func.dfg.value_type(val);
    let mut to_store = if cur != store_ty {
        if cur.bits() > store_ty.bits() {
            l.builder.ins().ireduce(store_ty, val)
        } else {
            l.builder.ins().uextend(store_ty, val)
        }
    } else {
        val
    };
    if memop.bswap {
        to_store = l.builder.ins().bswap(to_store);
    }
    l.builder
        .ins()
        .store(MemFlags::new(), to_store, fast.host_ptr, 0);

    if is_pair {
        let val_hi = l.read_iarg(op, 1, op.ty)?;
        let hi_ty = l.builder.func.dfg.value_type(val_hi);
        let to_store_hi = if hi_ty != types::I64 {
            l.builder.ins().uextend(types::I64, val_hi)
        } else {
            val_hi
        };
        l.builder
            .ins()
            .store(MemFlags::new(), to_store_hi, fast.host_ptr, 8);
    }

    l.builder.ins().jump(fast.rejoin, &[] as &[BlockArg]);

    // Slow path: call the store helper.
    l.builder.switch_to_block(fast.slow_block);
    let sig = l.builder.import_signature(st_helper_sig(store_ty));
    let helper_addr = l
        .builder
        .ins()
        .iconst(l.host_ptr_ty, helper_ptr as i64);
    let addr_i64 = if l.builder.func.dfg.value_type(guest_addr) != types::I64 {
        l.builder.ins().uextend(types::I64, guest_addr)
    } else {
        guest_addr
    };
    let env_i64 = l.env_val;
    // Coerce val to the helper's expected width.
    let cur_v = l.builder.func.dfg.value_type(val);
    let val_arg = if cur_v != store_ty {
        if cur_v.bits() > store_ty.bits() {
            l.builder.ins().ireduce(store_ty, val)
        } else {
            l.builder.ins().uextend(store_ty, val)
        }
    } else {
        val
    };
    let oi = l.builder.ins().iconst(types::I32, memop.raw_idx as i64);
    let retaddr = l.builder.ins().iconst(types::I64, 0);
    l.builder.ins().call_indirect(
        sig,
        helper_addr,
        &[env_i64, addr_i64, val_arg, oi, retaddr],
    );
    l.builder.ins().jump(fast.rejoin, &[] as &[BlockArg]);

    l.builder.switch_to_block(fast.rejoin);
    Ok(())
}
