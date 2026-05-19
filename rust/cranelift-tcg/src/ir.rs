//! Internal representation of the post-optimization TCG IR that the
//! C side hands us. Mirrors `struct CraneliftTcgOp` from the FFI header.

use crate::opc::{Op, TcgType};

/// A single TCG opcode after the optimizer pass.
///
/// Field order matches `CraneliftTcgOp` in `cranelift_bridge.h`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RawOp {
    pub opc: u16,
    pub nb_oargs: u8,
    pub nb_iargs: u8,
    pub nb_cargs: u8,
    pub type_: u8,
    pub flags: u16,
    /// Bit i set means args[i] is a TEMP_CONST value (not a temp idx).
    /// See CraneliftTcgOp::const_mask comment in cranelift_bridge.h.
    pub const_mask: u16,
    pub _pad: u32,
    pub args: [u64; 16],
}

/// Sanity-check that the layout matches the C side. Layout:
/// opc(2) + nb_oargs(1) + nb_iargs(1) + nb_cargs(1) + type_(1) +
/// flags(2) + const_mask(2) + pad(4) + args[16] (128). With u64
/// alignment the total is 144 bytes on every platform we target.
const _: () = {
    assert!(core::mem::size_of::<RawOp>() == 144);
};

/// Owned IR snapshot used by the worker thread once the C caller has
/// returned. We copy out of the C buffer so the caller can free it.
#[derive(Clone, Debug)]
pub struct OpSnapshot {
    pub ops: Vec<DecodedOp>,
}

#[derive(Clone, Debug)]
pub struct DecodedOp {
    pub op: Op,
    pub ty: TcgType,
    pub flags: u16,
    pub nb_oargs: u8,
    pub nb_iargs: u8,
    pub nb_cargs: u8,
    /// Bit i set means args[i] is a TEMP_CONST value (interpreted as
    /// signed int64) rather than a temp index. Covers positions
    /// 0..(nb_oargs + nb_iargs); cargs are always raw values.
    pub const_mask: u16,
    /// `args[0..nb_oargs]` = outputs (temp ids)
    /// `args[nb_oargs..nb_oargs+nb_iargs]` = inputs (temp ids OR consts per const_mask)
    /// `args[nb_oargs+nb_iargs..]` = consts (memop, label, condition, ...)
    pub args: [u64; 16],
}

impl DecodedOp {
    /// Output temp at the given index.
    pub fn oarg(&self, i: usize) -> u64 {
        debug_assert!(i < self.nb_oargs as usize);
        self.args[i]
    }
    /// Input temp at the given index. NOTE: may be a TEMP_CONST value
    /// if the corresponding bit is set in `const_mask`. Use
    /// `Lowering::read_iarg` to resolve into a Cranelift Value rather
    /// than calling read_temp directly on this.
    pub fn iarg(&self, i: usize) -> u64 {
        debug_assert!(i < self.nb_iargs as usize);
        self.args[self.nb_oargs as usize + i]
    }
    /// True if input arg `i` is a TEMP_CONST whose value sits in args[].
    pub fn iarg_is_const(&self, i: usize) -> bool {
        let pos = self.nb_oargs as usize + i;
        (self.const_mask >> pos) & 1 != 0
    }
    /// Constant arg at the given index.
    pub fn carg(&self, i: usize) -> u64 {
        let off = self.nb_oargs as usize + self.nb_iargs as usize;
        debug_assert!(i < self.nb_cargs as usize);
        self.args[off + i]
    }
}

impl OpSnapshot {
    /// Copy in a raw FFI buffer.
    pub fn from_raw(ptr: *const RawOp, len: usize) -> Self {
        if ptr.is_null() || len == 0 {
            return OpSnapshot { ops: Vec::new() };
        }
        // SAFETY: caller guarantees `[ptr, ptr+len)` is a valid slice for
        // the duration of this call; we copy out before returning.
        let raw_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        let mut ops = Vec::with_capacity(len);
        for raw in raw_slice {
            ops.push(DecodedOp {
                op: Op::from_raw(raw.opc),
                ty: TcgType::from_raw(raw.type_),
                flags: raw.flags,
                nb_oargs: raw.nb_oargs,
                nb_iargs: raw.nb_iargs,
                nb_cargs: raw.nb_cargs,
                const_mask: raw.const_mask,
                args: raw.args,
            });
        }
        OpSnapshot { ops }
    }
}
