//! TLB-entry layout the memory module needs to know in order to emit
//! the fast-path inline.
//!
//! These offsets MUST match `include/exec/tlb-common.h`. The
//! `tlb_fast_path_layout_check` C-side build assertion catches drift.

/// Offset of `addr_read` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_READ: u32 = 0;
/// Offset of `addr_write` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_WRITE: u32 = 4;
/// Offset of `addr_code` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_CODE: u32 = 8;
/// Offset of `addend` (host pointer to guest mem) inside a CPUTLBEntry.
/// Native pointer size on host - we always emit the load with a host
/// pointer width.
pub const TLB_ENTRY_ADDEND: u32 = 16;
/// Size of a single CPUTLBEntry in bytes.
pub const TLB_ENTRY_SIZE: u32 = 32;

/// Bits that mark a TLB entry as needing the slow path.
/// (TLB_NOTDIRTY | TLB_MMIO | TLB_WATCHPOINT | TLB_FORCE_SLOW)
pub const TLB_SLOW_FLAGS_MASK: u64 = 0x7F;

/// Page mask used to extract the TLB tag bits.
pub const TARGET_PAGE_MASK: u64 = !0xfff;

/// Memory operation size encoded in `MemOp`.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemOpSize {
    Mo8  = 0,
    Mo16 = 1,
    Mo32 = 2,
    Mo64 = 3,
}

impl MemOpSize {
    pub fn bytes(self) -> u32 {
        1u32 << (self as u32)
    }

    pub fn from_memop(memop: u32) -> Self {
        match memop & 0x07 {
            0 => MemOpSize::Mo8,
            1 => MemOpSize::Mo16,
            2 => MemOpSize::Mo32,
            _ => MemOpSize::Mo64,
        }
    }
}

/// Bit flags inside a TCG `MemOp`. Subset that the translator needs.
pub mod memop_flags {
    pub const MO_SIZE: u32  = 0x07;
    pub const MO_SIGN: u32  = 0x08;
    pub const MO_BSWAP: u32 = 0x10;
    pub const MO_LE: u32    = 0x00;
    pub const MO_BE: u32    = 0x10;
    pub const MO_AMASK: u32 = 0x70;
}
