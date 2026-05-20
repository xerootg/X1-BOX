//! TLB-entry layout the memory module needs to know in order to emit
//! the fast-path inline.
//!
//! These offsets MUST match `include/exec/tlb-common.h`. The
//! `tlb_fast_path_layout_check` C-side build assertion catches drift.

/* CPUTLBEntry layout (see include/exec/tlb-common.h):
 *   struct { uintptr_t addr_read, addr_write, addr_code, addend; };
 * uintptr_t == host pointer width. Cranelift is gated to aarch64
 * Android (64-bit host), so each field is 8 bytes and the next field
 * starts 8 bytes later. The C-side QEMU_BUILD_BUG_ONs in
 * accel/tcg/cputlb.c:109-114 pin the addr_{read,write,code} positions
 * to MMU_{DATA_LOAD,DATA_STORE,INST_FETCH} * sizeof(uintptr_t). */
/// Offset of `addr_read` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_READ: u32 = 0;
/// Offset of `addr_write` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_WRITE: u32 = 8;
/// Offset of `addr_code` inside a CPUTLBEntry.
pub const TLB_ENTRY_ADDR_CODE: u32 = 16;
/// Offset of `addend` (host pointer to guest mem) inside a CPUTLBEntry.
/// Was previously 16, which silently read `addr_code` and used it as
/// the addend — guaranteed wild host_ptr on every fast-path hit.
pub const TLB_ENTRY_ADDEND: u32 = 24;
/// Size of a single CPUTLBEntry in bytes (1 << CPU_TLB_ENTRY_BITS).
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
