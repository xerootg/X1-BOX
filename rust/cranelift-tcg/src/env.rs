//! Description of the guest CPU env layout as supplied by the C side.

use std::ffi::{c_char, CStr};

/// Per-global descriptor extracted from the env snapshot.
#[derive(Clone, Debug)]
pub struct GlobalDesc {
    pub offset: u32,
    pub size: u32,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct EnvDesc {
    pub env_size: u32,
    /// Absolute value of the negative offset from TCG_AREG0 (env) at
    /// which CPUNegativeOffsetState.tlb.f[fast_index(0)] lives.  The
    /// C side publishes the magnitude only; consumers compute the
    /// final addressing as `env - tlb_offset`.
    pub tlb_offset: u32,
    pub pc_offset: u32,
    pub guest_ptr_size: u32,
    pub host_ptr_size: u32,
    pub globals: Vec<GlobalDesc>,
    /// Address of `cranelift_chain_continue` in libxemu (accel/tcg/
    /// cpu-exec.c). Cranelift's GotoTb lowering calls this so chained
    /// TBs (audio mixing, video decode, animation loops) dispatch
    /// inline without a dispatcher round-trip per iteration. 0 means
    /// chain dispatch is disabled and we fall back to return-to-
    /// dispatcher.
    pub chain_continue_fn: u64,
    /// Address of `helper_lookup_tb_ptr`. The x86 frontend always emits
    /// this helper call immediately before `goto_ptr` (see
    /// tcg_gen_lookup_and_goto_ptr), but in tier-2 our goto_ptr
    /// lowering re-does the TB lookup via the chain helper, so the
    /// helper call is dead work. The translator pattern-matches the
    /// Call carg(0) against this address to elide it. 0 disables the
    /// elision (Call still emits, doubling the lookup cost per TB).
    pub lookup_tb_ptr_fn: u64,
    /// Address of `cranelift_helper_flcr(uint32_t mxcsr)` — the C-side
    /// MXCSR->FPCR translator. Tier-1 aarch64 inlines this as an MSR
    /// FPCR sequence; Cranelift IR has no MSR primitive so we emit a
    /// call_indirect here instead. 0 means flcr is unsupported and
    /// the lowering returns UnsupportedOp.
    pub flcr_fn: u64,
}

impl EnvDesc {
    /// Convenience: the signed offset to apply to the env pointer to
    /// reach the TLB array.  Returns `-(tlb_offset as i32)`; the C
    /// side stores the magnitude.
    pub fn tlb_signed_offset(&self) -> i32 {
        -(self.tlb_offset as i32)
    }

    /// Empty descriptor used by tests that do not exercise env access.
    pub fn dummy() -> Self {
        EnvDesc {
            env_size: 0x4000,
            tlb_offset: 0,
            pc_offset: 0,
            guest_ptr_size: 4,
            host_ptr_size: 8,
            globals: Vec::new(),
            chain_continue_fn: 0,
            lookup_tb_ptr_fn: 0,
            flcr_fn: 0,
        }
    }

    /// Build from the raw FFI struct. `name_pool` is a NUL-separated
    /// byte buffer; offsets reference into it.
    ///
    /// # Safety
    /// The caller must ensure `globals` points to `nb_globals * 3` u32s
    /// and `name_pool` is NUL-terminated and large enough to contain
    /// every name referenced from `globals[]`.
    pub unsafe fn from_raw(
        env_size: u32,
        tlb_offset: u32,
        pc_offset: u32,
        nb_globals: u32,
        globals: *const u32,
        name_pool: *const c_char,
        guest_ptr_size: u32,
        host_ptr_size: u32,
        chain_continue_fn: u64,
        lookup_tb_ptr_fn: u64,
        flcr_fn: u64,
    ) -> Self {
        let mut out = Vec::with_capacity(nb_globals as usize);
        if !globals.is_null() && nb_globals > 0 {
            // SAFETY: caller guarantee.
            let slice = unsafe {
                std::slice::from_raw_parts(globals, (nb_globals as usize) * 3)
            };
            for chunk in slice.chunks_exact(3) {
                let offset = chunk[0];
                let size = chunk[1];
                let name_off = chunk[2] as usize;
                let name = if name_pool.is_null() {
                    String::new()
                } else {
                    // SAFETY: caller guarantee plus bounds check via NUL.
                    unsafe {
                        let p = name_pool.add(name_off);
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                out.push(GlobalDesc { offset, size, name });
            }
        }
        EnvDesc {
            env_size,
            tlb_offset,
            pc_offset,
            guest_ptr_size,
            host_ptr_size,
            globals: out,
            chain_continue_fn,
            lookup_tb_ptr_fn,
            flcr_fn,
        }
    }
}
