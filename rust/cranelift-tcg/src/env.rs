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
        }
    }
}
