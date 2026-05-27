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
    /// Phase 3 (tier-2 TB chaining): address of
    /// `cranelift_chain_continue_v2` — slow path of GotoTb chain-slot
    /// miss; installs the target's SystemV entry into *from_slot
    /// before continuing the dispatch loop. 0 = chaining disabled,
    /// GotoTb stays on the legacy `chain_continue_fn` path.
    pub chain_continue_v2_fn: u64,
    /// Phase 3: signed byte offset from env to `cpu->interrupt_request`.
    /// Negative on every aarch64 build (CPUState precedes CPUArchState
    /// via CPUNegativeOffsetState). 0 = unknown; emitter must omit the
    /// IRQ check and fall back to legacy chain_continue.
    pub cpu_interrupt_request_offset: i32,
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
    /// Address of `helper_cc_compute_all(dst, src1, src2, op)` — the
    /// lazy-eflags materialisation helper from target/i386/tcg/
    /// cc_helper.c. The translator pattern-matches Call carg(0)
    /// against this and replaces the call with an inline if-ladder
    /// that handles CC_OP_EFLAGS (0 -> return src1) directly and
    /// falls through to a real call for everything else. 0 disables
    /// the inline path.
    pub cc_compute_all_fn: u64,

    /// Native fp80 inline lowering. Populated by the C side from
    /// `offsetof(CPUX86State, …)` and the resolved helper-symbol
    /// addresses (`__hard` variants on AArch64 HARD_FPU). If a helper
    /// pointer is 0, lower_call falls back to call_indirect for that
    /// op. See `helper::lower_call_impl` for the pattern-match.
    pub x87: X87Layout,
}

/// Offsets within `CPUX86State` and resolved helper-symbol addresses
/// the inline x87 lowering needs. All offsets are byte counts from the
/// `env` pointer (or, for `fpreg_native_d_off`, from the start of one
/// `FPReg` slot).
#[derive(Clone, Debug, Default)]
pub struct X87Layout {
    pub fpregs_offset: u32,
    pub fpreg_stride: u32,
    pub fpreg_native_d_off: u32,
    pub fpstt_offset: u32,
    pub ft0_native_offset: u32,
    pub fpus_offset: u32,
    pub fptags_offset: u32,
    pub cc_src_offset: u32,
    pub cc_dst_offset: u32,
    pub cc_src2_offset: u32,
    pub cc_op_offset: u32,

    pub fucom_st0_ft0_fn: u64,
    pub fcomi_st0_ft0_fn: u64,
    pub fucomi_st0_ft0_fn: u64,
    pub fxam_st0_fn: u64,
    pub fldl2t_st0_fn: u64,
    pub fldl2e_st0_fn: u64,
    pub fldpi_st0_fn: u64,
    pub fldlg2_st0_fn: u64,
    pub fldln2_st0_fn: u64,
    pub fld1_st0_fn: u64,
    pub fldz_st0_fn: u64,
    pub fldz_ft0_fn: u64,
    pub fpush_fn: u64,
    pub fpop_fn: u64,
    pub fdecstp_fn: u64,
    pub fincstp_fn: u64,
    pub ffree_stn_fn: u64,
    pub fmov_st0_ft0_fn: u64,
    pub fmov_ft0_stn_fn: u64,
    pub fmov_st0_stn_fn: u64,
    pub fmov_stn_st0_fn: u64,
    pub fxchg_st0_stn_fn: u64,
    pub fchs_st0_fn: u64,
    pub fabs_st0_fn: u64,
    pub fsqrt_fn: u64,
    pub fadd_st0_ft0_fn: u64,
    pub fmul_st0_ft0_fn: u64,
    pub fsub_st0_ft0_fn: u64,
    pub fsubr_st0_ft0_fn: u64,
    pub fdiv_st0_ft0_fn: u64,
    pub fdivr_st0_ft0_fn: u64,
    pub fadd_stn_st0_fn: u64,
    pub fmul_stn_st0_fn: u64,
    pub fsub_stn_st0_fn: u64,
    pub fsubr_stn_st0_fn: u64,
    pub fdiv_stn_st0_fn: u64,
    pub fdivr_stn_st0_fn: u64,
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
            chain_continue_v2_fn: 0,
            cpu_interrupt_request_offset: 0,
            lookup_tb_ptr_fn: 0,
            flcr_fn: 0,
            cc_compute_all_fn: 0,
            x87: X87Layout::default(),
        }
    }

    /// Build from the raw FFI struct. `name_pool` is a NUL-separated
    /// byte buffer; offsets reference into it.
    ///
    /// # Safety
    /// The caller must ensure `globals` points to `nb_globals * 3` u32s
    /// and `name_pool` is NUL-terminated and large enough to contain
    /// every name referenced from `globals[]`.
    #[allow(clippy::too_many_arguments)]
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
        chain_continue_v2_fn: u64,
        cpu_interrupt_request_offset: i32,
        lookup_tb_ptr_fn: u64,
        flcr_fn: u64,
        cc_compute_all_fn: u64,
        x87: X87Layout,
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
            chain_continue_v2_fn,
            cpu_interrupt_request_offset,
            lookup_tb_ptr_fn,
            flcr_fn,
            cc_compute_all_fn,
            x87,
        }
    }
}
