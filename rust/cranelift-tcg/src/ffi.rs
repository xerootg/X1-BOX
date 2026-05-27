//! C ABI surface. Mirrors `include/tcg/cranelift_bridge.h`.

use std::ffi::{c_char, c_void};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use once_cell::sync::OnceCell;

use crate::context::JitContext;
use crate::env::EnvDesc;
use crate::ir::{OpSnapshot, RawOp};
use crate::translator::Translator;

pub const CRANELIFT_TCG_OK: i32 = 0;
pub const CRANELIFT_TCG_ERR_NOT_IMPLEMENTED: i32 = -1;
pub const CRANELIFT_TCG_ERR_UNSUPPORTED_OP: i32 = -2;
pub const CRANELIFT_TCG_ERR_OUT_OF_MEMORY: i32 = -3;
pub const CRANELIFT_TCG_ERR_INVALID_IR: i32 = -4;
pub const CRANELIFT_TCG_ERR_VERIFIER_FAILED: i32 = -5;
pub const CRANELIFT_TCG_ERR_INTERNAL: i32 = -6;
pub const CRANELIFT_TCG_ERR_DISABLED: i32 = -7;
pub const CRANELIFT_TCG_ERR_BLACKLISTED: i32 = -8;

/// Mirror of `CraneliftTcgEnvDesc` in the C header.
#[repr(C)]
pub struct CraneliftTcgEnvDesc {
    pub env_size: u32,
    pub tlb_offset: u32,
    pub pc_offset: u32,
    pub nb_globals: u32,
    pub globals: *const u32,
    pub name_pool: *const c_char,
    pub guest_ptr_size: u32,
    pub host_ptr_size: u32,
    pub chain_continue_fn: usize,
    pub lookup_tb_ptr_fn: usize,
    pub flcr_fn: usize,
    pub cc_compute_all_fn: usize,

    /*
     * Native fp80 inline lowering surface (HARD_FPU mode).
     * Mirrors the matching fields in include/tcg/cranelift_bridge.h.
     * Helper pointers are 0 when the C side hasn't enabled inline
     * lowering for that op; in that case lower_call falls back to the
     * legacy call_indirect path.
     */
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
    pub x87_pad0: u32,

    pub x87_fucom_st0_ft0_fn: usize,
    pub x87_fcomi_st0_ft0_fn: usize,
    pub x87_fucomi_st0_ft0_fn: usize,
    pub x87_fxam_st0_fn: usize,
    pub x87_fldl2t_st0_fn: usize,
    pub x87_fldl2e_st0_fn: usize,
    pub x87_fldpi_st0_fn: usize,
    pub x87_fldlg2_st0_fn: usize,
    pub x87_fldln2_st0_fn: usize,
    pub x87_fld1_st0_fn: usize,
    pub x87_fldz_st0_fn: usize,
    pub x87_fldz_ft0_fn: usize,
    pub x87_fpush_fn: usize,
    pub x87_fpop_fn: usize,
    pub x87_fdecstp_fn: usize,
    pub x87_fincstp_fn: usize,
    pub x87_ffree_stn_fn: usize,
    pub x87_fmov_st0_ft0_fn: usize,
    pub x87_fmov_ft0_stn_fn: usize,
    pub x87_fmov_st0_stn_fn: usize,
    pub x87_fmov_stn_st0_fn: usize,
    pub x87_fxchg_st0_stn_fn: usize,
    pub x87_fchs_st0_fn: usize,
    pub x87_fabs_st0_fn: usize,
    pub x87_fsqrt_fn: usize,
    pub x87_fadd_st0_ft0_fn: usize,
    pub x87_fmul_st0_ft0_fn: usize,
    pub x87_fsub_st0_ft0_fn: usize,
    pub x87_fsubr_st0_ft0_fn: usize,
    pub x87_fdiv_st0_ft0_fn: usize,
    pub x87_fdivr_st0_ft0_fn: usize,
    pub x87_fadd_stn_st0_fn: usize,
    pub x87_fmul_stn_st0_fn: usize,
    pub x87_fsub_stn_st0_fn: usize,
    pub x87_fsubr_stn_st0_fn: usize,
    pub x87_fdiv_stn_st0_fn: usize,
    pub x87_fdivr_stn_st0_fn: usize,
}

/// Mirror of `CraneliftTcgStats` in the C header.
#[repr(C)]
pub struct CraneliftTcgStats {
    pub enqueued: u64,
    pub compiled_ok: u64,
    pub compiled_err: u64,
    pub fallback_unsupported_op: u64,
    pub blacklisted: u64,
    pub verify_ok: u64,
    pub verify_divergence: u64,
    pub total_compile_ns: u64,
    pub total_emitted_bytes: u64,
    pub active_entries: u32,
    pub worker_queue_depth: u32,
}

/// Global handle - the C side uses an opaque pointer but we only ever
/// have one context per process.
static GLOBAL: OnceCell<Arc<JitContext>> = OnceCell::new();

fn ctx_from_handle(handle: *const c_void) -> Option<Arc<JitContext>> {
    if handle.is_null() {
        return GLOBAL.get().cloned();
    }
    // The handle is the pointer we returned from init - which is the
    // pointer into the OnceCell-held Arc. We just look it back up.
    GLOBAL.get().cloned()
}

/// Install a panic hook that logs the message via __android_log_print
/// before the runtime abort()s. Without this, `panic = "abort"` would
/// terminate the process silently (no logcat trace, no tombstone hint).
fn install_panic_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let location = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown>".to_string());
            let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string payload>".to_string()
            };
            crate::dispatcher::log_to_android(&format!(
                "RUST PANIC at {location}: {payload}"
            ));
        }));
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_init(
    env: *const CraneliftTcgEnvDesc,
) -> *mut c_void {
    install_panic_hook();
    if let Some(existing) = GLOBAL.get() {
        return Arc::as_ptr(existing) as *mut c_void;
    }
    let desc = if env.is_null() {
        EnvDesc::dummy()
    } else {
        // SAFETY: caller guarantees `env` is a valid pointer for the
        // duration of this call.
        let e = unsafe { &*env };
        let x87 = crate::env::X87Layout {
            fpregs_offset: e.fpregs_offset,
            fpreg_stride: e.fpreg_stride,
            fpreg_native_d_off: e.fpreg_native_d_off,
            fpstt_offset: e.fpstt_offset,
            ft0_native_offset: e.ft0_native_offset,
            fpus_offset: e.fpus_offset,
            fptags_offset: e.fptags_offset,
            cc_src_offset: e.cc_src_offset,
            cc_dst_offset: e.cc_dst_offset,
            cc_src2_offset: e.cc_src2_offset,
            cc_op_offset: e.cc_op_offset,
            fucom_st0_ft0_fn: e.x87_fucom_st0_ft0_fn as u64,
            fcomi_st0_ft0_fn: e.x87_fcomi_st0_ft0_fn as u64,
            fucomi_st0_ft0_fn: e.x87_fucomi_st0_ft0_fn as u64,
            fxam_st0_fn: e.x87_fxam_st0_fn as u64,
            fldl2t_st0_fn: e.x87_fldl2t_st0_fn as u64,
            fldl2e_st0_fn: e.x87_fldl2e_st0_fn as u64,
            fldpi_st0_fn: e.x87_fldpi_st0_fn as u64,
            fldlg2_st0_fn: e.x87_fldlg2_st0_fn as u64,
            fldln2_st0_fn: e.x87_fldln2_st0_fn as u64,
            fld1_st0_fn: e.x87_fld1_st0_fn as u64,
            fldz_st0_fn: e.x87_fldz_st0_fn as u64,
            fldz_ft0_fn: e.x87_fldz_ft0_fn as u64,
            fpush_fn: e.x87_fpush_fn as u64,
            fpop_fn: e.x87_fpop_fn as u64,
            fdecstp_fn: e.x87_fdecstp_fn as u64,
            fincstp_fn: e.x87_fincstp_fn as u64,
            ffree_stn_fn: e.x87_ffree_stn_fn as u64,
            fmov_st0_ft0_fn: e.x87_fmov_st0_ft0_fn as u64,
            fmov_ft0_stn_fn: e.x87_fmov_ft0_stn_fn as u64,
            fmov_st0_stn_fn: e.x87_fmov_st0_stn_fn as u64,
            fmov_stn_st0_fn: e.x87_fmov_stn_st0_fn as u64,
            fxchg_st0_stn_fn: e.x87_fxchg_st0_stn_fn as u64,
            fchs_st0_fn: e.x87_fchs_st0_fn as u64,
            fabs_st0_fn: e.x87_fabs_st0_fn as u64,
            fsqrt_fn: e.x87_fsqrt_fn as u64,
            fadd_st0_ft0_fn: e.x87_fadd_st0_ft0_fn as u64,
            fmul_st0_ft0_fn: e.x87_fmul_st0_ft0_fn as u64,
            fsub_st0_ft0_fn: e.x87_fsub_st0_ft0_fn as u64,
            fsubr_st0_ft0_fn: e.x87_fsubr_st0_ft0_fn as u64,
            fdiv_st0_ft0_fn: e.x87_fdiv_st0_ft0_fn as u64,
            fdivr_st0_ft0_fn: e.x87_fdivr_st0_ft0_fn as u64,
            fadd_stn_st0_fn: e.x87_fadd_stn_st0_fn as u64,
            fmul_stn_st0_fn: e.x87_fmul_stn_st0_fn as u64,
            fsub_stn_st0_fn: e.x87_fsub_stn_st0_fn as u64,
            fsubr_stn_st0_fn: e.x87_fsubr_stn_st0_fn as u64,
            fdiv_stn_st0_fn: e.x87_fdiv_stn_st0_fn as u64,
            fdivr_stn_st0_fn: e.x87_fdivr_stn_st0_fn as u64,
        };
        unsafe {
            EnvDesc::from_raw(
                e.env_size,
                e.tlb_offset,
                e.pc_offset,
                e.nb_globals,
                e.globals,
                e.name_pool,
                e.guest_ptr_size,
                e.host_ptr_size,
                e.chain_continue_fn as u64,
                e.lookup_tb_ptr_fn as u64,
                e.flcr_fn as u64,
                e.cc_compute_all_fn as u64,
                x87,
            )
        }
    };
    let ctx = JitContext::new(desc);
    let ptr = Arc::as_ptr(&ctx) as *mut c_void;
    let _ = GLOBAL.set(ctx);
    ptr
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_destroy(_handle: *mut c_void) {
    if let Some(ctx) = GLOBAL.get() {
        ctx.shutdown();
    }
    // We deliberately do not clear GLOBAL: a second init() should
    // return the same context. The worker thread is gone, so further
    // enqueues will fail with ERR_INTERNAL until re-init.
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_set_enabled(
    handle: *mut c_void,
    enabled: i32,
) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.config.enabled.store(enabled != 0, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_is_enabled(handle: *const c_void) -> i32 {
    ctx_from_handle(handle)
        .map(|c| c.config.enabled.load(Ordering::Relaxed) as i32)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_set_hot_threshold(
    handle: *mut c_void,
    threshold: u32,
) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.config.hot_threshold.store(threshold, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_set_verify_mode(
    handle: *mut c_void,
    enabled: i32,
) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.config.verify_mode.store(enabled != 0, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_set_max_entries(
    handle: *mut c_void,
    max_entries: u32,
) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.config.max_entries.store(max_entries, Ordering::Relaxed);
    }
}

/// Publish QEMU's softmmu helper-pointer arrays.  Caller hands us two
/// 16-element arrays of `uintptr_t` (each entry is the absolute host
/// address of a `helper_*_mmu` function, or 0 for unsupported shapes).
/// We copy the values into the context's atomic helper table.
///
/// # Safety
/// `ld_helpers` and `st_helpers` must each point to at least 16
/// contiguous `usize` values for the duration of this call (or be
/// null, in which case the corresponding side is cleared).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_set_helpers(
    handle: *mut c_void,
    ld_helpers: *const usize,
    st_helpers: *const usize,
) {
    let Some(ctx) = ctx_from_handle(handle) else {
        return;
    };
    let mut ld = [0usize; crate::context::HELPER_ARRAY_LEN];
    let mut st = [0usize; crate::context::HELPER_ARRAY_LEN];
    if !ld_helpers.is_null() {
        // SAFETY: caller guarantee on length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ld_helpers,
                ld.as_mut_ptr(),
                crate::context::HELPER_ARRAY_LEN,
            );
        }
    }
    if !st_helpers.is_null() {
        // SAFETY: caller guarantee on length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                st_helpers,
                st.as_mut_ptr(),
                crate::context::HELPER_ARRAY_LEN,
            );
        }
    }
    ctx.helpers.publish(&ld, &st);
    crate::dispatcher::log_to_android(&format!(
        "rust-side helpers: ld[2]=0x{:x} ld[3]=0x{:x} ld[10]=0x{:x} st[2]=0x{:x}",
        ld[2], ld[3], ld[10], st[2]
    ));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_enqueue(
    handle: *mut c_void,
    tb_pc: u64,
    ops: *const RawOp,
    num_ops: usize,
    tier1_code_size: u32,
) -> i32 {
    let Some(ctx) = ctx_from_handle(handle) else {
        return CRANELIFT_TCG_ERR_INTERNAL;
    };
    if !ctx.config.enabled.load(Ordering::Relaxed) {
        return CRANELIFT_TCG_ERR_DISABLED;
    }
    if ctx.is_blacklisted(tb_pc) {
        return CRANELIFT_TCG_ERR_BLACKLISTED;
    }
    let snap = OpSnapshot::from_raw(ops, num_ops);
    let dispatcher_guard = ctx.dispatcher.lock();
    let Some(d) = dispatcher_guard.as_ref() else {
        return CRANELIFT_TCG_ERR_INTERNAL;
    };
    d.enqueue(
        &ctx,
        crate::dispatcher::CompileReq {
            tb_pc,
            ops: snap,
            tier1_code_size,
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_compile_sync(
    handle: *mut c_void,
    tb_pc: u64,
    ops: *const RawOp,
    num_ops: usize,
    out_code: *mut *const c_void,
    out_size: *mut usize,
) -> i32 {
    let Some(ctx) = ctx_from_handle(handle) else {
        return CRANELIFT_TCG_ERR_INTERNAL;
    };
    let snap = OpSnapshot::from_raw(ops, num_ops);

    // Synchronous compile happens on the calling thread; we maintain
    // a per-call translator since the global module lives on the
    // worker. This path is used by unit tests only.
    let mut t = match Translator::new(&ctx.env) {
        Ok(t) => t,
        Err(e) => return e.to_ffi(),
    };
    match t.compile(&ctx, tb_pc, &snap) {
        Ok((entry, _unwind)) => {
            if !out_code.is_null() {
                unsafe { *out_code = entry.code as *const c_void };
            }
            if !out_size.is_null() {
                unsafe { *out_size = entry.size };
            }
            ctx.insert_entry(entry);
            CRANELIFT_TCG_OK
        }
        Err(e) => e.to_ffi(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_poll_result(
    handle: *mut c_void,
    out_tb_pc: *mut u64,
    out_code: *mut *const c_void,
    out_size: *mut usize,
) -> i32 {
    // Legacy ABI: discard the unwind metadata since the caller didn't
    // ask for it. Kept for tests / verify-mode call sites that don't
    // care about synchronous-fault unwind support.
    unsafe {
        cranelift_tcg_poll_result_v2(
            handle,
            out_tb_pc,
            out_code,
            out_size,
            std::ptr::null_mut(),
        )
    }
}

/// Mirror of `CraneliftTcgUnwindMeta` in the C header. Pointers belong
/// to the Rust-owned `Box<UnwindBuf>` referenced by `_handle`; the
/// caller must invoke `cranelift_tcg_release_unwind(_handle)` once it
/// has finished consuming the arrays (typically right after copying
/// them into the C-side unwind index slab).
#[repr(C)]
pub struct CraneliftTcgUnwindMeta {
    pub n_insns: u32,
    pub n_rows: u32,
    pub host_end: *const u32,
    pub loc: *const u32,
    pub insn_data: *const u64,
    pub _handle: *mut c_void,
}

/// Drain one completed compile and (optionally) hand back its unwind
/// metadata. The unwind buffer lives on the Rust heap until the caller
/// passes its `_handle` to `cranelift_tcg_release_unwind`; ownership
/// transfers to the C side at poll time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_poll_result_v2(
    handle: *mut c_void,
    out_tb_pc: *mut u64,
    out_code: *mut *const c_void,
    out_size: *mut usize,
    out_unwind: *mut CraneliftTcgUnwindMeta,
) -> i32 {
    let Some(ctx) = ctx_from_handle(handle) else {
        return 0;
    };
    let rsp = {
        let guard = ctx.dispatcher.lock();
        match guard.as_ref() {
            Some(d) => d.poll(),
            None => None,
        }
    };
    let Some(rsp) = rsp else {
        return 0;
    };
    unsafe {
        if !out_tb_pc.is_null() {
            *out_tb_pc = rsp.tb_pc;
        }
        if !out_code.is_null() {
            *out_code = rsp.entry.code as *const c_void;
        }
        if !out_size.is_null() {
            *out_size = rsp.entry.size;
        }
    }

    // Leak the Box; ownership now belongs to the C side via the
    // returned `_handle`. If the caller passes a NULL `out_unwind`,
    // they've opted out of unwind support -- drop the Box right here.
    if out_unwind.is_null() {
        drop(rsp.unwind);
    } else {
        let raw: *mut crate::context::UnwindBuf = Box::into_raw(rsp.unwind);
        // SAFETY: `raw` is a unique pointer (we just leaked it); the
        // Box from_raw call lives long enough for the field reads.
        let buf: &crate::context::UnwindBuf = unsafe { &*raw };
        let meta = CraneliftTcgUnwindMeta {
            n_insns: buf.n_insns,
            n_rows: buf.host_end.len() as u32,
            host_end: buf.host_end.as_ptr(),
            loc: buf.loc.as_ptr(),
            insn_data: buf.insn_data.as_ptr(),
            _handle: raw as *mut c_void,
        };
        unsafe {
            *out_unwind = meta;
        }
    }
    1
}

/// Drop the unwind buffer referenced by `_handle`. Must be paired with
/// each non-NULL `out_unwind` returned by `cranelift_tcg_poll_result_v2`;
/// safe to call with a NULL handle (no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_release_unwind(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` was produced by Box::into_raw in poll_result_v2;
    // C side guarantees it calls this exactly once per non-NULL handle.
    let _: Box<crate::context::UnwindBuf> = unsafe {
        Box::from_raw(handle as *mut crate::context::UnwindBuf)
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_blacklist(
    handle: *mut c_void,
    pc_lo: u64,
    pc_hi: u64,
) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.record_blacklist(pc_lo, pc_hi);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_get_stats(
    handle: *const c_void,
    out: *mut CraneliftTcgStats,
) {
    if out.is_null() {
        return;
    }
    let Some(ctx) = ctx_from_handle(handle as *mut c_void) else {
        return;
    };
    let s = &ctx.stats;
    unsafe {
        *out = CraneliftTcgStats {
            enqueued: s.enqueued.load(Ordering::Relaxed),
            compiled_ok: s.compiled_ok.load(Ordering::Relaxed),
            compiled_err: s.compiled_err.load(Ordering::Relaxed),
            fallback_unsupported_op: s
                .fallback_unsupported_op
                .load(Ordering::Relaxed),
            blacklisted: s.blacklisted.load(Ordering::Relaxed),
            verify_ok: s.verify_ok.load(Ordering::Relaxed),
            verify_divergence: s.verify_divergence.load(Ordering::Relaxed),
            total_compile_ns: s.total_compile_ns.load(Ordering::Relaxed),
            total_emitted_bytes: s.total_emitted_bytes.load(Ordering::Relaxed),
            active_entries: s.active_entries.load(Ordering::Relaxed),
            worker_queue_depth: s.worker_queue_depth.load(Ordering::Relaxed),
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_reset_stats(handle: *mut c_void) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.stats.reset();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_reset_entries(handle: *mut c_void) {
    if let Some(ctx) = ctx_from_handle(handle) {
        ctx.reset_entries();
    }
}

/// Verification-mode comparison helper. Returns 0 if equal, otherwise
/// the byte offset of the first difference + 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cranelift_tcg_verify_env(
    handle: *mut c_void,
    tb_pc: u64,
    tcg_env: *const u8,
    cranelift_env: *const u8,
    len: usize,
) -> i32 {
    let Some(ctx) = ctx_from_handle(handle) else {
        return -1;
    };
    if tcg_env.is_null() || cranelift_env.is_null() {
        return -1;
    }
    let a = unsafe { std::slice::from_raw_parts(tcg_env, len) };
    let b = unsafe { std::slice::from_raw_parts(cranelift_env, len) };
    let res = crate::verify::compare_env(a, b);
    let ok = res.is_ok();
    crate::verify::record(&ctx, ok, tb_pc);
    match res {
        Ok(()) => 0,
        Err(off) => (off as i32).saturating_add(1),
    }
}
