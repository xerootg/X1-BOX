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
        Ok(entry) => {
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
    1
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
