//! Tier-2 dispatcher.
//!
//! Owns:
//! - A worker thread that drains the compile queue.
//! - The Cranelift `JITModule` (lives on the worker thread; never
//!   touched by the vCPU thread).
//! - A response channel feeding completed results back to the C side
//!   via [`crate::ffi::cranelift_tcg_poll_result`].
//!
//! Design rationale:
//! - vCPU thread MUST NOT block on compile. All enqueue paths use a
//!   bounded `crossbeam_channel::try_send`; if the queue is full the
//!   request is dropped (the block will get retried on its next hit).
//! - Result delivery is decoupled from the worker so the vCPU thread
//!   can poll once per main-loop iteration rather than per TB.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::context::{JitContext, TierTwoEntry};
use crate::ffi;
use crate::ir::OpSnapshot;
use crate::translator::Translator;

/// Print to Android's logcat with our tag.
pub(crate) fn log_to_android(msg: &str) {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let tag = CString::new("x1-cranelift").unwrap();
        let m = CString::new(msg).unwrap_or_else(|_| {
            CString::new("(log msg contained NUL)").unwrap()
        });
        // SAFETY: __android_log_print is variadic; we use the %s form
        // so a malformed `msg` can't trigger format-string UB.
        unsafe {
            __android_log_print(
                4, /* ANDROID_LOG_INFO */
                tag.as_ptr(),
                b"%s\0".as_ptr() as *const _,
                m.as_ptr(),
            );
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        eprintln!("{msg}");
    }
}

#[cfg(target_os = "android")]
unsafe extern "C" {
    fn __android_log_print(
        prio: std::ffi::c_int,
        tag: *const std::ffi::c_char,
        fmt: *const std::ffi::c_char,
        ...
    ) -> std::ffi::c_int;
}

/// One compile request.
pub struct CompileReq {
    pub tb_pc: u64,
    pub ops: OpSnapshot,
    pub tier1_code_size: u32,
}

/// One compile response.
pub struct CompileRsp {
    pub tb_pc: u64,
    pub entry: TierTwoEntry,
}

pub struct Dispatcher {
    req_tx: Sender<DispatcherMsg>,
    rsp_rx: Receiver<CompileRsp>,
    join: Option<JoinHandle<()>>,
}

enum DispatcherMsg {
    Compile(CompileReq),
    Shutdown,
}

impl Dispatcher {
    pub fn spawn(ctx: Arc<JitContext>) -> Self {
        // Bounded queue so a hot-loop can't OOM us. Drops on full.
        let (req_tx, req_rx) = bounded::<DispatcherMsg>(256);
        let (rsp_tx, rsp_rx) = bounded::<CompileRsp>(256);

        let worker_ctx = Arc::clone(&ctx);
        let join = std::thread::Builder::new()
            .name("cranelift-tcg".into())
            .spawn(move || worker_loop(worker_ctx, req_rx, rsp_tx))
            .expect("spawn cranelift-tcg worker");

        Dispatcher {
            req_tx,
            rsp_rx,
            join: Some(join),
        }
    }

    pub fn enqueue(&self, ctx: &JitContext, req: CompileReq) -> i32 {
        if !ctx.config.enabled.load(Ordering::Relaxed) {
            return ffi::CRANELIFT_TCG_ERR_DISABLED;
        }
        if ctx.is_blacklisted(req.tb_pc) {
            return ffi::CRANELIFT_TCG_ERR_BLACKLISTED;
        }
        match self.req_tx.try_send(DispatcherMsg::Compile(req)) {
            Ok(()) => {
                ctx.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                let depth = self.req_tx.len() as u32;
                ctx.stats.worker_queue_depth.store(depth, Ordering::Relaxed);
                ffi::CRANELIFT_TCG_OK
            }
            Err(TrySendError::Full(_)) => ffi::CRANELIFT_TCG_ERR_OUT_OF_MEMORY,
            Err(TrySendError::Disconnected(_)) => ffi::CRANELIFT_TCG_ERR_INTERNAL,
        }
    }

    pub fn poll(&self) -> Option<CompileRsp> {
        self.rsp_rx.try_recv().ok()
    }

    pub fn shutdown(mut self) {
        let _ = self.req_tx.try_send(DispatcherMsg::Shutdown);
        // Best-effort - drop the receivers so the worker exits cleanly.
        drop(self.req_tx);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

fn worker_loop(
    ctx: Arc<JitContext>,
    req_rx: Receiver<DispatcherMsg>,
    rsp_tx: Sender<CompileRsp>,
) {
    // The translator owns the JIT module; it is not Send so it must
    // live on this thread.
    let mut translator = match Translator::new(&ctx.env) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cranelift-tcg: translator init failed: {e:?}");
            return;
        }
    };

    // Per-error histograms so we can see WHY compiles are failing.
    let mut err_unsupported = std::collections::HashMap::<u16, u64>::new();
    let mut err_other = std::collections::HashMap::<&'static str, u64>::new();
    let mut total_compiles: u64 = 0;
    let last_log_ns = std::sync::Mutex::new(Instant::now());

    while let Ok(msg) = req_rx.recv() {
        match msg {
            DispatcherMsg::Shutdown => break,
            DispatcherMsg::Compile(req) => {
                let t0 = Instant::now();
                let result = translator.compile(&ctx, req.tb_pc, &req.ops);
                let elapsed_ns = t0.elapsed().as_nanos() as u64;
                total_compiles += 1;
                match result {
                    Ok(entry) => {
                        ctx.stats
                            .note_compile(elapsed_ns, entry.size as u64, true);
                        ctx.insert_entry(entry.clone());
                        let _ = rsp_tx.try_send(CompileRsp {
                            tb_pc: req.tb_pc,
                            entry,
                        });
                    }
                    Err(crate::translator::TransError::UnsupportedOp(opc)) => {
                        ctx.stats
                            .fallback_unsupported_op
                            .fetch_add(1, Ordering::Relaxed);
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_unsupported.entry(opc).or_insert(0) += 1;
                    }
                    Err(crate::translator::TransError::VerifierFailed(msg)) => {
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_other.entry("VerifierFailed").or_insert(0) += 1;
                        // Print a short excerpt of the first failure so
                        // we can see the actual Cranelift complaint.
                        static FIRST: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(true);
                        if FIRST.swap(false, Ordering::Relaxed) {
                            let trim: String = msg.chars().take(300).collect();
                            eprintln!("cranelift-tcg verifier (first): {trim}");
                            log_to_android(&format!(
                                "verifier err (first): {trim}"
                            ));
                        }
                    }
                    Err(crate::translator::TransError::InvalidIr(msg)) => {
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_other.entry(msg).or_insert(0) += 1;
                    }
                    Err(crate::translator::TransError::BadCondition(_)) => {
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_other.entry("BadCondition").or_insert(0) += 1;
                    }
                    Err(crate::translator::TransError::ModuleError(_)) => {
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_other.entry("ModuleError").or_insert(0) += 1;
                    }
                    Err(crate::translator::TransError::Internal(msg)) => {
                        ctx.stats.note_compile(elapsed_ns, 0, false);
                        *err_other.entry(msg).or_insert(0) += 1;
                    }
                }
                // Periodically dump the histograms so we can see what's
                // blocking compilation. Total compiles caps out near
                // ~700 for a steady-state Halo 2 perftest run, so the
                // previous power-of-two + 4096-stride gating only ever
                // fired at 512 and any earlier samples rolled out of
                // logcat's ring buffer. Log at a tight cadence (every
                // 16 compiles with a 1s floor) so the breakdown shows
                // up in steady-state with the dominant opcodes visible
                // by the time the user reads the log.
                let should_log = total_compiles % 16 == 0
                    || total_compiles == 1
                    || (err_unsupported.len() + err_other.len()) > 0
                        && total_compiles % 8 == 0;
                if should_log {
                    let mut last = last_log_ns.lock().unwrap();
                    if last.elapsed().as_millis() >= 1000 {
                        *last = Instant::now();
                        drop(last);
                        let mut top: Vec<(u16, u64)> = err_unsupported
                            .iter()
                            .map(|(&k, &v)| (k, v))
                            .collect();
                        top.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
                        let mut summary = format!(
                            "compile_err breakdown (total={total_compiles}): "
                        );
                        for (opc, count) in top.iter().take(12) {
                            summary.push_str(&format!("opc{opc}={count} "));
                        }
                        for (k, v) in &err_other {
                            summary.push_str(&format!("{k}={v} "));
                        }
                        log_to_android(&summary);
                    }
                }
                ctx.stats
                    .worker_queue_depth
                    .store(req_rx.len() as u32, Ordering::Relaxed);
            }
        }
    }
}
