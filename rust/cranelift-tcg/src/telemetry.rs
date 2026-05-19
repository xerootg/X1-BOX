//! Counters surfaced via [`crate::ffi::cranelift_tcg_get_stats`].
//!
//! Each counter is a relaxed `AtomicU64` - we never need
//! synchronisation between counters and the loss of precision is
//! preferable to lock contention.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Default, Debug)]
pub struct Telemetry {
    pub enqueued: AtomicU64,
    pub compiled_ok: AtomicU64,
    pub compiled_err: AtomicU64,
    pub fallback_unsupported_op: AtomicU64,
    pub blacklisted: AtomicU64,
    pub verify_ok: AtomicU64,
    pub verify_divergence: AtomicU64,
    pub total_compile_ns: AtomicU64,
    pub total_emitted_bytes: AtomicU64,
    pub active_entries: AtomicU32,
    pub worker_queue_depth: AtomicU32,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&self) {
        for c in [
            &self.enqueued,
            &self.compiled_ok,
            &self.compiled_err,
            &self.fallback_unsupported_op,
            &self.blacklisted,
            &self.verify_ok,
            &self.verify_divergence,
            &self.total_compile_ns,
            &self.total_emitted_bytes,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        self.active_entries.store(0, Ordering::Relaxed);
        self.worker_queue_depth.store(0, Ordering::Relaxed);
    }

    pub fn note_compile(&self, ns: u64, bytes: u64, ok: bool) {
        self.total_compile_ns.fetch_add(ns, Ordering::Relaxed);
        if ok {
            self.compiled_ok.fetch_add(1, Ordering::Relaxed);
            self.total_emitted_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        } else {
            self.compiled_err.fetch_add(1, Ordering::Relaxed);
        }
    }
}
