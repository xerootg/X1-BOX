//! Verification mode.
//!
//! When enabled (via `cranelift_tcg_set_verify_mode`) the dispatcher
//! does NOT swap the TB code pointer; instead the C side calls both
//! the TCG-emitted and Cranelift-emitted entries with isolated env
//! snapshots and compares the resulting register state. Divergence is
//! recorded as telemetry and the offending TB is added to the
//! blacklist.
//!
//! The actual lock-step harness lives in C (where the env snapshot
//! and TLB state can be mirrored cheaply); this module just provides
//! the comparison primitive.

use std::sync::atomic::Ordering;

use crate::context::JitContext;

/// Compare two raw byte slices that the C side claims represent the
/// post-TB env state under TCG vs Cranelift. Returns Ok(()) when
/// equal, Err(offset) on first byte that differs.
pub fn compare_env(tcg: &[u8], cranelift: &[u8]) -> Result<(), usize> {
    let n = tcg.len().min(cranelift.len());
    for i in 0..n {
        if tcg[i] != cranelift[i] {
            return Err(i);
        }
    }
    if tcg.len() != cranelift.len() {
        return Err(n);
    }
    Ok(())
}

/// Record the outcome of a single verification cycle.
pub fn record(ctx: &JitContext, ok: bool, tb_pc: u64) {
    if ok {
        ctx.stats.verify_ok.fetch_add(1, Ordering::Relaxed);
    } else {
        ctx.stats.verify_divergence.fetch_add(1, Ordering::Relaxed);
        // Blacklist a one-page window around the divergent PC so we
        // don't keep re-comparing the same bad TB.
        let pc_lo = tb_pc & !0xfff;
        let pc_hi = pc_lo + 0x1000;
        ctx.record_blacklist(pc_lo, pc_hi);
    }
}
