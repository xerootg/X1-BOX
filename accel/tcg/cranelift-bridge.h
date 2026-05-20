/*
 * Cranelift tier-2 JIT bridge - QEMU-side helpers.
 *
 * Internal header consumed by cpu-exec.c / translate-all.c. The
 * underlying Rust crate is reached through tcg/cranelift_bridge.h.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ACCEL_TCG_CRANELIFT_BRIDGE_H
#define ACCEL_TCG_CRANELIFT_BRIDGE_H

#include "qemu/osdep.h"
#include "exec/translation-block.h"
#include "tcg/tcg.h"

#if defined(XBOX) && defined(__aarch64__) && defined(__ANDROID__) && !defined(XEMU_DISABLE_CRANELIFT)
#define XEMU_HAVE_CRANELIFT 1
#else
#define XEMU_HAVE_CRANELIFT 0
#endif

#if XEMU_HAVE_CRANELIFT

/* Decide whether this TB has crossed the tier-2 threshold. */
bool cranelift_bridge_tb_should_promote(const TranslationBlock *tb);

/*
 * Take a snapshot of the post-optimization TCG IR and stash it in a
 * per-PC cache. Called from `tb_gen_code` for every translation. The
 * actual Cranelift compile is deferred until the block proves itself
 * hot via cpu_exec_loop's exec_count tracking.
 */
void cranelift_bridge_enqueue(TCGContext *s, TranslationBlock *tb);

/*
 * Hot-path hook: if this TB has crossed the tier-2 exec threshold,
 * look up its stashed IR snapshot and hand it to the Cranelift worker.
 * Idempotent; safe to call from cpu_exec_loop on every dispatch.
 */
void cranelift_bridge_maybe_compile(TranslationBlock *tb);

/* Drain completed compiles and RCU-swap them into the TB cache. */
void cranelift_bridge_drain(void);

/*
 * Hot-path hook: if a tier-2 compile result is pending for this TB's
 * guest PC, emit an ABI shim (SystemV -> TCG-prologue) and CAS-swap
 * tb->tc.ptr to point at the shim.  Subsequent dispatches execute the
 * Cranelift-compiled code.  Safe to call on every TB dispatch; the
 * common case (no pending swap) is a single ring read + compare.
 */
void cranelift_bridge_try_swap(TranslationBlock *tb);

/*
 * Return the installed tier-2 shim address for `tb`, or NULL if this
 * TB has not been tier-2 promoted.
 *
 * The shim is the per-TB ABI-translation blob that calls into Cranelift
 * code. We DON'T overwrite `tb->tc.ptr` because that field is the
 * lookup key for QEMU's `tcg_tb_lookup(host_pc)` search tree (used by
 * `cpu_io_recompile`, watchpoints, fault unwinding, …). Mutating it
 * makes the tree inconsistent and `tcg_tb_lookup` returns NULL for
 * any host PC inside the original tier-1 code -- which triggers a
 * `cpu_abort("can't recompile, no TB found")` the first time a tier-2
 * TB calls a helper that needs precise instruction-boundary state
 * (any MMIO load/store does).
 *
 * Instead, cpu_tb_exec calls this and dispatches to the shim when
 * present, falling back to tier-1 otherwise. The tree key stays
 * canonical.
 */
const void *cranelift_bridge_lookup_shim(const TranslationBlock *tb);

/*
 * Published by the aarch64 TCG backend (tcg_target_qemu_prologue) once
 * the prologue/epilogue blob has been emitted.  Zero until the prologue
 * has been initialised; cranelift_bridge_try_swap() refuses to install
 * shims until this is non-zero.
 */
extern uintptr_t cranelift_g_tb_ret_addr;

/*
 * Chain dispatch helper called by Cranelift-compiled TBs at goto_tb
 * sites. Defined in accel/tcg/cpu-exec.c where tb_lookup is visible.
 * The address is published to the Rust side via the env descriptor so
 * the lowering can emit a call to it without needing extern symbol
 * resolution at JIT time.
 */
uintptr_t cranelift_chain_continue(CPUArchState *env);

/* Runtime toggle + tuning knobs. */
void cranelift_bridge_set_enabled(bool enabled);
bool cranelift_bridge_is_enabled(void);
void cranelift_bridge_set_verify_mode(bool enabled);
void cranelift_bridge_set_threshold(uint32_t threshold);
void cranelift_bridge_blacklist(uint64_t pc_lo, uint64_t pc_hi);
/* Master switch for installing Cranelift-compiled shims into the TB
 * cache. Off by default (env: X1BOX_CRANELIFT_SWAP=1). */
void cranelift_bridge_set_swap_enabled(bool enabled);
bool cranelift_bridge_is_swap_enabled(void);

/* Dump current telemetry to the log. */
void cranelift_bridge_log_stats(void);

#else /* !XEMU_HAVE_CRANELIFT */

static inline bool
cranelift_bridge_tb_should_promote(const TranslationBlock *tb)
{
    (void)tb;
    return false;
}
static inline void
cranelift_bridge_enqueue(TCGContext *s, TranslationBlock *tb)
{
    (void)s; (void)tb;
}
static inline void
cranelift_bridge_maybe_compile(TranslationBlock *tb)
{
    (void)tb;
}
static inline void cranelift_bridge_drain(void) {}
static inline void cranelift_bridge_try_swap(TranslationBlock *tb)
{
    (void)tb;
}
static inline const void *
cranelift_bridge_lookup_shim(const TranslationBlock *tb)
{
    (void)tb;
    return NULL;
}
static inline void cranelift_bridge_set_enabled(bool e) { (void)e; }
static inline bool cranelift_bridge_is_enabled(void) { return false; }
static inline void cranelift_bridge_set_verify_mode(bool e) { (void)e; }
static inline void cranelift_bridge_set_threshold(uint32_t t) { (void)t; }
static inline void cranelift_bridge_blacklist(uint64_t lo, uint64_t hi)
{
    (void)lo; (void)hi;
}
static inline void cranelift_bridge_log_stats(void) {}
static inline void cranelift_bridge_set_swap_enabled(bool e) { (void)e; }
static inline bool cranelift_bridge_is_swap_enabled(void) { return false; }

#endif /* XEMU_HAVE_CRANELIFT */

#endif /* ACCEL_TCG_CRANELIFT_BRIDGE_H */
