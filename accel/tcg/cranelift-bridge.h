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
#include "accel/tcg/tb-cpu-state.h"

#if defined(XBOX) && defined(__aarch64__) && defined(__ANDROID__) && !defined(XEMU_DISABLE_CRANELIFT)
#define XEMU_HAVE_CRANELIFT 1
#else
#define XEMU_HAVE_CRANELIFT 0
#endif

#if XEMU_HAVE_CRANELIFT

/*
 * Published by cranelift-bridge.c. The dispatch hot path consults
 * these directly so we can short-circuit the bridge call when there's
 * nothing for it to do, which is by far the common case.
 */
extern uint32_t cranelift_bridge_g_tier2_threshold;
extern unsigned cranelift_bridge_g_pending_head;
extern unsigned cranelift_bridge_g_pending_tail;

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
 *
 * The inline wrapper below short-circuits the common case (TB not yet
 * hot or already tier-2) without paying function-call overhead. The
 * out-of-line slow path runs only when both gates would pass.
 */
void cranelift_bridge_maybe_compile_slow(TranslationBlock *tb);

/*
 * Tier-2 hot-PC hint table — direct-mapped, open-addressed via
 * Fibonacci hash. Populated by cranelift_bridge_jit_cache_open() from
 * the per-game hints.bin file and by cranelift_bridge_swap_install_one
 * on every successful tier-2 install. The maybe_compile inline below
 * uses this to bypass the exec_count threshold for PCs that were hot
 * in a previous session — they go straight to compile-enqueue on
 * first execution.
 *
 * Read without locks: a torn 64-bit read on aarch64 is impossible
 * (aligned u64 reads/writes are atomic) and even on hypothetical torn
 * archs the worst case is a false positive (compile a cold TB) or
 * false negative (fall back to the normal threshold path) — neither
 * affects correctness.
 */
#define CRANELIFT_HOT_PC_SLOTS 8192u
extern uint64_t cranelift_bridge_g_hot_pcs[CRANELIFT_HOT_PC_SLOTS];

static inline bool cranelift_bridge_pc_is_hot(uint64_t pc)
{
    if (pc == 0) {
        return false;
    }
    unsigned slot = (unsigned)((pc * 2654435761ull) & (CRANELIFT_HOT_PC_SLOTS - 1));
    return cranelift_bridge_g_hot_pcs[slot] == pc;
}

static inline void cranelift_bridge_maybe_compile(TranslationBlock *tb)
{
    if (tb->tier >= 2) {
        return;
    }
    /* Pre-warm: TBs at PCs that were hot in a prior session bypass the
     * exec_count gate so they compile on first execution. The table is
     * populated on every tier-2 install (live + load-on-init from the
     * per-game hints.bin). False positives here only cost a wasted
     * compile of a cold TB. */
    bool hot_hint = cranelift_bridge_pc_is_hot((uint64_t)tb->pc);
    if (!hot_hint &&
        tb->exec_count <
        qatomic_read(&cranelift_bridge_g_tier2_threshold)) {
        return;
    }
    cranelift_bridge_maybe_compile_slow(tb);
}

/* Drain completed compiles and RCU-swap them into the TB cache. */
void cranelift_bridge_drain(void);

/*
 * Hot-path hook: if a tier-2 compile result is pending for this TB's
 * guest PC, emit an ABI shim (SystemV -> TCG-prologue) and CAS-swap
 * tb->tc.ptr to point at the shim.  Subsequent dispatches execute the
 * Cranelift-compiled code.  Safe to call on every TB dispatch.
 *
 * The inline wrapper short-circuits when the pending ring is empty or
 * the TB is already tier-2. The slow path is reached only when work
 * may exist; with steady-state workloads the ring is empty 99%+ of
 * the time, so the fast path is a couple of cache-warm atomic reads.
 */
void cranelift_bridge_try_swap_slow(TranslationBlock *tb);

static inline void cranelift_bridge_try_swap(TranslationBlock *tb)
{
    if (tb->tier >= 2) {
        return;
    }
    /*
     * Per-TB short-circuit: only enter the slow path's mutex + ring
     * scan when THIS TB has a pending compile result. With the
     * threshold at 8 the ring is almost always non-empty for some
     * (other) TB, so the legacy head==tail check stopped being a
     * useful gate. drain() sets this bit after publishing to the ring;
     * try_swap_slow() clears it after consuming. Reads here are
     * uncontended, so qatomic_read is essentially a cache-hot load.
     */
    if (!qatomic_read(&tb->cranelift_pending)) {
        return;
    }
    cranelift_bridge_try_swap_slow(tb);
}

/*
 * Drop every cached tier-2 binding. MUST be called from
 * tb_flush__exclusive_or_serial() right after tcg_region_reset_all()
 * because that resets the TB allocator and the same TranslationBlock*
 * pointers get handed out for entirely different guest PCs. Without
 * this, the shim map keyed by TB pointer dispatches the new TBs into
 * stale Cranelift code -> guest #PF -> kernel cli/hlt.
 *
 * Caller must hold cpu_in_serial_context (which tb_flush already
 * asserts), so this is allowed to take internal mutexes and zero the
 * shim map without coordinating with vCPU readers.
 */
void cranelift_bridge_on_tb_flush(void);

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

/*
 * Per-chain quantum + per-guest-thread fingerprint tracker.
 *
 * cranelift_chain_init_quantum() reads X1BOX_CHAIN_MAX and
 * X1BOX_CHAIN_JITTER env vars; safe to call once at startup.
 *
 * cranelift_chain_get_stats() reports counters for the stats dump.
 * NULL out-params are tolerated.
 */
void cranelift_chain_init_quantum(void);
void cranelift_chain_get_stats(uint64_t *runs, uint64_t *iters,
                                uint64_t *spins, uint64_t *irq_exits,
                                uint32_t *thread_count,
                                unsigned *chain_max, uint32_t *jitter);

/*
 * Per-thread 2-slot LRU stats for helper_lookup_tb_ptr.
 * Hit/miss counters across all vCPU threads (best-effort, racey writes
 * tolerated; reader sees one of the in-flight values).
 */
void cranelift_get_helper_lookup_tb_lru_stats(uint64_t *hits, uint64_t *misses);

/*
 * Target-side helper: returns the top 16 bits of the guest stack pointer
 * (ESP & ~0xFFFFu on x86). Used by the chain-continue thread fingerprint.
 * Defined in target/i386/cpu.c so the accel-side stays target-opaque
 * (cpu-exec.c can't see CPUArchState layout from this layer).
 */
uint32_t xemu_chain_thread_fingerprint(CPUArchState *env);

/*
 * Target-side fast variant of get_tb_cpu_state for the chain dispatcher.
 *
 * Bypasses the cpu->cc->tcg_ops->get_tb_cpu_state vtable indirection that
 * x86_get_tb_cpu_state usually goes through; the chain calls this on every
 * iter so the indirect call + redundant env loads are measurable in the
 * profile (0.44% process samples in the 2026-05-24 Halo 2 capture).
 *
 * cflags is NOT set here — caller fills it from curr_cflags(cpu).
 */
TCGTBCPUState xemu_chain_get_tb_cpu_state(CPUArchState *env);

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

/*
 * Tier-2 disk cache (hint-cache v1).
 *
 * Saves per-game lists of guest PCs that were promoted to tier-2 in
 * the previous session. On the next boot the dispatcher pre-seeds
 * exec_count for those TBs so they enqueue compile on first execution
 * instead of waiting for X1BOX_CRANELIFT_THRESHOLD (default 4) hits.
 *
 * Location: $X1BOX_JIT_CACHE_DIR/hints.bin (env var set by the Android
 * launcher; sanitised game-title key matches the UI in
 * GameLibraryActivity.kt::jitCacheKey).
 *
 * File format:
 *   magic[4]: "X1JH"
 *   version: u32 (currently 1)
 *   build_id_len: u32, build_id[N]: bytes (CRANELIFT_BUILD_ID)
 *   xbe_sha1[20]: future use — currently zero
 *   count: u32
 *   tb_pcs: u64 × count
 *
 * On version/build-id mismatch the cache is discarded silently.
 *
 * `cranelift_bridge_jit_cache_open(dir_path)` is called once after the
 * game is identified; passing NULL disables both save and load (useful
 * on first boot before the path is known, and as a soft kill switch).
 *
 * `cranelift_bridge_jit_cache_save()` flushes the in-memory list to
 * disk; called from the game-shutdown path. Idempotent — safe to call
 * multiple times.
 */
void cranelift_bridge_jit_cache_open(const char *dir_path);
void cranelift_bridge_jit_cache_save(void);

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
static inline void cranelift_bridge_on_tb_flush(void) {}
static inline void cranelift_bridge_jit_cache_open(const char *p) { (void)p; }
static inline void cranelift_bridge_jit_cache_save(void) {}
static inline void cranelift_chain_init_quantum(void) {}
static inline void
cranelift_chain_get_stats(uint64_t *runs, uint64_t *iters,
                          uint64_t *spins, uint64_t *irq_exits,
                          uint32_t *thread_count,
                          unsigned *chain_max, uint32_t *jitter)
{
    if (runs) *runs = 0;
    if (iters) *iters = 0;
    if (spins) *spins = 0;
    if (irq_exits) *irq_exits = 0;
    if (thread_count) *thread_count = 0;
    if (chain_max) *chain_max = 0;
    if (jitter) *jitter = 0;
}

#endif /* XEMU_HAVE_CRANELIFT */

#endif /* ACCEL_TCG_CRANELIFT_BRIDGE_H */
