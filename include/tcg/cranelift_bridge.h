/*
 * Cranelift TCG bridge - tier-2 JIT backend FFI surface.
 *
 * Lifts QEMU TCG IR through Cranelift to produce higher-quality
 * ARM64 code than the hand-written TCG backend. Per-TB tier-2
 * compilation is dispatched on a worker thread and the resulting
 * code pointer is atomically swapped into the TB cache via RCU.
 *
 * All functions are safe to call from any thread once
 * cranelift_tcg_init() has returned non-NULL.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef TCG_CRANELIFT_BRIDGE_H
#define TCG_CRANELIFT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to the Cranelift JIT context (per-process). */
typedef struct CraneliftTcgContext CraneliftTcgContext;

/* Opaque handle to a single compile request / result. */
typedef struct CraneliftTcgCompile CraneliftTcgCompile;

/*
 * Return codes from cranelift_tcg_compile_*.
 * Negative = error, 0 = success, positive = success-with-warning.
 */
#define CRANELIFT_TCG_OK                     0
#define CRANELIFT_TCG_ERR_NOT_IMPLEMENTED   -1
#define CRANELIFT_TCG_ERR_UNSUPPORTED_OP    -2
#define CRANELIFT_TCG_ERR_OUT_OF_MEMORY     -3
#define CRANELIFT_TCG_ERR_INVALID_IR        -4
#define CRANELIFT_TCG_ERR_VERIFIER_FAILED   -5
#define CRANELIFT_TCG_ERR_INTERNAL          -6
#define CRANELIFT_TCG_ERR_DISABLED          -7
#define CRANELIFT_TCG_ERR_BLACKLISTED       -8

/*
 * Verification-mode result codes (set when verification is enabled).
 */
#define CRANELIFT_TCG_VERIFY_OK              0
#define CRANELIFT_TCG_VERIFY_DIVERGENCE      1

/*
 * Opaque IR snapshot passed from the C side. The bridge does not
 * iterate TCGContext directly (avoids a fragile dependency on QEMU's
 * internal types). Instead the C side serialises post-optimization
 * TCG ops into a flat array of CraneliftTcgOp records via the helpers
 * declared below.
 */
typedef struct CraneliftTcgOp {
    /* TCG opcode index (matches enum TCGOpcode in tcg-opc.h). */
    uint16_t opc;
    /* Number of output args (matches TCGOpDef::nb_oargs). */
    uint8_t  nb_oargs;
    /* Number of input args. */
    uint8_t  nb_iargs;
    /* Number of constant args (memop, label, etc.). */
    uint8_t  nb_cargs;
    /* Type hint: 0 = i32, 1 = i64, 2 = i128, 3 = v64, 4 = v128, 5 = v256. */
    uint8_t  type;
    /* TCG flags from TCGOpDef::flags. */
    uint16_t flags;
    /*
     * Bitmask covering positions 0..(nb_oargs + nb_iargs). When bit i
     * is set, args[i] holds the int64 constant value of a TEMP_CONST
     * source temp directly (rather than a dense temp index). Required
     * so the Rust side can emit `iconst(val)` instead of looking up a
     * local Variable seeded to zero — without this, every `mov dst,
     * TEMP_CONST` lowered to `dst = 0` and tier-2 blocks zeroed
     * register/PC state on every execution.
     */
    uint16_t const_mask;
    /* Padding to keep args[] 8-byte aligned. Total struct size = 144. */
    uint32_t _pad;
    /* Args; nb_oargs + nb_iargs + nb_cargs entries in [0..16]. */
    uint64_t args[16];
} CraneliftTcgOp;

/*
 * Describes the CPU environment layout the Cranelift IR needs to
 * reference. Set once at init time.
 */
typedef struct CraneliftTcgEnvDesc {
    /* sizeof(CPUArchState) - bounds for ld/st against env. */
    uint32_t env_size;
    /* Offset of the TLB array within env (target-specific). */
    uint32_t tlb_offset;
    /* Offset of the eip / nip equivalent. */
    uint32_t pc_offset;
    /* Number of guest GPRs the frontend exposes as TCG globals. */
    uint32_t nb_globals;
    /* Pointer to a flat array of global offsets/sizes/names.
     * Layout: u32 offset, u32 size, u32 name_offset (into name_pool).
     * 3 * nb_globals u32 entries.
     */
    const uint32_t *globals;
    /* Concatenated NUL-terminated names referenced from globals[]. */
    const char *name_pool;
    /* Guest pointer width: 4 or 8. */
    uint32_t guest_ptr_size;
    /* Host pointer width: 4 or 8. */
    uint32_t host_ptr_size;
    /* Address of cranelift_chain_continue helper (cpu-exec.c). The
     * GotoTb lowering calls this at every chain site to dispatch
     * successive TBs without going through the dispatcher loop, which
     * was the source of audio/cutscene chop once lower_call was on.
     * Pass 0 to disable chain dispatch (lowering falls back to a plain
     * return-to-dispatcher). */
    uintptr_t chain_continue_fn;
    /* Address of `helper_lookup_tb_ptr` (accel/tcg/cpu-exec.c). The
     * frontend always emits this helper immediately before `goto_ptr`
     * (see tcg_gen_lookup_and_goto_ptr). Tier-2 lowering of goto_ptr
     * routes back through cranelift_chain_continue, which re-does the
     * lookup itself by env->eip -- so the helper call is dead work in
     * tier-2 TBs. The translator pattern-matches against this address
     * to elide the redundant call. Pass 0 to disable the elision. */
    uintptr_t lookup_tb_ptr_fn;
} CraneliftTcgEnvDesc;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/*
 * Initialise the Cranelift backend. Spawns the worker thread.
 * `env` describes the guest CPU env layout and is copied internally.
 * Returns NULL if cranelift could not be initialised (e.g. ISA not
 * supported on this host). Safe to call once per process; subsequent
 * calls return the existing context.
 */
CraneliftTcgContext *cranelift_tcg_init(const CraneliftTcgEnvDesc *env);

/* Tear down the worker thread and release all JIT memory. */
void cranelift_tcg_destroy(CraneliftTcgContext *ctx);

/* Enable / disable the tier-2 backend at runtime (settings toggle). */
void cranelift_tcg_set_enabled(CraneliftTcgContext *ctx, int enabled);
int  cranelift_tcg_is_enabled(const CraneliftTcgContext *ctx);

/* Hot-block threshold (default 1000). */
void cranelift_tcg_set_hot_threshold(CraneliftTcgContext *ctx,
                                     uint32_t threshold);

/* Verification mode: when enabled, every Cranelift-compiled TB runs
 * alongside the TCG one and register state is compared. */
void cranelift_tcg_set_verify_mode(CraneliftTcgContext *ctx, int enabled);

/* Memory budget: cap on tier-2 entries (default 4096). */
void cranelift_tcg_set_max_entries(CraneliftTcgContext *ctx,
                                   uint32_t max_entries);

/*
 * Publish the qemu_ld_helpers[] / qemu_st_helpers[] arrays so the
 * tier-2 memory lowering can emit direct calls to QEMU's slow-path
 * MMIO helpers on TLB miss.  Both arrays are 16 elements indexed by
 * MemOp (size | sign | endian) bits; entries that are NULL on the C
 * side stay NULL on the Rust side and force a tier-2 fallback for
 * that MemOp shape.  Safe to call multiple times - the most recent
 * publish wins.
 */
void cranelift_tcg_set_helpers(void *ctx,
                               const uintptr_t ld_helpers[16],
                               const uintptr_t st_helpers[16]);

/* ------------------------------------------------------------------ */
/* Per-TB request flow                                                  */
/* ------------------------------------------------------------------ */

/*
 * Enqueue a hot TB for tier-2 compile. The caller passes:
 *   tb_pc           - guest PC of the TB start (also serves as key)
 *   ops, num_ops    - post-optimization TCG IR snapshot
 *   tier1_code_size - tier-1 code size in bytes (used for budgeting)
 *
 * Returns CRANELIFT_TCG_OK on enqueue, error code otherwise.
 * The caller may copy `ops`; the bridge owns the snapshot afterwards.
 */
int cranelift_tcg_enqueue(CraneliftTcgContext *ctx,
                          uint64_t tb_pc,
                          const CraneliftTcgOp *ops,
                          size_t num_ops,
                          uint32_t tier1_code_size);

/*
 * Synchronous compile path - used by the unit test harness and
 * Phase-2 verification. Returns a pointer to the emitted code in
 * `out_code`, its size in `out_size`. Caller does not own the buffer;
 * it lives in the JIT's code arena until cranelift_tcg_destroy().
 */
int cranelift_tcg_compile_sync(CraneliftTcgContext *ctx,
                               uint64_t tb_pc,
                               const CraneliftTcgOp *ops,
                               size_t num_ops,
                               const void **out_code,
                               size_t *out_size);

/*
 * Poll for a completed tier-2 compile. Returns 1 if a result is
 * available and was written to *out_tb_pc / *out_code / *out_size,
 * 0 if the queue is empty. Caller is expected to perform the RCU
 * TB-code-pointer swap.
 */
int cranelift_tcg_poll_result(CraneliftTcgContext *ctx,
                              uint64_t *out_tb_pc,
                              const void **out_code,
                              size_t *out_size);

/*
 * Blacklist a TB PC range after a crash inside Cranelift code.
 * Future enqueues with `tb_pc` in this range return
 * CRANELIFT_TCG_ERR_BLACKLISTED.
 */
void cranelift_tcg_blacklist(CraneliftTcgContext *ctx,
                             uint64_t pc_lo, uint64_t pc_hi);

/* ------------------------------------------------------------------ */
/* Telemetry                                                            */
/* ------------------------------------------------------------------ */

typedef struct CraneliftTcgStats {
    uint64_t enqueued;
    uint64_t compiled_ok;
    uint64_t compiled_err;
    uint64_t fallback_unsupported_op;
    uint64_t blacklisted;
    uint64_t verify_ok;
    uint64_t verify_divergence;
    uint64_t total_compile_ns;
    uint64_t total_emitted_bytes;
    uint32_t active_entries;
    uint32_t worker_queue_depth;
} CraneliftTcgStats;

void cranelift_tcg_get_stats(const CraneliftTcgContext *ctx,
                             CraneliftTcgStats *out);

/* Reset all counters (for benchmarking). */
void cranelift_tcg_reset_stats(CraneliftTcgContext *ctx);

/*
 * Drop all cached tier-2 entries. Called from the QEMU bridge after a
 * tb_flush, where every previously-cached TranslationBlock pointer has
 * just been freed back to the allocator and will alias new, unrelated
 * TBs. The worker keeps running and will accept new enqueues.
 */
void cranelift_tcg_reset_entries(CraneliftTcgContext *ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TCG_CRANELIFT_BRIDGE_H */
