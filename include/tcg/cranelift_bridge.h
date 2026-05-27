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
    /* Address of `cranelift_helper_flcr(uint32_t mxcsr)`. The x86
     * frontend emits INDEX_op_flcr for FLDCW/FNCLEX etc.; tier-1
     * aarch64 lowers it to an MSR FPCR sequence inline. Cranelift
     * can't emit MSR directly, so tier-2 routes flcr through a one-
     * argument helper that does the MXCSR->FPCR translation. 0 means
     * flcr is unsupported and the TB bails to tier-1. */
    uintptr_t flcr_fn;
    /* Address of `helper_cc_compute_all(dst, src1, src2, op)`
     * (target/i386/tcg/cc_helper.c). x86 frontend calls this whenever
     * lazy eflags need to materialise (sti/popf/iret, pushf/cpuid,
     * any cc-consuming branch when CC_OP is dynamic). Profile shows
     * ~1.25% of Halo 2 title-screen CPU sits in this single helper.
     * Tier-2 lowering peeks at carg(0); when it matches, we emit an
     * inline if-ladder that handles CC_OP_EFLAGS (= 0, returns src1
     * verbatim) and CC_OP_POPCNT (returns src ? 0 : CC_Z) and falls
     * through to the C helper for everything else. 0 disables the
     * inline path and routes every call through call_indirect. */
    uintptr_t cc_compute_all_fn;

    /*
     * Native fp80 (USE_HARD_FPU) inline lowering surface. AArch64
     * stores guest ST(N) as native double in env->fpregs[i].native_d
     * and FT0 in env->ft0_native, so the `__hard` variants of every
     * x87 arith/compare/classify helper reduce to a handful of f64
     * loads, an arith op, and a store. Cranelift can emit that inline
     * and skip the call_indirect entirely; the helpers below carry the
     * runtime addresses of each helper-pointer we know how to inline.
     *
     * Layout offsets + helper addresses live in a separate sub-struct
     * so the env-desc keeps a flat ABI. Any 0 helper pointer leaves
     * tier-2 on the legacy call_indirect path for that op.
     *
     * Exception-flag side effects of save_exception_flags/
     * merge_exception_flags are deliberately skipped on the inline
     * path: in HARD_FPU mode merge_exception_flags is a no-op and
     * save_exception_flags just zeros env->fp_status.exception_flags
     * — Xbox titles do not observe these IEEE exception bits. If a
     * future workload depends on env->fp_status.exception_flags being
     * cleared per-op, that's the first thing to revisit here.
     */
    uint32_t fpregs_offset;          /* offsetof(CPUX86State, fpregs[0])  */
    uint32_t fpreg_stride;           /* sizeof(FPReg)                     */
    uint32_t fpreg_native_d_off;     /* offsetof(FPReg, native_d)         */
    uint32_t fpstt_offset;           /* offsetof(CPUX86State, fpstt) [u32]*/
    uint32_t ft0_native_offset;      /* offsetof(CPUX86State, ft0_native) */
    uint32_t fpus_offset;            /* offsetof(CPUX86State, fpus) [u16] */
    uint32_t fptags_offset;          /* offsetof(CPUX86State, fptags[0])  */
    uint32_t cc_src_offset;          /* offsetof(CPUX86State, cc_src)     */
    uint32_t cc_dst_offset;          /* offsetof(CPUX86State, cc_dst)     */
    uint32_t cc_src2_offset;         /* offsetof(CPUX86State, cc_src2)    */
    uint32_t cc_op_offset;           /* offsetof(CPUX86State, cc_op) [i32]*/
    uint32_t x87_pad0;               /* keep helper pointers 8-aligned    */

    /* Comparison helpers — set fpus or cc_op/cc_src. */
    uintptr_t x87_fucom_st0_ft0_fn;
    uintptr_t x87_fcomi_st0_ft0_fn;
    uintptr_t x87_fucomi_st0_ft0_fn;

    /* Classification helper. */
    uintptr_t x87_fxam_st0_fn;

    /* Constant pushers — equivalent to fpush + ST0 = K. */
    uintptr_t x87_fldl2t_st0_fn;
    uintptr_t x87_fldl2e_st0_fn;
    uintptr_t x87_fldpi_st0_fn;
    uintptr_t x87_fldlg2_st0_fn;
    uintptr_t x87_fldln2_st0_fn;
    uintptr_t x87_fld1_st0_fn;
    uintptr_t x87_fldz_st0_fn;
    uintptr_t x87_fldz_ft0_fn;

    /* Stack-management helpers — fpush/fpop/fdecstp/fincstp/ffree. */
    uintptr_t x87_fpush_fn;
    uintptr_t x87_fpop_fn;
    uintptr_t x87_fdecstp_fn;
    uintptr_t x87_fincstp_fn;
    uintptr_t x87_ffree_stn_fn;

    /* Move helpers — fmov_*. */
    uintptr_t x87_fmov_st0_ft0_fn;
    uintptr_t x87_fmov_ft0_stn_fn;
    uintptr_t x87_fmov_st0_stn_fn;
    uintptr_t x87_fmov_stn_st0_fn;
    uintptr_t x87_fxchg_st0_stn_fn;

    /* Unary helpers — fchs/fabs/fsqrt. */
    uintptr_t x87_fchs_st0_fn;
    uintptr_t x87_fabs_st0_fn;
    uintptr_t x87_fsqrt_fn;

    /* Arithmetic helpers — fadd/fmul/fsub/fsubr/fdiv/fdivr ST0_FT0
     * and STN_ST0 (12 total). The inline-TCG path covers these when
     * g_use_fp_jit + HARD_FPU_HAS_TCG_FP_OPS are both set, but tier-2
     * still sees them as call_indirect when the bisect groups are
     * flipped or when the frontend opts out for any reason. */
    uintptr_t x87_fadd_st0_ft0_fn;
    uintptr_t x87_fmul_st0_ft0_fn;
    uintptr_t x87_fsub_st0_ft0_fn;
    uintptr_t x87_fsubr_st0_ft0_fn;
    uintptr_t x87_fdiv_st0_ft0_fn;
    uintptr_t x87_fdivr_st0_ft0_fn;
    uintptr_t x87_fadd_stn_st0_fn;
    uintptr_t x87_fmul_stn_st0_fn;
    uintptr_t x87_fsub_stn_st0_fn;
    uintptr_t x87_fsubr_stn_st0_fn;
    uintptr_t x87_fdiv_stn_st0_fn;
    uintptr_t x87_fdivr_stn_st0_fn;
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
 * Synchronous-fault unwind metadata for a single compiled TB. Produced
 * by the Rust translator from cranelift's per-IR-insn SourceLoc tags;
 * consumed by the C-side unwind index installer.
 *
 * Field semantics:
 *   - n_insns   : number of guest instructions in this TB. Same value
 *                 as `tb->icount` would carry on the tier-1 side.
 *   - n_rows    : number of (host_end, loc) rows. Cranelift may split
 *                 one guest insn across multiple non-contiguous host
 *                 byte ranges (block reordering, peephole), so n_rows
 *                 is in general >= n_insns.
 *   - host_end  : n_rows entries, each a byte offset (relative to the
 *                 cranelift function's start, i.e. the `code` pointer
 *                 returned via poll_result) one past the last byte of
 *                 the corresponding row. Sorted ascending so a fault
 *                 path can bsearch by host_pc - code_lo.
 *   - loc       : n_rows entries, parallel to host_end. loc[i] is the
 *                 0-based guest instruction index covering that range.
 *   - insn_data : n_insns * INSN_START_WORDS=3 uint64_t entries. Row
 *                 `loc[i]` of this table holds the values
 *                 `restore_state_to_opc` consumes when rewinding guest
 *                 state on synchronous fault.
 *   - _handle   : opaque pointer for cranelift_tcg_release_unwind.
 *                 Caller must invoke release exactly once per non-NULL
 *                 _handle returned by poll_result_v2.
 *
 * Lifetime: the pointer fields stay valid until the matching
 * cranelift_tcg_release_unwind(_handle) call. The C-side installer is
 * expected to memcpy the arrays into its own slab and immediately call
 * release; that decouples the C-side index lifetime from the Rust LRU.
 */
typedef struct CraneliftTcgUnwindMeta {
    uint32_t        n_insns;
    uint32_t        n_rows;
    const uint32_t *host_end;
    const uint32_t *loc;
    const uint64_t *insn_data;
    void           *_handle;
} CraneliftTcgUnwindMeta;

/*
 * Same as cranelift_tcg_poll_result, but additionally fills `out_unwind`
 * with the per-TB unwind metadata. Pass NULL for `out_unwind` to opt
 * out of unwind support (rust drops the buffer on poll). Returns 1 on
 * success, 0 if the queue is empty.
 */
int cranelift_tcg_poll_result_v2(CraneliftTcgContext *ctx,
                                 uint64_t *out_tb_pc,
                                 const void **out_code,
                                 size_t *out_size,
                                 CraneliftTcgUnwindMeta *out_unwind);

/*
 * Drop the Rust-owned unwind buffer referenced by `_handle`. Must be
 * called exactly once per non-NULL `_handle` returned by poll_result_v2.
 * Safe to invoke with NULL (no-op).
 */
void cranelift_tcg_release_unwind(void *handle);

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
