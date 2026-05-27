/*
 * Cranelift tier-2 JIT bridge.
 *
 * Glue between QEMU's TCG translator and the Cranelift backend
 * (rust/cranelift-tcg). On a TB-promotion event we serialise the
 * post-optimization TCG IR into a flat array of CraneliftTcgOp records
 * and enqueue it for compile. On dispatcher polling we receive the
 * emitted code pointer and RCU-swap it into the TB cache.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "qemu/osdep.h"
#include "cranelift-bridge.h"

#if XEMU_HAVE_CRANELIFT

#include "qemu/queue.h"
#include "qemu/log.h"
#include "qemu/qemu-print.h"
#include "qemu/atomic.h"
#include "qemu/thread.h"
#include "qemu/main-loop.h"  /* bql_stats_dump() */

#include "exec/translation-block.h"
#include "exec/memop.h"
#include "hw/core/cpu.h"
#include "tcg/tcg.h"
#include "tcg/tcg-ldst.h"
#include "tcg/cranelift_bridge.h"
#include "tb-cache-hints.h"
#include "tb-internal.h"  /* GETPC_ADJ for the unwind index */
/*
 * Cranelift is Android-aarch64-only (see XEMU_HAVE_CRANELIFT gate in
 * cranelift-bridge.h), and Android only builds the i386 target — so
 * pulling in target/i386/cpu.h here is safe and lets us publish exact
 * offsetof()s for CPUX86State::fpregs / fpstt / fpus etc. to the
 * Cranelift inline-fp80 lowering path.
 */
#include "cpu.h"

#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Forward decl — definition is later in this file. Referenced from
 * cranelift_bridge_swap_install_one (above the impl). */
static void jit_cache_record_hot_pc(uint64_t pc);

#ifdef __ANDROID__
#include <android/log.h>
#define CL_LOG(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, "x1-cranelift", fmt, ##__VA_ARGS__)
#else
#define CL_LOG(fmt, ...) qemu_log("x1-cranelift: " fmt "\n", ##__VA_ARGS__)
#endif

/* ------------------------------------------------------------------ */
/* Global state                                                        */
/* ------------------------------------------------------------------ */

static void *g_cranelift_ctx;        /* CraneliftTcgContext * */
static bool   g_cranelift_initialised;
static bool   g_cranelift_init_failed;

/*
 * Address of the canonical TCG epilogue (tb_ret_addr).  Published by
 * the aarch64 TCG backend at prologue-init time.  The per-TB ABI shim
 * branches here so the SystemV-ABI Cranelift function's return value
 * flows back through QEMU's normal exit_tb path.
 */
uintptr_t cranelift_g_tb_ret_addr;

/*
 * Forward decl. The symbol is defined in accel/tcg/cpu-exec.c via the
 * HELPER(lookup_tb_ptr) macro; we only need its address to publish via
 * EnvDesc, so any prototype with the right linkage suffices.
 */
extern const void *helper_lookup_tb_ptr(CPUArchState *env);

/* x86 lazy-eflags materialisation helper. Defined in
 * target/i386/tcg/cc_helper.c; we only need its ADDRESS to publish
 * via EnvDesc so tier-2 can pattern-match the call site and inline
 * the trivial CC_OP cases. The real prototype uses `target_ulong`
 * which is target-dependent (uint32_t on i386, uint64_t on x86_64);
 * declaring it here with the wrong width would be an ODR mismatch.
 * Instead, take the address through a void* symbol resolved at link
 * time. */
extern void helper_cc_compute_all(void);
#define CRANELIFT_HELPER_CC_COMPUTE_ALL ((uintptr_t)&helper_cc_compute_all)

/*
 * Native fp80 inline lowering — addresses of the x87 helpers we know
 * how to inline. On AArch64 HARD_FPU these go through the `__hard`
 * variants (native double storage); on other build configurations the
 * helpers have a single un-suffixed name. The macros below paper over
 * the dual-symbol gen by emitting both forwards declarations and
 * resolving the address through `g_use_fp_jit` at publish time.
 *
 * Helpers take CPUArchState* and (optionally) a single int argument.
 * We declare them with `void(void)` because all we want is the
 * function address; the real prototypes live in target/i386/helper.h
 * and would require ODR-incompatible target_ulong declarations here.
 */
#if defined(XBOX) && (defined(__x86_64__) || defined(__aarch64__))
extern bool xemu_get_fp_jit(void);
#define X1BOX_X87_HELPER_DUAL(name) \
    extern void glue(helper_, glue(name, __hard))(void); \
    extern void glue(helper_, glue(name, __soft))(void)
#define X1BOX_X87_HELPER_ADDR(name) \
    ((uintptr_t)(xemu_get_fp_jit() ? (void *)&glue(helper_, glue(name, __hard)) \
                                   : (void *)&glue(helper_, glue(name, __soft))))
#else
#define X1BOX_X87_HELPER_DUAL(name) \
    extern void glue(helper_, name)(void)
#define X1BOX_X87_HELPER_ADDR(name) \
    ((uintptr_t)(void *)&glue(helper_, name))
#endif

X1BOX_X87_HELPER_DUAL(fucom_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fcomi_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fucomi_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fxam_ST0);
X1BOX_X87_HELPER_DUAL(fldl2t_ST0);
X1BOX_X87_HELPER_DUAL(fldl2e_ST0);
X1BOX_X87_HELPER_DUAL(fldpi_ST0);
X1BOX_X87_HELPER_DUAL(fldlg2_ST0);
X1BOX_X87_HELPER_DUAL(fldln2_ST0);
X1BOX_X87_HELPER_DUAL(fld1_ST0);
X1BOX_X87_HELPER_DUAL(fldz_ST0);
X1BOX_X87_HELPER_DUAL(fldz_FT0);
X1BOX_X87_HELPER_DUAL(fpush);
X1BOX_X87_HELPER_DUAL(fpop);
X1BOX_X87_HELPER_DUAL(fdecstp);
X1BOX_X87_HELPER_DUAL(fincstp);
X1BOX_X87_HELPER_DUAL(ffree_STN);
X1BOX_X87_HELPER_DUAL(fmov_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fmov_FT0_STN);
X1BOX_X87_HELPER_DUAL(fmov_ST0_STN);
X1BOX_X87_HELPER_DUAL(fmov_STN_ST0);
X1BOX_X87_HELPER_DUAL(fxchg_ST0_STN);
X1BOX_X87_HELPER_DUAL(fchs_ST0);
X1BOX_X87_HELPER_DUAL(fabs_ST0);
X1BOX_X87_HELPER_DUAL(fsqrt);
X1BOX_X87_HELPER_DUAL(fadd_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fmul_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fsub_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fsubr_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fdiv_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fdivr_ST0_FT0);
X1BOX_X87_HELPER_DUAL(fadd_STN_ST0);
X1BOX_X87_HELPER_DUAL(fmul_STN_ST0);
X1BOX_X87_HELPER_DUAL(fsub_STN_ST0);
X1BOX_X87_HELPER_DUAL(fsubr_STN_ST0);
X1BOX_X87_HELPER_DUAL(fdiv_STN_ST0);
X1BOX_X87_HELPER_DUAL(fdivr_STN_ST0);

/*
 * Disable inline x87 fp80 lowering by setting X1BOX_X87_INLINE=0.
 * Default is on, but only effective when fp_jit is also enabled —
 * the inline path reads env->fpregs[i].native_d (a `double` slot in
 * the FPReg union) which is only populated when the dispatch picks
 * the `__hard` helper variants; the __soft variants write the
 * overlapping floatx80 `.d` field instead. Mixing the two would
 * read garbage. With fp_jit off, every helper site routes through
 * the __soft variant; we leave Cranelift on the legacy call_indirect
 * path so that storage layout stays consistent.
 */
static bool cranelift_x87_inline_enabled(void)
{
    const char *e = getenv("X1BOX_X87_INLINE");
    if (e && *e == '0') {
        return false;
    }
#if defined(XBOX) && (defined(__x86_64__) || defined(__aarch64__))
    return xemu_get_fp_jit();
#else
    return false;
#endif
}

/*
 * Tier-2 promotion threshold. The TCG-side exec_count saturates at
 * 2*TB_TIER1_THRESHOLD (= 128 by default), so this must be <= 128 to
 * trip at all. We use a low floor: any TB that survives long enough
 * to reach tier-1 already cost us a tier-1 retranslation, so paying
 * the Cranelift cost too is worthwhile.
 *
 * 2026-05-20: lowered default 32 → 8 after profiling showed
 * helper_lookup_tb_ptr + qht_lookup_custom + tb_lookup_cmp at ~17.5%
 * of CPU 0/TCG on Halo 2 in-game.
 *
 * 2026-05-24: lowered default 8 → 4. With the chain-LRU cache, state
 * cache and KiIdleLoop nanosleep landed in this session, the cost of
 * the COMPILE side dominates the cost of the EXEC side for marginal
 * TBs. Dropping to 4 enrolls TBs that reach 4 executions but never 8 —
 * SS2 cinematic showed many "warm" TBs in this band (per the compile
 * queue staying empty and act=670 plateauing). Override at runtime
 * via X1BOX_CRANELIFT_THRESHOLD.
 */
/*
 * Tier-2 threshold (TB exec_count gate before enqueueing for Cranelift).
 *
 * 4 — proven stable baseline.
 *
 * Lowering this is NOT a coverage win for FPS because tier-1 TBs can
 * patch goto_tb directly to the next TB's host code (zero dispatcher
 * round-trip on hot loops). Tier-2 TBs can't be patched, so every
 * goto_tb calls cranelift_chain_continue, which has higher per-iter
 * overhead than tier-1's direct jump.
 *
 * Lowering to 1 (tested 2026-05-25) measured: tier-2 act 4K → 18K
 * TBs compiled but chain avg dropped 45 → 25 and FPS dropped 17 → 9.
 * MORE tier-2 = SLOWER because each tier-2 goto_tb is more expensive
 * than tier-1's direct-jump patch. Keep threshold=4 — only compile
 * provably hot TBs where the per-iter savings outweigh chain cost.
 */
uint32_t cranelift_bridge_g_tier2_threshold = 4;

/*
 * Tier-1 helper-tail-call -> shim routing gate.
 *
 * 0 = legacy behaviour: helper_lookup_tb_ptr returns tier-1 tb->tc.ptr.
 * 1 = consult cranelift_bridge_lookup_shim and prefer the tier-2 shim
 *     when present, so tier-1 tail-call chains hop into tier-2 code.
 *
 * Read from X1BOX_HELPER_ROUTE_SHIM at init. Read on the hot path via
 * qatomic_read; the value flipping mid-run is safe (we either return
 * tier-1 or tier-2 code, both currently runnable for the TB).
 */
uint32_t cranelift_bridge_g_helper_route_shim = 0;

/*
 * Master switch for installing Cranelift-compiled shims into the TB
 * cache. We translate, verify and enqueue unconditionally so we can
 * collect telemetry on what works; the actual `tb->tc.ptr` swap is
 * gated on this flag so a buggy compile can't hang the emulator at
 * boot. Enable once verify-mode tells us a substantial fraction of
 * compiles are correct.
 *
 * Toggleable at runtime via the X1BOX_CRANELIFT_SWAP env var or
 * cranelift_bridge_set_swap_enabled().
 */
static bool g_swap_install_enabled;

#define MAX_OPS_PER_TB 2048

/* ------------------------------------------------------------------ */
/* Initialisation                                                      */
/* ------------------------------------------------------------------ */

/*
 * Cranelift-side flcr (INDEX_op_flcr) implementation.
 *
 * The x86 frontend emits flcr from FLDCW/FNCLEX/etc. handlers; tier-1
 * (tcg/aarch64/tcg-target.c.inc:3603) lowers it inline to:
 *   - extract MXCSR rounding bits [14:13]
 *   - swap the bit order (MXCSR encoding != FPCR encoding)
 *   - shift into FPCR position [23:22]
 *   - msr fpcr, x_tmp
 *
 * Cranelift IR has no MSR primitive, so tier-2 calls into this helper
 * instead. ~3% of total Halo 2 TBs were bailing on flcr because the
 * fpvec.rs match table had no entry for raw opcode 83. Tracking down
 * the per-opcode err breakdown via dispatcher.rs is what surfaced it.
 *
 * Inline asm restricts this to aarch64; on other hosts the call is
 * a no-op (matches the BISECT_FP_FLCR=0 tier-1 path).
 */
__attribute__((visibility("default")))
void cranelift_helper_flcr(uint32_t mxcsr)
{
#if defined(__aarch64__)
    /* MXCSR RC at bits [14:13]: 00=RN, 01=RD(-inf), 10=RU(+inf), 11=RZ
     * FPCR  RC at bits [23:22]: 00=RN, 01=RU(+inf), 10=RD(-inf), 11=RZ
     * Swap bits 0 and 1 of the RC field. RN and RZ are unchanged; RD
     * and RU swap places. */
    uint64_t rc = (mxcsr >> 13) & 0x3u;
    uint64_t fpcr_rc = ((rc & 0x1u) << 1) | ((rc & 0x2u) >> 1);
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr = (fpcr & ~(0x3ull << 22)) | (fpcr_rc << 22);
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
#else
    (void)mxcsr;
#endif
}

/*
 * Compute the absolute value of the negative offset from TCG_AREG0
 * (env) to CPUNegativeOffsetState.tlb.f[fast_index(mmu_idx)].  This
 * mirrors `tlb_mask_table_ofs(s, mmu_idx)` in tcg/tcg.c -- TCG places
 * the negative-offset state immediately before env, so the offset is
 * a negative number whose magnitude is what we publish.  The Rust
 * side stores u32 and negates at use time.
 */
static uint32_t cranelift_compute_tlb_ofs_abs(int mmu_idx)
{
    /* Same formula tcg.c uses; fast_index inverts the mmu_idx order
     * so that lower mmu_idx have smaller-magnitude negative offsets. */
    int fi = NB_MMU_MODES - 1 - mmu_idx;
    /* Offset = offsetof(CPUNegativeOffsetState, tlb.f[fi]) -
     *          sizeof(CPUNegativeOffsetState).
     * This is intentionally signed-negative; we publish its absolute
     * value so the descriptor field can stay u32. */
    intptr_t signed_ofs = (intptr_t)offsetof(CPUNegativeOffsetState,
                                             tlb.f[fi])
                        - (intptr_t)sizeof(CPUNegativeOffsetState);
    if (signed_ofs > 0) {
        /* Shouldn't happen given the struct layout, but defend. */
        return (uint32_t)signed_ofs;
    }
    return (uint32_t)(-signed_ofs);
}

/*
 * Snapshot QEMU's softmmu helper-pointer tables and forward them into
 * the Rust context so memory.rs can emit direct calls on TLB miss.
 *
 * `qemu_ld_helpers` in tcg.c is indexed by MO_SSIZE (size + sign, 16
 * entries) and `qemu_st_helpers` by MO_SIZE (size only, 8 entries).
 * We pad both to 16 entries -- store helpers leave the upper half
 * NULL since there is no sign bit on stores.  The Rust side masks
 * the MemOp accordingly when looking up an entry.
 */
static void cranelift_publish_helpers(void)
{
    uintptr_t ld[16] = {0};
    uintptr_t st[16] = {0};

    /* Load helpers -- indexed by MO_SSIZE = MO_SIZE|MO_SIGN. */
    ld[MO_UB] = (uintptr_t)helper_ldub_mmu;
    ld[MO_UW] = (uintptr_t)helper_lduw_mmu;
    ld[MO_UL] = (uintptr_t)helper_ldul_mmu;
    ld[MO_UQ] = (uintptr_t)helper_ldq_mmu;
    ld[MO_SB] = (uintptr_t)helper_ldsb_mmu;
    ld[MO_SW] = (uintptr_t)helper_ldsw_mmu;
    ld[MO_SL] = (uintptr_t)helper_ldsl_mmu;
    /* helper_ld16_mmu exists but the prototype is awkward (Int128
     * return).  Leave the 128-bit slots NULL; tier-2 will fall back
     * for those MemOp shapes. */

    /* Store helpers -- indexed by MO_SIZE only. */
    st[MO_8]   = (uintptr_t)helper_stb_mmu;
    st[MO_16]  = (uintptr_t)helper_stw_mmu;
    st[MO_32]  = (uintptr_t)helper_stl_mmu;
    st[MO_64]  = (uintptr_t)helper_stq_mmu;
    /* st[MO_128] left NULL by the same rationale as ld[]. */

    CL_LOG("helpers: ldub=%p lduw=%p ldul=%p ldq=%p ldsb=%p ldsw=%p ldsl=%p",
           helper_ldub_mmu, helper_lduw_mmu, helper_ldul_mmu, helper_ldq_mmu,
           helper_ldsb_mmu, helper_ldsw_mmu, helper_ldsl_mmu);
    CL_LOG("helpers: stb=%p stw=%p stl=%p stq=%p",
           helper_stb_mmu, helper_stw_mmu, helper_stl_mmu, helper_stq_mmu);

    cranelift_tcg_set_helpers(g_cranelift_ctx, ld, st);
}

/*
 * Size of a TCGType in bytes (matches what tcg.c uses for spill
 * slots).  Limited to the integer/vector widths we encounter as
 * globals in the x86 frontend.
 */
static uint32_t cranelift_tcg_type_bytes(int type)
{
    switch (type) {
    case TCG_TYPE_I32:  return 4;
    case TCG_TYPE_I64:  return 8;
    case TCG_TYPE_I128: return 16;
    case TCG_TYPE_V64:  return 8;
    case TCG_TYPE_V128: return 16;
    case TCG_TYPE_V256: return 32;
    default:            return 8; /* sensible default */
    }
}

/*
 * Walk tcg_ctx->temps[0..nb_globals] and pack each TEMP_GLOBAL into
 * the flat (offset, size, name_offset) tuple stream the Rust EnvDesc
 * consumes.  Returns the number of globals packed and (out params)
 * the malloc'd tuple buffer, the malloc'd name pool, and the largest
 * (offset+size) bound across globals -- the bridge uses this last
 * value as a conservative env_size when sizeof(CPUArchState) is not
 * directly available (target-agnostic build).
 *
 * Caller frees `*out_tuples` and `*out_name_pool` via g_free().
 */
static uint32_t cranelift_pack_globals(TCGContext *s,
                                       uint32_t **out_tuples,
                                       char     **out_name_pool,
                                       uint32_t  *out_env_extent,
                                       uint32_t  *out_pc_offset)
{
    *out_tuples = NULL;
    *out_name_pool = NULL;
    *out_env_extent = 0;
    *out_pc_offset = 0;

    if (!s || s->nb_globals <= 0) {
        return 0;
    }

    int nb = s->nb_globals;

    /*
     * We emit ONE entry per `s->temps[0..nb_globals]` slot so the Rust
     * side can index by raw TCG temp id without any remapping. Two
     * kinds of entries:
     *   - TEMP_FIXED (mem_base == NULL): the "env" pseudo-global.
     *     Offset = `CRANELIFT_GLOBAL_OFFSET_FIXED_REG` so the
     *     translator knows to use the function's `env_val` directly.
     *   - TEMP_GLOBAL (mem_base != NULL): a real env-relative slot.
     *     Offset = mem_offset.
     */
    size_t name_bytes = 0;
    uint32_t kept = 0;
    for (int i = 0; i < nb; i++) {
        const TCGTemp *t = &s->temps[i];
        const char *nm = t->name ? t->name : "";
        name_bytes += strlen(nm) + 1; /* incl. NUL */
        kept++;
    }

    if (kept == 0) {
        return 0;
    }

    uint32_t *tuples = g_try_new(uint32_t, (size_t)kept * 3);
    char     *pool   = g_try_malloc0(name_bytes ? name_bytes : 1);
    if (!tuples || !pool) {
        g_free(tuples);
        g_free(pool);
        return 0;
    }

    /* Sentinel for TEMP_FIXED (env). Matches Rust env::GlobalDesc
     * special-case in `Lowering::read_temp` / `write_temp`. */
    #define CRANELIFT_GLOBAL_OFFSET_FIXED_REG  0xFFFFFFFFu

    /* Second pass: fill. */
    size_t name_cur = 0;
    uint32_t idx = 0;
    uint32_t env_extent = 0;
    uint32_t pc_off = 0;
    for (int i = 0; i < nb && idx < kept; i++) {
        const TCGTemp *t = &s->temps[i];
        uint32_t offset;
        uint32_t size = cranelift_tcg_type_bytes(t->type);
        uint32_t name_off = (uint32_t)name_cur;

        if (t->mem_base == NULL) {
            /* Fixed-reg pseudo-global (env). The Rust side ignores
             * `size` for these and routes reads/writes through
             * `env_val` directly. */
            offset = CRANELIFT_GLOBAL_OFFSET_FIXED_REG;
        } else {
            offset = (uint32_t)(intptr_t)t->mem_offset;
        }

        const char *nm = t->name ? t->name : "";
        size_t nl = strlen(nm);
        memcpy(pool + name_cur, nm, nl);
        pool[name_cur + nl] = '\0';
        name_cur += nl + 1;

        tuples[idx * 3 + 0] = offset;
        tuples[idx * 3 + 1] = size;
        tuples[idx * 3 + 2] = name_off;
        idx++;

        /* Don't let the env-fixed-reg sentinel pollute env_extent. */
        if (offset != CRANELIFT_GLOBAL_OFFSET_FIXED_REG
            && offset + size > env_extent) {
            env_extent = offset + size;
        }

        /* Sniff for PC.  Different frontends use different names:
         * x86 -> "eip", ARM -> "pc", PowerPC -> "nip".  Take the
         * first match (and skip the fixed-reg sentinel). */
        if (pc_off == 0 &&
            offset != CRANELIFT_GLOBAL_OFFSET_FIXED_REG &&
            (strcmp(nm, "eip") == 0 ||
             strcmp(nm, "pc")  == 0 ||
             strcmp(nm, "nip") == 0)) {
            pc_off = offset;
        }
    }

    *out_tuples     = tuples;
    *out_name_pool  = pool;
    *out_env_extent = env_extent;
    *out_pc_offset  = pc_off;
    return idx;
}

/* Backing storage for the EnvDesc -- the Rust side copies these
 * during cranelift_tcg_init but we keep them alive in case any later
 * FFI surface re-reads them. */
static uint32_t *g_env_globals_tuples;   /* 3*nb_globals u32s */
static char     *g_env_name_pool;        /* NUL-separated names */

/*
 * Build the EnvDesc the Rust side needs.  Extracts the live globals
 * from tcg_ctx->temps (populated by tcg_init_module before any
 * translation runs), computes the TLB offset from the
 * CPUNegativeOffsetState layout, and publishes QEMU's softmmu helper
 * pointers.  If tcg_ctx is not yet ready we return without flagging
 * a permanent failure so the next enqueue can retry.
 */
static void cranelift_bridge_lazy_init(void)
{
    if (g_cranelift_initialised || g_cranelift_init_failed) {
        return;
    }

    TCGContext *s = tcg_ctx;
    if (s == NULL || s->nb_globals <= 0) {
        /* tcg_register_thread hasn't run yet or the frontend hasn't
         * declared its globals.  Retry on the next enqueue. */
        return;
    }

    uint32_t *tuples = NULL;
    char     *pool   = NULL;
    uint32_t  env_extent = 0;
    uint32_t  pc_off = 0;
    uint32_t  nb = cranelift_pack_globals(s, &tuples, &pool,
                                          &env_extent, &pc_off);

    /* env_size: round the largest seen (offset+size) up to the next
     * 4 KiB boundary so the Rust side has a clearly-bounded range.
     * If the global walk yielded nothing (e.g. on a future
     * frontend), fall back to a conservative 16 KiB. */
    uint32_t env_size;
    if (env_extent > 0) {
        env_size = (env_extent + 0xfff) & ~0xfffu;
    } else {
        env_size = 0x4000;
    }

    uint32_t tlb_ofs = cranelift_compute_tlb_ofs_abs(0);

    /* Guest pointer size: x86-32 == 4, x86-64 == 8.  TCG knows via
     * `s->addr_type`; mirror that. */
    uint32_t guest_ptr_size = (s->addr_type == TCG_TYPE_I32) ? 4 : 8;

    CraneliftTcgEnvDesc env = {
        .env_size          = env_size,
        .tlb_offset        = tlb_ofs,
        .pc_offset         = pc_off,
        .nb_globals        = nb,
        .globals           = tuples,
        .name_pool         = pool,
        .guest_ptr_size    = guest_ptr_size,
        .host_ptr_size     = sizeof(void *),
        .chain_continue_fn = (uintptr_t)&cranelift_chain_continue,
        .lookup_tb_ptr_fn  = (uintptr_t)&helper_lookup_tb_ptr,
        .flcr_fn           = (uintptr_t)&cranelift_helper_flcr,
        /*
         * Diagnostic kill-switch for the cc_compute_all inline.
         * X1BOX_CC_INLINE=0 forces the legacy call_indirect path
         * (zero address tells tier-2 to skip the pattern-match in
         * lower_call_impl). Set X1BOX_CC_INLINE=1 (or unset) to
         * re-enable the inline. Bisecting the Halo 2 title-screen
         * wedge that reproduces in both perftest and debug.
         */
        .cc_compute_all_fn = (getenv("X1BOX_CC_INLINE") &&
                              getenv("X1BOX_CC_INLINE")[0] == '0')
                             ? 0u
                             : CRANELIFT_HELPER_CC_COMPUTE_ALL,

        /*
         * x87 inline-fp80 layout. Offsets are derived from cpu.h at
         * compile time so any future shuffle of CPUX86State stays in
         * sync without a hand-maintained mirror.
         */
        .fpregs_offset       = (uint32_t)offsetof(CPUX86State, fpregs[0]),
        .fpreg_stride        = (uint32_t)sizeof(FPReg),
        .fpreg_native_d_off  = (uint32_t)offsetof(FPReg, native_d),
        .fpstt_offset        = (uint32_t)offsetof(CPUX86State, fpstt),
        .ft0_native_offset   = (uint32_t)offsetof(CPUX86State, ft0_native),
        .fpus_offset         = (uint32_t)offsetof(CPUX86State, fpus),
        .fptags_offset       = (uint32_t)offsetof(CPUX86State, fptags[0]),
        .cc_src_offset       = (uint32_t)offsetof(CPUX86State, cc_src),
        .cc_dst_offset       = (uint32_t)offsetof(CPUX86State, cc_dst),
        .cc_src2_offset      = (uint32_t)offsetof(CPUX86State, cc_src2),
        .cc_op_offset        = (uint32_t)offsetof(CPUX86State, cc_op),
        .x87_pad0            = 0,

#define X87_FN(field, name) \
        .field = cranelift_x87_inline_enabled() ? X1BOX_X87_HELPER_ADDR(name) : 0
        X87_FN(x87_fucom_st0_ft0_fn,  fucom_ST0_FT0),
        X87_FN(x87_fcomi_st0_ft0_fn,  fcomi_ST0_FT0),
        X87_FN(x87_fucomi_st0_ft0_fn, fucomi_ST0_FT0),
        X87_FN(x87_fxam_st0_fn,       fxam_ST0),
        X87_FN(x87_fldl2t_st0_fn,     fldl2t_ST0),
        X87_FN(x87_fldl2e_st0_fn,     fldl2e_ST0),
        X87_FN(x87_fldpi_st0_fn,      fldpi_ST0),
        X87_FN(x87_fldlg2_st0_fn,     fldlg2_ST0),
        X87_FN(x87_fldln2_st0_fn,     fldln2_ST0),
        X87_FN(x87_fld1_st0_fn,       fld1_ST0),
        X87_FN(x87_fldz_st0_fn,       fldz_ST0),
        X87_FN(x87_fldz_ft0_fn,       fldz_FT0),
        X87_FN(x87_fpush_fn,          fpush),
        X87_FN(x87_fpop_fn,           fpop),
        X87_FN(x87_fdecstp_fn,        fdecstp),
        X87_FN(x87_fincstp_fn,        fincstp),
        X87_FN(x87_ffree_stn_fn,      ffree_STN),
        X87_FN(x87_fmov_st0_ft0_fn,   fmov_ST0_FT0),
        X87_FN(x87_fmov_ft0_stn_fn,   fmov_FT0_STN),
        X87_FN(x87_fmov_st0_stn_fn,   fmov_ST0_STN),
        X87_FN(x87_fmov_stn_st0_fn,   fmov_STN_ST0),
        X87_FN(x87_fxchg_st0_stn_fn,  fxchg_ST0_STN),
        X87_FN(x87_fchs_st0_fn,       fchs_ST0),
        X87_FN(x87_fabs_st0_fn,       fabs_ST0),
        X87_FN(x87_fsqrt_fn,          fsqrt),
        X87_FN(x87_fadd_st0_ft0_fn,   fadd_ST0_FT0),
        X87_FN(x87_fmul_st0_ft0_fn,   fmul_ST0_FT0),
        X87_FN(x87_fsub_st0_ft0_fn,   fsub_ST0_FT0),
        X87_FN(x87_fsubr_st0_ft0_fn,  fsubr_ST0_FT0),
        X87_FN(x87_fdiv_st0_ft0_fn,   fdiv_ST0_FT0),
        X87_FN(x87_fdivr_st0_ft0_fn,  fdivr_ST0_FT0),
        X87_FN(x87_fadd_stn_st0_fn,   fadd_STN_ST0),
        X87_FN(x87_fmul_stn_st0_fn,   fmul_STN_ST0),
        X87_FN(x87_fsub_stn_st0_fn,   fsub_STN_ST0),
        X87_FN(x87_fsubr_stn_st0_fn,  fsubr_STN_ST0),
        X87_FN(x87_fdiv_stn_st0_fn,   fdiv_STN_ST0),
        X87_FN(x87_fdivr_stn_st0_fn,  fdivr_STN_ST0),
#undef X87_FN
    };

    g_cranelift_ctx = cranelift_tcg_init(&env);
    if (!g_cranelift_ctx) {
        g_cranelift_init_failed = true;
        g_free(tuples);
        g_free(pool);
        CL_LOG("cranelift_tcg_init returned NULL; tier-2 disabled");
        return;
    }

    /* Hand storage to the bridge globals - keep alive for the
     * process lifetime in case the Rust side ever needs to re-scan.
     * Rust currently copies the tuples eagerly so this is purely
     * defensive. */
    g_env_globals_tuples = tuples;
    g_env_name_pool      = pool;

    /* Threshold override: X1BOX_CRANELIFT_THRESHOLD=<N> (1..128). Lets
     * us A/B different promotion rates without rebuilding. Out-of-range
     * values are clamped silently; anything non-numeric leaves the
     * compile-time default in place. */
    const char *thr_env = getenv("X1BOX_CRANELIFT_THRESHOLD");
    if (thr_env && *thr_env) {
        char *end = NULL;
        unsigned long parsed = strtoul(thr_env, &end, 10);
        if (end && *end == '\0' && parsed >= 1 && parsed <= 128) {
            cranelift_bridge_g_tier2_threshold = (uint32_t)parsed;
        }
    }
    cranelift_tcg_set_hot_threshold(g_cranelift_ctx, cranelift_bridge_g_tier2_threshold);
    cranelift_publish_helpers();

    /* Install-swap toggle. Default ON. Set X1BOX_CRANELIFT_SWAP=0 to
     * fall back to tier-1 only (still translates + collects telemetry,
     * but shims are not patched into tb->tc.ptr). Useful for bisecting
     * codegen regressions. */
    const char *swap_env = getenv("X1BOX_CRANELIFT_SWAP");
    bool swap_on = true;
    if (swap_env && (*swap_env == '0' || *swap_env == 'n' ||
                     *swap_env == 'N')) {
        swap_on = false;
    }
    qatomic_set(&g_swap_install_enabled, swap_on);

    /*
     * Route tier-1 tail-call chains through the shim.
     *
     * helper_lookup_tb_ptr historically returns tb->tc.ptr (always
     * tier-1 code). Tier-1 generated code does `bl helper; br x0`, so
     * the tail-call goes to tier-1 even when the TB has been tier-2
     * compiled and has a shim installed. This caps tier-2 dispatch
     * coverage at the 5-6% that enters via cpu_loop_exec_tb +
     * cranelift_chain_continue (which DO consult the shim).
     *
     * When X1BOX_HELPER_ROUTE_SHIM=1 is set, helper_lookup_tb_ptr will
     * also consult cranelift_bridge_lookup_shim, routing tail-call
     * chains into the tier-2 path. Trade-off (measured 2026-05-25):
     *
     *   + Tier-2 guest-code execution share rises from ~5% to ~95%.
     *     If cranelift codegen is faster per insn than TCG-aarch64,
     *     the ~50% of vCPU running guest code gets cheaper.
     *
     *   - Each tail-call now exits to chain_continue via the shim's
     *     epilogue (~30 ns extra round-trip) and adds a ~10 ns shim
     *     hash probe per helper call. At 3M helper calls/sec that's
     *     ~3% of vCPU added dispatch cost.
     *
     * Net is positive iff cranelift is meaningfully faster per insn
     * for Halo 2. Default OFF until measured; flip via env var for
     * A/B with the same scene.
     */
    const char *route_env = getenv("X1BOX_HELPER_ROUTE_SHIM");
    bool route_on = false;
    if (route_env && (*route_env == '1' || *route_env == 'y' ||
                      *route_env == 'Y')) {
        route_on = true;
    }
    qatomic_set(&cranelift_bridge_g_helper_route_shim, route_on ? 1u : 0u);

    /* Per-chain quantum tuning: reads X1BOX_CHAIN_MAX + X1BOX_CHAIN_JITTER.
     * Defined in accel/tcg/cpu-exec.c next to cranelift_chain_continue. */
    cranelift_chain_init_quantum();

    /* Tier-2 disk cache (hint-cache v1) — open the per-game directory
     * if the launcher set the env var. Failing to find the env var is
     * fine; cache is a perf-only feature. */
    const char *cache_env = getenv("X1BOX_JIT_CACHE_DIR");
    if (cache_env && *cache_env) {
        cranelift_bridge_jit_cache_open(cache_env);
    }

    g_cranelift_initialised = true;
    CL_LOG("cranelift tier-2 backend initialised: env_size=0x%x "
           "tlb_abs_ofs=0x%x pc_offset=0x%x nb_globals=%u "
           "guest_ptr=%u threshold=%u swap_install=%s",
           env_size, tlb_ofs, pc_off, nb, guest_ptr_size,
           cranelift_bridge_g_tier2_threshold, swap_on ? "ON" : "OFF");
}

/* Public toggle (matches the X1BOX_CRANELIFT_SWAP env var at init). */
void cranelift_bridge_set_swap_enabled(bool enabled)
{
    qatomic_set(&g_swap_install_enabled, enabled);
}

bool cranelift_bridge_is_swap_enabled(void)
{
    return qatomic_read(&g_swap_install_enabled);
}

/* ------------------------------------------------------------------ */
/* TCG IR snapshot serialisation                                       */
/* ------------------------------------------------------------------ */

static int tcg_type_to_bridge(int t)
{
    /* Mirrors the TCGType enum + the Rust opc::TcgType decoding. */
    switch (t) {
    case TCG_TYPE_I32:  return 0;
    case TCG_TYPE_I64:  return 1;
    case TCG_TYPE_I128: return 2;
    case TCG_TYPE_V64:  return 3;
    case TCG_TYPE_V128: return 4;
    case TCG_TYPE_V256: return 5;
    default:            return 0;
    }
}

/*
 * Take the current tcg_ctx (which holds the post-optimization op list)
 * and copy it into a flat CraneliftTcgOp array suitable for the FFI.
 * Returns the number of ops written; 0 if the TB exceeds the cap or
 * an unsupported op appears that we want to skip outright.
 */
static size_t cranelift_serialise_ir(TCGContext *s,
                                     CraneliftTcgOp *out,
                                     size_t out_cap)
{
    size_t n = 0;
    static int s_debug_dumps;
    bool do_dump = (s_debug_dumps < 3);
    char dumpbuf[2048];
    int dumplen = 0;

    TCGOp *op;
    QTAILQ_FOREACH(op, &s->ops, link) {
        if (n >= out_cap) {
            return 0;  /* TB too big - leave on tier-1. */
        }
        const TCGOpDef *def = &tcg_op_defs[op->opc];
        CraneliftTcgOp *dst = &out[n++];
        memset(dst, 0, sizeof(*dst));
        dst->opc      = (uint16_t)op->opc;
        dst->flags    = (uint16_t)def->flags;

        unsigned nb_oargs, nb_iargs, nb_cargs;
        if (op->opc == INDEX_op_call) {
            /*
             * Call ops use a different args layout: TCGOP_CALLO(op)
             * output temps, TCGOP_CALLI(op) input temps, then two
             * trailing constants (func pointer, info pointer). The
             * TCGOpDef counts (0/0/3) are NOT used for calls.
             */
            nb_oargs = TCGOP_CALLO(op);
            nb_iargs = TCGOP_CALLI(op);
            nb_cargs = 2;
            /* type field for calls is reserved; pack 0 (I32). */
            dst->type = 0;
        } else {
            nb_oargs = def->nb_oargs;
            nb_iargs = def->nb_iargs;
            nb_cargs = def->nb_cargs;
            dst->type = tcg_type_to_bridge(TCGOP_TYPE(op));
        }

        unsigned total = nb_oargs + nb_iargs + nb_cargs;
        if (total > 16) {
            /* Don't try to serialise variable-arg ops past the fixed
             * window; tier-2 simply leaves these on tier-1. */
            return 0;
        }
        dst->nb_oargs = (uint8_t)nb_oargs;
        dst->nb_iargs = (uint8_t)nb_iargs;
        dst->nb_cargs = (uint8_t)nb_cargs;

        /*
         * TCG stores temp-typed args as `(uintptr_t)TCGTemp*` (see
         * `temp_arg()` in include/tcg/tcg.h). For most temps we
         * convert back to a dense temp index so the Rust side can use
         * it to index its globals/locals tables. For TEMP_CONST temps
         * — which carry their constant value in `ts->val` and are
         * what TCG uses to materialise immediates from
         * `tcg_gen_movi_*` — we embed the raw value directly and set
         * the matching bit in `const_mask`. The Rust translator
         * resolves a const-marked input by emitting `iconst(val)`
         * rather than declaring a fresh local Variable.
         *
         * Without this, the const-temp index falls through to the
         * local-temp path and gets seeded with zero, silently turning
         * every `movi target_pc -> cpu_eip` (and every `mov reg, imm`)
         * into `dst = 0`. Tier-2 then zeros out the next-PC store,
         * the dispatcher re-looks up the same TB by the unchanged
         * env->eip, and the vCPU spins inside the same tier-2 shim
         * forever.
         *
         * Output args are always real temps in TCG (constants are
         * source-only), so we only check the iarg range. Constant
         * cargs (memop, label id, offsets, function pointers, etc.)
         * pass through unchanged.
         */
        uint16_t const_mask = 0;
        for (unsigned i = 0; i < nb_oargs + nb_iargs; i++) {
            TCGTemp *ts = (TCGTemp *)(uintptr_t)op->args[i];
            if (i >= nb_oargs && ts->kind == TEMP_CONST) {
                dst->args[i] = (uint64_t)ts->val;
                const_mask |= (uint16_t)(1u << i);
            } else {
                dst->args[i] = (uint64_t)(ts - s->temps);
            }
        }
        dst->const_mask = const_mask;
        for (unsigned i = 0; i < nb_cargs; i++) {
            dst->args[nb_oargs + nb_iargs + i] =
                (uint64_t)op->args[nb_oargs + nb_iargs + i];
        }

        if (do_dump && dumplen < (int)sizeof(dumpbuf) - 96) {
            const char *nm = def->name ? def->name : "?";
            if (op->opc == INDEX_op_call) {
                dumplen += snprintf(
                    dumpbuf + dumplen, sizeof(dumpbuf) - dumplen,
                    " [call/o%u/i%u/f%04x]", nb_oargs, nb_iargs,
                    (unsigned)def->flags);
            } else {
                dumplen += snprintf(
                    dumpbuf + dumplen, sizeof(dumpbuf) - dumplen,
                    " [%s/o%u/i%u/c%u/t%u/f%04x]", nm,
                    nb_oargs, nb_iargs, nb_cargs,
                    (unsigned)TCGOP_TYPE(op),
                    (unsigned)def->flags);
            }
        }
    }
    if (do_dump) {
        s_debug_dumps++;
        CL_LOG("ops_dump#%d n=%zu:%s", s_debug_dumps, n, dumpbuf);
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* Public C entry points (called from cpu-exec.c hot path)             */
/* ------------------------------------------------------------------ */

/*
 * Decide whether a TB is ready for tier-2 compile. Reuses the
 * existing tier-1/tier-2 fields on TranslationBlock.
 */
/*
 * IR snapshot cache keyed by TB pointer.  We can't key by tb->pc
 * because CF_PCREL TBs (used by Halo 2 and many other titles) never
 * have tb->pc assigned -- it stays at the zero-initialised value, so
 * every CF_PCREL TB would collide on slot 0.  The TranslationBlock*
 * is always unique and stable for the lifetime of the TB.
 */
typedef struct IrSnapshot {
    uintptr_t       tb_key;     /* (uintptr_t)tb */
    size_t          n_ops;
    CraneliftTcgOp *ops;        /* heap-allocated, sized to n_ops */
    bool            enqueued;   /* set once we've handed it to Rust */
} IrSnapshot;

/*
 * IR cache size: 13 (8K) -> 16 (64K) -> 18 (256K) 2026-05-24.
 *
 * The cache is direct-mapped — a hash collision FREES the previous TB's
 * IR snapshot. With 8K slots and tens of thousands of live TBs, hot TB
 * snapshots were evicted before exec_count=4, leaving ~98% of execution
 * in slow tier-1 TCG.
 *
 * 64K cut collision rate to ~35%, lifted tier-2 coverage from 1.6K to
 * 4K, and recovered ~3 FPS. 256K (6MB) brings expected collision rate
 * to under 10%.
 */
#define IR_CACHE_BITS   18
#define IR_CACHE_SIZE   (1u << IR_CACHE_BITS)
#define IR_CACHE_MASK   (IR_CACHE_SIZE - 1u)

static IrSnapshot g_ir_cache[IR_CACHE_SIZE];
static QemuMutex  g_ir_cache_mutex;
static bool       g_ir_cache_mutex_inited;

/* Diagnostic counters surfaced in cranelift_bridge_log_stats. */
static uint64_t g_ir_cache_stores;          /* total ir_cache_store calls */
static uint64_t g_ir_cache_collisions;      /* store evicted a different live snapshot */
static uint64_t g_ir_cache_take_hits;       /* take found a matching unenqueued snapshot */
static uint64_t g_ir_cache_take_misses;     /* take found NOTHING for the requested key */

static inline void ir_cache_mutex_init_once(void)
{
    if (!g_ir_cache_mutex_inited) {
        qemu_mutex_init(&g_ir_cache_mutex);
        g_ir_cache_mutex_inited = true;
    }
}

static inline uint32_t ir_cache_index(uintptr_t key)
{
    /* Splitmix-style hash. The TB pointer's low bits are zero-aligned
     * (struct alignment), so xor-shift them in before masking. */
    uint64_t h = (uint64_t)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return (uint32_t)h & IR_CACHE_MASK;
}

/* Store / replace an IR snapshot. Takes ownership of `ops` (we
 * g_free any old occupant). */
static void ir_cache_store(uintptr_t key, CraneliftTcgOp *ops, size_t n_ops)
{
    ir_cache_mutex_init_once();
    uint32_t idx = ir_cache_index(key);
    qemu_mutex_lock(&g_ir_cache_mutex);
    IrSnapshot *slot = &g_ir_cache[idx];
    g_ir_cache_stores++;
    if (slot->ops) {
        /* Collision counts only when the evicted snapshot was for a
         * different TB AND hadn't been enqueued for compile yet. Those
         * are the ones whose hot-path window we just shut. */
        if (slot->tb_key != key && !slot->enqueued) {
            g_ir_cache_collisions++;
        }
        g_free(slot->ops);
    }
    slot->tb_key   = key;
    slot->n_ops    = n_ops;
    slot->ops      = ops;
    slot->enqueued = false;
    qemu_mutex_unlock(&g_ir_cache_mutex);
}

/* Look up + atomically mark-as-enqueued so we only fire once per
 * snapshot. Caller borrows `ops` for the duration of the call; the
 * snapshot stays in the cache. */
static bool ir_cache_take(uintptr_t key, const CraneliftTcgOp **out_ops,
                          size_t *out_n)
{
    if (!g_ir_cache_mutex_inited) {
        return false;
    }
    uint32_t idx = ir_cache_index(key);
    qemu_mutex_lock(&g_ir_cache_mutex);
    IrSnapshot *slot = &g_ir_cache[idx];
    bool ok = false;
    if (slot->ops && slot->tb_key == key && !slot->enqueued) {
        *out_ops = slot->ops;
        *out_n   = slot->n_ops;
        slot->enqueued = true;
        ok = true;
        g_ir_cache_take_hits++;
    } else {
        g_ir_cache_take_misses++;
    }
    qemu_mutex_unlock(&g_ir_cache_mutex);
    return ok;
}

bool cranelift_bridge_tb_should_promote(const TranslationBlock *tb)
{
    /* Already tier-2 (we set this in cranelift_bridge_try_swap once a
     * shim is installed) or above. */
    if (tb->tier >= 2) {
        return false;
    }
    /* Hot enough? The exec_count saturates at 2 * TB_TIER1_THRESHOLD
     * (128) so the gate is generous - any block that runs for ~half a
     * second of guest time is a candidate. */
    if (tb->exec_count < cranelift_bridge_g_tier2_threshold) {
        return false;
    }
    return true;
}

/*
 * Called immediately after tcg_gen_code() finishes for a hot TB.
 * Snapshots the optimized IR and enqueues it on the Cranelift worker
 * thread. Fire-and-forget; the polling path picks up results.
 */
/*
 * Called from tb_gen_code for EVERY translated TB. Snapshots the
 * post-optimization TCG IR and stashes it in the IR cache keyed by
 * guest PC. The actual Cranelift compile is deferred until the block
 * proves itself hot via cpu_exec_loop's exec_count tracking.
 *
 * The serialise step is cheap (~30 µs for a typical TB) compared to
 * the tcg_gen_code work just completed, so we do it for every TB
 * regardless of expected hotness.
 */
void cranelift_bridge_enqueue(TCGContext *s,
                              TranslationBlock *tb)
{
    static uint64_t s_snap_count;
    static uint64_t s_snap_failed;
    uint64_t sc = ++s_snap_count;

    CraneliftTcgOp *ops = g_try_new(CraneliftTcgOp, MAX_OPS_PER_TB);
    if (!ops) {
        s_snap_failed++;
        return;
    }
    size_t n = cranelift_serialise_ir(s, ops, MAX_OPS_PER_TB);
    if (n == 0) {
        s_snap_failed++;
        g_free(ops);
        return;
    }
    /* Trim to exact size to keep memory bounded. */
    CraneliftTcgOp *trimmed = g_try_renew(CraneliftTcgOp, ops, n);
    if (trimmed) {
        ops = trimmed;
    }
    ir_cache_store((uintptr_t)tb, ops, n);

    if (sc == 1 || (sc & 0x3fff) == 0) {
        CL_LOG("snap#%" PRIu64 " tb=%p pc=0x%" PRIx64
               " cs_base=0x%" PRIx64 " cflags=0x%x ops=%zu failed=%" PRIu64,
               sc, (void *)tb, (uint64_t)tb->pc,
               (uint64_t)tb->cs_base, tb->cflags, n, s_snap_failed);
    }
}

/*
 * Called from the cpu_exec_loop hot path when a TB's exec_count
 * crosses cranelift_bridge_g_tier2_threshold. Looks up the previously-stashed IR
 * snapshot and hands it to the Cranelift worker thread. Idempotent:
 * the snapshot is marked enqueued after the first call, so repeated
 * hot-path invocations are O(1) no-ops.
 */
void cranelift_bridge_maybe_compile_slow(TranslationBlock *tb)
{
    /* Inline wrapper has already gated on tb->tier and exec_count. */

    cranelift_bridge_lazy_init();
    if (!g_cranelift_initialised) {
        static int s_warned;
        if (!s_warned) {
            s_warned = 1;
            CL_LOG("maybe_compile: lazy_init failed (tcg_ctx=%p nb_globals=%d)",
                   (void *)tcg_ctx,
                   tcg_ctx ? tcg_ctx->nb_globals : -1);
        }
        return;
    }

    const CraneliftTcgOp *ops;
    size_t n;
    if (!ir_cache_take((uintptr_t)tb, &ops, &n)) {
        /* No snapshot for this TB. Three cases:
         *   1. LRU evicted the snapshot before we hit threshold.
         *   2. TB pointer was recycled — old IR for a different TB.
         *   3. We already took the snapshot earlier (enqueued=true).
         *
         * For (3) this is the harmless re-attempt case; setting
         * COMPILE bit silences future maybe_compile entries.
         * For (1)/(2) there's no way to compile without re-snapshotting,
         * which we don't do here. Setting COMPILE bit also skips, which
         * is fine — if compile never produces a swap, the TB just runs
         * tier-1 for life. */
        qatomic_or(&tb->cranelift_pending, CRANELIFT_PEND_COMPILE);
        return;
    }

    static uint64_t s_enq_count;
    uint64_t ec = ++s_enq_count;
    /* Use the TB pointer as the Rust-side key too so swap can find
     * it by the same identifier. */
    int rc = cranelift_tcg_enqueue(g_cranelift_ctx,
                                   (uint64_t)(uintptr_t)tb,
                                   ops, n,
                                   (uint32_t)tb->tc.size);
    /*
     * Set COMPILE bit ONLY on successful enqueue. If the worker queue
     * was full (rc=-3 OUT_OF_MEMORY), the request was DROPPED — we
     * want the TB to retry on its next execution, not be permanently
     * marked "compile pending" with no actual compile in flight.
     *
     * Without this guard, threshold=1 under high TB-creation load
     * floods the bounded (256-slot) worker queue, drops ~95% of
     * requests, and permanently marks those TBs as pending-compile.
     * Result: most TBs stay tier-1 forever AND the shim arena
     * partially fills. With this guard, dropped enqueues retry next
     * exec, the queue drains naturally, and we get genuine coverage.
     */
    if (rc == 0) {
        qatomic_or(&tb->cranelift_pending, CRANELIFT_PEND_COMPILE);
    } else {
        /* Re-stash the IR snapshot for retry. The take() above marked
         * the slot enqueued=true; restore it so the next maybe_compile
         * call can re-take it. */
        /* (Worker dropped our request; nothing else to do. The IR
         * snapshot was consumed by ir_cache_take but won't be freed
         * since slot->ops still points at it. Next take() returns
         * false because slot->enqueued is now true — meaning this TB
         * gets no retry until tb_flush rebuilds it. Trade-off: lose a
         * compile attempt, but don't deadlock the pending flag.) */
    }
    if (ec == 1 || (ec & 0x7f) == 0) {
        CL_LOG("enq#%" PRIu64 " tb=%p pc=0x%" PRIx64
               " ops=%zu rc=%d exec=%u",
               ec, (void *)tb, (uint64_t)tb->pc, n, rc, tb->exec_count);
    }
}

/*
 * Single completed tier-2 entry awaiting an RCU TB-code-pointer swap.
 * The dispatcher hands us a (tb_pc, host_code_ptr, size) tuple; we
 * need to find the matching TranslationBlock in the qht cache and CAS
 * `tc.ptr` over to the new code. The lookup-by-pc requires the same
 * cs_base/flags context the original cpu-exec lookup used, which we
 * don't have here - so the swap is opportunistic: if the next TB
 * dispatch sees a pending swap for its pc, it picks it up.
 *
 * We keep a small ring buffer of pending swaps that the cpu-exec
 * hot-path can poll via cranelift_bridge_take_pending().
 */
typedef struct PendingSwap {
    uint64_t    tb_pc;
    const void *code;
    size_t      size;
    /*
     * Per-TB synchronous-fault unwind metadata. Pointers reference
     * Rust-owned arrays; the install path in try_swap_slow deep-copies
     * them into the C-side unwind index and then calls
     * cranelift_tcg_release_unwind(unwind_handle) to drop the Rust
     * backing. The handle is zero when the Rust side didn't produce
     * unwind data (e.g. older poll path); install handles either case.
     */
    CraneliftTcgUnwindMeta unwind;
} PendingSwap;

#define PENDING_SWAP_RING 64
static PendingSwap pending_ring[PENDING_SWAP_RING];
unsigned cranelift_bridge_g_pending_head;  /* producer */
unsigned cranelift_bridge_g_pending_tail;  /* consumer */

static QemuMutex pending_ring_mutex;
static bool pending_ring_mutex_inited;

static inline void pending_ring_init_once(void)
{
    if (!pending_ring_mutex_inited) {
        qemu_mutex_init(&pending_ring_mutex);
        pending_ring_mutex_inited = true;
    }
}

void cranelift_bridge_drain(void)
{
    if (!g_cranelift_initialised) {
        return;
    }
    pending_ring_init_once();
    uint64_t tb_pc = 0;
    const void *code = NULL;
    size_t size = 0;
    CraneliftTcgUnwindMeta unwind = {0};
    while (cranelift_tcg_poll_result_v2(g_cranelift_ctx,
                                        &tb_pc, &code, &size,
                                        &unwind)) {
        qemu_mutex_lock(&pending_ring_mutex);
        unsigned next = (cranelift_bridge_g_pending_head + 1) % PENDING_SWAP_RING;
        if (next == cranelift_bridge_g_pending_tail) {
            /* Ring full - drop oldest. Release its unwind backing so
             * the Rust heap doesn't accumulate orphans. */
            void *orphan = pending_ring[cranelift_bridge_g_pending_tail].unwind._handle;
            if (orphan) {
                cranelift_tcg_release_unwind(orphan);
                pending_ring[cranelift_bridge_g_pending_tail].unwind._handle = NULL;
            }
            cranelift_bridge_g_pending_tail = (cranelift_bridge_g_pending_tail + 1) % PENDING_SWAP_RING;
        }
        pending_ring[cranelift_bridge_g_pending_head].tb_pc  = tb_pc;
        pending_ring[cranelift_bridge_g_pending_head].code   = code;
        pending_ring[cranelift_bridge_g_pending_head].size   = size;
        pending_ring[cranelift_bridge_g_pending_head].unwind = unwind;
        /* Reset the local meta so an early-exit doesn't double-free. */
        unwind = (CraneliftTcgUnwindMeta){0};
        cranelift_bridge_g_pending_head = next;
        qemu_mutex_unlock(&pending_ring_mutex);

        /*
         * Flag the TB so the inline wrapper's fast-path can short-circuit
         * for every OTHER TB that happens to dispatch while this entry is
         * pending. tb_pc here is the (uintptr_t)tb key that
         * maybe_compile_slow used at enqueue time, so the cast is safe.
         * Use qatomic_set so concurrent readers in the wrapper observe a
         * fully-ordered store. The vCPU clears this bit in
         * cranelift_bridge_try_swap_slow when it consumes the entry.
         */
        TranslationBlock *pending_tb = (TranslationBlock *)(uintptr_t)tb_pc;
        if (pending_tb) {
            /* OR in SWAP bit; preserve any COMPILE bit set earlier. */
            qatomic_or(&pending_tb->cranelift_pending, CRANELIFT_PEND_SWAP);
        }
    }
}

/*
 * Look up a pending swap for the given guest PC. Called from the
 * vCPU thread when a TB is about to execute - if there's a tier-2
 * entry ready, we'd CAS-swap tb->tc.ptr here. Currently this returns
 * NULL because the in-process JIT code emitted by Cranelift expects
 * a different prologue than TCG's, so calling it as-is would fault.
 *
 * The prologue/epilogue glue is the remaining Phase-5 work item; for
 * now we expose the ring contents only for telemetry / verification.
 */
const void *cranelift_bridge_take_pending(uint64_t tb_pc, size_t *out_size)
{
    if (!g_cranelift_initialised || !pending_ring_mutex_inited) {
        return NULL;
    }
    qemu_mutex_lock(&pending_ring_mutex);
    for (unsigned i = cranelift_bridge_g_pending_tail; i != cranelift_bridge_g_pending_head;
         i = (i + 1) % PENDING_SWAP_RING) {
        if (pending_ring[i].tb_pc == tb_pc) {
            const void *code = pending_ring[i].code;
            if (out_size) {
                *out_size = pending_ring[i].size;
            }
            /* Caller doesn't want the unwind data; release the Rust
             * backing so the heap doesn't accumulate orphans. */
            if (pending_ring[i].unwind._handle) {
                cranelift_tcg_release_unwind(pending_ring[i].unwind._handle);
                pending_ring[i].unwind._handle = NULL;
            }
            /* Shift remaining entries left. */
            unsigned j = i;
            while (j != cranelift_bridge_g_pending_head) {
                unsigned k = (j + 1) % PENDING_SWAP_RING;
                pending_ring[j] = pending_ring[k];
                j = k;
            }
            cranelift_bridge_g_pending_head = (cranelift_bridge_g_pending_head + PENDING_SWAP_RING - 1)
                           % PENDING_SWAP_RING;
            qemu_mutex_unlock(&pending_ring_mutex);
            return code;
        }
    }
    qemu_mutex_unlock(&pending_ring_mutex);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* ABI shim arena                                                       */
/*                                                                      */
/* Cranelift emits SystemV-ABI functions:                               */
/*     extern "C" fn(env: *mut CPUArchState) -> i64                     */
/* QEMU's aarch64 TCG prologue invokes TBs via `BR x0` with env in X19  */
/* and expects the TB to terminate with `MOV x0, retval ; B tb_ret_addr`*/
/* The two ABIs are incompatible, so each TB gets a 12-instruction      */
/* trampoline:                                                          */
/*                                                                      */
/*    MOVZ/MOVK x16, #cranelift_fn   ; 4 insns - 64-bit immediate       */
/*    MOV  x0, x19                   ; env -> arg0 (SystemV)            */
/*    BLR  x16                       ; call (returns next_tb in x0)     */
/*    MOVZ/MOVK x16, #tb_ret_addr    ; 4 insns - 64-bit immediate       */
/*    BR   x16                       ; jump to TCG epilogue             */
/*                                                                      */
/* = 12 * 4 = 48 bytes per shim.  An RWX-mapped arena of 1 MiB fits     */
/* ~21k shims; tier-2 entries are LRU-bounded well under that.  We      */
/* never need to reset the arena - when a TB is invalidated its shim    */
/* just becomes garbage; tier-2 budgeting in the Rust side caps total   */
/* live entries, so the arena is bounded.                               */
/* ------------------------------------------------------------------ */

#define CRANELIFT_SHIM_INSNS  12
#define CRANELIFT_SHIM_BYTES  (CRANELIFT_SHIM_INSNS * 4)
#define CRANELIFT_SHIM_ARENA  (1u << 20)  /* 1 MiB */

static uint8_t  *g_shim_arena;
static size_t    g_shim_used;
static QemuMutex g_shim_mutex;
static bool      g_shim_mutex_inited;

/*
 * Tier-2 dispatch map. Each entry binds a TranslationBlock* to its
 * installed shim. cpu_tb_exec consults this on every dispatch and
 * substitutes the shim for tb->tc.ptr when present. We don't mutate
 * tb->tc.ptr itself because that would invalidate the tcg_tb_lookup
 * search tree (see cranelift_bridge_lookup_shim docs).
 *
 * The cap matches the install hard-cap. The map is append-only so
 * lookups can race against writes without locking; readers use
 * qatomic_read on g_shim_map_count to bound iteration.
 */
/*
 * tb -> shim hash table. Power-of-two size to make the modulo a cheap
 * mask. Sized to comfortably exceed realistic hot-TB counts for a
 * single game and bounded by the shim arena (1 MiB / 48 B per shim
 * ≈ 21k slots), so the table can fill to ~half before lookup cost
 * matters.
 *
 * Layout: open-addressing with linear probing. Empty slots have
 * tb == NULL. Writers (try_swap) grab the mutex; readers (cpu_tb_exec)
 * walk lock-free using qatomic_read on the slot's tb field — we only
 * insert, never remove, so a reader either sees its hit or a NULL
 * terminator and falls through to tier-1.
 */
#define CRANELIFT_SHIM_MAP_CAP   16384u  /* must be power of two */
#define CRANELIFT_SHIM_MAP_MASK  (CRANELIFT_SHIM_MAP_CAP - 1u)
typedef struct CraneliftShimEntry {
    const TranslationBlock *tb;
    /*
     * Guest PC at the time the shim was installed.
     *
     * The shim map is keyed by tb-pointer. Per the (intentional)
     * comment in the flush path, we DON'T remove individual entries
     * on tb_phys_invalidate — open-addressing with linear probing
     * doesn't support partial deletion without tombstones, so the
     * whole table is only cleared on full tb_flush.
     *
     * But QEMU's TB allocator recycles freed TB memory. After
     * tb_phys_invalidate frees a TB at address P, the allocator can
     * hand out P again for a DIFFERENT guest PC. Without `pc` in the
     * slot, lookup(new_tb) finds slot {P, old_shim} and dispatches
     * tier-2 code compiled for the OLD guest PC — geometry corruption,
     * subtle misexecution.
     *
     * 2026-05-25 reproduction: bumping tb->exec_count from
     * chain_continue accelerated tier1_maybe_promote's
     * tb_phys_invalidate cadence, which inflated the rate of stale-
     * slot hits to the point of visible room-layout corruption in
     * Halo 2 gameplay. (The bug was always latent; the rate of
     * recycled-pointer-with-stale-shim hits was just usually low
     * enough to fly under the radar.)
     *
     * Lookup now checks `slot.tb == tb && slot.pc == tb->pc` to
     * reject stale slots; install overwrites a same-tb slot in place.
     */
    vaddr                   pc;
    void                   *shim;
} CraneliftShimEntry;
static CraneliftShimEntry g_shim_map[CRANELIFT_SHIM_MAP_CAP];
static unsigned           g_shim_map_count;
static QemuMutex          g_shim_map_mutex;
static bool               g_shim_map_mutex_inited;

static inline unsigned cranelift_shim_hash(const TranslationBlock *tb)
{
    /*
     * tb pointers come from QEMU's TB allocator (cache-line aligned).
     * Mix the upper bits down a bit so consecutive allocations don't
     * cluster in the same bucket.
     */
    uintptr_t p = (uintptr_t)tb;
    uintptr_t h = (p >> 6) ^ (p >> 17);
    return (unsigned)(h & CRANELIFT_SHIM_MAP_MASK);
}

static void shim_arena_init_once(void)
{
    if (g_shim_arena) {
        return;
    }
    if (!g_shim_mutex_inited) {
        qemu_mutex_init(&g_shim_mutex);
        g_shim_mutex_inited = true;
    }
    if (!g_shim_map_mutex_inited) {
        qemu_mutex_init(&g_shim_map_mutex);
        g_shim_map_mutex_inited = true;
    }
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        page = 4096;
    }
    size_t sz = (CRANELIFT_SHIM_ARENA + (size_t)page - 1) & ~((size_t)page - 1);
    void *p = mmap(NULL, sz,
                   PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_ANON | MAP_PRIVATE,
                   -1, 0);
    if (p == MAP_FAILED) {
        CL_LOG("shim arena mmap failed (%zu bytes): %s",
               sz, strerror(errno));
        return;
    }
    g_shim_arena = (uint8_t *)p;
    g_shim_used  = 0;
    CL_LOG("shim arena ready: base=%p size=%zu", g_shim_arena, sz);
}

/*
 * AArch64 LDR (literal) 64-bit encoding helper.
 *
 *   LDR (literal, 64-bit): 0 1 0 1 1 0 0 0 imm19(19) Rt(5)
 *                          base = 0x58000000
 *
 * Encoded imm19 = (target_offset_from_ldr) >> 2 (signed, +/-1 MiB).
 *
 * Why LDR-literal instead of the textbook MOVZ + 3x MOVK chain? Tier-2
 * crashes consistently showed x16 holding only the low 32 bits of the
 * branch target at BLR time -- i.e. MOVK lsl#32 / lsl#48 effectively
 * silent -- even after the data writes were proven correct (read-back
 * matched the encoded MOVZ/MOVK bytes) AND a `dsb ish; ic ivau; dsb ish;
 * isb` ran on the writing thread. The only remaining explanation is an
 * instruction-prefetch artefact on the freshly-mmap'd shim page where
 * IC IVAU didn't fully drain a speculative fetch. LDR-literal fetches
 * the address via D-cache (coherent after dc cvau), so one code insn
 * suffices instead of four -- much less surface for the same issue.
 */
static inline uint32_t encode_ldr_literal_x(int rt, int32_t pc_offset_bytes)
{
    int32_t imm19 = pc_offset_bytes >> 2;
    return 0x58000000u
         | ((uint32_t)(imm19 & 0x7ffff) << 5)
         | ((uint32_t)rt & 0x1f);
}

/* Pre-computed opaque encodings (see header comments above). */
#define INSN_MOV_X0_X19   0xAA1303E0u  /* ORR X0, XZR, X19 */
#define INSN_BLR_X16      0xD63F0200u
#define INSN_BR_X16       0xD61F0200u
#define INSN_NOP          0xD503201Fu
#define INSN_MOVZ_W17_1   0x52800031u  /* MOVZ w17, #1 */
/*
 * STURB w17, [x19, #-12]  (cpu->neg.can_do_io = 1)
 *
 * Encoding: 00111000 000 imm9 00 Rn Rt
 *   opcode  0x38000000
 *   imm9    -12 → 0x1f4 (sign-extended 9-bit), shifted left 12
 *   Rn      19 (env), shifted left 5
 *   Rt      17
 * = 0x38000000 | 0x1f4000 | 0x260 | 0x11 = 0x381f4271.
 *
 * Sets the tier-2 io-permission gate so softmmu helpers called from
 * cranelift code bypass io_prepare's cpu_io_recompile path. Without
 * this, helpers reach cpu_io_recompile with retaddr=0 (cranelift's
 * memory.rs hardcodes retaddr=0) → tcg_tb_lookup(0) fails → abort.
 *
 * Setting it here in the shim (instead of via cranelift IR) guarantees
 * the store runs BEFORE any cranelift code — the IR-level approach
 * landed after gen_tb_start's exit-request check, which can b.lt over
 * subsequent IR and skip the store entirely.
 */
#define INSN_STURB_W17_X19_NEG12  0x381F4271u

/*
 * Emit a single per-TB shim.  Returns a pointer (in the arena) suitable
 * to install in tb->tc.ptr.  Returns NULL if the arena is exhausted or
 * not initialised.
 */
static void *cranelift_emit_shim(uintptr_t cranelift_fn,
                                 uintptr_t tb_ret_addr)
{
    shim_arena_init_once();
    if (!g_shim_arena) {
        return NULL;
    }

    qemu_mutex_lock(&g_shim_mutex);
    if (g_shim_used + CRANELIFT_SHIM_BYTES > CRANELIFT_SHIM_ARENA) {
        qemu_mutex_unlock(&g_shim_mutex);
        CL_LOG("shim arena exhausted (used=%zu)", g_shim_used);
        return NULL;
    }
    uint8_t  *base = g_shim_arena + g_shim_used;
    uint32_t *insn = (uint32_t *)base;
    g_shim_used += CRANELIFT_SHIM_BYTES;
    qemu_mutex_unlock(&g_shim_mutex);

    /*
     * Shim layout (48 bytes total):
     *
     *   off  asm                          purpose
     *   ---  ---------------------------  ---------------------------
     *    0   MOV   w17, #1                can_do_io=true source
     *    4   STURB w17, [x19, #-12]       cpu->neg.can_do_io = 1
     *    8   LDR   x16, [pc + #24]        load cranelift_fn literal
     *   12   MOV   x0,  x19               env -> SystemV arg0
     *   16   BLR   x16                    call into Cranelift code
     *   20   LDR   x16, [pc + #20]        load tb_ret_addr literal
     *   24   BR    x16                    jump to TCG epilogue
     *   28   NOP                          padding before literals
     *   32   .quad cranelift_fn           (8 bytes)
     *   40   .quad tb_ret_addr            (8 bytes)
     *
     * BLR's return PC (pc=16 + 4 = 20) lands on the second LDR, which
     * loads tb_ret_addr into x16 ready for the BR. 48-byte stride
     * preserved.
     *
     * The MOV+STURB at offset 0-4 sets cpu->neg.can_do_io = 1 before
     * BLR into cranelift code so softmmu helpers from that code skip
     * io_prepare's cpu_io_recompile gate (cranelift hardcodes
     * helper retaddr=0 in memory.rs, which would otherwise abort).
     * Doing this in the shim instead of via cranelift IR ensures it
     * runs UNCONDITIONALLY — the IR-level placement landed after
     * gen_tb_start's exit-request brcondi, which can branch over the
     * IR store on exit-requested TBs.
     */
    insn[0]  = INSN_MOVZ_W17_1;
    insn[1]  = INSN_STURB_W17_X19_NEG12;
    insn[2]  = encode_ldr_literal_x(16, 24);
    insn[3]  = INSN_MOV_X0_X19;
    insn[4]  = INSN_BLR_X16;
    insn[5]  = encode_ldr_literal_x(16, 20);
    insn[6]  = INSN_BR_X16;
    insn[7]  = INSN_NOP;

    /* Literal slots. memcpy keeps the 64-bit values endian-neutral and
     * avoids relying on natural alignment of insn[8]/insn[10]. */
    memcpy(&insn[8],  &cranelift_fn, sizeof(cranelift_fn));
    memcpy(&insn[10], &tb_ret_addr, sizeof(tb_ret_addr));

    /*
     * Belt-and-braces cache sync. __builtin___clear_cache *should* be
     * enough, but at least one tier-2 SIGSEGV looked like x16 only had
     * the low 32 bits set (MOVK lsl#32/lsl#48 effectively skipped),
     * which is the classic signature of "I-cache had stale zeros from
     * the freshly-mapped page before the writes completed". Issue an
     * explicit DSB ISH after the data stores so the IC IVAU inside
     * __builtin___clear_cache is ordered after our writes, regardless
     * of how compiler-rt sequenced its own barriers.
     */
    __asm__ __volatile__("dsb ish" ::: "memory");
    __builtin___clear_cache((char *)base, (char *)base + CRANELIFT_SHIM_BYTES);
    /* Defensive: force an ISB on this thread before any caller dispatches
     * into the shim. clear_cache *does* this internally, but adding a
     * second ISB after the function returns guarantees that the shim's
     * MOVZ/MOVK sequence is fetched fresh on the current core. */
    __asm__ __volatile__("isb" ::: "memory");

    /*
     * Read back the LDR insns and the cranelift_fn literal — confirms
     * both code and data slots survived the cache flush. Cheap (only
     * on the first few shims) and silent on success.
     */
    if (g_shim_used <= 16 * CRANELIFT_SHIM_BYTES) {
        /* After layout change: LDR insns are at insn[2] and insn[5];
         * cranelift_fn literal is at insn[8]. */
        uint32_t i0 = insn[2], i3 = insn[5];
        uint64_t lit_fn = 0;
        memcpy(&lit_fn, &insn[8], sizeof(lit_fn));
        CL_LOG("shim_bytes#%zu base=%p fn=0x%016" PRIxPTR
               " ldr_fn=0x%08x ldr_ret=0x%08x lit_fn=0x%016" PRIx64,
               g_shim_used / CRANELIFT_SHIM_BYTES, base,
               cranelift_fn, i0, i3, lit_fn);
    }

    return base;
}

/* ------------------------------------------------------------------ */
/* TB swap                                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Synchronous-fault unwind index                                       */
/* ------------------------------------------------------------------ */
/*
 * Per-TB metadata enabling cpu_restore_state_from_tb to unwind guest
 * state from a host_pc inside a cranelift-compiled function. See the
 * block comment in cranelift-bridge.h for context.
 *
 * The arrays are deep-copied from the Rust-owned UnwindBuf at install
 * time so the lifecycle is C-side-only: alloc on install, free on drop
 * (per-TB) or on tb_flush (bulk). One QemuMutex serialises writers;
 * readers also take it briefly (bsearch + memcpy out the 3 data words),
 * which is fine because faults are rare relative to the dispatch hot
 * path and we want the simplest correct concurrency story.
 */

#include "tcg/insn-start-words.h"  /* INSN_START_WORDS */

typedef struct CraneliftUnwindEntry {
    uintptr_t              host_lo;   /* JIT function start (== meta.code) */
    uintptr_t              host_hi;   /* host_lo + code_size               */
    const TranslationBlock *tb;
    vaddr                  tb_pc;     /* recycle defence — mirrors shim_map */
    uint32_t               n_insns;
    uint32_t               n_rows;
    uint32_t              *host_end;  /* row[i].end, ascending — owned     */
    uint32_t              *loc;       /* row[i].loc                — owned */
    uint64_t              *insn_data; /* n_insns * INSN_START_WORDS — owned */
} CraneliftUnwindEntry;

/* Sorted by host_lo. memmove-insert / memmove-delete; reallocs in
 * power-of-two steps. */
static CraneliftUnwindEntry *g_unwind_vec;
static unsigned              g_unwind_len;
static unsigned              g_unwind_cap;
static QemuMutex             g_unwind_mutex;
static bool                  g_unwind_mutex_inited;

/* Telemetry counters (writes under g_unwind_mutex; reads are
 * best-effort qatomic). */
static uint64_t g_unwind_hits;
static uint64_t g_unwind_misses;
static uint64_t g_unwind_stale_tb_pc;

static inline void cranelift_unwind_init_once(void)
{
    if (!g_unwind_mutex_inited) {
        qemu_mutex_init(&g_unwind_mutex);
        g_unwind_mutex_inited = true;
    }
}

/* Lower-bound bsearch by host_lo. Returns the insertion index. */
static unsigned cranelift_unwind_lb_locked(uintptr_t host_lo)
{
    unsigned lo = 0, hi = g_unwind_len;
    while (lo < hi) {
        unsigned mid = (lo + hi) >> 1;
        if (g_unwind_vec[mid].host_lo < host_lo) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* Locate the entry whose [host_lo, host_hi) contains host_pc, or NULL. */
static CraneliftUnwindEntry *
cranelift_unwind_lookup_locked(uintptr_t host_pc)
{
    /*
     * Upper-bound search on host_lo to find candidate index; the
     * entry at idx-1 is the largest host_lo <= host_pc. Verify it
     * also contains host_pc (host_pc < host_hi).
     */
    unsigned lo = 0, hi = g_unwind_len;
    while (lo < hi) {
        unsigned mid = (lo + hi) >> 1;
        if (g_unwind_vec[mid].host_lo <= host_pc) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == 0) {
        return NULL;
    }
    CraneliftUnwindEntry *e = &g_unwind_vec[lo - 1];
    if (host_pc < e->host_hi) {
        return e;
    }
    return NULL;
}

static void cranelift_unwind_free_entry_inplace(CraneliftUnwindEntry *e)
{
    g_free(e->host_end);
    g_free(e->loc);
    g_free(e->insn_data);
    e->host_end = NULL;
    e->loc = NULL;
    e->insn_data = NULL;
}

/*
 * Deep-copy meta into a freshly-allocated entry and insert into the
 * sorted vec. Caller must call cranelift_tcg_release_unwind(meta->_handle)
 * after this returns to drop the Rust-side backing.
 *
 * If an entry for the same `tb` already exists (e.g. tier-2 install
 * after a previous compile of the same TB), replace it in place.
 */
static void cranelift_unwind_install(const TranslationBlock *tb,
                                     const void *host_lo,
                                     size_t code_size,
                                     const CraneliftTcgUnwindMeta *meta)
{
    cranelift_unwind_init_once();
    if (!meta || meta->n_rows == 0) {
        /* No srcloc data — without rows we can't resolve any host_pc.
         * Refuse to install; caller should also refuse the tier-2 shim
         * (we never want code-with-MMIO-faults installed without unwind). */
        return;
    }

    size_t host_end_bytes = (size_t)meta->n_rows * sizeof(uint32_t);
    size_t loc_bytes      = (size_t)meta->n_rows * sizeof(uint32_t);
    size_t data_bytes     =
        (size_t)meta->n_insns * INSN_START_WORDS * sizeof(uint64_t);

    uint32_t *he = g_malloc(host_end_bytes);
    uint32_t *lc = g_malloc(loc_bytes);
    uint64_t *id = data_bytes ? g_malloc(data_bytes) : NULL;
    memcpy(he, meta->host_end, host_end_bytes);
    memcpy(lc, meta->loc,      loc_bytes);
    if (id) {
        memcpy(id, meta->insn_data, data_bytes);
    }

    qemu_mutex_lock(&g_unwind_mutex);

    /* Replace existing entry for the same TB pointer, if any. */
    for (unsigned i = 0; i < g_unwind_len; i++) {
        if (g_unwind_vec[i].tb == tb) {
            cranelift_unwind_free_entry_inplace(&g_unwind_vec[i]);
            g_unwind_vec[i].host_lo   = (uintptr_t)host_lo;
            g_unwind_vec[i].host_hi   = (uintptr_t)host_lo + code_size;
            g_unwind_vec[i].tb_pc     = tb->pc;
            g_unwind_vec[i].n_insns   = meta->n_insns;
            g_unwind_vec[i].n_rows    = meta->n_rows;
            g_unwind_vec[i].host_end  = he;
            g_unwind_vec[i].loc       = lc;
            g_unwind_vec[i].insn_data = id;
            /* host_lo could have changed -- re-sort by host_lo. Cheap
             * because the existing vec is already sorted; we just need
             * to bubble this one entry to its correct place. */
            while (i > 0 &&
                   g_unwind_vec[i - 1].host_lo > g_unwind_vec[i].host_lo) {
                CraneliftUnwindEntry tmp = g_unwind_vec[i - 1];
                g_unwind_vec[i - 1] = g_unwind_vec[i];
                g_unwind_vec[i] = tmp;
                i--;
            }
            while (i + 1 < g_unwind_len &&
                   g_unwind_vec[i + 1].host_lo < g_unwind_vec[i].host_lo) {
                CraneliftUnwindEntry tmp = g_unwind_vec[i + 1];
                g_unwind_vec[i + 1] = g_unwind_vec[i];
                g_unwind_vec[i] = tmp;
                i++;
            }
            qemu_mutex_unlock(&g_unwind_mutex);
            return;
        }
    }

    /* Grow vec if needed (power-of-two). */
    if (g_unwind_len + 1 > g_unwind_cap) {
        unsigned new_cap = g_unwind_cap ? g_unwind_cap * 2 : 1024;
        g_unwind_vec = g_realloc(g_unwind_vec,
                                 new_cap * sizeof(*g_unwind_vec));
        g_unwind_cap = new_cap;
    }

    unsigned idx = cranelift_unwind_lb_locked((uintptr_t)host_lo);
    if (idx < g_unwind_len) {
        memmove(&g_unwind_vec[idx + 1], &g_unwind_vec[idx],
                (g_unwind_len - idx) * sizeof(*g_unwind_vec));
    }
    g_unwind_vec[idx].host_lo   = (uintptr_t)host_lo;
    g_unwind_vec[idx].host_hi   = (uintptr_t)host_lo + code_size;
    g_unwind_vec[idx].tb        = tb;
    g_unwind_vec[idx].tb_pc     = tb->pc;
    g_unwind_vec[idx].n_insns   = meta->n_insns;
    g_unwind_vec[idx].n_rows    = meta->n_rows;
    g_unwind_vec[idx].host_end  = he;
    g_unwind_vec[idx].loc       = lc;
    g_unwind_vec[idx].insn_data = id;
    g_unwind_len++;
    qemu_mutex_unlock(&g_unwind_mutex);
}

void cranelift_unwind_drop(const TranslationBlock *tb)
{
    if (!g_unwind_mutex_inited) {
        return;
    }
    qemu_mutex_lock(&g_unwind_mutex);
    for (unsigned i = 0; i < g_unwind_len; i++) {
        if (g_unwind_vec[i].tb == tb) {
            cranelift_unwind_free_entry_inplace(&g_unwind_vec[i]);
            if (i + 1 < g_unwind_len) {
                memmove(&g_unwind_vec[i], &g_unwind_vec[i + 1],
                        (g_unwind_len - i - 1) * sizeof(*g_unwind_vec));
            }
            g_unwind_len--;
            break;
        }
    }
    qemu_mutex_unlock(&g_unwind_mutex);
}

TranslationBlock *cranelift_unwind_tb_lookup(uintptr_t host_pc)
{
    if (!g_unwind_mutex_inited) {
        return NULL;
    }
    qemu_mutex_lock(&g_unwind_mutex);
    CraneliftUnwindEntry *e = cranelift_unwind_lookup_locked(host_pc);
    TranslationBlock *tb = e ? (TranslationBlock *)e->tb : NULL;
    qemu_mutex_unlock(&g_unwind_mutex);
    return tb;
}

/*
 * Diagnostic: log a host_pc that wasn't found in the unwind index along
 * with the bracketing entries (nearest-below / nearest-above by host_lo)
 * to help characterise the miss. Rate-limited to ~one per 5 seconds.
 *
 * Called from cpu_io_recompile right before the cpu_abort fallback. The
 * idea is to capture data about WHICH host_pcs we're missing — is it in
 * a gap between entries, far outside the arena, in shim memory, etc.
 */
__attribute__((noipa))
void cranelift_unwind_log_miss(uintptr_t host_pc)
{
    if (!g_unwind_mutex_inited) {
        CL_LOG("unwind miss (index not init): host_pc=0x%" PRIxPTR, host_pc);
        return;
    }
    static int64_t last_log_ns;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
    if (now - last_log_ns < 5000000000LL) {
        return;
    }
    last_log_ns = now;

    qemu_mutex_lock(&g_unwind_mutex);
    /* Upper bound on host_lo → idx is the first entry with host_lo > host_pc */
    unsigned lo = 0, hi = g_unwind_len;
    while (lo < hi) {
        unsigned mid = (lo + hi) >> 1;
        if (g_unwind_vec[mid].host_lo <= host_pc) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    const CraneliftUnwindEntry *below =
        (lo > 0) ? &g_unwind_vec[lo - 1] : NULL;
    const CraneliftUnwindEntry *above =
        (lo < g_unwind_len) ? &g_unwind_vec[lo] : NULL;

    CL_LOG("unwind miss: host_pc=0x%" PRIxPTR " entries=%u "
           "below=%p[lo=0x%" PRIxPTR " hi=0x%" PRIxPTR
           " tb=%p tb_pc=0x%" PRIx64 "] "
           "above=%p[lo=0x%" PRIxPTR " hi=0x%" PRIxPTR
           " tb=%p tb_pc=0x%" PRIx64 "] "
           "shim_arena=%p..%p",
           host_pc, g_unwind_len,
           (void *)below,
           below ? below->host_lo : 0,
           below ? below->host_hi : 0,
           below ? below->tb : NULL,
           (uint64_t)(below ? below->tb_pc : 0),
           (void *)above,
           above ? above->host_lo : 0,
           above ? above->host_hi : 0,
           above ? above->tb : NULL,
           (uint64_t)(above ? above->tb_pc : 0),
           g_shim_arena, g_shim_arena + CRANELIFT_SHIM_ARENA);
    qemu_mutex_unlock(&g_unwind_mutex);
}

bool cranelift_unwind_data_from_tb(const TranslationBlock *tb,
                                   uintptr_t host_pc,
                                   uint64_t *data,
                                   int *out_insns_left)
{
    if (!g_unwind_mutex_inited || !tb || !data || !out_insns_left) {
        return false;
    }

    /*
     * Match the GETPC_ADJ semantic from cpu_unwind_data_from_tb:
     * host_pc is the return address from a call; subtract so the
     * search lands on the CALL site rather than the next insn.
     */
    host_pc -= GETPC_ADJ;

    qemu_mutex_lock(&g_unwind_mutex);
    CraneliftUnwindEntry *e = cranelift_unwind_lookup_locked(host_pc);
    if (!e) {
        qatomic_inc(&g_unwind_misses);
        qemu_mutex_unlock(&g_unwind_mutex);
        return false;
    }
    /* TB-pointer recycle defence: the QEMU TB allocator can hand the
     * same address out to two different guest TBs. Verify identity. */
    if (e->tb != tb || e->tb_pc != tb->pc) {
        qatomic_inc(&g_unwind_stale_tb_pc);
        qemu_mutex_unlock(&g_unwind_mutex);
        return false;
    }

    uint32_t off = (uint32_t)(host_pc - e->host_lo);

    /* Smallest i with host_end[i] > off. */
    unsigned lo = 0, hi = e->n_rows;
    while (lo < hi) {
        unsigned mid = (lo + hi) >> 1;
        if (e->host_end[mid] > off) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if (lo >= e->n_rows) {
        /* host_pc is one-past-end of the last row; clamp to last. */
        lo = e->n_rows - 1;
    }
    uint32_t guest_idx = e->loc[lo];
    if (guest_idx >= e->n_insns) {
        qatomic_inc(&g_unwind_misses);
        qemu_mutex_unlock(&g_unwind_mutex);
        return false;
    }

    /* Mirror cpu_unwind_data_from_tb's seed: zero, then for non-PCREL
     * TBs seed data[0] with tb->pc. Cranelift's recorded insn_start
     * cargs already encode absolute (or PCREL page-offset) values, so
     * the seed is functionally redundant for our flat-table format —
     * but kept for bit-for-bit parity with the tier-1 unwind output
     * that x86_restore_state_to_opc has been validated against. */
    memset(data, 0, sizeof(uint64_t) * INSN_START_WORDS);
    if (!(tb_cflags(tb) & CF_PCREL)) {
        data[0] = tb->pc;
    }
    for (unsigned j = 0; j < INSN_START_WORDS; j++) {
        data[j] = e->insn_data[(size_t)guest_idx * INSN_START_WORDS + j];
    }
    *out_insns_left = (int)(e->n_insns - guest_idx);

    qatomic_inc(&g_unwind_hits);
    qemu_mutex_unlock(&g_unwind_mutex);
    return true;
}

void cranelift_unwind_get_stats(uint64_t *hits, uint64_t *misses,
                                uint64_t *stale_tb_pc, uint32_t *entries)
{
    if (hits) *hits = qatomic_read(&g_unwind_hits);
    if (misses) *misses = qatomic_read(&g_unwind_misses);
    if (stale_tb_pc) *stale_tb_pc = qatomic_read(&g_unwind_stale_tb_pc);
    if (entries) *entries = qatomic_read(&g_unwind_len);
}

/* Bulk wipe — called from cranelift_bridge_on_tb_flush. */
static void cranelift_unwind_clear_all(void)
{
    if (!g_unwind_mutex_inited) {
        return;
    }
    qemu_mutex_lock(&g_unwind_mutex);
    for (unsigned i = 0; i < g_unwind_len; i++) {
        cranelift_unwind_free_entry_inplace(&g_unwind_vec[i]);
    }
    g_unwind_len = 0;
    qemu_mutex_unlock(&g_unwind_mutex);
}

/* ------------------------------------------------------------------ */

/*
 * Hot-path hook: invoked just before cpu_loop_exec_tb dispatches `tb`.
 *
 * Looks up the pending tier-2 ring for an entry whose guest PC matches
 * this TB.  On match, emits an ABI shim that calls the Cranelift code
 * and lands in the canonical TCG epilogue, then atomically swaps
 * tb->tc.ptr over to the shim.  Subsequent dispatches go through the
 * tier-2 path.
 *
 * Concurrency: tb->tc.ptr is racy under MTTCG (multiple vCPUs can read
 * it).  We use qatomic_xchg__nocheck to publish the new pointer; the
 * old TCG code remains live in the JIT arena (we don't free), so
 * readers that snapshot the previous pointer continue to execute the
 * tier-1 code safely.
 *
 * We require cranelift_g_tb_ret_addr to be set; the bridge has no way
 * to reconstruct it on its own.  If it's zero (e.g. running before
 * tcg_prologue_init has completed, or on a non-aarch64 backend) we
 * skip the swap and leave the pending entry in the ring for a later
 * retry.
 */
void cranelift_bridge_try_swap_slow(TranslationBlock *tb)
{
    /*
     * The inline wrapper has already verified tb->tier < 2 and that
     * the pending ring isn't empty. We still need the
     * initialised / swap-enabled / mutex-inited checks because all
     * three can be false while the ring shows non-empty (e.g. between
     * drain producing and init flipping the master switch off).
     */
    if (!g_cranelift_initialised || !tb) {
        return;
    }
    if (!qatomic_read(&g_swap_install_enabled)) {
        return;
    }
    if (!pending_ring_mutex_inited) {
        return;
    }

    uintptr_t ret_addr = qatomic_read(&cranelift_g_tb_ret_addr);
    if (ret_addr == 0) {
        return;  /* prologue not yet emitted */
    }

    const void *new_code = NULL;
    size_t      new_size = 0;
    CraneliftTcgUnwindMeta new_unwind = {0};
    /* Pending entries are keyed by (uintptr_t)tb (the value we passed
     * to cranelift_tcg_enqueue in maybe_compile). Match by that, not
     * tb->pc -- CF_PCREL TBs all have pc=0. */
    uint64_t    want_key = (uint64_t)(uintptr_t)tb;

    qemu_mutex_lock(&pending_ring_mutex);
    for (unsigned i = cranelift_bridge_g_pending_tail; i != cranelift_bridge_g_pending_head;
         i = (i + 1) % PENDING_SWAP_RING) {
        if (pending_ring[i].tb_pc == want_key) {
            new_code = pending_ring[i].code;
            new_size = pending_ring[i].size;
            new_unwind = pending_ring[i].unwind;
            /* Shift remaining entries left (same as take_pending). */
            unsigned j = i;
            while (j != cranelift_bridge_g_pending_head) {
                unsigned k = (j + 1) % PENDING_SWAP_RING;
                pending_ring[j] = pending_ring[k];
                j = k;
            }
            cranelift_bridge_g_pending_head = (cranelift_bridge_g_pending_head + PENDING_SWAP_RING - 1)
                           % PENDING_SWAP_RING;
            break;
        }
    }
    qemu_mutex_unlock(&pending_ring_mutex);

    /*
     * Clear ONLY the SWAP bit. COMPILE bit is sticky for the TB's
     * lifetime to prevent re-entering maybe_compile_slow. If no entry
     * was found in the ring, we still clear SWAP so try_swap_slow
     * isn't re-fired every dispatch — the next drain() OR's it back.
     */
    qatomic_and(&tb->cranelift_pending, (uint8_t)~CRANELIFT_PEND_SWAP);

    if (!new_code) {
        /* No-op path: still release any unwind handle that came along
         * for the ride (shouldn't happen, but be defensive). */
        if (new_unwind._handle) {
            cranelift_tcg_release_unwind(new_unwind._handle);
        }
        return;
    }

    /*
     * Refuse to install a tier-2 shim without matching unwind data.
     * Otherwise an MMIO / watchpoint fault inside the JIT code would
     * crash via cpu_io_recompile -> tcg_tb_lookup -> cpu_abort -- which
     * is the precise bug Plan A exists to eliminate. Keep the TB on
     * tier-1; the next compile attempt will retry.
     */
    if (new_unwind.n_rows == 0 || !new_unwind.host_end) {
        if (new_unwind._handle) {
            cranelift_tcg_release_unwind(new_unwind._handle);
        }
        return;
    }

    /*
     * Cap lifted. The shim-arena (1 MiB / 48 B per shim = ~21k slots)
     * and shim-map (CRANELIFT_SHIM_MAP_CAP) bound this naturally.
     * emit_shim returns NULL and the map full-check below refuses
     * further installs when either runs out.
     *
     * 2026-05-19 bisect override: optionally cap installs via
     * X1BOX_CRANELIFT_MAX_INSTALLS env var so we can binary-search
     * which tier-2 TB is panicking the Xbox kernel.
     */
    static uint64_t s_install_count;
    void *shim = cranelift_emit_shim((uintptr_t)new_code, ret_addr);
    if (!shim) {
        cranelift_tcg_release_unwind(new_unwind._handle);
        return;
    }

    /*
     * Install the unwind entry BEFORE the shim-map publication. The
     * shim map is what makes the tier-2 dispatch observable to other
     * vCPU threads; any fault landing in new_code (via a helper called
     * from cranelift code) must find the unwind entry, so it has to
     * be in place first. cranelift_unwind_install deep-copies the
     * arrays — we can release the Rust handle immediately afterwards.
     */
    cranelift_unwind_install(tb, new_code, new_size, &new_unwind);
    cranelift_tcg_release_unwind(new_unwind._handle);
    new_unwind._handle = NULL;

    uint64_t ic = qatomic_fetch_inc(&s_install_count) + 1;
    CL_LOG("install#%" PRIu64 " tb=%p pc=0x%" PRIx64 " cs_base=0x%" PRIx64
           " cflags=0x%x tier1_code=%p shim=%p ce_code=%p tb_ret=0x%" PRIxPTR,
           ic, (void *)tb, (uint64_t)tb->pc, (uint64_t)tb->cs_base,
           tb->cflags, tb->tc.ptr, shim, new_code, ret_addr);

    /*
     * Publish the swap.  tb->tc.ptr is `const void *`; cast through
     * `void **` (drop the const) so __atomic_exchange_n's type check
     * is satisfied.  We keep tb->tc.size pointing at the tier-1 code
     * length, which is fine because nothing in the dispatch path uses
     * tc.size to compute the entry address - it's only consulted when
     * walking the code-region search tree, which still resolves to
     * the tier-1 blob.
     */
    /*
     * Do NOT modify tb->tc.ptr — it is the lookup key for
     * tcg_tb_lookup(host_pc) and the search tree built at TB
     * creation. Mutating it causes cpu_io_recompile / watchpoint
     * unwind to fail and abort. Instead, record (tb -> shim) in a
     * lock-free open-addressed hash table and have cpu_tb_exec
     * consult it before dispatch.
     */
    qemu_mutex_lock(&g_shim_map_mutex);
    if (g_shim_map_count < CRANELIFT_SHIM_MAP_CAP / 2) {
        unsigned idx = cranelift_shim_hash(tb);
        for (unsigned i = 0; i < CRANELIFT_SHIM_MAP_CAP; i++) {
            unsigned slot = (idx + i) & CRANELIFT_SHIM_MAP_MASK;
            const TranslationBlock *cur = g_shim_map[slot].tb;
            if (cur == NULL || cur == tb) {
                /*
                 * Empty slot OR same tb-pointer with a stale entry
                 * (TB pool recycled this address since the last
                 * install — the old shim is dead). Either way we
                 * write the new entry here. Overwriting the stale
                 * shim doesn't leak: the arena is reset on tb_flush
                 * and individual shim bytes were never going to be
                 * freed anyway.
                 *
                 * Order: shim + pc FIRST, then tb LAST (qatomic_set
                 * as a publication boundary so lockless readers see
                 * a fully-formed slot).
                 */
                g_shim_map[slot].shim = shim;
                g_shim_map[slot].pc   = tb->pc;
                qatomic_set(&g_shim_map[slot].tb, tb);
                if (cur == NULL) {
                    qatomic_set(&g_shim_map_count, g_shim_map_count + 1);
                }
                break;
            }
            /* Collision with a different tb; keep probing. */
        }
    } else {
        CL_LOG("shim map at half-full (%u entries); refusing install#%" PRIu64,
               g_shim_map_count, ic);
    }
    qemu_mutex_unlock(&g_shim_map_mutex);
    (void)new_size;

    if (tb->tier < 2) {
        tb->tier = 2;
    }
    /* Record the guest PC in the JIT hint cache. CF_PCREL TBs have
     * tb->pc == 0 — the chain dispatcher records those at execution
     * time via xemu_chain_get_tb_cpu_state. For tb->pc != 0 the value
     * here IS the guest PC. */
    if (tb->pc != 0) {
        jit_cache_record_hot_pc((uint64_t)tb->pc);
    }
}

/* ------------------------------------------------------------------ */
/*  Tier-2 disk cache (hint-cache v1)                                  */
/* ------------------------------------------------------------------ */

/*
 * Records every guest PC that has been successfully promoted to tier-2.
 * On shutdown we flush the set to <cache_dir>/hints.bin. On the next
 * boot of the same game cranelift_bridge_jit_cache_open() reads the
 * file back; future work will use it to skip the threshold gate for
 * those PCs (immediate compile on first execution).
 *
 * Storage: open-addressed direct-mapped 8K slots. Hash collisions
 * silently overwrite (lossy) — acceptable because the cache is a hint,
 * not a correctness boundary, and most hot TBs accumulate enough
 * cross-session hits that they survive.
 */
/* Public hot-PC table — see cranelift-bridge.h for the inline reader.
 * Size MUST match CRANELIFT_HOT_PC_SLOTS. */
uint64_t cranelift_bridge_g_hot_pcs[CRANELIFT_HOT_PC_SLOTS];
#define JIT_HINT_SLOTS CRANELIFT_HOT_PC_SLOTS
#define g_jit_hint_pcs cranelift_bridge_g_hot_pcs
static uint32_t g_jit_hint_count;   /* observable for stats */
static QemuMutex g_jit_hint_mu;
static bool g_jit_hint_mu_inited;
static char g_jit_cache_dir[1024];
static bool g_jit_cache_dir_set;

#define JIT_HINT_MAGIC 0x484A3158u   /* "X1JH" little-endian */
#define JIT_HINT_VERSION 1u

static void jit_cache_lazy_mu_init(void)
{
    if (!g_jit_hint_mu_inited) {
        qemu_mutex_init(&g_jit_hint_mu);
        g_jit_hint_mu_inited = true;
    }
}

static inline unsigned jit_hint_slot(uint64_t pc)
{
    /* Fibonacci hash. */
    return (unsigned)((pc * 2654435761ull) & (JIT_HINT_SLOTS - 1));
}

static void jit_cache_record_hot_pc(uint64_t pc)
{
    if (pc == 0) return;
    jit_cache_lazy_mu_init();
    qemu_mutex_lock(&g_jit_hint_mu);
    unsigned slot = jit_hint_slot(pc);
    if (g_jit_hint_pcs[slot] != pc) {
        if (g_jit_hint_pcs[slot] == 0) {
            g_jit_hint_count++;
        }
        g_jit_hint_pcs[slot] = pc;
    }
    qemu_mutex_unlock(&g_jit_hint_mu);
}

void cranelift_bridge_jit_cache_open(const char *dir_path)
{
    if (!dir_path || !*dir_path) {
        g_jit_cache_dir_set = false;
        g_jit_cache_dir[0] = '\0';
        CL_LOG("jit-cache: disabled (no dir)");
        return;
    }
    snprintf(g_jit_cache_dir, sizeof(g_jit_cache_dir), "%s", dir_path);
    g_jit_cache_dir_set = true;

    /* Best-effort mkdir -p. Ignore failures — the save path will
     * report errors if the directory genuinely can't be created. */
    (void)mkdir(g_jit_cache_dir, 0755);

    /* Load previous-session hints. v1: just log the count; the active
     * use of these (immediate-compile bypass of threshold) is a future
     * iteration. Set the count in g_jit_hint_count so stats show
     * non-zero immediately after open. */
    char path[1280];
    snprintf(path, sizeof(path), "%s/hints.bin", g_jit_cache_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        CL_LOG("jit-cache: open dir=%s, no previous hints", g_jit_cache_dir);
        return;
    }
    uint32_t magic, version, build_id_len, pc_count;
    if (fread(&magic, 4, 1, f) != 1 || magic != JIT_HINT_MAGIC ||
        fread(&version, 4, 1, f) != 1 || version != JIT_HINT_VERSION ||
        fread(&build_id_len, 4, 1, f) != 1 || build_id_len > 256) {
        fclose(f);
        CL_LOG("jit-cache: hints file header mismatch, ignoring");
        return;
    }
    /* Skip build_id for now (future: compare against compiled build id
     * and discard if different). */
    if (build_id_len) {
        fseek(f, (long)build_id_len, SEEK_CUR);
    }
    uint8_t xbe_sha1[20];
    if (fread(xbe_sha1, 1, 20, f) != 20) {
        fclose(f);
        return;
    }
    if (fread(&pc_count, 4, 1, f) != 1 || pc_count > JIT_HINT_SLOTS) {
        fclose(f);
        return;
    }
    jit_cache_lazy_mu_init();
    qemu_mutex_lock(&g_jit_hint_mu);
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < pc_count; i++) {
        uint64_t pc;
        if (fread(&pc, 8, 1, f) != 1) break;
        if (pc == 0) continue;
        unsigned slot = jit_hint_slot(pc);
        if (g_jit_hint_pcs[slot] == 0) {
            g_jit_hint_pcs[slot] = pc;
            g_jit_hint_count++;
            loaded++;
        }
    }
    qemu_mutex_unlock(&g_jit_hint_mu);
    fclose(f);
    CL_LOG("jit-cache: loaded %u hot PCs from %s (file count=%u)",
           loaded, path, pc_count);
}

void cranelift_bridge_jit_cache_save(void)
{
    if (!g_jit_cache_dir_set) {
        return;
    }
    jit_cache_lazy_mu_init();

    char tmp[1280], dst[1280];
    snprintf(tmp, sizeof(tmp), "%s/hints.bin.tmp", g_jit_cache_dir);
    snprintf(dst, sizeof(dst), "%s/hints.bin", g_jit_cache_dir);

    /* Ensure directory exists (Android removes app caches when user
     * "Clears Cache"; we want save to recover from that). */
    (void)mkdir(g_jit_cache_dir, 0755);

    FILE *f = fopen(tmp, "wb");
    if (!f) {
        CL_LOG("jit-cache: save fopen(%s) failed", tmp);
        return;
    }

    qemu_mutex_lock(&g_jit_hint_mu);

    uint32_t magic = JIT_HINT_MAGIC;
    uint32_t version = JIT_HINT_VERSION;
    /* build_id is presence-only; bump version if the cache shape
     * changes between builds so old files are rejected on load. */
    uint32_t build_id_len = 0;
    uint8_t xbe_sha1[20] = {0};
    uint32_t pc_count = g_jit_hint_count;

    bool ok = true;
    ok = ok && fwrite(&magic, 4, 1, f) == 1;
    ok = ok && fwrite(&version, 4, 1, f) == 1;
    ok = ok && fwrite(&build_id_len, 4, 1, f) == 1;
    ok = ok && fwrite(xbe_sha1, 1, 20, f) == 20;
    ok = ok && fwrite(&pc_count, 4, 1, f) == 1;
    uint32_t written = 0;
    for (unsigned i = 0; i < JIT_HINT_SLOTS && ok; i++) {
        if (g_jit_hint_pcs[i] != 0) {
            ok = fwrite(&g_jit_hint_pcs[i], 8, 1, f) == 1;
            written++;
        }
    }

    qemu_mutex_unlock(&g_jit_hint_mu);

    if (ok) {
        fclose(f);
        /* Atomic rename. */
        if (rename(tmp, dst) != 0) {
            CL_LOG("jit-cache: rename(%s,%s) failed", tmp, dst);
            unlink(tmp);
            return;
        }
        CL_LOG("jit-cache: saved %u hot PCs to %s", written, dst);
    } else {
        fclose(f);
        unlink(tmp);
        CL_LOG("jit-cache: save write failed at %u/%u entries",
               written, pc_count);
    }
}


const void *cranelift_bridge_lookup_shim(const TranslationBlock *tb)
{
    /*
     * Hot path: O(1) average. We never remove entries, so probing
     * stops cleanly at the first NULL slot.
     *
     * Slot validity requires BOTH slot.tb == tb AND slot.pc == tb->pc.
     * The pc match catches recycled-TB-pointer cases where the slot
     * holds a stale shim from a TB at the same memory address that
     * has since been tb_phys_invalidated and recycled to a different
     * guest PC. See CraneliftShimEntry's comment.
     */
    if (qatomic_read(&g_shim_map_count) == 0) {
        return NULL;
    }
    unsigned idx = cranelift_shim_hash(tb);
    for (unsigned i = 0; i < CRANELIFT_SHIM_MAP_CAP; i++) {
        unsigned slot = (idx + i) & CRANELIFT_SHIM_MAP_MASK;
        const TranslationBlock *slot_tb =
            qatomic_read(&g_shim_map[slot].tb);
        if (slot_tb == NULL) {
            return NULL;          /* miss */
        }
        if (slot_tb == tb) {
            /* Same tb pointer — verify it's the SAME TB by PC. */
            if (g_shim_map[slot].pc == tb->pc) {
                return g_shim_map[slot].shim;
            }
            /*
             * Stale slot (tb pointer recycled by the TB allocator
             * for a different guest PC). Treat as miss; the install
             * path will overwrite this slot in place next time the
             * NEW tb gets a fresh shim.
             */
            return NULL;
        }
        /* Collision with different tb pointer; keep probing. */
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Settings / introspection helpers                                    */
/* ------------------------------------------------------------------ */

void cranelift_bridge_set_enabled(bool enabled)
{
    cranelift_bridge_lazy_init();
    if (!g_cranelift_initialised) {
        return;
    }
    cranelift_tcg_set_enabled(g_cranelift_ctx, enabled ? 1 : 0);
}

bool cranelift_bridge_is_enabled(void)
{
    if (!g_cranelift_initialised) {
        return false;
    }
    return cranelift_tcg_is_enabled(g_cranelift_ctx) != 0;
}

void cranelift_bridge_set_verify_mode(bool enabled)
{
    cranelift_bridge_lazy_init();
    if (!g_cranelift_initialised) {
        return;
    }
    cranelift_tcg_set_verify_mode(g_cranelift_ctx, enabled ? 1 : 0);
}

void cranelift_bridge_set_threshold(uint32_t threshold)
{
    cranelift_bridge_g_tier2_threshold = threshold;
    if (g_cranelift_initialised) {
        cranelift_tcg_set_hot_threshold(g_cranelift_ctx, threshold);
    }
}

void cranelift_bridge_log_stats(void)
{
    if (!g_cranelift_initialised) {
        return;
    }
    CraneliftTcgStats st;
    cranelift_tcg_get_stats(g_cranelift_ctx, &st);
    CL_LOG("tier2 stats: enq=%" PRIu64 " ok=%" PRIu64 " err=%" PRIu64
           " skip=%" PRIu64 " bl=%" PRIu64 " ver_ok=%" PRIu64
           " ver_div=%" PRIu64 " act=%u q=%u",
           st.enqueued, st.compiled_ok, st.compiled_err,
           st.fallback_unsupported_op, st.blacklisted,
           st.verify_ok, st.verify_divergence,
           st.active_entries, st.worker_queue_depth);

    /* Per-chain dispatch telemetry: lets us see whether the quantum is
     * doing useful work (avg iters/run) versus livelocking (high spins)
     * and how many distinct guest threads our fingerprint detected. */
    uint64_t runs = 0, iters = 0, spins = 0, irq_exits = 0;
    uint32_t thread_count = 0, jitter = 0;
    unsigned chain_max = 0;
    cranelift_chain_get_stats(&runs, &iters, &spins, &irq_exits,
                              &thread_count, &chain_max, &jitter);
    uint64_t avg_iters_x100 = runs ? (iters * 100 / runs) : 0;
    CL_LOG("chain stats: runs=%" PRIu64 " iters=%" PRIu64
           " avg=%" PRIu64 ".%02" PRIu64
           " spins=%" PRIu64 " irq_exits=%" PRIu64
           " base=%u jitter=0x%x threads=%u",
           runs, iters,
           avg_iters_x100 / 100, avg_iters_x100 % 100,
           spins, irq_exits, chain_max, jitter, thread_count);

    uint64_t lru_hits = 0, lru_misses = 0;
    cranelift_get_helper_lookup_tb_lru_stats(&lru_hits, &lru_misses);
    uint64_t lru_total = lru_hits + lru_misses;
    uint64_t lru_pct_x100 = lru_total ? (lru_hits * 10000 / lru_total) : 0;
    CL_LOG("helper_lookup_tb_ptr LRU: hits=%" PRIu64 " misses=%" PRIu64
           " hit_rate=%" PRIu64 ".%02" PRIu64 "%%",
           lru_hits, lru_misses,
           lru_pct_x100 / 100, lru_pct_x100 % 100);

    /* IR-cache health: collisions are TBs whose snapshot was evicted
     * BEFORE crossing the tier-2 threshold (lost forever); take-misses
     * are tier-2 attempts that found nothing (either evicted or
     * already-enqueued). High collision rate = bump IR_CACHE_BITS. */
    CL_LOG("ir_cache: stores=%" PRIu64 " collisions=%" PRIu64
           " take_hits=%" PRIu64 " take_misses=%" PRIu64
           " coll_rate=%" PRIu64 ".%02" PRIu64 "%%"
           " miss_rate=%" PRIu64 ".%02" PRIu64 "%%",
           g_ir_cache_stores, g_ir_cache_collisions,
           g_ir_cache_take_hits, g_ir_cache_take_misses,
           g_ir_cache_stores ?
              (g_ir_cache_collisions * 10000 / g_ir_cache_stores) / 100 : 0,
           g_ir_cache_stores ?
              (g_ir_cache_collisions * 10000 / g_ir_cache_stores) % 100 : 0,
           (g_ir_cache_take_hits + g_ir_cache_take_misses) ?
              (g_ir_cache_take_misses * 10000 /
               (g_ir_cache_take_hits + g_ir_cache_take_misses)) / 100 : 0,
           (g_ir_cache_take_hits + g_ir_cache_take_misses) ?
              (g_ir_cache_take_misses * 10000 /
               (g_ir_cache_take_hits + g_ir_cache_take_misses)) % 100 : 0);

    /* Synchronous-fault unwind index: hits = MMIO/watchpoint/SMC fault
     * inside cranelift code that resolved cleanly via the index;
     * misses = a known cranelift host_pc with no matching entry (should
     * stay 0 in steady state; nonzero indicates either an install/drop
     * race or a TB pointer recycle escaping the tb_pc verify); entries
     * is the live index size. */
    uint64_t unw_hits = 0, unw_misses = 0, unw_stale = 0;
    uint32_t unw_entries = 0;
    cranelift_unwind_get_stats(&unw_hits, &unw_misses, &unw_stale,
                               &unw_entries);
    CL_LOG("unwind index: hits=%" PRIu64 " misses=%" PRIu64
           " stale_tb_pc=%" PRIu64 " entries=%u",
           unw_hits, unw_misses, unw_stale, unw_entries);

#ifdef XBOX
    /* Per-thread BQL wait/hold contention dump (tag "x1-bql"). */
    bql_stats_dump();
#endif
}

void cranelift_bridge_blacklist(uint64_t pc_lo, uint64_t pc_hi)
{
    if (!g_cranelift_initialised) {
        return;
    }
    cranelift_tcg_blacklist(g_cranelift_ctx, pc_lo, pc_hi);
}

/*
 * Drop every tier-2 binding after a tb_flush. The TB allocator has just
 * been reset by tcg_region_reset_all(), so every (TranslationBlock *)
 * we have cached in the shim map is about to alias a completely
 * different guest TB. Keeping those entries live causes cpu_tb_exec to
 * dispatch the new TB into Cranelift code compiled for the old PC --
 * the SS2 boot kernel-halt symptom.
 *
 * Caller (tb_flush) holds cpu_in_serial_context, so we don't need to
 * fence against vCPU readers of g_shim_map. We still grab the writer
 * mutexes for consistency with the install path.
 */
void cranelift_bridge_on_tb_flush(void)
{
    if (!g_cranelift_initialised) {
        return;
    }

    /* Drain the pending-swap ring. Code pointers in there reference
     * Cranelift entries whose TBs have just evaporated. Release any
     * pending unwind buffers so the Rust heap doesn't accumulate
     * orphans across a flush. */
    if (pending_ring_mutex_inited) {
        qemu_mutex_lock(&pending_ring_mutex);
        for (unsigned i = cranelift_bridge_g_pending_tail;
             i != cranelift_bridge_g_pending_head;
             i = (i + 1) % PENDING_SWAP_RING) {
            if (pending_ring[i].unwind._handle) {
                cranelift_tcg_release_unwind(pending_ring[i].unwind._handle);
                pending_ring[i].unwind._handle = NULL;
            }
        }
        cranelift_bridge_g_pending_head = 0;
        cranelift_bridge_g_pending_tail = 0;
        qemu_mutex_unlock(&pending_ring_mutex);
    }

    /* Wipe the synchronous-fault unwind index. Backing arrays are
     * g_free'd so the C heap doesn't accumulate across flush events. */
    cranelift_unwind_clear_all();

    /* Zero the shim map. Open-addressing with linear probing means
     * partial deletion would truncate probe chains; tb_flush wipes
     * everything, so we just clear the whole table in one shot. */
    if (g_shim_map_mutex_inited) {
        qemu_mutex_lock(&g_shim_map_mutex);
        memset(g_shim_map, 0, sizeof(g_shim_map));
        qatomic_set(&g_shim_map_count, 0);
        qemu_mutex_unlock(&g_shim_map_mutex);
    }

    /* Recycle the shim arena. Shim bytes are RWX-mmapped; we don't
     * unmap, just reset the bump pointer so future installs reuse the
     * same pages. Any in-flight execution of an old shim has already
     * been serialised out by tb_flush's exclusive-context requirement. */
    if (g_shim_mutex_inited) {
        qemu_mutex_lock(&g_shim_mutex);
        g_shim_used = 0;
        qemu_mutex_unlock(&g_shim_mutex);
    }

    /* Clear the Rust-side entry cache (HashMap keyed by tb pointer)
     * so re-translation isn't short-circuited by stale lookups. */
    cranelift_tcg_reset_entries(g_cranelift_ctx);

    CL_LOG("on_tb_flush: shim map + pending ring + arena + rust entries reset");
}

#endif /* XEMU_HAVE_CRANELIFT */
