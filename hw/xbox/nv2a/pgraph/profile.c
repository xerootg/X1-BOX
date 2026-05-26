/*
 * QEMU Geforce NV2A profiling helpers
 *
 * Copyright (c) 2020-2024 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "hw/xbox/nv2a/nv2a_int.h"
#include "hw/xbox/xpacks.h"
#ifdef XBOX
#include "qemu/burst_diag.h"
#endif
#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef XBOX
extern uint64_t tb_cache_stats_lookup_hits;
extern uint64_t tb_cache_stats_lookup_misses;
/* Per-frame stall detector — counters from cranelift_bridge_log_stats world. */
extern void cranelift_chain_get_stats(uint64_t *runs, uint64_t *iters,
                                       uint64_t *spins, uint64_t *irq_exits,
                                       uint32_t *thread_count,
                                       unsigned *chain_max, uint32_t *jitter);
extern void cranelift_get_helper_lookup_tb_lru_stats(uint64_t *hits,
                                                      uint64_t *misses);
extern void cranelift_get_helper_lookup_slot_hist(uint64_t out[8]);
extern void cranelift_get_helper_phys_pc_hint_stats(uint64_t *hits,
                                                     uint64_t *misses);
extern void cranelift_get_tb_jc_stats(uint64_t *hits, uint64_t *misses);
extern void cranelift_get_tb_pool_stats(uint64_t *gen, uint64_t *inval,
                                         uint64_t *flush);
extern void cranelift_get_vcpu_fpcr_observed(uint64_t *fpcr);
extern void cranelift_get_helper_dispatch_shim_stats(uint64_t *hits,
                                                      uint64_t *misses);
extern uint32_t cranelift_bridge_g_helper_route_shim;
extern void pfifo_get_ring_stats(uint64_t *samples, uint64_t *depth_sum,
                                  uint32_t *depth_max, uint64_t *empty,
                                  uint64_t *full, uint64_t *pusher_calls,
                                  uint64_t *words, uint32_t *dma_len,
                                  uint64_t *wait_spin, uint64_t *wait_idle,
                                  uint64_t *wait_ns);
extern void xbox_hle_get_idle_stats(uint64_t *idle_iters, uint64_t *yields,
                                     uint64_t *yield_ns_total);
#endif

NV2AStats g_nv2a_stats;

void nv2a_profile_increment(void)
{
#ifdef XBOX
    burst_diag_on_flip_stall();
#endif
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    const int64_t fps_update_interval = 250000;
    g_nv2a_stats.last_flip_time = now;

    static int64_t frame_count = 0;
    frame_count++;

    static int64_t ts = 0;
    int64_t delta = now - ts;
    if (delta >= fps_update_interval) {
        g_nv2a_stats.increment_fps = frame_count * 1000000 / delta;
        ts = now;
        frame_count = 0;
    }

#if defined(XBOX) && defined(__ANDROID__)
    /*
     * Per-frame stall detector.
     *
     * Snapshot interesting counters at every FLIP_STALL (guest frame
     * boundary). If the current interval is much longer than the
     * recent median (>2× of a 16-frame moving baseline), log what
     * changed during that frame. Helps localise which subsystem
     * caused a "long frame" without needing a profile-grade trace.
     *
     * Cost: a few qatomic_reads per FLIP_STALL (~30/sec) — negligible.
     */
    static int64_t last_flip_us;
    static int64_t baseline_us[16];
    static unsigned baseline_idx;
    static uint64_t last_chain_runs, last_chain_iters, last_irq_exits;
    static uint64_t last_lru_hits, last_lru_misses;
    static uint64_t last_phint_hits, last_phint_misses;
    static int64_t last_long_log_us;

    int64_t interval = last_flip_us ? (now - last_flip_us) : 0;
    last_flip_us = now;

    if (interval > 0) {
        /* Update rolling-median baseline (simple ring; medianish via
         * mean since spike detection doesn't need true median). */
        baseline_us[baseline_idx++ & 15] = interval;
        int64_t base_sum = 0;
        unsigned base_n = 0;
        for (unsigned i = 0; i < 16; i++) {
            if (baseline_us[i] > 0) {
                base_sum += baseline_us[i];
                base_n++;
            }
        }
        int64_t base_avg = base_n ? base_sum / base_n : 0;

        /* Snapshot deltas regardless — used in log if we trigger. */
        uint64_t chain_runs = 0, chain_iters = 0, irq_exits = 0;
        cranelift_chain_get_stats(&chain_runs, &chain_iters, NULL,
                                  &irq_exits, NULL, NULL, NULL);
        uint64_t lru_hits = 0, lru_misses = 0;
        cranelift_get_helper_lookup_tb_lru_stats(&lru_hits, &lru_misses);
        uint64_t phint_hits = 0, phint_misses = 0;
        cranelift_get_helper_phys_pc_hint_stats(&phint_hits, &phint_misses);

        uint64_t d_runs   = chain_runs   - last_chain_runs;
        uint64_t d_iters  = chain_iters  - last_chain_iters;
        uint64_t d_irq    = irq_exits    - last_irq_exits;
        uint64_t d_lhit   = lru_hits     - last_lru_hits;
        uint64_t d_lmiss  = lru_misses   - last_lru_misses;
        uint64_t d_phit   = phint_hits   - last_phint_hits;
        uint64_t d_pmiss  = phint_misses - last_phint_misses;

        last_chain_runs   = chain_runs;
        last_chain_iters  = chain_iters;
        last_irq_exits    = irq_exits;
        last_lru_hits     = lru_hits;
        last_lru_misses   = lru_misses;
        last_phint_hits   = phint_hits;
        last_phint_misses = phint_misses;

        /*
         * Periodic JIT stats: emit a structured `xemu-jit:` line every
         * ~2 seconds regardless of spikes. The long_frame line above
         * only fires on outliers; for steady-state diagnosis (TB churn,
         * jmp_cache hit rate, helper LRU + phint hit rate, chain depth)
         * we need a regular pulse so the MCP `jit_stats` tool always has
         * a fresh sample to read. 2 s smooths Halo 2's 15-20 Hz flip
         * cadence (~30-40 flips per sample) without burying logcat.
         *
         * d_jchit / d_jcmiss come from the per-CPU tb_jmp_cache; gen /
         * inval / flush come from the global TB-pool counters. Same
         * delta basis as the spike log so we can compare apples-to-
         * apples.
         */
        static int64_t last_jit_log_us;
        static uint64_t last_jc_hits, last_jc_misses;
        static uint64_t last_tb_gen, last_tb_inval, last_tb_flush;
        if (now - last_jit_log_us >= 2000000) {
            uint64_t jc_hits = 0, jc_misses = 0;
            cranelift_get_tb_jc_stats(&jc_hits, &jc_misses);
            uint64_t tb_gen = 0, tb_inval = 0, tb_flush = 0;
            cranelift_get_tb_pool_stats(&tb_gen, &tb_inval, &tb_flush);

            int64_t window_us = last_jit_log_us ? (now - last_jit_log_us) : 1;
            uint64_t d_jchit  = jc_hits   - last_jc_hits;
            uint64_t d_jcmiss = jc_misses - last_jc_misses;
            uint64_t d_gen    = tb_gen    - last_tb_gen;
            uint64_t d_inval  = tb_inval  - last_tb_inval;
            uint64_t d_flush  = tb_flush  - last_tb_flush;

            /* Re-read chain/lru/phint snapshots so they're delta'd over
             * the SAME 2-second window. The long_frame path above bumps
             * its "last_*" trackers on every flip, so we keep a
             * separate cadence here. */
            uint64_t cr = 0, ci = 0, ie = 0;
            cranelift_chain_get_stats(&cr, &ci, NULL, &ie, NULL, NULL, NULL);
            uint64_t lh = 0, lm = 0;
            cranelift_get_helper_lookup_tb_lru_stats(&lh, &lm);
            uint64_t ph = 0, pm = 0;
            cranelift_get_helper_phys_pc_hint_stats(&ph, &pm);

            static uint64_t last_cr, last_ci, last_ie;
            static uint64_t last_lh, last_lm, last_ph, last_pm;
            uint64_t d_cr = cr - last_cr, d_ci = ci - last_ci;
            uint64_t d_lh = lh - last_lh, d_lm = lm - last_lm;
            uint64_t d_ph = ph - last_ph, d_pm = pm - last_pm;

            uint64_t lru_total   = d_lh + d_lm;
            uint64_t phint_total = d_ph + d_pm;
            uint64_t jc_total    = d_jchit + d_jcmiss;
            unsigned lru_hit_pct   = lru_total   ? (unsigned)(d_lh * 100 / lru_total)   : 0;
            unsigned phint_hit_pct = phint_total ? (unsigned)(d_ph * 100 / phint_total) : 0;
            unsigned jc_hit_pct    = jc_total    ? (unsigned)(d_jchit * 100 / jc_total) : 0;
            uint64_t avg_chain_x100 = d_cr ? (d_ci * 100 / d_cr) : 0;

            uint64_t fpcr = 0;
            cranelift_get_vcpu_fpcr_observed(&fpcr);
            unsigned fz   = (unsigned)((fpcr >> 24) & 1);
            unsigned fz16 = (unsigned)((fpcr >> 19) & 1);
            unsigned dn   = (unsigned)((fpcr >> 25) & 1);

            uint64_t rh = 0, rm = 0;
            cranelift_get_helper_dispatch_shim_stats(&rh, &rm);
            static uint64_t last_rh, last_rm;
            uint64_t d_rh    = rh - last_rh;
            uint64_t d_rm    = rm - last_rm;
            last_rh = rh;
            last_rm = rm;
            uint64_t route_total = d_rh + d_rm;
            unsigned route_hit_pct = route_total
                ? (unsigned)(d_rh * 100 / route_total) : 0;
            unsigned route_on = qatomic_read(&cranelift_bridge_g_helper_route_shim);

            /*
             * Per-slot LRU hit histogram (Phase 1+2 SoA+NEON rewrite).
             * Emitted as `lru_slot=A/B/C/D/E/F/G/H` where A is slot 0
             * fast-path hits (should dominate), and B..H are SIMD-found
             * cold-scan promotions. A healthy distribution falls off
             * monotonically; any non-monotonic spike indicates the LRU
             * sizing is wrong for the current workload.
             */
            uint64_t slot_hist[8] = {0};
            cranelift_get_helper_lookup_slot_hist(slot_hist);
            static uint64_t last_slot_hist[8];
            uint64_t d_slot[8];
            for (unsigned _i = 0; _i < 8; _i++) {
                d_slot[_i] = slot_hist[_i] - last_slot_hist[_i];
                last_slot_hist[_i] = slot_hist[_i];
            }

            __android_log_print(ANDROID_LOG_INFO, "xemu-jit",
                "win=%lld_ms "
                "tb_gen/s=%llu tb_inval/s=%llu tb_flush/s=%llu "
                "jc_hit%%=%u jc_miss/s=%llu "
                "lru_hit%%=%u lru_call/s=%llu "
                "lru_slot=%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu "
                "phint_hit%%=%u phint_call/s=%llu "
                "chain_runs/s=%llu chain_avg=%llu.%02llu "
                "irq/s=%llu "
                "fpcr=0x%llx FZ=%u FZ16=%u DN=%u "
                "route_shim=%u route_hit%%=%u route_call/s=%llu",
                (long long)(window_us / 1000),
                (unsigned long long)(d_gen   * 1000000 / window_us),
                (unsigned long long)(d_inval * 1000000 / window_us),
                (unsigned long long)(d_flush * 1000000 / window_us),
                jc_hit_pct,
                (unsigned long long)(d_jcmiss * 1000000 / window_us),
                lru_hit_pct,
                (unsigned long long)(lru_total * 1000000 / window_us),
                (unsigned long long)d_slot[0], (unsigned long long)d_slot[1],
                (unsigned long long)d_slot[2], (unsigned long long)d_slot[3],
                (unsigned long long)d_slot[4], (unsigned long long)d_slot[5],
                (unsigned long long)d_slot[6], (unsigned long long)d_slot[7],
                phint_hit_pct,
                (unsigned long long)(phint_total * 1000000 / window_us),
                (unsigned long long)(d_cr * 1000000 / window_us),
                (unsigned long long)(avg_chain_x100 / 100),
                (unsigned long long)(avg_chain_x100 % 100),
                (unsigned long long)((ie - last_ie) * 1000000 / window_us),
                (unsigned long long)fpcr, fz, fz16, dn,
                route_on, route_hit_pct,
                (unsigned long long)(route_total * 1000000 / window_us));

            last_jit_log_us  = now;
            last_jc_hits     = jc_hits;
            last_jc_misses   = jc_misses;
            last_tb_gen      = tb_gen;
            last_tb_inval    = tb_inval;
            last_tb_flush    = tb_flush;
            last_cr          = cr;
            last_ci          = ci;
            last_ie          = ie;
            last_lh          = lh;
            last_lm          = lm;
            last_ph          = ph;
            last_pm          = pm;
        }

        /*
         * Periodic PIPELINE stats: PGRAPH ring fill-level + pfifo
         * wait reasons + vCPU HLE-nanosleep wall-time. Diagnoses
         * who-is-blocking-whom in the vCPU → pgraph-ring → pfifo →
         * vulkan render chain. Same 2-second cadence as xemu-jit so
         * the MCP `pipeline_stats` tool can correlate.
         *
         * Diagnostic key:
         *   ring_full% high   → pfifo can't drain fast enough; pfifo
         *                       is the bottleneck; CPU optimisations
         *                       on vCPU won't help.
         *   ring_empty% high  → vCPU produces commands slowly; vCPU
         *                       is the producer constraint; pfifo
         *                       sits waiting; CPU optimisations help.
         *   ring_full + ring_empty both low → balanced pipeline; the
         *                       bottleneck is elsewhere (audio sync,
         *                       wall-clock pacing, etc.).
         *   wait_idle/s high  → pfifo blocks on cond_wait often; vCPU
         *                       is feeding it bursty or sparse work.
         *   wait_spin/s high  → pfifo waking from short spin; vCPU is
         *                       producing in fast bursts.
         *   hle_yield_ms/s    → vCPU wall-time spent in our explicit
         *                       KiIdleLoop nanosleep. Subtract from
         *                       overall Sleep% to find "other" sleep
         *                       (BQL wait, futex on other devices).
         */
        static int64_t last_pipe_log_us;
        static uint64_t last_pfifo_samples, last_pfifo_depth_sum;
        static uint64_t last_pfifo_empty,   last_pfifo_full;
        static uint64_t last_pfifo_pcalls,  last_pfifo_words;
        static uint64_t last_pfifo_wspin,   last_pfifo_widle;
        static uint64_t last_pfifo_wait_ns;
        static uint64_t last_hle_idle, last_hle_yields, last_hle_yield_ns;
        static uint64_t last_uni_hits, last_uni_misses;
        static uint64_t last_mfh, last_mc;
        /* Phase 1.2 (index rewrite LRU) + 1.3 (surface part skip) */
        static uint64_t last_idx_hits, last_idx_misses, last_idx_evicts;
        static uint64_t last_surf_skip, last_surf_calls;
        if (now - last_pipe_log_us >= 2000000) {
            uint64_t ps = 0, pd_sum = 0, pe = 0, pf = 0;
            uint64_t pc = 0, pw = 0;
            uint32_t pmax = 0, pdma_len = 0;
            uint64_t pws = 0, pwi = 0, pwn = 0;
            pfifo_get_ring_stats(&ps, &pd_sum, &pmax, &pe, &pf, &pc,
                                 &pw, &pdma_len, &pws, &pwi, &pwn);

            uint64_t hi = 0, hy = 0, hyn = 0;
            xbox_hle_get_idle_stats(&hi, &hy, &hyn);

            int64_t pipe_window_us = last_pipe_log_us
                ? (now - last_pipe_log_us) : 1;

            uint64_t d_samples = ps  - last_pfifo_samples;
            uint64_t d_dsum    = pd_sum - last_pfifo_depth_sum;
            uint64_t d_empty   = pe  - last_pfifo_empty;
            uint64_t d_full    = pf  - last_pfifo_full;
            uint64_t d_pcalls  = pc  - last_pfifo_pcalls;
            uint64_t d_words   = pw  - last_pfifo_words;
            uint64_t d_wspin   = pws - last_pfifo_wspin;
            uint64_t d_widle   = pwi - last_pfifo_widle;
            uint64_t d_wait_ns = pwn - last_pfifo_wait_ns;

            uint64_t d_hyields = hy - last_hle_yields;
            uint64_t d_hidle   = hi - last_hle_idle;
            uint64_t d_hyn     = hyn - last_hle_yield_ns;

            uint64_t uh = g_nv2a_stats.shader_stats.uniform_fast_skip_hits;
            uint64_t um = g_nv2a_stats.shader_stats.uniform_fast_skip_misses;
            uint64_t d_uh = uh - last_uni_hits;
            uint64_t d_um = um - last_uni_misses;
            uint64_t d_un_total = d_uh + d_um;
            unsigned uni_hit_pct = d_un_total
                ? (unsigned)(d_uh * 100 / d_un_total) : 0;

            /* Phase 1.2: index-rewrite LRU cache */
            uint64_t ih = g_nv2a_stats.shader_stats.prim_rewrite_cache_hits;
            uint64_t im = g_nv2a_stats.shader_stats.prim_rewrite_cache_misses;
            uint64_t iv = g_nv2a_stats.shader_stats.prim_rewrite_cache_evicts;
            uint64_t d_ih = ih - last_idx_hits;
            uint64_t d_im = im - last_idx_misses;
            uint64_t d_iv = iv - last_idx_evicts;
            uint64_t d_idx_total = d_ih + d_im;
            unsigned idx_hit_pct = d_idx_total
                ? (unsigned)(d_ih * 100 / d_idx_total) : 0;

            /* Phase 1.3: update_surface_part short-circuit.
             * "calls" is the total entry count; "skip" is the fraction
             * that returned via the cache early-exit without rebuilding
             * the binding target or walking the full dirty bitmap. */
            uint64_t sc = g_nv2a_stats.surf_working.update_calls;
            uint64_t ss = g_nv2a_stats.surf_working.update_skip;
            uint64_t d_sc = sc - last_surf_calls;
            uint64_t d_ss = ss - last_surf_skip;
            unsigned surf_skip_pct = d_sc
                ? (unsigned)(d_ss * 100 / d_sc) : 0;

            unsigned ring_empty_pct = d_samples
                ? (unsigned)(d_empty * 100 / d_samples) : 0;
            unsigned ring_full_pct  = d_samples
                ? (unsigned)(d_full  * 100 / d_samples) : 0;
            uint64_t ring_avg = d_samples ? (d_dsum / d_samples) : 0;
            /*
             * pct_x100 = (wait_ns / total_ns) * 10000
             *          = wait_ns * 10000 / (window_us * 1000)
             *          = wait_ns * 10 / window_us
             *
             * Worst-case overflow: wait_ns ≈ 2e9 (2 sec window),
             * × 10 = 2e10, well below u64 max (1.8e19). Safe.
             */
            uint64_t pfifo_wait_pct_x100 = pipe_window_us
                ? (d_wait_ns * 10) / pipe_window_us : 0;
            uint64_t hle_yield_pct_x100 = pipe_window_us
                ? (d_hyn * 10) / pipe_window_us : 0;

            /* method_fast_hit accumulates per try_fast call by num_proc
             * (count of methods consumed). method_count is per method
             * header group. ratio = fast_hit/(fast_hit + slow_methods).
             * If the vertex fast-path is engaging, fast_hit jumps. */
            uint64_t mfh = g_nv2a_stats.cpu_working.method_fast_hit;
            uint64_t mc  = g_nv2a_stats.cpu_working.method_count;
            uint64_t d_mfh = mfh - last_mfh;
            uint64_t d_mc  = mc  - last_mc;

            __android_log_print(ANDROID_LOG_INFO, "xemu-pipe",
                "win=%lld_ms "
                "ring avg=%llu max=%u dma_len=%u empty%%=%u full%%=%u "
                "pusher_calls/s=%llu words/s=%llu "
                "pfifo wait_spin/s=%llu wait_idle/s=%llu wait%%=%llu.%02llu "
                "vcpu hle_yield/s=%llu hle_iters/s=%llu hle_yield_ms/s=%llu "
                "hle_yield%%=%llu.%02llu "
                "uni_skip_hit%%=%u uni_skip_total/s=%llu "
                "idx_cache_hit%%=%u idx_cache_total/s=%llu idx_cache_evicts/s=%llu "
                "surf_skip%%=%u surf_total/s=%llu "
                "mfh/s=%llu mc/s=%llu",
                (long long)(pipe_window_us / 1000),
                (unsigned long long)ring_avg,
                pmax, pdma_len,
                ring_empty_pct, ring_full_pct,
                (unsigned long long)(d_pcalls * 1000000 / pipe_window_us),
                (unsigned long long)(d_words  * 1000000 / pipe_window_us),
                (unsigned long long)(d_wspin  * 1000000 / pipe_window_us),
                (unsigned long long)(d_widle  * 1000000 / pipe_window_us),
                (unsigned long long)(pfifo_wait_pct_x100 / 100),
                (unsigned long long)(pfifo_wait_pct_x100 % 100),
                (unsigned long long)(d_hyields * 1000000 / pipe_window_us),
                (unsigned long long)(d_hidle   * 1000000 / pipe_window_us),
                (unsigned long long)(d_hyn / 1000000),  /* ms/s */
                (unsigned long long)(hle_yield_pct_x100 / 100),
                (unsigned long long)(hle_yield_pct_x100 % 100),
                uni_hit_pct,
                (unsigned long long)(d_un_total * 1000000 / pipe_window_us),
                idx_hit_pct,
                (unsigned long long)(d_idx_total * 1000000 / pipe_window_us),
                (unsigned long long)(d_iv * 1000000 / pipe_window_us),
                surf_skip_pct,
                (unsigned long long)(d_sc * 1000000 / pipe_window_us),
                (unsigned long long)(d_mfh * 1000000 / pipe_window_us),
                (unsigned long long)(d_mc  * 1000000 / pipe_window_us));

            last_pipe_log_us       = now;
            last_pfifo_samples     = ps;
            last_pfifo_depth_sum   = pd_sum;
            last_pfifo_empty       = pe;
            last_pfifo_full        = pf;
            last_pfifo_pcalls      = pc;
            last_pfifo_words       = pw;
            last_pfifo_wspin       = pws;
            last_pfifo_widle       = pwi;
            last_pfifo_wait_ns     = pwn;
            last_hle_idle          = hi;
            last_hle_yields        = hy;
            last_hle_yield_ns      = hyn;
            last_uni_hits          = uh;
            last_uni_misses        = um;
            last_idx_hits          = ih;
            last_idx_misses        = im;
            last_idx_evicts        = iv;
            last_surf_skip         = ss;
            last_surf_calls        = sc;
            last_mfh               = mfh;
            last_mc                = mc;
        }

        /*
         * Phase 2.1 + 2.2: per-NV2A-method-class histogram and per-draw
         * phase p99 + draws_per_submit. Emitted as the "xemu-method"
         * logcat tag on the same 2 s cadence as xemu-pipe so the MCP
         * tool can correlate.
         *
         * Method class indices must match METHOD_CLASS_* enum order in
         * hw/xbox/nv2a/pgraph/pgraph.c.
         */
        static int64_t last_method_log_us;
        static uint64_t last_mc_count[NV2A_METHOD_CLASS_COUNT];
        static uint64_t last_mc_cycles[NV2A_METHOD_CLASS_COUNT];
        if (now - last_method_log_us >= 2000000) {
            int64_t method_window_us = last_method_log_us
                ? (now - last_method_log_us) : 1;

            uint64_t d_count[NV2A_METHOD_CLASS_COUNT];
            uint64_t d_cycles[NV2A_METHOD_CLASS_COUNT];
            uint64_t total_cycles = 0;
            for (unsigned i = 0; i < NV2A_METHOD_CLASS_COUNT; i++) {
                uint64_t c = g_nv2a_stats.method_class_stats.count[i];
                uint64_t cy = g_nv2a_stats.method_class_stats.cycles[i];
                d_count[i]  = c  - last_mc_count[i];
                d_cycles[i] = cy - last_mc_cycles[i];
                total_cycles += d_cycles[i];
                last_mc_count[i]  = c;
                last_mc_cycles[i] = cy;
            }

            /* count/s by class, scaled to 1 s. */
            uint64_t r_vertex  = d_count[0] * 1000000 / method_window_us;
            uint64_t r_tex     = d_count[1] * 1000000 / method_window_us;
            uint64_t r_shader  = d_count[2] * 1000000 / method_window_us;
            uint64_t r_light   = d_count[3] * 1000000 / method_window_us;
            uint64_t r_render  = d_count[4] * 1000000 / method_window_us;
            uint64_t r_inline  = d_count[5] * 1000000 / method_window_us;
            uint64_t r_other   = d_count[6] * 1000000 / method_window_us;

            /* Cycle share % per class (integer percent, 0-100). */
            unsigned s_vertex = total_cycles
                ? (unsigned)(d_cycles[0] * 100 / total_cycles) : 0;
            unsigned s_tex    = total_cycles
                ? (unsigned)(d_cycles[1] * 100 / total_cycles) : 0;
            unsigned s_shader = total_cycles
                ? (unsigned)(d_cycles[2] * 100 / total_cycles) : 0;
            unsigned s_light  = total_cycles
                ? (unsigned)(d_cycles[3] * 100 / total_cycles) : 0;
            unsigned s_render = total_cycles
                ? (unsigned)(d_cycles[4] * 100 / total_cycles) : 0;
            unsigned s_inline = total_cycles
                ? (unsigned)(d_cycles[5] * 100 / total_cycles) : 0;
            unsigned s_other  = total_cycles
                ? (unsigned)(d_cycles[6] * 100 / total_cycles) : 0;

            FramePhaseTimingStats *ph = &g_nv2a_stats.phase;
            /* draws_per_submit is smoothed in the snapshot path. */
            unsigned dps_int   = (unsigned)ph->draws_per_submit;
            unsigned dps_frac  =
                (unsigned)((ph->draws_per_submit - (float)dps_int) * 100.0f);

            __android_log_print(ANDROID_LOG_INFO, "xemu-method",
                "win=%lld_ms "
                "vertex/s=%llu(%u%%) tex_state/s=%llu(%u%%) "
                "shader_state/s=%llu(%u%%) light/s=%llu(%u%%) "
                "render_state/s=%llu(%u%%) inline_draw/s=%llu(%u%%) "
                "other/s=%llu(%u%%) | draws_per_submit=%u.%02u | "
                "draw_vk_cmd_p99_us=%.2f draw_setup_p99_us=%.2f "
                "draw_vtx_attr_p99_us=%.2f draw_vtx_sync_p99_us=%.2f "
                "draw_prim_rw_p99_us=%.2f "
                "pipe_bind_tex_p99_us=%.2f pipe_bind_shd_p99_us=%.2f "
                "pipe_lookup_p99_us=%.2f",
                (long long)(method_window_us / 1000),
                (unsigned long long)r_vertex, s_vertex,
                (unsigned long long)r_tex,    s_tex,
                (unsigned long long)r_shader, s_shader,
                (unsigned long long)r_light,  s_light,
                (unsigned long long)r_render, s_render,
                (unsigned long long)r_inline, s_inline,
                (unsigned long long)r_other,  s_other,
                dps_int, dps_frac,
                ph->draw_vk_cmd_max_us,
                ph->draw_setup_max_us,
                ph->draw_vtx_attr_max_us,
                ph->draw_vtx_sync_max_us,
                ph->draw_prim_rw_max_us,
                ph->pipe_bind_tex_max_us,
                ph->pipe_bind_shd_max_us,
                ph->pipe_lookup_max_us);

            last_method_log_us = now;
        }

        /* Trigger: this frame > 2× baseline AND >50ms long.
         * Throttle to one log per 1s so we don't flood. */
        if (base_avg > 0 && interval > 2 * base_avg && interval > 50000 &&
            (now - last_long_log_us) > 1000000) {
            uint64_t avg_chain = d_runs ? (d_iters * 100 / d_runs) : 0;
            __android_log_print(ANDROID_LOG_INFO, "x1-stall",
                "long_frame: interval=%lld ms baseline=%lld ms "
                "chain_runs=%llu chain_iters=%llu avg=%llu.%02llu "
                "irq_exits=%llu lru_hits=%llu lru_misses=%llu "
                "phint_hits=%llu phint_misses=%llu",
                (long long)(interval / 1000),
                (long long)(base_avg / 1000),
                (unsigned long long)d_runs,
                (unsigned long long)d_iters,
                (unsigned long long)(avg_chain / 100),
                (unsigned long long)(avg_chain % 100),
                (unsigned long long)d_irq,
                (unsigned long long)d_lhit,
                (unsigned long long)d_lmiss,
                (unsigned long long)d_phit,
                (unsigned long long)d_pmiss);
            last_long_log_us = now;
        }
    }
#endif
}

static void snapshot_phase_timing(void)
{
    FramePhaseTimingWork *w = &g_nv2a_stats.phase_working;
    FramePhaseTimingStats *p = &g_nv2a_stats.phase;
    const float alpha = 0.2f;

#define SMOOTH(field) \
    p->field##_ms = p->field##_ms * (1.0f - alpha) + \
                    (float)(w->field##_ns) / 1e6f * alpha

    SMOOTH(surface_update);
    SMOOTH(texture_upload);
    SMOOTH(shader_compile);
    SMOOTH(draw_dispatch);
    SMOOTH(finish);
    SMOOTH(flip_idle);
    SMOOTH(fifo_idle);
    SMOOTH(fifo_idle_frame);
    SMOOTH(fifo_idle_starve);
    SMOOTH(draw_vtx_attr);
    SMOOTH(draw_vtx_sync);
    SMOOTH(draw_prim_rw);
    SMOOTH(draw_pipeline);
    SMOOTH(draw_desc_set);
    SMOOTH(draw_setup);
    SMOOTH(draw_vk_cmd);
    SMOOTH(pipe_bind_tex);
    SMOOTH(pipe_bind_shd);
    SMOOTH(pipe_lookup);
    SMOOTH(finish_fence);
    SMOOTH(finish_submit);
    SMOOTH(gpu_total);
    SMOOTH(gpu_render);
    SMOOTH(gpu_nonrender);
#undef SMOOTH

    /* Phase 2.2: per-window max (p99 proxy). Take the raw _max_ns,
     * convert to microseconds, and smooth so a single noisy outlier
     * doesn't dominate the displayed value. */
#define SMOOTH_MAX_US(field) \
    p->field##_max_us = p->field##_max_us * (1.0f - alpha) + \
                        (float)(w->field##_max_ns) / 1e3f * alpha
    SMOOTH_MAX_US(draw_vtx_attr);
    SMOOTH_MAX_US(draw_vtx_sync);
    SMOOTH_MAX_US(draw_prim_rw);
    SMOOTH_MAX_US(draw_setup);
    SMOOTH_MAX_US(draw_vk_cmd);
    SMOOTH_MAX_US(pipe_bind_tex);
    SMOOTH_MAX_US(pipe_bind_shd);
    SMOOTH_MAX_US(pipe_lookup);
#undef SMOOTH_MAX_US

#define SMOOTH_CNT(dst, src) \
    (dst) = (dst) * (1.0f - alpha) + (float)(src) * alpha
    SMOOTH_CNT(p->gpu_rp_count, w->gpu_rp_count);
    /* Phase 2.2: draws_per_submit ratio. submits_this_window is bumped
     * once per pgraph_vk_finish; draws_this_window is bumped per
     * vkCmdDraw* call. Ratio reflects how many draws each submit
     * batched together — high values motivate Phase 3 split. */
    if (w->submits_this_window > 0) {
        float dps =
            (float)w->draws_this_window / (float)w->submits_this_window;
        SMOOTH_CNT(p->draws_per_submit, dps);
    }
#undef SMOOTH_CNT

    p->total_ms = p->surface_update_ms + p->texture_upload_ms +
                  p->shader_compile_ms + p->draw_dispatch_ms +
                  p->finish_ms + p->flip_idle_ms + p->fifo_idle_ms;

    bool saved_post_flip = w->post_flip;
    memset(w, 0, sizeof(*w));
    w->post_flip = saved_post_flip;
}

static void snapshot_cpu_timing(void)
{
    CpuTimingWork *w = &g_nv2a_stats.cpu_working;
    CpuTimingStats *p = &g_nv2a_stats.cpu;
    const float alpha = 0.2f;

#define SMOOTH_MS(dst, src_ns) \
    (dst) = (dst) * (1.0f - alpha) + (float)(src_ns) / 1e6f * alpha

    SMOOTH_MS(p->lock_wait_ms, w->lock_wait_ns);
    SMOOTH_MS(p->pusher_run_ms, w->pusher_run_ns);
    SMOOTH_MS(p->method_exec_ms, w->method_exec_ns);
    SMOOTH_MS(p->puller_total_ms, w->puller_total_ns);
    SMOOTH_MS(p->puller_lock_ms, w->puller_lock_ns);
    SMOOTH_MS(p->puller_method_ms, w->puller_method_ns);

#undef SMOOTH_MS

#define SMOOTH_CNT(dst, src) \
    (dst) = (dst) * (1.0f - alpha) + (float)(src) * alpha

    SMOOTH_CNT(p->kick_count, w->kick_count);
    SMOOTH_CNT(p->kick_count_spun, w->kick_count_spun);
    SMOOTH_CNT(p->kick_count_idle, w->kick_count_idle);
    SMOOTH_CNT(p->pusher_words, w->pusher_words);
    SMOOTH_CNT(p->method_count, w->method_count);
    SMOOTH_CNT(p->method_fast_hit, w->method_fast_hit);
    SMOOTH_CNT(p->method_noninc_words, w->method_noninc_words);

#undef SMOOTH_CNT

#ifdef XBOX
    {
        uint64_t cur_hits   = tb_cache_stats_lookup_hits;
        uint64_t cur_misses = tb_cache_stats_lookup_misses;
        uint64_t dh = cur_hits   - w->tb_hits_snap;
        uint64_t dm = cur_misses - w->tb_misses_snap;
        float frame_pct = (dh + dm) > 0
                          ? (float)dh / (float)(dh + dm) * 100.0f
                          : 100.0f;
        p->tb_hit_pct = p->tb_hit_pct * (1.0f - alpha) + frame_pct * alpha;
        w->tb_hits_snap   = cur_hits;
        w->tb_misses_snap = cur_misses;
    }
#endif

    w->lock_wait_ns      = 0;
    w->pusher_run_ns     = 0;
    w->method_exec_ns    = 0;
    w->puller_total_ns   = 0;
    w->puller_lock_ns    = 0;
    w->puller_method_ns  = 0;
    w->kick_count        = 0;
    w->kick_count_spun   = 0;
    w->kick_count_idle   = 0;
    w->pusher_words      = 0;
    w->method_count      = 0;
    w->method_fast_hit   = 0;
    w->method_noninc_words = 0;
}

static void snapshot_vsync_timing(void)
{
    VsyncTimingWork *w = &g_nv2a_stats.vsync_working;
    VsyncTimingStats *p = &g_nv2a_stats.vsync;
    const float alpha = 0.2f;

#define SMOOTH_CNT(dst, src) \
    (dst) = (dst) * (1.0f - alpha) + (float)(src) * alpha

    SMOOTH_CNT(p->calls, w->calls);
    SMOOTH_CNT(p->reqs, w->reqs);
    SMOOTH_CNT(p->merged, w->merged);
    SMOOTH_CNT(p->dirty_count, w->dirty_count);
    SMOOTH_CNT(p->bytes_kb, w->bytes_copied / 1024.0f);

#undef SMOOTH_CNT

    memset(w, 0, sizeof(*w));
}

static void snapshot_surf_timing(void)
{
    SurfTimingWork *w = &g_nv2a_stats.surf_working;
    SurfTimingStats *p = &g_nv2a_stats.surf;
    const float alpha = 0.2f;

#define SMOOTH_MS(dst, src_ns) \
    (dst) = (dst) * (1.0f - alpha) + ((float)(src_ns) / 1e6f) * alpha
#define SMOOTH_CNT(dst, src) \
    (dst) = (dst) * (1.0f - alpha) + (float)(src) * alpha

    SMOOTH_MS(p->populate_ms, w->populate_ns);
    SMOOTH_MS(p->dirty_ms, w->dirty_ns);
    SMOOTH_MS(p->enrp_ms, w->enrp_ns);
    SMOOTH_MS(p->lk_hit_ms, w->lk_hit_ns);
    SMOOTH_MS(p->lk_evict_ms, w->lk_evict_ns);
    SMOOTH_MS(p->lk_nosurf_ms, w->lk_nosurf_ns);
    SMOOTH_MS(p->create_ms, w->create_ns);
    SMOOTH_MS(p->put_ms, w->put_ns);
    SMOOTH_MS(p->bind_ms, w->bind_ns);
    SMOOTH_MS(p->upload_ms, w->upload_ns);
    SMOOTH_MS(p->download_ms, w->download_ns);
    SMOOTH_MS(p->expire_ms, w->expire_ns);
    SMOOTH_MS(p->df_flush_ms, w->df_flush_ns);
    SMOOTH_MS(p->df_read_ms, w->df_read_ns);
    SMOOTH_CNT(p->update_calls, w->update_calls);
    SMOOTH_CNT(p->create_count, w->create_count);
    SMOOTH_CNT(p->hit_count, w->hit_count);
    SMOOTH_CNT(p->evict_count, w->evict_count);
    SMOOTH_CNT(p->upload_count, w->upload_count);
    SMOOTH_CNT(p->download_count, w->download_count);
    SMOOTH_CNT(p->miss_count, w->miss_count);

#undef SMOOTH_MS
#undef SMOOTH_CNT

    memset(w, 0, sizeof(*w));
}

void nv2a_profile_flip_stall(void)
{
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int64_t render_time = (now-g_nv2a_stats.last_flip_time)/1000;

    g_nv2a_stats.frame_working.mspf = render_time;
    g_nv2a_stats.frame_history[g_nv2a_stats.frame_ptr] =
        g_nv2a_stats.frame_working;
    g_nv2a_stats.frame_ptr =
        (g_nv2a_stats.frame_ptr + 1) % NV2A_PROF_NUM_FRAMES;
    g_nv2a_stats.frame_count++;
    memset(&g_nv2a_stats.frame_working, 0, sizeof(g_nv2a_stats.frame_working));

    /* xpacks: re-scan guest memory for any pending pattern_bytes patches.
     * Cheap when nothing is pending; throttled internally. */
    xpacks_tick();

    snapshot_phase_timing();
    snapshot_cpu_timing();
    snapshot_vsync_timing();
    snapshot_surf_timing();

    g_nv2a_stats.phase_working.post_flip = true;

    /* Track game frame time (flip-to-flip interval) */
    static int64_t prev_flip_us;
    if (prev_flip_us) {
        float frame_ms = (float)(now - prev_flip_us) / 1000.0f;
        FramePacingStats *p = &g_nv2a_stats.pacing;
        p->game_frame_ms = p->game_frame_ms * 0.8f + frame_ms * 0.2f;
        if (frame_ms < p->game_frame_min_ms || p->game_frame_min_ms == 0)
            p->game_frame_min_ms = frame_ms;
        if (frame_ms > p->game_frame_max_ms)
            p->game_frame_max_ms = frame_ms;
    }
    prev_flip_us = now;

    /* Drive xpack pattern-scan; throttled internally to ~1Hz. */
    xpacks_tick();

#if defined(__ANDROID__) && NV2A_PERF_LOG
    if ((g_nv2a_stats.frame_count % 60) == 0) {
        char buf[512];
        nv2a_profile_get_phase_timing_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "hakuX-phase", "%s", buf);
        nv2a_profile_get_cpu_timing_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "hakuX-cpu", "%s", buf);
        nv2a_profile_get_vsync_timing_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "xemu-vsync", "%s", buf);
        nv2a_profile_get_surf_timing_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "xemu-surf", "%s", buf);
        nv2a_profile_get_pacing_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "xemu-pace", "%s", buf);
        nv2a_profile_get_workload_str(buf, sizeof(buf));
        __android_log_print(ANDROID_LOG_INFO, "xemu-work", "%s", buf);

        {
            FramePhaseTimingStats *ph = &g_nv2a_stats.phase;
            __android_log_print(ANDROID_LOG_INFO, "xemu-gpu",
                "GPU: Tot:%.1f Rnd:%.1f Xfr:%.1f RP:%.0f",
                ph->gpu_total_ms,
                ph->gpu_render_ms,
                ph->gpu_nonrender_ms,
                ph->gpu_rp_count);
        }
    }
#endif
}

void nv2a_profile_get_pacing_str(char *buf, int bufsize)
{
    FramePacingStats *p = &g_nv2a_stats.pacing;
    snprintf(buf, bufsize,
             "G:%.1f(%.1f-%.1f) D:%.1f(%.1f-%.1f) S:%.1f J:%.1f Df:%u Vd:%.1f Ul:%c",
             p->game_frame_ms,
             p->game_frame_min_ms,
             p->game_frame_max_ms,
             p->display_frame_ms,
             p->display_frame_min_ms,
             p->display_frame_max_ms,
             p->swap_ms,
             p->vblank_jitter_ms,
             p->defers_total,
             p->vblank_delivery_ms,
             p->unlock_mode_active ? 'Y' : 'N');
    /* Reset min/max every call so the window reflects recent behavior */
    p->game_frame_min_ms = 0;
    p->game_frame_max_ms = 0;
    p->display_frame_min_ms = 0;
    p->display_frame_max_ms = 0;
    p->defers_total = 0;
}

void nv2a_profile_get_phase_timing_str(char *buf, int bufsize)
{
    FramePhaseTimingStats *p = &g_nv2a_stats.phase;
    snprintf(buf, bufsize,
             "Surf:%.1f Tex:%.1f Shd:%.1f Draw:%.1f "
             "[Vtx:%.1f Syn:%.1f Prw:%.1f Pipe:%.1f(Tx:%.1f Sh:%.1f Lu:%.1f) "
             "Desc:%.1f Setup:%.1f Cmd:%.1f] "
             "Fin:%.1f(Sub:%.1f Fen:%.1f) Flip:%.1f Idle:%.1f(Fr:%.1f St:%.1f) | Tot:%.1f ms",
             p->surface_update_ms,
             p->texture_upload_ms,
             p->shader_compile_ms,
             p->draw_dispatch_ms,
             p->draw_vtx_attr_ms,
             p->draw_vtx_sync_ms,
             p->draw_prim_rw_ms,
             p->draw_pipeline_ms,
             p->pipe_bind_tex_ms,
             p->pipe_bind_shd_ms,
             p->pipe_lookup_ms,
             p->draw_desc_set_ms,
             p->draw_setup_ms,
             p->draw_vk_cmd_ms,
             p->finish_ms,
             p->finish_submit_ms,
             p->finish_fence_ms,
             p->flip_idle_ms,
             p->fifo_idle_ms,
             p->fifo_idle_frame_ms,
             p->fifo_idle_starve_ms,
             p->total_ms);
}

void nv2a_profile_get_cpu_timing_str(char *buf, int bufsize)
{
    CpuTimingStats *p = &g_nv2a_stats.cpu;
    float spin_pct = p->kick_count > 0
                     ? p->kick_count_spun / p->kick_count * 100.0f
                     : 0.0f;
    float idle_pct = p->kick_count > 0
                     ? p->kick_count_idle / p->kick_count * 100.0f
                     : 0.0f;
    snprintf(buf, bufsize,
             "CPU: K:%.0f W:%.1fK M:%.0f(Fh:%.0f Ni:%.0f) "
             "Lock:%.1fms Push:%.1fms "
             "SpH:%.0f%% IdS:%.0f%% TbH:%.1f%%",
             p->kick_count,
             p->pusher_words / 1000.0f,
             p->method_count,
             p->method_fast_hit,
             p->method_noninc_words,
             p->lock_wait_ms,
             p->pusher_run_ms,
             spin_pct,
             idle_pct,
             p->tb_hit_pct);
}

void nv2a_profile_get_vsync_timing_str(char *buf, int bufsize)
{
    VsyncTimingStats *p = &g_nv2a_stats.vsync;
    snprintf(buf, bufsize,
             "Vsyn: C:%.0f R:%.0f M:%.0f D:%.0f %.0fKB",
             p->calls, p->reqs, p->merged, p->dirty_count, p->bytes_kb);
}

void nv2a_profile_get_surf_timing_str(char *buf, int bufsize)
{
    SurfTimingStats *p = &g_nv2a_stats.surf;
    snprintf(buf, bufsize,
             "Srf: C:%.0f pop:%.1f drty:%.1f enrp:%.1f "
             "lkH:%.1f lkE:%.1f lkN:%.1f "
             "cr:%.1f put:%.1f bnd:%.1f upl:%.1f dl:%.1f exp:%.1f "
             "dfF:%.1f dfR:%.1f "
             "| #cr:%.0f #hit:%.0f #ev:%.0f #upl:%.0f #dl:%.0f #miss:%.0f",
             p->update_calls,
             p->populate_ms, p->dirty_ms, p->enrp_ms,
             p->lk_hit_ms, p->lk_evict_ms, p->lk_nosurf_ms,
             p->create_ms, p->put_ms, p->bind_ms,
             p->upload_ms, p->download_ms, p->expire_ms,
             p->df_flush_ms, p->df_read_ms,
             p->create_count, p->hit_count, p->evict_count,
             p->upload_count, p->download_count, p->miss_count);
}

void nv2a_profile_get_shader_stats_str(char *buf, int bufsize)
{
    ShaderPipelineStats *s = &g_nv2a_stats.shader_stats;
    snprintf(buf, bufsize,
             "P:%u/%u S:%u/%u SPV:%u/%u L:%s W:%u",
             s->pipeline_cache_hits,
             s->pipeline_cache_hits + s->pipeline_cache_misses,
             s->shader_cache_hits,
             s->shader_cache_hits + s->shader_cache_misses,
             s->spv_cache_hits,
             s->spv_cache_hits + s->spv_cache_misses,
             s->pipeline_cache_disk_loaded ? "Y" : "N",
             s->pipeline_cache_disk_saved);
}

void nv2a_profile_get_workload_str(char *buf, int bufsize)
{
    unsigned int idx = (g_nv2a_stats.frame_ptr + NV2A_PROF_NUM_FRAMES - 1) %
                       NV2A_PROF_NUM_FRAMES;
    int *c = g_nv2a_stats.frame_history[idx].counters;
    snprintf(buf, bufsize,
             "BE:%d DA:%d IE:%d IB:%d IA:%d Clr:%d "
             "QS:%d/%d PGen:%d PBnd:%d PNd:%d RP:%d "
             "SGen:%d SBnd:%d SNd:%d UBOd:%d UBOn:%d "
             "TexU:%d GBU:%d/%d/%d/%d/%d "
             "Fin:Vbd%d Sc%d Sd%d Bs%d Fbd%d Pr%d Fl%d Flu%d St%d",
             c[NV2A_PROF_BEGIN_ENDS],
             c[NV2A_PROF_DRAW_ARRAYS],
             c[NV2A_PROF_INLINE_ELEMENTS],
             c[NV2A_PROF_INLINE_BUFFERS],
             c[NV2A_PROF_INLINE_ARRAYS],
             c[NV2A_PROF_CLEAR],
             c[NV2A_PROF_QUEUE_SUBMIT],
             c[NV2A_PROF_QUEUE_SUBMIT_AUX],
             c[NV2A_PROF_PIPELINE_GEN],
             c[NV2A_PROF_PIPELINE_BIND],
             c[NV2A_PROF_PIPELINE_NOTDIRTY],
             c[NV2A_PROF_PIPELINE_RENDERPASSES],
             c[NV2A_PROF_SHADER_GEN],
             c[NV2A_PROF_SHADER_BIND],
             c[NV2A_PROF_SHADER_BIND_NOTDIRTY],
             c[NV2A_PROF_SHADER_UBO_DIRTY],
             c[NV2A_PROF_SHADER_UBO_NOTDIRTY],
             c[NV2A_PROF_TEX_UPLOAD],
             c[NV2A_PROF_GEOM_BUFFER_UPDATE_1],
             c[NV2A_PROF_GEOM_BUFFER_UPDATE_2],
             c[NV2A_PROF_GEOM_BUFFER_UPDATE_3],
             c[NV2A_PROF_GEOM_BUFFER_UPDATE_4],
             c[NV2A_PROF_GEOM_BUFFER_UPDATE_4_NOTDIRTY],
             c[NV2A_PROF_FINISH_VERTEX_BUFFER_DIRTY],
             c[NV2A_PROF_FINISH_SURFACE_CREATE],
             c[NV2A_PROF_FINISH_SURFACE_DOWN],
             c[NV2A_PROF_FINISH_NEED_BUFFER_SPACE],
             c[NV2A_PROF_FINISH_FRAMEBUFFER_DIRTY],
             c[NV2A_PROF_FINISH_PRESENTING],
             c[NV2A_PROF_FINISH_FLIP_STALL],
             c[NV2A_PROF_FINISH_FLUSH],
             c[NV2A_PROF_FINISH_STALLED]);
}

const char *nv2a_profile_get_counter_name(unsigned int cnt)
{
    const char *default_names[NV2A_PROF__COUNT] = {
        #define _X(x) stringify(x),
        NV2A_PROF_COUNTERS_XMAC
        #undef _X
    };

    assert(cnt < NV2A_PROF__COUNT);
    return default_names[cnt] + 10; /* 'NV2A_PROF_' */
}

int nv2a_profile_get_counter_value(unsigned int cnt)
{
    assert(cnt < NV2A_PROF__COUNT);
    unsigned int idx = (g_nv2a_stats.frame_ptr + NV2A_PROF_NUM_FRAMES - 1) %
                       NV2A_PROF_NUM_FRAMES;
    return g_nv2a_stats.frame_history[idx].counters[cnt];
}
