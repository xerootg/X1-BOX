/*
 * The per-CPU TranslationBlock jump cache.
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ACCEL_TCG_TB_JMP_CACHE_H
#define ACCEL_TCG_TB_JMP_CACHE_H

#include "qemu/rcu.h"
#include "exec/cpu-common.h"

/*
 * Sizing the per-CPU TB jump cache.
 *
 * Default upstream: 12 bits → 4096 entries × 16 B = 64 KiB per CPU.
 * Diagnosed (2026-05-21, Halo 2 in-game @ scale=3): the X4-pinned vCPU
 * thread spends 10.93% in qht_lookup_custom + 5.13% in helper_lookup_tb_ptr
 * + 4.10% in tb_lookup_cmp — ~21% of TCG time. The hot path of
 * tb_lookup() returns via the per-CPU jc cache; the 21% number is what
 * we pay when that cache misses and falls through to the global QHT.
 *
 * Working-set sizing math from the profile: at 13.9 FPS, vCPU is doing
 * roughly 50 ms of guest work per frame. With Halo 2's typical TB span
 * of ~50 instructions, that's ~600 k TBs touched per frame, sampling
 * something like 20-40 k unique TBs in steady-state. A 4 k-entry direct-
 * mapped cache thrashes hard against a 20 k unique set (~5x oversubscribed),
 * giving a hit rate around 60-70%. Bumping to 13 bits (8192 entries,
 * 128 KiB) halves the conflict-miss rate without overflowing the X4's
 * 256 KiB L2 — and since each lookup also touches the TB struct itself
 * (cs_base/flags/cflags compare), the working set per lookup is already
 * larger than a single line, so spilling the cache to L2 is a small marginal
 * cost compared to halving qht trips.
 *
 * Net expected gain: qht_lookup_custom + tb_lookup_cmp drop from ~15%
 * combined to ~8%, recovering ~7% of TCG-per-frame on this scene.
 *
 * Followup #1 (2026-05-22, in-game): tried 14 bits (16 K entries =
 * 256 KiB exactly the X4 L2 size). qht_lookup_custom did NOT drop —
 * went up 4.07 → 4.52 — and `cpu_exec_loop` jumped to 4.53% on the
 * title screen (was ~1%), with tcg_flush_jmp_cache also up. The L2
 * saturation hypothesis won: 256 KiB jc + TB struct touches + other
 * working set evicts each other every lookup. Calibrated back to 13
 * bits (128 KiB jc) where qht stays bounded and flush walks are 2×
 * cheaper.
 *
 * If a future profile shows the working set has grown again, the next
 * lever is an associative or victim-cache structure rather than more
 * direct-mapped entries — at 13 bits we're already at half-L2.
 */
#define TB_JMP_CACHE_BITS 13
#define TB_JMP_CACHE_SIZE (1 << TB_JMP_CACHE_BITS)

/*
 * Invalidated in parallel; all accesses to 'tb' must be atomic.
 * A valid entry is read/written by a single CPU, therefore there is
 * no need for qatomic_rcu_read() and pc is always consistent with a
 * non-NULL value of 'tb'.  Strictly speaking pc is only needed for
 * CF_PCREL, but it's used always for simplicity.
 */
typedef struct CPUJumpCache {
    struct rcu_head rcu;
    struct {
        TranslationBlock *tb;
        vaddr pc;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
