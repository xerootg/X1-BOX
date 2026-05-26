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
 * 2026-05-25 (Zenfone 10 / Snapdragon 8 Gen 2 X3, 1 MiB private L2):
 * tried 15 bits (32 K entries × 16 B = 512 KiB per CPU) on the
 * hypothesis that the prior 14-bit regression was L2 saturation on the
 * Pixel-only A720 (256 KiB L2), and X3's bigger L2 would absorb it.
 * Halo 2 gameplay callgraph said otherwise:
 *
 *     symbol                         8K (13b)   32K (15b)
 *     helper_lookup_tb_ptr            3.62 %     9.78 %
 *     qht_lookup_custom               2.76 %     4.97 %
 *     tb_lookup_cmp                   ~0   %     1.56 %
 *     tcg_flush_jmp_cache             ~0   %     1.75 %
 *
 * The 32K cache LOST hit rate too — qht_lookup_custom went UP, not down.
 * Direct-mapped + 4× larger = more cold slots that displace hot ones
 * under the working-set hash pattern Halo 2 generates. Plus
 * tcg_flush_jmp_cache is a linear walk so the 4× bigger array costs 4×
 * more per flush, and flushes happen often (CR3, tb_phys_invalidate).
 *
 * 13 bits (8192 entries × 16 B = 128 KiB) is the calibrated optimum on
 * both Pixel A720 (256 KiB L2) and Zenfone X3 (1 MiB L2). Fix forward
 * means reverting size on fresh proof, not blind reverts on stale
 * comments — and the proof now exists.
 *
 * Next lever, if a future profile shows the working set has grown: an
 * associative or victim-cache structure, not more direct-mapped entries.
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
