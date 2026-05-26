/*
 * qht internal layout — exposed for type-specialised inline lookups.
 *
 * Upstream `util/qht.c` keeps these definitions private. The TCG dispatch
 * hot path benefits from inlining `tb_lookup_cmp` directly into the qht
 * bucket walk (eliminates an indirect call per probe and lets the
 * compiler hoist desc-loop-invariant loads). To do that, `cpu-exec.c`
 * needs to see the bucket / map layout. Splitting out a header is the
 * least-invasive way to share definitions between `util/qht.c` and a
 * type-specialised callsite.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef QEMU_QHT_INTERNAL_H
#define QEMU_QHT_INTERNAL_H

#include "qemu/qht.h"
#include "qemu/seqlock.h"
#include "qemu/thread.h"

#define QHT_BUCKET_ALIGN 64

/* define these to keep sizeof(qht_bucket) within QHT_BUCKET_ALIGN */
#if HOST_LONG_BITS == 32
#define QHT_BUCKET_ENTRIES 6
#else /* 64-bit */
#define QHT_BUCKET_ENTRIES 4
#endif

#define QHT_TSAN_BUCKET_LOCKS_BITS 4
#define QHT_TSAN_BUCKET_LOCKS (1 << QHT_TSAN_BUCKET_LOCKS_BITS)

struct qht_tsan_lock {
    QemuSpin lock;
} QEMU_ALIGNED(QHT_BUCKET_ALIGN);

struct qht_bucket {
    QemuSpin lock;
    QemuSeqLock sequence;
    uint32_t hashes[QHT_BUCKET_ENTRIES];
    void *pointers[QHT_BUCKET_ENTRIES];
    struct qht_bucket *next;
} QEMU_ALIGNED(QHT_BUCKET_ALIGN);

struct qht_map {
    struct rcu_head rcu;
    struct qht_bucket *buckets;
    size_t n_buckets;
    size_t n_added_buckets;
    size_t n_added_buckets_threshold;
#ifdef CONFIG_TSAN
    struct qht_tsan_lock tsan_bucket_locks[QHT_TSAN_BUCKET_LOCKS];
#endif
};

static inline struct qht_bucket *
qht_map_to_bucket(const struct qht_map *map, uint32_t hash)
{
    return &map->buckets[hash & (map->n_buckets - 1)];
}

#endif /* QEMU_QHT_INTERNAL_H */
