/*
 * Geforce NV2A PGRAPH Primitive Index Rewrite
 *
 * Rewrites NV2A primitive types to triangle/line/point lists on CPU.
 * Handles provoking vertex placement for flat shading correctness.
 *
 * Copyright (c) 2026 Matt Borgerson
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

#ifndef HW_XBOX_NV2A_PGRAPH_PRIM_REWRITE_H
#define HW_XBOX_NV2A_PGRAPH_PRIM_REWRITE_H

#include <stdbool.h>
#include <stdint.h>
#include "vsh_regs.h"

typedef struct PrimRewriteBuf {
    uint32_t *data;
    unsigned int capacity; /* in elements */
} PrimRewriteBuf;

typedef struct PrimRewrite {
    uint32_t *indices; /* points into PrimRewriteBuf; do NOT free */
    unsigned int num_indices;
} PrimRewrite;

typedef struct PrimAssemblyState {
    enum ShaderPrimitiveMode primitive_mode;
    enum ShaderPolygonMode polygon_mode;
    bool last_provoking;
    bool flat_shading;
} PrimAssemblyState;

/*
 * LRU cache for the output of pgraph_prim_rewrite_indexed /
 * pgraph_prim_rewrite_ranges. Halo 2 redraws the same meshes per-frame
 * so the rewrite output is highly repetitive. Sequential rewrite is
 * skipped (the rewrite itself is cheaper than a cache probe).
 *
 * Hash key = fast_hash(mode_bits, num_input_indices, input_indices[0..num])
 * for the indexed path, or (mode_bits, num_ranges, starts[..], counts[..])
 * for the ranges path. Secondary cheap hash is the tuple
 * (input_count, first, mid, last) used as a collision guard — both must
 * match for a hit.
 *
 * Entries hold an owned (g_realloc'd) index buffer. On hit, the cached
 * indices are copied into the per-renderer PrimRewriteBuf so callers
 * see the same `result.indices` pointer convention.
 */
#define PRIM_REWRITE_CACHE_ENTRIES 32

typedef struct PrimRewriteCacheEntry {
    bool valid;
    uint64_t key_hash;
    uint32_t secondary_hash;
    PrimAssemblyState mode;
    uint32_t input_count;
    uint32_t *indices;       /* owned; g_realloc'd */
    unsigned indices_cap;    /* in elements */
    unsigned num_indices;
    uint32_t lru_seq;
} PrimRewriteCacheEntry;

typedef struct PrimRewriteCache {
    PrimRewriteCacheEntry entries[PRIM_REWRITE_CACHE_ENTRIES];
    uint32_t lru_seq;
} PrimRewriteCache;

void pgraph_prim_rewrite_init(PrimRewriteBuf *buf);
void pgraph_prim_rewrite_finalize(PrimRewriteBuf *buf);

void pgraph_prim_rewrite_cache_init(PrimRewriteCache *cache);
void pgraph_prim_rewrite_cache_finalize(PrimRewriteCache *cache);
void pgraph_prim_rewrite_cache_invalidate(PrimRewriteCache *cache);
enum ShaderPrimitiveMode
pgraph_prim_rewrite_get_output_mode(enum ShaderPrimitiveMode primitive_mode,
                                    enum ShaderPolygonMode polygon_mode);

/*
 * `cache` may be NULL (GL renderer doesn't wire one up). The Vulkan
 * path passes &r->index_cache. The sequential helper skips the cache
 * by passing NULL — the rewrite cost is below a probe cost.
 */
PrimRewrite pgraph_prim_rewrite_indexed(PrimRewriteBuf *buf,
                                        PrimRewriteCache *cache,
                                        PrimAssemblyState mode,
                                        const uint32_t *input_indices,
                                        unsigned int num_input_indices);

PrimRewrite pgraph_prim_rewrite_ranges(PrimRewriteBuf *buf,
                                       PrimRewriteCache *cache,
                                       PrimAssemblyState mode,
                                       const int32_t *starts,
                                       const int32_t *counts,
                                       unsigned int num_ranges);

static inline PrimRewrite pgraph_prim_rewrite_sequential(PrimRewriteBuf *buf,
                                                         PrimAssemblyState mode,
                                                         int32_t start,
                                                         int32_t count)
{
    /* sequential: cache disabled — rewrite cost < probe cost. */
    return pgraph_prim_rewrite_ranges(buf, NULL, mode, &start, &count, 1);
}

#endif
