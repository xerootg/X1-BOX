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

#include "qemu/osdep.h"
#include "qemu/fast-hash.h"
#include "hw/xbox/nv2a/debug.h"
#include "prim_rewrite.h"

void pgraph_prim_rewrite_init(PrimRewriteBuf *buf)
{
    buf->data = NULL;
    buf->capacity = 0;
}

void pgraph_prim_rewrite_finalize(PrimRewriteBuf *buf)
{
    g_free(buf->data);
    buf->data = NULL;
    buf->capacity = 0;
}

/* Runtime gate for the index-rewrite cache. Default is DISABLED until we
 * confirm it doesn't cause Adreno GPU hangs (observed 2026-05-25 title
 * screen). Set X1BOX_ENABLE_IDX_CACHE=1 to opt in. */
static bool prim_cache_disabled(void)
{
    static int s = -1;
    if (s < 0) {
        const char *e = getenv("X1BOX_ENABLE_IDX_CACHE");
        s = (e && *e && *e != '0') ? 0 : 1;
    }
    return s != 0;
}

void pgraph_prim_rewrite_cache_init(PrimRewriteCache *cache)
{
    memset(cache, 0, sizeof(*cache));
}

void pgraph_prim_rewrite_cache_finalize(PrimRewriteCache *cache)
{
    for (unsigned i = 0; i < PRIM_REWRITE_CACHE_ENTRIES; i++) {
        g_free(cache->entries[i].indices);
        cache->entries[i].indices = NULL;
    }
    memset(cache, 0, sizeof(*cache));
}

void pgraph_prim_rewrite_cache_invalidate(PrimRewriteCache *cache)
{
    /* Drop validity, keep allocations so the next miss reuses them. */
    for (unsigned i = 0; i < PRIM_REWRITE_CACHE_ENTRIES; i++) {
        cache->entries[i].valid = false;
    }
    cache->lru_seq = 0;
}

/*
 * Secondary collision-guard hash. Three sample points + count;
 * cheap to compute, catches the trivial "same length different
 * mesh" collisions without the full memcmp.
 */
static inline uint32_t prim_secondary_hash_indexed(
        const uint32_t *idx, unsigned count)
{
    if (count == 0) {
        return 0;
    }
    uint32_t a = idx[0];
    uint32_t b = idx[count / 2];
    uint32_t c = idx[count - 1];
    return count * 0x9E3779B1u ^ (a * 0x85EBCA77u) ^
           (b * 0xC2B2AE3Du) ^ (c * 0x27D4EB2Fu);
}

static inline uint32_t prim_secondary_hash_ranges(
        const int32_t *starts, const int32_t *counts, unsigned num_ranges)
{
    if (num_ranges == 0) {
        return 0;
    }
    uint32_t a = (uint32_t)starts[0];
    uint32_t b = (uint32_t)counts[0];
    uint32_t c = (uint32_t)starts[num_ranges - 1];
    uint32_t d = (uint32_t)counts[num_ranges - 1];
    return num_ranges * 0x9E3779B1u ^ (a * 0x85EBCA77u) ^
           (b * 0xC2B2AE3Du) ^ (c * 0x27D4EB2Fu) ^
           (d * 0x165667B1u);
}

static inline bool prim_mode_equal(const PrimAssemblyState *a,
                                   const PrimAssemblyState *b)
{
    return a->primitive_mode == b->primitive_mode &&
           a->polygon_mode == b->polygon_mode &&
           a->last_provoking == b->last_provoking &&
           a->flat_shading == b->flat_shading;
}

/*
 * Look up a cached entry. Returns the index in `cache->entries` on
 * hit, -1 on miss. Bumps lru_seq on hit.
 */
static int prim_cache_lookup(PrimRewriteCache *cache,
                             uint64_t key_hash,
                             uint32_t secondary_hash,
                             const PrimAssemblyState *mode,
                             uint32_t input_count)
{
    for (unsigned i = 0; i < PRIM_REWRITE_CACHE_ENTRIES; i++) {
        PrimRewriteCacheEntry *e = &cache->entries[i];
        if (!e->valid) {
            continue;
        }
        if (e->key_hash == key_hash &&
            e->secondary_hash == secondary_hash &&
            e->input_count == input_count &&
            prim_mode_equal(&e->mode, mode)) {
            e->lru_seq = ++cache->lru_seq;
            return (int)i;
        }
    }
    return -1;
}

/*
 * Pick a slot for a fresh insertion. Prefers an invalid slot, falls
 * back to the LRU entry. Returns the chosen index; sets *was_eviction
 * to true if we displaced a valid entry.
 */
static unsigned prim_cache_pick_victim(PrimRewriteCache *cache,
                                       bool *was_eviction)
{
    unsigned victim = 0;
    uint32_t min_seq = UINT32_MAX;
    bool any_invalid = false;
    for (unsigned i = 0; i < PRIM_REWRITE_CACHE_ENTRIES; i++) {
        PrimRewriteCacheEntry *e = &cache->entries[i];
        if (!e->valid) {
            victim = i;
            any_invalid = true;
            break;
        }
        if (e->lru_seq < min_seq) {
            min_seq = e->lru_seq;
            victim = i;
        }
    }
    *was_eviction = !any_invalid;
    return victim;
}

static void prim_cache_store(PrimRewriteCache *cache,
                             uint64_t key_hash,
                             uint32_t secondary_hash,
                             const PrimAssemblyState *mode,
                             uint32_t input_count,
                             const uint32_t *indices,
                             unsigned num_indices)
{
    bool was_eviction = false;
    unsigned slot = prim_cache_pick_victim(cache, &was_eviction);
    PrimRewriteCacheEntry *e = &cache->entries[slot];

    if (was_eviction) {
        g_nv2a_stats.shader_stats.prim_rewrite_cache_evicts++;
    }

    if (e->indices_cap < num_indices) {
        unsigned new_cap = MAX(num_indices, e->indices_cap ? e->indices_cap * 2 : 32);
        e->indices = g_realloc(e->indices, new_cap * sizeof(uint32_t));
        e->indices_cap = new_cap;
    }
    if (num_indices > 0) {
        memcpy(e->indices, indices, num_indices * sizeof(uint32_t));
    }
    e->num_indices = num_indices;
    e->key_hash = key_hash;
    e->secondary_hash = secondary_hash;
    e->mode = *mode;
    e->input_count = input_count;
    e->lru_seq = ++cache->lru_seq;
    e->valid = true;
}

static void ensure_capacity(PrimRewriteBuf *buf, unsigned int needed)
{
    if (needed <= buf->capacity) {
        return;
    }
    buf->capacity = MAX(needed, buf->capacity * 2);
    buf->data = g_realloc(buf->data, buf->capacity * sizeof(uint32_t));
}

enum ShaderPrimitiveMode
pgraph_prim_rewrite_get_output_mode(enum ShaderPrimitiveMode primitive_mode,
                                    enum ShaderPolygonMode polygon_mode)
{
    switch (primitive_mode) {
    case PRIM_TYPE_POINTS:
        return PRIM_TYPE_POINTS;
    case PRIM_TYPE_LINES:
    case PRIM_TYPE_LINE_STRIP:
    case PRIM_TYPE_LINE_LOOP:
        return PRIM_TYPE_LINES;
    case PRIM_TYPE_TRIANGLES:
    case PRIM_TYPE_TRIANGLE_STRIP:
    case PRIM_TYPE_TRIANGLE_FAN:
        return PRIM_TYPE_TRIANGLES;
    case PRIM_TYPE_QUADS:
    case PRIM_TYPE_QUAD_STRIP:
    case PRIM_TYPE_POLYGON:
        return polygon_mode == POLY_MODE_LINE ? PRIM_TYPE_LINES :
                                                PRIM_TYPE_TRIANGLES;
    default:
        assert(!"Unexpected primitive mode");
        return primitive_mode;
    }
}

static inline bool needs_rewrite(PrimAssemblyState mode)
{
    switch (mode.primitive_mode) {
    case PRIM_TYPE_POINTS:
        return false;
    case PRIM_TYPE_LINES:
    case PRIM_TYPE_TRIANGLES:
        return mode.last_provoking && mode.flat_shading;
    default:
        return true;
    }
}

static unsigned int max_output_indices(enum ShaderPrimitiveMode mode,
                                       enum ShaderPolygonMode polygon_mode,
                                       unsigned int input_count)
{
    switch (mode) {
    case PRIM_TYPE_LINES:
        return input_count;
    case PRIM_TYPE_LINE_STRIP:
        return (input_count >= 2) ? (input_count - 1) * 2 : 0;
    case PRIM_TYPE_LINE_LOOP:
        return (input_count >= 2) ? input_count * 2 : 0;
    case PRIM_TYPE_TRIANGLES:
        return input_count;
    case PRIM_TYPE_TRIANGLE_STRIP:
    case PRIM_TYPE_TRIANGLE_FAN:
        return (input_count >= 3) ? (input_count - 2) * 3 : 0;
    case PRIM_TYPE_POLYGON:
        if (polygon_mode == POLY_MODE_LINE) {
            return (input_count >= 2) ? input_count * 2 : 0;
        }
        return (input_count >= 3) ? (input_count - 2) * 3 : 0;
    case PRIM_TYPE_QUADS:
        if (polygon_mode == POLY_MODE_LINE) {
            return (input_count / 4) * 8;
        }
        return (input_count / 4) * 6;
    case PRIM_TYPE_QUAD_STRIP:
        if (polygon_mode == POLY_MODE_LINE) {
            return (input_count >= 4) ? ((input_count - 2) / 2) * 8 : 0;
        }
        return (input_count >= 4) ? ((input_count - 2) / 2) * 6 : 0;
    default:
        return 0;
    }
}

static inline uint32_t idx_at(const uint32_t *idx, unsigned int i,
                              uint32_t base)
{
    return idx ? idx[i] : base + i;
}

static inline void emit_vertex(PrimRewrite *r, uint32_t v)
{
    r->indices[r->num_indices++] = v;
}

static inline void emit_line(PrimRewrite *r, uint32_t a, uint32_t b)
{
    emit_vertex(r, a);
    emit_vertex(r, b);
}

/* Place provoking vertex p at index 0. */
static inline void emit_line_pv(PrimRewrite *r, uint32_t a, uint32_t b,
                                uint32_t p)
{
    if (p == a) {
        emit_line(r, a, b);
    } else {
        emit_line(r, b, a);
    }
}

static inline void emit_tri(PrimRewrite *r, uint32_t a, uint32_t b, uint32_t c)
{
    emit_vertex(r, a);
    emit_vertex(r, b);
    emit_vertex(r, c);
}

/* Rotate provoking vertex p to index 0, preserving winding of (a, b, c). */
static inline void emit_tri_pv(PrimRewrite *r, uint32_t a, uint32_t b,
                               uint32_t c, uint32_t p)
{
    if (p == a) {
        emit_tri(r, a, b, c);
    } else if (p == b) {
        emit_tri(r, b, c, a);
    } else {
        emit_tri(r, c, a, b);
    }
}

static void rewrite_lines(PrimRewrite *r, const uint32_t *idx, uint32_t base,
                          unsigned int count, bool last_provoking)
{
    for (unsigned int i = 0; i + 1 < count; i += 2) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t pv = last_provoking ? v1 : v0;

        emit_line_pv(r, v0, v1, pv);
    }
}

static void rewrite_line_strip(PrimRewrite *r, const uint32_t *idx,
                               uint32_t base, unsigned int count,
                               bool last_provoking)
{
    for (unsigned int i = 0; i + 1 < count; i++) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t pv = last_provoking ? v1 : v0;

        emit_line_pv(r, v0, v1, pv);
    }
}

static void rewrite_line_loop(PrimRewrite *r, const uint32_t *idx,
                              uint32_t base, unsigned int count,
                              bool last_provoking)
{
    if (count < 2) {
        return;
    }

    for (unsigned int i = 0; i + 1 < count; i++) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t pv = last_provoking ? v1 : v0;

        emit_line_pv(r, v0, v1, pv);
    }

    uint32_t v_last = idx_at(idx, count - 1, base);
    uint32_t v_first = idx_at(idx, 0, base);
    uint32_t pv = last_provoking ? v_first : v_last;

    emit_line_pv(r, v_last, v_first, pv);
}

static void rewrite_triangles(PrimRewrite *r, const uint32_t *idx,
                              uint32_t base, unsigned int count,
                              bool last_provoking)
{
    for (unsigned int i = 0; i + 2 < count; i += 3) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t pv = last_provoking ? v2 : v0;

        emit_tri_pv(r, v0, v1, v2, pv);
    }
}

static void rewrite_triangle_strip(PrimRewrite *r, const uint32_t *idx,
                                   uint32_t base, unsigned int count,
                                   bool last_provoking)
{
    for (unsigned int i = 0; i + 2 < count; i++) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t pv = last_provoking ? v2 : v0;

        if (i & 1) {
            emit_tri_pv(r, v1, v0, v2, pv);
        } else {
            emit_tri_pv(r, v0, v1, v2, pv);
        }
    }
}

static void rewrite_triangle_fan(PrimRewrite *r, const uint32_t *idx,
                                 uint32_t base, unsigned int count,
                                 bool last_provoking)
{
    if (count < 3) {
        return;
    }

    uint32_t hub = idx_at(idx, 0, base);

    for (unsigned int i = 0; i + 2 < count; i++) {
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t pv = last_provoking ? v2 : v1;

        emit_tri_pv(r, hub, v1, v2, pv);
    }
}

static void rewrite_quads(PrimRewrite *r, const uint32_t *idx, uint32_t base,
                          unsigned int count, bool flat_shading)
{
    for (unsigned int i = 0; i + 3 < count; i += 4) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t v3 = idx_at(idx, i + 3, base);

        if (flat_shading) {
            /* Use v1-v3 diagonal so provoking vertex v3 is in both triangles.
             * This gives correct flat shading color but slightly different
             * depth slope vs hardware. */
            emit_tri(r, v3, v0, v1);
            emit_tri(r, v3, v1, v2);
        } else {
            /* v0-v2 diagonal: matches hardware quad tessellation */
            emit_tri(r, v0, v1, v2);
            emit_tri(r, v0, v2, v3);
        }
    }
}

static void rewrite_quads_line(PrimRewrite *r, const uint32_t *idx,
                               uint32_t base, unsigned int count)
{
    for (unsigned int i = 0; i + 3 < count; i += 4) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t v3 = idx_at(idx, i + 3, base);

        emit_line(r, v0, v1);
        emit_line(r, v1, v2);
        emit_line(r, v2, v3);
        emit_line(r, v3, v0);
    }
}

static void rewrite_quad_strip(PrimRewrite *r, const uint32_t *idx,
                               uint32_t base, unsigned int count,
                               bool flat_shading)
{
    if (count < 4) {
        return;
    }

    for (unsigned int i = 0; i + 3 < count; i += 2) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t v3 = idx_at(idx, i + 3, base);

        if (flat_shading) {
            /* Use v0-v3 diagonal so provoking vertex v3 is in both triangles.
             * This gives correct flat shading color but slightly different
             * depth slope vs hardware. */
            emit_tri(r, v3, v2, v0);
            emit_tri(r, v3, v0, v1);
        } else {
            /* v1-v2 diagonal: matches hardware quad strip tessellation */
            emit_tri(r, v0, v1, v2);
            emit_tri(r, v2, v1, v3);
        }
    }
}

static void rewrite_quad_strip_line(PrimRewrite *r, const uint32_t *idx,
                                    uint32_t base, unsigned int count)
{
    if (count < 4) {
        return;
    }

    for (unsigned int i = 0; i + 3 < count; i += 2) {
        uint32_t v0 = idx_at(idx, i, base);
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);
        uint32_t v3 = idx_at(idx, i + 3, base);

        emit_line(r, v0, v1);
        emit_line(r, v1, v3);
        emit_line(r, v3, v2);
        emit_line(r, v2, v0);
    }
}

static void rewrite_polygon(PrimRewrite *r, const uint32_t *idx, uint32_t base,
                            unsigned int count)
{
    if (count < 3) {
        return;
    }

    uint32_t hub = idx_at(idx, 0, base);

    for (unsigned int i = 0; i + 2 < count; i++) {
        uint32_t v1 = idx_at(idx, i + 1, base);
        uint32_t v2 = idx_at(idx, i + 2, base);

        emit_tri(r, hub, v1, v2);
    }
}

static void rewrite_polygon_line(PrimRewrite *r, const uint32_t *idx,
                                 uint32_t base, unsigned int count)
{
    if (count < 2) {
        return;
    }

    for (unsigned int i = 0; i + 1 < count; i++) {
        emit_line(r, idx_at(idx, i, base), idx_at(idx, i + 1, base));
    }

    /* Close the loop */
    emit_line(r, idx_at(idx, count - 1, base), idx_at(idx, 0, base));
}

static void rewrite_indices(PrimRewrite *r, const PrimAssemblyState *mode,
                            const uint32_t *idx, uint32_t base,
                            unsigned int num_indices)
{
    switch (mode->primitive_mode) {
    case PRIM_TYPE_LINES:
        rewrite_lines(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_LINE_STRIP:
        rewrite_line_strip(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_LINE_LOOP:
        rewrite_line_loop(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_TRIANGLES:
        rewrite_triangles(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        rewrite_triangle_strip(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
        rewrite_triangle_fan(r, idx, base, num_indices, mode->last_provoking);
        break;
    case PRIM_TYPE_QUADS:
        if (mode->polygon_mode == POLY_MODE_LINE) {
            rewrite_quads_line(r, idx, base, num_indices);
        } else {
            rewrite_quads(r, idx, base, num_indices, mode->flat_shading);
        }
        break;
    case PRIM_TYPE_QUAD_STRIP:
        if (mode->polygon_mode == POLY_MODE_LINE) {
            rewrite_quad_strip_line(r, idx, base, num_indices);
        } else {
            rewrite_quad_strip(r, idx, base, num_indices, mode->flat_shading);
        }
        break;
    case PRIM_TYPE_POLYGON:
        if (mode->polygon_mode == POLY_MODE_LINE) {
            rewrite_polygon_line(r, idx, base, num_indices);
        } else {
            rewrite_polygon(r, idx, base, num_indices);
        }
        break;
    default:
        assert(!"Unexpected primitive mode");
        break;
    }
}

/*
 * Build a stable 64-bit key from (mode, num_ranges, starts[], counts[]).
 * Mode bits are folded into the seed so two different rewrite shapes
 * with the same range layout don't collide.
 */
static uint64_t prim_key_ranges(PrimAssemblyState mode,
                                const int32_t *starts,
                                const int32_t *counts,
                                unsigned int num_ranges)
{
    struct {
        uint8_t pm;
        uint8_t poly;
        uint8_t lp;
        uint8_t flat;
        uint32_t num_ranges;
    } header = {
        (uint8_t)mode.primitive_mode,
        (uint8_t)mode.polygon_mode,
        (uint8_t)mode.last_provoking,
        (uint8_t)mode.flat_shading,
        num_ranges,
    };
    uint64_t h = fast_hash((const uint8_t *)&header, sizeof(header));
    if (num_ranges > 0) {
        h ^= fast_hash((const uint8_t *)starts,
                       num_ranges * sizeof(int32_t));
        h ^= fast_hash((const uint8_t *)counts,
                       num_ranges * sizeof(int32_t));
    }
    return h;
}

static uint64_t prim_key_indexed(PrimAssemblyState mode,
                                 const uint32_t *input_indices,
                                 unsigned int num_input_indices)
{
    struct {
        uint8_t pm;
        uint8_t poly;
        uint8_t lp;
        uint8_t flat;
        uint32_t num_input_indices;
    } header = {
        (uint8_t)mode.primitive_mode,
        (uint8_t)mode.polygon_mode,
        (uint8_t)mode.last_provoking,
        (uint8_t)mode.flat_shading,
        num_input_indices,
    };
    uint64_t h = fast_hash((const uint8_t *)&header, sizeof(header));
    if (num_input_indices > 0) {
        h ^= fast_hash((const uint8_t *)input_indices,
                       num_input_indices * sizeof(uint32_t));
    }
    return h;
}

PrimRewrite pgraph_prim_rewrite_ranges(PrimRewriteBuf *buf,
                                       PrimRewriteCache *cache,
                                       PrimAssemblyState mode,
                                       const int32_t *starts,
                                       const int32_t *counts,
                                       unsigned int num_ranges)
{
    PrimRewrite result = { 0 };

    /* Env kill switch — disable the LRU cache entirely. */
    if (prim_cache_disabled()) {
        cache = NULL;
    }

    assert(mode.polygon_mode != POLY_MODE_POINT ||
           mode.primitive_mode != PRIM_TYPE_POLYGON);

    if (!needs_rewrite(mode)) {
        return result;
    }

    unsigned int total_max_output = 0;
    unsigned int total_input = 0;
    for (unsigned int r = 0; r < num_ranges; r++) {
        total_max_output += max_output_indices(mode.primitive_mode,
                                               mode.polygon_mode, counts[r]);
        total_input += (unsigned int)counts[r];
    }

    if (total_max_output == 0) {
        return result;
    }

    if (cache) {
        uint64_t key = prim_key_ranges(mode, starts, counts, num_ranges);
        uint32_t sec = prim_secondary_hash_ranges(starts, counts, num_ranges);
        int hit = prim_cache_lookup(cache, key, sec, &mode, total_input);
        if (hit >= 0) {
            PrimRewriteCacheEntry *e = &cache->entries[hit];
            ensure_capacity(buf, e->num_indices);
            if (e->num_indices > 0) {
                memcpy(buf->data, e->indices,
                       e->num_indices * sizeof(uint32_t));
            }
            result.indices = buf->data;
            result.num_indices = e->num_indices;
            g_nv2a_stats.shader_stats.prim_rewrite_cache_hits++;
            return result;
        }
        g_nv2a_stats.shader_stats.prim_rewrite_cache_misses++;
    }

    ensure_capacity(buf, total_max_output);
    result.indices = buf->data;

    for (unsigned int r = 0; r < num_ranges; r++) {
        if (counts[r] == 0) {
            continue;
        }

        rewrite_indices(&result, &mode, NULL, starts[r], counts[r]);
    }

    if (cache) {
        uint64_t key = prim_key_ranges(mode, starts, counts, num_ranges);
        uint32_t sec = prim_secondary_hash_ranges(starts, counts, num_ranges);
        prim_cache_store(cache, key, sec, &mode, total_input,
                         result.indices, result.num_indices);
    }

    return result;
}

PrimRewrite pgraph_prim_rewrite_indexed(PrimRewriteBuf *buf,
                                        PrimRewriteCache *cache,
                                        PrimAssemblyState mode,
                                        const uint32_t *input_indices,
                                        unsigned int num_input_indices)
{
    PrimRewrite result = { 0 };

    /* Env kill switch — disable the LRU cache entirely. */
    if (prim_cache_disabled()) {
        cache = NULL;
    }

    assert(mode.polygon_mode != POLY_MODE_POINT ||
           mode.primitive_mode != PRIM_TYPE_POLYGON);

    if (!needs_rewrite(mode)) {
        return result;
    }

    unsigned int max_output = max_output_indices(
        mode.primitive_mode, mode.polygon_mode, num_input_indices);

    if (max_output == 0) {
        return result;
    }

    if (cache) {
        uint64_t key = prim_key_indexed(mode, input_indices, num_input_indices);
        uint32_t sec = prim_secondary_hash_indexed(input_indices,
                                                   num_input_indices);
        int hit = prim_cache_lookup(cache, key, sec, &mode, num_input_indices);
        if (hit >= 0) {
            PrimRewriteCacheEntry *e = &cache->entries[hit];
            ensure_capacity(buf, e->num_indices);
            if (e->num_indices > 0) {
                memcpy(buf->data, e->indices,
                       e->num_indices * sizeof(uint32_t));
            }
            result.indices = buf->data;
            result.num_indices = e->num_indices;
            g_nv2a_stats.shader_stats.prim_rewrite_cache_hits++;
            return result;
        }
        g_nv2a_stats.shader_stats.prim_rewrite_cache_misses++;
    }

    ensure_capacity(buf, max_output);
    result.indices = buf->data;

    rewrite_indices(&result, &mode, input_indices, 0, num_input_indices);

    if (cache) {
        uint64_t key = prim_key_indexed(mode, input_indices, num_input_indices);
        uint32_t sec = prim_secondary_hash_indexed(input_indices,
                                                   num_input_indices);
        prim_cache_store(cache, key, sec, &mode, num_input_indices,
                         result.indices, result.num_indices);
    }

    return result;
}
