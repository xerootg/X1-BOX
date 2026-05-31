/*
 * Geforce NV2A PGRAPH Vulkan Renderer
 *
 * Copyright (c) 2024 Matt Borgerson
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

#include "renderer.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

/* Mali speculatively prefetches past the last vertex of a draw. We over-
 * allocate fs->vertex_ram by this many bytes and lie about the usable size
 * in pgraph_vk_buffer_has_space_for(). Defined here so both the allocation
 * site and the size check below see it. */
#define MALI_DESCRIPTOR_TAIL_RESERVATION (1024u * 1024u)

typedef struct MemoryBudget {
    size_t total_heap;
    size_t renderer_budget;
    size_t vertex_inline_cap;
    size_t index_cap;
    size_t staging_cap;
    size_t perframe_vtx_cap;
    size_t perframe_idx_cap;
    size_t perframe_uni_cap;
    size_t perframe_stg_cap;
    size_t shader_module_cache_entries;
    size_t texture_cache_entries;
    int image_pool_max;
    int surface_image_pool_max;
    size_t surface_image_pool_bytes_max; /* 0 = uncapped (legacy) */
    size_t texture_cache_bytes_max;      /* 0 = uncapped (legacy) */
} MemoryBudget;

static MemoryBudget compute_memory_budget(PGRAPHVkState *r)
{
    const size_t mib = 1024 * 1024;
    const size_t gib = 1024 * mib;

    VkPhysicalDeviceMemoryProperties const *props;
    vmaGetMemoryProperties(r->allocator, &props);

    size_t total_heap = 0;
    for (uint32_t i = 0; i < props->memoryHeapCount; i++) {
        if (props->memoryHeaps[i].size > total_heap) {
            total_heap = props->memoryHeaps[i].size;
        }
    }

    MemoryBudget b = { .total_heap = total_heap };

#ifdef __ANDROID__
    if (total_heap <= 4 * gib) {
        b.renderer_budget = 512 * mib;
    } else if (total_heap <= 6 * gib) {
        b.renderer_budget = 768 * mib;
    } else if (total_heap <= 8 * gib) {
        b.renderer_budget = 1024 * mib;
    } else if (total_heap <= 12 * gib) {
        b.renderer_budget = 1536 * mib;
    } else if (total_heap <= 16 * gib) {
        b.renderer_budget = 2048 * mib;
    } else if (total_heap <= 22 * gib) {
        b.renderer_budget = 3072 * mib;
    } else {
        b.renderer_budget = SIZE_MAX;
    }
#else
    b.renderer_budget = SIZE_MAX;
#endif

    if (b.renderer_budget == SIZE_MAX) {
        b.vertex_inline_cap = SIZE_MAX;
        b.index_cap = SIZE_MAX;
        b.staging_cap = SIZE_MAX;
        b.perframe_vtx_cap = SIZE_MAX;
        b.perframe_idx_cap = SIZE_MAX;
        b.perframe_uni_cap = SIZE_MAX;
        b.perframe_stg_cap = SIZE_MAX;
        b.shader_module_cache_entries = 50 * 1024;
        b.texture_cache_entries = 1024;
        b.image_pool_max = 128;
        b.surface_image_pool_max = 64;
        b.surface_image_pool_bytes_max = 0; /* uncapped on desktop */
        b.texture_cache_bytes_max = 0;      /* uncapped on desktop */
    } else {
        size_t budget = b.renderer_budget;
        b.vertex_inline_cap = MAX(8 * mib, budget / 5);
        b.index_cap         = MAX(4 * mib, budget / 20);
        b.staging_cap       = MAX(16 * mib, budget * 18 / 100);
        b.perframe_vtx_cap  = MAX(8 * mib, budget * 10 / 100);
        b.perframe_idx_cap  = MAX(2 * mib, budget / 50);
        b.perframe_uni_cap  = MAX(8 * mib, budget * 6 / 100);
        b.perframe_stg_cap  = MAX(16 * mib, budget * 12 / 100);

        size_t budget_mib = budget / mib;
        b.shader_module_cache_entries = budget_mib * 4;
        if (b.shader_module_cache_entries < 2048) {
            b.shader_module_cache_entries = 2048;
        }
        if (b.shader_module_cache_entries > 50 * 1024) {
            b.shader_module_cache_entries = 50 * 1024;
        }

        if (budget_mib <= 768) {
            b.texture_cache_entries = 256;
            b.image_pool_max = 16;
            b.surface_image_pool_max = 8;
        } else if (budget_mib <= 1536) {
            b.texture_cache_entries = 512;
            b.image_pool_max = 32;
            b.surface_image_pool_max = 16;
        } else {
            b.texture_cache_entries = 1024;
            b.image_pool_max = 64;
            b.surface_image_pool_max = 32;
        }

        /*
         * Surface image pool bytes cap.
         *
         * Count-based cap alone breaks down at surface_scale>1 — each pooled
         * entry holds image + image_scratch, and at 3x scale a 640x480
         * surface becomes 1920x1440 RGBA8 = 21.6 MB per entry. A 32-entry
         * pool then pins ~700 MB of GPU memory and reliably OOMs Adreno
         * KGSL mid-session (observed: 10800 KB alloc refusals at FLIP_STALL,
         * abort via VK_CHECK on vmaCreateImage).
         *
         * Cap at ~15% of the renderer budget — gives the pool meaningful
         * headroom for hot-set reuse without monopolizing GPU memory that
         * shader/texture caches also need. Floor at 64 MB so 1x-scale users
         * keep effectively-unlimited pool behavior on small budgets.
         */
        size_t pool_bytes = budget * 15 / 100;
        if (pool_bytes < 64 * mib) {
            pool_bytes = 64 * mib;
        }
        b.surface_image_pool_bytes_max = pool_bytes;

        /*
         * Texture-cache bytes cap. The cache is otherwise count-only (1024
         * entries); a full cache of RGBA8+mip textures pins ~2.8 GB on Halo 2
         * — nearly 3x the entire renderer budget — and OOM-crashes the Adreno
         * KGSL allocator mid-combat (the other consumers all respect the
         * budget; the texture cache did not). Textures are the primary
         * legitimate GPU consumer, so give them the largest slice (~45% of the
         * renderer budget), floored at 192 MB so small-budget devices keep a
         * usable working set. The LRU sheds the cold tail to stay under this;
         * the live working set (bound/in-flight) is never evicted.
         */
        size_t tex_bytes = budget * 45 / 100;
        if (tex_bytes < 192 * mib) {
            tex_bytes = 192 * mib;
        }
        b.texture_cache_bytes_max = tex_bytes;
    }

    return b;
}

static const char *const buffer_names[BUFFER_COUNT] = {
    "BUFFER_STAGING_DST",
    "BUFFER_STAGING_SRC",
    "BUFFER_COMPUTE_DST",
    "BUFFER_COMPUTE_SRC",
    "BUFFER_INDEX",
    "BUFFER_INDEX_STAGING",
    "BUFFER_VERTEX_RAM",
    "BUFFER_VERTEX_INLINE",
    "BUFFER_VERTEX_INLINE_STAGING",
    "BUFFER_UNIFORM",
    "BUFFER_UNIFORM_STAGING",
};

static bool create_buffer(PGRAPHState *pg, StorageBuffer *buffer,
                          const char *name, Error **errp)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    VkBufferCreateInfo buffer_create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buffer->buffer_size,
        .usage = buffer->usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult result = vmaCreateBuffer(r->allocator, &buffer_create_info,
                                      &buffer->alloc_info, &buffer->buffer,
                                      &buffer->allocation, NULL);
    if (result != VK_SUCCESS) {
        error_setg(errp, "Failed to create Vulkan buffer %s (%zu bytes): %d",
                   name, buffer->buffer_size, result);
        return false;
    }
    return true;
}

static void destroy_buffer(PGRAPHState *pg, StorageBuffer *buffer)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    if (buffer->buffer == VK_NULL_HANDLE && buffer->allocation == VK_NULL_HANDLE) {
        return;
    }
    vmaDestroyBuffer(r->allocator, buffer->buffer, buffer->allocation);
    buffer->buffer = VK_NULL_HANDLE;
    buffer->allocation = VK_NULL_HANDLE;
}

bool pgraph_vk_init_buffers(NV2AState *d, Error **errp)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHVkState *r = pg->vk_renderer_state;

    const size_t mib = 1024 * 1024;
    size_t vram_size = memory_region_size(d->vram);

    MemoryBudget mb = compute_memory_budget(r);
    r->shader_module_cache_target = mb.shader_module_cache_entries;
    r->texture_cache_target = mb.texture_cache_entries;
    r->image_pool_max = mb.image_pool_max;
    r->surface_image_pool_max = mb.surface_image_pool_max;
    r->surface_image_pool_bytes_max = mb.surface_image_pool_bytes_max;
    r->texture_cache_bytes_max = mb.texture_cache_bytes_max;
    /* On-device tuning without a rebuild: X1BOX_TEX_CACHE_MB overrides the
     * texture-cache byte cap (MB). 0/unset keeps the computed default. */
    {
        const char *tex_mb_env = getenv("X1BOX_TEX_CACHE_MB");
        if (tex_mb_env && *tex_mb_env) {
            long v = strtol(tex_mb_env, NULL, 10);
            if (v > 0) {
                r->texture_cache_bytes_max = (size_t)v * mib;
            }
        }
    }

    VK_LOG_ERROR("memory_budget: total_heap=%zuMB budget=%s%zuMB "
                 "vtx_inline_cap=%zuMB index_cap=%zuMB staging_cap=%zuMB "
                 "pf_vtx=%zuMB pf_idx=%zuMB pf_uni=%zuMB pf_stg=%zuMB "
                 "shader_cache=%zu tex_cache=%zu img_pool=%d "
                 "surf_pool=%d surf_pool_bytes=%zuMB tex_cache_bytes=%zuMB",
                 mb.total_heap >> 20,
                 mb.renderer_budget == SIZE_MAX ? "uncapped/" : "",
                 mb.renderer_budget == SIZE_MAX ? 0 : mb.renderer_budget >> 20,
                 mb.vertex_inline_cap == SIZE_MAX ? 0 : mb.vertex_inline_cap >> 20,
                 mb.index_cap == SIZE_MAX ? 0 : mb.index_cap >> 20,
                 mb.staging_cap == SIZE_MAX ? 0 : mb.staging_cap >> 20,
                 mb.perframe_vtx_cap == SIZE_MAX ? 0 : mb.perframe_vtx_cap >> 20,
                 mb.perframe_idx_cap == SIZE_MAX ? 0 : mb.perframe_idx_cap >> 20,
                 mb.perframe_uni_cap == SIZE_MAX ? 0 : mb.perframe_uni_cap >> 20,
                 mb.perframe_stg_cap == SIZE_MAX ? 0 : mb.perframe_stg_cap >> 20,
                 mb.shader_module_cache_entries,
                 mb.texture_cache_entries,
                 mb.image_pool_max,
                 mb.surface_image_pool_max,
                 mb.surface_image_pool_bytes_max >> 20,
                 r->texture_cache_bytes_max >> 20);

    /* Xbox had 64 MB total UMA; no game stages more than 1× VRAM in flight
     * because the guest can't address that much. On Mali UMA (Android) the
     * persistent host-visible staging pair was the largest persistent
     * allocation we couldn't justify — 2× vram_size = 128 MB × 2 buffers =
     * 256 MB of zram-eligible anon memory. Cap at vram_size = 64 MB × 2
     * = 128 MB total. Saves ~128 MB of RSS pressure and is still a
     * full-frame worth of staging, which the existing code paths never
     * exceed. See memory note [project-xbox-uma-buffer-sizing].
     *
     * Desktop hosts keep the 2× headroom — they pay zero memory pressure
     * for it and benefit from the larger window when batched uploads are
     * fat (e.g. surface_scale=3 + bindless textures). */
#ifdef __ANDROID__
    size_t staging_size = vram_size;
#else
    size_t staging_size = vram_size * 2;
#endif
    if (staging_size < (32 * mib)) {
        staging_size = 32 * mib;
    }
    staging_size = MIN(staging_size, mb.staging_cap);

    size_t compute_size = vram_size * 2;
    if (compute_size < (64 * mib)) {
        compute_size = 64 * mib;
    }
#ifdef __ANDROID__
    if (compute_size > (64 * mib)) {
        compute_size = 64 * mib;
    }
#else
    if (compute_size > (256 * mib)) {
        compute_size = 256 * mib;
    }
#endif

    size_t index_size = sizeof(pg->inline_elements) * 100;
    index_size = MIN(index_size, mb.index_cap);

    size_t vertex_inline_size = NV2A_VERTEXSHADER_ATTRIBUTES *
                                NV2A_MAX_BATCH_LENGTH *
                                4 * sizeof(float) * 10;
    vertex_inline_size = MIN(vertex_inline_size, mb.vertex_inline_cap);

    VK_LOG("buffer_init: vram=%zu staging=%zu compute=%zu",
           vram_size, staging_size, compute_size);

    VmaAllocationCreateInfo host_alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    };
    /*
     * Write-combined streaming upload for the append-only staging buffers
     * (GPU_QUIRK_WC_STREAM_UPLOAD). SEQUENTIAL_WRITE lets VMA pick uncached /
     * write-combined host memory: CPU writes (pgraph_vk_append_to_buffer is a
     * pure forward memcpy) stream straight to RAM, the explicit vmaFlush in
     * sync_staging_buffer no-ops, and the GPU reads via the existing
     * HOST->TRANSFER barrier. ONLY valid for buffers the CPU never reads back —
     * so NOT STAGING_DST (surface readback) or vertex_ram (frame-propagation
     * memcpy reads it). Falls back to the cached RANDOM info when the quirk is
     * off (e.g. a future driver that streams WC poorly).
     */
    VmaAllocationCreateInfo host_wc_alloc_create_info = host_alloc_create_info;
    if (r->gpu_quirks & GPU_QUIRK_WC_STREAM_UPLOAD) {
        host_wc_alloc_create_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }
    VmaAllocationCreateInfo device_alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .flags = 0,
    };

    r->storage_buffers[BUFFER_STAGING_DST] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .buffer_size = staging_size,
    };

    r->storage_buffers[BUFFER_STAGING_SRC] = (StorageBuffer){
        .alloc_info = host_wc_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = staging_size,
    };

    r->storage_buffers[BUFFER_COMPUTE_DST] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .buffer_size = compute_size,
    };

    r->storage_buffers[BUFFER_COMPUTE_SRC] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .buffer_size = compute_size,
    };

    r->storage_buffers[BUFFER_INDEX] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        .buffer_size = index_size,
    };

    r->storage_buffers[BUFFER_INDEX_STAGING] = (StorageBuffer){
        .alloc_info = host_wc_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = index_size,
    };

    /* On Mali stock driver, vertex_ram is bound directly via
     * vkCmdBindVertexBuffers (skipping has_space_for's reservation logic).
     * Mali's tile-binner speculatively prefetches several cache lines past
     * the last vertex of a draw; without padding the buffer end, that
     * prefetch walks off the buffer's mapped pages and faults, surfacing
     * as VK_ERROR_DEVICE_LOST on the NEXT submit (which is why the kcpu
     * fence stalls 3 s before the validator can complain). Over-allocate
     * by MALI_DESCRIPTOR_TAIL_RESERVATION so prefetch always lands on a
     * mapped page. The extra bytes are never written; cost is ~1 MiB. */
    r->storage_buffers[BUFFER_VERTEX_RAM] = (StorageBuffer){
        .alloc_info = host_alloc_create_info,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .buffer_size = memory_region_size(d->vram)
                     + ((r->gpu_quirks & GPU_QUIRK_DESCRIPTOR_TAIL_PAD)
                            ? MALI_DESCRIPTOR_TAIL_RESERVATION : 0),
    };

    r->bitmap_size = memory_region_size(d->vram) / 4096;
    r->uploaded_bitmap = bitmap_new(r->bitmap_size);
    if (!r->uploaded_bitmap) {
        error_setg(errp, "Failed to allocate uploaded surface bitmap");
        return false;
    }
    bitmap_clear(r->uploaded_bitmap, 0, r->bitmap_size);
    r->vertex_ram_flush_min = VK_WHOLE_SIZE;
    r->vertex_ram_flush_max = 0;

    r->storage_buffers[BUFFER_VERTEX_INLINE] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .buffer_size = vertex_inline_size,
    };

    r->storage_buffers[BUFFER_VERTEX_INLINE_STAGING] = (StorageBuffer){
        .alloc_info = host_wc_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = vertex_inline_size,
    };

    extern int xemu_get_submit_frames(void);
    int nframes = xemu_get_submit_frames();

    size_t uniform_size;
    if (nframes >= 3)      uniform_size = 128 * mib;
    else if (nframes == 2) uniform_size = 64 * mib;
    else                   uniform_size = 32 * mib;

    r->storage_buffers[BUFFER_UNIFORM] = (StorageBuffer){
        .alloc_info = device_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .buffer_size = uniform_size,
    };

    r->storage_buffers[BUFFER_UNIFORM_STAGING] = (StorageBuffer){
        .alloc_info = host_wc_alloc_create_info,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .buffer_size = uniform_size,
    };

    for (int i = 0; i < BUFFER_COUNT; i++) {
        VK_LOG("buffer_init: create %s size=%zu",
               buffer_names[i], r->storage_buffers[i].buffer_size);
        if (!create_buffer(pg, &r->storage_buffers[i], buffer_names[i], errp)) {
            VK_LOG_ERROR("buffer_init: create %s FAILED", buffer_names[i]);
            goto fail;
        }
    }

    // FIXME: Add fallback path for device using host mapped memory

    int buffers_to_map[] = { BUFFER_VERTEX_RAM,
                             BUFFER_INDEX_STAGING,
                             BUFFER_VERTEX_INLINE_STAGING,
                             BUFFER_UNIFORM_STAGING,
                             BUFFER_STAGING_SRC,
                             BUFFER_STAGING_DST };

    for (int i = 0; i < ARRAY_SIZE(buffers_to_map); i++) {
        int idx = buffers_to_map[i];
        VK_LOG("buffer_init: map %s", buffer_names[idx]);
        VkResult result = vmaMapMemory(
            r->allocator, r->storage_buffers[idx].allocation,
            (void **)&r->storage_buffers[idx].mapped);
        if (result != VK_SUCCESS) {
            VK_LOG_ERROR("buffer_init: map %s FAILED: %d",
                         buffer_names[idx], result);
            error_setg(errp, "Failed to map Vulkan buffer %s (%zu bytes): %d",
                       buffer_names[idx], r->storage_buffers[idx].buffer_size,
                       result);
            goto fail;
        }
    }

    size_t idx_max, vtx_max, uni_max, stg_max;
    if (nframes >= 3) {
        idx_max = 32 * mib;
        vtx_max = 128 * mib;
        uni_max = 128 * mib;
        stg_max = 256 * mib;
    } else if (nframes == 2) {
        idx_max = 16 * mib;
        vtx_max = 64 * mib;
        uni_max = 64 * mib;
        stg_max = 128 * mib;
    } else {
        idx_max = 8 * mib;
        vtx_max = 32 * mib;
        uni_max = 32 * mib;
        stg_max = 64 * mib;
    }

    idx_max = MIN(idx_max, mb.perframe_idx_cap);
    vtx_max = MIN(vtx_max, mb.perframe_vtx_cap);
    uni_max = MIN(uni_max, mb.perframe_uni_cap);
    stg_max = MIN(stg_max, mb.perframe_stg_cap);

    for (int i = 0; i < NUM_SUBMIT_FRAMES; i++) {
        FrameStagingState *fs = &r->frame_staging[i];

        size_t idx_cap = MIN(r->storage_buffers[BUFFER_INDEX].buffer_size,
                             idx_max);
        size_t vtx_cap = MIN(r->storage_buffers[BUFFER_VERTEX_INLINE].buffer_size,
                             vtx_max);
        size_t uni_cap = MIN(r->storage_buffers[BUFFER_UNIFORM].buffer_size,
                             uni_max);
        size_t stg_cap = MIN(r->storage_buffers[BUFFER_STAGING_SRC].buffer_size,
                             stg_max);

        fs->index_staging = (StorageBuffer){
            .alloc_info = host_wc_alloc_create_info,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .buffer_size = idx_cap,
        };
        fs->vertex_inline_staging = (StorageBuffer){
            .alloc_info = host_wc_alloc_create_info,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .buffer_size = vtx_cap,
        };
        fs->uniform_staging = (StorageBuffer){
            .alloc_info = host_wc_alloc_create_info,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .buffer_size = uni_cap,
        };
        fs->staging_src = (StorageBuffer){
            .alloc_info = host_wc_alloc_create_info,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .buffer_size = stg_cap,
        };
        /* Per-frame vertex_ram: same Mali prefetch over-allocation rule
         * as the shared BUFFER_VERTEX_RAM above (see comment there). */
        fs->vertex_ram = (StorageBuffer){
            .alloc_info = host_alloc_create_info,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .buffer_size = memory_region_size(d->vram)
                         + ((r->gpu_quirks & GPU_QUIRK_DESCRIPTOR_TAIL_PAD)
                            ? MALI_DESCRIPTOR_TAIL_RESERVATION : 0),
        };
        fs->vertex_ram_flush_min = VK_WHOLE_SIZE;
        fs->vertex_ram_flush_max = 0;
        fs->vertex_ram_propagate_min = VK_WHOLE_SIZE;
        fs->vertex_ram_propagate_max = 0;
        fs->vertex_ram_initialized = false;

        char name[64];
        StorageBuffer *bufs[] = {
            &fs->index_staging, &fs->vertex_inline_staging,
            &fs->uniform_staging, &fs->staging_src, &fs->vertex_ram
        };
        const char *names[] = {
            "INDEX_STAGING", "VTXINLINE_STAGING", "UNIFORM_STAGING",
            "STAGING_SRC", "VERTEX_RAM"
        };
        for (int j = 0; j < ARRAY_SIZE(bufs); j++) {
            snprintf(name, sizeof(name), "FRAME%d_%s", i, names[j]);
            VK_LOG_ERROR("buffer_init: create %s size=%zu", name, bufs[j]->buffer_size);
            if (!create_buffer(pg, bufs[j], name, errp)) {
                goto fail;
            }
            VkResult res = vmaMapMemory(r->allocator, bufs[j]->allocation,
                                        (void **)&bufs[j]->mapped);
            if (res != VK_SUCCESS) {
                error_setg(errp, "Failed to map per-frame buffer %s: %d",
                           name, res);
                goto fail;
            }
        }

        fs->uploaded_bitmap = bitmap_new(r->bitmap_size);
        if (!fs->uploaded_bitmap) {
            error_setg(errp, "Failed to allocate per-frame uploaded bitmap");
            goto fail;
        }
        bitmap_clear(fs->uploaded_bitmap, 0, r->bitmap_size);
    }
    VK_LOG_ERROR("buffer_init: per-frame staging created (%d frames, "
                 "idx=%zuMB vtx=%zuMB uni=%zuMB stg=%zuMB per-frame)",
                 nframes, idx_max >> 20, vtx_max >> 20, uni_max >> 20,
                 stg_max >> 20);

    pgraph_prim_rewrite_init(&r->prim_rewrite_buf);
    pgraph_prim_rewrite_cache_init(&r->index_cache);

    r->draw_queue.index_buf = g_malloc0(INDEX_QUEUE_MAX * sizeof(uint32_t));

    return true;

fail:
    for (int i = 0; i < NUM_SUBMIT_FRAMES; i++) {
        FrameStagingState *fs = &r->frame_staging[i];
        StorageBuffer *bufs[] = {
            &fs->index_staging, &fs->vertex_inline_staging,
            &fs->uniform_staging, &fs->staging_src, &fs->vertex_ram
        };
        for (int j = 0; j < ARRAY_SIZE(bufs); j++) {
            if (bufs[j]->mapped) {
                vmaUnmapMemory(r->allocator, bufs[j]->allocation);
                bufs[j]->mapped = NULL;
            }
            destroy_buffer(pg, bufs[j]);
        }
        g_free(fs->uploaded_bitmap);
        fs->uploaded_bitmap = NULL;
    }
    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (r->storage_buffers[i].mapped) {
            vmaUnmapMemory(r->allocator, r->storage_buffers[i].allocation);
            r->storage_buffers[i].mapped = NULL;
        }
        destroy_buffer(pg, &r->storage_buffers[i]);
    }
    g_free(r->uploaded_bitmap);
    r->uploaded_bitmap = NULL;
    r->bitmap_size = 0;
    return false;
}

void pgraph_vk_finalize_buffers(NV2AState *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHVkState *r = pg->vk_renderer_state;

    for (int i = 0; i < NUM_SUBMIT_FRAMES; i++) {
        FrameStagingState *fs = &r->frame_staging[i];
        StorageBuffer *bufs[] = {
            &fs->index_staging, &fs->vertex_inline_staging,
            &fs->uniform_staging, &fs->staging_src, &fs->vertex_ram
        };
        for (int j = 0; j < ARRAY_SIZE(bufs); j++) {
            if (bufs[j]->mapped) {
                vmaUnmapMemory(r->allocator, bufs[j]->allocation);
            }
            destroy_buffer(pg, bufs[j]);
        }
        g_free(fs->uploaded_bitmap);
        fs->uploaded_bitmap = NULL;
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (r->storage_buffers[i].mapped) {
            vmaUnmapMemory(r->allocator, r->storage_buffers[i].allocation);
        }
        destroy_buffer(pg, &r->storage_buffers[i]);
    }

    pgraph_prim_rewrite_finalize(&r->prim_rewrite_buf);
    pgraph_prim_rewrite_cache_finalize(&r->index_cache);

    g_free(r->draw_queue.index_buf);
    r->draw_queue.index_buf = NULL;

    g_free(r->uploaded_bitmap);
    r->uploaded_bitmap = NULL;
}

/*
 * Tail reservation kept at the end of every descriptor-bound buffer on Mali.
 * Mali's stock driver speculatively prefetches several cache lines past the
 * end of the bound descriptor / vertex / index range; without this
 * reservation, bindings that land close to the buffer end have prefetch
 * walk off the buffer's mapped pages and fault, surfacing as
 * VK_ERROR_DEVICE_LOST on the NEXT submit (which is why the crash signature
 * looks unrelated to any specific draw).
 *
 * Two complementary applications of this constant:
 *   1. pgraph_vk_buffer_has_space_for() reports the staging buffers as
 *      MALI_DESCRIPTOR_TAIL_RESERVATION bytes smaller than they really are,
 *      so any UBO/vertex-inline/index ring allocation wraps early and the
 *      mirrored device-local bind never reaches the buffer's last page.
 *   2. fs->vertex_ram is OVER-ALLOCATED by MALI_DESCRIPTOR_TAIL_RESERVATION
 *      (see buffer_init below) since that buffer is bound directly without
 *      going through has_space_for. The extra tail is never written but
 *      ensures Mali prefetch off the last vertex always lands on a mapped
 *      page.
 *
 * 1 MiB is overkill but cheap (1 MiB on a 32+ MiB ring is <3% overhead) and
 * gives runway for any future Mali driver that prefetches further ahead than
 * current observed behaviour.
 *
 * The macro itself is defined at the top of this file so both uses see it.
 */

bool pgraph_vk_buffer_has_space_for(PGRAPHState *pg, int index,
                                    VkDeviceSize size,
                                    VkDeviceAddress alignment)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer *b = get_staging_buffer(r, index);
    VkDeviceSize usable = b->buffer_size;
    if ((r->gpu_quirks & GPU_QUIRK_DESCRIPTOR_TAIL_PAD) &&
        usable > MALI_DESCRIPTOR_TAIL_RESERVATION) {
        usable -= MALI_DESCRIPTOR_TAIL_RESERVATION;
    }
    return (ROUND_UP(b->buffer_offset, alignment) + size) <= usable;
}

VkDeviceSize pgraph_vk_append_to_buffer(PGRAPHState *pg, int index, void **data,
                                        VkDeviceSize *sizes, size_t count,
                                        VkDeviceAddress alignment)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    VkDeviceSize total_size = 0;
    for (int i = 0; i < count; i++) {
        total_size += sizes[i];
    }
    assert(pgraph_vk_buffer_has_space_for(pg, index, total_size, alignment));

    StorageBuffer *b = get_staging_buffer(r, index);
    VkDeviceSize starting_offset = ROUND_UP(b->buffer_offset, alignment);

    assert(b->mapped);

    for (int i = 0; i < count; i++) {
        b->buffer_offset = ROUND_UP(b->buffer_offset, alignment);
        memcpy(b->mapped + b->buffer_offset, data[i], sizes[i]);
        b->buffer_offset += sizes[i];
    }

    return starting_offset;
}

VkDeviceSize pgraph_vk_staging_alloc(PGRAPHState *pg, VkDeviceSize size)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    StorageBuffer *b = get_staging_buffer(r, BUFFER_STAGING_SRC);
    VkDeviceSize offset = ROUND_UP(b->buffer_offset, 16);
    if (offset + size > b->buffer_size) {
        return VK_WHOLE_SIZE;
    }
    b->buffer_offset = offset + size;
    return offset;
}

void pgraph_vk_staging_reset(PGRAPHState *pg)
{
    PGRAPHVkState *r = pg->vk_renderer_state;
    get_staging_buffer(r, BUFFER_STAGING_SRC)->buffer_offset = 0;
}

bool pgraph_vk_staging_reclaim_any(PGRAPHState *pg)
{
    PGRAPHVkState *r = pg->vk_renderer_state;

    for (int i = 0; i < r->num_active_frames; i++) {
        if (i == r->current_frame) {
            continue;
        }
        if (!qatomic_read(&r->frame_submitted[i])) {
            continue;
        }
        VkResult result = vkGetFenceStatus(r->device, r->frame_fences[i]);
        if (result == VK_SUCCESS) {
            r->frame_staging[i].staging_src.buffer_offset = 0;
            r->frame_staging[i].index_staging.buffer_offset = 0;
            r->frame_staging[i].vertex_inline_staging.buffer_offset = 0;
            r->frame_staging[i].uniform_staging.buffer_offset = 0;
            return true;
        }
    }
    return false;
}
