/*
 * Generic Vulkan probe runner. See probe_runner.h for the wire format.
 *
 * Lifecycle: at pgraph_vk_init time, look for probe_req.bin under
 * <ext>/x1box/gpu_probe/. If present, validate header, compile the
 * user's SPIR-V into a compute or graphics pipeline, dispatch/draw,
 * read back N uint32 outputs into probe_out.bin, drop probe_done.flag
 * as a sentinel, and rename the request to probe_req.consumed so
 * relaunches don't re-run a stale request.
 *
 * Pairs with the gpu_probe() MCP tool which writes the request,
 * triggers the emulator, and parses the binary output.
 *
 * Vertex shader for fragment-stage probes is reused from the existing
 * NaN probe — a 3-vertex fullscreen triangle, no descriptors.
 */

#include "qemu/osdep.h"
#include "renderer.h"
#include "probe_runner.h"
#include "qemu/android-paths.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

/* Reuse the existing fullscreen-triangle VS from the NaN probe. The .inc
 * defines these symbols with external linkage, so nan_probe.c is the sole
 * translation unit that #includes it; we just extern them here. */
extern unsigned char nan_probe_vert_spv[];
extern unsigned int  nan_probe_vert_spv_len;

/* Probe paths under <x1box_ext>/gpu_probe/. Resolved per-package at
 * runtime via android_x1box_ext_dir() so the perftest flavor reads
 * its own data dir; helpers below produce full paths into a caller-
 * supplied buffer. */
static int probe_path(char *out, size_t out_sz, const char *leaf)
{
    const char *base = android_x1box_ext_dir();
    if (!base) return -1;
    return snprintf(out, out_sz, "%s/gpu_probe/%s", base, leaf);
}

#define LOG_TAG "hakuX-gpu-probe"

#ifdef __ANDROID__
#define LOGW(fmt, ...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##__VA_ARGS__)
#else
#define LOGW(fmt, ...) ((void)0)
#define LOGE(fmt, ...) ((void)0)
#endif

/* --------------------------------------------------------------------- */
/* Request loading + validation                                          */
/* --------------------------------------------------------------------- */

struct probe_request {
    struct x1b_probe_req hdr;
    void   *spv;         /* malloc'd; freed by caller */
};

static bool load_request(struct probe_request *out, char *err, size_t err_sz)
{
    char req_path[512];
    if (probe_path(req_path, sizeof(req_path), "probe_req.bin") < 0) {
        return false;
    }
    FILE *f = fopen(req_path, "rb");
    if (!f) {
        return false;  /* missing — silent skip */
    }
    bool ok = false;
    if (fread(&out->hdr, sizeof(out->hdr), 1, f) != 1) {
        snprintf(err, err_sz, "short header read");
        goto done;
    }
    if (memcmp(out->hdr.magic, X1B_PROBE_REQ_MAGIC, 4) != 0) {
        snprintf(err, err_sz, "bad magic: %02x%02x%02x%02x",
                 (unsigned char)out->hdr.magic[0], (unsigned char)out->hdr.magic[1],
                 (unsigned char)out->hdr.magic[2], (unsigned char)out->hdr.magic[3]);
        goto done;
    }
    if (out->hdr.version != X1B_PROBE_VERSION) {
        snprintf(err, err_sz, "version mismatch: got %u, expected %u",
                 out->hdr.version, X1B_PROBE_VERSION);
        goto done;
    }
    if (out->hdr.stage != X1B_PROBE_STAGE_COMPUTE &&
        out->hdr.stage != X1B_PROBE_STAGE_FRAGMENT) {
        snprintf(err, err_sz, "invalid stage: %u", out->hdr.stage);
        goto done;
    }
    if (out->hdr.n_outputs == 0 || out->hdr.n_outputs > X1B_PROBE_MAX_OUTPUTS) {
        snprintf(err, err_sz, "n_outputs out of range: %u", out->hdr.n_outputs);
        goto done;
    }
    if (out->hdr.n_push_floats > X1B_PROBE_MAX_PUSH_FLOATS) {
        snprintf(err, err_sz, "n_push_floats out of range: %u",
                 out->hdr.n_push_floats);
        goto done;
    }
    if (out->hdr.spv_size == 0 || (out->hdr.spv_size % 4) != 0
        || out->hdr.spv_size > X1B_PROBE_MAX_SPV_BYTES) {
        snprintf(err, err_sz, "spv_size invalid: %u", out->hdr.spv_size);
        goto done;
    }
    out->spv = g_malloc(out->hdr.spv_size);
    if (fread(out->spv, 1, out->hdr.spv_size, f) != out->hdr.spv_size) {
        snprintf(err, err_sz, "short SPIR-V read");
        g_free(out->spv); out->spv = NULL;
        goto done;
    }
    ok = true;
done:
    fclose(f);
    return ok;
}

/* --------------------------------------------------------------------- */
/* Driver-property capture for the output header                         */
/* --------------------------------------------------------------------- */

static void fill_driver_props(PGRAPHVkState *r, struct x1b_probe_out *out)
{
    VkPhysicalDeviceFloatControlsProperties fc = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &fc,
    };
    vkGetPhysicalDeviceProperties2(r->physical_device, &p2);
    strncpy(out->gpu_name, p2.properties.deviceName, sizeof(out->gpu_name) - 1);
    out->gpu_name[sizeof(out->gpu_name) - 1] = '\0';
    out->vendor_id      = p2.properties.vendorID;
    out->device_id      = p2.properties.deviceID;
    out->driver_version = p2.properties.driverVersion;
    out->fp32_signed_zero_inf_nan_preserve =
        fc.shaderSignedZeroInfNanPreserveFloat32;
    out->fp32_denorm_preserve     = fc.shaderDenormPreserveFloat32;
    out->fp32_denorm_flush_to_zero = fc.shaderDenormFlushToZeroFloat32;
    out->fp32_rounding_mode_rte   = fc.shaderRoundingModeRTEFloat32;
    out->fp32_rounding_mode_rtz   = fc.shaderRoundingModeRTZFloat32;
}

/* --------------------------------------------------------------------- */
/* Output serialization                                                  */
/* --------------------------------------------------------------------- */

static void write_output(PGRAPHVkState *r, uint32_t status,
                         uint32_t n_outputs, const uint32_t *outputs,
                         const char *err_msg)
{
    struct x1b_probe_out hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, X1B_PROBE_OUT_MAGIC, 4);
    hdr.version   = X1B_PROBE_VERSION;
    hdr.status    = status;
    hdr.n_outputs = n_outputs;
    fill_driver_props(r, &hdr);
    size_t err_len = (status != X1B_PROBE_OK && err_msg) ? strlen(err_msg) : 0;
    hdr.error_msg_size = (uint32_t)err_len;

    /* Write atomically: temp file then rename. */
    char out_path[512], tmp_path[520], done_path[512];
    if (probe_path(out_path, sizeof(out_path), "probe_out.bin") < 0 ||
        probe_path(done_path, sizeof(done_path), "probe_done.flag") < 0) {
        LOGE("probe_path failed (android_x1box_ext_dir returned NULL?)");
        return;
    }
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_path);
    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        LOGE("fopen(%s) failed: %s", tmp_path, strerror(errno));
        return;
    }
    fwrite(&hdr, sizeof(hdr), 1, f);
    if (n_outputs > 0 && outputs) {
        fwrite(outputs, sizeof(uint32_t), n_outputs, f);
    }
    if (err_len > 0) {
        fwrite(err_msg, 1, err_len, f);
    }
    fclose(f);
    if (rename(tmp_path, out_path) != 0) {
        LOGE("rename(%s -> %s) failed: %s", tmp_path, out_path,
             strerror(errno));
        return;
    }

    /* Sentinel — touched only after probe_out.bin is fully written. */
    FILE *sf = fopen(done_path, "wb");
    if (sf) fclose(sf);
}

/* --------------------------------------------------------------------- */
/* Compute-stage probe                                                   */
/* --------------------------------------------------------------------- */

#ifdef __ANDROID__
static VkCommandBuffer begin_oneshot_cb(PGRAPHVkState *r, VkCommandPool *out_pool)
{
    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = r->queue_family,
    };
    if (vkCreateCommandPool(r->device, &pool_ci, NULL, out_pool) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = *out_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    if (vkAllocateCommandBuffers(r->device, &cb_ai, &cb) != VK_SUCCESS) {
        vkDestroyCommandPool(r->device, *out_pool, NULL);
        *out_pool = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static void submit_and_wait(PGRAPHVkState *r, VkCommandBuffer cb)
{
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    vkQueueSubmit(r->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(r->queue);
}

static uint32_t run_compute(PGRAPHVkState *r,
                            const struct x1b_probe_req *req,
                            const void *spv,
                            uint32_t *out_results,
                            char *err, size_t err_sz)
{
    VkResult vr;
    uint32_t status = X1B_PROBE_ERR_VK_RESOURCE;
    const VkDeviceSize buf_size = req->n_outputs * sizeof(uint32_t);

    VkBuffer       buffer = VK_NULL_HANDLE;
    VmaAllocation  buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo buf_info;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      pll = VK_NULL_HANDLE;
    VkShaderModule        sm  = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkDescriptorPool      dp  = VK_NULL_HANDLE;
    VkCommandPool         pool = VK_NULL_HANDLE;

    {
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buf_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo aci = {
            .usage = VMA_MEMORY_USAGE_AUTO,
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
        vr = vmaCreateBuffer(r->allocator, &bci, &aci,
                             &buffer, &buf_alloc, &buf_info);
        if (vr != VK_SUCCESS) {
            snprintf(err, err_sz, "vmaCreateBuffer compute-out: VkResult=%d", vr);
            goto out;
        }
        memset(buf_info.pMappedData, 0xAA, buf_size);
    }
    {
        VkDescriptorSetLayoutBinding b = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        VkDescriptorSetLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &b,
        };
        if (vkCreateDescriptorSetLayout(r->device, &ci, NULL, &dsl) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateDescriptorSetLayout failed");
            goto out;
        }
    }
    {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(float) * X1B_PROBE_MAX_PUSH_FLOATS,
        };
        VkPipelineLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &dsl,
            .pushConstantRangeCount = req->n_push_floats > 0 ? 1 : 0,
            .pPushConstantRanges = &pcr,
        };
        if (vkCreatePipelineLayout(r->device, &ci, NULL, &pll) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreatePipelineLayout failed");
            goto out;
        }
    }
    {
        VkShaderModuleCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = req->spv_size,
            .pCode = (const uint32_t *)spv,
        };
        if (vkCreateShaderModule(r->device, &ci, NULL, &sm) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateShaderModule compute failed");
            goto out;
        }
    }
    {
        VkComputePipelineCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = (VkPipelineShaderStageCreateInfo){
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm, .pName = "main",
            },
            .layout = pll,
        };
        if (vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 1, &ci,
                                     NULL, &pipeline) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateComputePipelines failed");
            goto out;
        }
    }
    {
        VkDescriptorPoolSize sz = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
        };
        VkDescriptorPoolCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &sz,
        };
        if (vkCreateDescriptorPool(r->device, &ci, NULL, &dp) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateDescriptorPool failed");
            goto out;
        }
    }
    VkDescriptorSet ds;
    {
        VkDescriptorSetAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
        };
        if (vkAllocateDescriptorSets(r->device, &ai, &ds) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkAllocateDescriptorSets failed");
            goto out;
        }
        VkDescriptorBufferInfo dbi = {
            .buffer = buffer, .offset = 0, .range = buf_size,
        };
        VkWriteDescriptorSet wds = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi,
        };
        vkUpdateDescriptorSets(r->device, 1, &wds, 0, NULL);
    }

    VkCommandBuffer cb = begin_oneshot_cb(r, &pool);
    if (cb == VK_NULL_HANDLE) {
        snprintf(err, err_sz, "command buffer allocation failed");
        goto out;
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pll, 0, 1, &ds, 0, NULL);
    if (req->n_push_floats > 0) {
        vkCmdPushConstants(cb, pll, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, req->n_push_floats * sizeof(float),
                           req->push_floats);
    }
    vkCmdDispatch(cb, 1, 1, 1);
    submit_and_wait(r, cb);

    memcpy(out_results, buf_info.pMappedData, buf_size);
    status = X1B_PROBE_OK;

out:
    if (pool)     vkDestroyCommandPool(r->device, pool, NULL);
    if (dp)       vkDestroyDescriptorPool(r->device, dp, NULL);
    if (pipeline) vkDestroyPipeline(r->device, pipeline, NULL);
    if (sm)       vkDestroyShaderModule(r->device, sm, NULL);
    if (pll)      vkDestroyPipelineLayout(r->device, pll, NULL);
    if (dsl)      vkDestroyDescriptorSetLayout(r->device, dsl, NULL);
    if (buffer)   vmaDestroyBuffer(r->allocator, buffer, buf_alloc);
    return status;
}

/* --------------------------------------------------------------------- */
/* Fragment-stage probe                                                  */
/* --------------------------------------------------------------------- */

static uint32_t run_fragment(PGRAPHVkState *r,
                             const struct x1b_probe_req *req,
                             const void *spv,
                             uint32_t *out_results,
                             char *err, size_t err_sz)
{
    VkResult vr;
    uint32_t status = X1B_PROBE_ERR_VK_RESOURCE;
    const uint32_t width  = req->n_outputs;
    const uint32_t height = 1;
    const VkFormat fmt    = VK_FORMAT_R32_UINT;
    const VkDeviceSize readback_size = width * sizeof(uint32_t);

    VkImage         image = VK_NULL_HANDLE;
    VmaAllocation   image_alloc = VK_NULL_HANDLE;
    VkImageView     image_view = VK_NULL_HANDLE;
    VkBuffer        readback = VK_NULL_HANDLE;
    VmaAllocation   readback_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo readback_info;
    VkRenderPass    rp = VK_NULL_HANDLE;
    VkFramebuffer   fb = VK_NULL_HANDLE;
    VkShaderModule  vsm = VK_NULL_HANDLE;
    VkShaderModule  fsm = VK_NULL_HANDLE;
    VkPipelineLayout pll = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkCommandPool    pool = VK_NULL_HANDLE;

    {
        VkImageCreateInfo ici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = fmt,
            .extent = { width, height, 1 },
            .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                   | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo aci = { .usage = VMA_MEMORY_USAGE_AUTO };
        vr = vmaCreateImage(r->allocator, &ici, &aci, &image, &image_alloc, NULL);
        if (vr != VK_SUCCESS) {
            snprintf(err, err_sz, "vmaCreateImage frag-out: VkResult=%d", vr);
            goto out;
        }
    }
    {
        VkImageViewCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = fmt,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0, .levelCount = 1,
                .baseArrayLayer = 0, .layerCount = 1,
            },
        };
        if (vkCreateImageView(r->device, &ci, NULL, &image_view) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateImageView failed");
            goto out;
        }
    }
    {
        VkBufferCreateInfo bci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = readback_size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo aci = {
            .usage = VMA_MEMORY_USAGE_AUTO,
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
        vr = vmaCreateBuffer(r->allocator, &bci, &aci, &readback,
                             &readback_alloc, &readback_info);
        if (vr != VK_SUCCESS) {
            snprintf(err, err_sz, "vmaCreateBuffer readback: VkResult=%d", vr);
            goto out;
        }
        memset(readback_info.pMappedData, 0xAA, readback_size);
    }
    {
        VkAttachmentDescription att = {
            .format = fmt,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        };
        VkAttachmentReference color_ref = {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        VkSubpassDescription sub = {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_ref,
        };
        VkRenderPassCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &att,
            .subpassCount = 1, .pSubpasses = &sub,
        };
        if (vkCreateRenderPass(r->device, &ci, NULL, &rp) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateRenderPass failed");
            goto out;
        }
    }
    {
        VkFramebufferCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = rp,
            .attachmentCount = 1, .pAttachments = &image_view,
            .width = width, .height = height, .layers = 1,
        };
        if (vkCreateFramebuffer(r->device, &ci, NULL, &fb) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateFramebuffer failed");
            goto out;
        }
    }
    {
        VkShaderModuleCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = nan_probe_vert_spv_len,
            .pCode = (const uint32_t *)nan_probe_vert_spv,
        };
        if (vkCreateShaderModule(r->device, &vci, NULL, &vsm) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateShaderModule vertex failed");
            goto out;
        }
        VkShaderModuleCreateInfo fci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = req->spv_size,
            .pCode = (const uint32_t *)spv,
        };
        if (vkCreateShaderModule(r->device, &fci, NULL, &fsm) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateShaderModule fragment failed");
            goto out;
        }
    }
    {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(float) * X1B_PROBE_MAX_PUSH_FLOATS,
        };
        VkPipelineLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = req->n_push_floats > 0 ? 1 : 0,
            .pPushConstantRanges = &pcr,
        };
        if (vkCreatePipelineLayout(r->device, &ci, NULL, &pll) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreatePipelineLayout failed");
            goto out;
        }
    }
    {
        VkPipelineShaderStageCreateInfo stages[2] = {
            { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_VERTEX_BIT,
              .module = vsm, .pName = "main" },
            { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
              .module = fsm, .pName = "main" },
        };
        VkPipelineVertexInputStateCreateInfo vis = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        };
        VkPipelineInputAssemblyStateCreateInfo ias = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        VkViewport vp = { .x = 0, .y = 0, .width = (float)width, .height = (float)height,
                          .minDepth = 0.0f, .maxDepth = 1.0f };
        VkRect2D sc = { .offset = {0,0}, .extent = { width, height } };
        VkPipelineViewportStateCreateInfo vps = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &vp,
            .scissorCount = 1, .pScissors = &sc,
        };
        VkPipelineRasterizationStateCreateInfo rs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f,
        };
        VkPipelineMultisampleStateCreateInfo ms = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        VkPipelineColorBlendAttachmentState cba = {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT,
        };
        VkPipelineColorBlendStateCreateInfo cbs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &cba,
        };
        VkGraphicsPipelineCreateInfo gpci = {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2, .pStages = stages,
            .pVertexInputState = &vis,
            .pInputAssemblyState = &ias,
            .pViewportState = &vps,
            .pRasterizationState = &rs,
            .pMultisampleState = &ms,
            .pColorBlendState = &cbs,
            .layout = pll, .renderPass = rp, .subpass = 0,
        };
        if (vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &gpci,
                                      NULL, &pipeline) != VK_SUCCESS) {
            snprintf(err, err_sz, "vkCreateGraphicsPipelines failed");
            goto out;
        }
    }

    VkCommandBuffer cb = begin_oneshot_cb(r, &pool);
    if (cb == VK_NULL_HANDLE) {
        snprintf(err, err_sz, "command buffer allocation failed");
        goto out;
    }
    VkRenderPassBeginInfo rpb = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = rp, .framebuffer = fb,
        .renderArea = { .offset = {0,0}, .extent = { width, height } },
    };
    vkCmdBeginRenderPass(cb, &rpb, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (req->n_push_floats > 0) {
        vkCmdPushConstants(cb, pll, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, req->n_push_floats * sizeof(float),
                           req->push_floats);
    }
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);

    VkBufferImageCopy region = {
        .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1,
        },
        .imageOffset = {0,0,0},
        .imageExtent = { width, height, 1 },
    };
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region);

    submit_and_wait(r, cb);

    memcpy(out_results, readback_info.pMappedData, readback_size);
    status = X1B_PROBE_OK;

out:
    if (pool)       vkDestroyCommandPool(r->device, pool, NULL);
    if (pipeline)   vkDestroyPipeline(r->device, pipeline, NULL);
    if (pll)        vkDestroyPipelineLayout(r->device, pll, NULL);
    if (fsm)        vkDestroyShaderModule(r->device, fsm, NULL);
    if (vsm)        vkDestroyShaderModule(r->device, vsm, NULL);
    if (fb)         vkDestroyFramebuffer(r->device, fb, NULL);
    if (rp)         vkDestroyRenderPass(r->device, rp, NULL);
    if (readback)   vmaDestroyBuffer(r->allocator, readback, readback_alloc);
    if (image_view) vkDestroyImageView(r->device, image_view, NULL);
    if (image)      vmaDestroyImage(r->allocator, image, image_alloc);
    return status;
}
#endif /* __ANDROID__ */

/* --------------------------------------------------------------------- */
/* Public entry point                                                    */
/* --------------------------------------------------------------------- */

void pgraph_vk_probe_runner_init(PGRAPHVkState *r)
{
#ifdef __ANDROID__
    /* Resolve paths under <x1box_ext>/gpu_probe/<...> for the running
     * package (per-flavor — debug vs perftest live in different data
     * dirs). */
    const char *base = android_x1box_ext_dir();
    if (!base) return;
    char base_dir[512], req_path[512], out_path[512],
         done_path[512], consumed_path[512];
    snprintf(base_dir,     sizeof(base_dir),      "%s/gpu_probe",          base);
    snprintf(req_path,     sizeof(req_path),      "%s/probe_req.bin",      base_dir);
    snprintf(out_path,     sizeof(out_path),      "%s/probe_out.bin",      base_dir);
    snprintf(done_path,    sizeof(done_path),     "%s/probe_done.flag",    base_dir);
    snprintf(consumed_path,sizeof(consumed_path), "%s/probe_req.consumed", base_dir);

    /* Skip entirely if there's no request file. No staleness check —
     * the MCP renames the file to .consumed after we write the output,
     * so a fresh run requires a fresh probe_req.bin. */
    struct stat st;
    if (stat(req_path, &st) != 0) {
        return;
    }

    /* Make sure parent dir + any leftover sentinel are sane. */
    g_mkdir_with_parents(base_dir, 0755);
    unlink(done_path);

    LOGW("probe request seen at %s — running", req_path);

    char err[512] = {0};
    struct probe_request req = {0};
    if (!load_request(&req, err, sizeof(err))) {
        if (err[0]) {
            LOGE("load_request failed: %s", err);
            write_output(r, X1B_PROBE_ERR_BAD_HEADER, 0, NULL, err);
        }
        rename(req_path, consumed_path);
        return;
    }

    uint32_t *outputs = g_malloc0(req.hdr.n_outputs * sizeof(uint32_t));
    uint32_t status;
    if (req.hdr.stage == X1B_PROBE_STAGE_COMPUTE) {
        status = run_compute(r, &req.hdr, req.spv, outputs, err, sizeof(err));
    } else {
        status = run_fragment(r, &req.hdr, req.spv, outputs, err, sizeof(err));
    }

    if (status == X1B_PROBE_OK) {
        LOGW("probe ok — wrote %u outputs to %s",
             req.hdr.n_outputs, out_path);
        write_output(r, status, req.hdr.n_outputs, outputs, NULL);
    } else {
        LOGE("probe FAILED (status=%u): %s", status, err);
        write_output(r, status, 0, NULL, err);
    }

    g_free(outputs);
    g_free(req.spv);
    rename(req_path, consumed_path);
#else
    (void)r;
#endif
}
