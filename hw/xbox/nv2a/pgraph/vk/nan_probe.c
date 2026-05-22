/*
 * One-shot Vulkan compute + fragment probes for NaN/Inf semantics
 * divergence between Mali and Adreno drivers. Gated behind either
 * $X1BOX_PROBE_NAN=1 or the sentinel file probe_nan.flag in the app's
 * external files dir. Runs once at pgraph_vk_init time and logs each
 * scalar's bit pattern to logcat (tag: hakuX-nan-probe).
 *
 * Background: in project_halo2_mali_specular_white.md the compute-stage
 * probe showed Mali and Adreno produce identical results for clamp/min/
 * max on NaN — disproving the original "Mali saturates to 1.0" theory.
 * The fragment-stage probe rules out the possibility that Mali's
 * graphics scheduler (Bifrost/Valhall fragment path) handles NaN
 * differently from the compute path.
 */

#include "qemu/osdep.h"
#include "renderer.h"
#include "qemu/android-paths.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

/* Compiled SPIR-V blobs, generated via:
 *   glslangValidator -V -S {comp,vert,frag} <src> -o <out>.spv
 *   xxd -i -n <symbol> <out>.spv > <out>.inc
 *
 * The compute & fragment shaders compute the same 24-slot probe array.
 * The vertex shader is a fullscreen-triangle pass-through for the
 * fragment probe. */
#include "nan_probe_spv.inc"        /* nan_probe_spv[]      / _len */
#include "nan_probe_vert_spv.inc"   /* nan_probe_vert_spv[] / _len */
#include "nan_probe_frag_spv.inc"   /* nan_probe_frag_spv[] / _len */

#define NAN_PROBE_SLOTS 24

struct nan_probe_label {
    const char *name;
    bool        as_int;  /* true: print decimal int; false: print float-bit-pattern + decoded float */
};

static const struct nan_probe_label k_labels[NAN_PROBE_SLOTS] = {
    [0]  = { "clamp(NaN, -1, 1)",       false },
    [1]  = { "max(NaN, 0)",             false },
    [2]  = { "max(NaN, -1)",            false },
    [3]  = { "min(NaN, 1)",             false },
    [4]  = { "NaN * NaN",               false },
    [5]  = { "clamp(+Inf, -1, 1)",      false },
    [6]  = { "clamp(-Inf, -1, 1)",      false },
    [7]  = { "NaN == NaN (0=ieee)",     true  },
    [8]  = { "abs(NaN)",                false },
    [9]  = { "-NaN",                    false },
    [10] = { "NaN < 0 (0=ieee)",        true  },
    [11] = { "NaN > 0 (0=ieee)",        true  },
    [12] = { "NaN + 1",                 false },
    [13] = { "isnan(NaN) (1=ieee)",     true  },
    [14] = { "mix(1, 2, NaN)",          false },
    [15] = { "smoothstep(0, 1, NaN)",   false },
    [16] = { "phong chain: NaN^2",      false },
    [17] = { "phong chain: NaN^4",      false },
    [18] = { "phong chain: clamp(^4)",  false },
    [19] = { "phong chain: (clamp)^2",  false },
    [20] = { "clamp(max(NaN,0)*-1,..)", false },
    [21] = { "+Inf - +Inf",             false },
    [22] = { "sqrt(-1)",                false },
    [23] = { "log(-1)",                 false },
};

static const char *classify_bits(uint32_t bits)
{
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t man = bits & 0x7FFFFF;
    if (exp == 0xFF && man != 0) return "NaN";
    if (exp == 0xFF && man == 0) return (bits >> 31) ? "-Inf" : "+Inf";
    if (exp == 0   && man == 0) return (bits >> 31) ? "-0"   : "+0";
    if (exp == 0   && man != 0) return (bits >> 31) ? "-denorm" : "+denorm";
    return "finite";
}

#ifdef __ANDROID__
static void log_probe_results(const char *stage, const uint32_t *out)
{
    __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
        "=== %s probe results (%d slots) ===", stage, NAN_PROBE_SLOTS);
    for (int i = 0; i < NAN_PROBE_SLOTS; i++) {
        uint32_t v = out[i];
        if (k_labels[i].as_int) {
            __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
                "  [%s] probe[%02d] %-30s = %u",
                stage, i, k_labels[i].name, v);
        } else {
            float f;
            memcpy(&f, &v, sizeof(f));
            __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
                "  [%s] probe[%02d] %-30s = 0x%08x  (%s)  %.6g",
                stage, i, k_labels[i].name, v, classify_bits(v), (double)f);
        }
    }
}

static void log_driver_props(PGRAPHVkState *r)
{
    VkPhysicalDeviceFloatControlsProperties fc = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &fc,
    };
    vkGetPhysicalDeviceProperties2(r->physical_device, &p2);
    __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
        "GPU: %s (vendor=0x%04x, device=0x%04x, driver=%u)",
        p2.properties.deviceName,
        p2.properties.vendorID, p2.properties.deviceID,
        p2.properties.driverVersion);
    __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
        "FloatControls.fp32: SZInfNaN preserve=%d, denorm preserve=%d, "
        "denorm flush-to-zero=%d, rounding modes RTE=%d RTZ=%d",
        fc.shaderSignedZeroInfNanPreserveFloat32,
        fc.shaderDenormPreserveFloat32,
        fc.shaderDenormFlushToZeroFloat32,
        fc.shaderRoundingModeRTEFloat32,
        fc.shaderRoundingModeRTZFloat32);
}

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

/* ---------------------- Compute-stage probe ---------------------- */

static bool run_compute_probe(PGRAPHVkState *r, uint32_t *out_results)
{
    VkResult vr;
    bool ok = false;
    const VkDeviceSize buf_size = NAN_PROBE_SLOTS * sizeof(uint32_t);

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
        VkBufferCreateInfo buf_ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buf_size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VmaAllocationCreateInfo buf_alloc_ci = {
            .usage = VMA_MEMORY_USAGE_AUTO,
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
        vr = vmaCreateBuffer(r->allocator, &buf_ci, &buf_alloc_ci,
                             &buffer, &buf_alloc, &buf_info);
        if (vr != VK_SUCCESS) {
            __android_log_print(ANDROID_LOG_ERROR, "hakuX-nan-probe",
                "compute: vmaCreateBuffer failed: %d", vr);
            return false;
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
        if (vkCreateDescriptorSetLayout(r->device, &ci, NULL, &dsl) != VK_SUCCESS) goto out;
    }
    {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0, .size = sizeof(float) * 3,
        };
        VkPipelineLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &dsl,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
        };
        if (vkCreatePipelineLayout(r->device, &ci, NULL, &pll) != VK_SUCCESS) goto out;
    }
    {
        VkShaderModuleCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = nan_probe_spv_len,
            .pCode = (const uint32_t *)nan_probe_spv,
        };
        if (vkCreateShaderModule(r->device, &ci, NULL, &sm) != VK_SUCCESS) goto out;
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
                                     NULL, &pipeline) != VK_SUCCESS) goto out;
    }
    {
        VkDescriptorPoolSize sz = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
        };
        VkDescriptorPoolCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &sz,
        };
        if (vkCreateDescriptorPool(r->device, &ci, NULL, &dp) != VK_SUCCESS) goto out;
    }
    VkDescriptorSet ds;
    {
        VkDescriptorSetAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl,
        };
        if (vkAllocateDescriptorSets(r->device, &ai, &ds) != VK_SUCCESS) goto out;

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
    if (cb == VK_NULL_HANDLE) goto out;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pll, 0, 1, &ds, 0, NULL);
    float pc[3] = { 0.0f, 1.0f, -1.0f };
    vkCmdPushConstants(cb, pll, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cb, 1, 1, 1);
    submit_and_wait(r, cb);

    memcpy(out_results, buf_info.pMappedData, buf_size);
    ok = true;

out:
    if (pool)     vkDestroyCommandPool(r->device, pool, NULL);
    if (dp)       vkDestroyDescriptorPool(r->device, dp, NULL);
    if (pipeline) vkDestroyPipeline(r->device, pipeline, NULL);
    if (sm)       vkDestroyShaderModule(r->device, sm, NULL);
    if (pll)      vkDestroyPipelineLayout(r->device, pll, NULL);
    if (dsl)      vkDestroyDescriptorSetLayout(r->device, dsl, NULL);
    if (buffer)   vmaDestroyBuffer(r->allocator, buffer, buf_alloc);
    return ok;
}

/* ---------------------- Fragment-stage probe ---------------------- *
 * Render a fullscreen triangle into a NAN_PROBE_SLOTS x 1 R32_UINT
 * color attachment; each pixel's gl_FragCoord.x selects which probe
 * slot to compute. Then copy the image to a host-visible buffer and
 * read back. Routes through Mali's graphics scheduler, which is
 * separate from the compute scheduler — that's the whole point.
 */

static bool run_fragment_probe(PGRAPHVkState *r, uint32_t *out_results)
{
    VkResult vr;
    bool ok = false;
    const uint32_t width  = NAN_PROBE_SLOTS;
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

    /* Color attachment image (R32_UINT) + transfer src for readback. */
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
            __android_log_print(ANDROID_LOG_ERROR, "hakuX-nan-probe",
                "fragment: vmaCreateImage failed: %d", vr);
            return false;
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
        if (vkCreateImageView(r->device, &ci, NULL, &image_view) != VK_SUCCESS) goto out;
    }

    /* Host-visible readback buffer. */
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
        if (vr != VK_SUCCESS) goto out;
        memset(readback_info.pMappedData, 0xAA, readback_size);
    }

    /* Render pass — single integer color attachment, DONT_CARE load, STORE store. */
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
        if (vkCreateRenderPass(r->device, &ci, NULL, &rp) != VK_SUCCESS) goto out;
    }
    {
        VkFramebufferCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = rp,
            .attachmentCount = 1, .pAttachments = &image_view,
            .width = width, .height = height, .layers = 1,
        };
        if (vkCreateFramebuffer(r->device, &ci, NULL, &fb) != VK_SUCCESS) goto out;
    }

    /* Shaders + graphics pipeline (no vertex input, no descriptors). */
    {
        VkShaderModuleCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = nan_probe_vert_spv_len,
            .pCode = (const uint32_t *)nan_probe_vert_spv,
        };
        if (vkCreateShaderModule(r->device, &vci, NULL, &vsm) != VK_SUCCESS) goto out;
        VkShaderModuleCreateInfo fci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = nan_probe_frag_spv_len,
            .pCode = (const uint32_t *)nan_probe_frag_spv,
        };
        if (vkCreateShaderModule(r->device, &fci, NULL, &fsm) != VK_SUCCESS) goto out;
    }
    {
        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0, .size = sizeof(float) * 3,
        };
        VkPipelineLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
        };
        if (vkCreatePipelineLayout(r->device, &ci, NULL, &pll) != VK_SUCCESS) goto out;
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
        /* Integer attachment: blending must be disabled. */
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
                                      NULL, &pipeline) != VK_SUCCESS) goto out;
    }

    /* Record + submit + wait. */
    VkCommandBuffer cb = begin_oneshot_cb(r, &pool);
    if (cb == VK_NULL_HANDLE) goto out;

    VkRenderPassBeginInfo rpb = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = rp, .framebuffer = fb,
        .renderArea = { .offset = {0,0}, .extent = { width, height } },
    };
    vkCmdBeginRenderPass(cb, &rpb, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    float pc[3] = { 0.0f, 1.0f, -1.0f };
    vkCmdPushConstants(cb, pll, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);

    /* Image is now in TRANSFER_SRC_OPTIMAL (final layout of the render pass).
     * Copy it into the readback buffer. */
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
    ok = true;

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
    return ok;
}
#endif /* __ANDROID__ */

void pgraph_vk_run_nan_probe(PGRAPHVkState *r)
{
#ifdef __ANDROID__
    /* Gate: enable when either $X1BOX_PROBE_NAN=1 or the sentinel file exists.
     * Android `am start` doesn't propagate env vars from the shell, so the
     * sentinel-file path is what the MCP / adb workflow actually uses.
     * Path is per-package via android_x1box_ext_dir() so the perftest
     * flavor reads its own data dir:
     *   adb shell 'touch /storage/emulated/0/Android/data/<pkg>/files/x1box/probe_nan.flag'
     */
    bool enabled = false;
    const char *env = getenv("X1BOX_PROBE_NAN");
    if (env && env[0] && env[0] != '0') enabled = true;
    if (!enabled) {
        const char *base = android_x1box_ext_dir();
        if (base) {
            char path[512];
            snprintf(path, sizeof(path), "%s/probe_nan.flag", base);
            struct stat st;
            if (stat(path, &st) == 0) {
                enabled = true;
            }
        }
    }
    if (!enabled) {
        return;
    }

    __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
        "running NaN/Inf semantics probes (compute + fragment)");
    log_driver_props(r);

    uint32_t compute_results[NAN_PROBE_SLOTS] = {0};
    uint32_t fragment_results[NAN_PROBE_SLOTS] = {0};

    if (run_compute_probe(r, compute_results)) {
        log_probe_results("compute", compute_results);
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "hakuX-nan-probe",
            "compute probe FAILED");
    }

    if (run_fragment_probe(r, fragment_results)) {
        log_probe_results("fragment", fragment_results);
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "hakuX-nan-probe",
            "fragment probe FAILED");
    }

    /* Diff summary — surface only the slots where the two stages disagree.
     * That's the whole point of running the fragment probe. */
    int diffs = 0;
    for (int i = 0; i < NAN_PROBE_SLOTS; i++) {
        if (compute_results[i] != fragment_results[i]) {
            __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
                "  DIFF [%02d] %-30s  compute=0x%08x  fragment=0x%08x",
                i, k_labels[i].name,
                compute_results[i], fragment_results[i]);
            diffs++;
        }
    }
    if (diffs == 0) {
        __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
            "compute and fragment results IDENTICAL — graphics scheduler "
            "is not the source of NaN-handling divergence");
    } else {
        __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
            "compute vs fragment differ in %d slot(s)", diffs);
    }

    __android_log_print(ANDROID_LOG_WARN, "hakuX-nan-probe",
        "=== probe complete ===");
#else
    (void)r;
#endif
}
