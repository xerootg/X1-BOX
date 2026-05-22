/*
 * Generic Vulkan probe runner — wire format + entry point.
 *
 * On-device companion to the gpu_probe() MCP tool. At pgraph_vk_init
 * time, if a request file is present in the app's external files dir,
 * compile + dispatch the user's SPIR-V probe, write a binary result
 * file, drop a done sentinel, and consume the request. No logcat
 * traffic, no per-slot decoding — that's the whole point.
 *
 * Wire format (little-endian, native struct layout — all uint32 + chars,
 * no implicit padding):
 *
 *   /storage/emulated/0/Android/data/com.izzy2lost.x1box/files/x1box/
 *       gpu_probe/probe_req.bin    request, written by MCP
 *       gpu_probe/probe_out.bin    result, written by xemu
 *       gpu_probe/probe_done.flag  sentinel, written by xemu after probe_out
 *
 * On consume, xemu renames probe_req.bin to probe_req.consumed so the
 * same launch doesn't re-run a stale probe.
 */

#ifndef HW_XBOX_NV2A_PGRAPH_VK_PROBE_RUNNER_H
#define HW_XBOX_NV2A_PGRAPH_VK_PROBE_RUNNER_H

#include <stdint.h>

#define X1B_PROBE_REQ_MAGIC        "XPRB"
#define X1B_PROBE_OUT_MAGIC        "XPRO"
#define X1B_PROBE_VERSION          1u
#define X1B_PROBE_STAGE_COMPUTE    0u
#define X1B_PROBE_STAGE_FRAGMENT   1u
#define X1B_PROBE_MAX_OUTPUTS      1024u
#define X1B_PROBE_MAX_PUSH_FLOATS  16u
#define X1B_PROBE_MAX_SPV_BYTES    (256u * 1024u)

/* Request file (probe_req.bin): this header, then `spv_size` bytes of
 * SPIR-V (must be 4-byte aligned and a multiple of 4 in length).
 *
 * Compute shaders:
 *   - layout(local_size_x = 1) in;
 *   - layout(std430, set=0, binding=0) buffer Out { uint probe[N]; };
 *   - layout(push_constant) uniform Push { float p[<= 16]; };
 *   - workgroup count is hardcoded to (1,1,1)
 *
 * Fragment shaders:
 *   - layout(location = 0) out uint result;   // R32_UINT attachment
 *   - layout(push_constant) uniform Push { float p[<= 16]; };
 *   - the runner supplies a fullscreen-triangle vertex shader
 *   - one fragment per output slot, slot = int(gl_FragCoord.x)
 *
 * All uint32_t fields are little-endian on disk. */
struct x1b_probe_req {
    char     magic[4];                  /* "XPRB" */
    uint32_t version;
    uint32_t stage;                     /* X1B_PROBE_STAGE_* */
    uint32_t n_outputs;                 /* 1..X1B_PROBE_MAX_OUTPUTS */
    uint32_t n_push_floats;             /* 0..X1B_PROBE_MAX_PUSH_FLOATS */
    uint32_t spv_size;                  /* bytes; <= X1B_PROBE_MAX_SPV_BYTES */
    uint32_t reserved[2];
    float    push_floats[X1B_PROBE_MAX_PUSH_FLOATS]; /* 16 slots, only first n_push_floats matter */
    /* SPIR-V follows immediately. */
};

/* Result file (probe_out.bin): this header, then `n_outputs` uint32
 * values (raw bit patterns; the host decodes), then an optional ASCII
 * error message of `error_msg_size` bytes (if status != 0). */
struct x1b_probe_out {
    char     magic[4];                  /* "XPRO" */
    uint32_t version;
    uint32_t status;                    /* 0 = ok; non-zero = errno-like (see below) */
    uint32_t n_outputs;                 /* echoes request on ok; may be 0 on error */
    char     gpu_name[256];             /* VkPhysicalDeviceProperties.deviceName, NUL-terminated */
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t driver_version;
    uint32_t fp32_signed_zero_inf_nan_preserve;
    uint32_t fp32_denorm_preserve;
    uint32_t fp32_denorm_flush_to_zero;
    uint32_t fp32_rounding_mode_rte;
    uint32_t fp32_rounding_mode_rtz;
    uint32_t error_msg_size;            /* bytes; 0 when status == 0 */
    uint32_t reserved[3];
    /* uint32_t outputs[n_outputs] follows, then error_msg if any. */
};

/* status codes — keep stable so the host parser can label them. */
enum {
    X1B_PROBE_OK             = 0,
    X1B_PROBE_ERR_NO_REQUEST = 1,  /* (not actually written to disk — runner just skips) */
    X1B_PROBE_ERR_BAD_MAGIC  = 2,
    X1B_PROBE_ERR_BAD_HEADER = 3,
    X1B_PROBE_ERR_SPV_READ   = 4,
    X1B_PROBE_ERR_VK_RESOURCE = 5,
    X1B_PROBE_ERR_VK_SUBMIT  = 6,
    X1B_PROBE_ERR_UNSUPPORTED = 7,
};

/* Forward-decl to avoid pulling all of renderer.h. */
struct PGRAPHVkState;

/* Called once from pgraph_vk_init after compute init. No-op if the
 * request file is absent or this is not Android. */
void pgraph_vk_probe_runner_init(struct PGRAPHVkState *r);

#endif /* HW_XBOX_NV2A_PGRAPH_VK_PROBE_RUNNER_H */
