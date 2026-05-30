/*
 * x87 native-fp64 vs true-floatx80 shadow differential (debug / validation).
 *
 * This file is compiled WITHOUT USE_HARD_FPU, so the floatx80_* symbols are the
 * REAL 80-bit softfloat functions (not the f64 macros in fpu_helper.c). For each
 * native fp64 float->int / float->f32 conversion the JIT emits (gated at
 * translate time by g_x87_shadow / env X1BOX_X87_SHADOW), we recompute the same
 * conversion in true x87 semantics — floatx80 at PC=53, honoring the guest
 * rounding-control bits — and log any disagreement.
 *
 * Since native fp64 == x87@PC=53 bit-for-bit for in-range values, this should
 * report ~0 disagreements; it flags only the documented edges (exponent-range
 * overflow, FISTP NaN/out-of-range). The helper is non-perturbing: it receives
 * ST0 as a passed bit pattern and never touches the inline FP register cache.
 */
#include "qemu/osdep.h"
#include "cpu.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include <android/log.h>

static uint64_t shadow_checks;
static uint64_t shadow_disagrees;

static int x1box_rc_from_fpuc(uint32_t fpuc)
{
    switch ((fpuc >> 10) & 3) {
    case 0:  return float_round_nearest_even;
    case 1:  return float_round_down;
    case 2:  return float_round_up;
    default: return float_round_to_zero;
    }
}

static void shadow_tick(void)
{
    if ((++shadow_checks & ((1ULL << 22) - 1)) == 0) {
        __android_log_print(ANDROID_LOG_INFO, "x1box-x87shadow",
                            "checked=%llu disagreements=%llu",
                            (unsigned long long)shadow_checks,
                            (unsigned long long)shadow_disagrees);
    }
}

static bool shadow_should_log(void)
{
    return shadow_disagrees < 40 || (shadow_disagrees & 0xFFF) == 0;
}

void helper_x87_shadow_fistl(CPUX86State *env, uint64_t st0_bits, int32_t native_res)
{
    float_status st = env->fp_status;
    set_float_rounding_mode(x1box_rc_from_fpuc(env->fpuc), &st);
    floatx80 fx = float64_to_floatx80((float64)st0_bits, &st);
    int32_t ref = floatx80_to_int32(fx, &st);
    shadow_tick();
    if (ref != native_res) {
        if (shadow_should_log()) {
            double d;
            memcpy(&d, &st0_bits, sizeof(d));
            __android_log_print(ANDROID_LOG_WARN, "x1box-x87shadow",
                "FISTP m32 #%llu st0=%.17g native=%d x87ref=%d rc=%u",
                (unsigned long long)shadow_disagrees, d, native_res, ref,
                (unsigned)((env->fpuc >> 10) & 3));
        }
        shadow_disagrees++;
    }
}

void helper_x87_shadow_fistll(CPUX86State *env, uint64_t st0_bits, int64_t native_res)
{
    float_status st = env->fp_status;
    set_float_rounding_mode(x1box_rc_from_fpuc(env->fpuc), &st);
    floatx80 fx = float64_to_floatx80((float64)st0_bits, &st);
    int64_t ref = floatx80_to_int64(fx, &st);
    shadow_tick();
    if (ref != native_res) {
        if (shadow_should_log()) {
            double d;
            memcpy(&d, &st0_bits, sizeof(d));
            __android_log_print(ANDROID_LOG_WARN, "x1box-x87shadow",
                "FISTP m64 #%llu st0=%.17g native=%lld x87ref=%lld rc=%u",
                (unsigned long long)shadow_disagrees, d,
                (long long)native_res, (long long)ref,
                (unsigned)((env->fpuc >> 10) & 3));
        }
        shadow_disagrees++;
    }
}

void helper_x87_shadow_fsts(CPUX86State *env, uint64_t st0_bits, uint32_t native_bits)
{
    float_status st = env->fp_status;
    set_float_rounding_mode(x1box_rc_from_fpuc(env->fpuc), &st);
    floatx80 fx = float64_to_floatx80((float64)st0_bits, &st);
    float32 ref = floatx80_to_float32(fx, &st);
    shadow_tick();
    if ((uint32_t)ref != native_bits) {
        /* both-NaN is not a real divergence (payload differences are benign) */
        if (!(float32_is_any_nan(ref) && float32_is_any_nan((float32)native_bits))) {
            if (shadow_should_log()) {
                double d;
                float nf, rf;
                memcpy(&d, &st0_bits, sizeof(d));
                memcpy(&nf, &native_bits, sizeof(nf));
                memcpy(&rf, &ref, sizeof(rf));
                __android_log_print(ANDROID_LOG_WARN, "x1box-x87shadow",
                    "FST m32 #%llu st0=%.17g native=%.9g x87ref=%.9g rc=%u",
                    (unsigned long long)shadow_disagrees, d,
                    (double)nf, (double)rf, (unsigned)((env->fpuc >> 10) & 3));
            }
            shadow_disagrees++;
        }
    }
}
