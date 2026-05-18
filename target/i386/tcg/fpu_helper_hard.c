/*
 * Compile fpu_helper.c a second time with USE_HARD_FPU defined to
 * produce helper_*__hard variants. On AArch64 these use native double
 * precision and consume the same native_d storage the inline TCG FP
 * path emits, keeping the two paths storage-compatible when fp_jit
 * is enabled. Native double has 52-bit mantissa vs x87's 64-bit, so
 * code that requires full extended-precision semantics should fall
 * back to the soft path via the fp_jit toggle.
 */
#if defined(XBOX) && (defined(__x86_64__) || defined(__aarch64__))
#define USE_HARD_FPU 1
#include "fpu_helper.c"
#endif
