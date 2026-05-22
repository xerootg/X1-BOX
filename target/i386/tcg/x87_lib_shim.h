#ifndef X87_LIB_SHIM_H
#define X87_LIB_SHIM_H

#include <stdint.h>
#include <stdbool.h>
#include "fpu/softfloat.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Thin C ABI over Aaron Giles' x87 library (x87::fp80_t).
 *
 * The library implements x87 ops at full 80-bit width with bit-exact
 * mantissa match against real Intel x87 hardware (100% on arithmetic;
 * 77-92% mantissa-exact on transcendentals, with the residual gap
 * confined to the C1 rounding-direction status bit which graphics
 * code never reads).
 *
 * Each function takes a float_status* (for rounding mode) and returns
 * the lib's x87 SW exception bits (IE/DE/ZE/OE/UE/PE in the low byte).
 * Use x87lib_apply_status_flags() to OR them into env->fp_status.
 *
 * floatx80 from softfloat-types.h and x87::fp80_t have identical binary
 * layout ({ uint64_t low/mantissa; uint16_t high/sign_exp; }), so we
 * reinterpret_cast inside the shim — no marshalling.
 */

bool xemu_get_x87_lib(void);
void xemu_set_x87_lib(bool enable);

/* Translate x87 SW exception bits to softfloat float_flag_* mask. */
void x87lib_apply_status_flags(uint16_t sw, float_status *status);

/* Transcendentals. Return x87 SW exception bits. */
uint16_t x87lib_f2xm1   (floatx80 src, floatx80 *dst, float_status *st);
uint16_t x87lib_fyl2x   (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fyl2xp1 (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fsin    (floatx80 src, floatx80 *dst, float_status *st);
uint16_t x87lib_fcos    (floatx80 src, floatx80 *dst, float_status *st);
uint16_t x87lib_fsincos (floatx80 src, floatx80 *dst_sin,
                         floatx80 *dst_cos, float_status *st);
uint16_t x87lib_fptan   (floatx80 src, floatx80 *dst,
                         floatx80 *one_out, float_status *st);
uint16_t x87lib_fpatan  (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fprem   (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fprem1  (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fscale  (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fxtract (floatx80 src, floatx80 *dst_exp,
                         floatx80 *dst_sig, float_status *st);
uint16_t x87lib_frndint (floatx80 src, floatx80 *dst, float_status *st);
uint16_t x87lib_fsqrt   (floatx80 src, floatx80 *dst, float_status *st);

#ifdef __cplusplus
}
#endif

#endif /* X87_LIB_SHIM_H */
