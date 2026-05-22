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

/* Basic arithmetic. Return x87 SW exception bits.
 * Hot in SS2 (Serious Engine 2 hits floatx80_mul / floatx80_addsub
 * for ~5% combined CPU) — wiring these through the bit-exact lib lets
 * fpu_helper.c skip the soft-float path when xemu_get_x87_lib() is on. */
uint16_t x87lib_fadd    (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fsub    (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fsubr   (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fmul    (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fdiv    (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);
uint16_t x87lib_fdivr   (floatx80 src1, floatx80 src2, floatx80 *dst,
                         float_status *st);

/* Compares. Return x87 SW C0/C2/C3 bits (no dst). */
uint16_t x87lib_fxam    (floatx80 src);
uint16_t x87lib_ftst    (floatx80 src);
uint16_t x87lib_fcom    (floatx80 src1, floatx80 src2);
uint16_t x87lib_fucom   (floatx80 src1, floatx80 src2);
uint16_t x87lib_fcomi   (floatx80 src1, floatx80 src2);
uint16_t x87lib_fucomi  (floatx80 src1, floatx80 src2);

/* Float / int <-> floatx80 conversions. Equivalent to the
 * float32_to_floatx80 / floatx80_to_float32 / int32_to_floatx80 /
 * floatx80_to_int32 family in fpu_helper.c — wires the conversion
 * helpers (hot SS2 bridge symbols, 0.86% + 0.78% of TCG) through the
 * lib's bit-exact x87_fld* / x87_fst* / x87_fild* / x87_fist* ops.
 * Truncating variants exist for the FISTT* opcode group. */
floatx80 x87lib_float32_to_floatx80 (float32 val, float_status *st);
floatx80 x87lib_float64_to_floatx80 (float64 val, float_status *st);
floatx80 x87lib_int16_to_floatx80   (int16_t val, float_status *st);
floatx80 x87lib_int32_to_floatx80   (int32_t val, float_status *st);
floatx80 x87lib_int64_to_floatx80   (int64_t val, float_status *st);

float32 x87lib_floatx80_to_float32  (floatx80 src, float_status *st);
float64 x87lib_floatx80_to_float64  (floatx80 src, float_status *st);
int16_t x87lib_floatx80_to_int16    (floatx80 src, float_status *st);
int32_t x87lib_floatx80_to_int32    (floatx80 src, float_status *st);
int64_t x87lib_floatx80_to_int64    (floatx80 src, float_status *st);
int16_t x87lib_floatx80_to_int16_trunc(floatx80 src, float_status *st);
int32_t x87lib_floatx80_to_int32_trunc(floatx80 src, float_status *st);
int64_t x87lib_floatx80_to_int64_trunc(floatx80 src, float_status *st);

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
