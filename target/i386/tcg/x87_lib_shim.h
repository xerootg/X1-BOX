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

/*
 * Per-instruction enable mask for bisecting which lib op causes a guest
 * regression. Default mask is 0xFFFFFFFF (all ops use lib when global
 * xemu_set_x87_lib(true) is set). Override at runtime via the env var
 *
 *   X1BOX_X87_LIB_MASK=0xNNNN   (parsed with strtoul base=0)
 *
 * Bit i (= X87_LIB_OP_*) cleared ⇒ that opcode falls back to soft-float
 * even when the global toggle is on. The set of gated ops below
 * corresponds 1:1 with the `if (xemu_get_x87_lib_op(...))` sites in
 * fpu_helper.c, so the mask is dense — binary search lands in O(log2 N).
 *
 * Quick bisection workflow (see [[project_x87_lib_slow_methods]]):
 *   X1BOX_X87_LIB_MASK=0           → all soft-float (baseline)
 *   X1BOX_X87_LIB_MASK=0xFFFFFFFF  → all lib (= prior default behavior)
 *   X1BOX_X87_LIB_MASK=0x000000FF  → only ops 0-7 use lib
 *   X1BOX_X87_LIB_MASK=0x00000001  → only F2XM1 uses lib, etc.
 */
enum X87LibOp {
    X87_LIB_OP_F2XM1 = 0,
    X87_LIB_OP_FPTAN,
    X87_LIB_OP_FPATAN,
    X87_LIB_OP_FXTRACT,
    X87_LIB_OP_FPREM,       /* covers both FPREM and FPREM1 */
    X87_LIB_OP_FYL2XP1,
    X87_LIB_OP_FYL2X,
    X87_LIB_OP_FSQRT,
    X87_LIB_OP_FSINCOS,
    X87_LIB_OP_FRNDINT,
    X87_LIB_OP_FSCALE,
    X87_LIB_OP_FSIN,
    X87_LIB_OP_FCOS,
    X87_LIB_OP_FADD,        /* covers helper_fadd_{ST0_FT0,STN_ST0} */
    X87_LIB_OP_FSUB,        /* covers FSUB and FSUBR */
    X87_LIB_OP_FMUL,
    X87_LIB_OP_FDIV,        /* covers FDIV and FDIVR */
    X87_LIB_OP_COUNT
};

bool xemu_get_x87_lib_op(unsigned op);
uint32_t xemu_get_x87_lib_mask(void);
void xemu_set_x87_lib_mask(uint32_t mask);

/*
 * Precision-guarded gate for the lib. Returns true ONLY when the
 * given op is enabled in the mask AND the guest's current x87 PC
 * (precision-control) bits select extended (64-bit mantissa) — the
 * lib operates exclusively at extended precision and ignores PC,
 * so routing single/double-precision computations through it would
 * silently compute extra mantissa bits the game never asked for.
 *
 * Halo 2 sets PC to extended for most code (Xbox default after
 * boot) but switches to single (24-bit) for some physics paths.
 * Without this guard, those physics computations get fp80 mantissa
 * accuracy from the lib instead of 24-bit-rounded from softfloat,
 * which manifests as position drift / momentum / sliding statics
 * once the guest's collision/integration math diverges from the
 * original Intel x87 rounding it was tuned against
 * (2026-05-24, after the FDIV/FSUB operand-swap fix unblocked
 * gameplay enough to expose the precision-control gap).
 */
static inline bool xemu_use_x87_lib_op(unsigned op,
                                       const float_status *st)
{
    if (!xemu_get_x87_lib_op(op)) {
        return false;
    }
    if (st && st->floatx80_rounding_precision != floatx80_precision_x) {
        return false;
    }
    return true;
}

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
