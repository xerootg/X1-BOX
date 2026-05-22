/*
 * C ABI shim for Aaron Giles' x87 library.
 *
 * Exposes a small set of x87 transcendental + helper ops to fpu_helper.c
 * with floatx80 in / floatx80 out. floatx80 (QEMU) and x87::fp80_t have
 * identical { uint64_t low/mantissa; uint16_t high/sign_exp; } layout —
 * we reinterpret across the boundary instead of marshalling.
 */

extern "C" {
#include "qemu/osdep.h"
#include "fpu/softfloat.h"
}

#include "x87fp80.h"
#include "x87common.h"
#include "x87_lib_shim.h"

#include <atomic>

namespace {

using namespace x87;

std::atomic<bool> g_x87_lib_enabled{false};

inline x87::fp80_t &to_lib(floatx80 &v)
{
    static_assert(sizeof(x87::fp80_t) == sizeof(floatx80) ||
                  sizeof(x87::fp80_t) <= sizeof(floatx80) + 6,
                  "fp80_t and floatx80 layout drift");
    return *reinterpret_cast<x87::fp80_t *>(&v);
}

inline const x87::fp80_t &to_lib(const floatx80 &v)
{
    return *reinterpret_cast<const x87::fp80_t *>(&v);
}

inline floatx80 from_lib(const x87::fp80_t &v)
{
    floatx80 out;
    out.low  = v.mantissa();
    out.high = v.sign_exp();
    return out;
}

inline x87::x87cw_t cw_from_status(float_status *st)
{
    if (!st) {
        return X87CW_ROUNDING_NEAREST;
    }
    switch (st->float_rounding_mode) {
    case float_round_nearest_even: return X87CW_ROUNDING_NEAREST;
    case float_round_down:         return X87CW_ROUNDING_DOWN;
    case float_round_up:           return X87CW_ROUNDING_UP;
    case float_round_to_zero:      return X87CW_ROUNDING_ZERO;
    default:                       return X87CW_ROUNDING_NEAREST;
    }
}

/* RAII wrapper: install the requested rounding mode on entry and restore
 * the previous one on scope exit. */
struct rounding_scope {
    x87::fpround_t guard;
    explicit rounding_scope(float_status *st)
        : guard(cw_from_status(st)) {}
};

} /* namespace */

extern "C" {

bool xemu_get_x87_lib(void)
{
    return g_x87_lib_enabled.load(std::memory_order_relaxed);
}

void xemu_set_x87_lib(bool enable)
{
    g_x87_lib_enabled.store(enable, std::memory_order_relaxed);
}

void x87lib_apply_status_flags(uint16_t sw, float_status *status)
{
    if (!status || !sw) {
        return;
    }
    uint16_t f = 0;
    if (sw & x87::X87SW_INVALID_EX)   f |= float_flag_invalid;
    if (sw & x87::X87SW_DIVZERO_EX)   f |= float_flag_divbyzero;
    if (sw & x87::X87SW_OVERFLOW_EX)  f |= float_flag_overflow;
    if (sw & x87::X87SW_UNDERFLOW_EX) f |= float_flag_underflow;
    if (sw & x87::X87SW_PRECISION_EX) f |= float_flag_inexact;
    if (sw & x87::X87SW_DENORM_EX)    f |= float_flag_input_denormal_flushed;
    if (f) {
        float_raise(f, status);
    }
}

#define LIB_UNARY(LIB_OP)                                            \
    rounding_scope _scope(st);                                       \
    x87::fp80_t out;                                                 \
    uint16_t sw = x87::fp80_t::LIB_OP(to_lib(src), out);              \
    *dst = from_lib(out);                                            \
    return sw

#define LIB_BINARY(LIB_OP)                                           \
    rounding_scope _scope(st);                                       \
    x87::fp80_t out;                                                 \
    uint16_t sw = x87::fp80_t::LIB_OP(to_lib(src1), to_lib(src2), out); \
    *dst = from_lib(out);                                            \
    return sw

uint16_t x87lib_f2xm1(floatx80 src, floatx80 *dst, float_status *st)
{
    LIB_UNARY(x87_f2xm1);
}

uint16_t x87lib_fsin(floatx80 src, floatx80 *dst, float_status *st)
{
    LIB_UNARY(x87_fsin);
}

uint16_t x87lib_fcos(floatx80 src, floatx80 *dst, float_status *st)
{
    LIB_UNARY(x87_fcos);
}

uint16_t x87lib_frndint(floatx80 src, floatx80 *dst, float_status *st)
{
    LIB_UNARY(x87_frndint);
}

uint16_t x87lib_fsqrt(floatx80 src, floatx80 *dst, float_status *st)
{
    LIB_UNARY(x87_fsqrt);
}

uint16_t x87lib_fyl2x(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fyl2x);
}

uint16_t x87lib_fyl2xp1(floatx80 src1, floatx80 src2, floatx80 *dst,
                        float_status *st)
{
    LIB_BINARY(x87_fyl2xp1);
}

uint16_t x87lib_fpatan(floatx80 src1, floatx80 src2, floatx80 *dst,
                       float_status *st)
{
    LIB_BINARY(x87_fpatan);
}

uint16_t x87lib_fprem(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fprem);
}

uint16_t x87lib_fprem1(floatx80 src1, floatx80 src2, floatx80 *dst,
                       float_status *st)
{
    LIB_BINARY(x87_fprem1);
}

uint16_t x87lib_fscale(floatx80 src1, floatx80 src2, floatx80 *dst,
                       float_status *st)
{
    LIB_BINARY(x87_fscale);
}

uint16_t x87lib_fsincos(floatx80 src, floatx80 *dst_sin, floatx80 *dst_cos,
                        float_status *st)
{
    rounding_scope _scope(st);
    /* Lib (per test/x87testasm.asm `fsincos80`): dst1 = cos, dst2 = sin. */
    x87::fp80_t cos_val, sin_val;
    uint16_t sw = x87::fp80_t::x87_fsincos(to_lib(src), cos_val, sin_val);
    *dst_sin = from_lib(sin_val);
    *dst_cos = from_lib(cos_val);
    return sw;
}

uint16_t x87lib_fptan(floatx80 src, floatx80 *dst, floatx80 *one_out,
                      float_status *st)
{
    rounding_scope _scope(st);
    /* Lib (per test/x87testasm.asm `fptan80`): dst1 = 1.0, dst2 = tan(x). */
    x87::fp80_t one_val, tan_val;
    uint16_t sw = x87::fp80_t::x87_fptan(to_lib(src), one_val, tan_val);
    *dst = from_lib(tan_val);
    *one_out = from_lib(one_val);
    return sw;
}

uint16_t x87lib_fxtract(floatx80 src, floatx80 *dst_exp, floatx80 *dst_sig,
                        float_status *st)
{
    rounding_scope _scope(st);
    x87::fp80_t sig, exp;
    /* Lib returns (dst1 = significand in [1.0, 2.0), dst2 = exponent). */
    uint16_t sw = x87::fp80_t::x87_fxtract(to_lib(src), sig, exp);
    *dst_exp = from_lib(exp);
    *dst_sig = from_lib(sig);
    return sw;
}

} /* extern "C" */
