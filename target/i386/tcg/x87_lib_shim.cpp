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

/* Basic arithmetic — bit-exact mantissa match against real Intel x87
 * hardware (per lib readme: 100% on these ops). Caller MUST OR the
 * returned SW bits into env->fp_status via x87lib_apply_status_flags
 * if it cares about exception/inexact flags. */
uint16_t x87lib_fadd(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fadd);
}

uint16_t x87lib_fsub(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fsub);
}

uint16_t x87lib_fsubr(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fsubr);
}

uint16_t x87lib_fmul(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fmul);
}

uint16_t x87lib_fdiv(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fdiv);
}

uint16_t x87lib_fdivr(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fdivr);
}

uint16_t x87lib_fxam(floatx80 src)
{
    return x87::fp80_t::x87_fxam(to_lib(src));
}

uint16_t x87lib_ftst(floatx80 src)
{
    return x87::fp80_t::x87_ftst(to_lib(src));
}

uint16_t x87lib_fcom(floatx80 src1, floatx80 src2)
{
    return x87::fp80_t::x87_fcom(to_lib(src1), to_lib(src2));
}

uint16_t x87lib_fucom(floatx80 src1, floatx80 src2)
{
    return x87::fp80_t::x87_fucom(to_lib(src1), to_lib(src2));
}

uint16_t x87lib_fcomi(floatx80 src1, floatx80 src2)
{
    return x87::fp80_t::x87_fcomi(to_lib(src1), to_lib(src2));
}

uint16_t x87lib_fucomi(floatx80 src1, floatx80 src2)
{
    return x87::fp80_t::x87_fucomi(to_lib(src1), to_lib(src2));
}

/* --- Conversions ---
 *
 * The lib's load helpers (x87_fld32 / fld64 / fild16/32/64) take cw+sw via
 * reference + dst by reference + src by void*. We marshal in our own
 * scope-installed rounding mode and discard the lib's sw output (fpu_helper
 * conversion helpers don't propagate exception flags except where they
 * already raised them manually). Pattern mirrors the as_*() member helpers
 * in x87fp80.h that the lib uses internally. */

#define LIB_LOAD_FROM(SrcT, LIB_OP)                                  \
    rounding_scope _scope(st);                                       \
    x87::fp80_t out;                                                 \
    x87::x87sw_t sw;                                                 \
    x87::fp80_t::LIB_OP(x87::fpround_t::get(), sw, out, &val);       \
    (void)sw;                                                        \
    return from_lib(out)

#define LIB_STORE_TO(DstT, LIB_OP)                                   \
    rounding_scope _scope(st);                                       \
    DstT out;                                                        \
    x87::x87sw_t sw;                                                 \
    x87::fp80_t::LIB_OP(x87::fpround_t::get(), sw, &out, to_lib(src)); \
    (void)sw;                                                        \
    return out

floatx80 x87lib_float32_to_floatx80(float32 val, float_status *st)
{
    LIB_LOAD_FROM(float32, x87_fld32);
}

floatx80 x87lib_float64_to_floatx80(float64 val, float_status *st)
{
    LIB_LOAD_FROM(float64, x87_fld64);
}

floatx80 x87lib_int16_to_floatx80(int16_t val, float_status *st)
{
    LIB_LOAD_FROM(int16_t, x87_fild16);
}

floatx80 x87lib_int32_to_floatx80(int32_t val, float_status *st)
{
    LIB_LOAD_FROM(int32_t, x87_fild32);
}

floatx80 x87lib_int64_to_floatx80(int64_t val, float_status *st)
{
    LIB_LOAD_FROM(int64_t, x87_fild64);
}

float32 x87lib_floatx80_to_float32(floatx80 src, float_status *st)
{
    LIB_STORE_TO(float32, x87_fst32);
}

float64 x87lib_floatx80_to_float64(floatx80 src, float_status *st)
{
    LIB_STORE_TO(float64, x87_fst64);
}

int16_t x87lib_floatx80_to_int16(floatx80 src, float_status *st)
{
    LIB_STORE_TO(int16_t, x87_fist16);
}

int32_t x87lib_floatx80_to_int32(floatx80 src, float_status *st)
{
    LIB_STORE_TO(int32_t, x87_fist32);
}

int64_t x87lib_floatx80_to_int64(floatx80 src, float_status *st)
{
    LIB_STORE_TO(int64_t, x87_fist64);
}

/* Truncating int stores (FISTT* opcode group): force round-toward-zero
 * for the duration of this call regardless of the guest's float_status. */
int16_t x87lib_floatx80_to_int16_trunc(floatx80 src, float_status *st)
{
    (void)st;
    x87::fpround_t guard(X87CW_ROUNDING_ZERO);
    int16_t out;
    x87::x87sw_t sw;
    x87::fp80_t::x87_fist16(x87::fpround_t::get(), sw, &out, to_lib(src));
    (void)sw;
    return out;
}

int32_t x87lib_floatx80_to_int32_trunc(floatx80 src, float_status *st)
{
    (void)st;
    x87::fpround_t guard(X87CW_ROUNDING_ZERO);
    int32_t out;
    x87::x87sw_t sw;
    x87::fp80_t::x87_fist32(x87::fpround_t::get(), sw, &out, to_lib(src));
    (void)sw;
    return out;
}

int64_t x87lib_floatx80_to_int64_trunc(floatx80 src, float_status *st)
{
    (void)st;
    x87::fpround_t guard(X87CW_ROUNDING_ZERO);
    int64_t out;
    x87::x87sw_t sw;
    x87::fp80_t::x87_fist64(x87::fpround_t::get(), sw, &out, to_lib(src));
    (void)sw;
    return out;
}

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
