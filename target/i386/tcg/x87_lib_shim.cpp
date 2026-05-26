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
#include <cstdlib>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#include <time.h>
#endif

namespace {

using namespace x87;

std::atomic<bool> g_x87_lib_enabled{false};

/*
 * Per-op enable bitmask. Bit i (= X87_LIB_OP_*) set ⇒ that opcode uses
 * lib when xemu_get_x87_lib() is true. Init lazily on first read from the
 * X1BOX_X87_LIB_MASK env var; default 0xFFFFFFFF = all ops on. See header
 * for the bisection workflow.
 */
std::atomic<uint32_t> g_x87_lib_op_mask{0xFFFFFFFFu};
std::atomic<int> g_x87_lib_mask_inited{0};

void init_op_mask_from_env_once()
{
    /*
     * Fast path: relaxed load — if already inited, no CAS, no memory
     * barrier. The CAS variant (used unconditionally previously) showed
     * up as ~6% of vCPU on SS2's profile via __aarch64_cas4_acq_rel
     * because every x87 helper hits xemu_get_x87_lib_op which calls
     * this. With the relaxed-load guard the steady-state cost is one
     * cache-hot load.
     */
    if (g_x87_lib_mask_inited.load(std::memory_order_acquire) != 0) {
        return;
    }
    int expected = 0;
    if (!g_x87_lib_mask_inited.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel)) {
        return;
    }
    const char *v = std::getenv("X1BOX_X87_LIB_MASK");
    if (v && v[0] != '\0') {
        char *end = nullptr;
        unsigned long m = std::strtoul(v, &end, 0); /* base 0 → accepts 0x/0/dec */
        if (end != v && *end == '\0') {
            g_x87_lib_op_mask.store(static_cast<uint32_t>(m),
                                    std::memory_order_relaxed);
        }
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "x1box-x87lib",
                        "op_mask=0x%08x (env X1BOX_X87_LIB_MASK=%s)",
                        g_x87_lib_op_mask.load(std::memory_order_relaxed),
                        v ? v : "<unset>");
#endif
}

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
        return X87CW_PRECISION_EXTENDED | X87CW_ROUNDING_NEAREST |
               X87CW_MASK_ALL_EX;
    }
    /* Rounding mode bits. */
    x87::x87cw_t rc;
    switch (st->float_rounding_mode) {
    case float_round_nearest_even: rc = X87CW_ROUNDING_NEAREST; break;
    case float_round_down:         rc = X87CW_ROUNDING_DOWN; break;
    case float_round_up:           rc = X87CW_ROUNDING_UP; break;
    case float_round_to_zero:      rc = X87CW_ROUNDING_ZERO; break;
    default:                       rc = X87CW_ROUNDING_NEAREST; break;
    }
    /* Precision-control bits — load-bearing for guest-x87 emulation.
     * Halo 2 sets PC=single (24-bit mantissa) for physics integration;
     * the lib's hand-rolled do_div/do_mul on aarch64 reads its CW via
     * read_x87_cw() which on non-x86 falls back to a global override.
     * We pipe st->floatx80_rounding_precision through that override at
     * each scope (via rounding_scope below) so do_* honor PC properly. */
    x87::x87cw_t pc;
    switch (st->floatx80_rounding_precision) {
    case floatx80_precision_s: pc = X87CW_PRECISION_SINGLE; break;
    case floatx80_precision_d: pc = X87CW_PRECISION_DOUBLE; break;
    case floatx80_precision_x:
    default:                   pc = X87CW_PRECISION_EXTENDED; break;
    }
    /* Always mask exceptions — the shim translates SW bits to softfloat
     * float_flag_* explicitly via x87lib_apply_status_flags; we never
     * want the lib's internal paths to throw SIGFPE. */
    return rc | pc | X87CW_MASK_ALL_EX;
}

/* X1BOX_CW_OVERRIDE_PATCH installs these into the lib on non-x86. On
 * x86 hosts the lib reads the real FPU CW via FNSTCW so the override
 * functions are absent; we provide weak no-op fallbacks here so the
 * shim compiles regardless. */
extern "C" {
#if !(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
void  x1box_x87_set_cw_override(x87::x87cw_t cw);
x87::x87cw_t x1box_x87_get_cw_override(void);
#else
__attribute__((weak)) inline void  x1box_x87_set_cw_override(x87::x87cw_t) {}
__attribute__((weak)) inline x87::x87cw_t x1box_x87_get_cw_override(void) { return 0; }
#endif
}

/* RAII wrapper: install the requested rounding mode + PC override on
 * entry and restore the previous on scope exit. Sets BOTH the host FPU
 * rounding mode (for hardware-fast lib paths on x86) AND the thread-
 * local x87 CW override (for software paths on aarch64 — see
 * x87fp80.cpp::read_x87_cw on non-x86). */
struct rounding_scope {
    x87::fpround_t guard;
    x87::x87cw_t prev_override;
    explicit rounding_scope(float_status *st)
        : guard(cw_from_status(st)),
          prev_override(x1box_x87_get_cw_override())
    {
        x1box_x87_set_cw_override(cw_from_status(st));
    }
    ~rounding_scope() {
        x1box_x87_set_cw_override(prev_override);
    }
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

uint32_t xemu_get_x87_lib_mask(void)
{
    init_op_mask_from_env_once();
    return g_x87_lib_op_mask.load(std::memory_order_relaxed);
}

void xemu_set_x87_lib_mask(uint32_t mask)
{
    /* Mark inited so env-var parse doesn't clobber a programmatic set. */
    g_x87_lib_mask_inited.store(1, std::memory_order_release);
    g_x87_lib_op_mask.store(mask, std::memory_order_relaxed);
}

#ifdef __ANDROID__
/*
 * On Android, also poll the `debug.x1box.x87mask` system property so
 * the bisection mask can be flipped live with `adb shell setprop` (no
 * rebuild).
 *
 * Refresh cadence: was wall-clock 500ms via `clock_gettime` per call.
 * That ate 4.47% of vCPU CPU because Halo 2 hits x87 ops millions of
 * times/sec and each call paid a vDSO `clock_gettime` to validate the
 * cache age. Switched 2026-05-24 to a per-thread CALL COUNTER —
 * refresh every 0x10000 (65536) calls. At 5M x87 ops/sec that's a
 * property re-read every ~13 ms, much faster than wall-clock 500 ms,
 * and with ZERO syscalls in the hot path. The setprop-based runtime
 * override still feels instantaneous to a human.
 *
 * setprop is the user-facing knob; the env var still works for
 * boot-time defaults. Property wins when present (non-empty).
 */
/*
 * Plain static (NOT thread_local). Xbox is single-vCPU and these are
 * only read/written from the vCPU thread inside xemu_get_x87_lib_op.
 * thread_local cost a __emutls_get_address + pthread_getspecific per
 * access, which on SS2 (x87-heavy) showed up as ~2% of vCPU. Plain
 * static = one direct load.
 */
static uint32_t s_prop_mask_cache = 0xFFFFFFFFu;
static uint32_t s_prop_mask_calls_left = 0;
static bool     s_prop_mask_seen = false;
static constexpr uint32_t PROP_MASK_REFRESH_CALLS = 0x10000u;

static uint32_t resolve_op_mask_android(uint32_t fallback)
{
    if (__builtin_expect(s_prop_mask_seen && s_prop_mask_calls_left > 0, 1)) {
        s_prop_mask_calls_left--;
        return s_prop_mask_cache;
    }
    s_prop_mask_calls_left = PROP_MASK_REFRESH_CALLS;

    /* Use `debug.*` namespace — SELinux's default policy lets shell write
     * arbitrary debug.* props without an explicit allowlist. (Bare
     * `x1box.x87.mask` fails with "failed to set property".)
     * Set live with:  adb shell setprop debug.x1box.x87mask 0xFF */
    char buf[PROP_VALUE_MAX] = {0};
    int n = __system_property_get("debug.x1box.x87mask", buf);
    if (n > 0 && buf[0] != '\0') {
        char *end = nullptr;
        unsigned long m = std::strtoul(buf, &end, 0);
        if (end != buf && (*end == '\0' || *end == '\n')) {
            s_prop_mask_cache = static_cast<uint32_t>(m);
            s_prop_mask_seen = true;
            return s_prop_mask_cache;
        }
    }
    /* Property absent or unparseable — fall back to env-derived mask. */
    s_prop_mask_cache = fallback;
    s_prop_mask_seen = true;
    return fallback;
}
#endif

bool xemu_get_x87_lib_op(unsigned op)
{
    if (!g_x87_lib_enabled.load(std::memory_order_relaxed)) {
        return false;
    }
    if (op >= 32) {
        /* Out-of-range bit can't be selected by uint32 mask — treat as off
         * so a future op enum extension past 32 entries fails closed. */
        return false;
    }
    init_op_mask_from_env_once();
    uint32_t m = g_x87_lib_op_mask.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    m = resolve_op_mask_android(m);
#endif
    return ((m >> op) & 1u) != 0u;
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

/*
 * Precision-control wiring. The lib's hand-rolled do_div/do_mul on
 * non-x86 hosts uses read_x87_cw() to decide where to round, and
 * that function reads our thread-local override which rounding_scope
 * sets from st->floatx80_rounding_precision. The lib now produces
 * the correct PC-rounded result directly — no second softfloat
 * round-pack step (which would double-round and violate IEEE 754).
 */
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

/*
 * x87 lib FSUB convention compensation — same FDIVP-encoding gotcha
 * as the FDIV pair above (see comment there). The lib defines:
 *   x87_fsub (a, b) returns b - a   (matches FSUBP encoding)
 *   x87_fsubr(a, b) returns a - b   (matches FSUBRP encoding)
 * QEMU's helper_fsub_ST0_FT0 wants src1 - src2 = ST0 - FT0, which
 * means we must call x87_fsubr (a - b) to get that. Symmetric swap
 * for fsubr. Without this, subtractions return their NEGATION —
 * which doesn't crash on its own (FADD+FSUB-only bisection was
 * cosmetically OK) but produces wrong audio amplitudes + texture
 * coordinates downstream when other ops feed it real data.
 */
uint16_t x87lib_fsub(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fsubr);
}

uint16_t x87lib_fsubr(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fsub);
}

uint16_t x87lib_fmul(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fmul);
}

/*
 * x87 lib FDIV convention compensation.
 *
 * The lib uses FDIVP-encoding semantics for its API: per x87fp80.cpp:1459
 *   x87_fdiv (a, b) returns b / a   (matches FDIVP)
 *   x87_fdivr(a, b) returns a / b   (matches FDIVRP)
 *
 * QEMU's helper_fdiv_ST0_FT0 expects mathematical convention:
 *   x87lib_fdiv (src1, src2, dst) → *dst = src1 / src2
 *   x87lib_fdivr(src1, src2, dst) → *dst = src2 / src1
 *
 * To get src1/src2 from the lib we call x87_fdivr (which returns a/b),
 * and for src2/src1 we call x87_fdiv (which returns b/a). Without this
 * swap, every division produced its RECIPROCAL — values were typically
 * off by a factor of (numerator/denominator)², which propagated as
 * out-of-range floats into integer pointer math in Bink's audio decoder.
 * The cascading garbage corrupted the kernel Rtl heap's free-list and
 * triggered KMODE_EXCEPTION_NOT_HANDLED (0x1E / STATUS_ACCESS_VIOLATION)
 * at the next free, bugcheck-halting Halo 2's title screen
 * (2026-05-24, isolated via X1BOX_X87_LIB_MASK bisection).
 *
 * Sibling x87fp80.cpp callers that get this right for reference:
 *   - operator/(a,b): uses x87_fdivr to get a/b
 *   - x87fp80trans.cpp:1801: x87_fdivr(one, pio2_fp80, …) for 1/(π/2)
 *   - test/x87aarch64_smoke.cpp:81: explicit "Asm convention" comment
 */
uint16_t x87lib_fdiv(floatx80 src1, floatx80 src2, floatx80 *dst,
                     float_status *st)
{
    LIB_BINARY(x87_fdivr);
}

uint16_t x87lib_fdivr(floatx80 src1, floatx80 src2, floatx80 *dst,
                      float_status *st)
{
    LIB_BINARY(x87_fdiv);
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
