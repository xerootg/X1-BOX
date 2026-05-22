/*
 * audio_trace.h — runtime-gated trace helpers shared by apu.c and vp.c.
 *
 * Three log streams under tags `hakuX-apu-prod` / `hakuX-apu-cons` /
 * `hakuX-apu-throt` / `hakuX-apu-vdisp` are emitted on Android when
 * either of these holds:
 *   - $X1BOX_AUDIO_TRACE=1 in the launched process's env
 *   - <ext>/x1box/audio_trace.flag exists on disk
 *
 * The check stat()s the flag once per translation unit (cached in a
 * static atomic), so the hot-path cost when disabled is a single
 * relaxed load.
 *
 * The MCP tool `audio_trace_analyze` drains these tags and reports
 * underruns, throttle snaps, slow frames, and silent-frame patterns
 * — see android/tools/x1box_debugger_mcp.py.
 */

#ifndef HW_XBOX_MCPX_APU_AUDIO_TRACE_H
#define HW_XBOX_MCPX_APU_AUDIO_TRACE_H

#include "qemu/osdep.h"
#include "qemu/atomic.h"

#ifdef __ANDROID__
#include "qemu/android-paths.h"
#include <android/log.h>
#include <sys/stat.h>
#include <time.h>

/*
 * Trade-off note: this is `static inline` rather than a single shared
 * extern function on purpose. Each translation unit gets its own
 * one-shot cache, which means we pay one extra getenv + one extra
 * stat() per TU at first hit (apu.c and vp.c — two total). The win
 * is that the disabled-path is a single load with no PLT/GOT jump,
 * matching the pgraph.c frame_stats.flag pattern.
 */
static int g_audio_trace_cached __attribute__((unused)) = -1;

static inline bool audio_trace_enabled(void)
{
    int v = qatomic_read(&g_audio_trace_cached);
    if (v >= 0) return v != 0;
    const char *e = getenv("X1BOX_AUDIO_TRACE");
    int decided = (e && e[0] && e[0] != '0') ? 1 : 0;
    /* DEBUG: log what each gate decided so we can see in logcat why
     * tracing isn't activating. Fires exactly once per process (on the
     * first call), tag `hakuX-apu-init`. Remove once the trace gate is
     * working in steady state. */
    int gate_env = decided;
    int gate_ext = 0;
    int gate_int = 0;
    char ext_path[512] = "(no ext base)";
    char int_path[512] = "(no pkg)";
    struct stat st;
    /* Check 1: external app-files dir. */
    const char *base = android_x1box_ext_dir();
    if (base) {
        snprintf(ext_path, sizeof(ext_path), "%s/audio_trace.flag", base);
        if (stat(ext_path, &st) == 0) {
            gate_ext = 1;
            if (!decided) decided = 1;
        }
    }
    /* Check 2: internal /data/data/<pkg>/files/. */
    const char *pkg = android_package_name();
    if (pkg) {
        snprintf(int_path, sizeof(int_path),
                 "/data/data/%s/files/audio_trace.flag", pkg);
        if (stat(int_path, &st) == 0) {
            gate_int = 1;
            if (!decided) decided = 1;
        }
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "hakuX-apu-init",
        "audio_trace decided=%d env=%d ext=%d int=%d ext_path='%s' int_path='%s'",
        decided, gate_env, gate_ext, gate_int, ext_path, int_path);
#endif
    qatomic_set(&g_audio_trace_cached, decided);
    return decided != 0;
}

static inline int64_t audio_trace_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

#else /* not __ANDROID__ */

static inline bool audio_trace_enabled(void) { return false; }
static inline int64_t audio_trace_now_ns(void) { return 0; }

#endif /* __ANDROID__ */

#endif /* HW_XBOX_MCPX_APU_AUDIO_TRACE_H */
