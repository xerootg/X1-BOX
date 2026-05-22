/*
 * Android Dynamic Performance Framework (ADPF) — system_server-relayed
 * uclamp boost for app-uid processes. Sidesteps the CAP_SYS_NICE check
 * that blocks the direct sched_setattr path used by sched-android.c
 * (confirmed empirically: EPERM on every uclamp.min raise on Pixel 10a
 * / Android 14; see project_pixel_sched_underclock.md).
 *
 * What this does: at init, dlsym the APerformanceHint_* symbols from
 * libandroid.so, enumerate /proc/self/task for our perf threads,
 * create an APerformanceHintSession with those TIDs and a target work
 * duration, then spawn a tiny reporter thread that periodically calls
 * reportActualWorkDuration so the kernel-side policy can converge.
 *
 * Gated on $X1BOX_ADPF_ENABLED (default OFF) so it's opt-in. Reads
 * from the same sched_config.txt the affinity layer uses, so the
 * user has a single runtime knob file.
 */

#include "qemu/osdep.h"
#include "qemu/adpf-android.h"
#include "qemu/android-paths.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <dlfcn.h>
#endif
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define LOG_TAG "x1box-adpf"

#ifdef __ANDROID__
#define ADPF_LOGI(fmt, ...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, fmt, ##__VA_ARGS__)
#define ADPF_LOGW(fmt, ...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, fmt, ##__VA_ARGS__)
#define ADPF_LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##__VA_ARGS__)
#else
#define ADPF_LOGI(fmt, ...) ((void)0)
#define ADPF_LOGW(fmt, ...) ((void)0)
#define ADPF_LOGE(fmt, ...) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* ADPF symbols — dlsym'd at runtime to support pre-Android-13 builds */
/* ------------------------------------------------------------------ */

/* Opaque types from <android/performance_hint.h>; we forward-declare so
 * we don't need to require NDK r25+ headers in the source tree. */
struct APerformanceHintManager;
struct APerformanceHintSession;
struct AWorkDuration;

typedef struct APerformanceHintManager APerformanceHintManager;
typedef struct APerformanceHintSession APerformanceHintSession;
typedef struct AWorkDuration           AWorkDuration;

static APerformanceHintManager *(*p_getManager)(void);
static int64_t (*p_getPreferredUpdateRateNanos)(APerformanceHintManager *);
static APerformanceHintSession *(*p_createSession)(
    APerformanceHintManager *, const int32_t *, size_t, int64_t);
static int (*p_updateTargetWorkDuration)(APerformanceHintSession *, int64_t);
static int (*p_reportActualWorkDuration)(APerformanceHintSession *, int64_t);
static void (*p_closeSession)(APerformanceHintSession *);
/* API 34+ — used by add_thread to extend the session at runtime when the
 * sched-android scanner spots a new perf-class TID (e.g. cranelift-tcg).
 * Optional: stays NULL on older Android. */
static int (*p_setThreads)(APerformanceHintSession *, const int32_t *, size_t);

/* API 34+ split-duration reporting. Lets ADPF distinguish CPU work from
 * GPU work so it boosts the right side. Optional — falls back to the
 * single-value path on older Android. */
static AWorkDuration *(*p_AWorkDuration_create)(void);
static void           (*p_AWorkDuration_release)(AWorkDuration *);
static void           (*p_AWorkDuration_setWorkPeriodStartTimestampNanos)(AWorkDuration *, int64_t);
static void           (*p_AWorkDuration_setActualTotalDurationNanos)(AWorkDuration *, int64_t);
static void           (*p_AWorkDuration_setActualCpuDurationNanos)(AWorkDuration *, int64_t);
static void           (*p_AWorkDuration_setActualGpuDurationNanos)(AWorkDuration *, int64_t);
static int            (*p_reportActualWorkDuration2)(APerformanceHintSession *, AWorkDuration *);

static bool resolve_adpf_symbols(void)
{
#ifdef __ANDROID__
    void *h = dlopen("libandroid.so", RTLD_NOW);
    if (!h) {
        ADPF_LOGE("dlopen(libandroid.so) failed: %s", dlerror());
        return false;
    }
    /* All symbols are required. If any are missing, this is an older
     * Android (<13) or a stripped libandroid; bail. */
    p_getManager                  = (APerformanceHintManager *(*)(void))
        dlsym(h, "APerformanceHint_getManager");
    p_getPreferredUpdateRateNanos = (int64_t (*)(APerformanceHintManager *))
        dlsym(h, "APerformanceHint_getPreferredUpdateRateNanos");
    p_createSession               = (APerformanceHintSession *(*)(
        APerformanceHintManager *, const int32_t *, size_t, int64_t))
        dlsym(h, "APerformanceHint_createSession");
    p_updateTargetWorkDuration    = (int (*)(APerformanceHintSession *, int64_t))
        dlsym(h, "APerformanceHint_updateTargetWorkDuration");
    p_reportActualWorkDuration    = (int (*)(APerformanceHintSession *, int64_t))
        dlsym(h, "APerformanceHint_reportActualWorkDuration");
    p_closeSession                = (void (*)(APerformanceHintSession *))
        dlsym(h, "APerformanceHint_closeSession");
    /* setThreads is API 34+ (Android 14). NULL on older devices —
     * adpf_android_add_thread() degrades gracefully to a no-op. */
    p_setThreads                  = (int (*)(
        APerformanceHintSession *, const int32_t *, size_t))
        dlsym(h, "APerformanceHint_setThreads");

    /* Split-duration API (API 34+). All-or-nothing: we treat the whole
     * group as a single optional capability. If any symbol is missing,
     * adpf_android_report_frame_split() falls back to the single-value
     * path. */
    p_AWorkDuration_create =
        (AWorkDuration *(*)(void))
        dlsym(h, "AWorkDuration_create");
    p_AWorkDuration_release =
        (void (*)(AWorkDuration *))
        dlsym(h, "AWorkDuration_release");
    p_AWorkDuration_setWorkPeriodStartTimestampNanos =
        (void (*)(AWorkDuration *, int64_t))
        dlsym(h, "AWorkDuration_setWorkPeriodStartTimestampNanos");
    p_AWorkDuration_setActualTotalDurationNanos =
        (void (*)(AWorkDuration *, int64_t))
        dlsym(h, "AWorkDuration_setActualTotalDurationNanos");
    p_AWorkDuration_setActualCpuDurationNanos =
        (void (*)(AWorkDuration *, int64_t))
        dlsym(h, "AWorkDuration_setActualCpuDurationNanos");
    p_AWorkDuration_setActualGpuDurationNanos =
        (void (*)(AWorkDuration *, int64_t))
        dlsym(h, "AWorkDuration_setActualGpuDurationNanos");
    p_reportActualWorkDuration2 =
        (int (*)(APerformanceHintSession *, AWorkDuration *))
        dlsym(h, "APerformanceHint_reportActualWorkDuration2");

    if (!p_getManager || !p_createSession || !p_updateTargetWorkDuration
        || !p_reportActualWorkDuration || !p_closeSession) {
        ADPF_LOGW("ADPF symbols unavailable (Android < 13 or stripped libandroid)");
        return false;
    }
    return true;
#else
    return false;
#endif
}

/* ------------------------------------------------------------------ */
/* Config (file-backed; shares the sched-android knob file)            */
/* ------------------------------------------------------------------ */

#define MAX_KV 64
static struct { char k[64]; char v[160]; } g_kv[MAX_KV];
static int g_kv_n;

static void load_kv(void)
{
    g_kv_n = 0;
    const char *base = android_x1box_ext_dir();
    if (!base) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/sched_config.txt", base);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f) && g_kv_n < MAX_KV) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0' || *p == '\n') continue;
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        char *k = p, *v = eq + 1;
        size_t kl = strlen(k);
        while (kl && (k[kl-1] == ' ' || k[kl-1] == '\t')) k[--kl] = '\0';
        while (*v == ' ' || *v == '\t') v++;
        size_t vl = strlen(v);
        while (vl && (v[vl-1] == '\n' || v[vl-1] == '\r'
                      || v[vl-1] == ' ' || v[vl-1] == '\t')) v[--vl] = '\0';
        if (!*k) continue;
        strncpy(g_kv[g_kv_n].k, k, sizeof(g_kv[g_kv_n].k) - 1);
        g_kv[g_kv_n].k[sizeof(g_kv[g_kv_n].k) - 1] = '\0';
        strncpy(g_kv[g_kv_n].v, v, sizeof(g_kv[g_kv_n].v) - 1);
        g_kv[g_kv_n].v[sizeof(g_kv[g_kv_n].v) - 1] = '\0';
        g_kv_n++;
    }
    fclose(f);
}

static const char *lookup_str(const char *name)
{
    for (int i = 0; i < g_kv_n; i++) {
        if (!strcmp(g_kv[i].k, name)) return g_kv[i].v;
    }
    return getenv(name);
}

static int64_t cfg_i64(const char *name, int64_t dflt)
{
    const char *v = lookup_str(name);
    if (!v || !*v) return dflt;
    char *end = NULL;
    long long n = strtoll(v, &end, 10);
    if (!end || *end != '\0') return dflt;
    return (int64_t)n;
}

static bool cfg_bool(const char *name, bool dflt)
{
    const char *v = lookup_str(name);
    if (!v || !*v) return dflt;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T'
        || v[0] == 'y' || v[0] == 'Y';
}

/* ------------------------------------------------------------------ */
/* Perf-thread enumeration                                             */
/* ------------------------------------------------------------------ */

/* Same prefix table as sched-android's classify(), kept in sync so
 * a thread that opted in to affinity also opts into ADPF. The
 * filter env var lets the user narrow further (e.g. "hot,hotwarm"
 * to leave the voice workers out of the session if it overshoots
 * the 32-TID limit). */
typedef enum {
    ATC_NONE = 0,
    ATC_HOT,
    ATC_HOTWARM,
    ATC_WARM,
} AdpfThreadClass;

static AdpfThreadClass classify_comm(const char *comm)
{
    if (!comm || !*comm) return ATC_NONE;
    if (!strncmp(comm, "CPU ",           4))  return ATC_HOT;       /* CPU N/TCG */
    if (!strncmp(comm, "nv2a.pfifo",    10))  return ATC_HOTWARM;
    if (!strncmp(comm, "cranelift-tcg", 13))  return ATC_WARM;
    if (!strncmp(comm, "pgraph.vk.rend",14))  return ATC_WARM;
    if (!strncmp(comm, "pgraph.vk.comp",14))  return ATC_WARM;
    if (!strncmp(comm, "mcpx.apu",       8))  return ATC_WARM;
    if (!strncmp(comm, "mcpx.voice",    10))  return ATC_WARM;
    if (!strncmp(comm, "SDLAudioP2",    10))  return ATC_WARM;
    return ATC_NONE;
}

static bool filter_allows(const char *filter, AdpfThreadClass c)
{
    /* Filter is a comma-separated list of class names. Empty/missing
     * defaults to all classes. */
    if (!filter || !*filter) return true;
    const char *want = NULL;
    switch (c) {
    case ATC_HOT:     want = "hot";     break;
    case ATC_HOTWARM: want = "hotwarm"; break;
    case ATC_WARM:    want = "warm";    break;
    default: return false;
    }
    const char *p = filter;
    while (*p) {
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        /* trim spaces */
        while (len && (*p == ' ' || *p == '\t')) { p++; len--; }
        while (len && (p[len-1] == ' ' || p[len-1] == '\t')) len--;
        if (len == strlen(want) && !strncasecmp(p, want, len)) return true;
        if (!comma) break;
        p = comma + 1;
    }
    return false;
}

#define MAX_SESSION_TIDS 32

struct enumerated_tids {
    int32_t tids[MAX_SESSION_TIDS];
    char    names[MAX_SESSION_TIDS][32];
    int     n;
};

static int read_comm(const char *task_dir, int tid, char *out, size_t out_sz)
{
    char path[128];
    snprintf(path, sizeof(path), "%s/%d/comm", task_dir, tid);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(out, (int)out_sz, f)) { fclose(f); return -1; }
    fclose(f);
    /* strip trailing newline */
    size_t n = strlen(out);
    while (n && (out[n-1] == '\n' || out[n-1] == '\r')) out[--n] = '\0';
    return 0;
}

static void enumerate_perf_tids(const char *filter, struct enumerated_tids *out)
{
    out->n = 0;
    char task_dir[] = "/proc/self/task";
    DIR *d = opendir(task_dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && out->n < MAX_SESSION_TIDS) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        int tid = atoi(e->d_name);
        if (tid <= 0) continue;
        char comm[32] = {0};
        if (read_comm(task_dir, tid, comm, sizeof(comm)) != 0) continue;
        AdpfThreadClass c = classify_comm(comm);
        if (c == ATC_NONE) continue;
        if (!filter_allows(filter, c)) continue;
        out->tids[out->n] = (int32_t)tid;
        strncpy(out->names[out->n], comm, sizeof(out->names[0]) - 1);
        out->names[out->n][sizeof(out->names[0]) - 1] = '\0';
        out->n++;
    }
    closedir(d);
}

/* ------------------------------------------------------------------ */
/* Session state + reporter thread                                     */
/* ------------------------------------------------------------------ */

static struct {
    bool                          initialized;
    bool                          debug;
    APerformanceHintManager      *mgr;
    APerformanceHintSession      *session;
    int64_t                       target_ns;
    int64_t                       report_period_ns;
    pthread_t                     reporter;
    bool                          reporter_running;
    /* Live TID list — mutated by add_thread when the sched scanner
     * spots a new perf-class thread, then re-pushed via setThreads
     * to system_server. Guarded by tid_lock for cross-thread updates. */
    pthread_mutex_t               tid_lock;
    int32_t                       tids[MAX_SESSION_TIDS];
    int                           n_tids;
    char                          tid_summary[256];
    /* Set every time adpf_android_report_frame() is called (from the
     * NV097_FLIP_STALL handler). Reporter thread reads this to decide
     * whether to fall back to polling (no real frame reports coming
     * in) or back off to a 1 Hz heartbeat (proper reports flowing).
     * Plain int64_t — __atomic_* builtins work on it, the _Atomic
     * qualifier breaks the builtin's operand-type check. */
    int64_t                       last_real_report_ns;
    /* Reusable AWorkDuration. We allocate it once at session creation
     * to avoid the per-frame malloc/free that AWorkDuration_create()
     * would impose. NULL on Android < 14 — the split-report API
     * degrades to the single-value path in that case. */
    AWorkDuration                *work_duration;
} g_state;

static int64_t monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static void *reporter_loop(void *arg)
{
    (void)arg;
#ifdef __ANDROID__
    /* Give ourselves a recognizable comm so the affinity layer doesn't
     * sweep us into a perf class — we're a reporting helper, not a
     * worker. */
    syscall(__NR_prctl, 15 /* PR_SET_NAME */, "x1box-adpf-rpt", 0, 0, 0);

    /* If a real per-frame report has arrived within FRAME_REPORT_STALE_NS
     * (500 ms), the NV097_FLIP_STALL hook is feeding ADPF accurate
     * data — back off to a 1 Hz heartbeat just to keep the session
     * from timing out and avoid drowning the system with synthetic
     * "elapsed wall-clock" reports. If no real reports show up (game
     * paused, between titles, ADPF_FRAME_REPORT disabled), fall back
     * to the original polling cadence. */
    const int64_t FRAME_REPORT_STALE_NS = 500LL * 1000 * 1000;
    const int64_t HEARTBEAT_NS         = 1000LL * 1000 * 1000;  /* 1 Hz */

    int64_t prev = monotonic_ns();
    while (__atomic_load_n(&g_state.reporter_running, __ATOMIC_ACQUIRE)) {
        int64_t now = monotonic_ns();
        int64_t last_real = __atomic_load_n(&g_state.last_real_report_ns,
                                            __ATOMIC_RELAXED);
        bool real_flowing = (last_real != 0)
                          && (now - last_real < FRAME_REPORT_STALE_NS);

        int64_t sleep_ns = real_flowing ? HEARTBEAT_NS
                                        : g_state.report_period_ns;
        struct timespec ts = {
            .tv_sec  = sleep_ns / 1000000000LL,
            .tv_nsec = sleep_ns % 1000000000LL,
        };
        nanosleep(&ts, NULL);

        now = monotonic_ns();
        int64_t actual = now - prev;
        prev = now;

        /* Only synthesize a report when no real frame data is coming
         * in. Otherwise we'd drown the actual signal. */
        last_real = __atomic_load_n(&g_state.last_real_report_ns,
                                     __ATOMIC_RELAXED);
        real_flowing = (last_real != 0)
                     && (now - last_real < FRAME_REPORT_STALE_NS);
        if (!real_flowing && g_state.session && p_reportActualWorkDuration) {
            p_reportActualWorkDuration(g_state.session, actual);
        }
    }
#endif
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void adpf_android_init(void)
{
#ifdef __ANDROID__
    if (g_state.initialized) return;
    load_kv();

    if (!cfg_bool("X1BOX_ADPF_ENABLED", false)) {
        ADPF_LOGI("disabled (X1BOX_ADPF_ENABLED != 1)");
        g_state.initialized = true;
        return;
    }
    g_state.debug = cfg_bool("X1BOX_ADPF_DEBUG", false);
    g_state.target_ns = cfg_i64("X1BOX_ADPF_TARGET_NS", 33333333);  /* 30 FPS */
    int64_t period_ms = cfg_i64("X1BOX_ADPF_REPORT_PERIOD_MS", 16);
    g_state.report_period_ns = period_ms * 1000000LL;

    if (!resolve_adpf_symbols()) {
        ADPF_LOGW("ADPF symbol resolve failed; init skipped");
        g_state.initialized = true;
        return;
    }

    g_state.mgr = p_getManager();
    if (!g_state.mgr) {
        ADPF_LOGW("APerformanceHint_getManager returned NULL");
        g_state.initialized = true;
        return;
    }
    if (p_getPreferredUpdateRateNanos) {
        int64_t pref = p_getPreferredUpdateRateNanos(g_state.mgr);
        ADPF_LOGI("preferred update rate = %" PRId64 " ns (%.2f ms)",
                  pref, (double)pref / 1e6);
        /* If the user didn't override the period, prefer what the
         * system asks for — but clamp to a sane window so we don't
         * spin if the system reports something very small. */
        if (!lookup_str("X1BOX_ADPF_REPORT_PERIOD_MS") && pref > 1000000LL) {
            g_state.report_period_ns = pref;
        }
    }

    const char *filter = lookup_str("X1BOX_ADPF_THREAD_FILTER");
    struct enumerated_tids et;
    enumerate_perf_tids(filter, &et);

    if (et.n == 0) {
        ADPF_LOGW("no perf threads found; ADPF session not created");
        g_state.initialized = true;
        return;
    }

    g_state.session = p_createSession(g_state.mgr, et.tids,
                                      (size_t)et.n, g_state.target_ns);
    if (!g_state.session) {
        ADPF_LOGE("createSession failed (n_tids=%d target_ns=%" PRId64 ")",
                  et.n, g_state.target_ns);
        g_state.initialized = true;
        return;
    }

    pthread_mutex_init(&g_state.tid_lock, NULL);
    /* Pre-allocate the reusable AWorkDuration object once. If the
     * symbols aren't available (Android < 14) we leave it NULL and
     * the split-report path falls back to the single-value API. */
    if (p_AWorkDuration_create && p_reportActualWorkDuration2) {
        g_state.work_duration = p_AWorkDuration_create();
        if (!g_state.work_duration) {
            ADPF_LOGW("AWorkDuration_create returned NULL — split reports disabled");
        }
    }
    g_state.n_tids = et.n;
    for (int i = 0; i < et.n; i++) {
        g_state.tids[i] = et.tids[i];
    }
    size_t off = 0;
    for (int i = 0; i < et.n && off < sizeof(g_state.tid_summary) - 1; i++) {
        int n = snprintf(g_state.tid_summary + off,
                         sizeof(g_state.tid_summary) - off,
                         "%s%s(%d)", i ? "," : "",
                         et.names[i], (int)et.tids[i]);
        if (n < 0 || (size_t)n >= sizeof(g_state.tid_summary) - off) break;
        off += (size_t)n;
    }

    ADPF_LOGI("session created: n_tids=%d target=%" PRId64 " ns period=%" PRId64 " ns",
              et.n, g_state.target_ns, g_state.report_period_ns);
    if (g_state.debug) {
        ADPF_LOGI("  tids = %s", g_state.tid_summary);
    }

    __atomic_store_n(&g_state.reporter_running, true, __ATOMIC_RELEASE);
    if (pthread_create(&g_state.reporter, NULL, reporter_loop, NULL) != 0) {
        ADPF_LOGE("reporter thread create failed");
        __atomic_store_n(&g_state.reporter_running, false, __ATOMIC_RELEASE);
    }

    g_state.initialized = true;
#endif /* __ANDROID__ */
}

void adpf_android_add_thread(int tid)
{
#ifdef __ANDROID__
    if (!g_state.session || tid <= 0) return;
    /* setThreads is API 34+. On older Android the session's TID list is
     * frozen at creation time and there's no way to extend it; degrade
     * to a logged no-op so the user knows the thread isn't being
     * managed by ADPF. The sched-android affinity layer still works
     * via direct syscall — only uclamp.min stays at 0. */
    if (!p_setThreads) {
        if (g_state.debug) {
            ADPF_LOGW("add_thread(%d) skipped — APerformanceHint_setThreads "
                      "is API 34+ and not available on this device", tid);
        }
        return;
    }

    pthread_mutex_lock(&g_state.tid_lock);
    bool already = false;
    for (int i = 0; i < g_state.n_tids; i++) {
        if (g_state.tids[i] == tid) { already = true; break; }
    }
    if (already || g_state.n_tids >= MAX_SESSION_TIDS) {
        pthread_mutex_unlock(&g_state.tid_lock);
        return;
    }
    g_state.tids[g_state.n_tids++] = (int32_t)tid;
    int n = g_state.n_tids;
    int32_t snapshot[MAX_SESSION_TIDS];
    memcpy(snapshot, g_state.tids, (size_t)n * sizeof(int32_t));
    pthread_mutex_unlock(&g_state.tid_lock);

    int rc = p_setThreads(g_state.session, snapshot, (size_t)n);
    if (rc != 0) {
        ADPF_LOGW("setThreads add tid=%d failed: rc=%d (errno=%d %s)",
                  tid, rc, errno, strerror(errno));
        /* Don't pop the TID from our list — the kernel may have accepted
         * a subset, and we'll just re-push on next add. */
    } else if (g_state.debug) {
        ADPF_LOGI("setThreads added tid=%d (n_tids=%d)", tid, n);
    }
#else
    (void)tid;
#endif
}

void adpf_android_set_target(int64_t target_ns)
{
#ifdef __ANDROID__
    if (!g_state.session || !p_updateTargetWorkDuration) return;
    /* Clamp from above. IPowerHintSession only raises uclamp.min when
     * reported_actual > target — so callers that adaptively raise the
     * target to match a slipping frame time silently disable the boost
     * feedback. Cap defaults to X1BOX_ADPF_TARGET_NS (the init-time
     * ambition, 30 FPS by default), NOT g_state.target_ns — the latter
     * is mutated by every call below and would let the cap latch to
     * the first observed target. User can widen via
     * X1BOX_ADPF_TARGET_MAX_NS if a title actually wants a higher
     * rate (e.g. 60 FPS = 16666666). */
    int64_t cap = cfg_i64("X1BOX_ADPF_TARGET_MAX_NS",
                          cfg_i64("X1BOX_ADPF_TARGET_NS", 33333333));
    if (cap > 0 && target_ns > cap) {
        if (g_state.debug) {
            ADPF_LOGI("set_target(%" PRId64 ") clamped to %" PRId64,
                      target_ns, cap);
        }
        target_ns = cap;
    }
    g_state.target_ns = target_ns;
    p_updateTargetWorkDuration(g_state.session, target_ns);
    if (g_state.debug) {
        ADPF_LOGI("set_target(%" PRId64 " ns)", target_ns);
    }
#else
    (void)target_ns;
#endif
}

void adpf_android_report_frame(int64_t actual_ns)
{
#ifdef __ANDROID__
    if (!g_state.session || !p_reportActualWorkDuration) return;
    if (actual_ns <= 0) return;
    p_reportActualWorkDuration(g_state.session, actual_ns);
    /* Mark that a real per-frame report came through so the polling
     * reporter loop can step down. */
    __atomic_store_n(&g_state.last_real_report_ns, monotonic_ns(),
                     __ATOMIC_RELAXED);
#else
    (void)actual_ns;
#endif
}

void adpf_android_report_frame_split(int64_t cpu_ns, int64_t gpu_ns,
                                     int64_t start_ns)
{
#ifdef __ANDROID__
    if (!g_state.session) return;
    if (cpu_ns <= 0) return;

    int64_t total_ns = cpu_ns + (gpu_ns > 0 ? gpu_ns : 0);
    /* On API 34+ with the split-report symbols and AWorkDuration
     * allocated, use the rich path so the system can see CPU vs GPU
     * separately and boost the binding side only. */
    if (g_state.work_duration && p_reportActualWorkDuration2) {
        if (start_ns > 0 && p_AWorkDuration_setWorkPeriodStartTimestampNanos) {
            p_AWorkDuration_setWorkPeriodStartTimestampNanos(
                g_state.work_duration, start_ns);
        }
        if (p_AWorkDuration_setActualTotalDurationNanos) {
            p_AWorkDuration_setActualTotalDurationNanos(
                g_state.work_duration, total_ns);
        }
        if (p_AWorkDuration_setActualCpuDurationNanos) {
            p_AWorkDuration_setActualCpuDurationNanos(
                g_state.work_duration, cpu_ns);
        }
        if (p_AWorkDuration_setActualGpuDurationNanos) {
            /* Per ADPF docs, gpu_ns = 0 tells the system "no GPU work
             * pending, don't boost GPU clocks." Exactly what we want
             * when xemu is CPU-bound (the common case for Halo 2 on
             * Tensor G4 — Mali is the cool side). */
            p_AWorkDuration_setActualGpuDurationNanos(
                g_state.work_duration, gpu_ns < 0 ? 0 : gpu_ns);
        }
        p_reportActualWorkDuration2(g_state.session, g_state.work_duration);
    } else if (p_reportActualWorkDuration) {
        /* API 33 fallback — collapse to total. ADPF can't see the
         * CPU/GPU split but at least gets a real frame-work number. */
        p_reportActualWorkDuration(g_state.session, total_ns);
    } else {
        return;
    }
    __atomic_store_n(&g_state.last_real_report_ns, monotonic_ns(),
                     __ATOMIC_RELAXED);
#else
    (void)cpu_ns; (void)gpu_ns; (void)start_ns;
#endif
}

void adpf_android_shutdown(void)
{
#ifdef __ANDROID__
    if (!g_state.initialized) return;
    if (g_state.reporter_running) {
        __atomic_store_n(&g_state.reporter_running, false, __ATOMIC_RELEASE);
        pthread_join(g_state.reporter, NULL);
    }
    if (g_state.session && p_closeSession) {
        p_closeSession(g_state.session);
        g_state.session = NULL;
    }
    if (g_state.work_duration && p_AWorkDuration_release) {
        p_AWorkDuration_release(g_state.work_duration);
        g_state.work_duration = NULL;
    }
    g_state.initialized = false;
#endif
}

char *adpf_android_describe(void)
{
    char buf[512];
    snprintf(buf, sizeof(buf),
             "initialized=%d session=%p target_ns=%" PRId64
             " period_ns=%" PRId64 " n_tids=%d\n  tids: %s\n",
             (int)g_state.initialized, (void *)g_state.session,
             g_state.target_ns, g_state.report_period_ns,
             g_state.n_tids,
             g_state.tid_summary[0] ? g_state.tid_summary : "(none)");
    return strdup(buf);
}
