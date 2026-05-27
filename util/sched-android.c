/*
 * Android-specific per-thread CPU scheduling. See sched-android.h for
 * background and tunables.
 *
 * What this does, in one paragraph: on init, it parses
 * /sys/devices/system/cpu/cpuN/cpu_capacity for every online CPU and
 * classifies cores into little / mid / big buckets by capacity. Then
 * each time a QEMU thread starts, we look up the thread name in a
 * static policy table, and, if a match exists, issue sched_setattr(2)
 * with SCHED_FLAG_UTIL_CLAMP_MIN (so the freq governor doesn't idle the
 * core) plus sched_setaffinity(2) to constrain placement. Both are
 * raw syscalls because bionic doesn't export them on every NDK level.
 *
 * The table is intentionally simple — name prefix → policy class — and
 * each profile maps classes to (uclamp.min, allowed-cluster-mask). The
 * user can override class values via env vars; defaults are tuned
 * against the Halo 2 Pixel 10a benchmark (see the benchmark write-up
 * in project_halo2_pixel_sched_underclock.md).
 */

#include "qemu/osdep.h"
#include "qemu/sched-android.h"
#include "qemu/adpf-android.h"
#include "qemu/android-paths.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <dirent.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* We use a raw uint64_t bitmask for CPU affinity rather than cpu_set_t —
 * bionic's <sched.h> only provides the CPU_* macros when _GNU_SOURCE is
 * defined, and the osdep include order doesn't reliably set it on
 * Android NDK r26+. pfifo.c uses the same workaround. The bitmask
 * supports up to 64 CPUs which exceeds anything we'll see on a phone. */
typedef uint64_t cpu_mask_t;
#define CPU_MASK_ZERO()        ((cpu_mask_t)0)
#define CPU_MASK_SET(i, m)     (*(m) |= ((cpu_mask_t)1 << (i)))
#define CPU_MASK_COUNT(m)      ((int)__builtin_popcountll(m))
#define CPU_MASK_GET(i, m)     (((m) >> (i)) & 1)

#define LOG_TAG "x1box-sched"

#ifdef __ANDROID__
#define SCHED_LOGI(fmt, ...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, fmt, ##__VA_ARGS__)
#define SCHED_LOGW(fmt, ...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, fmt, ##__VA_ARGS__)
#define SCHED_LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##__VA_ARGS__)
#else
#define SCHED_LOGI(fmt, ...) ((void)0)
#define SCHED_LOGW(fmt, ...) ((void)0)
#define SCHED_LOGE(fmt, ...) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* sched_setattr glue                                                  */
/* ------------------------------------------------------------------ */

#ifndef SCHED_FLAG_KEEP_PARAMS
#define SCHED_FLAG_KEEP_PARAMS    0x10
#endif
#ifndef SCHED_FLAG_UTIL_CLAMP_MIN
#define SCHED_FLAG_UTIL_CLAMP_MIN 0x20
#endif
#ifndef SCHED_FLAG_UTIL_CLAMP_MAX
#define SCHED_FLAG_UTIL_CLAMP_MAX 0x40
#endif

#ifndef __NR_sched_setattr
#  if defined(__aarch64__)
#    define __NR_sched_setattr 274
#  elif defined(__x86_64__)
#    define __NR_sched_setattr 314
#  elif defined(__arm__)
#    define __NR_sched_setattr 380
#  elif defined(__i386__)
#    define __NR_sched_setattr 351
#  endif
#endif

/* Mirror of struct sched_attr from <linux/sched/types.h>, kept local
 * because the header isn't reliably present on every NDK level. */
struct x1b_sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    /* SCHED_DEADLINE */
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
    /* SCHED_FLAG_UTIL_CLAMP_* */
    uint32_t sched_util_min;
    uint32_t sched_util_max;
};

static int x1b_sched_setattr(pid_t pid, const struct x1b_sched_attr *attr,
                             unsigned int flags)
{
#ifdef __NR_sched_setattr
    return (int)syscall(__NR_sched_setattr, pid, attr, flags);
#else
    (void)pid; (void)attr; (void)flags;
    errno = ENOSYS;
    return -1;
#endif
}

/* ------------------------------------------------------------------ */
/* Topology detection                                                  */
/* ------------------------------------------------------------------ */

#define MAX_CPUS 32

struct topology {
    int n_cpus;
    int capacity[MAX_CPUS];   /* 0..1024 from cpu_capacity; -1 if unknown */
    int max_capacity;
    int min_capacity;
    int biggest_cpu;          /* highest-capacity CPU; -1 if no info */
    cpu_mask_t big_mask;      /* highest-capacity cluster */
    cpu_mask_t midbig_mask;   /* everything except the smallest-capacity cluster */
    /* Mid-only: midbig minus the biggest cluster. Used for HOT_WARM pin —
     * keeps pfifo on the A720 cluster (Tensor G4) / A715-A710 cluster
     * (SD 8 Gen 2) so it never contends with TCG for the single big core.
     * Empty when topology has no clear three-tier split (then HOT_WARM
     * falls back to the midbig mask, same as WARM). */
    cpu_mask_t mid_only_mask;
    cpu_mask_t all_mask;
    bool valid;
};

static struct topology g_topo;

static int read_int_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static bool detect_topology(struct topology *t)
{
    memset(t, 0, sizeof(*t));
    t->big_mask = CPU_MASK_ZERO();
    t->midbig_mask = CPU_MASK_ZERO();
    t->mid_only_mask = CPU_MASK_ZERO();
    t->all_mask = CPU_MASK_ZERO();
    t->biggest_cpu = -1;
    t->max_capacity = 0;
    t->min_capacity = INT32_MAX;

    /* Use sysconf to get processor count — works without parsing
     * /proc/cpuinfo. Online-only is fine for our purposes. */
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online <= 0 || online > MAX_CPUS) {
        return false;
    }
    t->n_cpus = (int)online;

    int known = 0;
    for (int i = 0; i < t->n_cpus; i++) {
        CPU_MASK_SET(i, &t->all_mask);
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
        int cap = read_int_file(path);
        t->capacity[i] = cap;
        if (cap > 0) {
            known++;
            if (cap > t->max_capacity) {
                t->max_capacity = cap;
                t->biggest_cpu  = i;
            }
            if (cap < t->min_capacity) t->min_capacity = cap;
        }
    }

    if (known == 0) {
        /* No cpu_capacity info (older kernel or non-bigLITTLE SoC).
         * Mark valid=false; the apply step will skip affinity but can
         * still set uclamp on each thread if asked. */
        return false;
    }

    /* big_mask = cpus with capacity == max_capacity.
     * midbig_mask = cpus with capacity > min_capacity (excludes
     *               smallest cluster only). */
    for (int i = 0; i < t->n_cpus; i++) {
        if (t->capacity[i] < 0) continue;
        if (t->capacity[i] == t->max_capacity) CPU_MASK_SET(i, &t->big_mask);
        if (t->capacity[i]  > t->min_capacity) CPU_MASK_SET(i, &t->midbig_mask);
    }
    /* If everything has the same capacity (no bigLITTLE), midbig_mask
     * would be empty and big_mask = all; treat midbig as all. */
    if (CPU_MASK_COUNT(t->midbig_mask) == 0) {
        t->midbig_mask = t->all_mask;
    }

    /* mid_only = midbig minus big. Empty for a 2-tier SoC (only big +
     * little); for 3-tier (Tensor G4: X4 + 3 A720 + 4 A520, SD 8 Gen 2:
     * X3 + 4 A715/A710 + 3 A510), this is the middle cluster. */
    t->mid_only_mask = t->midbig_mask & ~t->big_mask;

    t->valid = true;
    return true;
}

/* ------------------------------------------------------------------ */
/* Profile + per-class policy                                          */
/* ------------------------------------------------------------------ */

typedef enum {
    POLICY_NONE = 0,
    POLICY_HOT,        /* CPU N/TCG — pinned to the biggest core */
    POLICY_HOT_WARM,   /* nv2a.pfifo — pinned to the mid cluster */
    POLICY_WARM,       /* mcpx/voice/render/cranelift — exclude little only */
    POLICY_COOL,       /* qemu_main / SDL / event threads — light touch */
    POLICY_COUNT,
} PolicyClass;

/* Name-prefix → class. First match wins; order matters. nv2a.pfifo gets
 * its own HOT_WARM class because it's the next-loudest thread after TCG
 * and shares vertex/draw cache lines with TCG's writes — keeping it on
 * a dedicated, uclamp-max'd mid core (rather than letting the scheduler
 * shuffle it through whichever A720 looks idle) noticeably cuts the
 * cache-miss tail on tlb_reset_dirty per the project benchmarks. */
struct name_class { const char *prefix; PolicyClass cls; };
static const struct name_class k_classes[] = {
    { "CPU ",           POLICY_HOT      },   /* "CPU 0/TCG", "CPU 1/TCG", ... */
    { "nv2a.pfifo",     POLICY_HOT_WARM },
    /* AudioTrack is Android's JNI audio-callback thread, created by SDL
     * when it opens an audio device. Empirically lands on the big core
     * (cpu7 on Tensor G4) and steals slices from CPU 0/TCG. WARM is the
     * right class — low-CPU but we still want it off the big core; the
     * WARM pin_mid_warm path gives it a dedicated mid core. */
    { "AudioTrack",     POLICY_WARM     },
    { "cranelift-tcg",  POLICY_WARM     },
    { "pgraph.vk.rend", POLICY_WARM     },
    { "pgraph.vk.comp", POLICY_WARM     },
    { "mcpx.apu",       POLICY_WARM     },
    { "mcpx.voice",     POLICY_WARM     },
    { "SDLAudioP2",     POLICY_WARM     },
    { "qemu_main",      POLICY_COOL     },
    { "nv2a.pgraph",    POLICY_WARM     },
    { NULL, POLICY_NONE },
};

static PolicyClass classify(const char *name)
{
    if (!name || !*name) return POLICY_NONE;
    for (const struct name_class *c = k_classes; c->prefix; c++) {
        if (strncmp(name, c->prefix, strlen(c->prefix)) == 0) {
            return c->cls;
        }
    }
    return POLICY_NONE;
}

/* Resolved per-class policy values; populated by sched_android_init from
 * the active profile + env overrides. */
struct class_policy {
    uint32_t uclamp_min;       /* 0 = don't set */
    bool     set_affinity;     /* whether to call sched_setaffinity */
    bool     pin_biggest;      /* affinity = single biggest CPU only */
    bool     pin_mid;          /* affinity = a single hashed mid core */
    bool     exclude_little;   /* affinity = midbig (skip little cluster) */
};

/*
 * On pin_mid: we use the whole mid_only_mask (e.g. cpu4-6 on Tensor G4)
 * instead of pinning each thread to a single mid core. Tried per-thread
 * single-core pinning on 2026-05-26 (Halo 2 Pixel): each WARM thread on
 * its own cpu had per-core utilization too low (~15-20%) to convince
 * sched_pixel to bump the cluster freq — cores stayed at the 578 MHz
 * idle floor. Letting the kernel float threads within mid_only_mask
 * lets EAS collapse them onto whichever core wins the EAS lottery,
 * which earns a freq bump. The cost is some intra-cluster bouncing,
 * but never to cpu7 (which is reserved for CPU 0/TCG).
 */

static struct sched_config {
    SchedProfile profile;
    bool debug;
    bool boost_enabled;        /* runtime toggle (foreground/background) */
    struct class_policy by_class[POLICY_COUNT];
} g_cfg;

/* Config-file path (Android-only; `am start` doesn't propagate env vars).
 * Each line is `KEY=VALUE`; lines starting with '#' or empty are skipped.
 * Values from this file OVERRIDE env vars so the user has a writable
 * knob without rebuilding/restart-with-env. Path is per-package, so
 * the perftest flavor reads its own data dir. */
#define CONFIG_LINES_MAX 64
#define CONFIG_LINE_MAX  256

struct kv { char k[64]; char v[128]; };
static struct kv g_file_kv[CONFIG_LINES_MAX];
static int       g_file_kv_n;

static void load_config_file(void)
{
    g_file_kv_n = 0;
#ifdef __ANDROID__
    const char *base = android_x1box_ext_dir();
    if (!base) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/sched_config.txt", base);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[CONFIG_LINE_MAX];
    while (fgets(line, sizeof(line), f) && g_file_kv_n < CONFIG_LINES_MAX) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\0' || *p == '\n') continue;
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = '\0';
        char *k = p, *v = eq + 1;
        /* rtrim k */
        size_t kl = strlen(k);
        while (kl && (k[kl-1] == ' ' || k[kl-1] == '\t')) k[--kl] = '\0';
        /* trim v */
        while (*v == ' ' || *v == '\t') v++;
        size_t vl = strlen(v);
        while (vl && (v[vl-1] == '\n' || v[vl-1] == '\r'
                      || v[vl-1] == ' ' || v[vl-1] == '\t')) v[--vl] = '\0';
        if (!*k) continue;
        struct kv *e = &g_file_kv[g_file_kv_n++];
        strncpy(e->k, k, sizeof(e->k) - 1); e->k[sizeof(e->k)-1] = '\0';
        strncpy(e->v, v, sizeof(e->v) - 1); e->v[sizeof(e->v)-1] = '\0';
    }
    fclose(f);
#endif
}

static const char *lookup_str(const char *name)
{
    /* file overrides env */
    for (int i = 0; i < g_file_kv_n; i++) {
        if (!strcmp(g_file_kv[i].k, name)) return g_file_kv[i].v;
    }
    return getenv(name);
}

static int env_int(const char *name, int dflt)
{
    const char *v = lookup_str(name);
    if (!v || !*v) return dflt;
    char *end = NULL;
    long n = strtol(v, &end, 10);
    if (!end || *end != '\0') return dflt;
    return (int)n;
}

static bool env_bool(const char *name, bool dflt)
{
    const char *v = lookup_str(name);
    if (!v || !*v) return dflt;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

static SchedProfile parse_profile(const char *v)
{
    if (!v || !*v) return X1BOX_SCHED_OFF;
    if (!strcasecmp(v, "off")      || !strcmp(v, "0")) return X1BOX_SCHED_OFF;
    if (!strcasecmp(v, "balanced") || !strcmp(v, "1")) return X1BOX_SCHED_BALANCED;
    if (!strcasecmp(v, "max")      || !strcmp(v, "2")) return X1BOX_SCHED_MAX;
    return X1BOX_SCHED_OFF;
}

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

static void apply_profile_defaults(SchedProfile p, struct sched_config *c)
{
    /* Defaults per profile — overridden by env below. */
    int hot_min = 0, hotwarm_min = 0, warm_min = 0, cool_min = 0;
    bool excl_little_hot = false, excl_little_warm = false;
    bool pin_big_hot = false;
    bool pin_mid_pfifo = false;
    bool pin_mid_warm = false;
    switch (p) {
    case X1BOX_SCHED_OFF:
        /* leave everything 0/false */
        break;
    case X1BOX_SCHED_BALANCED:
        hot_min          = 512;   /* ask for ~50% capacity */
        hotwarm_min      = 512;   /* pfifo treated as a real worker, not a helper */
        warm_min         = 256;
        cool_min         = 0;
        excl_little_hot  = true;
        excl_little_warm = false;
        pin_big_hot      = false;
        pin_mid_pfifo    = true;  /* pfifo + AudioTrack get a dedicated mid core */
        pin_mid_warm     = false;
        break;
    case X1BOX_SCHED_MAX:
        hot_min          = 1024;  /* full capacity request */
        hotwarm_min      = 1024;  /* pfifo: ask for full capacity on its mid core */
        warm_min         = 768;
        cool_min         = 256;
        excl_little_hot  = true;
        excl_little_warm = true;
        pin_big_hot      = true;
        pin_mid_pfifo    = true;
        /* MAX hashes each WARM thread name to its own mid core so
         * mcpx.apu_thread, pgraph.vk.rende, cranelift-tcg etc. don't
         * fight inside the mid cluster — measured 2026-05-26 with
         * Halo 2: each was bouncing across 3 cores at 16-43% per
         * core, none warming any single cache. */
        pin_mid_warm     = true;
        break;
    }

    hot_min          = env_int("X1BOX_SCHED_UCLAMP_HOT",     hot_min);
    hotwarm_min      = env_int("X1BOX_SCHED_UCLAMP_HOTWARM", hotwarm_min);
    warm_min         = env_int("X1BOX_SCHED_UCLAMP_WARM",    warm_min);
    cool_min         = env_int("X1BOX_SCHED_UCLAMP_COOL",    cool_min);
    excl_little_hot  = env_bool("X1BOX_SCHED_EXCLUDE_LITTLE", excl_little_hot);
    /* Apply EXCLUDE_LITTLE both to hot and warm so a single env var
     * matches user expectation; finer split needs class-specific vars. */
    excl_little_warm = excl_little_hot;
    pin_big_hot      = env_bool("X1BOX_SCHED_PIN_TCG_BIG",   pin_big_hot);
    pin_mid_pfifo    = env_bool("X1BOX_SCHED_PIN_PFIFO_MID", pin_mid_pfifo);
    pin_mid_warm     = env_bool("X1BOX_SCHED_PIN_WARM_MID",  pin_mid_warm);

    c->by_class[POLICY_NONE] = (struct class_policy){0};
    c->by_class[POLICY_HOT]  = (struct class_policy){
        .uclamp_min     = (uint32_t)(hot_min > 0 ? hot_min : 0),
        .set_affinity   = excl_little_hot || pin_big_hot,
        .pin_biggest    = pin_big_hot,
        .pin_mid        = false,
        .exclude_little = excl_little_hot,
    };
    /* HOT_WARM: nv2a.pfifo. pin_mid uses mid_only_mask (= midbig & ~big)
     * so pfifo lives on the A720 cluster and never competes with TCG for
     * the X4. If the SoC isn't 3-tier (mid_only_mask is empty), the
     * apply path falls back to the midbig mask, matching WARM. */
    c->by_class[POLICY_HOT_WARM] = (struct class_policy){
        .uclamp_min     = (uint32_t)(hotwarm_min > 0 ? hotwarm_min : 0),
        .set_affinity   = excl_little_warm || pin_mid_pfifo,
        .pin_biggest    = false,
        .pin_mid        = pin_mid_pfifo,
        .exclude_little = excl_little_warm,
    };
    c->by_class[POLICY_WARM] = (struct class_policy){
        .uclamp_min     = (uint32_t)(warm_min > 0 ? warm_min : 0),
        .set_affinity   = excl_little_warm || pin_mid_warm,
        .pin_biggest    = false,
        .pin_mid        = pin_mid_warm,
        .exclude_little = excl_little_warm,
    };
    c->by_class[POLICY_COOL] = (struct class_policy){
        .uclamp_min     = (uint32_t)(cool_min > 0 ? cool_min : 0),
        .set_affinity   = false,
        .pin_biggest    = false,
        .pin_mid        = false,
        .exclude_little = false,
    };
}

static void init_once_locked(void)
{
    memset(&g_cfg, 0, sizeof(g_cfg));
    load_config_file();
    g_cfg.profile = parse_profile(lookup_str("X1BOX_SCHED_PROFILE"));
    g_cfg.debug   = env_bool("X1BOX_SCHED_DEBUG", false);
    g_cfg.boost_enabled = true;

    bool topo_ok = detect_topology(&g_topo);
    apply_profile_defaults(g_cfg.profile, &g_cfg);

#ifdef __ANDROID__
    SCHED_LOGI("init: profile=%d debug=%d topo_valid=%d "
               "n_cpus=%d biggest_cpu=%d min_cap=%d max_cap=%d",
               (int)g_cfg.profile, (int)g_cfg.debug, (int)topo_ok,
               g_topo.n_cpus, g_topo.biggest_cpu,
               g_topo.min_capacity, g_topo.max_capacity);
    if (g_cfg.debug) {
        for (int i = 0; i < g_topo.n_cpus; i++) {
            SCHED_LOGI("  cpu%d capacity=%d", i, g_topo.capacity[i]);
        }
        SCHED_LOGI("  mid_only_mask=0x%llx  midbig_mask=0x%llx  big_mask=0x%llx",
                   (unsigned long long)g_topo.mid_only_mask,
                   (unsigned long long)g_topo.midbig_mask,
                   (unsigned long long)g_topo.big_mask);
        SCHED_LOGI("  policy HOT      : uclamp=%u  pin_big=%d  excl_little=%d",
                   g_cfg.by_class[POLICY_HOT].uclamp_min,
                   (int)g_cfg.by_class[POLICY_HOT].pin_biggest,
                   (int)g_cfg.by_class[POLICY_HOT].exclude_little);
        SCHED_LOGI("  policy HOT_WARM : uclamp=%u  pin_mid=%d  excl_little=%d",
                   g_cfg.by_class[POLICY_HOT_WARM].uclamp_min,
                   (int)g_cfg.by_class[POLICY_HOT_WARM].pin_mid,
                   (int)g_cfg.by_class[POLICY_HOT_WARM].exclude_little);
        SCHED_LOGI("  policy WARM     : uclamp=%u  pin_mid=%d  excl_little=%d",
                   g_cfg.by_class[POLICY_WARM].uclamp_min,
                   (int)g_cfg.by_class[POLICY_WARM].pin_mid,
                   (int)g_cfg.by_class[POLICY_WARM].exclude_little);
        SCHED_LOGI("  policy COOL     : uclamp=%u",
                   g_cfg.by_class[POLICY_COOL].uclamp_min);
    }
#else
    (void)topo_ok;
#endif
}

void sched_android_init(void)
{
    pthread_once(&g_init_once, init_once_locked);
}

/* ------------------------------------------------------------------ */
/* Apply                                                               */
/* ------------------------------------------------------------------ */

/* tid=0 means caller; non-zero targets a specific TID within our own
 * process — sched_setattr/sched_setaffinity both accept this for any
 * task we own, no special capability needed (uclamp.min raise is the
 * one operation gated on CAP_SYS_NICE, which we know EPERMs). */
static void apply_uclamp_min(pid_t tid, uint32_t min_val)
{
    struct x1b_sched_attr a;
    memset(&a, 0, sizeof(a));
    a.size           = sizeof(a);
    a.sched_policy   = SCHED_OTHER;
    a.sched_flags    = SCHED_FLAG_UTIL_CLAMP_MIN | SCHED_FLAG_KEEP_PARAMS;
    a.sched_util_min = min_val;
    a.sched_util_max = 1024;
    if (x1b_sched_setattr(tid, &a, 0) != 0) {
        if (g_cfg.debug) {
            SCHED_LOGW("sched_setattr(tid=%d uclamp.min=%u) failed: errno=%d (%s)",
                       (int)tid, min_val, errno, strerror(errno));
        }
    }
}

static void apply_affinity(pid_t tid, cpu_mask_t mask)
{
    /* Use raw syscall path — matches pfifo.c and avoids needing
     * pthread_setaffinity_np across NDK versions. The kernel
     * sched_setaffinity ABI accepts a bitmap of arbitrary length. */
    if (syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask) != 0) {
        if (g_cfg.debug) {
            SCHED_LOGW("sched_setaffinity(tid=%d) failed: errno=%d (%s)",
                       (int)tid, errno, strerror(errno));
        }
    }
}

/* Core policy applier — works for caller (tid=0) or any peer TID
 * (tid != 0). Used by apply_for_thread (per-creation hook in
 * qemu-thread-posix) and the late-scanner loop below (catches threads
 * spawned outside qemu_thread_create, like Rust std::thread::spawn
 * for cranelift-tcg). */
/*
 * Cemu's signal that we've been missing: setpriority(nice=-10) on hot
 * threads. Stock Android allows unprivileged apps to lower nice down to
 * ~−10 without CAP_SYS_NICE; system_server's task profiles set ~−10 on
 * the foreground app's primary thread for the same reason. Lowering nice
 * makes sched_pixel's per-task EAS treat the thread as priority work,
 * which (a) reduces the chance of it being pre-empted by small system
 * threads, and (b) raises the per-cluster utilization estimate that
 * drives the cpufreq governor — without us trying to write uclamp.min
 * (which EPERMs for unprivileged apps).
 *
 * Per Cemu/coreinit_Thread.cpp:1556 + helpers.cpp:702.
 */
static void apply_nice(pid_t tid, int nice_value)
{
#ifdef __ANDROID__
    /* setpriority(PRIO_PROCESS, who=0) sets the calling task on Linux;
     * for a target tid we use the syscall directly with PRIO_PROCESS.
     * The kernel allows lowering nice for tasks within the same process
     * (same uid) without CAP_SYS_NICE up to RLIMIT_NICE (default 20,
     * i.e. nice >= 0). Below 0 needs CAP_SYS_NICE in theory; in practice
     * the foreground-app task profile leaves us at nice=-2 and we can
     * lower from there to about −10. Silently ignore EPERM. */
    if (syscall(__NR_setpriority, /*PRIO_PROCESS=*/0, tid, nice_value) != 0) {
        if (g_cfg.debug && errno != EACCES && errno != EPERM) {
            SCHED_LOGW("setpriority(tid=%d nice=%d) failed: errno=%d (%s)",
                       (int)tid, nice_value, errno, strerror(errno));
        }
    }
#else
    (void)tid; (void)nice_value;
#endif
}

static void apply_policy_to(pid_t tid, const char *name)
{
    if (g_cfg.profile == X1BOX_SCHED_OFF) return;
    PolicyClass cls = classify(name);
    if (cls == POLICY_NONE) return;
    struct class_policy *p = &g_cfg.by_class[cls];

    if (p->set_affinity && g_topo.valid) {
        if (p->pin_biggest && g_topo.biggest_cpu >= 0) {
            cpu_mask_t m = CPU_MASK_ZERO();
            CPU_MASK_SET(g_topo.biggest_cpu, &m);
            apply_affinity(tid, m);
        } else if (p->pin_mid && CPU_MASK_COUNT(g_topo.mid_only_mask) > 0) {
            apply_affinity(tid, g_topo.mid_only_mask);
        } else if (p->exclude_little) {
            apply_affinity(tid, g_topo.midbig_mask);
        }
    }

    uint32_t want = g_cfg.boost_enabled ? p->uclamp_min : 0;
    if (want > 0) {
        apply_uclamp_min(tid, want);
    }

    /*
     * Per-class nice value. Hot threads get the deepest priority drop
     * (Cemu uses -10 for its OSSched threads); cooler classes get less
     * so they don't starve genuinely-idle system work.
     */
    int nice_value = 0;
    switch (cls) {
    case POLICY_HOT:       nice_value = -10; break;
    case POLICY_HOT_WARM:  nice_value =  -8; break;
    case POLICY_WARM:      nice_value =  -5; break;
    case POLICY_COOL:      nice_value =   0; break;
    default: break;
    }
    if (nice_value < 0) {
        apply_nice(tid, nice_value);
    }

    if (g_cfg.debug) {
        SCHED_LOGI("apply tid=%d name='%s' class=%d uclamp=%u nice=%d "
                   "set_aff=%d pin_big=%d pin_mid=%d excl_little=%d",
                   (int)tid, name, (int)cls, want, nice_value,
                   (int)p->set_affinity, (int)p->pin_biggest,
                   (int)p->pin_mid, (int)p->exclude_little);
    }
}

/* ------------------------------------------------------------------ */
/* Scanner thread — catches threads spawned outside qemu_thread_create */
/* ------------------------------------------------------------------ */
/*
 * cranelift-tcg (and any other Rust std::thread::spawn child) never
 * goes through util/qemu-thread-posix.c's hook, so apply_for_thread is
 * never called on it. Result: it inherits the parent thread's affinity
 * (often cpu=7 because TCG creates it) and never gets the policy
 * we'd otherwise set for its WARM class (cpus=4-7).
 *
 * Fix: a small background scanner thread that periodically walks
 * /proc/self/task, classifies each TID by comm, and applies policy
 * to any new perf-class TIDs we haven't already handled. The actual
 * apply uses sched_setattr/sched_setaffinity with the target TID as
 * the first arg — both syscalls allow peer-task modification within
 * the same process without special privilege (except uclamp.min raise,
 * which we already know EPERMs; ADPF handles that side).
 */

#define SCANNER_TID_TRACK_MAX 256

static pid_t      g_scanner_tids[SCANNER_TID_TRACK_MAX];
static int        g_scanner_tids_n;
static pthread_t  g_scanner_thread;
static bool       g_scanner_running;

static bool scanner_seen(pid_t tid)
{
    for (int i = 0; i < g_scanner_tids_n; i++) {
        if (g_scanner_tids[i] == tid) return true;
    }
    return false;
}

static void scanner_remember(pid_t tid)
{
    if (g_scanner_tids_n < SCANNER_TID_TRACK_MAX) {
        g_scanner_tids[g_scanner_tids_n++] = tid;
    }
}

static int read_task_comm(pid_t tid, char *out, size_t out_sz)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/task/%d/comm", (int)tid);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(out, (int)out_sz, f)) { fclose(f); return -1; }
    fclose(f);
    size_t n = strlen(out);
    while (n && (out[n-1] == '\n' || out[n-1] == '\r')) out[--n] = '\0';
    return 0;
}

static void scanner_pass(void)
{
    DIR *d = opendir("/proc/self/task");
    if (!d) return;
    struct dirent *e;
    int newly_applied = 0;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        pid_t tid = (pid_t)atoi(e->d_name);
        if (tid <= 0) continue;
        if (scanner_seen(tid)) continue;
        char comm[32] = {0};
        if (read_task_comm(tid, comm, sizeof(comm)) != 0) continue;
        PolicyClass cls = classify(comm);
        if (cls == POLICY_NONE) {
            /* Don't remember non-perf threads — there are hundreds of
             * Android binder/JIT threads that come and go; tracking
             * them all would overflow our small fixed array. We will
             * re-classify them next pass (cheap, ~32-byte comm read). */
            continue;
        }
        /* Skip self — we run as the scanner thread and the policy
         * table doesn't include "x1box-sched-scan". (Defense in depth.) */
        if ((int)tid == (int)syscall(__NR_gettid)) continue;
        apply_policy_to(tid, comm);
        /* Also enroll into the ADPF session so system_server raises
         * uclamp.min on this TID. No-op on Android < 14 (setThreads
         * unavailable) — affinity is still applied. */
        adpf_android_add_thread((int)tid);
        scanner_remember(tid);
        newly_applied++;
    }
    closedir(d);
    if (g_cfg.debug && newly_applied > 0) {
        SCHED_LOGI("scanner: %d new perf-class TIDs handled (total tracked=%d)",
                   newly_applied, g_scanner_tids_n);
    }
}

/*
 * Periodically re-apply policy to every TID we already classified.
 *
 * Android can quietly migrate a thread off its pinned mask in a few
 * ways: (a) cpuset cgroup transitions when the app moves between the
 * top-app / foreground / background cgroups, (b) Game Mode interventions
 * changing affinity masks mid-play, (c) PowerManager / power-hint
 * services tinkering with sched attrs on perceived idle. Cemu's
 * pinning watchdog exists for exactly these scenarios.
 *
 * This is pure defence in depth. If nothing disturbs us, every
 * sched_setaffinity call here is a no-op (kernel sees mask is already
 * the requested one). Walks our fixed-size tracked-TID array, re-reads
 * comm to handle TID reuse (a recycled TID may now be a non-perf
 * thread we should drop), and re-applies the same policy that
 * scanner_pass() used at first-discovery time.
 */
static void scanner_repin_pass(void)
{
    pid_t self_tid = (pid_t)syscall(__NR_gettid);
    int repinned = 0;
    int dropped = 0;
    for (int i = 0; i < g_scanner_tids_n; ) {
        pid_t tid = g_scanner_tids[i];
        if (tid == self_tid) { i++; continue; }
        char comm[32] = {0};
        if (read_task_comm(tid, comm, sizeof(comm)) != 0) {
            /* Thread is gone (TID died). Compact the array. */
            g_scanner_tids[i] = g_scanner_tids[--g_scanner_tids_n];
            dropped++;
            continue;
        }
        PolicyClass cls = classify(comm);
        if (cls == POLICY_NONE) {
            /* TID was recycled into something we don't manage; drop. */
            g_scanner_tids[i] = g_scanner_tids[--g_scanner_tids_n];
            dropped++;
            continue;
        }
        apply_policy_to(tid, comm);
        repinned++;
        i++;
    }
    if (g_cfg.debug && (repinned > 0 || dropped > 0)) {
        SCHED_LOGI("scanner: re-pinned %d, dropped %d (total tracked=%d)",
                   repinned, dropped, g_scanner_tids_n);
    }
}

static void *scanner_loop(void *arg)
{
    (void)arg;
#ifdef __ANDROID__
    /* Mark this thread so classify() won't pick it up + so logcat is
     * grep-friendly. */
    syscall(__NR_prctl, 15 /* PR_SET_NAME */, "x1box-sched-scan", 0, 0, 0);
    /* Fast initial cadence to catch threads spawned during boot, then
     * slow down once everyone we care about has been seen. In steady
     * state we also run scanner_repin_pass() once per loop to defend
     * against kernel- or Game-Mode-initiated migration. */
    int fast_passes = 30;        /* 30 × 200ms = 6s of fast scanning */
    while (__atomic_load_n(&g_scanner_running, __ATOMIC_ACQUIRE)) {
        scanner_pass();
        struct timespec ts;
        if (fast_passes > 0) {
            /* During discovery phase don't bother re-pinning — every
             * iteration of scanner_pass() already touches new TIDs and
             * the kernel hasn't had time to drift anyone off mask. */
            ts.tv_sec = 0; ts.tv_nsec = 200 * 1000 * 1000;  /* 200ms */
            fast_passes--;
        } else {
            /* Steady state: 2 Hz combined sweep (match Cemu's
             * helpers.cpp:518 pinWatchdogLoop). Android's
             * SetTaskProfiles can reset affinity on any cpuset cgroup
             * change — foreground/background, top-app moves, Game Mode
             * intervention, even idle-screen-on transitions on Tensor.
             * 1 Hz left a perceptible "drift to mid for 700 ms" window
             * during which the heavy threads landed on the wrong core,
             * the cluster freq dropped, and FPS visibly hitched. 500 ms
             * keeps the worst-case drift below one Halo 2 sim tick
             * (33 ms) which is the perceptual threshold.
             *
             * Cost is negligible (one opendir, ~12 comm reads, ~12
             * sched_setaffinity no-op syscalls per pass = under
             * 0.05% CPU). */
            scanner_repin_pass();
            ts.tv_sec = 0; ts.tv_nsec = 500 * 1000 * 1000;  /* 500ms */
        }
        nanosleep(&ts, NULL);
    }
#endif
    return NULL;
}

static void scanner_start_once(void)
{
#ifdef __ANDROID__
    if (g_cfg.profile == X1BOX_SCHED_OFF) return;
    if (g_scanner_running) return;
    __atomic_store_n(&g_scanner_running, true, __ATOMIC_RELEASE);
    if (pthread_create(&g_scanner_thread, NULL, scanner_loop, NULL) != 0) {
        SCHED_LOGE("scanner thread create failed; cranelift-tcg won't be re-pinned");
        __atomic_store_n(&g_scanner_running, false, __ATOMIC_RELEASE);
    } else {
        SCHED_LOGI("scanner thread started (catches Rust-spawned threads)");
    }
#endif
}

void sched_android_apply_for_thread(const char *name)
{
#ifdef __ANDROID__
    sched_android_init();
    apply_policy_to(0, name);
    /* First time we're called from a real perf-class thread, kick the
     * scanner. We do it lazily here (rather than from init_once_locked)
     * because the scanner needs a real /proc/self/task to walk; doing
     * it before the first qemu_thread_create has fired means we'd just
     * spin uselessly on launcher threads. */
    static pthread_once_t scanner_once = PTHREAD_ONCE_INIT;
    if (classify(name) != POLICY_NONE) {
        pthread_once(&scanner_once, scanner_start_once);
    }
#else
    (void)name;
#endif
}

/* ------------------------------------------------------------------ */
/* Foreground/background hook                                          */
/* ------------------------------------------------------------------ */

void sched_android_set_boost(bool enable)
{
#ifdef __ANDROID__
    sched_android_init();
    if (g_cfg.profile == X1BOX_SCHED_OFF) return;
    if (g_cfg.boost_enabled == enable) return;
    g_cfg.boost_enabled = enable;

    /* We can't enumerate every running QEMU thread here without
     * pulling more headers; instead, we lower OUR OWN thread's uclamp
     * and rely on a future per-thread refresh path. For most apps
     * backgrounding moves us into a /background cgroup which already
     * tightens the cgroup-level uclamp ceiling, so this is mostly
     * cosmetic — but harmless. */
    if (!enable) {
        apply_uclamp_min(0, 0);  /* tid=0 self, min_val=0 to drop the floor */
    }
    SCHED_LOGI("set_boost(%d)", (int)enable);
#else
    (void)enable;
#endif
}

/* ------------------------------------------------------------------ */
/* Status                                                              */
/* ------------------------------------------------------------------ */

char *sched_android_describe(void)
{
    sched_android_init();
    /* Compose a multi-line string with key state. Caller frees. */
    char buf[1024];
    int n = snprintf(buf, sizeof(buf),
                     "profile=%d debug=%d boost=%d topo_valid=%d\n"
                     "  n_cpus=%d biggest_cpu=%d cap_min=%d cap_max=%d\n"
                     "  big_mask=0x%llx midbig_mask=0x%llx mid_only_mask=0x%llx\n"
                     "  HOT      uclamp=%u set_aff=%d pin_big=%d excl_little=%d\n"
                     "  HOT_WARM uclamp=%u set_aff=%d pin_mid=%d excl_little=%d\n"
                     "  WARM     uclamp=%u pin_mid=%d excl_little=%d\n"
                     "  COOL     uclamp=%u\n",
                     (int)g_cfg.profile, (int)g_cfg.debug,
                     (int)g_cfg.boost_enabled, (int)g_topo.valid,
                     g_topo.n_cpus, g_topo.biggest_cpu,
                     g_topo.min_capacity, g_topo.max_capacity,
                     (unsigned long long)g_topo.big_mask,
                     (unsigned long long)g_topo.midbig_mask,
                     (unsigned long long)g_topo.mid_only_mask,
                     g_cfg.by_class[POLICY_HOT].uclamp_min,
                     (int)g_cfg.by_class[POLICY_HOT].set_affinity,
                     (int)g_cfg.by_class[POLICY_HOT].pin_biggest,
                     (int)g_cfg.by_class[POLICY_HOT].exclude_little,
                     g_cfg.by_class[POLICY_HOT_WARM].uclamp_min,
                     (int)g_cfg.by_class[POLICY_HOT_WARM].set_affinity,
                     (int)g_cfg.by_class[POLICY_HOT_WARM].pin_mid,
                     (int)g_cfg.by_class[POLICY_HOT_WARM].exclude_little,
                     g_cfg.by_class[POLICY_WARM].uclamp_min,
                     (int)g_cfg.by_class[POLICY_WARM].pin_mid,
                     (int)g_cfg.by_class[POLICY_WARM].exclude_little,
                     g_cfg.by_class[POLICY_COOL].uclamp_min);
    (void)n;
    return strdup(buf);
}
