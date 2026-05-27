/*
 * util/sched-android.h — per-thread CPU scheduling policy for Android.
 *
 * Targets the Tensor G4 (sched_pixel governor) underclock bug observed
 * in Halo 2 on Pixel 10a: with the workload's bursty CPU pattern, the
 * mid-cluster A720 cores idle at ~30-35% of max scaling freq, dropping
 * effective FPS to ~16. Setting uclamp.min on hot threads via
 * sched_setattr(2) tells the scheduler the task needs sustained CPU
 * capacity; pairing with sched_setaffinity narrows where it lands.
 *
 * Power: default profile is OFF — no behavior change. The user opts in
 * via $X1BOX_SCHED_PROFILE=balanced|max. BALANCED keeps the existing
 * affinity flexible and adds a moderate uclamp. MAX pins the single
 * hottest interpreter thread (CPU N/TCG) to the biggest core.
 *
 * Tunables (env vars; default in parens):
 *   X1BOX_SCHED_PROFILE=off|balanced|max     (off)
 *   X1BOX_SCHED_UCLAMP_HOT=N                 (profile default)  uclamp.min for CPU N/TCG
 *   X1BOX_SCHED_UCLAMP_HOTWARM=N             (profile default)  uclamp.min for nv2a.pfifo
 *   X1BOX_SCHED_UCLAMP_WARM=N                (profile default)  uclamp.min for other warm threads
 *   X1BOX_SCHED_UCLAMP_COOL=N                (profile default)  uclamp.min for qemu_main / cool helpers
 *   X1BOX_SCHED_EXCLUDE_LITTLE=0|1           (profile default)  mask out the little cluster
 *   X1BOX_SCHED_PIN_TCG_BIG=0|1              (profile default)  pin CPU N/TCG to biggest core
 *   X1BOX_SCHED_PIN_PFIFO_MID=0|1            (profile default)  pin nv2a.pfifo + AudioTrack to a single
 *                                                               hashed mid core (deterministic per thread
 *                                                               name) — keeps them off the biggest core
 *                                                               (TCG's) and off the little cluster
 *   X1BOX_SCHED_PIN_WARM_MID=0|1             (profile default)  pin each WARM thread (mcpx.apu_thread,
 *                                                               pgraph.vk.rend*, cranelift-tcg, mcpx.voice,
 *                                                               SDLAudioP2, nv2a.pgraph) to a single hashed
 *                                                               mid core — stops cluster-internal bouncing
 *   X1BOX_SCHED_DEBUG=0|1                    (0)                log every policy decision
 *
 * Why the extra class: pfifo is the next-loudest thread after TCG, and once
 * TCG is pinned to cpu 7 (Cortex-X4 on Tensor G4) the scheduler will only
 * place pfifo on the mid A720s. Empirically those A720s get governor-
 * underclocked unless something asks for sustained capacity. Pinning pfifo
 * to the mid-only mask with uclamp=1024 keeps a dedicated A720 awake at
 * peak clock and avoids any chance of contending with TCG on the X4. See
 * project_pixel_sched_underclock.md.
 */

#ifndef QEMU_UTIL_SCHED_ANDROID_H
#define QEMU_UTIL_SCHED_ANDROID_H

#include <stdbool.h>

typedef enum {
    X1BOX_SCHED_OFF      = 0,
    X1BOX_SCHED_BALANCED = 1,
    X1BOX_SCHED_MAX      = 2,
} SchedProfile;

/* Policy classes — kept in this header so headers stay in sync with
 * sched-android.c. The numeric ordering is significant for the by_class
 * array. */
typedef enum {
    X1BOX_POLICY_NONE     = 0,
    X1BOX_POLICY_HOT      = 1, /* CPU N/TCG — pinned to the biggest core */
    X1BOX_POLICY_HOT_WARM = 2, /* nv2a.pfifo — pinned to the mid cluster */
    X1BOX_POLICY_WARM     = 3, /* mcpx/voice/render — exclude little only */
    X1BOX_POLICY_COOL     = 4, /* qemu_main — light boost, no affinity */
    X1BOX_POLICY_COUNT    = 5,
} SchedPolicyClass;

/* Read env, detect CPU-capacity topology. Idempotent; safe to call from
 * multiple threads (init is mutex-guarded). Cheap: ~9 sysfs reads. */
void sched_android_init(void);

/* Apply the configured profile to the calling thread, matched by name.
 * Must be called inside the target thread AFTER prctl(PR_SET_NAME, …)
 * has set /proc/self/task/<tid>/comm. Match is by prefix of `name`.
 * No-op when profile is OFF or platform is not Android. */
void sched_android_apply_for_thread(const char *name);

/* Foreground/background hook. When enable=false, the next call to
 * sched_android_apply_for_thread for any new thread will skip the
 * boost, AND previously-boosted tracked threads have their uclamp.min
 * reset to 0 (allowing the governor to power down idle cores).
 * Currently not wired to Android lifecycle — call manually from the
 * JNI layer or via a future hook in SDL_APP_*_BACKGROUND handling. */
void sched_android_set_boost(bool enable);

/* Diagnostics — exposed for the `sched_status` MCP tool / debug print.
 * Returns a malloc'd string the caller must free. */
char *sched_android_describe(void);

#endif /* QEMU_UTIL_SCHED_ANDROID_H */
