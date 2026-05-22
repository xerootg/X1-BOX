/*
 * util/adpf-android.h — Android Dynamic Performance Framework (ADPF)
 * integration. Provides the OS-supported path to raise per-task
 * uclamp.min for app-uid processes; bypasses the CAP_SYS_NICE check
 * that blocks the direct sched_setattr(SCHED_FLAG_UTIL_CLAMP_MIN)
 * approach used by sched-android.c.
 *
 * Mechanism: APerformanceHint_createSession registers a set of TIDs
 * with system_server, which then performs the kernel-side uclamp
 * escalation on the app's behalf. Reporting actual work duration each
 * frame lets the system converge on an appropriate boost level.
 *
 * Requires Android 13+ (API level 33) for the base hint API. Falls
 * back to a no-op on older devices via dlsym detection.
 *
 * Tunables (read at init from $X1BOX_ADPF_* env or the same
 * <ext>/x1box/sched_config.txt file used by sched-android):
 *
 *   X1BOX_ADPF_ENABLED=0|1            (0)  master switch (independent of SCHED_PROFILE)
 *   X1BOX_ADPF_TARGET_NS=<int64>      (33333333 = 30 FPS target)
 *   X1BOX_ADPF_REPORT_PERIOD_MS=<int> (16)  reporter-thread sleep
 *   X1BOX_ADPF_THREAD_FILTER=<str>    ("hot,hotwarm,warm")  which sched classes to enroll
 *   X1BOX_ADPF_DEBUG=0|1              (0)
 */

#ifndef QEMU_UTIL_ADPF_ANDROID_H
#define QEMU_UTIL_ADPF_ANDROID_H

#include <stdbool.h>
#include <stdint.h>

/* Read env/config, dlsym the ADPF API symbols, enumerate our perf
 * threads, create a HintSession, and start the reporter thread.
 *
 * Must be called AFTER xemu's hot threads have been spawned (TCG,
 * pfifo, voice workers, apu, render) so the enumeration finds them.
 * pgraph_vk_init is a natural call site. Idempotent — second calls
 * are no-ops. */
void adpf_android_init(void);

/* Update the session's target work duration. Useful when the host
 * wants to ask for higher boost (e.g. user toggled "performance mode"
 * in xemu's settings). Caller passes nanoseconds, e.g. 16666666 for
 * 60 FPS target. Safe to call from any thread. */
void adpf_android_set_target(int64_t target_ns);

/* Per-frame report. Most callers will use the auto-reporter thread
 * spawned by init; this is exposed for the future case where xemu's
 * render loop wants to call it directly from SDL_GL_SwapWindow or the
 * NV097_FLIP_STALL handler for tighter convergence. */
void adpf_android_report_frame(int64_t actual_ns);

/* Split-duration report (API 34+ / Android 14+). Lets the system see
 * separately how much of the frame budget went to CPU vs GPU work, so
 * it can target DVFS boost at the binding side rather than running
 * both at max. Pass `gpu_ns = 0` on a CPU-bound workload; the system
 * interprets that as "no GPU work pending, don't boost GPU clocks."
 *
 * `start_ns` is the CLOCK_MONOTONIC nanosecond timestamp at which the
 * frame's work began (== previous FLIP_STALL).  Pass 0 to let the
 * implementation use its own clock.
 *
 * Falls back to the single-value path (cpu_ns + gpu_ns) on older
 * Android. No-op when ADPF is disabled. */
void adpf_android_report_frame_split(int64_t cpu_ns, int64_t gpu_ns,
                                     int64_t start_ns);

/* Extend the active session with an additional TID. Used by the
 * sched-android late-scanner to enroll threads spawned outside
 * qemu_thread_create (e.g. the Rust cranelift-tcg JIT worker, which
 * doesn't go through our per-creation hook and so wasn't in the TID
 * list when the session was originally built at pgraph_vk_init time).
 * Requires APerformanceHint_setThreads (API 34+ / Android 14); no-op
 * on older devices. Idempotent — duplicates are filtered. */
void adpf_android_add_thread(int tid);

/* Shutdown the reporter thread and close the session. Currently
 * unused — the session lives for the process lifetime. */
void adpf_android_shutdown(void);

/* Diagnostics — multi-line malloc'd string the caller must free. */
char *adpf_android_describe(void);

#endif /* QEMU_UTIL_ADPF_ANDROID_H */
