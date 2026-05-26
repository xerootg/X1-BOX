/*
 * QEMU TCG Multi Threaded vCPUs implementation
 *
 * Copyright (c) 2003-2008 Fabrice Bellard
 * Copyright (c) 2014 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "system/tcg.h"
#include "system/replay.h"
#include "exec/icount.h"
#include "qemu/main-loop.h"
#include "qemu/notify.h"
#include "qemu/guest-random.h"
#include "hw/boards.h"
#include "tcg/startup.h"
#include "tcg-accel-ops.h"
#include "tcg-accel-ops-mttcg.h"

typedef struct MttcgForceRcuNotifier {
    Notifier notifier;
    CPUState *cpu;
} MttcgForceRcuNotifier;

static void do_nothing(CPUState *cpu, run_on_cpu_data d)
{
}

static void mttcg_force_rcu(Notifier *notify, void *data)
{
    CPUState *cpu = container_of(notify, MttcgForceRcuNotifier, notifier)->cpu;

    /*
     * Called with rcu_registry_lock held, using async_run_on_cpu() ensures
     * that there are no deadlocks.
     */
    async_run_on_cpu(cpu, do_nothing, RUN_ON_CPU_NULL);
}

/*
 * In the multi-threaded case each vCPU has its own thread. The TLS
 * variable current_cpu can be used deep in the code to find the
 * current CPUState for a given thread.
 */

#if defined(XBOX) && defined(__aarch64__)
#include <android/log.h>
/*
 * Pre-set host FPCR to SSE-compatible defaults on vCPU thread entry.
 *
 * Default Android aarch64 FPCR has FZ=1 (flush denormals to zero) which
 * diverges from SSE MXCSR default (FZ=0, preserves denormals). We
 * clear FZ / FZ16 / DN here so that if anything tier-2 or per-op
 * X1BOX_SSE_INLINE_* opts back IN to NEON-lowered scalar SSE, those
 * ops at least match SSE on denormals and NaN propagation.
 *
 * This is NOT sufficient on its own to make NEON FMUL.S bit-identical
 * to x86 MULSS — proven by the 2026-05-25 attempt to re-enable
 * X1BOX_SSE_INLINE=1 with both (a) this startup FPCR write AND
 * (b) full MXCSR -> FPCR routing on LDMXCSR. Halo 2 NPCs still slid.
 * The residual drift is architectural, not configurable via FPCR.
 *
 * The startup write is still kept because:
 *   1. It's correct (matches SSE intent) and free if SSE_INLINE=0.
 *   2. It's load-bearing if a per-op gate opts in for bisection.
 *   3. Without it, NEON FZ=1 would flush denormals coming OUT of
 *      math libraries used by helpers, etc.
 *
 * FPCR bits (ARMv8 D17.4):
 *   bit 24: FZ        (flush-to-zero, non-FP16)
 *   bit 19: FZ16      (flush-to-zero, FP16)
 *   bit 25: DN        (default NaN — propagating NaN payload is the
 *                      SSE-matching choice, so DN=0)
 *   bits 22-23: RM    (rounding mode — leave at default round-nearest)
 *
 * See project_sse_neon_physics_drift.md for the bisection record.
 */
static inline void xemu_vcpu_match_sse_fpcr(void)
{
    uint64_t fpcr_before, fpcr_after;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr_before));
    uint64_t fpcr = fpcr_before & ~((1ULL << 24) | (1ULL << 19) | (1ULL << 25));
    asm volatile("msr fpcr, %0" : : "r"(fpcr));
    asm volatile("mrs %0, fpcr" : "=r"(fpcr_after));
    /*
     * Log so we can see WHETHER the FPCR write actually landed —
     * historical debugging: physics drift persisted on 2026-05-25
     * after this change, so we need to confirm whether FPCR really
     * sticks at FZ=0 in the running vCPU thread, or whether something
     * downstream (libc, helper, JITted op-fence) restores the bit.
     */
    __android_log_print(ANDROID_LOG_INFO, "x1-fpcr",
        "vcpu_fpcr: before=0x%llx after=0x%llx FZ=%d FZ16=%d DN=%d",
        (unsigned long long)fpcr_before,
        (unsigned long long)fpcr_after,
        (int)((fpcr_after >> 24) & 1),
        (int)((fpcr_after >> 19) & 1),
        (int)((fpcr_after >> 25) & 1));
}

/*
 * Read FPCR from the current thread. Used by the periodic emitter in
 * profile.c to confirm the FZ=0 setting from startup is still active
 * after the JIT, helpers, and audio thread have run for a while.
 */
uint64_t xemu_vcpu_read_fpcr(void)
{
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
}
#else
uint64_t xemu_vcpu_read_fpcr(void) { return 0; }
#endif

static void *mttcg_cpu_thread_fn(void *arg)
{
    MttcgForceRcuNotifier force_rcu;
    CPUState *cpu = arg;

    assert(tcg_enabled());
    g_assert(!icount_enabled());

#if defined(XBOX) && defined(__aarch64__)
    xemu_vcpu_match_sse_fpcr();
#endif

    rcu_register_thread();
    force_rcu.notifier.notify = mttcg_force_rcu;
    force_rcu.cpu = cpu;
    rcu_add_force_rcu_notifier(&force_rcu.notifier);
    tcg_register_thread();

    bql_lock();
    qemu_thread_get_self(cpu->thread);

    cpu->thread_id = qemu_get_thread_id();
    cpu->neg.can_do_io = true;
    current_cpu = cpu;
    cpu_thread_signal_created(cpu);
    qemu_guest_random_seed_thread_part2(cpu->random_seed);

    do {
        qemu_process_cpu_events(cpu);

        if (cpu_can_run(cpu)) {
            int r;
            bql_unlock();
            r = tcg_cpu_exec(cpu);
            bql_lock();
#ifdef XBOX
            {
                static int dbg_mttcg = 0;
                if (dbg_mttcg < 30) {
                    error_report("[MTTCG] cpu_exec returned r=%d "
                                 "halted=%d stop=%d exit_req=%d",
                                 r, cpu->halted, cpu->stop,
                                 qatomic_read(&cpu->exit_request));
                    dbg_mttcg++;
                }
            }
#endif
            switch (r) {
            case EXCP_DEBUG:
                cpu_handle_guest_debug(cpu);
                break;
            case EXCP_HALTED:
                /*
                 * Usually cpu->halted is set, but may have already been
                 * reset by another thread by the time we arrive here.
                 */
                break;
            case EXCP_ATOMIC:
                bql_unlock();
                cpu_exec_step_atomic(cpu);
                bql_lock();
            default:
                /* Ignore everything else? */
                break;
            }
        }
    } while (!cpu->unplug || cpu_can_run(cpu));

    tcg_cpu_destroy(cpu);
    bql_unlock();
    rcu_remove_force_rcu_notifier(&force_rcu.notifier);
    rcu_unregister_thread();
    return NULL;
}

void mttcg_start_vcpu_thread(CPUState *cpu)
{
    char thread_name[VCPU_THREAD_NAME_SIZE];

    g_assert(tcg_enabled());
    tcg_cpu_init_cflags(cpu, current_machine->smp.max_cpus > 1);

    /* create a thread per vCPU with TCG (MTTCG) */
    snprintf(thread_name, VCPU_THREAD_NAME_SIZE, "CPU %d/TCG",
             cpu->cpu_index);

    qemu_thread_create(cpu->thread, thread_name, mttcg_cpu_thread_fn,
                       cpu, QEMU_THREAD_JOINABLE);
}
