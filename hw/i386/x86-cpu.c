/*
 * Copyright (c) 2003-2004 Fabrice Bellard
 * Copyright (c) 2019, 2024 Red Hat, Inc.
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
#include "system/whpx.h"
#include "system/cpu-timers.h"
#include "trace.h"

#include "hw/i386/x86.h"
#include "target/i386/cpu.h"
#include "hw/intc/i8259.h"
#include "hw/irq.h"
#include "system/kvm.h"

/* TSC handling */
uint64_t cpu_get_tsc(CPUX86State *env)
{
#ifdef XBOX
    /*
     * Xbox CPU runs at 733.333 MHz. Profiled Halo 2 first level:
     * vDSO __kernel_clock_gettime was 4.87% of total CPU on the vCPU
     * thread, and KeQueryPerformanceCounter HLE hits=0 — so Halo 2
     * polls RDTSC directly (compiled into game code, never going
     * through a kernel hook). Each RDTSC was costing us a
     * qemu_clock_get_ns → cpus_get_virtual_clock → clock_gettime
     * syscall path; that's ~50 cycles for a one-instruction RDTSC.
     *
     * On aarch64 hosts we sidestep the entire timer stack by reading
     * ARM's cntvct_el0 directly (1 mrs instruction, ~1 cycle) and
     * scaling to 733 MHz with a cached 96.32 fixed-point multiplier
     * computed once from cntfrq_el0. That's a single mrs + umulh per
     * RDTSC. Halo 2's game timing only cares about monotonic
     * elapsed-cycle counts, not host-vs-guest pause semantics.
     *
     * Non-aarch64 builds (host testing) keep the slow path.
     */
#if defined(__aarch64__)
    {
        static uint64_t s_mul;
        static bool s_init;
        if (__builtin_expect(!s_init, 0)) {
            uint64_t cntfrq;
            __asm__("mrs %0, cntfrq_el0" : "=r"(cntfrq));
            if (cntfrq == 0) {
                cntfrq = 19200000ULL;
            }
            /* mul = (733333333 << 32) / cntfrq. Fits in 64 bits:
             * 733333333 * 2^32 ≈ 3.15e18 < 2^64. */
            s_mul = (733333333ULL << 32) / cntfrq;
            s_init = true;
        }
        uint64_t v;
        __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
        return (uint64_t)(((unsigned __int128)v * s_mul) >> 32);
    }
#else
    return muldiv64(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL), 733333333,
                    NANOSECONDS_PER_SECOND);
#endif
#else
    return cpus_get_elapsed_ticks();
#endif
}

/* IRQ handling */
static void pic_irq_request(void *opaque, int irq, int level)
{
    CPUState *cs = first_cpu;
    X86CPU *cpu = X86_CPU(cs);

    trace_x86_pic_interrupt(irq, level);
    if (cpu_is_apic_enabled(cpu->apic_state) && !kvm_irqchip_in_kernel() &&
        !whpx_apic_in_platform()) {
        CPU_FOREACH(cs) {
            cpu = X86_CPU(cs);
            if (apic_accept_pic_intr(cpu->apic_state)) {
                apic_deliver_pic_intr(cpu->apic_state, level);
            }
        }
    } else {
        if (level) {
            cpu_interrupt(cs, CPU_INTERRUPT_HARD);
        } else {
            cpu_reset_interrupt(cs, CPU_INTERRUPT_HARD);
        }
    }
}

qemu_irq x86_allocate_cpu_irq(void)
{
    return qemu_allocate_irq(pic_irq_request, NULL, 0);
}

int cpu_get_pic_interrupt(CPUX86State *env)
{
    X86CPU *cpu = env_archcpu(env);
    int intno;

    if (!kvm_irqchip_in_kernel() && !whpx_apic_in_platform()) {
        intno = apic_get_interrupt(cpu->apic_state);
        if (intno >= 0) {
            return intno;
        }
        /* read the irq from the PIC */
        if (!apic_accept_pic_intr(cpu->apic_state)) {
            return -1;
        }
    }

    intno = pic_read_irq(isa_pic);
    return intno;
}

APICCommonState *cpu_get_current_apic(void)
{
    if (current_cpu) {
        X86CPU *cpu = X86_CPU(current_cpu);
        return cpu->apic_state;
    } else {
        return NULL;
    }
}
