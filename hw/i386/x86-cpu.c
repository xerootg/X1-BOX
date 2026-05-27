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
     * Xbox CPU runs at 733.333 MHz. RDTSC must tick from the SAME
     * clock source as the PIT IRQ that drives KeTickCount (the kernel
     * global behind XAPILIB::GetTickCount): both come from
     * QEMU_CLOCK_VIRTUAL. Halo 2's Bink decoder (FUN_003e4560) reads
     * RDTSC for elapsed ms AND GetTickCount as a sanity check; if the
     * two clocks drift apart, the bink's drift accumulator at
     * DAT_00570a14 grows unbounded until iVar2+drift wraps so far
     * negative that the (iVar2 - DAT_00484ad8) > 0xc0000000 backward-
     * jump guard fires, FUN_003e4560 returns its cached ms forever,
     * BinkWait's timer condition never resolves, and halo_bink_pump_
     * tick @ 0x00155fc8 spins for 60+ seconds on every Bink frame.
     * Sampled wedge state: drift=-1.85M ms, host advanced 10s with
     * the cached ms field literally not moving.
     *
     * Prior aarch64 fast path (commit before 2026-05-26) read
     * cntvct_el0 directly — host monotonic, doesn't pause when the
     * VM does. That made RDTSC drift forward of KeTickCount on every
     * vm_stop/vm_start (gdb break, audio thread serialization, frame
     * throttle, anything that briefly disables ticks via
     * cpu_disable_ticks). The drift assumption "virtual and host
     * clocks track within ns" doesn't hold over a 20-minute attract-
     * loop run. Going back through qemu_clock_get_ns costs ~5% vCPU
     * on the RDTSC hotspot, but the wedge is unrecoverable until the
     * bink itself ages out a movie. Correctness over throughput.
     */
    return muldiv64(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL), 733333333,
                    NANOSECONDS_PER_SECOND);
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
