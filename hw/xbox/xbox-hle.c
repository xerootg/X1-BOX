/*
 * Xbox kernel HLE — implementation.
 *
 * Pipeline:
 *   1. xbox_hle_init reads env vars; if HLE is off we bail with no
 *      runtime cost beyond a single atomic read on the fast path.
 *   2. On each cpu_exec_loop entry, the dispatcher nudges
 *      xbox_hle_resolve_kernel() until it walks the PE export table at
 *      0x80010000 and installs hooks for known ordinals.
 *   3. Per-TB, the dispatcher calls xbox_hle_check(cpu, tb->pc). On a
 *      hit, the handler runs in host C and sets EIP to the caller's
 *      return address; the TB itself is skipped.
 *
 * Hooks are PC-keyed via a 64-slot open-addressing hash. The kernel
 * itself only exposes ~6 hooks today so collisions are rare; the table
 * has headroom for future Tier 2/3 entries (D3D8 fast paths, math libs).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/atomic.h"
#include "qemu/timer.h"  /* qemu_clock_get_ns for KeQueryPerformanceCounter */
#include "hw/core/cpu.h"
#include "exec/cpu-common.h"
#include "exec/target_page.h"
#include "system/dma.h"
#include "system/hw_accel.h"
#include "system/memory.h"
#include "cpu.h"  /* target/i386 -- X86CPU, CPUX86State */
#include "xbox-hle.h"
#include <unistd.h>
#include <string.h>

/* ARM SHA-1 crypto extensions for XcShaTransform HLE. */
#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_SHA1
#define HWCAP_SHA1 (1 << 5)
#endif
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define HLE_LOG(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, "x1-hle", fmt, ##__VA_ARGS__)
#else
#define HLE_LOG(fmt, ...) qemu_log("x1-hle: " fmt "\n", ##__VA_ARGS__)
#endif

/* ------------------------------------------------------------------ */
/*  Guest memory helpers (mirrors xpacks.c — virt→phys + cross-page)   */
/* ------------------------------------------------------------------ */

static int virt_to_phys_guest(uint32_t vaddr, hwaddr *out_phys)
{
    CPUState *cs = first_cpu;
    if (!cs) return -1;
    cpu_synchronize_state(cs);
    MemTxAttrs attrs;
    hwaddr gpa = cpu_get_phys_page_attrs_debug(cs,
                    vaddr & TARGET_PAGE_MASK, &attrs);
    if (gpa == (hwaddr)-1) return -1;
    *out_phys = gpa + (vaddr & ~TARGET_PAGE_MASK);
    return 0;
}

static bool g_read(uint32_t vaddr, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        hwaddr phys;
        if (virt_to_phys_guest(vaddr + done, &phys) != 0) return false;
        size_t in_page = TARGET_PAGE_SIZE - (phys & ~TARGET_PAGE_MASK);
        size_t n = MIN(len - done, in_page);
        if (dma_memory_read(&address_space_memory, phys,
                            (uint8_t *)buf + done, n,
                            MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            return false;
        }
        done += n;
    }
    return true;
}

static bool g_write(uint32_t vaddr, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        hwaddr phys;
        if (virt_to_phys_guest(vaddr + done, &phys) != 0) return false;
        size_t in_page = TARGET_PAGE_SIZE - (phys & ~TARGET_PAGE_MASK);
        size_t n = MIN(len - done, in_page);
        if (dma_memory_write(&address_space_memory, phys,
                             (const uint8_t *)buf + done, n,
                             MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            return false;
        }
        done += n;
    }
    return true;
}

static inline bool g_read32(uint32_t vaddr, uint32_t *out)
{
    return g_read(vaddr, out, 4);
}

static inline bool g_write32(uint32_t vaddr, uint32_t val)
{
    return g_write(vaddr, &val, 4);
}

/* ------------------------------------------------------------------ */
/*  Handler signature                                                  */
/* ------------------------------------------------------------------ */

/*
 * A handler mutates env->eip + env->regs to simulate a return, and
 * does the actual work in host code. It does NOT pop the call stack
 * itself for fastcall functions — those have no stack args. For
 * stdcall, the handler must pop ret + N*4.
 *
 * Return:
 *   true  — handler took ownership of the call. Dispatcher skips the TB.
 *   false — handler declined (e.g., contended-lock path that needs the
 *           real kernel's KeWait). Dispatcher executes the TB normally,
 *           env state MUST be untouched.
 */
typedef bool (*XboxHleHandler)(X86CPU *cpu);

struct hle_entry {
    uint32_t pc;            /* guest virtual addr of function entry; 0 = empty */
    XboxHleHandler handler;
    const char *name;
    uint64_t hits;          /* telemetry — handler returned true */
    uint64_t declines;      /* telemetry — handler returned false (still entered) */
};

#define HLE_TABLE_SIZE 64u
static struct hle_entry g_hle_table[HLE_TABLE_SIZE];

/* ------------------------------------------------------------------ */
/*  Per-handler gates + master enable                                  */
/* ------------------------------------------------------------------ */

static bool g_hle_enabled;          /* master, set by xbox_hle_init */
static bool g_hle_resolved;         /* PE walk succeeded once */
static bool g_gate_rtl = true;      /* RtlMoveMemory + RtlFillMemory */
static bool g_gate_kf  = true;      /* KfAcquireSpinLock + Release */
static bool g_gate_yield = true;    /* KeDelay... + NtYieldExecution */

static uint64_t g_resolve_attempts;
static uint64_t g_resolve_failures;

/*
 * Unhooked-PC profiler. Counts every kernel-range PC that hits the
 * HLE check WITHOUT matching a hook. Periodically dumped by
 * xbox_hle_log_stats() — the top entries are our next HLE candidates.
 *
 * 256-slot Fibonacci-hashed direct-mapped table. Hash collisions cause
 * lossy aggregation but the heavy hitters stay visible. Don't enable
 * tracking on every TB (too expensive) — only fires when we already
 * know we're in the kernel image (post the PC-range fast-path).
 */
#define HLE_PROFILE_SLOTS 256u
static struct {
    uint32_t pc;
    uint64_t count;
} g_unhooked_pcs[HLE_PROFILE_SLOTS];

static inline void unhooked_pc_count(uint32_t pc)
{
    unsigned slot = (unsigned)((pc * 2654435761u) >> (32 - 8));
    if (g_unhooked_pcs[slot].pc == pc) {
        g_unhooked_pcs[slot].count++;
    } else if (g_unhooked_pcs[slot].count < 8) {
        /* Replace cold entries (count < 8) — gives heavy hitters
         * stability while letting new entries register. */
        g_unhooked_pcs[slot].pc = pc;
        g_unhooked_pcs[slot].count = 1;
    }
}

/* ------------------------------------------------------------------ */
/*  Hook table accessors                                               */
/* ------------------------------------------------------------------ */

static inline unsigned hle_slot(uint32_t pc)
{
    /* Fibonacci hash → 6 bits. */
    return (unsigned)((pc * 2654435761u) >> (32 - 6));
}

static void hle_install(const char *name, uint32_t pc,
                        XboxHleHandler handler)
{
    if (!pc) return;
    unsigned slot = hle_slot(pc);
    for (unsigned probe = 0; probe < HLE_TABLE_SIZE; probe++) {
        unsigned i = (slot + probe) & (HLE_TABLE_SIZE - 1);
        if (g_hle_table[i].pc == 0 || g_hle_table[i].pc == pc) {
            g_hle_table[i].pc = pc;
            g_hle_table[i].handler = handler;
            g_hle_table[i].name = name;
            HLE_LOG("registered %s @ 0x%08x (slot %u)", name, pc, i);
            return;
        }
    }
    HLE_LOG("hook table full installing %s @ 0x%08x", name, pc);
}

static inline struct hle_entry *hle_lookup(uint32_t pc)
{
    unsigned slot = hle_slot(pc);
    for (unsigned probe = 0; probe < HLE_TABLE_SIZE; probe++) {
        unsigned i = (slot + probe) & (HLE_TABLE_SIZE - 1);
        if (g_hle_table[i].pc == 0) return NULL;
        if (g_hle_table[i].pc == pc) return &g_hle_table[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Calling-convention helpers                                         */
/* ------------------------------------------------------------------ */

/*
 * Stdcall return: callee pops ret+N*4. Used by Rtl*, Ke*, Nt*.
 *   ret addr at [esp+0], args at [esp+4]+.
 */
static bool hle_return_stdcall(CPUX86State *env, unsigned nargs)
{
    uint32_t esp = env->regs[R_ESP];
    uint32_t ret_addr;
    if (!g_read32(esp, &ret_addr)) return false;
    env->regs[R_ESP] = esp + 4 + 4 * nargs;
    env->eip = ret_addr;
    return true;
}

/*
 * Fastcall return (Kf*): args in ECX/EDX, no stack args. Just pop the
 * return address.
 */
static bool hle_return_fastcall(CPUX86State *env)
{
    uint32_t esp = env->regs[R_ESP];
    uint32_t ret_addr;
    if (!g_read32(esp, &ret_addr)) return false;
    env->regs[R_ESP] = esp + 4;
    env->eip = ret_addr;
    return true;
}

static inline uint32_t hle_arg32(CPUX86State *env, unsigned idx)
{
    uint32_t v = 0;
    g_read32(env->regs[R_ESP] + 4 + 4 * idx, &v);
    return v;
}

/* ------------------------------------------------------------------ */
/*  Handlers                                                           */
/* ------------------------------------------------------------------ */

/* ----------------- Kernel IRQL global ----------------- */

/*
 * The kernel stores PrcbData.CurrentIrql at a fixed absolute address —
 * confirmed via Ghidra decompile of KfRaiseIrql/KfLowerIrql at
 * 0x80014338/0x80014368 (both use `mov [0x80035c00], cl` etc., not the
 * fs:[KPCR_IRQL_OFFSET] redirection). Using the absolute address keeps
 * us in sync with the kernel regardless of FS segment setup.
 */
#define XKRNL_CURRENT_IRQL_VA 0x80035c00u

static uint8_t kirql_get(void)
{
    /* Always use slow path g_read for now — the va_cache is suspected
     * of returning stale phys at some boot phase. KfRaiseIrql is only
     * 1-byte read+write so the overhead is acceptable. */
    uint8_t v = 0;
    g_read(XKRNL_CURRENT_IRQL_VA, &v, 1);
    return v;
}

static void kirql_set(uint8_t v)
{
    g_write(XKRNL_CURRENT_IRQL_VA, &v, 1);
}

/*
 * KIRQL FASTCALL KfRaiseIrql(KIRQL NewIrql)
 *
 * ECX = new IRQL (low byte). Returns old IRQL in AL.
 * Reference disasm from Ghidra (xboxkrnl.exe @ 0x80014338):
 *   xor   eax, eax
 *   mov   al, [0x80035c00]      ; load old IRQL
 *   mov   [0x80035c00], cl      ; store new IRQL
 *   ret
 *
 * We don't model the IRQL-raise side-effect (deferring the pending-IRQ
 * dispatch) because raising IRQL by definition can't fire any new IRQ;
 * the real function does no PIC manipulation here either. So this HLE
 * is byte-for-byte semantically equivalent to the kernel.
 */
static bool hle_kf_raise_irql(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint8_t new_irql = (uint8_t)(env->regs[R_ECX] & 0xff);
    uint8_t old_irql = kirql_get();
    kirql_set(new_irql);
    env->regs[R_EAX] = old_irql; /* return value in AL; high bits zeroed
                                  * because the kernel's xor eax,eax
                                  * comes first */
    hle_return_fastcall(env);
    return true;
}

/*
 * KIRQL FASTCALL KfLowerIrql(KIRQL NewIrql)
 *
 * Kernel disasm (0x80014368): set IRQL byte, then check the pending-
 * IRQ mask AND the level-mask at [0x800142a4 + new_irql*4] — if any
 * IRQs are pending at or below the new level, the slow path bit-scans
 * the mask and dispatches handlers (a guest function table at
 * PTR_caseD_0_80035938).
 *
 * 2026-05-22 profile showed PC 0x80014386 (inside the slow-path
 * bit-scan loop) at 122K hits/sec on Halo 2 title screen — by far the
 * dominant unhooked kernel hot-spot.
 *
 * HLE strategy:
 *   1. Read pending mask + level mask BEFORE mutating state.
 *   2. If (pending & level) == 0: fast path. Write the new IRQL byte
 *      and return. Skips the kernel's bit-scan + I/O ports + guest
 *      handler dispatch.
 *   3. Else: decline. Kernel runs the real slow path so the pending
 *      handler actually fires. We don't touch env so the function
 *      re-runs cleanly from its entry.
 *
 * Cost: 2 g_read32 calls before commit. Saves ~30 kernel instructions
 * + I/O port writes on the common case where no IRQ is eligible.
 */
#define XKRNL_PENDING_IRQ_MASK_VA  0x80035a1cu
#define XKRNL_LEVEL_MASK_TABLE_VA  0x800142a4u

static bool hle_kf_lower_irql(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint8_t new_irql = (uint8_t)(env->regs[R_ECX] & 0xff);

    /*
     * Strategy (matches Cxbx-R's EmuKrnl.cpp:423-467 implementation):
     *   1. Read HalInterruptRequestRegister (0x80035a1c) and the
     *      IrqlMasks[NewIrql] entry (0x800142a4 + new_irql*4).
     *   2. If (pending & level_mask) == 0 → fast path: no pending soft
     *      interrupts at or below the new level, so just write IRQL and
     *      return.
     *   3. Else: decline. Real kernel BSR-picks the highest pending
     *      IRQL and calls the soft-interrupt handler. We can't
     *      replicate that without owning the DPC-drain plumbing.
     *
     * Earlier attempts used a phys-VA cache; that broke boot because the
     * cache returned stale phys after kernel paging stabilized. The
     * direct g_read32/g_write path is ~150ns per call vs ~30ns cached,
     * but the kernel's slow path is several hundred ns + I/O port
     * writes, so the fast path is still a big win.
     */
    uint32_t pending_mask = 0, level_mask = 0;
    if (!g_read32(XKRNL_PENDING_IRQ_MASK_VA, &pending_mask)) return false;
    if (!g_read32(XKRNL_LEVEL_MASK_TABLE_VA + (uint32_t)new_irql * 4,
                  &level_mask)) {
        return false;
    }
    if (pending_mask & level_mask) {
        return false;  /* slow path → real kernel dispatches the soft IRQ */
    }

    /*
     * Critical: the kernel's KfLowerIrql does `mov [0x80035c00], ecx` —
     * a 4-byte write — so it zeros bytes 1-3 (APC/DPC soft-interrupt
     * indicator bytes) at the same time as setting CurrentIrql.
     * Writing only 1 byte leaves those indicators set; the dispatcher
     * thinks APC/DPC need delivery but the IRQL says PASSIVE → state-
     * machine deadlock. Black-screen-at-boot reproducer 2026-05-22.
     */
    uint32_t new_irql_dword = (uint32_t)new_irql;  /* zero-extends bytes 1-3 */
    if (!g_write(XKRNL_CURRENT_IRQL_VA, &new_irql_dword, 4)) return false;
    hle_return_fastcall(env);
    return true;
}

/*
 * KIRQL __stdcall KeRaiseIrqlToDpcLevel(VOID)
 *
 * Kernel disasm (0x80014348):
 *   xor   eax, eax
 *   mov   al, [0x80035c00]      ; load old IRQL
 *   mov   byte [0x80035c00], 2  ; write DISPATCH_LEVEL
 *   ret
 *
 * Returns previous IRQL byte in AL. Convenience wrapper around
 * KfRaiseIrql(DISPATCH_LEVEL); used heavily by every kernel function
 * that needs to lock the dispatcher database (which on uniprocessor
 * Xbox is the same as raising to DISPATCH).
 *
 * Same safety argument as KfRaiseIrql: raising IRQL by definition
 * cannot fire any new soft interrupts, so no pending-IRQ check needed.
 */
static bool hle_ke_raise_irql_to_dpc_level(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint8_t old_irql = kirql_get();
    /* 4-byte write (matches kernel's `mov byte [],2` followed by the
     * zeroing of soft-interrupt bytes in KfLowerIrql; the raise side
     * uses `mov byte` here but writing 4 bytes is also correct since
     * the high bytes get zeroed). */
    uint32_t two = 2;
    if (!g_write(XKRNL_CURRENT_IRQL_VA, &two, 1)) return false;
    env->regs[R_EAX] = old_irql;  /* return value in AL, high bits zeroed */
    hle_return_stdcall(env, 0);
    return true;
}

/*
 * VOID __fastcall ObfDereferenceObject(PVOID Object)
 *
 * Kernel disasm (0x80020491) — fastcall, ECX = Object. The object
 * header is at Object-0x10:
 *   header[+0x00] = PointerCount (LONG)
 *   header[+0x08] = Type (POBJECT_TYPE)
 *   Type[+0x04]   = DeleteProcedure
 *   Type[+0x0c]   = CloseProcedure
 *
 * Decompile shows: read PointerCount, decrement, store. If the OLD
 * value was 1 (i.e. now 0), call the Type's CloseProcedure (if non-NULL)
 * then DeleteProcedure.
 *
 * HLE fast path: if PointerCount > 1, just decrement and return.
 * If PointerCount == 1, decline (let kernel run the destructor chain).
 *
 * Called constantly: every Ob{ReferenceObjectByHandle,ByName,…} pairs
 * with a deref. Halo 2 file/event/thread handle usage hits this in the
 * hundreds-per-second range during normal play.
 */
static bool hle_obf_dereference_object(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t obj_ptr = env->regs[R_ECX];
    uint32_t header_ptr = obj_ptr - 0x10;

    int32_t pointer_count = 0;
    if (!g_read(header_ptr, &pointer_count, 4)) return false;
    if (pointer_count <= 1) {
        /* Decline: kernel needs to run CloseProcedure + DeleteProcedure. */
        return false;
    }
    pointer_count--;
    if (!g_write(header_ptr, &pointer_count, 4)) return false;
    hle_return_fastcall(env);
    return true;
}

/*
 * BOOLEAN __stdcall KeInsertQueueDpc(KDPC *Dpc, PVOID SystemArg1, PVOID SystemArg2)
 *
 * Kernel disasm (0x8001a3c3): KfRaiseIrql to 0x1f (HIGH_LEVEL), test
 * Dpc->Inserted, if zero link into the per-CPU DPC tail and conditionally
 * fire HalRequestSoftwareInterrupt(DISPATCH_LEVEL). Returns whether the
 * insert happened.
 *
 * KDPC layout:
 *   +0x00 Type      (UCHAR, 0x13 for DPC)
 *   +0x01 Number    (UCHAR)
 *   +0x02 Inserted  (UCHAR — set/cleared by queue ops)
 *   +0x04 Flink     (used as LIST_ENTRY when queued)
 *   +0x08 Blink
 *   +0x0C DeferredRoutine
 *   +0x10 DeferredContext
 *   +0x14 SystemArgument1
 *   +0x18 SystemArgument2
 *
 * KPCR DPC list head:
 *   0x80035c2c — head.Flink
 *   0x80035c30 — head.Blink
 *   0x80035c34 — DpcRoutineActive (currently draining)
 *   0x80035c28 — DpcInterruptRequested (dispatch is queued)
 *
 * HLE strategy:
 *   1. Already inserted (Inserted != 0): return FALSE immediately.
 *   2. Not inserted AND (active=0 AND requested=0): would need to call
 *      HalRequestSoftwareInterrupt → decline; kernel runs full path
 *      including the PIC manipulation.
 *   3. Not inserted AND (active!=0 OR requested!=0): linkage only — do
 *      the writes ourselves and return TRUE. No interrupt trigger
 *      needed; the active drain or pending interrupt will pick it up.
 *
 * The "linkage only" case is the hot path during a burst of DPC
 * scheduling (interrupt handler queuing several DPCs in a row, or
 * gameplay audio mixer / render thread firing multiple DPCs per frame).
 */
#define KDPC_INSERTED_OFFSET    0x02
#define KDPC_FLINK_OFFSET       0x04
#define KDPC_BLINK_OFFSET       0x08
#define KDPC_SYSARG1_OFFSET     0x14
#define KDPC_SYSARG2_OFFSET     0x18
#define XKRNL_DPC_HEAD_FLINK_VA 0x80035c2cu
#define XKRNL_DPC_HEAD_BLINK_VA 0x80035c30u
#define XKRNL_DPC_REQUESTED_VA  0x80035c28u
#define XKRNL_DPC_ACTIVE_VA     0x80035c34u

static bool hle_ke_insert_queue_dpc(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t dpc_ptr   = hle_arg32(env, 0);
    uint32_t sys_arg1  = hle_arg32(env, 1);
    uint32_t sys_arg2  = hle_arg32(env, 2);

    uint8_t inserted = 0;
    if (!g_read(dpc_ptr + KDPC_INSERTED_OFFSET, &inserted, 1)) return false;
    if (inserted) {
        /* Already in queue — return FALSE. */
        env->regs[R_EAX] = 0;
        hle_return_stdcall(env, 3);
        return true;
    }

    /* Need to insert. Check whether HalRequestSoftwareInterrupt would
     * also fire — if so, decline. */
    uint8_t active = 0, requested = 0;
    if (!g_read(XKRNL_DPC_ACTIVE_VA,    &active,    1)) return false;
    if (!g_read(XKRNL_DPC_REQUESTED_VA, &requested, 1)) return false;
    if (active == 0 && requested == 0) {
        /* First DPC of the moment — kernel must call
         * HalRequestSoftwareInterrupt (PIC + IMR manipulation). Decline. */
        return false;
    }

    /* Safe path: link into tail of DPC list, no interrupt trigger needed. */
    uint32_t prev_tail = 0;
    if (!g_read32(XKRNL_DPC_HEAD_BLINK_VA, &prev_tail)) return false;

    /* DPC.SystemArgument1/2 */
    if (!g_write(dpc_ptr + KDPC_SYSARG1_OFFSET, &sys_arg1, 4)) return false;
    if (!g_write(dpc_ptr + KDPC_SYSARG2_OFFSET, &sys_arg2, 4)) return false;

    /* DPC.Flink = &head (loops back via head.Flink VA). */
    uint32_t head_flink_va = XKRNL_DPC_HEAD_FLINK_VA;
    if (!g_write(dpc_ptr + KDPC_FLINK_OFFSET, &head_flink_va, 4)) return false;

    /* DPC.Blink = previous tail. */
    if (!g_write(dpc_ptr + KDPC_BLINK_OFFSET, &prev_tail, 4)) return false;

    /* previous_tail->Flink = &DPC.Flink_node (= dpc_ptr + 4). */
    uint32_t dpc_flink_node_va = dpc_ptr + KDPC_FLINK_OFFSET;
    if (!g_write(prev_tail, &dpc_flink_node_va, 4)) return false;

    /* head.Blink = &DPC.Flink_node. */
    if (!g_write(XKRNL_DPC_HEAD_BLINK_VA, &dpc_flink_node_va, 4)) return false;

    /* Mark inserted. */
    uint8_t one = 1;
    if (!g_write(dpc_ptr + KDPC_INSERTED_OFFSET, &one, 1)) return false;

    env->regs[R_EAX] = 1;  /* TRUE — we inserted */
    hle_return_stdcall(env, 3);
    return true;
}

/*
 * BOOLEAN __stdcall KeRemoveQueueDpc(KDPC *Dpc)
 *
 * Kernel disasm (0x8001a3a2): cli; test Dpc->Inserted; if non-zero do
 * doubly-linked-list unlink (prev.Flink = our flink; next.Blink = our
 * blink) and clear Inserted. No IRQL raise — caller's responsibility.
 * Returns BOOLEAN (was-queued).
 *
 * Standard circular doubly-linked list unlink: with Dpc->Flink pointing
 * to next entry's Flink_node (or head.Flink_addr if we're tail), and
 * Dpc->Blink pointing to previous entry's Flink_node (or head.Blink_addr
 * if we're head), the unlink is:
 *   *Blink = Flink         (prev.Flink updated to skip us)
 *   *(Flink+4) = Blink     (next.Blink updated to skip us)
 */
static bool hle_ke_remove_queue_dpc(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t dpc_ptr = hle_arg32(env, 0);

    uint8_t inserted = 0;
    if (!g_read(dpc_ptr + KDPC_INSERTED_OFFSET, &inserted, 1)) return false;

    if (inserted) {
        uint32_t flink = 0, blink = 0;
        if (!g_read32(dpc_ptr + KDPC_FLINK_OFFSET, &flink)) return false;
        if (!g_read32(dpc_ptr + KDPC_BLINK_OFFSET, &blink)) return false;
        /* prev.Flink = our Flink (forward link points around us) */
        if (!g_write(blink, &flink, 4)) return false;
        /* next.Blink = our Blink (backward link points around us).
         * Next entry's Blink is at +4 from its Flink_node, which is what
         * `flink` currently holds the address of. */
        if (!g_write(flink + 4, &blink, 4)) return false;
        /* Clear Inserted byte. */
        uint8_t zero = 0;
        if (!g_write(dpc_ptr + KDPC_INSERTED_OFFSET, &zero, 1)) return false;
    }

    env->regs[R_EAX] = inserted ? 1 : 0;
    hle_return_stdcall(env, 1);
    return true;
}

/*
 * BOOLEAN __stdcall RtlEqualString(
 *     PCANSI_STRING String1, PCANSI_STRING String2,
 *     BOOLEAN CaseInSensitive)
 *
 * ANSI_STRING layout: USHORT Length @ +0, USHORT MaxLength @ +2,
 *                     PSTR Buffer @ +4.
 *
 * Kernel disasm (0x80022678): compare lengths first (cheap), then byte
 * loop, optionally case-folded via the kernel's RtlUpperChar (which
 * handles ANSI code page specifics for high-bit chars).
 *
 * HLE strategy:
 *   - Length-mismatch fast path: return FALSE immediately.
 *   - Case-sensitive: byte-equality, full HLE.
 *   - Case-insensitive: only HLE for pure ASCII A-Z/a-z folding; decline
 *     when any byte is >=0x80 (kernel does ANSI code page mapping).
 *   - Cap string length at 0x800 — anything longer than that is
 *     unlikely to be a string comparison and not worth HLE'ing.
 */
static bool hle_rtl_equal_string(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t s1_ptr = hle_arg32(env, 0);
    uint32_t s2_ptr = hle_arg32(env, 1);
    uint8_t case_insensitive = (uint8_t)(hle_arg32(env, 2) & 0xff);

    uint16_t len1 = 0, len2 = 0;
    if (!g_read(s1_ptr,     &len1, 2)) return false;
    if (!g_read(s2_ptr,     &len2, 2)) return false;

    if (len1 != len2) {
        env->regs[R_EAX] = 0;  /* FALSE */
        hle_return_stdcall(env, 3);
        return true;
    }
    if (len1 == 0) {
        env->regs[R_EAX] = 1;  /* TRUE: both empty */
        hle_return_stdcall(env, 3);
        return true;
    }
    if (len1 > 0x800) {
        return false;  /* too long; let kernel handle */
    }

    uint32_t buf1 = 0, buf2 = 0;
    if (!g_read(s1_ptr + 4, &buf1, 4)) return false;
    if (!g_read(s2_ptr + 4, &buf2, 4)) return false;

    uint8_t b1[0x800], b2[0x800];
    if (!g_read(buf1, b1, len1)) return false;
    if (!g_read(buf2, b2, len1)) return false;

    bool equal = true;
    for (uint16_t i = 0; i < len1; i++) {
        uint8_t c1 = b1[i], c2 = b2[i];
        if (case_insensitive) {
            if (c1 >= 0x80 || c2 >= 0x80) {
                /* Non-ASCII; defer to kernel for ANSI code page handling. */
                return false;
            }
            if (c1 >= 'a' && c1 <= 'z') c1 -= 32;
            if (c2 >= 'a' && c2 <= 'z') c2 -= 32;
        }
        if (c1 != c2) { equal = false; break; }
    }

    env->regs[R_EAX] = equal ? 1 : 0;
    hle_return_stdcall(env, 3);
    return true;
}

/* ------------------------------------------------------------------ */
/*  SHA-1 native transform — for XcShaTransform HLE                    */
/* ------------------------------------------------------------------ */

/*
 * Plain C SHA-1 block transform (RFC 3174). Always available as a
 * fallback. ~10k cycles per block on native ARM vs ~500k cycles when
 * the guest x86 SHA-1 transform is run through TCG/Cranelift — 50x
 * speedup just from running native instead of emulated.
 */
#define SHA1_ROL(x, n) (((uint32_t)(x) << (n)) | ((uint32_t)(x) >> (32 - (n))))

static void sha1_transform_plain(uint32_t state[5], const uint8_t block[64])
{
    uint32_t w[80];
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24)
             | ((uint32_t)block[i * 4 + 1] << 16)
             | ((uint32_t)block[i * 4 + 2] <<  8)
             |  (uint32_t)block[i * 4 + 3];
    }
    for (i = 16; i < 80; i++) {
        w[i] = SHA1_ROL(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];

    for (i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999u;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDCu;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6u;
        }
        uint32_t tmp = SHA1_ROL(a, 5) + f + e + k + w[i];
        e = d; d = c; c = SHA1_ROL(b, 30); b = a; a = tmp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

#if defined(__aarch64__)
/*
 * ARMv8 SHA-1 crypto extension transform. ~80 cycles per block —
 * ~125x faster than plain C, ~6000x faster than emulated x86 SHA-1.
 *
 * Bit-exact reference: Jeffrey Walton's noloader/SHA-Intrinsics
 * sha1-arm.c. Matches OpenSSL crypto/sha/asm/sha1-armv8.pl and Linux
 * arch/arm64/crypto/sha1-ce-core.S patterns.
 *
 * Critical detail (the bug in the first attempt): TMP0/TMP1 are
 * double-buffered — each group's TMP is PREPARED in group N-2, so
 * K-constant transitions happen at:
 *   group 4 prep (rounds 12-15): K0→K1 (for group 6, rounds 20-23)
 *   group 9 prep (rounds 32-35): K1→K2 (for group 11, rounds 40-43)
 *   group 14 prep (rounds 52-55): K2→K3 (for group 16, rounds 60-63)
 *
 * Tail structure (groups 18-20):
 *   Group 18 (rounds 68-71): final add + final su1, NO su0
 *   Groups 19-20 (rounds 72-79): bare sha1h + sha1pq, consuming
 *     already-prepared TMP0/TMP1
 */
__attribute__((target("crypto")))
static void sha1_transform_neon(uint32_t state[5], const uint8_t block[64])
{
    const uint32x4_t k0 = vdupq_n_u32(0x5A827999u);
    const uint32x4_t k1 = vdupq_n_u32(0x6ED9EBA1u);
    const uint32x4_t k2 = vdupq_n_u32(0x8F1BBCDCu);
    const uint32x4_t k3 = vdupq_n_u32(0xCA62C1D6u);

    uint32x4_t ABCD = vld1q_u32(state);
    uint32_t   E0   = state[4];
    uint32x4_t ABCD_SAVED = ABCD;
    uint32_t   E0_SAVED   = E0;

    /* Load message + byte-swap LE → BE. */
    uint32x4_t MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block)));
    uint32x4_t MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 16)));
    uint32x4_t MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 32)));
    uint32x4_t MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 48)));

    uint32x4_t TMP0 = vaddq_u32(MSG0, k0);
    uint32x4_t TMP1 = vaddq_u32(MSG1, k0);
    uint32_t   E1;

    /* Group 1: rounds 0-3 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1cq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG2, k0);
    MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

    /* Group 2: rounds 4-7 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1cq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG3, k0);
    MSG0 = vsha1su1q_u32(MSG0, MSG3);
    MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

    /* Group 3: rounds 8-11 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1cq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG0, k0);
    MSG1 = vsha1su1q_u32(MSG1, MSG0);
    MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

    /* Group 4: rounds 12-15 — K0 → K1 prep for group 6 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1cq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG1, k1);
    MSG2 = vsha1su1q_u32(MSG2, MSG1);
    MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

    /* Group 5: rounds 16-19 — last sha1c */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1cq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG2, k1);
    MSG3 = vsha1su1q_u32(MSG3, MSG2);
    MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

    /* Group 6: rounds 20-23 — first sha1p */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG3, k1);
    MSG0 = vsha1su1q_u32(MSG0, MSG3);
    MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

    /* Group 7: rounds 24-27 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG0, k1);
    MSG1 = vsha1su1q_u32(MSG1, MSG0);
    MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

    /* Group 8: rounds 28-31 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG1, k1);
    MSG2 = vsha1su1q_u32(MSG2, MSG1);
    MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

    /* Group 9: rounds 32-35 — K1 → K2 prep for group 11 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG2, k2);
    MSG3 = vsha1su1q_u32(MSG3, MSG2);
    MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

    /* Group 10: rounds 36-39 — last sha1p in first p-phase */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG3, k2);
    MSG0 = vsha1su1q_u32(MSG0, MSG3);
    MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

    /* Group 11: rounds 40-43 — first sha1m */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1mq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG0, k2);
    MSG1 = vsha1su1q_u32(MSG1, MSG0);
    MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

    /* Group 12: rounds 44-47 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1mq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG1, k2);
    MSG2 = vsha1su1q_u32(MSG2, MSG1);
    MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

    /* Group 13: rounds 48-51 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1mq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG2, k2);
    MSG3 = vsha1su1q_u32(MSG3, MSG2);
    MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

    /* Group 14: rounds 52-55 — K2 → K3 prep for group 16 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1mq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG3, k3);
    MSG0 = vsha1su1q_u32(MSG0, MSG3);
    MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

    /* Group 15: rounds 56-59 — last sha1m */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1mq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG0, k3);
    MSG1 = vsha1su1q_u32(MSG1, MSG0);
    MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

    /* Group 16: rounds 60-63 — second sha1p phase */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG1, k3);
    MSG2 = vsha1su1q_u32(MSG2, MSG1);
    MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

    /* Group 17: rounds 64-67 */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E0, TMP0);
    TMP0 = vaddq_u32(MSG2, k3);
    MSG3 = vsha1su1q_u32(MSG3, MSG2);
    MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

    /* Group 18: rounds 68-71 — final add, final su1, NO su0 */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);
    TMP1 = vaddq_u32(MSG3, k3);
    MSG0 = vsha1su1q_u32(MSG0, MSG3);

    /* Group 19: rounds 72-75 — bare */
    E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E0, TMP0);

    /* Group 20: rounds 76-79 — bare */
    E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
    ABCD = vsha1pq_u32(ABCD, E1, TMP1);

    /* Final state add. */
    E0 += E0_SAVED;
    ABCD = vaddq_u32(ABCD_SAVED, ABCD);

    vst1q_u32(state, ABCD);
    state[4] = E0;
}
#endif /* __aarch64__ */

static bool g_sha1_use_neon;

/*
 * SHA-1 self-test using NIST FIPS-180 "abc" test vector.
 *
 * Block layout for SHA-1("abc"):
 *   bytes 0..2:  0x61 0x62 0x63 ('a' 'b' 'c')
 *   byte  3:     0x80 (padding bit)
 *   bytes 4..55: zeros
 *   bytes 56-63: 0x00 00 00 00 00 00 00 0x18 (length=24 bits, big-endian)
 *
 * Initial state (SHA-1 IV):  67452301 efcdab89 98badcfe 10325476 c3d2e1f0
 * Expected after 1 transform: a9993e36 4706816a ba3e2571 7850c26c 9cd0d89d
 *
 * Returns true if the transform produced the correct digest.
 */
static bool sha1_self_test(void (*xform)(uint32_t[5], const uint8_t[64]))
{
    static const uint8_t abc_block[64] = {
        'a','b','c',0x80, 0,0,0,0,  0,0,0,0,  0,0,0,0,
        0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,
        0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,
        0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0x18
    };
    static const uint32_t expected[5] = {
        0xa9993e36u, 0x4706816au, 0xba3e2571u,
        0x7850c26cu, 0x9cd0d89du
    };
    uint32_t state[5] = {
        0x67452301u, 0xefcdab89u, 0x98badcfeu,
        0x10325476u, 0xc3d2e1f0u
    };
    xform(state, abc_block);
    return memcmp(state, expected, sizeof(state)) == 0;
}

static void sha1_init(void)
{
    static bool initialized;
    if (initialized) return;
    initialized = true;

    /* Always verify plain C first — if this fails the build is broken. */
    if (!sha1_self_test(sha1_transform_plain)) {
        HLE_LOG("sha1 PLAIN C SELF-TEST FAILED — SHA-1 HLE disabled");
        g_sha1_use_neon = false;
        return;
    }

#if defined(__aarch64__)
    unsigned long caps = getauxval(AT_HWCAP);
    if (caps & HWCAP_SHA1) {
        if (sha1_self_test(sha1_transform_neon)) {
            g_sha1_use_neon = true;
            HLE_LOG("sha1 backend: ARMv8 crypto extension (validated)");
            return;
        }
        HLE_LOG("sha1 NEON SELF-TEST FAILED — falling back to plain C");
    } else {
        HLE_LOG("sha1 backend: plain C (HWCAP_SHA1 not present)");
    }
#else
    HLE_LOG("sha1 backend: plain C");
#endif
    g_sha1_use_neon = false;
}

static inline void sha1_transform(uint32_t state[5], const uint8_t block[64])
{
#if defined(__aarch64__)
    if (g_sha1_use_neon) {
        sha1_transform_neon(state, block);
        return;
    }
#endif
    sha1_transform_plain(state, block);
}

/*
 * void __stdcall XcShaTransform(uint32_t state[5], const uint8_t block[64])
 *
 * Internal SHA-1 transform helper at 0x80031550, called from
 * XcSHAUpdate per 64-byte block. ret 8 (stdcall, 2 args).
 *
 * Halo 2 hits this ~140k inner-PC samples per 10s window (~30
 * calls/s × ~80 round-iter PCs each). HLE replaces the entire 80-round
 * x86 implementation with a single native call.
 */
static bool hle_xc_sha_transform(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t state_ptr = hle_arg32(env, 0);
    uint32_t block_ptr = hle_arg32(env, 1);

    uint32_t state[5];
    uint8_t block[64];
    if (!g_read(state_ptr, state, sizeof(state))) return false;
    if (!g_read(block_ptr, block, sizeof(block))) return false;

    sha1_transform(state, block);

    if (!g_write(state_ptr, state, sizeof(state))) return false;
    hle_return_stdcall(env, 2);
    return true;
}

/*
 * VOID __stdcall KeStallExecutionProcessor(ULONG Microseconds)
 *
 * Kernel impl (0x800153a0) uses CPUID as a serialization barrier and
 * then spin-reads the TSC until enough cycles have elapsed. The first
 * cut of this HLE called host usleep() to be "helpful" but that was a
 * disaster: Halo 2 calls KeStall ~270 times/sec with tiny microsecond
 * values, and host usleep granularity is ~100µs+, so the vCPU thread
 * was sleeping ~30% of wall time and FPS collapsed (2026-05-22).
 *
 * On the emulator the guest x86 is already much slower than real
 * hardware due to TCG dispatch overhead — any stall the kernel asks
 * for is effectively already taken just by entering this function.
 * Return immediately. Halo 2 is calling KeStall for legacy SMBus/USB
 * register settling and the host can't observe those anyway, so
 * skipping the wait has no functional effect.
 */
static bool hle_ke_stall_execution_processor(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    hle_return_stdcall(env, 1);
    return true;
}

/*
 * NTSTATUS KeDelayExecutionThread(KPROCESSOR_MODE WaitMode,
 *                                  BOOLEAN Alertable,
 *                                  PLARGE_INTEGER Interval)
 * __stdcall, 3 args. We special-case Interval==0 (and Interval->Quad==0)
 * as a pure yield — return STATUS_SUCCESS immediately. Real timeouts
 * still go through the kernel; we don't intercept those because they
 * need real wait-queue plumbing.
 *
 * For now: return STATUS_SUCCESS unconditionally (treat as instant
 * timeout). This matches the "yield-as-flush-window" goal but is more
 * aggressive than strictly correct. Gated by X1BOX_HLE_YIELD=1.
 */
static bool hle_ke_delay_execution_thread(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    /* args ignored — we'd need to check Interval to be strictly correct */
    env->regs[R_EAX] = 0; /* STATUS_SUCCESS */
    hle_return_stdcall(env, 3);
    return true;
}

/*
 * ULONG __stdcall KeQueryPerformanceFrequency(VOID)
 *
 * Kernel disasm (0x80015350):
 *   mov eax, 0x337f98
 *   xor edx, edx
 *   ret
 *
 * The Xbox kernel returns a hardcoded 3,375,000 Hz — the scaled rate
 * of the ACPI PMTMR at I/O port 0x8008. Trivial HLE: just stuff the
 * constant in EAX and return.
 *
 * Note this returns a ULONG (32-bit) on Xbox, not a LARGE_INTEGER like
 * Windows. EDX is zeroed by the kernel for ABI cleanliness.
 */
#define XKRNL_PERF_FREQUENCY_HZ 3375000u

static bool hle_ke_query_performance_frequency(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    env->regs[R_EAX] = XKRNL_PERF_FREQUENCY_HZ;
    env->regs[R_EDX] = 0;
    hle_return_stdcall(env, 0);
    return true;
}

/*
 * LARGE_INTEGER __stdcall KeQueryPerformanceCounter(VOID)
 *
 * Kernel disasm (0x80015314) reads I/O port 0x8008 and combines it
 * with a kernel-maintained high-half tracker (DAT_80035a78 etc.).
 * Each call triggers an MMIO emulation roundtrip in xemu.
 *
 * HLE strategy: bypass the I/O port entirely and convert QEMU's
 * VIRTUAL clock (nanoseconds in guest virtual time) into ticks at
 * the kernel's reported 3.375 MHz frequency.
 *
 * Using QEMU_CLOCK_VIRTUAL (not host monotonic) is load-bearing:
 * xemu's PMTMR emulation also ticks off the virtual clock, so this
 * HLE returns values consistent with what the rest of the emulator
 * believes about elapsed guest time. Otherwise frame-delta math in
 * Halo 2 would desync with the audio/video subsystems.
 *
 * Conversion: ticks = ns * 3375000 / 1e9 = ns * 27 / 8000. Exact.
 *
 * Returns 64-bit value via EDX:EAX (stdcall LARGE_INTEGER return).
 */
static bool hle_ke_query_performance_counter(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    int64_t virtual_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    uint64_t ticks = ((uint64_t)virtual_ns * 27ull) / 8000ull;
    env->regs[R_EAX] = (uint32_t)(ticks & 0xFFFFFFFFu);
    env->regs[R_EDX] = (uint32_t)(ticks >> 32);
    hle_return_stdcall(env, 0);
    return true;
}

/* Dispatcher object header offsets (shared by KeSetEvent, KeWait*, etc). */
#define KOBJ_TYPE_OFFSET         0x00
#define KOBJ_SIGNAL_STATE_OFFSET 0x04
#define KOBJ_WAIT_LIST_OFFSET    0x08
#define KOBJ_TYPE_EVENT_NOTIFY   0
#define KOBJ_TYPE_EVENT_SYNC     1
#define KOBJ_TYPE_SEMAPHORE      5

/*
 * LONG __stdcall KeSetEvent(PRKEVENT Event, KPRIORITY Increment, BOOLEAN Wait)
 *
 * Kernel disasm (0x8001a518):
 *   raise IRQL to DISPATCH (FUN_80014348)
 *   read old SignalState
 *   if (no waiters): SignalState = 1
 *   else: wake waiters via FUN_8001befb
 *   if (Wait == FALSE): call FUN_80018ba0 (lower IRQL / swap thread)
 *   else: save IRQL state into KPCR[0x56] + KPCR[0x54] for caller's next wait
 *   return old SignalState
 *
 * The Wait=TRUE side effect is CRITICAL: the caller intends to call a
 * Wait function next, and the kernel keeps IRQL at DISPATCH and stows
 * the saved IRQL into KPCR so the chained wait can pick it up. Skipping
 * that breaks the SetEvent+Wait pattern and the thread never unblocks
 * cleanly → kernel idle loop hits 285K/s (observed 2026-05-22).
 *
 * HLE fast path:
 *   - Wait must be FALSE (decline TRUE; can't simulate IRQL save)
 *   - Wait list empty (decline if waiters; can't easily wake them)
 *   - Just read SignalState, write 1, return old SignalState
 *
 * On single-vCPU the kernel's IRQL raise/lower around the SignalState
 * mutation is a net no-op (same IRQL in and out) since we can't take
 * an interrupt mid-handler. So we can skip the IRQL dance entirely
 * for the fast path.
 */
static bool hle_ke_set_event(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t event_ptr = hle_arg32(env, 0);
    uint32_t wait_arg  = hle_arg32(env, 2);  /* 3rd stdcall arg = Wait */

    if (wait_arg != 0) {
        /* Caller intends to immediately wait — kernel must save IRQL
         * into KPCR for the chained wait call. Decline. */
        return false;
    }

    /* Wait list empty? */
    uint32_t flink = 0;
    if (!g_read32(event_ptr + KOBJ_WAIT_LIST_OFFSET, &flink)) return false;
    if (flink != event_ptr + KOBJ_WAIT_LIST_OFFSET) {
        /* Waiters exist — kernel needs to wake them. Decline. */
        return false;
    }

    int32_t old_state = 0;
    if (!g_read(event_ptr + KOBJ_SIGNAL_STATE_OFFSET, &old_state, 4)) {
        return false;
    }
    int32_t one = 1;
    g_write(event_ptr + KOBJ_SIGNAL_STATE_OFFSET, &one, 4);

    env->regs[R_EAX] = old_state;  /* return previous SignalState */
    hle_return_stdcall(env, 3);
    return true;
}

/*
 * VOID __stdcall KeQuerySystemTime(PLARGE_INTEGER CurrentTime)
 *
 * Kernel disasm (0x80019f60) does the standard 64-bit-read-with-
 * consistency-check from kernel-tracked globals at 0x8003a880/884/888.
 * Returns the system time as a 100-nanosecond count.
 *
 * HLE bypasses the global-polling and computes directly from
 * QEMU_CLOCK_VIRTUAL. Same logic as KeQueryPerformanceCounter but
 * different unit conversion (100ns Windows-FILETIME style).
 */
static bool hle_ke_query_system_time(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t out_ptr = hle_arg32(env, 0);

    int64_t virtual_ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    /* 100ns units (Windows FILETIME semantics). */
    int64_t xbox_time = virtual_ns / 100;

    if (!g_write(out_ptr, &xbox_time, 8)) {
        return false;
    }
    hle_return_stdcall(env, 1);
    return true;
}

/*
 * NTSTATUS __stdcall KeWaitForSingleObject(
 *     PVOID Object,
 *     KWAIT_REASON WaitReason,
 *     KPROCESSOR_MODE WaitMode,
 *     BOOLEAN Alertable,
 *     PLARGE_INTEGER Timeout)
 *
 * The full kernel implementation (0x80019cdc) handles mutex acquisition,
 * semaphore decrement, APC delivery, alertability, timeout queuing — all
 * the wait-queue plumbing we can't easily replicate.
 *
 * BUT — empirically the dominant case in Halo 2 gameplay is "the object
 * is already signalled when we ask, so just return SUCCESS without
 * blocking". D3D fences, audio buffer-ready events, completed file I/O
 * all sit in this pattern. We HLE only that path and decline everything
 * else; the kernel's real implementation runs untouched for the rare
 * actually-blocking case.
 *
 * Dispatcher object header (DISPATCHER_HEADER):
 *   +0x00 Type           (UCHAR: 0=EventNotify, 1=EventSync, 2=Mutex,
 *                                4=Queue, 5=Semaphore, 6=Thread, …)
 *   +0x01 Absolute
 *   +0x02 Size
 *   +0x03 Inserted
 *   +0x04 SignalState    (LONG)
 *   +0x08 WaitListHead.Flink
 *   +0x0c WaitListHead.Blink
 *
 * Fast path: SignalState >= 1 AND Type is Event{Notify,Sync,Semaphore}.
 *   Type 0 (notification event): leave SignalState alone, return SUCCESS.
 *   Type 1 (synchronization event): clear SignalState to 0, return SUCCESS.
 *   Type 5 (semaphore): decrement SignalState by 1, return SUCCESS.
 *
 * Anything else (Type=Mutex, SignalState<1, anything weird): decline,
 * let the kernel run its real path.
 *
 * Halo 2 often passes Alertable=FALSE / Timeout=NULL on the hot path,
 * so we don't need to inspect those when we're in the signalled-fast-path.
 */
static bool hle_ke_wait_for_single_object(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t obj_ptr = hle_arg32(env, 0);

    uint8_t type = 0xff;
    if (!g_read(obj_ptr + KOBJ_TYPE_OFFSET, &type, 1)) return false;
    if (type != KOBJ_TYPE_EVENT_NOTIFY &&
        type != KOBJ_TYPE_EVENT_SYNC &&
        type != KOBJ_TYPE_SEMAPHORE) {
        /* Mutex, thread, queue, etc. — too complex for fast path. */
        return false;
    }

    int32_t signal_state = 0;
    if (!g_read(obj_ptr + KOBJ_SIGNAL_STATE_OFFSET, &signal_state, 4)) {
        return false;
    }
    if (signal_state < 1) {
        /* Not signalled — would need to actually block. Decline. */
        return false;
    }

    /* Check the wait list — if anyone is already queued ahead of us,
     * the kernel needs to do its priority ordering. Decline. */
    uint32_t wait_flink = 0;
    if (!g_read32(obj_ptr + KOBJ_WAIT_LIST_OFFSET, &wait_flink)) {
        return false;
    }
    /* WaitListHead is a circular doubly-linked list; if empty, Flink
     * points back to the head (obj_ptr + 0x08). */
    if (wait_flink != obj_ptr + KOBJ_WAIT_LIST_OFFSET) {
        return false;
    }

    /* Signalled-fast-path: update signal state per object type. */
    if (type == KOBJ_TYPE_EVENT_SYNC) {
        int32_t zero = 0;
        g_write(obj_ptr + KOBJ_SIGNAL_STATE_OFFSET, &zero, 4);
    } else if (type == KOBJ_TYPE_SEMAPHORE) {
        signal_state--;
        g_write(obj_ptr + KOBJ_SIGNAL_STATE_OFFSET, &signal_state, 4);
    }
    /* EventNotify: leave SignalState set (notification events stay
     * signalled until KeResetEvent). */

    env->regs[R_EAX] = 0;  /* STATUS_SUCCESS */
    hle_return_stdcall(env, 5);
    return true;
}

/*
 * VOID __stdcall RtlEnterCriticalSection(PRTL_CRITICAL_SECTION CriticalSection)
 *
 * Kernel disasm (0x80023120):
 *   mov ecx, [PTR_DAT_80035c04]   ; KPCR.PrcbData.CurrentThread
 *   mov edx, [esp+4]              ; CS ptr
 *   inc [edx+0x10]                ; LockCount++
 *   jnz contended
 *   mov [edx+0x18], ecx           ; OwningThread = current
 *   mov [edx+0x14], 1             ; RecursionCount = 1
 *   ret 4
 * contended:
 *   cmp [edx+0x18], ecx           ; OwningThread == current?
 *   jne wait
 *   inc [edx+0x14]                ; recursive: RecursionCount++
 *   ret 4
 * wait:
 *   call KeWaitForSingleObject ...
 *   ...
 *
 * RTL_CRITICAL_SECTION layout:
 *   +0x00 DebugInfo
 *   +0x10 LockCount         (-1 = unheld, 0 = held by 1, >0 = contended)
 *   +0x14 RecursionCount    (acquire depth)
 *   +0x18 OwningThread      (KTHREAD*)
 *   +0x1c LockSemaphore     (KSEMAPHORE for contention)
 *
 * Two fast paths we HLE:
 *   1. Uncontended (LockCount was -1): just claim it.
 *   2. Recursive (same thread re-entering): bump RecursionCount.
 *
 * Contended-by-other-thread case (LockCount >= 0 && OwningThread !=
 * current): decline → kernel handles the real KeWait. We do NOT mutate
 * env state in that case; the kernel function runs from PC X normally.
 */
#define CS_LOCK_COUNT_OFFSET     0x10
#define CS_RECURSION_OFFSET      0x14
#define CS_OWNING_THREAD_OFFSET  0x18
#define XKRNL_CURRENT_THREAD_PTR 0x80035c04u

static bool hle_rtl_enter_critical_section(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t cs_ptr = hle_arg32(env, 0);

    /* Pre-check: is this lock held by a *different* thread? If so, we
     * need the kernel's wait path. Do NOT mutate state. */
    int32_t lock_count = 0;
    if (!g_read(cs_ptr + CS_LOCK_COUNT_OFFSET, &lock_count, 4)) return false;

    uint32_t current_thread = 0;
    if (!g_read32(XKRNL_CURRENT_THREAD_PTR, &current_thread)) return false;

    if (lock_count != -1) {
        /* Lock is held — by whom? */
        uint32_t owner = 0;
        if (!g_read(cs_ptr + CS_OWNING_THREAD_OFFSET, &owner, 4)) return false;
        if (owner != current_thread) {
            /* Contended by other thread — decline, let kernel KeWait. */
            return false;
        }
    }

    /* Safe to handle: either uncontended or recursive. */
    lock_count++;
    g_write(cs_ptr + CS_LOCK_COUNT_OFFSET, &lock_count, 4);

    if (lock_count == 0) {
        /* First acquire (was -1) — set owner + recursion=1. */
        g_write(cs_ptr + CS_OWNING_THREAD_OFFSET, &current_thread, 4);
        int32_t one = 1;
        g_write(cs_ptr + CS_RECURSION_OFFSET, &one, 4);
    } else {
        /* Recursive — bump recursion count. */
        int32_t recursion = 0;
        g_read(cs_ptr + CS_RECURSION_OFFSET, &recursion, 4);
        recursion++;
        g_write(cs_ptr + CS_RECURSION_OFFSET, &recursion, 4);
    }

    hle_return_stdcall(env, 1);
    return true;
}

/*
 * VOID __stdcall RtlLeaveCriticalSection(PRTL_CRITICAL_SECTION CS)
 *
 * Kernel disasm (0x800231d0):
 *   mov edx, [esp+4]            ; CS ptr
 *   xor eax, eax
 *   dec [edx+0x14]              ; RecursionCount--
 *   jnz still_held
 *   mov [edx+0x18], 0           ; OwningThread = NULL
 *   ... LockCount-- with overflow detect → KeReleaseSemaphore if waiters
 * still_held:
 *   dec [edx+0x10]              ; LockCount--
 *   ret 4
 *
 * We HLE everything EXCEPT the path where LockCount-- crosses to a value
 * indicating waiters need to be woken (KeReleaseSemaphore). That path
 * requires real wait-queue plumbing; decline → kernel handles it.
 *
 * Detection: if recursion goes to 0 AND LockCount was > 0 (waiters exist),
 * decline. Otherwise HLE.
 */
static bool hle_rtl_leave_critical_section(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    uint32_t cs_ptr = hle_arg32(env, 0);

    int32_t recursion = 0, lock_count = 0;
    if (!g_read(cs_ptr + CS_RECURSION_OFFSET, &recursion, 4)) return false;
    if (!g_read(cs_ptr + CS_LOCK_COUNT_OFFSET, &lock_count, 4)) return false;

    /* If this is the last release AND there are waiters, decline so the
     * kernel can KeReleaseSemaphore the next waiter. */
    if (recursion == 1 && lock_count > 0) {
        return false;
    }

    recursion--;
    g_write(cs_ptr + CS_RECURSION_OFFSET, &recursion, 4);

    if (recursion != 0) {
        /* Still held recursively — just decrement LockCount. */
        lock_count--;
        g_write(cs_ptr + CS_LOCK_COUNT_OFFSET, &lock_count, 4);
    } else {
        /* Final release with no waiters. Clear owner + decrement to -1. */
        int32_t zero = 0;
        g_write(cs_ptr + CS_OWNING_THREAD_OFFSET, &zero, 4);
        lock_count--;
        g_write(cs_ptr + CS_LOCK_COUNT_OFFSET, &lock_count, 4);
    }

    hle_return_stdcall(env, 1);
    return true;
}

/*
 * NTSTATUS NtYieldExecution(VOID)  __stdcall, 0 args.
 *
 * Kernel disasm (0x8001aef3):
 *   cmp [0x8003aa00], 0        ; ready-queue mask
 *   push edi
 *   mov edi, 0x40000024        ; STATUS_NO_YIELD_PERFORMED preloaded
 *   jz .return_no_yield        ; queue empty -> nothing to swap to
 *   ...                        ; else rotate ready queue + KiSwapThread
 *
 * On single-vCPU emulation the only correct return is
 * STATUS_NO_YIELD_PERFORMED (0x40000024) — there is no other host CPU
 * to swap to, and any guest-thread scheduling would have to be done by
 * the kernel itself (which we'd skip with this HLE). Returning
 * STATUS_SUCCESS (the previous behaviour) makes the caller assume a
 * yield happened — Halo 2's yield-poll loops then hog the vCPU
 * because they think render/audio threads already got time. That
 * caused the 22.7 → 14.9 FPS regression observed 2026-05-22.
 *
 * The previous version of this handler also called cpu_exit() to
 * force a dispatcher round-trip on every yield. That made FPS jitter
 * MUCH worse — the Cranelift chain helper started exiting at avg 4
 * iters with 99.9% spin-detect false-positives, and audio/render
 * frame pacing collapsed. The dispatcher already re-checks
 * cpu->interrupt_request between every chain iteration, so the
 * forced exit was redundant *and* destroyed the chain helper's
 * amortisation. Removed 2026-05-22.
 */
static bool hle_nt_yield_execution(X86CPU *cpu)
{
    CPUX86State *env = &cpu->env;
    env->regs[R_EAX] = 0x40000024;             /* STATUS_NO_YIELD_PERFORMED */
    hle_return_stdcall(env, 0);
    return true;
}

/*
 * VOID KiIdleLoop(VOID)  — kernel idle thread, infinite loop:
 *   do {
 *     do {
 *       sti; nop; nop; cli;
 *       if (DPC_list != empty) drain;          ; PTR_LOOP_80035c2c
 *       if (NextThread == NULL) continue;      ; DAT_80035c08
 *     } while (NextThread == NULL);
 *     CurrentThread = NextThread; NextThread = NULL;
 *     KiSwapThread();
 *   } while (1);
 *
 * After all the HLE additions, this is the dominant unhooked PC group
 * on Halo 2 title screen: ~982k hits per 10s window. The vCPU burns
 * iters whenever the guest goes idle (waiting for audio/video sync /
 * timer fire to wake a thread).
 *
 * Hook installed at 0x8001b02e (inner-loop top, post-sti). When both
 * DPC list is empty AND NextThread is NULL, call sched_yield() — host
 * scheduler gets a chance to advance audio/render threads + deliver
 * timer interrupts that feed NextThread.
 *
 * Always DECLINE (return false) — the kernel still runs its own spin
 * TB. cpu->interrupt_request is checked between chain iters, so any
 * IRQ posted during our yield gets delivered on the next iter.
 *
 * Rate limited: only check + yield every 64 iters (thread-local
 * counter). sched_yield is cheap (~50ns uncontended) but we don't want
 * to thrash the scheduler.
 */
/*
 * QEMU/TCG splits TBs at `sti` (the IF flag change makes it a system
 * instruction). So the kernel's `sti; nop; nop; cli; cmp ebp,[ebp]`
 * sequence at 0x8001b02e is actually two TBs: one ending AT sti, and
 * the next one starting AT the nop at 0x8001b02f. Hooking at 0x8001b02e
 * gets 0 hits because the chain back-edge from 0x8001b047 (jz -0x1b)
 * jumps to 0x8001b02e but the TB at 0x8001b02e ends immediately after
 * sti, and the *next* TB (the one running the spin body) starts at
 * 0x8001b02f. Empirically: 0x8001b02e hits=0, 0x8001b02f hits=710k.
 */
#define KI_IDLE_LOOP_INNER_VA    0x8001b02fu
#define XKRNL_DPC_LIST_HEAD_VA   0x80035c2cu  /* KPCR + 0x50 */
#define XKRNL_NEXT_THREAD_VA     0x80035c08u  /* KPCR + 0x2c */

/*
 * Telemetry for the KiIdleLoop spin path. Three counters tell us why
 * the guest is parked: actual idle (both DPC and NextThread empty),
 * waiting for a queued DPC, or waiting for a thread swap.
 *
 * The initial attempt called sched_yield() on actual-idle iters
 * (2026-05-22). FPS dropped from ~25 to ~17 — sched_yield while holding
 * BQL doesn't help because BQL-waiting threads (audio, render) still
 * can't take BQL. We just added latency to the IRQ-delivery path that
 * would have woken the spin naturally. Removed; handler is now pure
 * telemetry.
 */
static uint64_t g_idle_loop_idle;             /* both empty */
static uint64_t g_idle_loop_dpc_pending;
static uint64_t g_idle_loop_next_thread_pending;

static bool hle_ki_idle_loop_spin(X86CPU *cpu)
{
    (void)cpu;
    static __thread uint32_t iter;
    iter++;
    if ((iter & 0x3F) != 0) return false;  /* sample every 64 iters */

    /* DPC list head is "empty" when its Flink points back to the head VA
     * itself (circular linked list with sentinel). */
    uint32_t dpc_flink = 0;
    if (!g_read32(XKRNL_DPC_LIST_HEAD_VA, &dpc_flink)) return false;
    if (dpc_flink != XKRNL_DPC_LIST_HEAD_VA) {
        g_idle_loop_dpc_pending++;
        return false;
    }

    uint32_t next_thread = 0;
    if (!g_read32(XKRNL_NEXT_THREAD_VA, &next_thread)) return false;
    if (next_thread != 0) {
        g_idle_loop_next_thread_pending++;
        return false;
    }

    g_idle_loop_idle++;
    return false;
}

/* ------------------------------------------------------------------ */
/*  PE export resolver                                                 */
/* ------------------------------------------------------------------ */

/*
 * Xbox kernel image base. The MCPX BIOS loads xboxkrnl.exe at this VA
 * by convention; every retail kernel revision uses the same address.
 */
#define XKRNL_IMAGE_BASE 0x80010000u

/*
 * Ordinal → handler table.
 *
 * Ordinals from xboxkrnl.exe public exports. Cross-referenced against
 * the OpenXDK kernel header (xboxkrnl.def) and confirmed against the
 * Cxbx-Reloaded ordinal table.
 *
 * If a future kernel build remaps these we'll miss; xbox_hle_log_stats
 * surfaces hits=0 as a debugging signal.
 */
struct ord_entry {
    uint16_t ordinal;
    const char *name;
    XboxHleHandler handler;
    bool *gate;             /* points at g_gate_* */
    /*
     * Prologue signature: a sequence of expected bytes at the function
     * entry. The resolver verifies these match before installing the
     * hook. NULL or zero-length means "log the actual bytes so we can
     * harvest a signature, but don't install yet" — see
     * X1BOX_HLE_PROBE flow below.
     */
    const uint8_t *prologue;
    size_t prologue_len;
};

/*
 * Canonical Xbox kernel ordinals. The values below come from the
 * external_locations of Halo 2 retail (loaded into Ghidra) — Ghidra's
 * XBE plugin stores the kernel ordinal in the `address` field as
 * `0x80000000 | ordinal`, which is the real export ordinal index used
 * by the kernel's PE export table at 0x800102c0.
 *
 * 2026-05-21 history: my first cut guessed MS NT ordinals; ord 161
 * landed on KfLowerIrql (which expects an IRQL byte in CL — a small
 * integer like 2/30), my "spinlock acquire" handler treated CL as a
 * pointer and wrote 1 to that virtual address → NULL-page write →
 * KeBugCheckEx 0x1e KMODE_EXCEPTION_NOT_HANDLED. Lesson load-bearing
 * in feedback_hle_ordinal_ground_truth.
 *
 * Addresses + first-8-byte signatures harvested 2026-05-22 from the
 * running Halo 2 retail kernel via the xboxkrnl.exe dump in Ghidra
 * (project xbox_OG, /tmp/xboxkrnl.bin). Functions visible at:
 *   ord  99 KeDelayExecutionThread       @ 0x8001980a
 *   ord 126 KeQueryPerformanceCounter    @ 0x80015314
 *   ord 127 KeQueryPerformanceFrequency  @ 0x80015350
 *   ord 128 KeQuerySystemTime            @ 0x80019f60
 *   ord 151 KeStallExecutionProcessor    @ 0x800153a0
 *   ord 158 KeWaitForMultipleObjects     @ 0x800199db
 *   ord 159 KeWaitForSingleObject        @ 0x80019cdc
 *   ord 160 KfRaiseIrql                  @ 0x80014338
 *   ord 161 KfLowerIrql                  @ 0x80014368
 *   ord 238 NtYieldExecution             @ 0x8001aef3
 */

/* First-8-byte function prologues, from Ghidra read_memory on the
 * kernel image. Long enough to be unique across 366 exports; tolerates
 * minor compiler variants by being short. */
static const uint8_t SIG_NT_YIELD_EXECUTION[] = {
    /* cmp [0x8003aa00], 0; push edi; mov edi, 0x40000024 */
    0x83, 0x3d, 0x00, 0xaa, 0x03, 0x80, 0x00, 0x57
};
static const uint8_t SIG_KE_DELAY_EXECUTION_THREAD[] = {
    /* push ebp; mov ebp, esp; sub esp, 0x10; push ebx; push esi */
    0x55, 0x8b, 0xec, 0x83, 0xec, 0x10, 0x53, 0x56
};
static const uint8_t SIG_KF_RAISE_IRQL[] = {
    /* xor eax, eax; mov al, [0x80035c00]; mov [0x80035c00], cl */
    0x33, 0xc0, 0xa0, 0x00, 0x5c, 0x03, 0x80, 0x88
};
static const uint8_t SIG_KF_LOWER_IRQL[] = {
    /* and ecx, 0xff; pushfd; cli; mov [0x80035c00], ecx */
    0x81, 0xe1, 0xff, 0x00, 0x00, 0x00, 0x9c, 0xfa
};
static const uint8_t SIG_KE_STALL_EXECUTION_PROCESSOR[] = {
    /* push ebx; xor eax, eax; cpuid; pop ebx; mov ecx, [esp+4] */
    0x53, 0x33, 0xc0, 0x0f, 0xa2, 0x5b, 0x8b, 0x4c
};
static const uint8_t SIG_KE_QUERY_PERF_COUNTER[] = {
    /* push ebx; mov ecx, [DAT_80035a7c]; mov ebx, [DAT_80035a74]; cmp ecx, [DAT_80035a78] */
    0x53, 0x8b, 0x0d, 0x7c, 0x5a, 0x03, 0x80, 0x8b
};
static const uint8_t SIG_KE_QUERY_PERF_FREQUENCY[] = {
    /* mov eax, 0x337f98; xor edx, edx; ret */
    0xb8, 0x98, 0x7f, 0x33, 0x00, 0x33, 0xd2, 0xc3
};
static const uint8_t SIG_RTL_ENTER_CRITICAL_SECTION[] = {
    /* mov ecx, [0x80035c04]; mov edx, [esp+4]; inc [edx+0x10]; jnz */
    0x8b, 0x0d, 0x04, 0x5c, 0x03, 0x80, 0x8b, 0x54
};
static const uint8_t SIG_RTL_LEAVE_CRITICAL_SECTION[] = {
    /* mov edx, [esp+4]; xor eax, eax; dec [edx+0x14]; jnz */
    0x8b, 0x54, 0x24, 0x04, 0x33, 0xc0, 0xff, 0x4a
};
static const uint8_t SIG_KE_WAIT_FOR_SINGLE_OBJECT[] = {
    /* push ebp; mov ebp, esp; sub esp, 0x2c; push ebx; push esi;
     * mov esi, [0x80035c04] */
    0x55, 0x8b, 0xec, 0x83, 0xec, 0x2c, 0x53, 0x56
};
static const uint8_t SIG_KE_SET_EVENT[] = {
    /* push ebx; push esi; push edi; call FUN_80014348; mov ecx, [esp+0x10] */
    0x53, 0x56, 0x57, 0xe8, 0x28, 0x9e, 0xff, 0xff
};
static const uint8_t SIG_KE_QUERY_SYSTEM_TIME[] = {
    /* jmp +2; pause; mov eax, [0x8003a884]; mov ecx, [0x8003a880] */
    0xeb, 0x02, 0xf3, 0x90, 0xa1, 0x84, 0xa8, 0x03
};
static const uint8_t SIG_KE_RAISE_IRQL_TO_DPC_LEVEL[] = {
    /* xor eax, eax; mov al, [0x80035c00]; mov byte [0x80035c00], 2 */
    0x33, 0xc0, 0xa0, 0x00, 0x5c, 0x03, 0x80, 0xc6
};
static const uint8_t SIG_OBF_DEREFERENCE_OBJECT[] = {
    /* push ebp; mov ebp, esp; push ecx; push ecx; push esi; lea esi, [ecx-0x10] */
    0x55, 0x8b, 0xec, 0x51, 0x51, 0x56, 0x8d, 0x71
};
static const uint8_t SIG_KE_INSERT_QUEUE_DPC[] = {
    /* push ebp; mov ebp,esp; push ecx; push ebx; mov cl,0x1f; call KfRaiseIrql */
    0x55, 0x8b, 0xec, 0x51, 0x53, 0xb1, 0x1f, 0xe8
};
static const uint8_t SIG_KE_REMOVE_QUEUE_DPC[] = {
    /* cli; mov ecx,[esp+4]; mov al,[ecx+2]; test al, al */
    0xfa, 0x8b, 0x4c, 0x24, 0x04, 0x8a, 0x41, 0x02
};
static const uint8_t SIG_RTL_EQUAL_STRING[] = {
    /* push ebp; mov ebp,esp; mov ecx,[ebp+8]; mov edx,[ebp+0xC] */
    0x55, 0x8b, 0xec, 0x8b, 0x4d, 0x08, 0x8b, 0x55
};

#define SIG(arr) arr, sizeof(arr)

static const struct ord_entry g_ordinal_table[] = {
    /*
     * NtYieldExecution: zero-arg stdcall. Single-vCPU has no other
     * processors to yield to; the only meaningful side effect is
     * letting the host scheduler pick another thread. Our handler
     * returns STATUS_SUCCESS and pops 0 stack args. Halo 2's main
     * loop calls this and ignores the return value.
     */
    { 238, "NtYieldExecution",          hle_nt_yield_execution,        &g_gate_yield,
        SIG(SIG_NT_YIELD_EXECUTION) },
    /*
     * KfRaiseIrql: fastcall, ECX = new IRQL byte, returns old IRQL in
     * AL. Faithful HLE: read/write [0x80035c00] (the kernel's absolute
     * PrcbData.CurrentIrql). No side effects beyond the byte update —
     * raising IRQL cannot fire new IRQs by definition.
     */
    { 160, "KfRaiseIrql",               hle_kf_raise_irql,             &g_gate_kf,
        SIG(SIG_KF_RAISE_IRQL) },
    /*
     * KeStallExecutionProcessor: stdcall, [esp+4] = microseconds.
     * Kernel spin-reads TSC; we replace with host usleep so the OS can
     * schedule other host threads during the stall. Clamped at 1ms.
     */
    { 151, "KeStallExecutionProcessor", hle_ke_stall_execution_processor, &g_gate_kf,
        SIG(SIG_KE_STALL_EXECUTION_PROCESSOR) },
    /*
     * KeQueryPerformanceCounter / Frequency: high-frequency timer reads.
     * Halo 2 calls KeQueryPerformanceCounter for every frame's delta,
     * potentially multiple times per frame for sub-frame timings. Each
     * call in the original hits I/O port 0x8008 via xemu's MMIO emulation
     * (~500-1000 host cycles). HLE bypasses the port by reading
     * qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) directly.
     */
    { 126, "KeQueryPerformanceCounter",   hle_ke_query_performance_counter,   &g_gate_kf,
        SIG(SIG_KE_QUERY_PERF_COUNTER) },
    { 127, "KeQueryPerformanceFrequency", hle_ke_query_performance_frequency, &g_gate_kf,
        SIG(SIG_KE_QUERY_PERF_FREQUENCY) },
    /*
     * Rtl{Enter,Leave}CriticalSection — user-mode locks. Halo 2 calls
     * these heavily for thread-safe game state mutation. On single-vCPU
     * uncontended (the norm) they're just inc/dec on the CS struct.
     * Contended-by-other-thread case declines back to the kernel so the
     * real KeWaitForSingleObject runs.
     */
    { 277, "RtlEnterCriticalSection",     hle_rtl_enter_critical_section,     &g_gate_rtl,
        SIG(SIG_RTL_ENTER_CRITICAL_SECTION) },
    { 294, "RtlLeaveCriticalSection",     hle_rtl_leave_critical_section,     &g_gate_rtl,
        SIG(SIG_RTL_LEAVE_CRITICAL_SECTION) },
    /*
     * KeWaitForSingleObject — signalled-event fast path. Every D3D fence
     * check, audio buffer-ready event, file-I/O completion wait. The full
     * kernel implementation is heavyweight; the handler only takes the
     * call when the object is already signalled (no actual blocking
     * needed). Mutex/thread/queue/non-signalled paths decline and run
     * the real kernel.
     */
    { 159, "KeWaitForSingleObject",       hle_ke_wait_for_single_object,      &g_gate_kf,
        SIG(SIG_KE_WAIT_FOR_SINGLE_OBJECT) },
    /* KeSetEvent fast path (no waiters): cheap signal flip. */
    { 145, "KeSetEvent",                  hle_ke_set_event,                   &g_gate_kf,
        SIG(SIG_KE_SET_EVENT) },
    /* KeQuerySystemTime: read host virtual clock instead of kernel
     * globals. 100ns FILETIME units. */
    { 128, "KeQuerySystemTime",           hle_ke_query_system_time,           &g_gate_kf,
        SIG(SIG_KE_QUERY_SYSTEM_TIME) },
    /*
     * KeDelayExecutionThread: probe-only. Blanket STATUS_SUCCESS would
     * break non-zero-Interval callers (frame pacing, asset loaders).
     * Needs an Interval-inspect path that only HLE's the Interval==0
     * yield case.
     */
    {  99, "KeDelayExecutionThread",    NULL,                          &g_gate_yield,
        SIG(SIG_KE_DELAY_EXECUTION_THREAD) },
    /*
     * KfLowerIrql: dominant unhooked hotspot (0x80014386 at 122K/s on
     * Halo 2 title screen). Fast path = no pending IRQs eligible at
     * new level → just write IRQL byte and return. Slow path declines
     * to kernel which dispatches the pending handler.
     */
    { 161, "KfLowerIrql",               hle_kf_lower_irql,             &g_gate_kf,
        SIG(SIG_KF_LOWER_IRQL) },
    /*
     * KeRaiseIrqlToDpcLevel: stdcall 0-arg helper that's a thin wrapper
     * around KfRaiseIrql(DISPATCH_LEVEL). Used by every dispatcher-lock
     * site in the kernel (KiLockDispatcherDatabase aliases to it on
     * uniprocessor Xbox per Cxbx-R EmuKrnlKi.h:250). Trivial HLE.
     */
    { 129, "KeRaiseIrqlToDpcLevel",     hle_ke_raise_irql_to_dpc_level, &g_gate_kf,
        SIG(SIG_KE_RAISE_IRQL_TO_DPC_LEVEL) },
    /*
     * ObfDereferenceObject: fastcall (ECX=Object). Hot path is
     * "refcount > 1, just decrement and return". Halo 2 hits this on
     * every file/event/thread handle deref pair. Decline when refcount
     * reaches 1 so the kernel runs the proper destructor chain.
     */
    { 250, "ObfDereferenceObject",      hle_obf_dereference_object,    &g_gate_kf,
        SIG(SIG_OBF_DEREFERENCE_OBJECT) },
    /*
     * KeInsertQueueDpc: stdcall(3). Audio mixer + render thread fire
     * DPCs per frame; this gets hot during gameplay. Fast path covers
     * "DPC already queued" + "linkage-only" cases; trigger path (first
     * DPC of a batch) declines so the kernel calls HalRequestSoftware-
     * Interrupt with its PIC/IMR manipulation. Telemetry should show
     * hits >> declines once gameplay DPC traffic kicks in.
     */
    { 119, "KeInsertQueueDpc",          hle_ke_insert_queue_dpc,       &g_gate_kf,
        SIG(SIG_KE_INSERT_QUEUE_DPC) },
    /*
     * KeRemoveQueueDpc: stdcall(1). Symmetric with KeInsertQueueDpc —
     * unlinks DPC from the queue if currently queued, clears Inserted,
     * returns BOOLEAN previous-Inserted-state. Pure pointer surgery; no
     * IRQL state needed (caller's responsibility).
     */
    { 137, "KeRemoveQueueDpc",          hle_ke_remove_queue_dpc,       &g_gate_kf,
        SIG(SIG_KE_REMOVE_QUEUE_DPC) },
    /*
     * RtlEqualString: stdcall(3). Length-mismatch fast-path is just one
     * USHORT compare. ASCII-only case-insensitive supported; non-ASCII
     * declines so the kernel handles ANSI code page semantics.
     */
    { 279, "RtlEqualString",            hle_rtl_equal_string,          &g_gate_rtl,
        SIG(SIG_RTL_EQUAL_STRING) },
};

/*
 * Non-ordinal hot PCs. These are addresses INSIDE kernel functions —
 * usually inner-loop TB targets — that we want to hook for telemetry
 * or to inject host-side scheduling assists. They can't be resolved
 * via the EAT, so we verify them by hard-coded VA + prologue signature.
 *
 * Signature verification protects against drift across kernel revisions:
 * if the byte sequence at the VA doesn't match, the install is silently
 * skipped (no boot wedge).
 */
struct extra_hook {
    uint32_t va;
    const char *name;
    XboxHleHandler handler;
    bool *gate;
    const uint8_t *prologue;
    size_t prologue_len;
};

static const uint8_t SIG_KI_IDLE_LOOP_INNER[] = {
    /* post-sti TB start: nop; nop; cli; cmp ebp, [ebp]; jz +0xc */
    0x90, 0x90, 0xfa, 0x3b, 0x6d, 0x00, 0x74, 0x0c
};
static const uint8_t SIG_XC_SHA_TRANSFORM[] = {
    /* push esi; push edi; push ebx; push ebp; mov edx, [esp+0x18] */
    0x56, 0x57, 0x53, 0x55, 0x8b, 0x54, 0x24, 0x18
};

#define XC_SHA_TRANSFORM_VA 0x80031550u

static const struct extra_hook g_extra_hooks[] = {
    {
        KI_IDLE_LOOP_INNER_VA,
        "KiIdleLoop.inner_spin",
        hle_ki_idle_loop_spin,
        &g_gate_kf,
        SIG(SIG_KI_IDLE_LOOP_INNER),
    },
    {
        XC_SHA_TRANSFORM_VA,
        "XcShaTransform",
        hle_xc_sha_transform,
        &g_gate_kf,
        SIG(SIG_XC_SHA_TRANSFORM),
    },
};

/*
 * Verify the function at `va` matches the expected prologue. NULL/zero
 * signature means "probe mode": log the first 16 bytes so we can build
 * a real signature for a future build, then refuse to install (safe by
 * default). Real signatures cause install on byte-match, skip on miss.
 */
static bool prologue_matches(uint32_t va, const uint8_t *sig, size_t len,
                              const char *name)
{
    uint8_t buf[16];
    if (!g_read(va, buf, sizeof(buf))) return false;
    if (!sig || len == 0) {
        HLE_LOG("probe %s @ 0x%08x bytes=%02x%02x%02x%02x%02x%02x%02x%02x"
                "%02x%02x%02x%02x%02x%02x%02x%02x",
                name, va,
                buf[0], buf[1], buf[2], buf[3],
                buf[4], buf[5], buf[6], buf[7],
                buf[8], buf[9], buf[10], buf[11],
                buf[12], buf[13], buf[14], buf[15]);
        return false;
    }
    if (len > sizeof(buf)) len = sizeof(buf);
    return memcmp(buf, sig, len) == 0;
}

static const struct ord_entry *ord_lookup(uint16_t ord)
{
    for (size_t i = 0; i < ARRAY_SIZE(g_ordinal_table); i++) {
        if (g_ordinal_table[i].ordinal == ord) return &g_ordinal_table[i];
    }
    return NULL;
}

bool xbox_hle_resolve_kernel(CPUState *cs)
{
    if (g_hle_resolved) return true;
    if (!g_hle_enabled) return false;
    g_resolve_attempts++;

    /* Need the CPU stopped enough to read RAM via dma_memory_read.
     * Caller (cpu_exec_loop entry) already holds BQL so this is fine. */
    uint8_t hdr[1024];
    if (!g_read(XKRNL_IMAGE_BASE, hdr, sizeof(hdr))) {
        g_resolve_failures++;
        return false;
    }
    if (hdr[0] != 'M' || hdr[1] != 'Z') {
        /* Kernel not loaded yet — MCPX BIOS hasn't handed off. */
        g_resolve_failures++;
        return false;
    }
    uint32_t e_lfanew = ldl_le_p(hdr + 0x3C);
    if (e_lfanew + 24 + 96 + 8 >= sizeof(hdr)) {
        g_resolve_failures++;
        return false;
    }
    if (hdr[e_lfanew] != 'P' || hdr[e_lfanew + 1] != 'E') {
        g_resolve_failures++;
        return false;
    }

    /* COFF header at e_lfanew+4 (skip "PE\0\0"). Optional header
     * starts at e_lfanew + 24. DataDirectory[0] (Export) starts at
     * optional+96. */
    uint32_t opt_off = e_lfanew + 24;
    uint32_t exp_rva  = ldl_le_p(hdr + opt_off + 96);
    uint32_t exp_size = ldl_le_p(hdr + opt_off + 100);
    if (!exp_rva || !exp_size) {
        g_resolve_failures++;
        return false;
    }

    /* Read the Export Directory itself. */
    uint8_t exp[40];
    if (!g_read(XKRNL_IMAGE_BASE + exp_rva, exp, sizeof(exp))) {
        g_resolve_failures++;
        return false;
    }
    uint32_t ord_base   = ldl_le_p(exp + 16);
    uint32_t num_funcs  = ldl_le_p(exp + 20);
    uint32_t func_rva   = ldl_le_p(exp + 28);
    if (num_funcs > 512) {
        g_resolve_failures++;
        return false;
    }

    /* Walk AddressOfFunctions. For each ordinal we care about, register
     * the hook at (image_base + EAT[ord - base]). */
    bool any = false;
    for (size_t i = 0; i < ARRAY_SIZE(g_ordinal_table); i++) {
        const struct ord_entry *e = &g_ordinal_table[i];
        if (e->gate && !*e->gate) continue;
        if (e->ordinal < ord_base) continue;
        uint32_t idx = e->ordinal - ord_base;
        if (idx >= num_funcs) continue;

        uint32_t rva;
        if (!g_read32(XKRNL_IMAGE_BASE + func_rva + idx * 4, &rva)) continue;
        if (!rva) continue;
        uint32_t va = XKRNL_IMAGE_BASE + rva;
        /* Sanity: addresses must land in the kernel image. */
        if (va < XKRNL_IMAGE_BASE || va > XKRNL_IMAGE_BASE + (4u << 20)) {
            continue;
        }
        /*
         * Prologue signature check + probe logging. The check logs the
         * actual prologue bytes when signature is NULL (harvest mode),
         * so we can capture real bytes on a running kernel without
         * a separate Ghidra import of xboxkrnl.exe.
         *
         * Without this, an ordinal-table drift across kernel revisions
         * silently installs handlers on the wrong function (2026-05-21
         * bugcheck 0x1e).
         */
        if (!prologue_matches(va, e->prologue, e->prologue_len, e->name)) {
            continue;
        }
        /* Handler-less probe entries (KfRaiseIrql/KfLowerIrql today) get
         * resolved + logged but never installed — they're here to help
         * harvest signatures and confirm the ordinal table. */
        if (!e->handler) {
            HLE_LOG("resolved %s @ 0x%08x (no handler — probe only)",
                    e->name, va);
            continue;
        }
        hle_install(e->name, va, e->handler);
        any = true;
    }

    /* Extra non-ordinal hot-PC hooks (in-function TB targets). Same
     * signature-verify discipline as the ord loop. */
    for (size_t i = 0; i < ARRAY_SIZE(g_extra_hooks); i++) {
        const struct extra_hook *h = &g_extra_hooks[i];
        if (h->gate && !*h->gate) continue;
        if (h->va < XKRNL_IMAGE_BASE ||
            h->va > XKRNL_IMAGE_BASE + (4u << 20)) {
            continue;
        }
        if (!prologue_matches(h->va, h->prologue, h->prologue_len, h->name)) {
            continue;
        }
        if (!h->handler) {
            HLE_LOG("resolved %s @ 0x%08x (no handler — probe only)",
                    h->name, h->va);
            continue;
        }
        hle_install(h->name, h->va, h->handler);
        any = true;
    }

    if (any) {
        g_hle_resolved = true;
        HLE_LOG("kernel resolved: image_base=0x%08x exp_rva=0x%x",
                XKRNL_IMAGE_BASE, exp_rva);
    } else {
        g_resolve_failures++;
    }
    return g_hle_resolved;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

void xbox_hle_init(void)
{
    static bool inited;
    if (inited) return;
    inited = true;

    const char *e = getenv("X1BOX_HLE");
    g_hle_enabled = (e && *e && *e != '0' && *e != 'n' && *e != 'N');
    if (!g_hle_enabled) {
        HLE_LOG("HLE off (X1BOX_HLE not set)");
        return;
    }

    const struct { const char *name; bool *gate; } gates[] = {
        { "X1BOX_HLE_RTL",   &g_gate_rtl   },
        { "X1BOX_HLE_KF",    &g_gate_kf    },
        { "X1BOX_HLE_YIELD", &g_gate_yield },
    };
    for (size_t i = 0; i < ARRAY_SIZE(gates); i++) {
        const char *v = getenv(gates[i].name);
        if (v && (*v == '0' || *v == 'n' || *v == 'N')) {
            *gates[i].gate = false;
        }
    }
    HLE_LOG("HLE on: rtl=%d kf=%d yield=%d",
            g_gate_rtl, g_gate_kf, g_gate_yield);
    sha1_init();
}

bool xbox_hle_is_enabled(void)
{
    return g_hle_enabled;
}

bool xbox_hle_check(CPUState *cs, uint32_t pc)
{
    static bool init_done;
    if (!init_done) {
        init_done = true;
        xbox_hle_init();
    }
    if (!g_hle_enabled) return false;
    /*
     * Fast-path range filter: every HLE hook today lives in the Xbox
     * kernel image at 0x80010000+. Game code (Halo 2 XBE) is at much
     * lower addresses (< 0x800000). Reject the >99% of TBs that
     * aren't in the kernel range with a single compare — eliminates
     * the hash-table lookup cost for them. At 12M TB/s this is the
     * dominant cost of the dispatcher hook.
     *
     * Update XKRNL_HOOK_RANGE_* if Tier 2 hooks ever land outside the
     * kernel image (e.g., D3D8 fast paths).
     */
#define XKRNL_HOOK_RANGE_LO 0x80010000u
#define XKRNL_HOOK_RANGE_HI 0x800b0000u
    if (pc < XKRNL_HOOK_RANGE_LO || pc >= XKRNL_HOOK_RANGE_HI) {
        return false;
    }
    if (!g_hle_resolved) {
        /* Throttle the PE walk — it touches guest memory which isn't
         * mapped early in boot. Try every ~256 calls until it lands. */
        static uint32_t throttle;
        if ((throttle++ & 0xFF) == 0) {
            xbox_hle_resolve_kernel(cs);
        }
        if (!g_hle_resolved) return false;
    }
    struct hle_entry *e = hle_lookup(pc);
    if (!e || !e->handler) {
        /* Profile unhooked kernel-range PCs so we know what to HLE next.
         * The PC-range filter at function top already gated this to the
         * kernel image, so we're only counting kernel entries. */
        unhooked_pc_count(pc);
        return false;
    }
    X86CPU *x86 = X86_CPU(cs);
    bool handled = e->handler(x86);
    if (handled) {
        e->hits++;
    } else {
        e->declines++;
    }
    return handled;
}

void xbox_hle_log_stats(void)
{
    if (!g_hle_enabled) return;
    if (!g_hle_resolved) {
        HLE_LOG("HLE not yet resolved (attempts=%" PRIu64
                " fails=%" PRIu64 ")",
                g_resolve_attempts, g_resolve_failures);
        return;
    }
    for (unsigned i = 0; i < HLE_TABLE_SIZE; i++) {
        if (g_hle_table[i].pc) {
            HLE_LOG("hle %s @ 0x%08x hits=%" PRIu64 " declines=%" PRIu64,
                    g_hle_table[i].name, g_hle_table[i].pc,
                    g_hle_table[i].hits, g_hle_table[i].declines);
        }
    }
    HLE_LOG("idle_loop idle=%" PRIu64 " dpc_pending=%" PRIu64
            " next_thread_pending=%" PRIu64,
            g_idle_loop_idle, g_idle_loop_dpc_pending,
            g_idle_loop_next_thread_pending);

    /* Top 10 unhooked kernel PCs — these are the next HLE candidates.
     * Simple selection sort; HLE_PROFILE_SLOTS=256 so this is cheap. */
    struct { uint32_t pc; uint64_t count; } top[10] = {0};
    for (unsigned i = 0; i < HLE_PROFILE_SLOTS; i++) {
        if (!g_unhooked_pcs[i].count) continue;
        for (unsigned t = 0; t < 10; t++) {
            if (g_unhooked_pcs[i].count > top[t].count) {
                /* Shift right + insert */
                for (unsigned s = 9; s > t; s--) top[s] = top[s-1];
                top[t].pc = g_unhooked_pcs[i].pc;
                top[t].count = g_unhooked_pcs[i].count;
                break;
            }
        }
    }
    for (unsigned t = 0; t < 10; t++) {
        if (top[t].count) {
            HLE_LOG("unhooked-pc[%u] 0x%08x hits=%" PRIu64,
                    t, top[t].pc, top[t].count);
        }
    }
}
