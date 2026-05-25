/*
 * Xbox kernel HLE (High-Level Emulation) hooks.
 *
 * Detects entry to specific Xbox kernel functions (resolved from the
 * loaded kernel's PE export table) and short-circuits them with host C
 * implementations. Cuts the guest x86 instruction count for hot kernel
 * calls — primarily Rtl memory copies, Kf spinlocks, and explicit-yield
 * primitives.
 *
 * Off by default; enable with X1BOX_HLE=1. Per-function gates exist for
 * fine-grained bisection (X1BOX_HLE_RTL=0 etc).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef HW_XBOX_HLE_H
#define HW_XBOX_HLE_H

#include "qemu/osdep.h"
#include "hw/core/cpu.h"

/*
 * One-time init: reads env vars, sets master enable, allocates the
 * PC->handler table. Safe to call multiple times (no-op after first).
 */
void xbox_hle_init(void);

/*
 * Returns true iff HLE is enabled at runtime. Cheap (single atomic
 * read); the dispatcher fast-path uses this to skip the PC check
 * entirely when HLE is off.
 */
bool xbox_hle_is_enabled(void);

/*
 * Dispatcher hot-path. Called from cpu_exec_loop with the current
 * guest PC. If `pc` matches a registered HLE entry, the corresponding
 * handler runs (mutates guest CPU state, sets EIP to the caller's
 * return address) and we return true. The caller MUST skip the TB
 * execution that would otherwise have happened.
 *
 * Returns false when:
 *   - HLE is disabled
 *   - `pc` is not in the hook table
 *   - The kernel has not been resolved yet
 */
bool xbox_hle_check(CPUState *cs, uint32_t pc);

/*
 * Lazy resolver. Reads the Xbox kernel PE export table at 0x80010000
 * once per session and installs hooks for all known ordinals. The
 * dispatcher fast-path nudges this on each cpu_exec_loop entry until
 * it succeeds; failure is silent (just no HLE).
 *
 * Returns true if hooks were installed (now or previously).
 */
bool xbox_hle_resolve_kernel(CPUState *cs);

/*
 * Telemetry. Reports per-handler hit counts so we can confirm which
 * HLE entries are actually getting exercised and which are dead code
 * paths. Mirrors the cranelift chain-stats pattern.
 */
void xbox_hle_log_stats(void);

/*
 * True when DSound HLE is taking over voice work — MCPX vp.c reads this
 * to short-circuit voice_process so we don't double-process audio that
 * the HLE has already routed to its own backend. Cheap (one branch on a
 * static bool, predicted-not-taken in normal flow).
 *
 * Set once by xbox_hle_init() from the X1BOX_HLE_DSOUND env var. Stays
 * stable for the session — flipping live would race with vp's frame
 * pump.
 */
bool xbox_hle_dsound_active(void);

/*
 * True when the per-voice voice_process bypass is opted in via
 * X1BOX_HLE_DSOUND_BYPASS=1. Separate gate from xbox_hle_dsound_active()
 * because the bypass needs a minimal voice state-machine (CBO advance +
 * voice_off at end-of-buffer) to keep MCPX IRQ generation flowing —
 * without that, guest threads HLT-wait on audio IRQs that never come.
 * vp.c reads this in voice_process to skip the expensive sample math.
 */
bool xbox_hle_dsound_bypass_active(void);

#endif /* HW_XBOX_HLE_H */
