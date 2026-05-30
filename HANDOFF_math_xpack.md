# HANDOFF — Math Xpack (native aarch64 fp64 x87 substitution)

Status: **design + evidence complete; implementation starting with the FISTP RC fix.**
Date: 2026-05-29. Companion to `HANDOFF_x87_lib_bisection.md`.

---

## 0. TL;DR

- x87 80-bit emulation is the perf bottleneck. Halo 2 & Serious Sam II use x87 as an
  **effectively-fp64 machine** (control word `0x027F` = PC=53/double; they never spill 80-bit to
  memory). So routing x87 through native aarch64 `double` is a sound substitution.
- That native path **already exists in the tree** (`fp_jit` → the `_f32/_f64` TCG-FP generators in
  `target/i386/ops_fpu.h`, plus the Cranelift tier-2 inline path). **It is NOT a new backend.**
- BUT on the test device (Zenfone 10) the app's **"Hard FPU" toggle is OFF**, so x87 currently runs
  on **softfloat** (slow, correct). The math xpack is therefore a real **opt-in to enable the native
  path per validated title** — not a relabel of something already on.
- The native path has a **confirmed, device-verified bug**: `FISTP`/`FIST` lower to aarch64
  `FCVTZS` (round-toward-zero), ignoring the guest rounding mode → off-by-one under the games'
  default round-to-nearest, plus `NaN→0` instead of x87's `0x80000000`. **Fixing this is a hard
  prerequisite** before the native path can be enabled for any title that uses default-rounding FISTP
  (Halo 2 does — it never reprograms RC).

---

## 1. Evidence (all verified this session)

### 1.1 Ghidra — precision & 80-bit traffic (Halo 2 `4D530064` + Serious Sam II)
Method: `ghidra-mcp run_script_inline` walking `listing.getInstructions(true)`, classifying each x87
op by primary opcode (`0xD8–0xDF`) + ModRM.reg width.

| metric | Halo 2 | Serious Sam II |
|---|---|---|
| FLD m32 / m64 / **m80** | 9698 / 394 / **14** | 48561 / 323 / **18** |
| FST·FSTP m32 / m64 / **m80** | 6246 / 249 / **3** | 31806 / 479 / **4** |
| BCD-80 (FBLD/FBSTP) | 0 | 0 |
| `FLDCW` (all) | 10 (all CRT) | 28 |
| `FNINIT`/`FINIT` | 0 | 0 |

- **Steady-state control word = `0x027F`** (RN, all exceptions masked, **PC=53/double**). Proven by
  CRT math helpers doing `CMP word ptr [...],0x27F; JZ skip; FLDCW` (skip restore when caller CW is
  already the default): Halo2 `@0x327f7e`,`@0x32802a`; SS2 `@0x41f65e`,`@0x41f70a`.
- **80-bit never reaches memory in game code** — the handful of REAL10 ops are all CRT/x87-lib
  internal. Data is overwhelmingly fp32, thin fp64 minority. The 80-bit register width is a transient.
- **RC used, PC not.** Halo 2 has **zero game-code `FLDCW`**. SS2's engine (`0x76xxxx`) reprograms
  only the **rounding-control** bits (`FNSTCW; OR EAX,0xC00; FLDCW`, the `ftol` truncate idiom) +
  a precomputed truncate-CW global; it never touches PC. ⇒ the substitution **must honor the rounding
  mode** but **may hard-assume PC=53**.

### 1.2 Ghidra — C4 scan: does game code branch on FP exception flags? (NO)
Refined `FNSTSW`→`AND/TEST` classifier separating `AH` (condition codes C0–C3, high byte = FP
*compare* branches, benign) from `AL` (exception flags, low byte = the real risk):

| | total `FNSTSW` | benign `AH` compares | exception-byte (`AL`) tests | PE-bit (0x20) tests |
|---|---:|---:|---:|---:|
| Halo 2 | 1331 | 1318 | 5 | 3 |
| Serious Sam II | 3441 | 3423 | 5 | 3 |

**Every** exception-byte test (including all `AND AX,0x20; JZ` precision-exception checks) is inside
the CRT math library (`0x32xxxx`/`0x37xxxx`/`0x41xxxx`) — transcendental-helper epilogues. **Zero
game-engine FP-exception branches.** Since transcendentals stay on the libm/XEFU path, the native
path's lack of exception-flag modeling is safe for game logic. **C4 retired.**

### 1.3 Device — FISTP bug A/B (Zenfone 10, live gdbstub injection)
Method: inject `FNINIT; fld qword[eax]; fistp dword[eax+8]` at EIP, scratch in `.data` (`0x500000`),
`IF` cleared so no interrupt clobbers the 3-instruction window, run through the live JIT to a bp,
read the stored int. (FNINIT pins RC=nearest independent of guest state.)

| input | `fp_jit` OFF (softfloat) | `fp_jit` ON (native) | correct x87 @ RC-nearest |
|---:|:---:|:---:|:---:|
| 2.0 | 2 | 2 | 2 |
| **2.7** | 3 ✓ | **2** ❌ | 3 |
| **3.5** | 4 ✓ | **3** ❌ | 4 |
| -2.7 | -3 ✓ | (−2, implied) | -3 |
| **NaN** | 0x80000000 ✓ | **0** ❌ | 0x80000000 |

`fp_jit` state read from `show_config` (`[perf] fp_jit`); driven by SharedPref `setting_hard_fpu`
("Hard FPU" toggle) → `xemu.toml`. The softfloat `NaN→0x80000000` vs native `NaN→0` cleanly proves
which path executed. **Bug confirmed on the native path.** Device was restored to as-found (Hard FPU off).

---

## 2. What exists vs. the gap

| Capability | Status | Location (verify line drift) |
|---|---|---|
| `fp_jit` flag + per-op codegen selection (no per-instruction branch) | exists; **toggle, off on test device** | `xemu_settings_android.cc:109,546-547`; pref `setting_hard_fpu` |
| TCG-native FP arith FADD/FMUL/FSUB/FDIV/FSQRT on `double` stack | exists, live when `fp_jit` on | `target/i386/ops_fpu.h:213-276` |
| native fp64 register stack `env->fpregs[i].native_d` / `ft0_native` | exists | `ops_fpu.h:131-148` |
| Cranelift tier-2 inline F64 path | exists | `rust/cranelift-tcg/src/x87.rs` |
| **arith RC → host FPCR** via `gen_flcr` (masks guest CW `0xc00`) | exists | `translate.c:1726-1739`; called `ops_fpu.h:57,76,90,110`; Cranelift `cranelift-bridge.c:574` |
| FPCR FZ/FZ16/DN=0 forced at vCPU thread entry | exists | `accel/tcg/tcg-accel-ops-mttcg.c:97-144` |
| transcendentals via libm fp64 (lib bypassed under hard path) | exists | `fpu_helper.c`; `xemu_use_x87_lib_op` sites `#if !defined(USE_HARD_FPU)` |
| **FIST/FISTP honors RC** | **GAP — broken (FCVTZS), device-confirmed** | `ops_fpu.h:328-336` → `tcg-target.c.inc:682-685,3490-3513` |
| `-ffp-contract=off` on FP TUs (C) + no `Opcode::Fma` (Cranelift) | **GAP** | CMake/meson; `feedback_lto_tls_link_flag` (must be in `target_link_options` too) |
| per-title validation gate + self-test + per-op bisection mask | **GAP** | — |
| xpack kind that flips runtime CPU/FPU state | **GAP — xpacks are memory-patch-only** | `hw/xbox/xpacks.c` kinds: bytes/pattern_bytes/cave/shader |

**This xpack adds:** (1) the FIST RC fix; (2) no-contract build + no-FTZ guarantee + an init
self-test; (3) a SHA1-gated `fpu_mode` xpack kind as the per-title "validated" marker; (4) a per-op
`X1BOX_X87_FP64_MASK` bisection + kill switch.

---

## 3. The FISTP RC fix (implementation target — part B)

Root cause: `gen_fistl_ST0`/`gen_fistll_ST0` (`ops_fpu.h:328-336`) emit `tcg_gen_cvtNf_iM` with **no
rounding mode**; aarch64 lowers all `cvt*f_i*` to `FCVTZS` (`tcg-target.c.inc:682-685, 3490-3513`).
`FISTTP` (SSE3) is *supposed* to truncate, so `FCVTZS` is correct there — only plain `FIST`/`FISTP`
must honor RC.

**Chosen approach — round-to-integral-then-truncate (FPCR-driven, dynamic-RC-correct):**
`gen_flcr` already mirrors the guest RC bits into the host FPCR at runtime. So FISTP should:
`FRINTI d, d` (round to integral float using FPCR rounding mode — tracks dynamic RC like SS2 ftol for
free, no per-RC codegen dispatch) then `FCVTZS w, d` (now exact, value is integral). FISTTP keeps
plain `FCVTZS`. This needs a TCG "round-to-integral, current mode" float op lowered to aarch64
`FRINTI` (and a portable softfloat fallback for the i386/host build).

Alternative considered: explicit `FCVTNS/FCVTMS/FCVTPS/FCVTZS` selected by guest RC. Rejected as
primary because RC is not a TB constant (SS2 changes it at runtime) → would need a runtime branch on
`fpuc & 0xC00`; the FRINTI approach gets dynamic-RC correctness from the FPCR `gen_flcr` already sets.

**Out of scope of the rounding fix (track separately, flagged by self-test):** NaN/overflow result.
x87 (masked invalid) gives `0x80000000` for NaN and both-direction overflow; `FCVTZS` gives `0` for
NaN and saturates (`0x7FFFFFFF`/`0x80000000`). Games rarely FISTP out-of-range; match later if needed.

Mirror the fix in `rust/cranelift-tcg/src/x87.rs` (the tier-2 inline path).

---

## 4. Xpack format extension (design)

New `[[patch]] kind = "fpu_mode"`, riding the existing per-pack SHA1 gate (`hw/xbox/xpacks.c`
`sha1_matches` over the 256-byte xbe digsig). Concrete Halo 2 pack:

```toml
schema = 1
name   = "Halo 2 — native fp64 x87 (PC53 fast math)"
[match]
title_id      = 0x4D530064
xbe_code_sha1 = "<40 hex — build-locked; only the image proven PC=53>"
[[patch]]
kind = "fpu_mode"
mode = "native_fp64_pc53"
```

- Parser: `SEC_PATCH_FPU` section + scalar `x1box_fpu_mode fpu_mode` on `XPack`; `"fpu_mode"` kind in
  the kind dispatch; unknown `mode` → `LOG_GUEST_ERROR`, drop directive.
- Apply: inside `xpacks_apply_for_xbe` **within** the `sha1_matches` block; **refuse if no
  `xbe_code_sha1`** (we only proved PC=53 for specific builds). Layering: `xpacks.h` declares
  `xpacks_cpu_fpu_directive()`, implemented in `translate.c` (owns the flag); it runs the self-test,
  on pass sets the verified marker + enables the FIST-RC fix + `xemu_set_x87_lib(false)` +
  `tb_flush` (the apply hook at `xemu-xbe.c` runs *after* the loader built TBs, so the flush is
  load-bearing) + asserts live CW==`0x027F`; on fail logs & leaves the safe path.
- UX: disabled-by-default in `enabled.txt`; user opts in via Manage Mods; flip default-on per title
  only after the ship-gate passes for that SHA1.

---

## 5. Accuracy contract & self-test

Divergence native-fp64 vs x87@PC53 (oracle = softfloat `floatx80_precision_d`):
mantissa 53 vs 53 → identical; **no double-rounding** (PC already 53); exponent 15- vs 11-bit →
bounded, can't arise from fp32×fp32, shadow-canary it; **FMA contraction / FTZ / NaN** → guarded by
`-ffp-contract=off` + FZ=0 + NaN-by-class; **FIST RC** → §3 fix; transcendentals → stay on libm.

Init self-test (mandatory; `feedback_hle_crypto_self_test`), run on the vCPU thread **with `fp_jit`
enabled**, through both a tier-1.5 and a tier-2 TB, vs the softfloat-PC53 oracle: RNE-tie arith,
FST→m32 single-round, correctly-rounded FSQRT (proves not `FRSQRTE`), an FMA-trap triple, a subnormal
product (detects FTZ), `0/0`→NaN-class & `1/0`→∞, **FIST under all 4 RC modes**, fcom ordering incl.
NaN. Mismatch → clear the verified marker, fall back to softfloat, log the failing vector via in-app
Debug Logging (`feedback_vk_check_logging`; stderr drops on Android). The on-device gdb injection
harness from §1.3 is exactly this self-test in miniature and can be promoted.

---

## 6. Validation & rollout

- Scenes: **Halo 2** wall-slide + Warthog (the `project_sse_neon_physics_drift` failure mode);
  **SS2** ftol/RC canary (HUD/score off-by-one). FPS via guest FLIP_STALL normalized to 30 Hz
  (`feedback_user_visible_fps_definition`, `project_halo2_native_framerate`); `feedback_never_revert`.
- Bisection: **`X1BOX_X87_FP64_MASK`** (env `strtoul` base=0 + Android prop `debug.x1box.x87fp64mask`,
  same parse site as `X1BOX_X87_LIB_MASK`); transcendental bits separate + unsettable. Kill switch:
  `=0` (parity with `X1BOX_X87_INLINE=0`) + self-test auto-disable.
- Ladder per title: default safe → ship xpack disabled → user opt-in → default-on per SHA1 after gate.

---

## 7. Implementation order (dependency-sorted)

1. **(in progress) FISTP RC fix** — `ops_fpu.h` FIST generators + aarch64 `FRINTI` lowering (new TCG
   round-to-integral op or equivalent) + softfloat fallback; keep FISTTP truncating; mirror in `x87.rs`.
   Verify on device with the §1.3 harness (expect 2.7→3, 3.5→4 under `fp_jit` on).
2. Build flags: `-ffp-contract=off` + `#pragma STDC FP_CONTRACT OFF` on `fpu_helper*.c` /
   `x87_lib_shim.cpp`, in `c_args` **and** `target_link_options`; `debug_assert!` no-`Opcode::Fma` in `x87.rs`.
3. Self-test harness (§5) → clears verified marker on mismatch.
4. `X1BOX_X87_FP64_MASK` + transcendental-bit separation.
5. `fpu_mode` xpack kind (parser/struct/loader + `xpacks_cpu_fpu_directive` in `translate.c`).
6. `xpacks/SPEC.md` — document `kind="fpu_mode"`.
7. Ship the two SHA1-locked packs, run the ship-gate per title.

Full exhaustive design (workflow synthesis, incl. red-team risk register): workflow run
`wf_76a200b2-77d`, output at `tasks/wy2c9qjzg.output`.

### Key files
`target/i386/ops_fpu.h:213-336` · `target/i386/tcg/translate.c:1726-1739,4397` ·
`tcg/aarch64/tcg-target.c.inc:682-685,3490-3513` · `tcg/tcg-op-fp.c` · `rust/cranelift-tcg/src/x87.rs` ·
`accel/tcg/tcg-accel-ops-mttcg.c:97-144` · `accel/tcg/tb-maint.c` (Cranelift TB invalidation) ·
`hw/xbox/xpacks.c` + `xpacks.h` · `hw/xbox/xemu-xbe.c` (apply hook) · `xpacks/SPEC.md` ·
`android/app/src/main/cpp/xemu_settings_android.cc:109` (fp_jit default + env parse site).
