# Halo 2 x86 → AArch64 Optimization Inventory

Generated: 2026-05-29 via Ghidra static analysis (13,783 functions, 1,116,947 total instructions)

---

## Instruction Frequency Baseline

| Category            | Count    | % of total |
|---------------------|----------|------------|
| Total instructions  | 1,116,947| —          |
| SSE scalar (MOVSS/MULSS/ADDSS/SUBSS/DIVSS/…) | 83,951 | 7.5% |
| SSE packed (MOVAPS/MULPS/ADDPS/SHUFPS/…)     | 35,347 | 3.2% |
| MMX                 | 1,087    | 0.1%       |
| x87 FPU             | 39,487   | 3.5%       |
| **SIMD total**      | **120,385** | **10.8%** |

### Top mnemonics (selected)

```
MOV     319376   28.6%
PUSH    107868    9.7%
CMP      49145    4.4%
CALL     48758    4.4%
MOVSS    44677    4.0%   ← scalar SSE float move (biggest SIMD cost)
MOVAPS   18781    1.7%   ← packed float move
MULSS    16200    1.5%   ← scalar float multiply
FLD      11657    1.0%   ← x87 load
ADDSS     8709    0.8%
FSTP      8396    0.8%   ← x87 store
SUBSS     6603    0.6%
COMISS    5904    0.5%
FMUL      5130    0.5%   ← x87 multiply
SHUFPS    3252    0.3%   ← SSE shuffle (vector swizzle/broadcast)
MULPS     2965    0.3%   ← packed float multiply
ADDPS     1941    0.2%
CVTSI2SS  1588    0.1%   ← int→float conversion
SUBPS     1308    0.1%
DIVSS      995    0.1%
RSQRTSS    118           ← fast reciprocal sqrt
```

### x87 transcendentals breakdown

| Op        | Count | Purpose              |
|-----------|-------|----------------------|
| FSQRT     | 545   | sqrt                 |
| FCOS      | 260   | cosine               |
| FSIN      | 235   | sine                 |
| FPATAN    | 80    | atan2(y,x) from stack|
| FYL2X     | 52    | y·log₂(x) → log     |
| F2XM1     | 24    | 2^x-1 → exp         |
| FSCALE    | 28    | scale by power of 2  |
| FPTAN     | 19    | tan                  |
| FSINCOS   | 1     | sin+cos simultaneous |

---

## Identified Functions

### Confirmed by decompile

| Address     | Name / Role                              | Key evidence |
|-------------|------------------------------------------|--------------|
| 0x0031c940  | **Havok physics constraint solver**      | `hkKeycode_validate()` call; iterates constraint pairs; 1037 packed + 439 x87 in 2208 instrs |
| 0x00360710  | **Radix-4 FFT (audio DSP)**              | Butterfly pattern, twiddle table, `param_2 >> 4/6/8/10` recursive sizing; 593 packed SSE in 784 |
| 0x00361e30  | FFT variant A (same structure)           | 593 packed SSE in 784 |
| 0x00363450  | FFT variant B                            | 590 packed SSE in 825 |
| 0x00364ae0  | FFT variant B'                           | 590 packed SSE in 825 |
| 0x0035fba0  | FFT variant C                            | 588 packed SSE in 766 |
| 0x003612c0  | FFT variant C'                           | 588 packed SSE in 766 |
| 0x003701b0  | **32-pt FFT butterfly stage**            | Reads/writes EAX[0..0x1f] via x87; butterfly tree structure; 98.7% x87 (514/521) |
| 0x0036fc20  | FFT stage (small, 99.3% x87)             | 432 x87 in 435 |
| 0x0036ea60  | FFT stage (85% x87)                      | 578 x87 in 680 |
| 0x0036e200  | FFT stage (84.3% x87)                    | 558 x87 in 662 |
| 0x00143600  | **Rigid body inertia tensor builder**    | 9 quaternion inputs → higher-order rotation matrices; 1624 scalar + 80 packed in 1768 (96.4%) |
| 0x0023e460  | **Vertex shader sin/cos table builder**  | Calls `D3D8LTCG::D3DDevice_SetVertexShaderConstantNotInlineFast(0x58)`; 32 FSIN/FCOS calls in one function |
| 0x00030bf0  | `vector3d_normalize`                     | `1/sqrt(x²+y²+z²) * (x,y,z)` via FSQRT; EAX convention |
| 0x0002b400  | `vector2d_normalize`                     | Same pattern for 2D |
| 0x002db2b0  | `hkConvexSweep_castRay`                  | Havok sweep test; RDTSC + SHUFPS heavy |
| 0x002f7c40  | `hkAgent_multiSphereTri`                 | Havok sphere-triangle collision |
| 0x002eb670  | `hkMopp_query`                           | Havok MOPP (optimized BSP) query |
| 0x0030fcb0  | `hkWorld_simulate`                       | Havok world simulation step; 5 RDTSC |
| 0x0030fef0  | `hkWorld_simulateAndIntegrate`           | Havok full integrate; 6 RDTSC |
| 0x002fa2f0  | `hkConvexWelder_run`                     | Havok mesh welding |
| 0x002fbb70  | `TthkShapeCollection_castRay`            | Havok shape cast |
| 0x00331d2b  | `QueryPerformanceCounter`                | RDTSC wrapper (7 instrs) |

---

## Optimization Opportunities

### Priority 1 — x87 Transcendental → Native HLE (highest leverage per function)

Each x87 transcendental goes through the full emulator x87 stack simulation.
On Cortex-A720, native `sinf/cosf/sqrtf/atan2f` are 4–20× cheaper than
the emulated FSIN/FCOS/FSQRT/FPATAN path.

#### 1a. FSQRT (545 sites across 250+ functions)

- AArch64 direct: `fsqrt s0, s0` (1 instruction, 11 cycles on A720)
- Faster alternative: `frsqrte s0, s0` + `frsqrts` refinement (12-bit → ~23-bit precision, 6 cycles)
- **Primary HLE targets already known**: `vector3d_normalize@0x00030bf0`, `vector2d_normalize@0x0002b400`
- Also `_CI` math dispatch stub (planned transcendental HLE expansion)
- Havok physics (0x002fxxxx family): FUN_002f3030 (191 x87), FUN_002f28f0 (184 x87), FUN_002f4440/002f4f00/002f5960/002f63d0/002f6db0 (155-158 x87 each)

#### 1b. FSIN / FCOS (495 sites)

- AArch64: call Android libm `sinf`/`cosf` (neon-optimized polynomial, ~15 cycles)
- **Hottest single instance**: `FUN_0023e460@0x0023e460`
  - Calls 16× FSIN + 16× FCOS to populate a vertex shader constant table
  - Called every frame (guarded by `DAT_00485af0` dirty flag)
  - Entire function is a candidate for full HLE — detect by `D3DDevice_SetVertexShaderConstantNotInlineFast(0x58)` call pattern
  - XPack `pattern_bytes` on the function prologue + replacement trampoline to a native sin/cos loop
- Also `FUN_00039e50@0x00039e50` (4 FSIN/FCOS in 338 total, 94.7% scalar SSE); called from particle/animation code

#### 1c. FPATAN / atan2 (80 sites)

- Pattern: `FLD y; FLD x; FPATAN` → `atan2f(y, x)` on AArch64
- `FUN_0027aa0@0x00027aa0`: 5 transcendentals (FCOS, FPTAN, FPATAN, FSIN) in 653 total (39.8%)
- `FUN_001fe620@0x001fe620`: 11 transcendentals (all types) in 467 (25.1% SIMD)
- `FUN_0034acA0@0x0034aca0`: 2 FPATAN in 183 (48.6% x87) — very high FPATAN density

#### 1d. FYL2X + FLDLN2 → logf (52 FYL2X sites)

- Pattern: `FLDLN2; FLD x; FYL2X` = `ln(x)` → `logf(x)` on AArch64
- Key functions: `FUN_0033e9a1@0x0033e9a1` (343 x87 in 633, uses FYL2X repeatedly)

#### 1e. FLDL2E + F2XM1 + FSCALE → expf (24 F2XM1 sites)

- Pattern: `FLDL2E; FMUL; F2XM1; FLD1; FADD; FSCALE` = `exp(x)` → `expf(x)`
- Key functions: `FUN_0018ae80@0x0018ae80` (17 transcendentals: FLDLG2, FYL2X, FLDL2E, F2XM1, FSCALE in 928 total)
- `FUN_0021fa80@0x0021fa80`: 6 transcendentals (all log/exp types) in 563

---

### Priority 2 — SSE Scalar Inline Gaps

SSE scalar inline already landed for MULSS/ADDSS/SUBSS. Check remaining gaps:

#### 2a. DIVSS (995 occurrences)

- AArch64: `fdiv s0, s0, s1` (9–17 cycles, but still cheaper than helper dispatch)
- SSE `DIVSS` is NOT in the current inline set (was excluded due to precision concerns)
- At 995 sites, inlining saves ~995 × (helper_call_overhead − fdiv_latency) per hot path
- Gate behind `X1BOX_SSE_DIVSS_INLINE=1` env var

#### 2b. RSQRTSS (118 occurrences)

- SSE `RSQRTSS` = 12-bit reciprocal sqrt approximation
- AArch64 equivalent: `frsqrte s0, s1` (8-bit) + one `frsqrts` refinement → ~23-bit
- Used in Havok physics and specular highlight calculations
- The Newton-Raphson refinement is optional for game code (12-bit is often sufficient)
- Pattern: `RSQRTSS; MULSS; MULSS; ADDSS` = standard refinement loop; can emit entire sequence

#### 2c. CVTSI2SS / CVTTSS2SI (1,835 combined occurrences)

- `CVTSI2SS`: int → float = AArch64 `scvtf s0, w0`
- `CVTTSS2SI`: float → int (truncate) = AArch64 `fcvtzs w0, s0`
- Verify these are already in the SSE scalar inline path — if not, high-value gap

#### 2d. COMISS / UCOMISS (6,661 combined)

- SSE float compare → flags; used with JZ/JNZ/JBE/etc.
- AArch64: `fcmp s0, s1` + branch
- These should be in the inline path; verify they aren't going through helper

---

### Priority 3 — NEON Width Expansion (scalar sequences → packed)

These are opportunities where x86 does scalar ops in a loop that NEON could do 4-wide.

#### 3a. FFT Butterfly Family (6 functions, ~590 packed SSE each)

**FUN_00360710 / 00361e30 / 00363450 / 00364ae0 / 0035fba0 / 003612c0**

These are radix-4 FFT butterflies (likely MCPX audio DSP or Bink audio).
The x86 SSE version already uses MOVAPS/ADDPS/SUBPS/MULPS (packed 4-wide).
The emulator executes these one op at a time through TCG helpers.

Key patterns in the emitted x86:
- `(float)((uint)(a - b) ^ SIGN_BIT)` — negation via XOR with sign bit mask (XORPS trick)
  → AArch64: `fneg.4s v0, v0` or `eor.16b v0, v0, sign_mask`
- Complex butterfly `(a·c − b·d, a·d + b·c)` with real/imag interleaved
  → AArch64: `fmul.4s` + `fmls.4s` + `fmla.4s` (fused multiply-accumulate)
- Four butterflies in parallel (4-wide MOVAPS/ADDPS/SUBPS)
  → Already 4-wide on x86; NEON `fadd.4s/fsub.4s` is a direct drop-in

These 6 functions together contribute ~3,500 packed SSE instructions. Under Cranelift tier-2,
the packed ops currently fall through to helpers. A NEON-native xpack replacement for the FFT
entry point would run the entire FFT in native NEON without any TCG involvement.

**Approach**: xpack `pattern_bytes` on the FFT function prologue → trampoline to native NEON FFT
(e.g. NE10 or manually written AArch64 asm butterfly).

#### 3b. FUN_003701b0 — 32-pt FFT Butterfly (98.7% x87)

This function reads 32 floats from a struct (via EAX) and writes 32 floats back.
It uses x87 exclusively, running a 32-point butterfly tree with twiddle multipliers
passed in `param_1+4..+0x24`.

Under x87 emulation this is extremely expensive (514 FLD/FMUL/FADD/FSTP in 521 instrs).
A native NEON implementation of the same 32-pt butterfly:
- Load 8× float32x4 = 32 floats via `vld1.4s` (8 instructions)
- Compute 4-level butterfly: 4 stages × 8 FADD.4S + 8 FSUB.4S = 64 NEON instructions
- Multiply twiddle: 8 FMUL.4S per butterfly group
- Store: 8× `vst1.4s`
- Total: ~100 AArch64 instructions vs 514 x87 ops through the emulator

**AArch64 reduction estimate**: ~5× speedup on this function in isolation.

#### 3c. SHUFPS / Broadcast Patterns (3,252 SHUFPS total)

Top SHUFPS users: FUN_0031c940 (136), FUN_0031f620 (85), FUN_003048a0 (81)

Common patterns:
- `SHUFPS xmm0, xmm0, 0x00` = broadcast lane 0 to all → `DUP.4S v0, v0.s[0]`
- `SHUFPS xmm0, xmm1, 0x44` = low halves → `ZIP1.4S v0, v0, v1`
- `SHUFPS xmm0, xmm1, 0xE4` = identity-ish → can be recognized and eliminated
- `UNPCKLPS / UNPCKHPS` → `ZIP1.4S` / `ZIP2.4S`
- `MOVLHPS / MOVHLPS` → `INS v0.d[1], v1.d[0]` / `INS v0.d[0], v1.d[1]`

In Cranelift tier-2, if SHUFPS is emitted as a helper call (likely), each occurrence costs
the full helper dispatch. Adding shuffle as an inline case in the Cranelift lowering would
save ~3,252 × helper_call_overhead per hot-loop pass.

#### 3d. MOVLPS / MOVHPS Patterns (492 combined)

Used for complex-number arithmetic (load/store high/low 64-bit lanes):
- `MOVLPS xmm0, [mem]` → `LDR d0, [mem]` (load lower 64 bits)
- `MOVHPS xmm0, [mem]` → `LD1 {v0.2s}[2], [mem]` (insert upper 64 bits)
- `MOVLPS [mem], xmm0` → `STR d0, [mem]`

---

### Priority 4 — Havok Physics RDTSC Cluster

**100+ functions in 0x002exxxx–0x002fxxxx** use RDTSC for internal performance profiling.
Every RDTSC pair goes through `cpu_get_tsc()` → TB exit → re-entry.

Key clusters:
- `FUN_00223af0/00223b20/00223b60/00223c20` — 4 tiny functions (12–29 instrs), 1 RDTSC each = `QueryPerformanceCounter` wrappers
- `FUN_002e6e90/002e7030/002e71f0/…` — ~30 Havok profiling wrappers (90–165 instrs, 3 RDTSC each)
- `hkWorld_simulate@0x0030fcb0` — 5 RDTSC; `hkWorld_simulateAndIntegrate@0x0030fef0` — 6 RDTSC

**Approach**: HLE the `QueryPerformanceCounter@0x00331d2b` wrapper (7 instrs, 1 RDTSC) to return a
monotonic counter from `clock_gettime(CLOCK_MONOTONIC)`. This would eliminate RDTSC cost in all
100+ Havok profiling sites that call through it.

The tiny 12-instr `FUN_00223af0` family appears to be inlined wrappers that read RDTSC directly —
these would need pattern_bytes xpack patches to replace the RDTSC pair with a zero or monotonic value.

---

### Priority 5 — MMX / Integer SIMD (smaller, but clean mapping)

MMX total: 1,087 instructions (mostly PCM audio processing)

| Pattern                | x86 MMX | AArch64 NEON |
|------------------------|---------|--------------|
| PUNPCKLWD (73 sites)   | interleave 16-bit words | `ZIP1.8H` |
| PADDD (39 sites)        | add 32-bit lanes | `ADD.4S` |
| PADDW (80 sites)        | add 16-bit lanes | `ADD.8H` |
| PSLLW/PSRLD/PSRAD       | shift | `SHL/USHR/SSHR` |
| PMULHW (16 sites)       | high 16-bit multiply | `SQDMULH.8H` |
| PUNPCKHWD / PACKUSWB    | pack/unpack | `ZIP2/UZP1` |

The MMX usage is concentrated in audio resampling / PCM mixing code.
EMMS (38 occurrences) marks the end of MMX regions — these are discrete, bounded functions.

---

## Functions NOT Worth Targeting

| Address    | Why |
|------------|-----|
| 0x00143600 | Rigid body inertia tensor: 74 unique float variables, 3 nested code paths. Structure is too complex for xpack replacement. Math xpack (global fp64 mode) is the correct lever. |
| 0x00123110 | Busy-wait intro state machine (MMX XOR-fold, from memory). XPACK state-counter advance is the right lever per prior analysis. |
| 0x0031c940 | Havok constraint solver: correct fix is Havok's own SSE path becoming effective via NEON; too stateful for xpack. |

---

## AArch64 Instruction Mapping Reference

| x86 SSE/x87                | AArch64 NEON / scalar FP        | Notes |
|----------------------------|----------------------------------|-------|
| `MOVAPS xmm, [mem]`        | `LDR q0, [mem]`                  | 16-byte aligned |
| `MOVUPS xmm, [mem]`        | `LDR q0, [mem]` (unaligned ok)  | AArch64 LDR is always unaligned-safe |
| `ADDPS xmm0, xmm1`        | `FADD v0.4s, v0.4s, v1.4s`      | |
| `SUBPS xmm0, xmm1`        | `FSUB v0.4s, v0.4s, v1.4s`      | |
| `MULPS xmm0, xmm1`        | `FMUL v0.4s, v0.4s, v1.4s`      | |
| `DIVPS xmm0, xmm1`        | `FDIV v0.4s, v0.4s, v1.4s`      | |
| `XORPS xmm0, [sign_mask]` | `FNEG v0.4s, v0.4s`             | When mask = 0x80000000×4 |
| `ANDPS xmm0, [abs_mask]`  | `FABS v0.4s, v0.4s`             | When mask = 0x7FFFFFFF×4 |
| `SHUFPS xmm0,xmm0, 0x00` | `DUP v0.4s, v0.s[0]`            | broadcast lane 0 |
| `SHUFPS xmm0,xmm1, 0x44` | `ZIP1 v0.4s, v0.4s, v1.4s`      | low 2 from each |
| `UNPCKLPS xmm0, xmm1`    | `ZIP1 v0.4s, v0.4s, v1.4s`      | interleave low |
| `UNPCKHPS xmm0, xmm1`    | `ZIP2 v0.4s, v0.4s, v1.4s`      | interleave high |
| `MOVLHPS xmm0, xmm1`     | `INS v0.d[1], v1.d[0]`           | copy low→high |
| `MOVHLPS xmm0, xmm1`     | `INS v0.d[0], v1.d[1]`           | copy high→low |
| `RSQRTPS xmm0, xmm1`     | `FRSQRTE v0.4s, v1.4s`           | 8-bit approx; add FRSQRTS for 23-bit |
| `RSQRTSS xmm0, xmm1`     | `FRSQRTE s0, s1`                  | scalar |
| `SQRTSS xmm0, xmm1`      | `FSQRT s0, s1`                    | full precision |
| `CVTSI2SS xmm0, r32`     | `SCVTF s0, w0`                    | |
| `CVTTSS2SI r32, xmm0`    | `FCVTZS w0, s0`                   | truncate toward zero |
| `MOVMSKPS r32, xmm0`     | `UMOV w0, v0.s[3]; …` + shifts   | slow; 4 extracts + shift/or |
| `COMISS / UCOMISS`        | `FCMP s0, s1`                     | |
| `FSQRT`                   | `FSQRT s0, s0`                    | x87 stack → scalar reg |
| `FSIN / FCOS`             | `BL sinf` / `BL cosf`             | libm PLT |
| `FPATAN`                  | `BL atan2f`                       | pop y, pop x, push result |
| `FYL2X`                   | `BL logf` + `FMUL`                | y×log₂(x) = y×ln(x)/ln(2) |
| `FLDL2E + F2XM1 + FSCALE`| `BL expf`                          | |
| `PUNPCKLWD mm, mm`        | `ZIP1 v0.8h, v0.8h, v1.8h`       | |
| `PADDD mm, mm`            | `ADD v0.4s, v0.4s, v1.4s`        | (use SIMD not MMX regs) |
| `PMULHW mm, mm`           | `SQDMULH v0.8h, v0.8h, v1.8h`    | signed 16-bit high multiply |

---

## Implementation Paths

### Path A: xpack pattern_bytes / HLE (no emulator source changes)

Best for: complete function replacements, vertex shader setup, normalize functions

1. `vector3d_normalize@0x00030bf0` — signature scan → trampoline to AArch64 `fsqrt`-based normalize
2. `vector2d_normalize@0x0002b400` — same
3. `FUN_0023e460@0x0023e460` — vertex shader sin/cos table; replace 32× FSIN/FCOS calls with
   a native loop using `sinf/cosf`; gate on `DAT_00485af0` flag (already checked in the function)
4. `QueryPerformanceCounter@0x00331d2b` — HLE to return `clock_gettime` value
5. `FUN_00223af0` family (RDTSC-only tiny wrappers) — xpack `bytes` to replace RDTSC with
   `XOR EAX,EAX; XOR EDX,EDX; RET` (return 0) or a static increment

### Path B: TCG inline translator additions (emulator source changes)

Best for: instruction-level patterns appearing thousands of times

1. **FSQRT inline** — emit `FSQRT s0, s0` (or FRSQRTE+FRSQRTS) instead of helper call
2. **DIVSS inline** — emit `FDIV s0, s0, s1` behind `X1BOX_SSE_DIVSS_INLINE=1`
3. **RSQRTSS inline** — emit `FRSQRTE + FRSQRTS` refinement
4. **FSIN/FCOS inline** — emit `BL sinf / cosf` (PLT call, but avoids x87 emulator path)
5. **FPATAN inline** — detect `FLD; FLD; FPATAN` sequence, emit `BL atan2f`
6. **SHUFPS → DUP / ZIP** — add immediate-indexed lowering in Cranelift `fpvec.rs`

### Path C: Math xpack (SHA1-gated fp64 native mode)

Already in progress — enables Hard FPU path. Will cover all FSQRT, FMUL, FADD x87 ops.
Blocked on physics divergence root-cause (Hard FPU breaks Halo 2 physics).

---

## Coverage Summary

| Opportunity                     | Sites | Path | Est. difficulty |
|---------------------------------|-------|------|-----------------|
| FSQRT → native                  | 545   | B/A  | Low             |
| FSIN/FCOS → sinf/cosf           | 495   | B/A  | Low             |
| DIVSS inline                    | 995   | B    | Low             |
| RSQRTSS inline                  | 118   | B    | Low             |
| CVTSI2SS / CVTTSS2SI (verify)   | 1835  | B    | Low (verify only)|
| FPATAN → atan2f                 | 80    | B    | Medium          |
| FYL2X → logf                    | 52    | B    | Medium          |
| F2XM1+FSCALE → expf             | 24    | B    | Medium          |
| SHUFPS inline (Cranelift)       | 3252  | B    | Medium          |
| FFT function family (6 fns)     | ~3500 | A    | High            |
| 32-pt FFT butterfly (003701b0)  | 514   | A    | High            |
| vector3d_normalize HLE          | 38    | A    | Low             |
| Vertex shader sin/cos table HLE | 32    | A    | Low-Medium      |
| RDTSC/QPC HLE (Havok profiling) | 100+  | A    | Low             |
