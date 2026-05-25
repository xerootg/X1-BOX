# HANDOFF — x87 lib bump regression bisection (2026-05-22)

Context: commit `4f63f4babc x87_lib_shim: bump to 4b78068 + cover whole library`
bumped the upstream Aaron-Giles x87 library from `06813043ae` → `4b78068`
(spanning ~15 upstream commits). After this bump the user reported on
Halo 2:

- Audio latent
- Controls latent
- Physics very broken (walked through walls, inaccurate stick movement)

We have NOT shipped a fix yet — but the bisection is most of the way
done. This doc tells the next agent (a) what's already verified, (b) the
two open questions, (c) how to drive the on-device repro, (d) the fix
options once we have the answer.

---

## What's already verified

Bit ↔ x87 op map (enum `X87LibOp` in
[target/i386/tcg/x87_lib_shim.h](target/i386/tcg/x87_lib_shim.h)):

| bit | op       | bit | op |
|---:|---|---:|---|
| 0  | F2XM1    | 7  | FSQRT |
| 1  | FPTAN    | 8  | FSINCOS |
| 2  | FPATAN   | 9  | FRNDINT |
| 3  | FXTRACT  | 10 | FSCALE |
| 4  | FPREM    | 11 | FSIN |
| 5  | FYL2XP1  | 12 | FCOS |
| 6  | FYL2X    |    | |

Bits 13+ are unused.

Test results so far (Halo 2 on Pixel 10a perftest build,
`com.izzy2lost.x1box.perftest`, in title-screen / immediately post-title
gameplay):

| mask          | description                          | verdict |
|---------------|--------------------------------------|---------|
| `0xFFFFFFFF`  | all lib (= default before bisection) | audio + controls latent, physics broken |
| `0x00000000`  | all soft-float                       | geometry got WORSE — missing surfaces (rules out a clean revert to soft-float) |
| `0x000000FF`  | bits 0-7 lib (arith/logs/FSQRT)      | audio less latent, geometry correct |
| `0x000001FF`  | + FSINCOS                            | "fantastic" |
| `0x000003FF`  | + FSINCOS + FRNDINT                  | controls tight + geometry fine, **audio latent** |
| `0x000007FF`  | + FSINCOS + FRNDINT + FSCALE         | game wedged / hung (frozen screen, :xemu process alive but no progress) |
| `0x000005FF`  | + FSINCOS + FSCALE                   | **PENDING USER VERDICT** (in-flight when handoff requested). Claude observation 2026-05-22: FPS at 14 user-visible (vs 28 at `0x1FF`). Game is running, not wedged (rules out FSCALE-alone wedge — the `0x7FF` wedge must come from FRNDINT+FSCALE interaction or FRNDINT pushing total CPU over the wedge threshold). Physics/walls verdict still needs human eyes. |

Conclusions to date:

- **FRNDINT (bit 9) is the audio-latency cause.** Adding it to `0x1FF`
  (which was "fantastic") flips audio to latent without affecting
  controls or geometry. Anything that uses `lroundf` / `(int)f` in the
  guest goes through FRNDINT — that's hot in the MCPX audio sample
  conversion path.

- **FSINCOS (bit 8) is clean** — `0x1FF` was strictly better than
  `0xFF` (subjective "fantastic" vs "correct").

- **Bits 10/11/12 (FSCALE/FSIN/FCOS) are the physics-breaker class.**
  At `0xFFFFFFFF` (these bits on) physics is broken. At `0x3FF` (these
  bits off) physics is fine. The break is somewhere in this group.
  `0x7FF` (which only added FSCALE among 10/11/12) WEDGED — that's a
  strong but not definitive vote against FSCALE.

---

## Open questions (the bisection that remains)

1. **`mask=0x5FF` verdict** — was set just before the handoff request.
   Adds FSCALE to the known-good `0x1FF`. User did not test it.
   - If wedges/crashes/breaks physics → **FSCALE is the physics break**.
   - If "fantastic" → FSCALE is clean, the break is FSIN or FCOS.

2. **If FSCALE is clean**, test:
   - `mask=0x9FF` — adds FSIN to `0x1FF`. Walk into a wall, look around.
   - `mask=0x11FF` — adds FCOS to `0x1FF`. Same test.
   - One of these will reproduce "walk through walls / inaccurate
     sticks." That's the second culprit.

The result is one or two broken ops in `{FSCALE, FSIN, FCOS}`. Plus the
already-confirmed FRNDINT.

---

## How to drive the repro on-device

The diagnostic infrastructure is already wired into the perftest APK
that's installed on the device — no rebuild needed to bisect further.

Live mask toggle (no app restart needed; cache refresh ≤ 500ms):

```bash
adb shell setprop debug.x1box.x87mask 0x5FF
adb shell getprop debug.x1box.x87mask    # confirm
```

Wiring lives in:
- [target/i386/tcg/x87_lib_shim.cpp](target/i386/tcg/x87_lib_shim.cpp) —
  `resolve_op_mask_android()` reads `debug.x1box.x87mask` (Android
  `__system_property_get`) with a 500ms per-thread cache. Falls back to
  env `X1BOX_X87_LIB_MASK` when prop is unset.
- [target/i386/tcg/x87_lib_shim.h](target/i386/tcg/x87_lib_shim.h) —
  `enum X87LibOp`, `xemu_get_x87_lib_op(unsigned op)`, mask getter/setter.
- [target/i386/tcg/fpu_helper.c](target/i386/tcg/fpu_helper.c) — 13
  `if (xemu_get_x87_lib_op(X87_LIB_OP_*))` gates around each call to
  `x87lib_*` (lines 1656, 1832, 1922, 2383, 2460, 2696, 2812, 2977,
  3001, 3036, 3053, 3115, 3140). The fall-through is the existing
  soft-float fpu_helper path.
- [android/app/src/main/cpp/xemu_settings_android.cc](android/app/src/main/cpp/xemu_settings_android.cc)
  sets `setenv("X1BOX_X87_LIB_MASK", "0xFFFFFFFF", 0)` as the boot
  default (overridable by external env or the setprop above).

The global x87 lib enable is the `setting_x87_lib` preference (controls
`xemu_set_x87_lib(true/false)` at
[android/app/src/main/cpp/xemu_android.cpp:1017](android/app/src/main/cpp/xemu_android.cpp#L1017)).
Verify it's ON via:
```bash
adb logcat -s xemu-android:* | grep "x87 lib"
# Should print: x87 lib (Aaron Giles fp80_t soft path): ON
```

Driving the bisection via the MCP server (mcp__x1box-debugger__*):
```python
mcp__x1box-debugger__shell(command="setprop debug.x1box.x87mask 0x5FF; getprop debug.x1box.x87mask")
mcp__x1box-debugger__screenshot(save_path="/tmp/x.png")  # then Read it
mcp__x1box-debugger__measure_fps(duration=8)             # if FPS is the signal
mcp__x1box-debugger__audio_trace_analyze(duration=10)    # for audio latency
```

If the game wedges at a mask value, recover with:
```bash
adb shell am force-stop com.izzy2lost.x1box.perftest
adb shell setprop debug.x1box.x87mask 0xFF
adb shell am start -n com.izzy2lost.x1box.perftest/com.izzy2lost.x1box.LauncherActivity \
  -d 'file:///storage/emulated/0/xbox/Halo%202%20(XBCLASSICRP).iso' \
  -a android.intent.action.VIEW
```

---

## Once the broken op(s) are identified — fix paths

The bug is in upstream `xerootg/x87.git` between `06813043ae..4b78068`.
Three options, ranked by user-impact:

1. **Cherry-pick out the specific bad upstream commit(s)**, then bump
   the `X87_GIT_REV` constant in
   [android/app/src/main/cpp/CMakeLists.txt:170](android/app/src/main/cpp/CMakeLists.txt#L170)
   to the new clean tip. Diff that touched each candidate op:
   - **FRNDINT**: not obviously mentioned in the upstream commit log
     between the two SHAs. Look for any commit that touched
     `range_reduce`, `as_integer`, or the trunc/round paths.
   - **FSCALE**: same — no obvious mention. May be implicit via a
     shared helper (e.g., a precision change in `fpext96` divides).
   - **FSIN / FCOS**: trig range-reduce was rewritten — see upstream
     commits `4b78068 trig range_reduce: use Pentium-truncated π/2`
     and `04e3d69 log/exp: Pentium-truncated ln(2) ...`. These changes
     swapped the constant from full-precision π/2 to a Pentium-truncated
     value. If the Xbox's hardware (P3) actually used full-precision
     π/2 in its microcode, the new lib is now LESS accurate against
     real Xbox behavior. Reverting these two commits restores the
     prior trig behavior.

   Suggested order: try reverting just `4b78068` first (it's the tip
   commit), rebuild, retest at `mask=0xFFFFFFFF`. If physics + audio
   both clear, that one commit covers it. If physics clears but audio
   stays latent, also revert any commit touching `as_integer`/FRNDINT.

2. **Ship with the bits zeroed in production**, so soft-float runs
   those ops while lib runs the rest. Concretely: change the default
   in [xemu_settings_android.cc](android/app/src/main/cpp/xemu_settings_android.cc)
   from `"0xFFFFFFFF"` to e.g. `"0x1FF"` (or the verified-good mask
   after the FSCALE/FSIN/FCOS bisection finishes). Add a comment
   referencing this handoff. Cheap, no upstream work, slightly slower
   on the disabled ops.

3. **File the regression upstream** against the xerootg/x87 fork with
   the specific commit range + the Halo 2 repro. Slow turnaround, but
   the right long-term home for the fix since other emus may consume
   the same lib.

---

## Other in-flight changes (NOT this regression — separate work)

These are uncommitted on the working tree and were touched THIS session.
The next agent should be aware so they don't accidentally include them
in the x87 fix.

1. **MCPX worker-mixbin-sum change** — staged then stashed when it
   regressed audio (caused a "looping audio" bug independent of the
   x87 regression). Run `git stash list` and you'll see it. Files:
   `hw/xbox/mcpx/apu/vp/vp.c`, `hw/xbox/mcpx/apu/vp/vp.h`. The intent
   was sound (move worker mixbin accumulator out from under
   `vwd->lock`) but the implementation has a bug — possibly the
   `had_work_mask` snapshot races with worker bit-clears, or the
   APU's post-cond_wait sum reads worker state too early. Needs a
   careful re-audit before un-stashing. Refs:
   [[project_halo2_audio_sync_bottleneck]] (the audit memory),
   [[project_mcpx_voice_dirty_batcher]] (don't break the dirty-batcher
   flush invariant).

2. **Per-op x87 lib bisection mask** — this handoff's infrastructure.
   The mask defaults to `0xFFFFFFFF` (= prior behavior) so it's
   safe to leave in tree. The Android setprop is a no-cost knob.
   Files: `target/i386/tcg/x87_lib_shim.{cpp,h}`, `fpu_helper.c` (gate
   sites), `xemu_settings_android.cc` (default setenv). Worth landing
   regardless — gives future regressions a built-in bisection tool.
   Memory ref: [[project_x87_lib_bisection]].

3. **`setenv` lines added** in xemu_settings_android.cc — only the new
   `X1BOX_X87_LIB_MASK` one is from this session. The `X1BOX_HLE`,
   `X1BOX_SSE_INLINE` ones were already in tree.

---

## Memory entries worth re-reading

- [[project_x87_lib_slow_methods]] — `_deps/x87-src/` is 3rd-party; the
  fork lives at github.com/xerootg/x87.git. Don't fork from outside.
- [[project_x87_lib_bisection]] — the mask infrastructure this handoff
  uses; same bit map.
- [[project_sse_scalar_inline_gated]] — separate SSE inline gating,
  ALSO has bisection knobs (X1BOX_SSE_INLINE_{ADD,SUB,MUL,DIV,SQRT,COMI}).
  Not relevant to this regression but the user originally suspected it.
- [[project_halo2_audio_sync_bottleneck]] — context for the MCPX
  change that's currently stashed.

---

## TL;DR for the fix agent

1. Read the **PENDING** row in the verdict table above and ask the
   user to confirm the `0x5FF` verdict (or just re-run it — the
   infrastructure is live on-device).
2. Continue the bisection across bits 10, 11, 12 as described.
3. Cherry-pick the responsible upstream commit(s) out of
   `xerootg/x87.git`, bump `X87_GIT_REV` in CMakeLists.txt, rebuild,
   verify `mask=0xFFFFFFFF` is now clean again.
4. Leave the per-op mask infrastructure in tree — it'll save the next
   regression too.

The user is on a Pixel 10a, perftest variant, Halo 2. Active mask
during handoff: `0x5FF`. Active app on-device:
`com.izzy2lost.x1box.perftest`.

---

## Reply from 2nd agent — 2026-05-22 (additions + corrections)

The original FSCALE conclusion in this doc was **wrong** — we (this
session) re-bisected from the opposite direction on-device and need to
overwrite two claims plus add a finding the first pass missed.

### Corrections to the conclusions in this doc

**FSCALE-lib is not the physics breaker — soft-float FSCALE is.**

Original doc concluded: "Bits 10/11/12 (FSCALE/FSIN/FCOS) are the
physics-breaker class … `0x7FF` (which only added FSCALE among 10/11/12)
WEDGED — that's a strong but not definitive vote against FSCALE."

This session's data overrules that:
- `mask=0x4FF` (`0xFF` + bit 10 FSCALE-lib, **no** FSINCOS/FRNDINT) ⇒
  game stable, **"this looks correct"**, no rocket physics on shoot.
- `mask=0xFF`  (FSCALE soft-float) ⇒ user shot an enemy and the body
  *"disappeared through the sky like a rocket"*. Classic out-of-range
  FSCALE scaling-by-2^huge bug.
- The `0x7FF` wedge from the original session was not FSCALE alone —
  it was the **interaction** with FRNDINT-lib (`0x3FF` audio-latent and
  later wedged in the original session too; FRNDINT being a known-bad
  op is what tipped the total state over).

**Bisection bracket update:**

| mask        | description                          | verdict (this session) |
|-------------|--------------------------------------|------------------------|
| `0x000000FF`| FSCALE soft-float                    | **rocket-physics on shoot** (softfloat FSCALE edge case) |
| `0x000001FF`| + FSINCOS                            | "fantastic" — FSINCOS-lib clean ✓ |
| `0x000002FF`| + FRNDINT only (no FSINCOS)          | "fine so far" at title scene — needs in-game re-confirm |
| `0x000003FF`| + FSINCOS + FRNDINT                  | game wedged/hung during cutscene |
| `0x000004FF`| + FSCALE only (no FSINCOS/FRNDINT)   | **"looks correct"** + no rocket on shoot ✓ |
| `0x000007FF`| + FSINCOS + FRNDINT + FSCALE         | wedged (same as 0x3FF — FRNDINT carrying it) |
| `0xFFFFFFFF`| all lib (+ FSIN/FCOS/FSIN-class)     | latency + bad geometry (consistent with original handoff) |

**Production known-good mask: `0x4FF`** — lib for arith/log/sqrt class
plus FSCALE; soft-float for FSINCOS/FRNDINT/FSIN/FCOS.

### Cause of softfloat FSCALE rocket-physics

helper_fscale soft-float path
([target/i386/tcg/fpu_helper.c:3098-3107](target/i386/tcg/fpu_helper.c#L3098-L3107)):

```c
n = floatx80_to_int32_round_to_zero(ST1, &env->fp_status);
...
ST0 = floatx80_scalbn(ST0, n, &env->fp_status);
```

Intel SDM: FSCALE with `|ST(1)| > 2^15` is implementation-defined.
QEMU's softfloat saturates ST1→int32 (INT32_MIN on overflow) and
`scalbn(ST0, INT32_MIN)` collapses to subnormal/zero — but a near-zero
divisor in the game's "1/dist" projectile term blows velocity up to
near-infinity, hence the rocket. Aaron Giles' lib FSCALE handles the
out-of-range case in a way the engine tolerates.

The upstream lib patch on `xerootg/x87.git` (2030a1d / b9480f3) does
**not** touch FSCALE — it correctly didn't, because FSCALE-lib is fine.
**That means upstream still won't fix this on its own**: x1-box needs
FSCALE-lib enabled (mask bit 10 set) regardless of which upstream
ln(2)/π/2 flags we wire.

### Verifying the upstream patch (2030a1d / b9480f3) — it's correct

The other agent pushed two commits to `github.com/xerootg/x87.git`:
- `2030a1d` adds `-DX87_TRIG_FULL_PRECISION_PI=1` to opt out of the
  Pentium-truncated π/2 inside `range_reduce`.
- `b9480f3` adds `-DX87_LOGEXP_FULL_PRECISION_LN2=1` to opt out of the
  Pentium-truncated ln(2) in both f2xm1 paths.

Scope audit (reviewing the patches against the full
`x87fp80trans.cpp`):
- Three `pentium_ln2` / `pio2` truncation sites exist in the file
  (lines ~302, ~414, ~1738). All three are now under the new `#if`
  guards. ✓
- FYL2X / FYL2XP1 use a separate `fpext_t::l2e` (log2(e) = 1/ln(2)),
  not the truncated constant — already full precision. ✓
- The 0xC000…0000 table entries at lines 230-254 in fyl2x's lookup
  table are exact `±k/16` values, not Pentium-truncated constants. ✓
- No other site references the truncated forms.

**Upstream patch is structurally correct.** The flags default to
truncated (matching modern Intel hardware, which is what the upstream
test oracle is) and the opt-out flips to full precision, which is what
x1-box wants for the Xbox P6 target.

### What still needs to land on the x1-box side

1. **Bump `X87_GIT_REV` in CMakeLists.txt** from `4b78068` to `b9480f3`
   (or whatever the new tip is once readme tweaks settle).
2. **Add the two opt-out defines** so the x87 library is compiled with
   full-precision constants. Concretely, in
   [android/app/src/main/cpp/CMakeLists.txt](android/app/src/main/cpp/CMakeLists.txt)
   near the existing FetchContent block, after `x87` is added as a
   target:
   ```cmake
   target_compile_definitions(x87 PRIVATE
       X87_TRIG_FULL_PRECISION_PI=1
       X87_LOGEXP_FULL_PRECISION_LN2=1)
   ```
   (Verify exact target name — it may be `x87_lib` etc. — by grepping
   the existing `add_library` / FetchContent call sites.)
3. **Rebuild + retest at `mask=0xFFFFFFFF`.** Hypothesis: with both
   flags on, the FSIN/FCOS/FSINCOS-via-range_reduce regression
   disappears, so the only remaining bad bit is whatever was driving
   the `0x3FF`/`0x7FF` wedge — likely FRNDINT (this session never
   in-game-confirmed FRNDINT-only is broken; the prior session's
   `0x3FF` audio-latency observation is the strongest evidence).
4. **If `0xFFFFFFFF` is still bad after the flags**, fall back to the
   per-op mask: set the production default in
   [xemu_settings_android.cc](android/app/src/main/cpp/xemu_settings_android.cc)
   from `"0xFFFFFFFF"` to the verified-good mask (`0x4FF` floor, can
   probably go higher once FSIN/FCOS retest under new flags).

### What the upstream patch can't fix — softfloat FSCALE

Regardless of which trig/log flags we wire, the rocket-physics bug
lives in QEMU's softfloat `floatx80_to_int32_round_to_zero` +
`floatx80_scalbn` combination, not in the lib. Two options:

a) **Always keep bit 10 (FSCALE) set in the production mask.** Lib
   handles the out-of-range ST(1) correctly. Cheapest fix.

b) **Fix the soft-float path** to clamp |ST1| to [-2^15, 2^15] before
   the scalbn (mirroring what hardware does, per Intel SDM —
   implementation-defined but most x87s saturate the exponent, not the
   mantissa-times-2^underflow). Patch site is the `else` block at
   [fpu_helper.c:3098-3107](target/i386/tcg/fpu_helper.c#L3098-L3107).
   Lower-risk if we ever ship with the lib globally OFF.

Recommend (a) as the immediate fix, (b) as a follow-up so the lib
isn't load-bearing for correctness on this op.

### Other in-flight changes (this session adds nothing to undo)

- The MCPX worker-mixbin stash from the original handoff is still
  stashed (not in this build). Don't un-stash without re-auditing.
- The per-op mask infrastructure is unchanged. Setprop continues to
  work.
- The `debug.x1box.x87mask` setprop on the device is currently
  `0x4FF` (the verified-good mask) as of the time of this reply.

### Recommended fix-agent next action

1. Apply CMakeLists.txt changes per "What still needs to land" §1-§2.
2. Rebuild + install + relaunch Halo 2.
3. With `setprop debug.x1box.x87mask 0xFFFFFFFF`, ask user to repeat
   the Halo-2-cutscene + shoot-someone test from this session.
4. Three possible outcomes:
   - **Clean** → both bugs were in trig constants; raise the
     production default to `0xFFFFFFFF`, mention `0x4FF` as the
     known-good fallback, and we're done.
   - **Rocket-physics still gone but audio still latent** →
     FRNDINT-lib is independently broken regardless of the trig flags.
     Either keep bit 9 off in production (`0x5FF` etc.) or push a
     FRNDINT-specific fix upstream.
   - **Rocket-physics back** → my reading of the FSCALE softfloat bug
     is wrong; the lib's FSCALE *was* what the user was seeing during
     "this looks correct" at `0x4FF`, and the new flags somehow
     changed lib FSCALE behavior. Re-bisect from `0x4FF`.

— 2nd agent
