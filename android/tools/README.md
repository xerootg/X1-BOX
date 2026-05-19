# x1box-debugger — MCP server for live X1 BOX debugging

ADB-backed MCP server (`x1box_debugger_mcp.py`) that gives an agent direct
control of a running X1 BOX (xemu-on-Android) install: process state, native
memory peek/poke, GPU/Vulkan introspection, log capture, build/install, native
profiling — plus **real QEMU gdbstub control** of the guest Xbox i386 CPU
(breakpoints, single-step, registers, guest memory).

**One paragraph for a fresh agent:** if the user is debugging X1 BOX on a
phone, you almost certainly want these tools instead of raw `adb shell`. They
encode the package, process layout, prefs paths, and the modern-QEMU gdbstub
quirks (multiprocess thread selection, generated register-feature table)
that a naive ADB session will get wrong.

---

## When you'd use this

- "Why did Halo 2 crash?" → `pull_log` / `search_log` / `logcat` for the
  native VK_CHECK + libc abort lines.
- "Where in the BIOS does the guest stop?" → `gdb_enable(pause_at_boot=True)`
  → `launch_app(rom=...)` → `gdb_connect` → `gdb_read_registers`.
- "I changed `xemu_android.cpp`, deploy and test." →
  `build_and_install(variant='debug')` → `launch_app(...)`.
- "What's the renderer doing under load?" → `profile(duration=5)` and
  `get_gfxinfo` / `get_gpu_stats`.
- "Wipe the shader cache before this run." → `clear_cache(kind='shader')`.

---

## Prerequisites

| What | Why |
|---|---|
| `adb` reachable on `PATH` and one device authorised | Every tool shells out via `adb`. |
| **Debug build** of `com.izzy2lost.x1box` installed | Anything that uses `run-as` (memory peek/poke, prefs editing, simpleperf, log pulls from internal storage, `clear_cache`) only works on a debuggable build. Release builds are not debuggable. |
| Python `mcp` package | `from mcp.server.fastmcp import FastMCP`. The server is registered in `~/.claude.json` as `x1box-debugger`; restart Claude Code after installing to pick up the tools. |

To build + install a debug APK in one shot:

```python
build_and_install(variant="debug")   # ./gradlew assembleDebug && adb install -r ...
```

---

## On-device layout this server assumes

- **Package**: `com.izzy2lost.x1box`
- **Launcher activity**: `.LauncherActivity` (handles `file://` / `content://` ROM intents)
- **Emulation activity**: `.MainActivity`, runs in **`:xemu`** process (qemu_main + JIT + Vulkan renderer)
- **xemu runtime files** (external — readable directly):
  `/storage/emulated/0/Android/data/com.izzy2lost.x1box/files/x1box/`
  → `xemu.toml`, `eeprom.bin`, `mcpx.bin`, `flash.bin`, `hdd.img`, `dvd.iso`,
    `inline_aio_required.flag`
- **Debug logs + prefs** (internal — needs `run-as`):
  `/data/data/com.izzy2lost.x1box/files/x1box/debug-logs/{ui,xemu}-{debug,logcat}.log`
  `/data/data/com.izzy2lost.x1box/shared_prefs/x1box_prefs.xml`

Two processes:

- `pidof com.izzy2lost.x1box` — UI / launcher
- `pidof com.izzy2lost.x1box:xemu` — emulation (where the interesting stuff lives)

All tools that take a `target` arg default to `"emu"` (the `:xemu` process).
Pass `target="ui"` to inspect the launcher process instead.

---

## Tool catalogue (50 total)

### Lifecycle

| Tool | Notes |
|---|---|
| `list_devices()` | `adb devices`. |
| `launch_app(rom="")` | Empty arg = open launcher. With a path or URI, sends `action.VIEW` to `LauncherActivity`, which boots straight into emulation when core files are set up. |
| `stop_app()` | Force-stops both processes. Call before editing prefs so reads-on-boot pick up changes. |
| `screenshot(save_path)` | Pulls a PNG. |

### Process inspection

| Tool | Notes |
|---|---|
| `get_pid()` | UI + `:xemu` pids. |
| `get_threads(target='emu' \| 'ui' \| 'all')` | tid, state, name. Useful with `profile()`. |
| `pause_process(target)` / `resume_process(target)` | `SIGSTOP`/`SIGCONT` the host process. Different from `gdb_break` — this freezes the **host process** (Vulkan threads and all), not the guest CPU. |
| `get_memory_maps(filter, target)` | `/proc/<pid>/maps`. Grep for `libxemu`, `[stack]`, `mali` etc. |
| `read_memory(address, size, fmt, target)` | Reads **host** virtual address space (`/proc/<pid>/mem` via `run-as`). For **guest** memory use `gdb_read_guest_memory` instead. fmt: `hex \| u32le \| f32le \| bytes`. |
| `write_memory(address, hex_bytes, target)` | Host poke. Same scope warning as above. |

### GPU / Vulkan

| Tool | Notes |
|---|---|
| `get_gpu_info()` | `dumpsys gpu` + SurfaceFlinger driver lines. |
| `get_gfxinfo()` | Frame timings / jank counts. |
| `get_gpu_stats()` | Mali sysfs + Adreno KGSL + meminfo for UI **and** `:xemu`. |
| `get_vulkan_layers()` | Discoverable validation layers. |
| `get_gpu_driver_dir()` | adrenotools custom-driver install dir under the app sandbox. |

### Logs

| Tool | Notes |
|---|---|
| `enable_debug_logs(enabled=True)` | Toggles `setting_debug_logs_enabled` in prefs — needed for the file-based log capture in `DebugLog.kt`. |
| `pull_log(name='xemu', tail_lines, grep)` | `name` ∈ `xemu, ui, logcat, uilogcat`. |
| `search_log(pattern, name, context_lines, max_matches)` | Regex with surrounding context. |
| `clear_log(name='all')` | Truncate one or all. |
| `watch_log(pattern, name, timeout_secs)` | Poll until regex hits. |
| `logcat(lines, tag, grep)` | Raw `adb logcat -d`. |

### Cache / config / state

| Tool | Notes |
|---|---|
| `show_config()` | Cat `xemu.toml`. |
| `push_config(local_path)` | Replace `xemu.toml`. |
| `list_state(kind)` | `kind` ∈ `all, config, logs, shader_cache, snapshots, gpu_driver`. |
| `clear_cache(kind)` | `kind` ∈ `shader, logs, snapshots, covers, disc_format_cache`. |

### Build / install / FS

| Tool | Notes |
|---|---|
| `install_apk(apk_path="", variant)` | Empty path picks newest APK in `app/build/outputs/apk/<variant>/`. |
| `build_and_install(variant)` | `./gradlew assemble<Variant>` then install. |
| `shell(command)` | Raw `adb shell`. Use `run-as com.izzy2lost.x1box sh -c '...'` for sandboxed paths. |
| `push_file` / `pull_file` | Plain `adb push/pull`. For app-internal paths, prefer the dedicated tools (they handle `run-as` for you). |

### Native profiling

| Tool | Notes |
|---|---|
| `profile(duration=3, tid="", target='emu')` | `simpleperf record --call-graph fp` then `report --sort symbol`. Symbols resolve from `libxemu.so`. Caps duration at 60s. |

### Native debugger bootstrap (host-side, not guest)

| Tool | Notes |
|---|---|
| `attach_native_debugger(port=5039, lldb_server_path="")` | Pushes `lldb-server` from your NDK into the app sandbox, launches it as `platform --listen *:<port>`, and runs `adb forward`. Use this when you want to set breakpoints in **`libxemu.so`** (Vulkan renderer, JIT, glue code) — i.e. the host process, NOT the emulated CPU. |
| `detach_native_debugger(port)` | Kills lldb-server in the sandbox and removes the forward. |

### Guest CPU debugging via QEMU gdbstub (`gdb_*` family)

**This is the headline feature.** Real breakpoints, step, register dumps, and
guest-memory peek/poke on the emulated **Xbox i386 CPU** — not host-side.

| Tool | Notes |
|---|---|
| `gdb_enable(port=1234, pause_at_boot=False)` | Writes `XEMU_ANDROID_GDB_PORT` (and optionally `XEMU_ANDROID_GDB_PAUSE=1`) into the app's `env_vars` SharedPreference. `xemu_android.cpp` reads these and injects `-gdb tcp::PORT` (+ `-S`) into QEMU's argv on next launch. Force-stops the app first so the new prefs take effect. |
| `gdb_disable()` | Strip those env vars. |
| `gdb_connect(port=1234)` | `adb forward` then open RSP session. Negotiates `multiprocess+` and auto-selects the current thread (modern QEMU gdbstub needs `Hg<tid>` before `g`/`p`/`m` work — the client handles this; don't issue it yourself). |
| `gdb_disconnect(port=0)` | Close RSP. Pass `port>0` to also drop the `adb forward`. |
| `gdb_status()` | Stop reason (`T05thread:p01.01;` style). |
| `gdb_break()` | Halt the guest CPU (`\x03` over RSP). Different from `pause_process` — this stops the emulated x86, not the Android process. |
| `gdb_continue(wait=False, timeout=5.0)` | Resume. `wait=True` blocks until the next stop. |
| `gdb_step()` | Single guest instruction. |
| `gdb_set_breakpoint(address, kind=1)` / `gdb_clear_breakpoint(...)` | Software BPs (`Z0`). `address` is a hex string. |
| `gdb_set_watchpoint(address, length=4, kind='w' \| 'r' \| 'a')` | Hardware watchpoint via `Z2/Z3/Z4`. x86 limit is 4 slots. |
| `gdb_list_breakpoints()` | Client-tracked set (gdbstub itself doesn't list). |
| `gdb_read_registers()` | i386 GPRs + EFLAGS + segment selectors. |
| `gdb_write_register(name, value)` | `value` parsed via `int(value, 0)` so `0xdeadbeef` and `1234` both work. |
| `gdb_read_guest_memory(address, size=64, fmt='hex' \| 'u32le' \| 'f32le' \| 'bytes')` | Walks QEMU's MMU — sees the **guest's** view of RAM/MMIO. |
| `gdb_write_guest_memory(address, hex_bytes)` | Guest poke. `gdb_break` first for a coherent write. |

---

## Recipes

### 1. Halt guest at boot, inspect the reset vector

```python
stop_app()
gdb_enable(port=1234, pause_at_boot=True)
launch_app(rom="/storage/emulated/0/xbox/Halo 2 (XBCLASSICRP).iso")

# wait briefly for the gdbstub to bind (qemu_init ~1.3s on a Pixel 10a)
import time; time.sleep(2)

gdb_connect(1234)
print(gdb_read_registers())   # eip should be 0xfff0, eflags 0x2, cs 0xf000
gdb_read_guest_memory("0xffff0000", 64, fmt="u32le")   # MCPX shadow
```

### 2. Set a guest BP, run, observe

```python
gdb_set_breakpoint("0x80012345")
gdb_continue(wait=True, timeout=30)   # blocks until BP hits (or timeout)
print(gdb_read_registers())
gdb_clear_breakpoint("0x80012345")
gdb_continue()
```

### 3. Live-edit code path while running (host side)

```python
build_and_install(variant="debug")    # iterate edit -> rebuild -> reinstall
launch_app(rom="...")
profile(duration=5)                    # see hotspots
search_log("VK_CHECK", name="xemu", context_lines=3)
```

### 4. Native breakpoint in libxemu.so (renderer / JIT)

```python
attach_native_debugger(port=5039)
# Then on the host, run: lldb -> platform select remote-android ->
#   platform connect connect://localhost:5039 -> attach --pid <emu pid>
```

### 5. Reproduce the Mali device-lost without burning state between runs

```python
clear_cache(kind="shader")
clear_log(name="all")
enable_debug_logs(True)
launch_app(rom="...")
watch_log("VK_ERROR_DEVICE_LOST", name="xemu", timeout_secs=120)
# inspect captured logs
pull_log(name="xemu", tail_lines=200, grep="vk")
```

---

## Gotchas

- **Halo 2 hits a Mali `VK_ERROR_DEVICE_LOST` ~30 s in**, surfaced via a libc
  assertion abort at `hw/xbox/nv2a/pgraph/vk/renderer.c:1479`
  (`vkWaitForFences`). The renderer keeps submitting GPU work even when the
  guest CPU is paused via `gdb_break` or `-S`, so `pause_at_boot` does **not**
  shield you from this crash. To validate gdbstub end-to-end, connect within
  the ~30 s window or use a lighter title.
- **Debug build required for most things.** Release isn't debuggable, so
  `run-as` fails → `read_memory`, `write_memory`, `enable_debug_logs`,
  `clear_log`, `clear_cache`, `profile`, `gdb_enable`/`gdb_disable`,
  `attach_native_debugger` all bail out with "Install the debug build first".
- **Prefs edits require `stop_app` first.** Already done by `gdb_enable` /
  `gdb_disable` / `enable_debug_logs` internally — if you write to prefs by
  hand, force-stop yourself so the next launch reads the new values.
- **`gdb_connect` race.** The gdbstub binds inside `qemu_init`, which takes
  ~1–2 s after `:xemu` spawns. If your first connect dies with "gdb peer
  closed connection," sleep 1.5 s and try again. `gdb_connect` itself is
  cheap — re-call freely.
- **Modern QEMU gdbstub thread selection.** Per-register reads (`p<N>`) and
  the legacy `g` packet need an explicit `Hg<tid>` first, with **no space**
  between `Hg` and the thread id (`Hgp01.01`, not `Hg p1.1` — the latter
  returns `E22`). The MCP client auto-selects the thread parsed from the
  stop reply on `gdb_connect`; only worry about this if you send raw RSP.
- **Watch the host-vs-guest scope.** `read_memory` reads the Android
  process's `/proc/<pid>/mem` (libxemu, Vulkan structs, JIT-emitted code).
  `gdb_read_guest_memory` reads the emulated Xbox CPU's address space (game
  RAM, BIOS shadow, MMIO). They're not interchangeable.

---

## Wiring (one screen)

```
LauncherActivity (UI process)
  └─ MainActivity (:xemu process)
       └─ SDL_main → xemu_android_main → qemu_init(argc, argv) → qemu_main
                                          │
                                          └─ argv built from:
                                              -accel tcg,thread=...,tb-size=...
                                              -add-fd fd=…,set=0          (DVD)
                                              -gdb tcp::PORT              ← XEMU_ANDROID_GDB_PORT
                                              -S                          ← XEMU_ANDROID_GDB_PAUSE=1
```

Env vars come from the `env_vars` SharedPreference (multi-line `KEY=VALUE`
entries, one per line), read in `SyncSetupFiles()` and `setenv()`'d before
`SDL_main` builds `arg_storage`. The `gdb_enable` tool edits that pref via
`run-as`. See [xemu_android.cpp:1325-1342](../app/src/main/cpp/xemu_android.cpp#L1325)
for the `-gdb` injection point.

Build-system fixes needed for the gdbstub to work on Android (already in
[CMakeLists.txt](../app/src/main/cpp/CMakeLists.txt)):

- `chardev/char-socket.c` is built into an `OBJECT` library
  (`xemu_force_keep_objs`), bypassing the dead-code-elim that the
  `xemu_core` STATIC archive otherwise applies to .o files whose only
  references are `__attribute__((constructor))` init functions. Without
  this the `socket` chardev type isn't registered and `-gdb tcp::PORT`
  fails with `'socket' is not a valid char driver name`.
- `gdb-static-features.c` is generated at build time from
  `gdb-xml/i386-{32bit,32bit-linux}.xml` via `scripts/feature_to_c.py`,
  replacing the empty `stubs/gdbstub.c`. Without the populated
  `gdb_static_features[]` table, register reads (`g`, `p<N>`) return
  empty / `E14`.

---

## Adding new tools

The server is a single file: [x1box_debugger_mcp.py](./x1box_debugger_mcp.py).
Decorate a function with `@mcp.tool()` and it shows up after a Claude Code
restart. Use the helpers near the top:

- `_adb(*args)` / `_sh(*args)` — adb / adb shell
- `_device()` — None if no device attached (return early)
- `_is_debuggable()` — call before anything that needs `run-as`
- `_emu_pid()` / `_ui_pid()` — process PIDs
- `_pull_to_tmp(device_path)` — adb pull with `run-as cat` fallback for
  `/data/...` paths
- `_rewrite_env_vars(set_lines, drop_keys)` — round-trip `env_vars` pref
  cleanly (cat + parse + push + run-as cp)
- `_GDB` (module-level `_GdbClient`) — persistent RSP session
