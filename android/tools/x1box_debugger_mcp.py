#!/usr/bin/env python3
"""
X1 BOX Android MCP Debugger
ADB-backed tools for debugging X1 BOX (xemu-on-Android) on a real device:
- Process control (pause/resume, PIDs, threads, /proc maps, memory peek/poke)
- Vulkan & GPU introspection (dumpsys gpu, gfxinfo, Mali/Adreno stats)
- Native profiling via simpleperf on the :xemu emulation process
- Log capture (xemu native, UI, logcat) with grep/regex/tail/watch
- Cache and config management (xemu.toml, debug-logs, shader cache)
- Title launch via intent (file path or content:// URI)
- Build / install (gradlew assembleDebug + adb install)
- Native debugger bootstrap (lldb-server in app sandbox + adb port forward)
- QEMU gdbstub control over the GDB Remote Serial Protocol: enable/disable
  the stub, set/clear breakpoints, read/write guest registers and memory,
  interrupt/step/continue. Drives the guest i386 CPU directly.

Mirrors the layout of cemu_debugger_mcp.py so the two feel the same.
"""

from __future__ import annotations

import os
import re
import socket
import struct
import subprocess
import tempfile
import time
import xml.sax.saxutils as _saxutil
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("x1box-debugger")

# ---------------------------------------------------------------------------
# Constants — package / activity / process layout
# ---------------------------------------------------------------------------

PKG = "com.izzy2lost.x1box"
LAUNCHER_ACTIVITY = f"{PKG}/.LauncherActivity"
EMU_PROCESS = f"{PKG}:xemu"            # MainActivity runs in :xemu (see AndroidManifest)

# Filesystem paths used by the running app.
# External (world-readable via the app id sandbox): xemu's runtime files.
EXT_BASE = f"/storage/emulated/0/Android/data/{PKG}/files/x1box"
CONFIG_TOML = f"{EXT_BASE}/xemu.toml"
EEPROM_BIN = f"{EXT_BASE}/eeprom.bin"

# Internal (requires run-as on debuggable build): debug-log directory.
INT_FILES_DIR = f"/data/data/{PKG}/files"
LOG_DIR = f"{INT_FILES_DIR}/x1box/debug-logs"
UI_DEBUG_LOG = f"{LOG_DIR}/ui-debug.log"
XEMU_DEBUG_LOG = f"{LOG_DIR}/xemu-debug.log"
UI_LOGCAT_LOG = f"{LOG_DIR}/ui-logcat.log"
XEMU_LOGCAT_LOG = f"{LOG_DIR}/xemu-logcat.log"
PREFS_XML = f"/data/data/{PKG}/shared_prefs/x1box_prefs.xml"
SHADER_CACHE_DIR = f"{INT_FILES_DIR}/cache"   # xemu cache_shaders lives under app cache; cleared with cache wipe

ANDROID_ROOT = Path(__file__).resolve().parent.parent  # /home/.../x1-box/android
APK_OUT_DIR = ANDROID_ROOT / "app/build/outputs/apk"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True, check=False)

def _sh(*args: str) -> str:
    r = _adb("shell", *args)
    return (r.stdout + r.stderr).strip()

def _device() -> str | None:
    lines = _adb("devices").stdout.strip().splitlines()[1:]
    devs = [l.split()[0] for l in lines if l.strip() and "device" in l]
    return devs[0] if devs else None

def _ui_pid() -> str | None:
    """PID of the main UI process."""
    return _sh(f"pidof {PKG}").strip() or None

def _emu_pid() -> str | None:
    """PID of the :xemu emulation process (where qemu_main runs).

    Android's `pidof` matches on /proc/<pid>/comm which the kernel
    truncates to 15 chars. The pseudo-process name "<pkg>:xemu" is
    26 chars, so `pidof com.izzy2lost.x1box:xemu` always returns
    empty even when the process is running. Use cmdline matching
    instead — /proc/<pid>/cmdline preserves the full name.

    Falls back to the UI pid when :xemu isn't running yet.
    """
    # Walk /proc/<pid>/cmdline directly. One forked grep, vs many.
    out = _sh(
        f"grep -l -F '{EMU_PROCESS}' /proc/[0-9]*/cmdline 2>/dev/null"
    ).strip()
    for line in out.splitlines():
        # /proc/<pid>/cmdline
        parts = line.split("/")
        if len(parts) >= 3 and parts[2].isdigit():
            return parts[2]
    return _ui_pid()

def _is_debuggable() -> bool:
    """Best-effort check: run-as must work for memory peek/poke + log pulls."""
    out = _sh(f"run-as {PKG} id 2>&1")
    return "uid=" in out

def _pull_to_tmp(device_path: str, via_run_as: bool = True) -> str | None:
    """Pull a device file to a local temp file. For paths inside the app's
    internal storage, copy via `run-as cat` because `adb pull` can't see
    /data/data even on debuggable builds.
    """
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as f:
        tmp = f.name

    if via_run_as and device_path.startswith("/data/"):
        r = subprocess.run(
            ["adb", "exec-out", f"run-as {PKG} cat {device_path}"],
            capture_output=True, check=False,
        )
        if r.returncode != 0 or not r.stdout:
            os.unlink(tmp)
            return None
        with open(tmp, "wb") as f:
            f.write(r.stdout)
        return tmp

    r = _adb("pull", device_path, tmp)
    if r.returncode != 0:
        os.unlink(tmp)
        return None
    return tmp

def _resolve_apk(variant: str) -> Path | None:
    d = APK_OUT_DIR / variant
    if not d.is_dir():
        return None
    apks = list(d.glob("*.apk"))
    if not apks:
        return None
    return max(apks, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Device / lifecycle
# ---------------------------------------------------------------------------

@mcp.tool()
def list_devices() -> str:
    """List connected ADB devices."""
    return _adb("devices").stdout.strip()

@mcp.tool()
def launch_app(rom: str = "") -> str:
    """Launch X1 BOX. With no args, opens the launcher; with `rom` set, hands
    that path/URI to LauncherActivity (which boots straight into emulation
    when core setup is complete).

    rom: absolute device path (`/sdcard/Games/foo.iso`) or a content://
         URI the app already has read permission for. Leave empty to just
         open the launcher UI.
    """
    if not _device(): return "No device connected."
    if rom:
        if rom.startswith("content://") or rom.startswith("file://"):
            data = rom
        else:
            data = f"file://{rom}"
        out = _sh(
            "am start -a android.intent.action.VIEW "
            f"-d '{data}' -n {LAUNCHER_ACTIVITY}"
        )
    else:
        out = _sh(f"am start -n {LAUNCHER_ACTIVITY}")
    return out or "Launch sent."

@mcp.tool()
def stop_app() -> str:
    """Force-stop both the UI and :xemu processes."""
    if not _device(): return "No device connected."
    return _sh(f"am force-stop {PKG}") or f"{PKG} stopped."

@mcp.tool()
def screenshot(save_path: str = "/tmp/x1box_screen.png") -> str:
    """Take a device screenshot and save locally. Returns the local path."""
    if not _device(): return "No device connected."
    _sh("screencap -p /sdcard/_x1box_ss.png")
    r = _adb("pull", "/sdcard/_x1box_ss.png", save_path)
    _sh("rm -f /sdcard/_x1box_ss.png")
    if r.returncode != 0: return f"Screenshot failed: {r.stderr.strip()}"
    return save_path


# ---------------------------------------------------------------------------
# Process inspection
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pid() -> str:
    """Return the X1 BOX PIDs (UI + :xemu emulation, when present)."""
    if not _device(): return "No device connected."
    ui = _ui_pid()
    emu = _sh(f"pidof {EMU_PROCESS}").strip()
    if not ui and not emu:
        return f"{PKG}: not running"
    parts = [f"pkg={PKG}"]
    if ui:  parts.append(f"ui={ui}")
    if emu: parts.append(f"emu={emu}")
    return "  ".join(parts)

@mcp.tool()
def get_threads(target: str = "emu") -> str:
    """List threads of the X1 BOX process with name and state.
    target: 'emu' (default, :xemu emulation process), 'ui' (launcher process),
            or 'all' to dump both.
    """
    if not _device(): return "No device connected."

    def dump(pid: str | None, label: str) -> str:
        if not pid: return f"{label}: not running"
        tids = _sh(f"ls /proc/{pid}/task").split()
        rows = []
        for tid in tids:
            name = _sh(f"cat /proc/{pid}/task/{tid}/comm 2>/dev/null").strip()
            state = _sh(
                f"grep -m1 '^State:' /proc/{pid}/task/{tid}/status 2>/dev/null"
            ).replace("State:", "").strip()
            rows.append(f"tid={tid:>6}  state={state:<20} name={name}")
        return f"=== {label} (pid={pid}) ===\n" + ("\n".join(rows) or "(no threads)")

    target = target.lower()
    if target == "all":
        return dump(_ui_pid(), "ui") + "\n\n" + dump(_sh(f"pidof {EMU_PROCESS}").strip() or None, "emu")
    if target == "ui":
        return dump(_ui_pid(), "ui")
    return dump(_emu_pid(), "emu")

@mcp.tool()
def pause_process(target: str = "emu") -> str:
    """SIGSTOP the emulation (or UI) process to freeze it for memory inspection.
    Call resume_process() to continue.
    target: 'emu' (default) or 'ui'.
    """
    if not _device(): return "No device connected."
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"
    return _sh(f"kill -STOP {pid}") or f"SIGSTOP -> pid {pid} ({target})"

@mcp.tool()
def resume_process(target: str = "emu") -> str:
    """SIGCONT the X1 BOX process (resume after pause_process)."""
    if not _device(): return "No device connected."
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"
    return _sh(f"kill -CONT {pid}") or f"SIGCONT -> pid {pid} ({target})"

@mcp.tool()
def get_memory_maps(filter: str = "", target: str = "emu") -> str:
    """Read /proc/<pid>/maps for the chosen process.
    filter: optional substring to keep only matching lines (e.g. 'libxemu', 'stack').
    target: 'emu' (default) or 'ui'.
    """
    if not _device(): return "No device connected."
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"
    out = _sh(f"cat /proc/{pid}/maps")
    if filter:
        out = "\n".join(l for l in out.splitlines() if filter.lower() in l.lower())
    return out or "(no matching mappings)"

@mcp.tool()
def read_memory(address: str, size: int = 64, fmt: str = "hex", target: str = "emu") -> str:
    """Read raw bytes from a process's virtual address space.

    address: hex address string like '0xf5807800'
    size: number of bytes (capped at 4096)
    fmt: 'hex' | 'u32le' | 'f32le' | 'bytes'
    target: 'emu' (default) or 'ui'

    Pauses the process briefly for a consistent snapshot. Requires a
    debuggable build (X1 BOX's debug variant is debuggable; release is not).
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return ("run-as failed: install the debug build "
                "(./gradlew assembleDebug && install_apk('debug')) to peek/poke memory.")
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"

    size = min(max(size, 1), 4096)
    try:
        addr_int = int(address, 16)
    except ValueError:
        return f"Bad address: {address}"

    _sh(f"kill -STOP {pid}")
    try:
        cmd = f"dd if=/proc/{pid}/mem bs=1 skip={addr_int} count={size} 2>/dev/null | xxd"
        out = _sh(f"run-as {PKG} sh -c '{cmd}'")
    finally:
        _sh(f"kill -CONT {pid}")

    if not out.strip():
        return f"Could not read memory at {address} (permission denied or invalid address)."

    if fmt == "hex":
        return out

    raw = bytearray()
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) < 2: continue
        hex_part = parts[1].split("  ")[0].strip()
        for b in hex_part.split():
            try: raw.append(int(b, 16))
            except ValueError: pass

    if fmt == "u32le":
        vals = struct.unpack_from(f"<{len(raw)//4}I", bytes(raw[:len(raw)//4*4]))
        body = "\n".join(f"  [{i*4:+#06x}]  0x{v:08x}  ({v})" for i, v in enumerate(vals))
        return f"uint32_t[{len(vals)}] @ {address}:\n{body}"
    if fmt == "f32le":
        vals = struct.unpack_from(f"<{len(raw)//4}f", bytes(raw[:len(raw)//4*4]))
        body = "\n".join(f"  [{i*4:+#06x}]  {v:.6g}" for i, v in enumerate(vals))
        return f"float[{len(vals)}] @ {address}:\n{body}"
    return out

@mcp.tool()
def write_memory(address: str, hex_bytes: str, target: str = "emu") -> str:
    """Poke raw bytes into a process's virtual address space.

    address: hex address (e.g. '0xf5807800')
    hex_bytes: contiguous hex string ('deadbeef') or space/comma-separated
               byte values ('de ad be ef' or 'de,ad,be,ef')
    target: 'emu' (default) or 'ui'

    Pauses the process during the write. Requires the debuggable build.
    Use with extreme care - writing to live emulator state can corrupt
    in-flight Vulkan buffers and crash the renderer.
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return ("run-as failed: install the debug build to poke memory.")
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"
    try:
        addr_int = int(address, 16)
    except ValueError:
        return f"Bad address: {address}"

    clean = re.sub(r"[,\s]", "", hex_bytes)
    if len(clean) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", clean or ""):
        return f"Bad hex_bytes: {hex_bytes!r}"
    raw = bytes.fromhex(clean)
    if not raw: return "No bytes to write."

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(raw)
        local = f.name
    remote = f"/data/local/tmp/_x1box_poke_{os.getpid()}.bin"
    try:
        if _adb("push", local, remote).returncode != 0:
            return "Failed to stage payload on device."
        _sh(f"kill -STOP {pid}")
        try:
            # dd of=/proc/<pid>/mem seek=<addr> writes at the given offset.
            cmd = (
                f"dd if={remote} of=/proc/{pid}/mem bs=1 "
                f"seek={addr_int} count={len(raw)} conv=notrunc 2>&1"
            )
            out = _sh(f"run-as {PKG} sh -c '{cmd}'")
        finally:
            _sh(f"kill -CONT {pid}")
            _sh(f"rm -f {remote}")
    finally:
        os.unlink(local)
    return f"Wrote {len(raw)} bytes at {address}: {out or 'ok'}"


# ---------------------------------------------------------------------------
# GPU / Vulkan introspection
# ---------------------------------------------------------------------------

@mcp.tool()
def get_gpu_info() -> str:
    """Dump GPU driver info (dumpsys gpu + SurfaceFlinger driver lines)."""
    if not _device(): return "No device connected."
    parts = ["=== dumpsys gpu ===", _sh("dumpsys gpu"),
             "\n=== SurfaceFlinger GPU/driver lines ===",
             _sh("dumpsys SurfaceFlinger | grep -i "
                 "'gpu\\|vulkan\\|mali\\|adreno\\|driver\\|version' | head -40")]
    return "\n".join(parts)

@mcp.tool()
def get_gfxinfo() -> str:
    """Frame timing / jank stats for X1 BOX (dumpsys gfxinfo)."""
    if not _device(): return "No device connected."
    return _sh(f"dumpsys gfxinfo {PKG}")

@mcp.tool()
def get_gpu_stats() -> str:
    """GPU utilisation / memory: Mali sysfs, Adreno KGSL, app meminfo."""
    if not _device(): return "No device connected."
    parts = []
    mali = _sh(
        "cat /sys/class/misc/mali0/device/utilization 2>/dev/null || "
        "cat /sys/devices/platform/*/utilization 2>/dev/null | head -5"
    )
    if mali: parts.append(f"Mali utilization:\n{mali}")
    adreno = _sh("cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null")
    if adreno: parts.append(f"Adreno busy%: {adreno}")
    parts.append("\n=== meminfo (UI process) ===")
    parts.append(_sh(f"dumpsys meminfo {PKG} | head -40"))
    emu = _sh(f"pidof {EMU_PROCESS}").strip()
    if emu:
        parts.append("\n=== meminfo (:xemu) ===")
        parts.append(_sh(f"dumpsys meminfo {emu} | head -40"))
    return "\n".join(parts) or "No GPU stats available."

@mcp.tool()
def get_vulkan_layers() -> str:
    """List Vulkan validation layers discoverable on the device."""
    if not _device(): return "No device connected."
    found = _sh(
        "find /data/local/debug/vulkan /vendor/lib64/libVkLayer* "
        "/system/lib64/libVkLayer* 2>/dev/null | head -30"
    )
    if found: return found
    return _sh("getprop debug.vulkan.layers") or "(none found)"

@mcp.tool()
def get_gpu_driver_dir() -> str:
    """List the custom GPU driver install dir (adrenotools)."""
    if not _device(): return "No device connected."
    return _sh(f"run-as {PKG} ls -la files/gpu_driver/ 2>&1") or "(empty)"


# ---------------------------------------------------------------------------
# Per-thread CPU / runqueue-wait / sleep profiler
#
# Useful for "why isn't this thread running?" type questions. Reads only
# /proc/<pid>/task/<tid>/{stat,schedstat,wchan,status}, so it works on a
# debuggable build without root and without instrumenting the binary.
# ---------------------------------------------------------------------------

_THREAD_PROFILE_SCRIPT = r"""
PID="$1"
DUR_MS="$2"
INTERVAL_MS="$3"
if [ -z "$PID" ] || [ ! -d "/proc/$PID/task" ]; then
  echo "ERR no_pid"
  exit 1
fi

# Sleep arg as "0.NNN" seconds. Toybox sleep accepts floats.
SLEEP_ARG=$(awk -v ms="$INTERVAL_MS" 'BEGIN { printf "%.3f", ms / 1000.0 }')

# How many sample iterations fit in the window. Computed up front so the
# inner loop never has to fork to check time.
ITERATIONS=$(( DUR_MS / INTERVAL_MS ))
[ "$ITERATIONS" -lt 1 ] && ITERATIONS=1

# Read /proc/uptime once via awk to get a monotonic-ns timestamp.
mono_ns() {
  awk '{
    split($1, p, ".");
    secs = p[1] + 0;
    frac = (length(p[2]) ? p[2] : "0");
    while (length(frac) < 9) frac = frac "0";
    frac = substr(frac, 1, 9) + 0;
    printf "%d\n", secs * 1000000000 + frac;
  }' /proc/uptime
}

# Snapshot all per-tid {comm,stat,schedstat,status.ctxt_switches} in a
# SINGLE awk pass over the relevant files. Avoids ~5 forks per thread.
# Output layout: one record per file because awk can't directly join
# multiple files per tid; we emit a tag identifying which field came
# from which file, and Python re-joins by tid.
snap() {
  tag="$1"
  awk -v tag="$tag" '
    BEGIN { OFS = "|" }
    FNR == 1 {
      # FILENAME is /proc/PID/task/TID/<comm|stat|schedstat|status>
      n = split(FILENAME, parts, "/")
      tid = parts[n-1]
      kind = parts[n]
      payload = $0
    }
    # status is multi-line; collect just the two ctxt-switch counters.
    FILENAME ~ /\/status$/ {
      if (/^voluntary_ctxt_switches:/)   vol[tid] = $2
      if (/^nonvoluntary_ctxt_switches:/) nv[tid]  = $2
      next
    }
    FNR == 1 {
      val[tid, kind] = payload
      tids[tid] = 1
      next
    }
    END {
      for (t in tids) {
        v = (t in vol) ? vol[t] : "0"
        nvv = (t in nv) ? nv[t] : "0"
        print tag, t, val[t, "comm"], val[t, "stat"], \
              val[t, "schedstat"], v, nvv
      }
    }
  ' /proc/$PID/task/*/comm \
     /proc/$PID/task/*/stat \
     /proc/$PID/task/*/schedstat \
     /proc/$PID/task/*/status \
     2>/dev/null
}

echo "PID=$PID"
echo "MONO_START=$(mono_ns)"
snap A

# Sampling loop — ONE awk reading every wchan file per iteration.
# That's a single fork per sample regardless of how many threads exist.
i=0
while [ "$i" -lt "$ITERATIONS" ]; do
  awk '
    FNR == 1 {
      n = split(FILENAME, p, "/")
      # Skip the "0" wchan (running threads) — that is not a sleep site.
      if ($0 != "0" && $0 != "") {
        print "W|" p[n-1] "|" $0
      }
    }
  ' /proc/$PID/task/*/wchan 2>/dev/null
  sleep "$SLEEP_ARG" 2>/dev/null || sleep 1
  i=$((i + 1))
done

echo "MONO_END=$(mono_ns)"
snap B
echo "DONE"
"""


@mcp.tool()
def thread_wait_profile(duration: int = 5, target: str = "emu",
                        sample_interval_ms: int = 50,
                        min_active_pct: float = 0.5,
                        top_n: int = 25,
                        include_idle_summary: bool = True) -> str:
    """
    Per-thread CPU + runqueue-wait + sleep profiler.

    Samples /proc/<pid>/task/<tid>/{stat,schedstat,wchan,status} over a
    `duration`-second window. For each thread reports:

      * CPU%   — fraction of wall the thread ran on a CPU
                 (from schedstat run_time delta)
      * RunQ%  — fraction it was runnable but waiting in the runqueue
                 (priority inversion / core oversubscription if high)
      * Sleep% — fraction it was blocked sleeping (the remainder)
      * vCS/s  — voluntary context switches per second
                 (high = the thread blocks itself a lot on a lock / cv / IO)
      * iCS/s  — involuntary context switches per second
                 (high = scheduler is preempting it)
      * wchan top — most common kernel functions it was sleeping in
                    (futex_wait_queue_me => mutex/cond; do_nanosleep => sleep;
                     poll_schedule_timeout => poll/epoll; io_schedule => disk)

    Threads with CPU% < min_active_pct AND Sleep% < 10 are folded into an
    "idle" bucket so the output focuses on the active workers.

    Args:
      duration: window length in seconds (default 5).
      target: 'emu' (default :xemu emulation process), 'ui' (launcher), or
              a literal numeric PID string.
      sample_interval_ms: how often (ms) to sample wchan inside the
              window. 50ms gives ~20Hz; reduce for finer wchan
              resolution at the cost of more measurement perturbation.
      min_active_pct: threshold below which a thread is considered idle
              and folded.
      top_n: limit on number of active threads printed (sorted by
              CPU% + Sleep% desc).
      include_idle_summary: print a 1-line summary of folded idle threads.

    Useful for:
      * vCPU stall analysis: a vCPU thread showing high Sleep% + wchan
        in futex_wait means it's blocked on a mutex (likely the BQL or
        a device lock). High RunQ% means it's getting starved by the
        scheduler (priority inversion).
      * Lock contention: many threads sleeping in futex_wait => find
        the holder.
      * Audio sync issues: SDLAudioP2 with high Sleep% + wchan in
        skb_recv or do_nanosleep => producer not keeping up.
    """
    if not _device():
        return "No device connected."
    target = target.strip()
    if target.isdigit():
        pid = target
    elif target == "ui":
        pid = _ui_pid()
    else:
        pid = _emu_pid()
    if not pid:
        return f"target '{target}' not running"

    duration = max(1, int(duration))
    interval = max(5, int(sample_interval_ms))
    dur_ms = duration * 1000

    # Execute the sampler on-device in a single adb shell call. We need
    # the script's positional $1/$2/$3 — wrap it with a leading `set --`.
    # Subprocess timeout is `duration + 10s` so the host can never hang
    # forever even if the on-device script gets wedged.
    script = f"set -- {pid} {dur_ms} {interval}\n{_THREAD_PROFILE_SCRIPT}"
    try:
        r = subprocess.run(
            ["adb", "exec-out", "sh", "-c", script],
            capture_output=True, text=True, check=False,
            timeout=duration + 10,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or b"").decode("utf-8", errors="replace")
        return (
            f"profile timed out after {duration + 10}s. "
            f"Partial output ({len(partial)} bytes):\n"
            + partial[-1000:]
        )
    raw = r.stdout
    if not raw.strip():
        err = r.stderr.strip() or "(no output)"
        return f"profile failed: {err}"

    # Parse output.
    snaps: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    wchan_samples: dict[str, dict[str, int]] = {}  # tid -> {wchan: count}
    comm_map: dict[str, str] = {}
    mono_start_ns = mono_end_ns = None
    sample_count = 0

    for line in raw.splitlines():
        if line.startswith("MONO_START="):
            mono_start_ns = int(line.split("=", 1)[1])
            continue
        if line.startswith("MONO_END="):
            mono_end_ns = int(line.split("=", 1)[1])
            continue
        if line.startswith("PID="):
            continue
        if line.startswith("ERR"):
            return f"profile failed: {line}"

        parts = line.split("|")
        if not parts:
            continue

        kind = parts[0]
        if kind in ("A", "B") and len(parts) >= 7:
            tid, comm, stat, sched, vol, nv = parts[1:7]
            stat_fields = stat.split()
            # /proc/.../stat: field 14 = utime, 15 = stime (1-indexed).
            try:
                utime_ticks = int(stat_fields[13])
                stime_ticks = int(stat_fields[14])
                last_cpu = int(stat_fields[38]) if len(stat_fields) > 38 else -1
            except (IndexError, ValueError):
                utime_ticks = stime_ticks = 0
                last_cpu = -1
            sc_fields = sched.split()
            try:
                run_time_ns = int(sc_fields[0])
                run_delay_ns = int(sc_fields[1])
            except (IndexError, ValueError):
                run_time_ns = run_delay_ns = 0
            try:
                vol_i = int(vol); nv_i = int(nv)
            except ValueError:
                vol_i = nv_i = 0
            snaps[kind][tid] = {
                "utime": utime_ticks,
                "stime": stime_ticks,
                "run_time_ns": run_time_ns,
                "run_delay_ns": run_delay_ns,
                "vol": vol_i,
                "nv": nv_i,
                "last_cpu": last_cpu,
            }
            comm_map[tid] = comm or "?"

        elif kind == "W" and len(parts) >= 3:
            tid = parts[1]
            wchan = parts[2] or "-"
            wchan_samples.setdefault(tid, {})
            wchan_samples[tid][wchan] = wchan_samples[tid].get(wchan, 0) + 1
            if tid in (next(iter(wchan_samples)),):  # noqa
                pass

    if mono_start_ns is None or mono_end_ns is None:
        return f"profile failed: missing time markers\nraw:\n{raw[:500]}"
    wall_ns = mono_end_ns - mono_start_ns
    if wall_ns <= 0:
        return "profile failed: zero-duration window"
    # Sample count is per-tid; just take the max as the loop iteration count.
    if wchan_samples:
        sample_count = max(sum(d.values()) for d in wchan_samples.values())

    # Build per-tid rows from threads present in BOTH snapshots.
    rows = []
    for tid, a in snaps["A"].items():
        b = snaps["B"].get(tid)
        if not b:
            continue
        run_time_delta = max(0, b["run_time_ns"] - a["run_time_ns"])
        run_delay_delta = max(0, b["run_delay_ns"] - a["run_delay_ns"])
        sleep_ns = max(0, wall_ns - run_time_delta - run_delay_delta)
        vol_delta = max(0, b["vol"] - a["vol"])
        nv_delta = max(0, b["nv"] - a["nv"])

        cpu_pct = run_time_delta / wall_ns * 100.0
        runq_pct = run_delay_delta / wall_ns * 100.0
        sleep_pct = sleep_ns / wall_ns * 100.0
        wsec = wall_ns / 1e9

        # Build wchan top-3
        wmap = wchan_samples.get(tid, {})
        wtop = sorted(wmap.items(), key=lambda kv: -kv[1])[:3]
        wtot = sum(wmap.values()) or 1
        wstr = ", ".join(f"{w}:{c * 100 // wtot}%" for w, c in wtop) or "-"

        rows.append({
            "tid": tid, "comm": comm_map.get(tid, "?"),
            "cpu_pct": cpu_pct, "runq_pct": runq_pct, "sleep_pct": sleep_pct,
            "vol_cs_ps": vol_delta / wsec, "nv_cs_ps": nv_delta / wsec,
            "last_cpu": b.get("last_cpu", -1),
            "wchan": wstr,
        })

    # Sort: active first by (CPU% + Sleep%), then drop idle.
    active = [r for r in rows
              if r["cpu_pct"] >= min_active_pct or r["sleep_pct"] >= 10.0]
    idle = [r for r in rows if r not in active]
    active.sort(key=lambda r: (r["cpu_pct"] + r["sleep_pct"]), reverse=True)
    active = active[:top_n]

    lines = []
    lines.append(
        f"pid={pid}  window={wall_ns / 1e9:.2f}s  "
        f"wchan_samples={sample_count}  "
        f"threads_active={len(rows) - len(idle)}/{len(rows)}"
    )
    if sample_count == 0:
        lines.append(
            "[!] wchan_top is empty: Android masks /proc/<tid>/wchan to '0' "
            "for non-root readers (kptr_restrict). CPU%/RunQ%/Sleep% and "
            "ctxt-switch rates are still valid. For per-thread blocking "
            "stacks, capture off-CPU samples with simpleperf or `adb root`."
        )
    lines.append("")
    lines.append(
        f"{'TID':>6}  {'COMM':<16}  {'CPU%':>5}  {'RunQ%':>5}  "
        f"{'Sleep%':>6}  {'vCS/s':>6}  {'iCS/s':>6}  cpu  wchan_top3"
    )
    lines.append("-" * 96)
    for r in active:
        lines.append(
            f"{r['tid']:>6}  {r['comm'][:16]:<16}  "
            f"{r['cpu_pct']:5.1f}  {r['runq_pct']:5.1f}  "
            f"{r['sleep_pct']:6.1f}  {r['vol_cs_ps']:6.0f}  {r['nv_cs_ps']:6.0f}  "
            f"{r['last_cpu']:>3}  {r['wchan']}"
        )

    if include_idle_summary and idle:
        idle_cpu_sum = sum(r["cpu_pct"] for r in idle)
        lines.append("")
        lines.append(
            f"+ {len(idle)} idle thread(s) folded "
            f"(total CPU%={idle_cpu_sum:.2f})"
        )

    # Aggregate wchan across all active threads — a quick "where is the
    # process collectively blocking?" view.
    if any(r["sleep_pct"] > 5.0 for r in active):
        agg: dict[str, int] = {}
        for tid, wmap in wchan_samples.items():
            if tid not in {r["tid"] for r in active}:
                continue
            for w, c in wmap.items():
                if w == "0" or w == "-":
                    continue
                agg[w] = agg.get(w, 0) + c
        if agg:
            total = sum(agg.values())
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
            lines.append("")
            lines.append("Aggregate wchan (active threads only, excludes running):")
            for w, c in top:
                lines.append(f"  {c * 100 / total:5.1f}%  {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logs (xemu native + UI + logcat capture)
# ---------------------------------------------------------------------------

LOGS = {
    "xemu":    XEMU_DEBUG_LOG,
    "ui":      UI_DEBUG_LOG,
    "logcat":  XEMU_LOGCAT_LOG,
    "uilogcat": UI_LOGCAT_LOG,
}

def _resolve_log(name: str) -> str | None:
    return LOGS.get(name.lower())

@mcp.tool()
def enable_debug_logs(enabled: bool = True) -> str:
    """Toggle the in-app debug log capture (setting_debug_logs_enabled in prefs).
    Off by default; flips on the file-based UI/native/logcat capture in DebugLog.kt.
    Requires a running app for the broadcast path; falls back to a direct prefs
    edit (debug build only).
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return ("Cannot edit shared_prefs without run-as. "
                "Toggle via Settings > Debug logs in the app, or install the debug build.")
    # Direct prefs edit while the app is stopped.
    _sh(f"am force-stop {PKG}")
    val = "true" if enabled else "false"
    sed = (
        f"sed -i -E "
        f"'s#<boolean name=\"setting_debug_logs_enabled\" value=\"[^\"]*\" ?/>#"
        f"<boolean name=\"setting_debug_logs_enabled\" value=\"{val}\" />#' "
        f"{PREFS_XML}"
    )
    out = _sh(f"run-as {PKG} sh -c '{sed} 2>&1; grep setting_debug_logs_enabled {PREFS_XML} || echo MISSING'")
    if "MISSING" in out:
        # Insert the entry before </map>
        ins = (
            f"sed -i 's#</map>#"
            f"<boolean name=\"setting_debug_logs_enabled\" value=\"{val}\" />\\n</map>#' "
            f"{PREFS_XML}"
        )
        out = _sh(f"run-as {PKG} sh -c '{ins} 2>&1'")
    return f"setting_debug_logs_enabled={val}\n{out}"

@mcp.tool()
def pull_log(name: str = "xemu", tail_lines: int = 0, grep: str = "") -> str:
    """Pull one of the on-device log files.

    name:       'xemu' (native xemu.log) | 'ui' | 'logcat' (xemu process logcat capture)
                | 'uilogcat'
    tail_lines: keep only the last N lines (0 = all)
    grep:       case-insensitive substring filter
    """
    if not _device(): return "No device connected."
    path = _resolve_log(name)
    if not path: return f"Unknown log '{name}'. Options: {', '.join(LOGS)}"
    tmp = _pull_to_tmp(path)
    if not tmp:
        return (f"Log {path} not found or unreadable. "
                f"Enable via enable_debug_logs(True) and reproduce.")
    try:
        with open(tmp) as f:
            lines = f.readlines()
    finally:
        os.unlink(tmp)
    if grep:
        lines = [l for l in lines if grep.lower() in l.lower()]
    if tail_lines > 0:
        lines = lines[-tail_lines:]
    return "".join(lines) or "(no matching lines)"

@mcp.tool()
def search_log(pattern: str, name: str = "xemu", context_lines: int = 2, max_matches: int = 50) -> str:
    """Regex-search a log file with surrounding context."""
    if not _device(): return "No device connected."
    path = _resolve_log(name)
    if not path: return f"Unknown log '{name}'. Options: {', '.join(LOGS)}"
    tmp = _pull_to_tmp(path)
    if not tmp: return "Log not found on device."
    try:
        with open(tmp) as f:
            lines = f.readlines()
    finally:
        os.unlink(tmp)
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Bad regex: {e}"
    out, count, n, matched = [], 0, len(lines), set()
    for i, l in enumerate(lines):
        if rx.search(l):
            if count >= max_matches: break
            count += 1
            lo = max(0, i - context_lines)
            hi = min(n, i + context_lines + 1)
            if any(j not in matched for j in range(lo, hi)):
                out.append(f"--- match #{count} (lines {lo+1}-{hi}) ---")
                for j in range(lo, hi):
                    marker = ">>>" if j == i else "   "
                    out.append(f"{marker} {j+1:5d}: {lines[j].rstrip()}")
            matched.update(range(lo, hi))
    return "\n".join(out) or "No matches found."

@mcp.tool()
def clear_log(name: str = "all") -> str:
    """Truncate one or all debug-log files. name: 'xemu'|'ui'|'logcat'|'uilogcat'|'all'."""
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return "run-as failed. Install the debug build to clear internal logs."
    targets = list(LOGS.values()) if name.lower() == "all" else [_resolve_log(name)]
    if any(t is None for t in targets):
        return f"Unknown log '{name}'. Options: {', '.join(LOGS)}, all"
    cmds = " && ".join(f": > {t} 2>/dev/null || rm -f {t}" for t in targets)
    _sh(f"run-as {PKG} sh -c '{cmds}'")
    return f"Cleared: {', '.join(targets)}"

@mcp.tool()
def watch_log(pattern: str, name: str = "xemu", timeout_secs: int = 30, poll_interval: float = 0.5) -> str:
    """Poll a log until a regex pattern appears (or timeout)."""
    if not _device(): return "No device connected."
    path = _resolve_log(name)
    if not path: return f"Unknown log '{name}'. Options: {', '.join(LOGS)}"
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Bad regex: {e}"
    deadline = time.monotonic() + timeout_secs
    seen = 0
    while time.monotonic() < deadline:
        tmp = _pull_to_tmp(path)
        if tmp:
            try:
                with open(tmp) as f:
                    lines = f.readlines()
            finally:
                os.unlink(tmp)
            new = lines[seen:]
            seen = len(lines)
            hits = [l.rstrip() for l in new if rx.search(l)]
            if hits:
                return "Match:\n" + "\n".join(hits)
        time.sleep(poll_interval)
    return f"Timed out after {timeout_secs}s waiting for '{pattern}'."

@mcp.tool()
def logcat(lines: int = 200, tag: str = "", grep: str = "") -> str:
    """Dump recent logcat output filtered to X1 BOX-relevant traffic.

    tag:  Android log tag filter (e.g. 'xemu-android', 'Vulkan', 'libxemu')
    grep: additional case-insensitive substring filter
    """
    if not _device(): return "No device connected."
    cmd = ["adb", "logcat", "-d", "-t", str(lines * 3)]
    if tag: cmd += ["-s", tag]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = r.stdout.splitlines()
    if grep:
        out = [l for l in out if grep.lower() in l.lower()]
    return "\n".join(out[-lines:]) or "(no output)"


# ---------------------------------------------------------------------------
# Cache / config / save-state management
# ---------------------------------------------------------------------------

@mcp.tool()
def show_config() -> str:
    """Print the active xemu.toml from the device."""
    if not _device(): return "No device connected."
    return _sh(f"cat {CONFIG_TOML}") or "(no config or unreadable)"

@mcp.tool()
def push_config(local_path: str) -> str:
    """Replace the on-device xemu.toml with a local file."""
    if not _device(): return "No device connected."
    if not os.path.isfile(local_path): return f"Not found locally: {local_path}"
    r = _adb("push", local_path, CONFIG_TOML)
    return (r.stdout + r.stderr).strip() or "Config pushed."

@mcp.tool()
def list_state(kind: str = "all") -> str:
    """List on-device emulator state for inspection or cleanup.

    kind: 'all' (default), 'config' (xemu.toml), 'logs', 'shader_cache',
          'snapshots' (x1box/snapshots in internal files dir), 'gpu_driver'
    """
    if not _device(): return "No device connected."
    def ls(path: str, via_run_as: bool) -> str:
        if via_run_as:
            return _sh(f"run-as {PKG} ls -la {path.replace(INT_FILES_DIR + '/', 'files/')} 2>&1")
        return _sh(f"ls -la {path} 2>&1")
    parts = []
    if kind in ("all", "config"):
        parts.append("=== xemu.toml ===\n" + _sh(f"ls -la {CONFIG_TOML} 2>&1"))
        parts.append("=== x1box dir (external) ===\n" + _sh(f"ls -la {EXT_BASE} 2>&1"))
    if kind in ("all", "logs"):
        parts.append("=== debug-logs ===\n" + ls(LOG_DIR, True))
    if kind in ("all", "shader_cache"):
        parts.append("=== shader/cache dir ===\n" + ls(SHADER_CACHE_DIR, True))
    if kind in ("all", "snapshots"):
        parts.append("=== snapshots ===\n" + ls(f"{INT_FILES_DIR}/x1box/snapshots", True))
    if kind in ("all", "gpu_driver"):
        parts.append("=== gpu_driver ===\n" + ls(f"{INT_FILES_DIR}/gpu_driver", True))
    return "\n\n".join(parts)

@mcp.tool()
def clear_cache(kind: str = "shader") -> str:
    """Wipe a cache or state category from the device.

    kind: 'shader' (app cache dir), 'logs' (all debug-log files),
          'snapshots' (save-state previews / files in x1box/snapshots),
          'covers' (downloaded_covers + custom_covers), 'disc_format_cache'
    """
    if not _device(): return "No device connected."
    if not _is_debuggable() and kind not in ("snapshots", "logs"):
        return "run-as failed. Install the debug build to clear internal data."
    targets = {
        "shader":             ["cache"],
        "logs":               ["files/x1box/debug-logs/*"],
        "snapshots":          ["files/x1box/snapshots/*"],
        "covers":             ["files/downloaded_covers/*", "files/custom_covers/*"],
        "disc_format_cache":  ["files/disc_format_cache.tsv"],
    }
    if kind not in targets:
        return f"Unknown kind '{kind}'. Options: {', '.join(targets)}"
    cmds = " ; ".join(f"rm -rf {t}" for t in targets[kind])
    out = _sh(f"run-as {PKG} sh -c '{cmds} 2>&1; echo done'")
    return f"Cleared {kind}: {out}"


# ---------------------------------------------------------------------------
# Build / install
# ---------------------------------------------------------------------------

@mcp.tool()
def install_apk(apk_path: str = "", variant: str = "debug") -> str:
    """Install an X1 BOX APK. With no apk_path, picks the most recent build
    of the requested variant from android/app/build/outputs/apk/<variant>/.
    """
    if not _device(): return "No device connected."
    v = (variant or "debug").lower()
    if v not in ("debug", "release"):
        return f"Unknown variant '{variant}' (expected 'debug' or 'release')."
    if not apk_path:
        apk = _resolve_apk(v)
        if not apk: return f"No APK in {APK_OUT_DIR / v}. Build first."
        apk_path = str(apk)
    if not os.path.exists(apk_path):
        return f"APK not found: {apk_path}"
    r = _adb("install", "-r", apk_path)
    return (r.stdout + r.stderr).strip() or f"Installed ({v} -> {apk_path})."

@mcp.tool()
def build_and_install(variant: str = "debug") -> str:
    """Run `./gradlew assemble<Variant>` in android/, then install the APK."""
    if not _device(): return "No device connected."
    v = (variant or "debug").lower()
    if v not in ("debug", "release"):
        return f"Unknown variant '{variant}' (expected 'debug' or 'release')."
    task = f"assemble{v.capitalize()}"
    r = subprocess.run(
        ["./gradlew", task],
        cwd=ANDROID_ROOT, capture_output=True, text=True, check=False,
    )
    tail = (r.stdout + r.stderr)[-2000:]
    if r.returncode != 0:
        return f"Build FAILED ({task}):\n{tail}"
    return install_apk(variant=v) + f"\n\nBuild log (tail):\n{tail}"


# ---------------------------------------------------------------------------
# File transfer / arbitrary shell
# ---------------------------------------------------------------------------

@mcp.tool()
def shell(command: str) -> str:
    """Run an arbitrary adb shell command. Combine with run-as <pkg> sh -c '...'
    when you need access to the app's private data dir."""
    if not _device(): return "No device connected."
    return _sh(command)

@mcp.tool()
def push_file(local_path: str, device_path: str) -> str:
    """Push a local file to a device path (adb push)."""
    if not _device(): return "No device connected."
    if not os.path.exists(local_path): return f"Not found locally: {local_path}"
    r = _adb("push", local_path, device_path)
    return (r.stdout + r.stderr).strip() or "Push complete."

@mcp.tool()
def pull_file(device_path: str, local_path: str) -> str:
    """Pull a device file to a local path. For paths under /data/<pkg>/ use
    run-as via the `shell` tool instead."""
    if not _device(): return "No device connected."
    r = _adb("pull", device_path, local_path)
    return (r.stdout + r.stderr).strip() or "Pull complete."


# ---------------------------------------------------------------------------
# Native profiling (simpleperf)
# ---------------------------------------------------------------------------

@mcp.tool()
def profile(duration: int = 3, tid: str = "", target: str = "emu") -> str:
    """Profile X1 BOX native code with simpleperf and return top hotspots.

    Samples the :xemu emulation process by default (where qemu_main + JIT
    run). Requires the debug build (which is debuggable). Uses frame-pointer
    call-graph sampling so symbols resolve from libxemu.so.

    duration: seconds (1-60, default 3)
    tid:      optional thread id to restrict sampling
    target:   'emu' (default) or 'ui'
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return "run-as failed. Install the debug build to use simpleperf."
    pid = _emu_pid() if target != "ui" else _ui_pid()
    if not pid: return f"{PKG}: not running"

    duration = max(1, min(duration, 60))
    perf_data = f"/data/user/0/{PKG}/simpleperf.data"
    record_args = f"-p {pid} --duration {duration} --call-graph fp -o {perf_data}"
    if tid: record_args += f" -t {tid}"

    record_out = _sh(f"run-as {PKG} simpleperf record {record_args} 2>&1")
    if "error" in record_out.lower() and "simpleperf" not in record_out.lower():
        return f"simpleperf record failed:\n{record_out}"
    report_out = _sh(
        f"run-as {PKG} simpleperf report -i {perf_data} "
        f"--sort symbol --percent-limit 0.1 2>&1 | head -60"
    )
    return (f"=== simpleperf {duration}s profile (pid={pid}{', tid='+tid if tid else ''}) ===\n"
            f"Record: {record_out.strip()}\n\n=== Top symbols ===\n{report_out}")


# ---------------------------------------------------------------------------
# Native debugger bootstrap (lldb-server in the app sandbox)
# ---------------------------------------------------------------------------

@mcp.tool()
def attach_native_debugger(port: int = 5039, lldb_server_path: str = "") -> str:
    """Start lldb-server inside the X1 BOX app sandbox so the host can attach
    over `adb forward`. Use this to set breakpoints in libxemu.so / native code.

    Steps performed:
      1. Push `lldb-server` (from the NDK, if not already present in /data/local/tmp)
      2. `run-as <pkg>` to copy it into the app's data dir (so it has the right SELinux context)
      3. Launch `lldb-server platform --listen \\*:<port>` inside the app sandbox
      4. `adb forward tcp:<port> tcp:<port>` so a host lldb can connect

    lldb_server_path: optional path to a host-side lldb-server arm64 binary
                      (usually $ANDROID_NDK/toolchains/llvm/prebuilt/.../lib/clang/.../lib/linux/aarch64/lldb-server
                      or $ANDROID_NDK/lldb-server/lldb-server). If empty, we
                      try to autodetect.

    Returns the host-side connect instructions on success.
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return "Install the debug build first; release isn't debuggable."

    # If lldb-server is already in /data/local/tmp, skip the push step.
    have_remote = _sh("ls -la /data/local/tmp/lldb-server 2>/dev/null | wc -l").strip()
    if have_remote == "0":
        # Try to locate a host copy.
        candidates: list[Path] = []
        if lldb_server_path:
            candidates.append(Path(lldb_server_path))
        ndk = os.environ.get("ANDROID_NDK") or os.environ.get("ANDROID_NDK_ROOT")
        if ndk:
            ndk_path = Path(ndk)
            candidates += list(ndk_path.glob(
                "toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/lldb-server"
            ))
            candidates += list(ndk_path.glob(
                "toolchains/llvm/prebuilt/*/lib64/clang/*/lib/linux/aarch64/lldb-server"
            ))
        candidates += [Path("/opt/android-sdk/ndk").glob("*/toolchains/llvm/prebuilt/*/lib/clang/*/lib/linux/aarch64/lldb-server")] \
            if Path("/opt/android-sdk/ndk").is_dir() else []
        # Flatten (some entries are generators)
        resolved: list[Path] = []
        for c in candidates:
            if isinstance(c, Path):
                if c.is_file(): resolved.append(c)
            else:
                resolved.extend(p for p in c if p.is_file())
        if not resolved:
            return ("Could not locate host lldb-server. Set ANDROID_NDK or pass "
                    "lldb_server_path=/abs/path. Example: $ANDROID_NDK/toolchains/"
                    "llvm/prebuilt/linux-x86_64/lib/clang/.../lib/linux/aarch64/lldb-server")
        src = resolved[0]
        r = _adb("push", str(src), "/data/local/tmp/lldb-server")
        if r.returncode != 0:
            return f"Failed to push lldb-server: {(r.stdout+r.stderr).strip()}"
        _sh("chmod 755 /data/local/tmp/lldb-server")

    # Stage inside the app sandbox so SELinux lets us trace the app's PIDs.
    _sh(
        f"run-as {PKG} sh -c "
        f"'cp /data/local/tmp/lldb-server ./lldb-server 2>/dev/null; "
        f"chmod 755 ./lldb-server'"
    )

    # Kill any prior listener.
    _sh(f"run-as {PKG} sh -c 'pkill -f lldb-server 2>/dev/null; true'")

    # Launch in background. Use exec-out + nohup so it survives the adb exit.
    subprocess.Popen(
        ["adb", "shell",
         f"run-as {PKG} sh -c 'nohup ./lldb-server platform --listen \"*:{port}\" "
         f">/dev/null 2>&1 &'"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.4)

    # Forward the port to the host.
    fwd = _adb("forward", f"tcp:{port}", f"tcp:{port}")

    pid_emu = _emu_pid() or "(launch a game first)"
    return (
        f"lldb-server running in {PKG} sandbox on port {port}.\n"
        f"adb forward: {(fwd.stdout+fwd.stderr).strip() or 'ok'}\n\n"
        f"Connect from host lldb:\n"
        f"  $ lldb\n"
        f"  (lldb) platform select remote-android\n"
        f"  (lldb) platform connect connect://localhost:{port}\n"
        f"  (lldb) attach --pid {pid_emu}\n\n"
        f"To set a breakpoint in xemu source before attach:\n"
        f"  (lldb) target create --no-dependents /path/to/libxemu.so\n"
        f"  (lldb) breakpoint set --name <symbol>\n"
        f"Then attach as above."
    )

@mcp.tool()
def detach_native_debugger(port: int = 5039) -> str:
    """Kill the in-sandbox lldb-server and drop the adb-forward entry."""
    if not _device(): return "No device connected."
    _sh(f"run-as {PKG} sh -c 'pkill -f lldb-server 2>/dev/null; true'")
    _adb("forward", "--remove", f"tcp:{port}")
    return f"lldb-server stopped, tcp:{port} forward removed."


# ---------------------------------------------------------------------------
# QEMU gdbstub control (real guest-CPU debugging over GDB RSP)
#
# Wiring:
#   1. xemu_android.cpp picks up XEMU_ANDROID_GDB_PORT from env and appends
#      `-gdb tcp::PORT` to QEMU's argv. With XEMU_ANDROID_GDB_PAUSE=1 it also
#      appends `-S` (halt at boot).
#   2. The env_vars SharedPreference (read in SyncSetupFiles) is the channel
#      for setenv on the device side. gdb_enable() writes the right entries
#      into x1box_prefs.xml under run-as.
#   3. After a game launch starts QEMU, gdb_connect() runs `adb forward
#      tcp:PORT tcp:PORT` and opens a local TCP socket. All subsequent
#      gdb_* tools talk Remote Serial Protocol over that socket.
#
# Target: xemu emulates an Xbox CPU (i386). The legacy `g` packet returns
# 16 32-bit registers in order:
#   eax, ecx, edx, ebx, esp, ebp, esi, edi, eip, eflags, cs, ss, ds, es, fs, gs
# ---------------------------------------------------------------------------

GDB_PREF_PORT_KEY = "XEMU_ANDROID_GDB_PORT"
GDB_PREF_PAUSE_KEY = "XEMU_ANDROID_GDB_PAUSE"

I386_REG_NAMES = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
    "eip", "eflags", "cs", "ss", "ds", "es", "fs", "gs",
]


class _GdbClient:
    """Minimal GDB Remote Serial Protocol client for QEMU's gdbstub.

    Persistent across MCP calls — instantiated once, stays connected until
    gdb_disconnect() or process exit.
    """

    def __init__(self) -> None:
        self.sock: socket.socket | None = None
        self.port: int = 0
        self.bps: set[int] = set()       # client-tracked software breakpoints
        self.last_stop: str = ""         # last stop-reply payload

    # --- low-level packet I/O ------------------------------------------------

    @staticmethod
    def _checksum(data: bytes) -> bytes:
        return f"{sum(data) & 0xff:02x}".encode()

    def _send(self, payload: str) -> None:
        if not self.sock:
            raise RuntimeError("gdb not connected — call gdb_connect() first")
        body = payload.encode()
        pkt = b"$" + body + b"#" + self._checksum(body)
        self.sock.sendall(pkt)
        # Wait for ack ('+' or '-' resend).
        ack = self.sock.recv(1)
        if ack == b"-":
            self.sock.sendall(pkt)
            self.sock.recv(1)

    def _recv(self, timeout: float = 5.0) -> str:
        """Receive one packet, ack it, return its payload string."""
        if not self.sock:
            raise RuntimeError("gdb not connected")
        self.sock.settimeout(timeout)
        buf = bytearray()
        state = "start"
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("gdb peer closed connection")
            buf.extend(chunk)
            while buf:
                if state == "start":
                    i = buf.find(b"$")
                    if i < 0:
                        buf.clear()
                        break
                    del buf[:i + 1]
                    payload = bytearray()
                    state = "body"
                if state == "body":
                    j = buf.find(b"#")
                    if j < 0:
                        payload.extend(buf)
                        buf.clear()
                        break
                    payload.extend(buf[:j])
                    del buf[:j + 1]
                    if len(buf) < 2:
                        # need 2 hex chars of cksum
                        more = self.sock.recv(2 - len(buf))
                        if not more:
                            raise RuntimeError("gdb peer closed connection")
                        buf.extend(more)
                    # consume checksum, ack, return payload
                    del buf[:2]
                    self.sock.sendall(b"+")
                    return payload.decode("latin-1")

    def _cmd(self, payload: str, timeout: float = 5.0) -> str:
        self._send(payload)
        return self._recv(timeout)

    # --- lifecycle -----------------------------------------------------------

    def connect(self, port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> str:
        if self.sock:
            self.disconnect()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.settimeout(None)
        self.sock = s
        self.port = port
        # Negotiate multiprocess+ so vCont thread specs and stop replies use
        # the modern pPID.TID form QEMU expects from a multiprocess-aware peer.
        try:
            self._cmd("qSupported:multiprocess+;swbreak+;hwbreak+;qRelocInsn+",
                      timeout=timeout)
        except Exception:
            pass
        reply = self._cmd("?", timeout=timeout)
        self.last_stop = reply
        # Parse "Txx[thread:<tid>;...]" or "Sxx" into a thread id to select
        # for subsequent `g`/`p`/`m`/etc. Without this, QEMU's modern
        # gdbstub returns empty for register reads.
        self._select_default_thread(reply, timeout=timeout)
        return f"connected tcp:{port}, initial stop: {reply}"

    def _select_default_thread(self, stop_reply: str, timeout: float = 5.0) -> None:
        """Find the current thread id in a stop reply and Hg-select it."""
        # Stop reply may be "T05thread:p01.01;..." (multiprocess+) or
        # "T05thread:01;..." (legacy). Extract the thread id verbatim.
        tid: str | None = None
        if stop_reply.startswith("T") and len(stop_reply) >= 3:
            for token in stop_reply[3:].split(";"):
                if token.startswith("thread:"):
                    tid = token.split(":", 1)[1]
                    break
        if not tid:
            # Fall back to qC (current thread).
            try:
                qc = self._cmd("qC", timeout=timeout)
                if qc.startswith("QC"):
                    tid = qc[2:]
            except Exception:
                return
        if not tid:
            return
        # `Hg<tid>` selects the thread for "general operations" (g, p, m, etc.).
        # QEMU rejects a space between Hg and the tid (returns E22), so build
        # the packet directly.
        try:
            self._cmd(f"Hg{tid}", timeout=timeout)
        except Exception:
            pass

    def disconnect(self) -> str:
        if not self.sock:
            return "already disconnected"
        try:
            self._send("D")
            self._recv(timeout=1.0)
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None
        return "disconnected"

    # --- execution control ---------------------------------------------------

    def interrupt(self) -> str:
        """Send Ctrl-C (\\x03) to halt the guest. Reply is a stop packet."""
        if not self.sock:
            raise RuntimeError("gdb not connected")
        self.sock.sendall(b"\x03")
        reply = self._recv(timeout=5.0)
        self.last_stop = reply
        return reply

    def cont(self, wait: bool = False, timeout: float = 5.0) -> str:
        """Continue. If wait=True, block until guest stops again."""
        self._send("c")
        if not wait:
            return "running"
        try:
            reply = self._recv(timeout=timeout)
        except socket.timeout:
            return "running (no stop within timeout)"
        self.last_stop = reply
        return reply

    def step(self) -> str:
        """Single instruction step. Returns stop reply."""
        reply = self._cmd("s")
        self.last_stop = reply
        return reply

    def status(self) -> str:
        reply = self._cmd("?")
        self.last_stop = reply
        return reply

    # --- breakpoints ---------------------------------------------------------

    def set_breakpoint(self, addr: int, kind: int = 1) -> str:
        """Z0,<addr>,<kind> — software breakpoint via INT3."""
        reply = self._cmd(f"Z0,{addr:x},{kind}")
        if reply.upper() == "OK":
            self.bps.add(addr)
            return f"breakpoint set @ 0x{addr:x}"
        return f"failed: {reply or '(empty reply, gdbstub may not support swbp)'}"

    def clear_breakpoint(self, addr: int, kind: int = 1) -> str:
        reply = self._cmd(f"z0,{addr:x},{kind}")
        if reply.upper() == "OK":
            self.bps.discard(addr)
            return f"breakpoint cleared @ 0x{addr:x}"
        return f"failed: {reply}"

    def set_watchpoint(self, addr: int, length: int, kind: str = "w") -> str:
        """Z2/Z3/Z4 — write/read/access watchpoint (hardware)."""
        z = {"w": 2, "r": 3, "a": 4}.get(kind)
        if z is None:
            return f"bad kind {kind!r} (use 'w', 'r', or 'a')"
        reply = self._cmd(f"Z{z},{addr:x},{length}")
        return f"{kind}-watchpoint @ 0x{addr:x}:{length} -> {reply or '(empty)'}"

    def list_breakpoints(self) -> str:
        if not self.bps:
            return "(no client-tracked breakpoints)"
        return "\n".join(f"  0x{a:x}" for a in sorted(self.bps))

    # --- registers / memory --------------------------------------------------

    def read_registers(self) -> dict[str, int]:
        reply = self._cmd("g")
        if reply.startswith("E"):
            raise RuntimeError(f"gdbstub error: {reply}")
        raw = bytes.fromhex(reply)
        out: dict[str, int] = {}
        # i386 layout: 16 * 4 bytes; if more (FPU/SSE) is returned, ignore the tail.
        for i, name in enumerate(I386_REG_NAMES):
            chunk = raw[i * 4:(i + 1) * 4]
            if len(chunk) < 4:
                break
            out[name] = int.from_bytes(chunk, "little")
        return out

    def write_register(self, name: str, value: int) -> str:
        if name not in I386_REG_NAMES:
            return f"unknown register {name!r}"
        regs = self.read_registers()
        regs[name] = value & 0xffffffff
        raw = b"".join(int(regs[n]).to_bytes(4, "little") for n in I386_REG_NAMES)
        reply = self._cmd("G" + raw.hex())
        return f"wrote {name}=0x{value:x}: {reply or '(no reply)'}"

    def read_mem(self, addr: int, length: int) -> bytes:
        # Chunk to avoid huge packets; QEMU typically caps at PacketSize.
        out = bytearray()
        remaining = length
        cur = addr
        while remaining > 0:
            n = min(remaining, 1024)
            reply = self._cmd(f"m{cur:x},{n:x}")
            if reply.startswith("E") or not reply:
                raise RuntimeError(f"gdbstub error at 0x{cur:x}: {reply or '(empty)'}")
            data = bytes.fromhex(reply)
            out.extend(data)
            if len(data) < n:
                break
            cur += n
            remaining -= n
        return bytes(out)

    def write_mem(self, addr: int, data: bytes) -> str:
        # Same chunking concern.
        cur = addr
        i = 0
        while i < len(data):
            chunk = data[i:i + 1024]
            reply = self._cmd(f"M{cur:x},{len(chunk):x}:" + chunk.hex())
            if reply.upper() != "OK":
                return f"failed at 0x{cur:x}: {reply or '(empty)'}"
            cur += len(chunk)
            i += len(chunk)
        return f"wrote {len(data)} bytes @ 0x{addr:x}"


_GDB = _GdbClient()


def _xml_escape(s: str) -> str:
    return _saxutil.escape(s, {'"': "&quot;"})


def _rewrite_env_vars(set_lines: list[str] | None = None,
                      drop_keys: list[str] | None = None) -> str | None:
    """Edit the `env_vars` <string> entry in x1box_prefs.xml.

    set_lines:  KEY=VALUE lines to add / overwrite (overwrite by KEY=)
    drop_keys:  KEY prefixes to strip from existing lines before adding

    Pulls the prefs file via `run-as cat`, edits it in Python, pushes the
    modified copy to /data/local/tmp, and `cp`s it back under run-as. Returns
    None on success or an error string.
    """
    set_lines = set_lines or []
    drop_keys = drop_keys or []

    # Pull current prefs.
    r = subprocess.run(
        ["adb", "exec-out", f"run-as {PKG} cat {PREFS_XML}"],
        capture_output=True, check=False,
    )
    if r.returncode != 0 or not r.stdout:
        return f"failed to read {PREFS_XML}: {r.stderr.decode(errors='replace').strip()}"
    text = r.stdout.decode("utf-8", errors="replace")

    # Find / decode existing env_vars value.
    m = re.search(r'<string name="env_vars">(.*?)</string>', text, re.DOTALL)
    if m:
        encoded = m.group(1)
        decoded = (encoded.replace("&#10;", "\n").replace("&apos;", "'")
                          .replace("&quot;", '"').replace("&amp;", "&"))
        existing_lines = decoded.splitlines()
    else:
        existing_lines = []

    # Strip drop_keys and any keys we're about to set anyway.
    set_keys = {l.split("=", 1)[0] for l in set_lines}
    kept = [l for l in existing_lines
            if not any(l.startswith(k + "=") for k in drop_keys)
            and l.split("=", 1)[0] not in set_keys]
    new_lines = kept + set_lines
    new_val = "\n".join(new_lines)

    new_entry_body = _xml_escape(new_val).replace("\n", "&#10;")
    new_entry = f'<string name="env_vars">{new_entry_body}</string>'

    if m:
        new_text = text[:m.start()] + new_entry + text[m.end():]
    elif new_val:
        # Insert before </map>.
        new_text = re.sub(r"</map>", new_entry + "\n</map>", text, count=1)
    else:
        # Nothing to do.
        return None

    # Push the modified prefs back.
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(new_text)
        local = f.name
    remote_stage = f"/data/local/tmp/_x1box_prefs_{os.getpid()}.xml"
    try:
        if _adb("push", local, remote_stage).returncode != 0:
            return "failed to stage prefs file"
        out = _sh(
            f"run-as {PKG} sh -c "
            f"'cp {remote_stage} {PREFS_XML} && chmod 660 {PREFS_XML} 2>&1'"
        )
        if out and "error" in out.lower():
            return f"cp into shared_prefs failed: {out}"
    finally:
        os.unlink(local)
        _sh(f"rm -f {remote_stage}")
    return None


@mcp.tool()
def gdb_enable(port: int = 1234, pause_at_boot: bool = False) -> str:
    """Configure xemu to start QEMU's gdbstub on the next launch.

    Writes XEMU_ANDROID_GDB_PORT (and optionally XEMU_ANDROID_GDB_PAUSE=1)
    into the `env_vars` SharedPreference. xemu_android.cpp reads env_vars at
    startup and SDL_main injects `-gdb tcp::PORT` (plus `-S` when pause is on)
    into QEMU's argv.

    Requires the debug build (uses run-as to edit shared_prefs). Stops the
    app first so the new prefs take effect on next launch_app().

    pause_at_boot=True is recommended for cold-boot breakpoints — without it
    the guest hits BIOS code before you can attach.
    """
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return "Install the debug build first (run-as required to edit prefs)."
    if port < 1 or port > 65535:
        return f"Invalid port {port}."

    _sh(f"am force-stop {PKG}")

    extra_lines = [f"{GDB_PREF_PORT_KEY}={port}"]
    if pause_at_boot:
        extra_lines.append(f"{GDB_PREF_PAUSE_KEY}=1")
    err = _rewrite_env_vars(set_lines=extra_lines,
                            drop_keys=[GDB_PREF_PORT_KEY, GDB_PREF_PAUSE_KEY])
    if err: return err

    verify = _sh(
        f"run-as {PKG} sh -c 'grep -o {GDB_PREF_PORT_KEY} {PREFS_XML} | head -1'"
    )
    return (f"gdbstub configured: port={port}, pause_at_boot={pause_at_boot}\n"
            f"Prefs verified: {verify or '(key missing — investigate)'}\n"
            f"Next steps:\n"
            f"  1. launch_app('/sdcard/Games/foo.iso')   # start a game\n"
            f"  2. gdb_connect({port})                   # adb forward + RSP attach\n"
            f"  3. gdb_set_breakpoint('0x...'), gdb_continue(), ...")

@mcp.tool()
def gdb_disable() -> str:
    """Remove the gdbstub env vars from prefs. Launches after this run
    without `-gdb`. Restart the app for it to take effect."""
    if not _device(): return "No device connected."
    if not _is_debuggable():
        return "Debug build required to edit prefs."
    _sh(f"am force-stop {PKG}")
    err = _rewrite_env_vars(drop_keys=[GDB_PREF_PORT_KEY, GDB_PREF_PAUSE_KEY])
    if err: return err
    return "gdbstub disabled in prefs. Next launch will not start the stub."

@mcp.tool()
def gdb_connect(port: int = 1234) -> str:
    """`adb forward tcp:PORT tcp:PORT` then open an RSP session to the gdbstub.
    Run this after launch_app() — the gdbstub binds inside QEMU once
    `qemu_init` parses argv. The first command also issues qSupported and `?`."""
    fwd = _adb("forward", f"tcp:{port}", f"tcp:{port}")
    if fwd.returncode != 0:
        return f"adb forward failed: {(fwd.stdout+fwd.stderr).strip()}"
    try:
        info = _GDB.connect(port)
    except Exception as e:
        return (f"connect failed: {e}\n"
                f"Check: gdb_enable() was called, game is running, "
                f"and `pull_log('xemu', grep='gdbstub')` shows the listener.")
    return info

@mcp.tool()
def gdb_disconnect(port: int = 0) -> str:
    """Close the RSP session. Pass port>0 to also `adb forward --remove`."""
    msg = _GDB.disconnect()
    if port:
        _adb("forward", "--remove", f"tcp:{port}")
        msg += f", tcp:{port} forward removed"
    return msg

@mcp.tool()
def gdb_status() -> str:
    """Send `?` — report current stop reason / signal."""
    if not _GDB.sock: return "not connected"
    return f"stop reply: {_GDB.status()}"

@mcp.tool()
def gdb_break() -> str:
    """Halt the guest CPU (sends \\x03 over RSP). Use before reading state."""
    if not _GDB.sock: return "not connected"
    return f"halted: {_GDB.interrupt()}"

@mcp.tool()
def gdb_continue(wait: bool = False, timeout: float = 5.0) -> str:
    """Resume execution. With wait=True, block until next stop and return it."""
    if not _GDB.sock: return "not connected"
    return _GDB.cont(wait=wait, timeout=timeout)

@mcp.tool()
def gdb_step() -> str:
    """Single-step the guest one instruction."""
    if not _GDB.sock: return "not connected"
    return f"stop after step: {_GDB.step()}"

@mcp.tool()
def gdb_set_breakpoint(address: str, kind: int = 1) -> str:
    """Set a software breakpoint (INT3) at a guest virtual address.

    address: hex address ('0x80012345' or '80012345')
    kind: instruction-length hint (1 is fine for x86)
    """
    if not _GDB.sock: return "not connected"
    try:
        a = int(address, 16)
    except ValueError:
        return f"bad address {address!r}"
    return _GDB.set_breakpoint(a, kind)

@mcp.tool()
def gdb_clear_breakpoint(address: str, kind: int = 1) -> str:
    """Remove a software breakpoint."""
    if not _GDB.sock: return "not connected"
    try:
        a = int(address, 16)
    except ValueError:
        return f"bad address {address!r}"
    return _GDB.clear_breakpoint(a, kind)

@mcp.tool()
def gdb_list_breakpoints() -> str:
    """List client-tracked software breakpoints."""
    return _GDB.list_breakpoints()

@mcp.tool()
def gdb_set_watchpoint(address: str, length: int = 4, kind: str = "w") -> str:
    """Set a hardware watchpoint on guest memory.

    kind: 'w' (write), 'r' (read), or 'a' (access). x86 hw watchpoints
    are limited to 4 slots and lengths of 1/2/4/8.
    """
    if not _GDB.sock: return "not connected"
    try:
        a = int(address, 16)
    except ValueError:
        return f"bad address {address!r}"
    return _GDB.set_watchpoint(a, length, kind)

@mcp.tool()
def gdb_read_registers() -> str:
    """Dump the guest CPU's i386 general-purpose registers + EFLAGS + segs."""
    if not _GDB.sock: return "not connected"
    regs = _GDB.read_registers()
    if not regs: return "no registers returned"
    lines = []
    for name, val in regs.items():
        if name in ("cs", "ss", "ds", "es", "fs", "gs"):
            lines.append(f"  {name:>7} = 0x{val:04x}")
        else:
            lines.append(f"  {name:>7} = 0x{val:08x}  ({val:>10})")
    return "\n".join(lines)

@mcp.tool()
def gdb_write_register(name: str, value: str) -> str:
    """Set a guest register. value is parsed as hex ('0xdeadbeef') or decimal."""
    if not _GDB.sock: return "not connected"
    try:
        v = int(value, 0)
    except ValueError:
        return f"bad value {value!r}"
    return _GDB.write_register(name, v)

@mcp.tool()
def gdb_read_guest_memory(address: str, size: int = 64, fmt: str = "hex") -> str:
    """Read guest virtual memory through the gdbstub.

    Unlike read_memory (which reads the *host* process's address space),
    this goes through QEMU's MMU and sees the guest's view of RAM/MMIO.

    address: hex address (e.g. '0x80012345')
    size: bytes (max 16384)
    fmt: 'hex' | 'u32le' | 'f32le' | 'bytes'
    """
    if not _GDB.sock: return "not connected"
    try:
        a = int(address, 16)
    except ValueError:
        return f"bad address {address!r}"
    size = max(1, min(size, 16384))
    try:
        data = _GDB.read_mem(a, size)
    except Exception as e:
        return f"read failed: {e}"

    if fmt == "hex":
        # 16-byte rows.
        rows = []
        for off in range(0, len(data), 16):
            chunk = data[off:off + 16]
            hexs = " ".join(f"{b:02x}" for b in chunk)
            ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            rows.append(f"  {a + off:08x}: {hexs:<47}  {ascii_}")
        return "\n".join(rows)
    if fmt == "u32le":
        n = len(data) // 4
        vals = struct.unpack_from(f"<{n}I", data)
        return "\n".join(f"  [{i*4:+#06x}]  0x{v:08x}  ({v})" for i, v in enumerate(vals))
    if fmt == "f32le":
        n = len(data) // 4
        vals = struct.unpack_from(f"<{n}f", data)
        return "\n".join(f"  [{i*4:+#06x}]  {v:.6g}" for i, v in enumerate(vals))
    return data.hex()

@mcp.tool()
def gdb_write_guest_memory(address: str, hex_bytes: str) -> str:
    """Poke guest virtual memory through the gdbstub. Pauses the guest
    implicitly via gdb (you should `gdb_break` first if you want a coherent
    write while it's running). Use with care."""
    if not _GDB.sock: return "not connected"
    try:
        a = int(address, 16)
    except ValueError:
        return f"bad address {address!r}"
    clean = re.sub(r"[,\s]", "", hex_bytes)
    if len(clean) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", clean or ""):
        return f"bad hex_bytes {hex_bytes!r}"
    raw = bytes.fromhex(clean)
    if not raw: return "no bytes"
    return _GDB.write_mem(a, raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
