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
import shlex
import socket
import struct
import subprocess
import tempfile
import time
import xml.etree.ElementTree as _ET
import xml.sax.saxutils as _saxutil
from pathlib import Path
from urllib.parse import unquote as _urlunquote
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
SHADER_DUMP_DIR = f"{EXT_BASE}/shader_dump"   # per-pipeline-create SPIR-V/GLSL dump (see draw.c, Android-only)
SHADER_DUMP_LOG_TAG = "hakuX-vk-shaderdump"   # logcat tag the dumper emits per shader

# Arm Performance Studio (malioc) — optional; honoured when set or default exists.
ARM_PERFORMANCE_STUDIO_HOME = os.environ.get(
    "ARM_PERFORMANCE_STUDIO_HOME",
    "/home/xero/repos/Arm_Performance_Studio_2026.2",
)
MALIOC_BIN = f"{ARM_PERFORMANCE_STUDIO_HOME}/mali_offline_compiler/malioc"

# Generic GPU probe runner (see hw/xbox/nv2a/pgraph/vk/probe_runner.{c,h}).
GPU_PROBE_DIR       = f"{EXT_BASE}/gpu_probe"
GPU_PROBE_REQ       = f"{GPU_PROBE_DIR}/probe_req.bin"
GPU_PROBE_REQ_USED  = f"{GPU_PROBE_DIR}/probe_req.consumed"
GPU_PROBE_OUT       = f"{GPU_PROBE_DIR}/probe_out.bin"
GPU_PROBE_DONE      = f"{GPU_PROBE_DIR}/probe_done.flag"
GPU_PROBE_VERSION       = 1
GPU_PROBE_STAGE_COMPUTE = 0
GPU_PROBE_STAGE_FRAGMENT = 1

ANDROID_ROOT = Path(__file__).resolve().parent.parent  # /home/.../x1-box/android
APK_OUT_DIR = ANDROID_ROOT / "app/build/outputs/apk"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Active-device selector. When set, exported as ANDROID_SERIAL so every adb
# subprocess (including the `simpleperf` flow inside profile_native and the
# `adb exec-out run-as` paths in _pull_to_tmp) targets the same device. Set
# via set_active_device(); cleared with set_active_device("").
_ACTIVE_DEVICE: str | None = None


def _sync_active_device_env() -> None:
    if _ACTIVE_DEVICE:
        os.environ["ANDROID_SERIAL"] = _ACTIVE_DEVICE
    else:
        os.environ.pop("ANDROID_SERIAL", None)


def _adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True, check=False)

def _sh(*args: str) -> str:
    r = _adb("shell", *args)
    return (r.stdout + r.stderr).strip()

def _list_attached_devices() -> list[str]:
    lines = _adb("devices").stdout.strip().splitlines()[1:]
    return [l.split()[0] for l in lines if l.strip() and "device" in l]

def _device() -> str | None:
    # If an active device is selected, prefer it (and confirm it's still attached).
    devs = _list_attached_devices()
    if _ACTIVE_DEVICE and _ACTIVE_DEVICE in devs:
        return _ACTIVE_DEVICE
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


# Supported disc-image extensions — mirrors GameLibraryActivity.gameExts.
GAME_EXTS = (".iso", ".xiso", ".cso", ".cci")


def _read_prefs() -> dict[str, str] | None:
    """Return <string> entries from x1box_prefs.xml. Needs the debug build
    (run-as). Returns None if unreadable.
    """
    if not _is_debuggable():
        return None
    r = subprocess.run(
        ["adb", "exec-out", f"run-as {PKG} cat {PREFS_XML}"],
        capture_output=True, check=False,
    )
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        root = _ET.fromstring(r.stdout.decode("utf-8", errors="replace"))
    except _ET.ParseError:
        return None
    out: dict[str, str] = {}
    for child in root:
        name = child.get("name")
        if not name:
            continue
        if child.tag == "string":
            out[name] = child.text or ""
        else:
            v = child.get("value")
            if v is not None:
                out[name] = v
    return out


def _games_folder_fs_path() -> str | None:
    """Translate the saved gamesFolderUri SAF tree URI to a filesystem path.
    Mirrors FrontendLaunchHelper.treeUriToFilesystemPath in Kotlin so the
    discovery and launch paths agree on the same root.
    """
    prefs = _read_prefs()
    if not prefs:
        return None
    uri = prefs.get("gamesFolderUri")
    if not uri:
        return None
    m = re.search(r"/tree/([^/?#]+)", uri)
    if not m:
        return None
    doc_id = _urlunquote(m.group(1))
    if ":" in doc_id:
        volume, relative = doc_id.split(":", 1)
    else:
        volume, relative = doc_id, ""
    vol = volume.lower()
    if vol == "primary":
        base = "/storage/emulated/0"
    elif vol == "home":
        base = "/storage/emulated/0/Documents"
    elif volume:
        base = f"/storage/{volume}"
    else:
        return None
    return base if not relative else f"{base}/{relative.strip('/')}"


def _scan_games(folder: str) -> list[str]:
    """Enumerate disc images under `folder` via adb shell `find`. Returns
    absolute device paths sorted case-insensitively.
    """
    exts = " -o ".join(f"-iname '*{e}'" for e in GAME_EXTS)
    out = _sh(
        f"find {shlex.quote(folder)} -maxdepth 6 -type f \\( {exts} \\) 2>/dev/null"
    )
    files = [l.strip() for l in out.splitlines() if l.strip()]
    return sorted(files, key=str.lower)


def _game_title_from_path(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return re.sub(r"[._]+", " ", stem).strip() or name


# ---------------------------------------------------------------------------
# Device / lifecycle
# ---------------------------------------------------------------------------

@mcp.tool()
def list_devices() -> str:
    """List connected ADB devices, annotated with model/manufacturer/SoC
    properties so it's easy to map a serial to a physical phone, and with a
    marker showing which (if any) is the currently active device.

    Active-device selection is process-global within this MCP server: see
    `set_active_device(serial)` to pick one and `get_active_device()` to
    query it. When set, ANDROID_SERIAL is exported to the environment so
    every subsequent adb call targets that device.
    """
    raw = _adb("devices").stdout.strip()
    devs = _list_attached_devices()
    if not devs:
        return raw
    rows = [raw, ""]
    rows.append(f"{'serial':<24}  {'model':<22}  {'manufacturer':<14}  {'soc':<14}  active")
    rows.append("-" * 96)
    for serial in devs:
        def gp(prop: str) -> str:
            r = subprocess.run(
                ["adb", "-s", serial, "shell", "getprop", prop],
                capture_output=True, text=True, check=False, timeout=5,
            )
            return (r.stdout or "").strip()
        model = gp("ro.product.model") or "?"
        manuf = gp("ro.product.manufacturer") or "?"
        soc = (gp("ro.soc.model")
               or gp("ro.hardware")
               or gp("ro.board.platform")
               or "?")
        active = "*" if serial == _ACTIVE_DEVICE else ""
        rows.append(f"{serial:<24}  {model:<22}  {manuf:<14}  {soc:<14}  {active}")
    return "\n".join(rows)

@mcp.tool()
def set_active_device(serial: str = "") -> str:
    """Pin all subsequent MCP tool calls to a specific ADB serial.

    Used for A/B comparisons across two devices in the same session (e.g.
    Pixel 10a vs Zenfone 10 for Halo 2 perf work). Sets ANDROID_SERIAL in
    the process environment, which is honored by every `adb` invocation in
    this server. Pass an empty string to clear the pin (then `_device()`
    falls back to picking the first attached device).

    Returns the new active device, or an error if `serial` isn't attached.
    """
    global _ACTIVE_DEVICE
    serial = serial.strip()
    if not serial:
        _ACTIVE_DEVICE = None
        _sync_active_device_env()
        return "Active device cleared (will fall back to first attached device)."
    devs = _list_attached_devices()
    if serial not in devs:
        return (f"Serial '{serial}' not attached. Currently attached: "
                f"{', '.join(devs) if devs else '(none)'}")
    _ACTIVE_DEVICE = serial
    _sync_active_device_env()
    return f"Active device set to {serial}."

@mcp.tool()
def get_active_device() -> str:
    """Return the currently pinned ADB serial, or '(none)' if not set."""
    return _ACTIVE_DEVICE or "(none)"

@mcp.tool()
def launch_app(rom: str = "") -> str:
    """Launch X1 BOX. With no args, opens the launcher; with `rom` set, hands
    that path/URI to LauncherActivity (which boots straight into emulation
    when core setup is complete).

    rom: absolute device path (`/sdcard/Games/foo.iso`) or a content://
         URI the app already has read permission for. Leave empty to just
         open the launcher UI.

    Prefer `launch_game(name)` for title boots — it discovers what's
    installed via `list_games()` and avoids path-guessing. Use this tool
    when you already have an exact path or content:// URI.
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
def list_games() -> str:
    """List Xbox titles inside the user-selected Games folder.

    Reads `gamesFolderUri` from x1box_prefs.xml (requires the debug build —
    we go through run-as), translates the SAF tree URI to a filesystem path
    using the same mapping the app uses (`primary:` → /storage/emulated/0,
    other volumes → /storage/<volume>), and enumerates supported disc
    images (.iso/.xiso/.cso/.cci) up to 6 directories deep.

    Output format: one game per line as `<title>\\t<path>`, prefixed by a
    `Games folder:` header. Pass the title to `launch_game(name)` or the
    path to `launch_app(rom=...)`.
    """
    if not _device(): return "No device connected."
    folder = _games_folder_fs_path()
    if not folder:
        if not _is_debuggable():
            return ("Cannot list games without run-as. Install the debug "
                    "build (./gradlew assembleDebug && install_apk('debug')).")
        return ("No games folder configured. Open the X1 BOX app and pick "
                "a Games folder via the Setup Wizard or Library, then retry.")
    files = _scan_games(folder)
    if not files:
        return f"Games folder: {folder}\n(no supported disc images found)"
    rows = [f"{_game_title_from_path(p)}\t{p}" for p in files]
    return f"Games folder: {folder}\n" + "\n".join(rows)

@mcp.tool()
def launch_game(name: str) -> str:
    """Launch a title by name (case-insensitive substring match).

    Resolves `name` against the discovered games folder (see list_games),
    matching first on the derived title and falling back to the filename.
    On a unique match, hands the path to LauncherActivity — same path the
    in-app library tile tap uses. On ambiguity, returns up to 10 candidate
    paths so you can refine the query.
    """
    if not _device(): return "No device connected."
    needle = name.strip().lower()
    if not needle:
        return "Provide a name substring to match against."
    folder = _games_folder_fs_path()
    if not folder:
        if not _is_debuggable():
            return ("Cannot resolve games without run-as. Install the debug "
                    "build, or pass an exact path to launch_app(rom=...).")
        return ("No games folder configured. Open the X1 BOX app and pick "
                "a Games folder, then retry.")
    files = _scan_games(folder)
    if not files:
        return f"No game files found under {folder}."

    title_hits = [p for p in files if needle in _game_title_from_path(p).lower()]
    matches = title_hits or [p for p in files if needle in p.rsplit("/", 1)[-1].lower()]
    if not matches:
        return (f"No game matched '{name}'. Run list_games() to see "
                f"available titles.")
    if len(matches) > 1:
        lines = "\n".join(f"  {_game_title_from_path(p)}\t{p}" for p in matches[:10])
        suffix = "" if len(matches) <= 10 else f"\n  ... ({len(matches) - 10} more)"
        return f"Ambiguous match for '{name}' ({len(matches)} hits):\n{lines}{suffix}"
    return launch_app(matches[0])

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


# Friendly aliases for Android keycodes commonly needed when driving xemu's
# title screens / menus. Numeric codes are accepted too — passed straight to
# `input keyevent`. Source: https://developer.android.com/reference/android/view/KeyEvent
_KEY_ALIASES = {
    "DPAD_UP": 19, "DPAD_DOWN": 20, "DPAD_LEFT": 21, "DPAD_RIGHT": 22,
    "DPAD_CENTER": 23,
    "UP": 19, "DOWN": 20, "LEFT": 21, "RIGHT": 22, "CENTER": 23,
    "ENTER": 66, "ESCAPE": 111, "BACK": 4, "HOME": 3, "MENU": 82,
    "TAB": 61, "SPACE": 62,
    "BUTTON_A": 96, "BUTTON_B": 97, "BUTTON_X": 99, "BUTTON_Y": 100,
    "BUTTON_L1": 102, "BUTTON_R1": 103, "BUTTON_L2": 104, "BUTTON_R2": 105,
    "BUTTON_THUMBL": 106, "BUTTON_THUMBR": 107,
    "BUTTON_START": 108, "BUTTON_SELECT": 109, "BUTTON_MODE": 110,
    "START": 108, "SELECT": 109,
    "A": 96, "B": 97, "X": 99, "Y": 100,
    "WAKEUP": 224, "POWER": 26,
}


# Valid sources for Android's `input` command. Using `gamepad` makes the
# event look like it came from a connected joypad, which goes through a
# different dispatcher path than touchscreen events — vendor "Game Genie"
# / "lock-touch" overlays (ASUS Zenfone, ROG, etc.) only block touch and
# leave gamepad events flowing to the focused activity below.
_INPUT_SOURCES = {
    "gamepad", "keyboard", "dpad", "joystick", "touchpad", "touchscreen",
    "touchnavigation", "mouse", "stylus", "trackball", "default",
}


@mcp.tool()
def input_keyevent(key: str, repeat: int = 1, delay_ms: int = 150,
                   source: str = "gamepad") -> str:
    """Send an Android key event to whatever has focus on the active device.

    Built for driving xemu through its title / menu screens without a
    physical gamepad: SDL's Android backend translates these into joypad
    events, so `BUTTON_A`/`BUTTON_START`/`DPAD_UP` reach the running game.

    Args:
      key:      Key name (BUTTON_A, BUTTON_START, DPAD_UP, ENTER, ...) or
                a numeric keycode like '96'. Case-insensitive.
      repeat:   Number of presses (1-32). Useful for menu navigation
                ('DPAD_DOWN' repeat=3). Default 1.
      delay_ms: Delay between presses when repeat > 1 (0-2000). Default 150.
      source:   Android `input` source tag: 'gamepad' (default), 'keyboard',
                'dpad', 'touchscreen', etc. The default `gamepad` value
                bypasses ASUS/ROG GameGenie "lock-touch" overlays which only
                block touchscreen events; gamepad events keep flowing to the
                focused activity. Pass 'default' or '' to use the legacy
                `input keyevent` path with no explicit source.

    Returns a one-line summary of what was sent.
    """
    if not _device(): return "No device connected."
    raw = key.strip().upper()
    if raw.isdigit():
        code = int(raw)
    else:
        if raw.startswith("KEYCODE_"):
            raw = raw[len("KEYCODE_"):]
        code = _KEY_ALIASES.get(raw)
        if code is None:
            known = ", ".join(sorted(set(_KEY_ALIASES))[:12])
            return (f"Unknown key '{key}'. Pass a numeric keycode or one of: "
                    f"{known}, ...")
    repeat = max(1, min(int(repeat), 32))
    delay_ms = max(0, min(int(delay_ms), 2000))

    src = (source or "").strip().lower()
    if src in ("", "default"):
        prefix = "input"
    elif src in _INPUT_SOURCES:
        prefix = f"input {src}"
    else:
        return (f"Unknown source '{source}'. Use one of: "
                f"{', '.join(sorted(_INPUT_SOURCES))} (or '' / 'default').")

    if repeat == 1:
        _sh(f"{prefix} keyevent {code}")
        return f"sent {prefix} keyevent {code} ({key}) x1"

    # Compose a single on-device shell command so we keep adb round-trips
    # to one regardless of repeat count.
    cmd_parts = []
    for i in range(repeat):
        cmd_parts.append(f"{prefix} keyevent {code}")
        if i < repeat - 1 and delay_ms:
            cmd_parts.append(f"sleep {delay_ms / 1000:.3f}")
    _sh(" ; ".join(cmd_parts))
    return f"sent {prefix} keyevent {code} ({key}) x{repeat} (delay {delay_ms}ms)"


# ---------------------------------------------------------------------------
# Keep-alive daemon
#
# Persistent on-device shell loop that (a) wakes the screen when it goes to
# sleep and (b) periodically nudges the running game with a gamepad button
# (DPAD_UP by default) to prevent the Halo 2 attract-mode demo from kicking
# in. Runs entirely in adb shell — no app changes — and is identified by a
# PID file under /data/local/tmp so we can stop / status-check it from any
# subsequent MCP tool call.
# ---------------------------------------------------------------------------

# Files live under /data/local/tmp because that's world-writable to the shell
# user (no run-as needed) and survives across our adb sessions.
KEEPALIVE_DIR = "/data/local/tmp"
KEEPALIVE_SCRIPT = f"{KEEPALIVE_DIR}/x1box_keepalive.sh"
KEEPALIVE_PID = f"{KEEPALIVE_DIR}/x1box_keepalive.pid"
KEEPALIVE_LOG = f"{KEEPALIVE_DIR}/x1box_keepalive.log"
KEEPALIVE_HB = f"{KEEPALIVE_DIR}/x1box_keepalive.hb"

# The actual loop run on-device. Uses POSIX shell so it works on toybox /
# busybox without bash. `setsid` detaches it from the controlling tty so it
# survives the spawning adb shell exiting. Each iteration:
#   1. If `keep_awake` is set and the screen is off, send a WAKEUP keyevent.
#   2. Send `input <source> keyevent <key>` to nudge the game.
#   3. Touch the heartbeat file with the current epoch.
#   4. Sleep `interval`.
# Exits cleanly if the pid-file is removed or replaced — that's how
# keepalive_stop kills us.
_KEEPALIVE_LOOP = r"""#!/system/bin/sh
INTERVAL="${1:-90}"
KEY="${2:-19}"
SRC="${3:-gamepad}"
KEEP_AWAKE="${4:-1}"
PID_FILE="__PID_FILE__"
HB_FILE="__HB_FILE__"
LOG_FILE="__LOG_FILE__"

# Compose the input command once. 'default' source = plain `input keyevent`.
if [ "$SRC" = "default" ] || [ -z "$SRC" ]; then
  INPUT_CMD="input keyevent $KEY"
else
  INPUT_CMD="input $SRC keyevent $KEY"
fi

echo "$$" > "$PID_FILE"
echo "$(date +%s) keepalive started: interval=${INTERVAL}s key=${KEY} src=${SRC} awake=${KEEP_AWAKE}" >> "$LOG_FILE"

while :; do
  # Stop signal: pid-file deleted or owned by a different pid.
  if [ ! -f "$PID_FILE" ]; then
    echo "$(date +%s) pid-file gone, exiting" >> "$LOG_FILE"
    exit 0
  fi
  PF_PID=$(cat "$PID_FILE" 2>/dev/null)
  if [ "$PF_PID" != "$$" ]; then
    echo "$(date +%s) pid-file replaced (now=$PF_PID, me=$$), exiting" >> "$LOG_FILE"
    exit 0
  fi

  if [ "$KEEP_AWAKE" = "1" ]; then
    # `dumpsys power` field — works on every Android we care about.
    STATE=$(dumpsys power 2>/dev/null | grep -m1 'mWakefulness=' | sed 's/.*mWakefulness=//;s/ .*//')
    if [ "$STATE" != "Awake" ]; then
      input keyevent 224 >/dev/null 2>&1
      echo "$(date +%s) woke from state=$STATE" >> "$LOG_FILE"
    fi
  fi

  $INPUT_CMD >/dev/null 2>&1
  date +%s > "$HB_FILE"

  # toybox sleep accepts integer seconds.
  sleep "$INTERVAL"
done
"""


@mcp.tool()
def keepalive_start(interval_s: int = 90, key: str = "DPAD_UP",
                    source: str = "gamepad", keep_awake: bool = True) -> str:
    """Start a persistent on-device daemon that prevents the screen from
    sleeping and nudges the running game with a gamepad button on an
    interval — so an idle Halo 2 doesn't drift into the attract-mode demo
    while we're collecting perf samples.

    The daemon lives on the active device under /data/local/tmp/ and
    survives across MCP tool calls. Stops automatically when the pid-file
    is removed (see `keepalive_stop`).

    Args:
      interval_s: seconds between key nudges (default 90; Halo 2's attract
                  timeout is well over 2 minutes so ~90s leaves comfortable
                  margin without spamming inputs).
      key:        button name (e.g. 'DPAD_UP', 'BUTTON_A') or numeric code.
                  Default 'DPAD_UP' — does no menu action but resets the
                  idle timer.
      source:     `input` source tag. Default 'gamepad' so the event
                  bypasses ASUS GameGenie's lock-touch overlay and is
                  dispatched as a joypad button (SDL maps it through to
                  xemu the same as a real controller).
      keep_awake: if True, also send WAKEUP keyevent whenever the screen
                  goes to sleep.

    Returns the on-device PID of the daemon and current settings.
    """
    if not _device():
        return "No device connected."

    raw = key.strip().upper()
    if raw.isdigit():
        code = int(raw)
    else:
        if raw.startswith("KEYCODE_"):
            raw = raw[len("KEYCODE_"):]
        code = _KEY_ALIASES.get(raw)
        if code is None:
            return f"Unknown key '{key}'. See input_keyevent for valid names."

    src = (source or "").strip().lower()
    if src and src != "default" and src not in _INPUT_SOURCES:
        return (f"Unknown source '{source}'. Use one of: "
                f"{', '.join(sorted(_INPUT_SOURCES))} (or 'default').")

    interval = max(5, min(int(interval_s), 3600))

    # Stop any existing daemon first (idempotent).
    keepalive_stop()

    # Push the script.
    script = (_KEEPALIVE_LOOP
              .replace("__PID_FILE__", KEEPALIVE_PID)
              .replace("__HB_FILE__", KEEPALIVE_HB)
              .replace("__LOG_FILE__", KEEPALIVE_LOG))
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        local = f.name
    try:
        if _adb("push", local, KEEPALIVE_SCRIPT).returncode != 0:
            return "Failed to stage keepalive script."
    finally:
        os.unlink(local)
    _sh(f"chmod 755 {KEEPALIVE_SCRIPT}")
    _sh(f": > {KEEPALIVE_LOG}; rm -f {KEEPALIVE_HB} {KEEPALIVE_PID}")

    # Launch detached. `setsid` is sometimes absent on Android shells; fall
    # back to a double-fork via `nohup ... &` which produces the same effect
    # (parent shell exits, daemon's parent becomes init, no controlling tty).
    awake = "1" if keep_awake else "0"
    src_arg = src or "default"
    spawn = (
        f"cd /data/local/tmp && "
        f"nohup sh {KEEPALIVE_SCRIPT} {interval} {code} {src_arg} {awake} "
        f">> {KEEPALIVE_LOG} 2>&1 < /dev/null &"
    )
    _sh(spawn)

    # Wait briefly for the daemon to write its pid.
    pid = ""
    for _ in range(20):
        out = _sh(f"cat {KEEPALIVE_PID} 2>/dev/null").strip()
        if out and out.isdigit():
            pid = out
            break
        time.sleep(0.1)

    if not pid:
        tail = _sh(f"tail -n 20 {KEEPALIVE_LOG} 2>/dev/null")
        return f"Daemon did not announce its PID. Log tail:\n{tail or '(empty)'}"

    return (f"keepalive: pid={pid} interval={interval}s "
            f"key={key}({code}) source={src_arg} keep_awake={keep_awake}")


@mcp.tool()
def keepalive_stop() -> str:
    """Stop the on-device keepalive daemon (if running) and remove its
    pid-file / heartbeat. Idempotent — safe to call when nothing is
    running.
    """
    if not _device():
        return "No device connected."
    pid = _sh(f"cat {KEEPALIVE_PID} 2>/dev/null").strip()
    if pid and pid.isdigit():
        _sh(f"kill {pid} 2>/dev/null; sleep 0.2; kill -9 {pid} 2>/dev/null")
        _sh(f"rm -f {KEEPALIVE_PID} {KEEPALIVE_HB}")
        return f"keepalive: stopped pid={pid}"
    _sh(f"rm -f {KEEPALIVE_PID} {KEEPALIVE_HB}")
    return "keepalive: not running"


@mcp.tool()
def keepalive_status() -> str:
    """Report whether the keepalive daemon is alive on the active device,
    its PID, age, last heartbeat, and the tail of its log.
    """
    if not _device():
        return "No device connected."
    pid = _sh(f"cat {KEEPALIVE_PID} 2>/dev/null").strip()
    if not pid or not pid.isdigit():
        return "keepalive: not running"
    alive = _sh(f"test -d /proc/{pid} && echo yes || echo no").strip()
    if alive != "yes":
        return f"keepalive: pid={pid} but /proc/{pid} missing (stale)"
    hb = _sh(f"cat {KEEPALIVE_HB} 2>/dev/null").strip()
    now = _sh("date +%s").strip()
    try:
        age = int(now) - int(hb) if hb else None
    except ValueError:
        age = None
    tail = _sh(f"tail -n 5 {KEEPALIVE_LOG} 2>/dev/null")
    parts = [f"keepalive: pid={pid} alive"]
    if age is not None:
        parts.append(f"last_hb={age}s ago")
    if tail:
        parts.append("log_tail:\n" + tail)
    return "  ".join(parts[:2]) + (("\n" + parts[2]) if len(parts) > 2 else "")


@mcp.tool()
def device_status() -> str:
    """One-shot health snapshot of the active device: screen power,
    foreground activity, x1box pids, recent xemu-reported FPS, GPU busy %,
    battery temp, and keepalive-daemon state. Use this as the quick "is
    the rig still in a useful state?" check between heavy probes.

    All values come from cheap reads (dumpsys / sysfs / /proc); the call
    completes in well under a second on a healthy device.
    """
    if not _device():
        return "No device connected."

    # Screen power + wakefulness — one dumpsys call.
    wake = _sh(
        "dumpsys power 2>/dev/null | "
        "grep -E 'mWakefulness=|mScreenState=|mDisplayState=' | head -4"
    )

    # Foreground activity.
    top = _sh(
        "dumpsys activity activities 2>/dev/null | "
        "grep -m1 'topResumedActivity=' | sed 's/^ *//'"
    )

    # x1box pids.
    ui = _ui_pid()
    emu = _sh(f"pidof {EMU_PROCESS}").strip()

    # Battery temperature (deci-degC) — cheap one-line read.
    batt_temp_raw = _sh("cat /sys/class/power_supply/battery/temp 2>/dev/null").strip()
    try:
        batt_temp_c = int(batt_temp_raw) / 10.0
        batt_temp = f"{batt_temp_c:.1f}°C"
    except ValueError:
        batt_temp = batt_temp_raw or "?"

    # GPU busy %. Try Mali sysfs, then Adreno KGSL.
    gpu = _sh(
        "cat /sys/class/misc/mali0/device/utilization 2>/dev/null || "
        "cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null"
    ).strip()
    if not gpu:
        gpu = "?"

    # FPS from xemu's own overlay (writes to logcat tag "xemu" / on-screen).
    # As a cheap proxy, pull the most recent gfxinfo line.
    gfx = _sh(
        f"dumpsys gfxinfo {PKG} 2>/dev/null | "
        f"grep -m1 'Janky frames:' || true"
    ).strip()

    # Keepalive daemon.
    ka_pid = _sh(f"cat {KEEPALIVE_PID} 2>/dev/null").strip()
    if ka_pid and ka_pid.isdigit():
        alive = _sh(f"test -d /proc/{ka_pid} && echo yes || echo no").strip()
        hb = _sh(f"cat {KEEPALIVE_HB} 2>/dev/null").strip()
        now = _sh("date +%s").strip()
        try:
            age = int(now) - int(hb) if hb else "?"
        except ValueError:
            age = "?"
        ka = f"pid={ka_pid} alive={alive} last_hb={age}s"
    else:
        ka = "not running"

    lines = [
        "=== device_status ===",
        f"power:        {wake or '(unknown)'}",
        f"foreground:   {top or '(unknown)'}",
        f"x1box.ui:     {ui or 'not running'}",
        f"x1box.emu:    {emu or 'not running'}",
        f"battery_temp: {batt_temp}",
        f"gpu_busy:     {gpu}",
        f"gfx_jank:     {gfx or '(no gfxinfo)'}",
        f"keepalive:    {ka}",
    ]
    return "\n".join(lines)


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


# User-visible FPS is measured from xemu's own host-swap callsite. ui/xemu.c
# emits one logcat line per SDL_GL_SwapWindow under the tag "xemu-fps":
#
#     11-20 14:32:10.123 12345 12999 I xemu-fps: f=4711 g=823 t=98765432109876
#
# Three counters are captured per swap:
#   f = g_android_frame_counter         (host swap counter)
#   g = g_nv2a_stats.frame_count        (guest NV097_FLIP_STALL ticks —
#                                        rate the Xbox produced new frames)
#   t = CLOCK_MONOTONIC ns after swap returns
#
# Host swap fires every ~16ms regardless of whether the guest produced
# anything new, so delta(f)/dt is misleading. delta(g)/dt is the rate
# at which a NEW guest framebuffer reached the swap site — i.e. the
# rate of unique frames the user actually saw on screen.
#
# Why not SurfaceFlinger --latency? On Android 14+ with BLAST surfaces the
# legacy --latency dumper does not populate frame triples for SurfaceView-
# backed apps like x1-box — empirically it returns just the refresh period.
_FPS_LOGCAT_RE = re.compile(r"\bf=(\d+)\s+g=(\d+)\s+t=(\d+)\b")
# Backward-compat: APKs from the first iteration of the instrumentation
# emit only f= and t= (no g=). Parse those too and report degraded mode.
_FPS_LOGCAT_RE_LEGACY = re.compile(r"\bf=(\d+)\s+t=(\d+)\b")


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@mcp.tool()
def measure_fps(duration: float = 5.0) -> str:
    """User-visible FPS: unique guest-rendered frames per second.

    Drains logcat for the `xemu-fps` tag emitted by ui/xemu.c after every
    SDL_GL_SwapWindow. The host swaps a buffer ~every refresh tick whether
    or not the guest produced anything new, so swap-rate is the wrong
    answer — what the user actually sees is unique guest frames. We get
    that from nv2a's NV097_FLIP_STALL counter (`g=`), snapshot at swap
    time, and report:
        - User-visible FPS: delta(guest flip counter) / window seconds
          — the rate of unique frames hitting the screen, what the
          eye perceives. NOT the panel refresh rate, NOT nv2a's
          internal sim rate (which doesn't account for whether the
          host swapped the guest frame).
        - Host swap FPS:   delta(swap counter) / window — for diagnostic
          context.
        - Unique-frame ratio: how many host swaps showed a new guest
          frame vs. re-blitted the previous one.
        - Per-unique-frame intervals + percentiles, computed from the
          host-side timestamps of swaps where g advanced.

    Args:
        duration: sampling window in seconds (default 5.0).

    Requires the `g=` field in the log line (added 2026-05-20). Older
    builds fall back to swap-only reporting with a warning.
    """
    if not _device():
        return "No device connected."
    if duration <= 0:
        return "duration must be > 0."

    # xemu-fps is emitted from the :xemu process, NOT the launcher UI.
    # We don't filter by pid since the tag is unique to xemu's swap site
    # and :xemu can respawn mid-sample.
    if not _emu_pid() and not _ui_pid():
        return f"{PKG} is not running. Launch it before sampling."

    _adb("logcat", "-c")
    time.sleep(duration)
    raw = _sh("logcat -d -s xemu-fps:I")
    lines = raw.splitlines() if raw else []

    # Each sample: (host_swap_count, guest_flip_count, mono_ns).
    # legacy samples (no g=) record guest=None.
    samples: list[tuple[int, int | None, int]] = []
    for ln in lines:
        m = _FPS_LOGCAT_RE.search(ln)
        if m:
            samples.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        m = _FPS_LOGCAT_RE_LEGACY.search(ln)
        if m:
            samples.append((int(m.group(1)), None, int(m.group(2))))

    if len(samples) < 2:
        return (f"Got {len(samples)} xemu-fps samples in {duration:.2f}s. "
                f"Either xemu isn't swapping (paused / background / boot "
                f"animation in slot 0), or the running APK predates the "
                f"`xemu-fps` instrumentation in ui/xemu.c. "
                f"Try `logcat -d -s xemu-fps:I` to inspect manually.")

    samples.sort(key=lambda s: s[2])
    swap_counter = [s[0] for s in samples]
    guest_counter = [s[1] for s in samples]
    timestamps = [s[2] for s in samples]

    legacy_mode = any(g is None for g in guest_counter)

    span_ns = timestamps[-1] - timestamps[0]
    if span_ns <= 0:
        return f"Got {len(samples)} samples but zero-duration window."

    swap_delta = swap_counter[-1] - swap_counter[0]
    swap_fps = swap_delta / (span_ns / 1e9)

    out: list[str] = ["=== User-visible FPS (xemu host-swap + nv2a flip) ==="]
    out.append(f"Sample window    : {duration:.2f}s requested, "
               f"{span_ns/1e9:.2f}s observed")
    out.append(f"Swap log lines   : {len(samples)} "
               f"(swap-counter delta {swap_delta})")
    out.append("")

    if legacy_mode:
        out.append("WARNING: log lines missing g= field — running an older APK. "
                   "Rebuild from current source to get the user-visible (guest "
                   "flip) FPS. Reporting swap-rate only:")
        out.append(f"Host swap FPS    : {swap_fps:6.2f}")
        return "\n".join(out)

    guest_delta = guest_counter[-1] - guest_counter[0]
    user_fps = guest_delta / (span_ns / 1e9)

    # Find swaps that carried a NEW guest frame (g advanced since previous
    # swap). Per-unique-frame intervals come from the host monotonic ns at
    # those swaps — that's when the user's eyes saw the frame change.
    unique_swap_ts: list[int] = [timestamps[0]] if guest_counter[0] is not None else []
    for i in range(1, len(samples)):
        if guest_counter[i] != guest_counter[i - 1]:
            unique_swap_ts.append(timestamps[i])

    out.append(f"Guest flips (NV097_FLIP_STALL) : {guest_delta}")
    out.append(f"Unique-frame swaps             : {len(unique_swap_ts)}")
    duplicate = swap_delta - guest_delta
    duplicate_pct = (100.0 * duplicate / swap_delta) if swap_delta > 0 else 0.0
    out.append(f"Duplicate swaps                : {duplicate} ({duplicate_pct:.1f}% of "
               f"swaps re-blitted the previous frame)")
    out.append("")
    out.append(f"USER-VISIBLE FPS : {user_fps:6.2f}   <-- what the eye sees")
    out.append(f"Host swap FPS    : {swap_fps:6.2f}   (for context — capped at "
               f"display refresh)")
    out.append("")

    intervals_ns = [b - a for a, b in zip(unique_swap_ts, unique_swap_ts[1:]) if b > a]
    long_stall_threshold_ns = 2_000_000_000
    stalls = [i for i in intervals_ns if i >= long_stall_threshold_ns]
    clean = [i for i in intervals_ns if i < long_stall_threshold_ns]

    if len(clean) >= 2:
        mean_int = sum(clean) / len(clean)
        sc = sorted(clean)
        median_int = _percentile(sc, 50.0)
        p5_int = _percentile(sc, 5.0)
        p95_int = _percentile(sc, 95.0)
        p99_int = _percentile(sc, 99.0)
        fps_median = 1e9 / median_int if median_int > 0 else 0.0
        fps_best5 = 1e9 / p5_int if p5_int > 0 else 0.0
        fps_low5 = 1e9 / p95_int if p95_int > 0 else 0.0
        fps_low1 = 1e9 / p99_int if p99_int > 0 else 0.0
        var = sum((i - mean_int) ** 2 for i in clean) / (len(clean) - 1) \
              if len(clean) > 1 else 0.0
        stddev_ms = (var ** 0.5) / 1e6
        jitter_pct = 100.0 * (var ** 0.5) / mean_int if mean_int else 0.0

        out.append(f"FPS median       : {fps_median:6.2f}")
        out.append(f"FPS 1%-low       : {fps_low1:6.2f}   (slowest 1% of unique frames)")
        out.append(f"FPS 5%-low       : {fps_low5:6.2f}   (slowest 5% of unique frames)")
        out.append(f"FPS best 5%      : {fps_best5:6.2f}")
        out.append("")
        out.append(f"Unique-frame interval : mean {mean_int/1e6:.2f} ms, "
                   f"median {median_int/1e6:.2f} ms")
        out.append(f"Jitter           : stddev {stddev_ms:.2f} ms ({jitter_pct:.1f}%)")
        if stalls:
            out.append(f"Long stalls      : {len(stalls)} (>2s, excluded from "
                       f"percentile math)")
    elif user_fps < 0.1:
        out.append("(guest counter did not advance — emulator paused, on a static "
                   "menu xemu hasn't reached, or pre-FLIP_STALL boot)")
    else:
        out.append("(too few unique frames in window for percentile stats; "
                   "increase duration)")

    return "\n".join(out)


# JIT periodic stats line from hw/xbox/nv2a/pgraph/profile.c (every ~2s):
#   xemu-jit: win=<ms> tb_gen/s=N tb_inval/s=N tb_flush/s=N
#             jc_hit%=P jc_miss/s=N lru_hit%=P lru_call/s=N
#             phint_hit%=P phint_call/s=N
#             chain_runs/s=N chain_avg=A.BB irq/s=N
# Only emitted by debug + perftest builds in the :xemu process. If you
# don't see lines, either the build predates the instrumentation or
# the emulator hasn't reached FLIP_STALL yet (still on the boot logo).
_JIT_STAT_RE = re.compile(
    r"win=(?P<win_ms>\d+)_ms\s+"
    r"tb_gen/s=(?P<tb_gen>\d+)\s+"
    r"tb_inval/s=(?P<tb_inval>\d+)\s+"
    r"tb_flush/s=(?P<tb_flush>\d+)\s+"
    r"jc_hit%=(?P<jc_hit_pct>\d+)\s+"
    r"jc_miss/s=(?P<jc_miss>\d+)\s+"
    r"lru_hit%=(?P<lru_hit_pct>\d+)\s+"
    r"lru_call/s=(?P<lru_calls>\d+)\s+"
    r"phint_hit%=(?P<phint_hit_pct>\d+)\s+"
    r"phint_call/s=(?P<phint_calls>\d+)\s+"
    r"chain_runs/s=(?P<chain_runs>\d+)\s+"
    r"chain_avg=(?P<chain_avg_int>\d+)\.(?P<chain_avg_frac>\d+)\s+"
    r"irq/s=(?P<irq>\d+)"
    r"(?:\s+fpcr=(?P<fpcr>0x[0-9a-fA-F]+)\s+FZ=(?P<fz>\d+)\s+FZ16=(?P<fz16>\d+)\s+DN=(?P<dn>\d+))?"
    r"(?:\s+route_shim=(?P<route_on>\d+)\s+route_hit%=(?P<route_hit_pct>\d+)\s+route_call/s=(?P<route_calls>\d+))?"
)


@mcp.tool()
def jit_stats(samples: int = 5) -> str:
    """JIT pipeline health: TB churn, jmp_cache hit rate, helper LRU + phint
    hit rate, cranelift chain depth, IRQ exit rate.

    Halo 2 callgraphs frequently surface helper_lookup_tb_ptr +
    tb_htable_lookup as a huge chunk of vCPU, and tb_gen_code as a huge
    chunk of cpu_exec_loop. The flat profile alone can't say WHY —
    working-set thrash on the jmp_cache vs. constant tb_phys_invalidate
    waves vs. helper LRU misses all show up the same way in cpu-clock
    samples. This drains the structured `xemu-jit:` lines profile.c
    emits every ~2s with the actual per-second rates so you can tell:

      * tb_gen/s high + tb_inval/s low  → jmp_cache (or TB pool) too
        small for working set; TBs evicted and re-translated.
      * tb_gen/s high + tb_inval/s high → self-modifying guest or
        page-protect churn evicting TBs; cache size won't fix it.
      * tb_flush/s spikes                → guest tripping tb_flush
        triggers (CR3 storms, pool exhaustion).
      * lru_hit% low                     → 4-slot LRU not absorbing
        the dispatch pattern; bigger LRU or different structure.
      * phint_hit% low                   → same-page tb_htable lookups
        are rare; the phys_pc hint isn't saving page walks here.
      * jc_hit% low                      → tb_jmp_cache miss rate is
        the dominant cost source.
      * chain_avg low (<8)               → cranelift chain dispatcher
        bails out fast; not amortising the dispatcher round-trip.

    Args:
      samples: how many recent xemu-jit samples to display (default 5
               = last ~10 seconds; pass 30 for the last minute).

    The emitter sleeps ~2 s between lines, so a sample of N covers
    ~2N seconds.
    """
    if not _device():
        return "No device connected."
    pid = _emu_pid()
    if not pid:
        return f"{PKG}:xemu is not running. Launch it before sampling."
    if samples <= 0:
        return "samples must be > 0."

    raw = _sh(f"logcat -d -s xemu-jit:I --pid={pid}")
    lines = [ln for ln in (raw or "").splitlines() if "win=" in ln]
    if not lines:
        return ("No xemu-jit lines yet. Either the emulator hasn't reached "
                "FLIP_STALL (still on the boot animation) or the running "
                "build predates the JIT-stats instrumentation in "
                "hw/xbox/nv2a/pgraph/profile.c — rebuild + reinstall.")

    parsed = []
    for ln in lines[-samples:]:
        m = _JIT_STAT_RE.search(ln)
        if not m:
            continue
        d = m.groupdict()
        parsed.append({
            "win_ms":     int(d["win_ms"]),
            "tb_gen":     int(d["tb_gen"]),
            "tb_inval":   int(d["tb_inval"]),
            "tb_flush":   int(d["tb_flush"]),
            "jc_hit_pct": int(d["jc_hit_pct"]),
            "jc_miss":    int(d["jc_miss"]),
            "lru_hit_pct":int(d["lru_hit_pct"]),
            "lru_calls":  int(d["lru_calls"]),
            "phint_hit_pct": int(d["phint_hit_pct"]),
            "phint_calls":   int(d["phint_calls"]),
            "chain_runs": int(d["chain_runs"]),
            "chain_avg":  float(f"{d['chain_avg_int']}.{d['chain_avg_frac']}"),
            "irq":        int(d["irq"]),
            "fpcr":       d.get("fpcr"),
            "fz":         int(d["fz"])   if d.get("fz")   else None,
            "fz16":       int(d["fz16"]) if d.get("fz16") else None,
            "dn":         int(d["dn"])   if d.get("dn")   else None,
            "route_on":       int(d["route_on"])      if d.get("route_on")      else None,
            "route_hit_pct":  int(d["route_hit_pct"]) if d.get("route_hit_pct") else None,
            "route_calls":    int(d["route_calls"])   if d.get("route_calls")   else None,
        })
    if not parsed:
        return ("Found xemu-jit lines but none matched the expected shape. "
                "Check the format in hw/xbox/nv2a/pgraph/profile.c "
                "vs _JIT_STAT_RE in this tool.")

    out = []
    out.append(f"Last {len(parsed)} xemu-jit samples (each window ~2s):")
    out.append("")
    out.append("  TB pool                  jmp_cache       helper LRU   "
               "phint        chain         irq/s")
    out.append("  gen/s inval/s flush/s   hit%  miss/s     hit% call/s   "
               "hit% call/s  runs/s  avg")
    out.append("  " + "-" * 96)
    for r in parsed:
        out.append(
            f"  {r['tb_gen']:>5} {r['tb_inval']:>7} {r['tb_flush']:>7}   "
            f"{r['jc_hit_pct']:>3}%  {r['jc_miss']:>7}    "
            f"{r['lru_hit_pct']:>3}% {r['lru_calls']:>6}   "
            f"{r['phint_hit_pct']:>3}% {r['phint_calls']:>6}  "
            f"{r['chain_runs']:>6}  {r['chain_avg']:>4.1f}   "
            f"{r['irq']:>5}"
        )

    last = parsed[-1]
    if last["fpcr"] is not None:
        fz_warn = " ⚠ FZ=1 — denormal flush ON" if last["fz"] else ""
        dn_warn = " ⚠ DN=1 — default NaN ON"    if last["dn"] else ""
        out.append("")
        out.append(f"vCPU FPCR (last sample): {last['fpcr']} "
                   f"FZ={last['fz']} FZ16={last['fz16']} DN={last['dn']}"
                   f"{fz_warn}{dn_warn}")
    if last.get("route_on") is not None:
        out.append(
            f"helper→shim routing: {'ON' if last['route_on'] else 'OFF'}  "
            f"hit%={last['route_hit_pct']}  calls/s={last['route_calls']:,} "
            f"(set X1BOX_HELPER_ROUTE_SHIM=1/0 then relaunch to A/B)"
        )
    out.append("")
    out.append("Diagnosis hints:")
    if last["tb_gen"] > 2000:
        out.append(f"  * tb_gen/s={last['tb_gen']}: heavy TB translation. "
                   f"Likely working-set thrash or self-mod.")
    if last["tb_inval"] > 1000:
        out.append(f"  * tb_inval/s={last['tb_inval']}: TBs being "
                   f"invalidated — self-modifying code or page churn.")
    if last["tb_flush"] > 0:
        out.append(f"  * tb_flush/s={last['tb_flush']}: full TB cache "
                   f"wipes — investigate pool exhaustion / CR3 storms.")
    if last["jc_hit_pct"] < 80 and last["jc_miss"] > 50000:
        out.append(f"  * jc_hit%={last['jc_hit_pct']}%: tb_jmp_cache miss "
                   f"rate high — falling through to QHT often.")
    if last["lru_hit_pct"] < 30:
        out.append(f"  * lru_hit%={last['lru_hit_pct']}%: helper 4-slot LRU "
                   f"not absorbing dispatch pattern.")
    if last["phint_calls"] > 0 and last["phint_hit_pct"] < 20:
        out.append(f"  * phint_hit%={last['phint_hit_pct']}%: same-page LRU "
                   f"side-channel rarely fires — page walks not being "
                   f"skipped much.")
    if last["chain_avg"] < 8 and last["chain_runs"] > 1000:
        out.append(f"  * chain_avg={last['chain_avg']}: cranelift chain "
                   f"dispatcher bails out fast — dispatcher round-trip "
                   f"not amortised.")
    if last["irq"] > 500:
        out.append(f"  * irq/s={last['irq']}: high IRQ exit rate — vCPU "
                   f"frequently bailing back to the dispatcher.")
    if len(out) == 5 + len(parsed):  # no hint lines appended
        out.append("  (no anomalies detected at default thresholds)")
    return "\n".join(out)


# Pipeline stats line from hw/xbox/nv2a/pgraph/profile.c (every ~2s):
#   xemu-pipe: win=<ms> ring avg=<bytes> max=<bytes> dma_len=<bytes>
#              empty%=<pct> full%=<pct>
#              pusher_calls/s=<n> words/s=<n>
#              pfifo wait_spin/s=<n> wait_idle/s=<n> wait%=<pct>
#              vcpu hle_yield/s=<n> hle_iters/s=<n> hle_yield_ms/s=<n>
#              hle_yield%=<pct>
#              uni_skip_hit%=<pct> uni_skip_total/s=<n>
#              idx_cache_hit%=<pct> idx_cache_total/s=<n> idx_cache_evicts/s=<n>
#              surf_skip%=<pct> surf_total/s=<n>
#
# Diagnoses producer-consumer in the vCPU → pgraph-ring → nv2a.pfifo_thre
# → vulkan-render pipeline. CPU% alone (from thread_wait_profile) doesn't
# tell you WHO IS BLOCKING WHOM — pipeline_stats does.
#
# The trailing uni_skip / idx_cache / surf_skip fields are optional in
# the regex — older builds that predate Phase 1.2/1.3 wiring won't have
# them, and we still want to parse the rest of the line cleanly.
_PIPE_STAT_RE = re.compile(
    r"win=(?P<win_ms>\d+)_ms\s+"
    r"ring\s+avg=(?P<ring_avg>\d+)\s+max=(?P<ring_max>\d+)\s+"
    r"dma_len=(?P<dma_len>\d+)\s+"
    r"empty%=(?P<empty_pct>\d+)\s+full%=(?P<full_pct>\d+)\s+"
    r"pusher_calls/s=(?P<pcalls>\d+)\s+words/s=(?P<words>\d+)\s+"
    r"pfifo\s+wait_spin/s=(?P<wspin>\d+)\s+wait_idle/s=(?P<widle>\d+)\s+"
    r"wait%=(?P<wpct_int>\d+)\.(?P<wpct_frac>\d+)\s+"
    r"vcpu\s+hle_yield/s=(?P<hyields>\d+)\s+hle_iters/s=(?P<hiters>\d+)\s+"
    r"hle_yield_ms/s=(?P<hyms>\d+)\s+"
    r"hle_yield%=(?P<hypct_int>\d+)\.(?P<hypct_frac>\d+)"
    r"(?:\s+uni_skip_hit%=(?P<uni_pct>\d+)\s+uni_skip_total/s=(?P<uni_total>\d+))?"
    r"(?:\s+idx_cache_hit%=(?P<idx_pct>\d+)\s+idx_cache_total/s=(?P<idx_total>\d+)"
    r"\s+idx_cache_evicts/s=(?P<idx_evicts>\d+))?"
    r"(?:\s+surf_skip%=(?P<surf_pct>\d+)\s+surf_total/s=(?P<surf_total>\d+))?"
)


def _pct_str(x):
    return f"{x:>5.1f}%"


@mcp.tool()
def pipeline_stats(samples: int = 5) -> str:
    """Pipeline producer-consumer diagnosis: WHO IS BLOCKING WHOM.

    `thread_wait_profile` and `profile_native` tell us CPU% per thread,
    but a high CPU% on one thread doesn't tell you whether it's the
    pipeline gate or is itself being held back. To diagnose Halo 2's
    14-17 FPS plateau (well below the 30 FPS native target) we need
    producer-consumer visibility.

    This drains the `xemu-pipe:` lines that profile.c emits every ~2s
    and renders them with a diagnostic header.

    Decision keys:

      * ring full%   high  → vCPU produces GPU cmds faster than pfifo
                             drains them. PFIFO IS THE BOTTLENECK.
                             CPU optimisations on vCPU won't help.
      * ring empty%  high  → vCPU is slow-feeding pfifo. VCPU IS THE
                             BOTTLENECK. Pfifo sits waiting.
      * both low           → balanced; bottleneck is elsewhere (audio
                             sync, wall-clock pacing, frame-throttle).
      * wait_idle/s  high  → pfifo blocks on cond_wait often; vCPU
                             produces in bursts or sparse work.
      * wait_spin/s  high  → pfifo waking from short spin (good — low
                             latency).
      * hle_yield%   high  → vCPU spends X% of wall in KiIdleLoop
                             nanosleep. Compare with overall Sleep%
                             from thread_wait_profile: anything above
                             is OTHER sleep (BQL waits, futex on
                             pfifo/audio/etc.). High `other sleep` is
                             a smoking gun for hidden contention.

    Args:
      samples: how many recent xemu-pipe samples to display
               (default 5 = last ~10s; pass 30 for the last minute).
    """
    if not _device():
        return "No device connected."
    pid = _emu_pid()
    if not pid:
        return f"{PKG}:xemu is not running. Launch it before sampling."
    if samples <= 0:
        return "samples must be > 0."

    raw = _sh(f"logcat -d -s xemu-pipe:I --pid={pid}")
    lines = [ln for ln in (raw or "").splitlines() if "win=" in ln]
    if not lines:
        return ("No xemu-pipe lines yet. Either the emulator hasn't "
                "reached FLIP_STALL (still on the boot animation) or the "
                "running build predates the pipeline-stats instrumentation "
                "in hw/xbox/nv2a/pgraph/profile.c — rebuild + reinstall.")

    parsed = []
    for ln in lines[-samples:]:
        m = _PIPE_STAT_RE.search(ln)
        if not m:
            continue
        d = m.groupdict()
        # Trailing groups are optional (older builds may omit them).
        def _opt_int(key):
            v = d.get(key)
            return int(v) if v is not None else None

        parsed.append({
            "win_ms":    int(d["win_ms"]),
            "ring_avg":  int(d["ring_avg"]),
            "ring_max":  int(d["ring_max"]),
            "dma_len":   int(d["dma_len"]),
            "empty_pct": int(d["empty_pct"]),
            "full_pct":  int(d["full_pct"]),
            "pcalls":    int(d["pcalls"]),
            "words":     int(d["words"]),
            "wspin":     int(d["wspin"]),
            "widle":     int(d["widle"]),
            "wpct":      float(f"{d['wpct_int']}.{d['wpct_frac']}"),
            "hyields":   int(d["hyields"]),
            "hiters":    int(d["hiters"]),
            "hyms":      int(d["hyms"]),
            "hypct":     float(f"{d['hypct_int']}.{d['hypct_frac']}"),
            # Phase 1.1 (uniform fast-skip) + 1.2 (index LRU) + 1.3 (surface skip)
            "uni_pct":    _opt_int("uni_pct"),
            "uni_total":  _opt_int("uni_total"),
            "idx_pct":    _opt_int("idx_pct"),
            "idx_total":  _opt_int("idx_total"),
            "idx_evicts": _opt_int("idx_evicts"),
            "surf_pct":   _opt_int("surf_pct"),
            "surf_total": _opt_int("surf_total"),
        })
    if not parsed:
        return ("Found xemu-pipe lines but none matched the expected shape. "
                "Check the format in hw/xbox/nv2a/pgraph/profile.c "
                "vs _PIPE_STAT_RE in this tool.")

    out = []
    out.append(f"Last {len(parsed)} xemu-pipe samples (each window ~2s):")
    out.append("")
    out.append("  PGRAPH ring                pusher          pfifo wait   "
               "vCPU KiIdleLoop")
    out.append("  avg%  max%  empty%  full%  calls/s words/s  spin/s idle/s wait%  "
               "yield/s iters/s yield%")
    out.append("  " + "-" * 96)
    for r in parsed:
        avg_pct = (r["ring_avg"] * 100.0 / r["dma_len"]) if r["dma_len"] else 0
        max_pct = (r["ring_max"] * 100.0 / r["dma_len"]) if r["dma_len"] else 0
        out.append(
            f"  {avg_pct:>4.1f} {max_pct:>5.1f}  {r['empty_pct']:>5}%  "
            f"{r['full_pct']:>4}%  {r['pcalls']:>7} {r['words']:>7}  "
            f"{r['wspin']:>6} {r['widle']:>6} {r['wpct']:>4.1f}%  "
            f"{r['hyields']:>7} {r['hiters']:>7} {r['hypct']:>4.1f}%"
        )

    # Cache-effectiveness sub-table (Phase 1.1/1.2/1.3 telemetry).
    # Only render if at least one sample carries the new fields — older
    # APKs predate the wiring and would render an all-"n/a" table.
    have_cache_fields = any(
        r["uni_pct"] is not None or r["idx_pct"] is not None
        or r["surf_pct"] is not None
        for r in parsed
    )
    if have_cache_fields:
        out.append("")
        out.append("  Cache hit rates (uniform fast-skip / prim-rewrite LRU / "
                   "surface-part skip)")
        out.append("  uni_hit%  uni_total/s  idx_hit%  idx_total/s  idx_evicts/s  "
                   "surf_skip%  surf_total/s")
        out.append("  " + "-" * 84)
        for r in parsed:
            def _pct(v):
                return f"{v:>5}%" if v is not None else "  n/a"

            def _num(v):
                return f"{v:>9}" if v is not None else "      n/a"

            out.append(
                f"  {_pct(r['uni_pct']):>7}  {_num(r['uni_total']):>10}  "
                f"{_pct(r['idx_pct']):>7}  {_num(r['idx_total']):>10}  "
                f"{_num(r['idx_evicts']):>11}  "
                f"{_pct(r['surf_pct']):>9}  {_num(r['surf_total']):>11}"
            )

    last = parsed[-1]
    out.append("")
    out.append("Diagnosis:")
    if last["full_pct"] >= 30:
        out.append(f"  *** ring_full%={last['full_pct']}% — PFIFO IS THE "
                   f"BOTTLENECK. vCPU is feeding pgraph faster than "
                   f"nv2a.pfifo_thre can drain. Investigate pfifo "
                   f"throughput; CPU optimisations on vCPU won't help.")
    elif last["empty_pct"] >= 50:
        out.append(f"  *** ring_empty%={last['empty_pct']}% — vCPU IS THE "
                   f"PRODUCER GATE. pgraph ring is usually empty; pfifo "
                   f"sits waiting for new commands. vCPU-side perf "
                   f"improvements will help; pfifo isn't the bottleneck.")
    else:
        out.append(f"  --- ring balanced (empty {last['empty_pct']}% / "
                   f"full {last['full_pct']}%). The bottleneck is NOT "
                   f"in the vCPU → pfifo pipeline. Look at audio sync, "
                   f"wall-clock pacing, frame throttle, or GPU render.")

    if last["wpct"] >= 50:
        out.append(f"  * pfifo wait%={last['wpct']:.1f}% — pfifo spends "
                   f"over half its wall blocked on the ring being empty. "
                   f"Confirms vCPU under-feeding.")

    if last["hypct"] >= 10:
        out.append(f"  * vCPU hle_yield%={last['hypct']:.1f}% — KiIdleLoop "
                   f"nanosleeps eat this much wall time. If "
                   f"thread_wait_profile Sleep% is much higher, the "
                   f"difference is `other sleep` (BQL futex waits, "
                   f"audio sync, etc.) — chase that next.")

    if last["widle"] > 0 and last["wspin"] > 0:
        spin_ratio = last["wspin"] * 100 // (last["wspin"] + last["widle"])
        out.append(f"  * pfifo wake mix: {spin_ratio}% spin-wakes / "
                   f"{100-spin_ratio}% cond_wait-wakes. High spin% is good "
                   f"(low-latency producer signal); high cond_wait% means "
                   f"vCPU goes silent for long stretches.")

    if last["pcalls"] > 0:
        words_per_call = last["words"] / max(1, last["pcalls"])
        out.append(f"  * pusher: {words_per_call:.0f} words drained per "
                   f"call, {last['words']:,} words/s total — that's the "
                   f"actual GPU command throughput the pipeline supports.")

    # Phase 1 cache diagnostics.
    if last["idx_pct"] is not None:
        if last["idx_total"] and last["idx_pct"] < 40:
            out.append(f"  * idx_cache_hit%={last['idx_pct']}% — index-rewrite "
                       f"LRU is missing most lookups; key may be wrong or "
                       f"workload not actually repetitive on this scene.")
        elif last["idx_total"]:
            out.append(f"  * idx_cache_hit%={last['idx_pct']}% over "
                       f"{last['idx_total']}/s lookups (evicts "
                       f"{last['idx_evicts']}/s) — Phase 1.2 LRU effective.")
    if last["surf_pct"] is not None:
        if last["surf_total"] and last["surf_pct"] < 5:
            out.append(f"  * surf_skip%={last['surf_pct']}% over "
                       f"{last['surf_total']}/s update_surface_part calls — "
                       f"short-circuit barely firing; check that "
                       f"surface_binding_inputs_gen bump set covers all "
                       f"buffer_dirty write sites.")
        elif last["surf_total"]:
            out.append(f"  * surf_skip%={last['surf_pct']}% over "
                       f"{last['surf_total']}/s update_surface_part calls — "
                       f"Phase 1.3 short-circuit landing.")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# pgraph_method_stats — Phase 2.1 + 2.2 method-class histogram & per-draw p99
# ---------------------------------------------------------------------------
#
# profile.c emits the "xemu-method" tag every ~2 seconds with the format:
#   win=<ms> vertex/s=N(C%%) tex_state/s=N(C%%) shader_state/s=N(C%%) light/s=N(C%%)
#            render_state/s=N(C%%) inline_draw/s=N(C%%) other/s=N(C%%)
#          | draws_per_submit=X.YY
#          | draw_vk_cmd_p99_us=X.XX draw_setup_p99_us=X.XX
#            draw_vtx_attr_p99_us=X.XX draw_vtx_sync_p99_us=X.XX
#            draw_prim_rw_p99_us=X.XX
#            pipe_bind_tex_p99_us=X.XX pipe_bind_shd_p99_us=X.XX
#            pipe_lookup_p99_us=X.XX
#
# `count/s` is methods of that class per second; `(C%)` is the percent of
# total method-class cycles spent in that class (sums to ~100). The p99 values
# are exponential-MA-smoothed max-per-snapshot in microseconds.
_METHOD_CLASSES = [
    ("vertex",       "Vertex data"),
    ("tex_state",    "Texture state"),
    ("shader_state", "Shader state"),
    ("light",        "Lighting"),
    ("render_state", "Render state"),
    ("inline_draw",  "Inline draw"),
    ("other",        "Other"),
]
_METHOD_STAT_RE = re.compile(
    r"win=(?P<win_ms>\d+)_ms\s+"
    r"vertex/s=(?P<v_rate>\d+)\((?P<v_pct>\d+)%\)\s+"
    r"tex_state/s=(?P<t_rate>\d+)\((?P<t_pct>\d+)%\)\s+"
    r"shader_state/s=(?P<s_rate>\d+)\((?P<s_pct>\d+)%\)\s+"
    r"light/s=(?P<l_rate>\d+)\((?P<l_pct>\d+)%\)\s+"
    r"render_state/s=(?P<r_rate>\d+)\((?P<r_pct>\d+)%\)\s+"
    r"inline_draw/s=(?P<i_rate>\d+)\((?P<i_pct>\d+)%\)\s+"
    r"other/s=(?P<o_rate>\d+)\((?P<o_pct>\d+)%\)\s+\|\s+"
    r"draws_per_submit=(?P<dps_int>\d+)\.(?P<dps_frac>\d+)\s+\|\s+"
    r"draw_vk_cmd_p99_us=(?P<p99_vk>[\d.]+)\s+"
    r"draw_setup_p99_us=(?P<p99_setup>[\d.]+)\s+"
    r"draw_vtx_attr_p99_us=(?P<p99_vatt>[\d.]+)\s+"
    r"draw_vtx_sync_p99_us=(?P<p99_vsync>[\d.]+)\s+"
    r"draw_prim_rw_p99_us=(?P<p99_prim>[\d.]+)\s+"
    r"pipe_bind_tex_p99_us=(?P<p99_bindt>[\d.]+)\s+"
    r"pipe_bind_shd_p99_us=(?P<p99_binds>[\d.]+)\s+"
    r"pipe_lookup_p99_us=(?P<p99_lookup>[\d.]+)"
)


@mcp.tool()
def pgraph_method_stats(samples: int = 15) -> str:
    """Phase 2.1 + 2.2: NV2A method-class histogram and per-draw phase p99.

    Phase 1's caches addressed cache-miss CPU costs in the pgraph slow
    path. To decide whether Phase 3's producer/consumer thread split is
    worthwhile we need to know:
      (a) Concentration — is one method class >50% of pfifo cycles? If
          yes, the split has high yield. Spread across 4+ classes means
          lower yield from a single structural change.
      (b) Encode-vs-parse ratio — `draw_vk_cmd_p99_us` relative to the
          other per-draw phases. High vk_cmd p99 + high
          `draws_per_submit` means a non-trivial chunk of pfifo cost is
          pure Vulkan encode, which IS splittable. Low vk_cmd p99 means
          parse/setup dominates and the split won't move the needle.

    Drains recent `xemu-method` logcat lines and renders:
      * Method-class histogram with count/s and cycle share per class
      * Per-draw phase p99 (microseconds, smoothed max)
      * draws_per_submit ratio

    Args:
      samples: how many recent xemu-method windows to display
               (default 15 = last ~30s; pass 60 for the last ~2 min).
    """
    if not _device():
        return "No device connected."
    pid = _emu_pid()
    if not pid:
        return f"{PKG}:xemu is not running. Launch it before sampling."
    if samples <= 0:
        return "samples must be > 0."

    raw = _sh(f"logcat -d -s xemu-method:I --pid={pid}")
    lines = [ln for ln in (raw or "").splitlines() if "win=" in ln]
    if not lines:
        return ("No xemu-method lines yet. Either the emulator hasn't "
                "reached FLIP_STALL (still on the boot animation) or the "
                "running build predates the Phase 2.1/2.2 instrumentation "
                "in hw/xbox/nv2a/pgraph/profile.c — rebuild + reinstall.")

    parsed = []
    for ln in lines[-samples:]:
        m = _METHOD_STAT_RE.search(ln)
        if not m:
            continue
        d = m.groupdict()
        parsed.append({
            "win_ms":    int(d["win_ms"]),
            "rates":     {
                "vertex":       int(d["v_rate"]),
                "tex_state":    int(d["t_rate"]),
                "shader_state": int(d["s_rate"]),
                "light":        int(d["l_rate"]),
                "render_state": int(d["r_rate"]),
                "inline_draw":  int(d["i_rate"]),
                "other":        int(d["o_rate"]),
            },
            "pcts":      {
                "vertex":       int(d["v_pct"]),
                "tex_state":    int(d["t_pct"]),
                "shader_state": int(d["s_pct"]),
                "light":        int(d["l_pct"]),
                "render_state": int(d["r_pct"]),
                "inline_draw":  int(d["i_pct"]),
                "other":        int(d["o_pct"]),
            },
            "dps":       float(f"{d['dps_int']}.{d['dps_frac']}"),
            "p99_vk":    float(d["p99_vk"]),
            "p99_setup": float(d["p99_setup"]),
            "p99_vatt":  float(d["p99_vatt"]),
            "p99_vsync": float(d["p99_vsync"]),
            "p99_prim":  float(d["p99_prim"]),
            "p99_bindt": float(d["p99_bindt"]),
            "p99_binds": float(d["p99_binds"]),
            "p99_lookup":float(d["p99_lookup"]),
        })
    if not parsed:
        return ("Found xemu-method lines but none matched the expected "
                "shape. Check the format in hw/xbox/nv2a/pgraph/profile.c "
                "vs _METHOD_STAT_RE in this tool.")

    out = []
    out.append(f"Last {len(parsed)} xemu-method samples (each window ~2s):")
    out.append("")
    out.append("  Method-class histogram (averaged across window)")
    out.append("  " + "-" * 76)
    # Average across samples for the headline table.
    n = len(parsed)
    avg_rate = {k: 0 for k, _ in _METHOD_CLASSES}
    avg_pct  = {k: 0 for k, _ in _METHOD_CLASSES}
    for p in parsed:
        for k, _ in _METHOD_CLASSES:
            avg_rate[k] += p["rates"][k]
            avg_pct[k]  += p["pcts"][k]
    for k, _ in _METHOD_CLASSES:
        avg_rate[k] //= n
        avg_pct[k]  //= n

    out.append(f"  {'Class':<14}  {'count/s':>10}  {'cycle %':>8}")
    for k, label in _METHOD_CLASSES:
        bar = "#" * (avg_pct[k] // 2)
        out.append(f"  {label:<14}  {avg_rate[k]:>10}  {avg_pct[k]:>6}%  {bar}")

    # Per-sample p99 table (last N samples)
    out.append("")
    out.append("  Per-draw phase p99 (microseconds, smoothed max per "
               "snapshot)  +  draws/submit")
    out.append("  win  vk_cmd  setup  vtx_att vtx_sync prim_rw "
               "bind_tex bind_shd pipe_lkp  dps")
    out.append("  " + "-" * 84)
    for p in parsed:
        out.append(
            f"  {p['win_ms']:>4} "
            f"{p['p99_vk']:>6.2f} {p['p99_setup']:>6.2f} "
            f"{p['p99_vatt']:>7.2f} {p['p99_vsync']:>7.2f} "
            f"{p['p99_prim']:>7.2f} {p['p99_bindt']:>7.2f} "
            f"{p['p99_binds']:>7.2f} {p['p99_lookup']:>8.2f} "
            f"{p['dps']:>5.2f}"
        )

    # Diagnosis: identify the dominant class + the dominant per-draw phase.
    last = parsed[-1]
    out.append("")
    out.append("Diagnosis:")
    sorted_classes = sorted(
        [(label, last["pcts"][k]) for k, label in _METHOD_CLASSES],
        key=lambda x: x[1],
        reverse=True,
    )
    top_label, top_pct = sorted_classes[0]
    if top_pct >= 50:
        out.append(f"  *** {top_label} dominates at {top_pct}% of pfifo "
                   f"cycles — single-class concentration favours Phase 3 "
                   f"structural split or class-targeted fast-path.")
    elif top_pct >= 30:
        out.append(f"  --- {top_label} leads at {top_pct}% — moderate "
                   f"concentration; consider class-targeted optimisations "
                   f"before a structural split.")
    else:
        out.append(f"  --- pfifo cycles spread evenly across classes "
                   f"(top: {top_label} at {top_pct}%) — Phase 3 split "
                   f"yields will be lower; chase the largest per-draw "
                   f"phase instead.")

    # Encode-vs-parse ratio
    phases = [
        ("draw_vk_cmd",    last["p99_vk"]),
        ("draw_setup",     last["p99_setup"]),
        ("draw_vtx_attr",  last["p99_vatt"]),
        ("draw_vtx_sync",  last["p99_vsync"]),
        ("draw_prim_rw",   last["p99_prim"]),
        ("pipe_bind_tex",  last["p99_bindt"]),
        ("pipe_bind_shd",  last["p99_binds"]),
        ("pipe_lookup",    last["p99_lookup"]),
    ]
    total_per_draw = sum(v for _, v in phases) or 1.0
    vk_cmd_share = last["p99_vk"] / total_per_draw
    out.append(f"  * draw_vk_cmd_p99={last['p99_vk']:.2f} us "
               f"({vk_cmd_share*100:.0f}% of per-draw p99 sum) — "
               f"{'encode-heavy' if vk_cmd_share >= 0.30 else 'parse-heavy'}")
    if vk_cmd_share >= 0.30 and last["dps"] >= 20:
        out.append(f"    + draws_per_submit={last['dps']:.2f} (>= 20) — "
                   f"Phase 3 encode-thread split has favourable conditions.")
    elif vk_cmd_share < 0.20:
        out.append(f"    + Phase 3 split won't help — encode portion is "
                   f"<20% of per-draw cost; attack parse/setup instead.")
    elif last["dps"] < 10:
        out.append(f"    + draws_per_submit={last['dps']:.2f} (<10) — "
                   f"Phase 3 split's payoff is limited by small batches.")

    # Pick the largest per-draw phase that ISN'T draw_vk_cmd as a target
    # if encode-vs-parse leans parse.
    if vk_cmd_share < 0.30:
        non_vk = sorted(
            [p for p in phases if p[0] != "draw_vk_cmd"],
            key=lambda x: x[1], reverse=True,
        )
        if non_vk:
            name, val = non_vk[0]
            out.append(f"  * largest non-encode phase: {name} "
                       f"p99={val:.2f} us — attack here first.")

    return "\n".join(out)


# Audio-trace log line shape from hw/xbox/mcpx/apu/apu.c:
#   prod : hakuX-apu-prod : t=<ns> q_pre=<bytes> q_post=<bytes> peak=<int> work_us=<us> div=<frame_div>
#   cons : hakuX-apu-cons : t=<ns> req=<bytes> avail=<bytes> copied=<bytes> underrun=<0|1>
#   throt: hakuX-apu-throt: t=<ns> reason=deadline qb=<bytes> slip_us=<int> limit_us=<int>
# Enabled by $X1BOX_AUDIO_TRACE=1 OR by touch <ext>/x1box/audio_trace.flag.
# The file-based gate is the easy-to-flip path — no rebuild required.
_AUDIO_PROD_RE  = re.compile(r"t=(\d+)\s+q_pre=(\d+)\s+q_post=(\d+)\s+peak=(\d+)\s+work_us=(\d+)\s+div=(\d+)")
_AUDIO_CONS_RE  = re.compile(
    r"t=(\d+)\s+req=(\d+)\s+avail=(\d+)\s+copied=(\d+)\s+underrun=([01])"
    r"(?:\s+dur_us=(\d+))?"
)
_AUDIO_FFWAIT_RE = re.compile(
    r"t=(\d+)\s+kind=ffwait\s+wait_us=(\d+)\s+qb_entry=(\d+)\s+"
    r"qb_exit=(\d+)\s+iters=(\d+)"
)
_AUDIO_THROT_RE = re.compile(r"t=(\d+)\s+reason=(\w+)\s+qb=(-?\d+)\s+slip_us=(-?\d+)\s+limit_us=(\d+)")
_AUDIO_VDISP_RE = re.compile(r"t=(\d+)\s+total=(\d+)\s+silent_env=(\d+)\s+silent_vol=(\d+)\s+processed=(\d+)\s+work_us=(\d+)\s+mode=(\w+)")
# hakuX-apu-bql is emitted only on slow acquisitions/runs (filter is in C).
# Two shapes:
#   kind=bql_lock     acq_us=<int>             — bql_lock held for >1ms by vCPU
#   kind=dlock_post_irq acq_us=<int>           — d->lock reacquire after IRQ held >1ms
#   kind=se_or_dlock  se_us=<int> dlock_us=<int> — se_frame >5ms OR dlock reacquire >1ms
_AUDIO_BQL_RE_SIMPLE = re.compile(r"t=(\d+)\s+kind=(bql_lock|dlock_post_irq)\s+acq_us=(\d+)")
_AUDIO_BQL_RE_SED    = re.compile(r"t=(\d+)\s+kind=se_or_dlock\s+se_us=(\d+)\s+dlock_us=(\d+)")


def _audio_trace_flag_paths() -> list[str]:
    """Both possible flag-file locations — standard + perftest packages.
    The C code's android_x1box_ext_dir() reads /proc/self/cmdline to pick
    the right one at runtime; here we return both so the MCP user can
    enable for whichever variant they happen to be running."""
    return [
        "/storage/emulated/0/Android/data/com.izzy2lost.x1box/files/x1box/audio_trace.flag",
        "/storage/emulated/0/Android/data/com.izzy2lost.x1box.perftest/files/x1box/audio_trace.flag",
    ]


@mcp.tool()
def audio_trace_enable() -> str:
    """Enable APU producer/consumer/throttle tracing without a rebuild.

    Creates `<ext>/x1box/audio_trace.flag` in BOTH the standard and
    perftest data dirs so whichever variant launches next picks it up.
    The C-side check `audio_trace_enabled()` stats this file at first
    call per process and caches the result, so a re-launch of :xemu is
    required for the flag change to take effect (force_stop the package
    or restart from the launcher).

    Once enabled, ~17 KB/s of logcat is emitted under tags
    `hakuX-apu-prod`, `hakuX-apu-cons`, `hakuX-apu-throt`. Drain via
    `audio_trace_analyze(duration_s)`.
    """
    if not _device(): return "No device connected."
    # IMPORTANT history: a plain `touch` from the adb shell uid on the
    # external path creates a file the app uid CANNOT stat() — FUSE/
    # sdcardfs remaps writes from non-app uids and gates app-side reads
    # of those files (permission denied even with mode 660). Writing via
    # `run-as <pkg>` to external storage *also* fails because the FUSE
    # layer rejects the write context.
    #
    # The C-side now (audio_trace.h) checks BOTH:
    #   1. <ext>/x1box/audio_trace.flag (matches frame_stats.flag pattern)
    #   2. /data/data/<pkg>/files/audio_trace.flag (this tool's path)
    # We write the INTERNAL path via run-as, which works reliably for
    # any debuggable build, and the C code finds it on check #2.
    created = []
    failed  = []
    packages = ["com.izzy2lost.x1box", "com.izzy2lost.x1box.perftest"]
    for pkg in packages:
        # /data/data/<pkg>/files/ is owned by the app uid; run-as has
        # full uid context there. Create files/ if missing (it's the
        # normal AppContext.getFilesDir() target, but might not exist
        # until the app has actually written to it).
        path = f"/data/data/{pkg}/files/audio_trace.flag"
        result = _sh(
            f"run-as {pkg} mkdir -p /data/data/{pkg}/files 2>&1; "
            f"run-as {pkg} touch {path} 2>&1; "
            f"run-as {pkg} ls -la {path} 2>&1"
        )
        if result and "audio_trace.flag" in result and "denied" not in result.lower():
            created.append(f"{path}  (uid={pkg})")
        else:
            failed.append(f"{path}: {result.strip() or 'unknown error'}")
    out = ["=== audio_trace_enable ==="]
    for p in created:
        out.append(f"  set: {p}")
    for f in failed:
        out.append(f"  fail: {f}")
    if not created:
        out.append("No flags set. The packages may not be debuggable, or "
                   "the files dir doesn't exist (launch the app once first).")
    out.append("\nNOTE: re-launch :xemu (force_stop or relaunch from "
               "launcher) so the cached `audio_trace_enabled` flag picks "
               "up the new state.")
    return "\n".join(out)


@mcp.tool()
def audio_trace_disable() -> str:
    """Remove the audio_trace.flag from both data dirs.

    The C-side flag is process-cached at first frame, so a running :xemu
    will continue emitting trace lines until restart even after the flag
    is gone. Re-launch to actually stop the spam.
    """
    if not _device(): return "No device connected."
    out = ["=== audio_trace_disable ==="]
    # Remove from the internal-data path (where audio_trace_enable now
    # writes) AND clean up any stale flag in the external path from
    # older versions of this tool.
    for pkg in ["com.izzy2lost.x1box", "com.izzy2lost.x1box.perftest"]:
        internal_path = f"/data/data/{pkg}/files/audio_trace.flag"
        external_path = (f"/storage/emulated/0/Android/data/{pkg}/files/"
                         f"x1box/audio_trace.flag")
        _sh(f"run-as {pkg} rm -f {internal_path} 2>&1")
        _sh(f"rm -f {external_path} 2>&1")
        out.append(f"  rm -f {internal_path}")
        out.append(f"  rm -f {external_path}")
    out.append("\nNOTE: re-launch :xemu to stop emission. Running process "
               "has the enabled state cached for its lifetime.")
    return "\n".join(out)


@mcp.tool()
def audio_trace_analyze(duration: float = 10.0, package: str = "") -> str:
    """Diagnose audio-stutter cause from on-device APU trace logs.

    Enable on-device first by setting X1BOX_AUDIO_TRACE=1 in the running
    process's env. The simplest path is to add it to the launcher's
    setenv list in xemu_settings_android.cc — once flipped, every APU
    frame push, every audio-callback drain, and every throttle catch-up
    snap emits a structured line under tags `hakuX-apu-{prod,cons,throt}`.

    This tool:
      1. Clears logcat (loses unrelated history — accept).
      2. Sleeps `duration` seconds.
      3. Drains the three tags and parses every line.
      4. Reports the *cause* of any audio stutter visible in the window:
            - Underrun count + cadence (consumer copied < requested).
            - Throttle snap count (producer slipped past 43ms catchup
              window, dropping audio frames).
            - Slow-frame count (producer work_us > EP_FRAME_US=5333).
            - Silent-but-claimed frames (peak == 0 surrounded by audible
              frames — points at fast-path mis-silencing).
            - Producer inter-push gap percentiles (regular = healthy).
            - Consumer inter-callback gap percentiles (AAudio burst-size
              jitter shows up here).

    A "clip + blank @ 250-500ms" stutter pattern is conclusively
    identified by:
      - Underrun events on the consumer side at ~3-4 Hz cadence
      - Throttle snaps in the same window, OR producer inter-push gaps
        > 6ms in the same window
      - The PRODUCER's `q_post` at the snap-or-slow moment shows the
        FIFO state at the moment of the gap

    Args:
        duration: sampling window in seconds (default 10). 5-30 is sane.
        package: 'standard' (com.izzy2lost.x1box, default), 'perftest'
            (com.izzy2lost.x1box.perftest), or empty to auto-detect.

    Returns a multi-section human-readable report.
    """
    if not _device():
        return "No device connected."
    if duration <= 0:
        return "duration must be > 0."

    # Verify a target process is running before we clear logcat. The
    # caller wants the *xemu* subprocess specifically since the trace
    # is emitted from it, not the launcher.
    if not _emu_pid():
        return ("xemu emulation process not running. Launch a game first.\n"
                "Then enable tracing by ensuring the running APK has\n"
                "X1BOX_AUDIO_TRACE=1 set (via xemu_settings_android.cc\n"
                "setenv, or by directly exporting it before launch).")

    _adb("logcat", "-c")
    time.sleep(duration)
    raw = _sh(
        "logcat -d -s hakuX-apu-prod:I hakuX-apu-cons:I "
        "hakuX-apu-throt:W hakuX-apu-vdisp:I hakuX-apu-bql:W"
    )
    if not raw:
        return "logcat returned nothing — trace flag may not be enabled."

    prod: list[tuple[int, int, int, int, int, int]] = []  # t, q_pre, q_post, peak, work_us, div
    cons: list[tuple[int, int, int, int, int, int]] = []  # t, req, avail, copied, underrun, dur_us
    throt: list[tuple[int, str, int, int, int]] = []     # t, reason, qb, slip_us, limit_us
    vdisp: list[tuple[int, int, int, int, int, int, str]] = []  # t, total, silent_env, silent_vol, processed, work_us, mode
    bql: list[tuple[int, str, int]] = []                # t, kind, acq_us  (bql_lock | dlock_post_irq | se | dlock)

    for ln in raw.splitlines():
        m = _AUDIO_PROD_RE.search(ln)
        if m:
            prod.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                         int(m.group(4)), int(m.group(5)), int(m.group(6))))
            continue
        m = _AUDIO_CONS_RE.search(ln)
        if m:
            dur_us = int(m.group(6)) if m.group(6) else -1
            cons.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                         int(m.group(4)), int(m.group(5)), dur_us))
            continue
        m = _AUDIO_FFWAIT_RE.search(ln)
        if m:
            bql.append((int(m.group(1)), "ffwait", int(m.group(2))))
            continue
        m = _AUDIO_THROT_RE.search(ln)
        if m:
            throt.append((int(m.group(1)), m.group(2), int(m.group(3)),
                          int(m.group(4)), int(m.group(5))))
            continue
        m = _AUDIO_VDISP_RE.search(ln)
        if m:
            vdisp.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                          int(m.group(4)), int(m.group(5)), int(m.group(6)),
                          m.group(7)))
            continue
        m = _AUDIO_BQL_RE_SIMPLE.search(ln)
        if m:
            bql.append((int(m.group(1)), m.group(2), int(m.group(3))))
            continue
        m = _AUDIO_BQL_RE_SED.search(ln)
        if m:
            t = int(m.group(1))
            se_us = int(m.group(2))
            dlock_us = int(m.group(3))
            if se_us > 5000:
                bql.append((t, "se_frame_slow", se_us))
            if dlock_us > 1000:
                bql.append((t, "dlock_post_se", dlock_us))

    if not prod and not cons and not throt and not vdisp and not bql:
        return (f"Got 0 trace events in {duration:.1f}s.\n"
                f"Either X1BOX_AUDIO_TRACE is not set in the running\n"
                f"process, or the build doesn't have the trace instrumentation.\n"
                f"To enable: add `setenv(\"X1BOX_AUDIO_TRACE\", \"1\", 1)`\n"
                f"to android/app/src/main/cpp/xemu_settings_android.cc,\n"
                f"rebuild, install, relaunch.")

    out: list[str] = []
    out.append("=== Audio trace analysis ===")
    out.append(f"Window: {duration:.1f}s   prod={len(prod)}  cons={len(cons)}  "
               f"throt={len(throt)}  vdisp={len(vdisp)}  bql_stalls={len(bql)}")

    # --- Underruns (consumer ran out of FIFO data) ---
    underruns = [c for c in cons if c[4] == 1]
    if underruns:
        # Compute spacing between underruns to detect cadence.
        if len(underruns) > 1:
            gaps_ms = [(underruns[i+1][0] - underruns[i][0]) / 1e6
                       for i in range(len(underruns) - 1)]
            gaps_ms.sort()
            p50_gap = _percentile(gaps_ms, 50)
            p95_gap = _percentile(gaps_ms, 95)
            mean_rate = len(underruns) / duration
            out.append(f"\n>>> UNDERRUNS: {len(underruns)} ({mean_rate:.2f}/s)  "
                       f"gap_ms p50={p50_gap:.0f}  p95={p95_gap:.0f}")
        else:
            out.append(f"\n>>> UNDERRUNS: 1 (insufficient for cadence)")
        # Sample the first few underruns to show shortfall sizes.
        for i, c in enumerate(underruns[:5]):
            t, req, avail, copied, _, _ = c
            short = req - copied
            out.append(f"  [{i}] t={t} req={req}B avail={avail}B copied={copied}B "
                       f"shortfall={short}B ({short/(req or 1)*100:.0f}%)")
        if len(underruns) > 5:
            out.append(f"  ...({len(underruns)-5} more)")
    else:
        out.append("\nUNDERRUNS: none — consumer was always fed in this window.")

    # --- Throttle catch-up snaps (producer lost time) ---
    if throt:
        snap_rate = len(throt) / duration
        slips_us = sorted(t[3] for t in throt)
        out.append(f"\n>>> THROTTLE SNAPS: {len(throt)} ({snap_rate:.2f}/s) — "
                   f"producer slipped past catch-up budget")
        out.append(f"  slip_us p50={_percentile(slips_us, 50):.0f}  "
                   f"p95={_percentile(slips_us, 95):.0f}  "
                   f"max={slips_us[-1]:.0f}")
        for i, ev in enumerate(throt[:5]):
            t, reason, qb, slip, limit = ev
            out.append(f"  [{i}] t={t} reason={reason} qb={qb}B "
                       f"slip={slip}us limit={limit}us")
        if len(throt) > 5:
            out.append(f"  ...({len(throt)-5} more)")
    else:
        out.append("\nTHROTTLE SNAPS: none — producer kept pace in this window.")

    # --- Producer inter-push gap analysis ---
    if len(prod) > 1:
        prod_sorted = sorted(prod, key=lambda x: x[0])
        deltas_ms = [(prod_sorted[i+1][0] - prod_sorted[i][0]) / 1e6
                     for i in range(len(prod_sorted) - 1)]
        deltas_ms.sort()
        # Producer should push every EP_FRAME_US = 5.333ms when not throttled.
        # Anything > 6 ms is a slip; > 10 ms is severe.
        slips = [d for d in deltas_ms if d > 6.0]
        severe = [d for d in deltas_ms if d > 10.0]
        out.append(f"\n--- Producer inter-push gaps (target ~5.33ms) ---")
        out.append(f"  p50={_percentile(deltas_ms, 50):.2f}ms  "
                   f"p95={_percentile(deltas_ms, 95):.2f}ms  "
                   f"p99={_percentile(deltas_ms, 99):.2f}ms  "
                   f"max={deltas_ms[-1]:.2f}ms")
        out.append(f"  slips>6ms: {len(slips)}  severe>10ms: {len(severe)}")

        # Work-time analysis: how long was se_frame busy each push?
        works_us = sorted(p[4] for p in prod)
        out.append(f"--- Producer work_us (EP_FRAME_US=5333) ---")
        out.append(f"  p50={_percentile(works_us, 50):.0f}us  "
                   f"p95={_percentile(works_us, 95):.0f}us  "
                   f"p99={_percentile(works_us, 99):.0f}us  "
                   f"max={works_us[-1]:.0f}us")
        slow = [w for w in works_us if w > 5333]
        if slow:
            out.append(f"  SLOW FRAMES (work_us > 5333): {len(slow)} "
                       f"({len(slow)*100/len(works_us):.1f}% of total)")

        # FIFO occupancy at push time (q_post tells us if we're refilling
        # from low watermark — points at recent underrun recovery).
        q_posts = sorted(p[2] for p in prod)
        out.append(f"--- FIFO occupancy at push time ---")
        out.append(f"  q_post p5={_percentile(q_posts, 5):.0f}B  "
                   f"p50={_percentile(q_posts, 50):.0f}B  "
                   f"p95={_percentile(q_posts, 95):.0f}B")

    # --- Silent-frame detection ---
    # A 'suspicious silent' frame = peak==0 within a run of audible frames
    # (audible neighbors on both sides). This is the signature of voice
    # processing returning zeros for a frame that should have had audio.
    if len(prod) > 2:
        prod_sorted = sorted(prod, key=lambda x: x[0])
        peaks = [p[3] for p in prod_sorted]
        suspicious = []
        for i in range(1, len(peaks) - 1):
            if peaks[i] == 0 and peaks[i-1] > 100 and peaks[i+1] > 100:
                suspicious.append((prod_sorted[i][0], peaks[i-1], peaks[i+1]))
        total_silent = sum(1 for p in peaks if p == 0)
        out.append(f"\n--- Silent frames (peak==0) ---")
        out.append(f"  total: {total_silent} / {len(peaks)} "
                   f"({total_silent*100/len(peaks):.1f}%)")
        if suspicious:
            out.append(f"  SUSPICIOUS (silent between audible neighbors): "
                       f"{len(suspicious)}")
            for i, (t, prev_peak, next_peak) in enumerate(suspicious[:5]):
                out.append(f"    [{i}] t={t} prev_peak={prev_peak} "
                           f"next_peak={next_peak}")

    # --- Lock-acquisition stalls (the actual stall source if it's lock-related) ---
    if bql:
        by_kind: dict[str, list[int]] = {}
        for t, kind, acq_us in bql:
            by_kind.setdefault(kind, []).append(acq_us)
        out.append(f"\n>>> LOCK-STALL EVENTS: {len(bql)} total")
        for kind, vals in sorted(by_kind.items()):
            vs = sorted(vals)
            rate = len(vs) / duration
            out.append(f"  kind={kind:18s} n={len(vs):4d} ({rate:.2f}/s)  "
                       f"p50={_percentile(vs, 50):.0f}us  "
                       f"p95={_percentile(vs, 95):.0f}us  "
                       f"max={vs[-1]}us")
        # Show the top 5 longest events as full traces.
        bql_sorted = sorted(bql, key=lambda x: -x[2])
        out.append("  longest stalls:")
        for i, (t, kind, acq_us) in enumerate(bql_sorted[:5]):
            out.append(f"    [{i}] t={t} kind={kind} acq={acq_us}us "
                       f"(={acq_us/1000:.1f}ms)")
    else:
        out.append("\nLOCK-STALL EVENTS: none above thresholds "
                   "(>1ms bql/d->lock, >5ms se_frame).")

    # --- Voice dispatch breakdown (silent-voice fast-path elision rate) ---
    if vdisp:
        vd_sorted = sorted(vdisp, key=lambda x: x[0])
        totals = [v[1] for v in vd_sorted]
        env_skips = [v[2] for v in vd_sorted]
        vol_skips = [v[3] for v in vd_sorted]
        processed = [v[4] for v in vd_sorted]
        vd_works = [v[5] for v in vd_sorted]
        total_sum = sum(totals)
        env_sum = sum(env_skips)
        vol_sum = sum(vol_skips)
        proc_sum = sum(processed)
        out.append(f"\n--- Voice dispatch (per-frame, mode={vd_sorted[-1][6]}) ---")
        out.append(f"  frames: {len(vd_sorted)}   "
                   f"voices total: {total_sum}   "
                   f"~{total_sum/max(1,len(vd_sorted)):.1f}/frame")
        if total_sum > 0:
            out.append(f"  silent_env: {env_sum} ({env_sum*100/max(1,total_sum):.1f}%)  "
                       f"silent_vol: {vol_sum} ({vol_sum*100/max(1,total_sum):.1f}%)  "
                       f"processed: {proc_sum} ({proc_sum*100/max(1,total_sum):.1f}%)")
        works_sorted = sorted(vd_works)
        out.append(f"  vdisp work_us p50={_percentile(works_sorted, 50):.0f}  "
                   f"p95={_percentile(works_sorted, 95):.0f}  "
                   f"max={works_sorted[-1] if works_sorted else 0}")

        # If silent-voice elision rate spikes briefly, that's the
        # smoking-gun for clip+blank stutter. Find frames where
        # (silent_env + silent_vol) / total > 90% surrounded by frames
        # where it's < 50%.
        silent_spikes = []
        for i in range(1, len(vd_sorted) - 1):
            tot, se, sv = vd_sorted[i][1], vd_sorted[i][2], vd_sorted[i][3]
            if tot < 4:
                continue
            ratio = (se + sv) / tot
            prev_tot = vd_sorted[i-1][1]
            next_tot = vd_sorted[i+1][1]
            prev_ratio = ((vd_sorted[i-1][2] + vd_sorted[i-1][3]) /
                          prev_tot) if prev_tot > 0 else 0.0
            next_ratio = ((vd_sorted[i+1][2] + vd_sorted[i+1][3]) /
                          next_tot) if next_tot > 0 else 0.0
            if ratio > 0.9 and prev_ratio < 0.5 and next_ratio < 0.5:
                silent_spikes.append((vd_sorted[i][0], tot, se, sv,
                                      vd_sorted[i][4]))
        if silent_spikes:
            out.append(f"  SILENT-ELISION SPIKES (>90% elided between "
                       f"<50% neighbors): {len(silent_spikes)}")
            for i, (t, tot, se, sv, proc) in enumerate(silent_spikes[:5]):
                out.append(f"    [{i}] t={t} total={tot} env={se} vol={sv} "
                           f"processed={proc}")

    # --- Cross-correlation: underrun moments vs voice dispatch state ---
    # For each underrun, find the closest vdisp event (within ±5ms) and
    # report what was happening voice-wise. If underruns consistently
    # follow high-elision frames, the silent-voice gate is the cause.
    if underruns and vdisp:
        vd_sorted = sorted(vdisp, key=lambda x: x[0])
        vd_times = [v[0] for v in vd_sorted]
        out.append("\n--- Underrun-vs-vdisp correlation ---")
        elide_at_underrun = []
        for ur in underruns[:20]:
            ur_t = ur[0]
            # Binary search would be cleaner; len(vd_times) is small enough.
            best = min(vd_sorted, key=lambda v: abs(v[0] - ur_t))
            if abs(best[0] - ur_t) <= 5_000_000:  # within 5ms
                tot = best[1]
                elide_pct = ((best[2] + best[3]) / tot * 100) if tot > 0 else 0
                elide_at_underrun.append(elide_pct)
        if elide_at_underrun:
            elide_at_underrun.sort()
            out.append(f"  elide% at underrun moments (n={len(elide_at_underrun)}): "
                       f"p50={_percentile(elide_at_underrun, 50):.1f}  "
                       f"max={elide_at_underrun[-1]:.1f}")

    # --- Consumer inter-callback gap analysis ---
    if len(cons) > 1:
        cons_sorted = sorted(cons, key=lambda x: x[0])
        deltas_ms = sorted((cons_sorted[i+1][0] - cons_sorted[i][0]) / 1e6
                           for i in range(len(cons_sorted) - 1))
        out.append(f"\n--- Consumer inter-callback gaps ---")
        out.append(f"  p50={_percentile(deltas_ms, 50):.2f}ms  "
                   f"p95={_percentile(deltas_ms, 95):.2f}ms  "
                   f"p99={_percentile(deltas_ms, 99):.2f}ms  "
                   f"max={deltas_ms[-1]:.2f}ms")

        # Callback duration: if dur_us is available, this tells us how
        # much time WE spent inside the callback. Combined with the
        # inter-callback gap, this discriminates "audio backend isn't
        # calling us" (long gap, short dur) from "our code is slow"
        # (long dur).
        durs_us = [c[5] for c in cons_sorted if c[5] >= 0]
        if durs_us:
            durs_sorted = sorted(durs_us)
            slow_cb = [d for d in durs_us if d > 5000]  # >5ms is slow
            out.append(f"--- Consumer callback duration (time inside our cb) ---")
            out.append(f"  p50={_percentile(durs_sorted, 50):.0f}us  "
                       f"p95={_percentile(durs_sorted, 95):.0f}us  "
                       f"p99={_percentile(durs_sorted, 99):.0f}us  "
                       f"max={durs_sorted[-1]}us")
            if slow_cb:
                out.append(f"  SLOW CALLBACKS (dur > 5ms): {len(slow_cb)}")

            # Discriminate cause of long inter-callback gaps. If the
            # NEXT callback's dur is short but the gap before it was
            # long, the audio backend held us off (silence happens
            # because no audio was demanded). If dur is large the
            # callback itself stalled.
            backend_gates = 0
            self_stalls = 0
            for i in range(1, len(cons_sorted)):
                gap_ms = (cons_sorted[i][0] - cons_sorted[i-1][0]) / 1e6
                if gap_ms < 50:  # threshold for "long gap"
                    continue
                prev_dur_ms = cons_sorted[i-1][5] / 1000.0 \
                              if cons_sorted[i-1][5] >= 0 else 0
                if prev_dur_ms < 5.0:
                    backend_gates += 1
                else:
                    self_stalls += 1
            if backend_gates or self_stalls:
                out.append(f"  >50ms gaps: backend-gated={backend_gates}  "
                           f"self-stalled={self_stalls}")

    # --- Verdict ---
    out.append("\n=== Verdict ===")
    bql_kinds = {k for _, k, _ in bql}
    has_bql_stall = any(k == "bql_lock" for _, k, _ in bql)
    has_dlock_stall = any(k.startswith("dlock") for _, k, _ in bql)
    has_se_slow = any(k == "se_frame_slow" for _, k, _ in bql)
    has_ffwait = any(k == "ffwait" for _, k, _ in bql)
    # Magnitude check: pick the biggest single stall and compare to
    # observed producer-gap p99. If max bql_lock < 1/3 of p99 gap, BQL
    # contention can't be the primary cause.
    max_bql_acq_us = max([x[2] for x in bql if x[1] == "bql_lock"], default=0)
    max_ffwait_us = max([x[2] for x in bql if x[1] == "ffwait"], default=0)
    prod_p99_us = 0
    if len(prod) > 1:
        prod_sorted = sorted(prod, key=lambda x: x[0])
        gaps_us = sorted((prod_sorted[i+1][0] - prod_sorted[i][0]) / 1000
                         for i in range(len(prod_sorted) - 1))
        prod_p99_us = int(_percentile(gaps_us, 99))

    if not underruns and not throt:
        out.append("Audio path looks CLEAN in this window. No underruns and "
                   "no throttle snaps. If the user reports stutter, it is "
                   "either intermittent (try a longer window) or downstream "
                   "of this code path (Android audio HAL / openslES burst).")
    elif has_ffwait and max_ffwait_us > prod_p99_us * 0.5:
        out.append("PRIMARY CAUSE: Consumer (audio backend) is gating the FIFO.")
        out.append(f"  Producer enters throttle's FIFO-full wait loop for up to "
                   f"{max_ffwait_us}us at a time — that's the FIFO being full "
                   f"because the audio callback isn't draining it.")
        out.append("  Check the consumer 'backend-gated' vs 'self-stalled' "
                   "counts above. If 'backend-gated' dominates, OpenSL/AAudio "
                   "isn't calling us often enough — that's the audio backend "
                   "buffer config, not xemu.")
        out.append("  Fix path: shrink the host-side audio device buffer "
                   "(XEMU_ANDROID_AUDIO_SAMPLES env) so the backend calls more "
                   "often, or pump frames in a non-callback driven loop.")
    elif has_bql_stall and max_bql_acq_us > prod_p99_us / 3:
        out.append("PRIMARY CAUSE: BQL contention from vCPU.")
        out.append(f"  Underruns coincide with bql_lock acquisitions up to "
                   f"{max_bql_acq_us}us; producer p99 gap is {prod_p99_us}us.")
        out.append("  The vCPU is holding BQL during MMIO emulation / DMA / TB "
                   "execution longer than the audio frame budget allows.")
        out.append("  Fix path: reduce BQL hold duration on vCPU. Candidates:")
        out.append("    * lower CHAIN_MAX in cranelift_chain_continue further")
        out.append("    * drop BQL more aggressively in nv2a MMIO write paths")
        out.append("    * check for a specific guest MMIO that's pinning BQL")
    elif has_bql_stall:
        out.append(f"BQL stalls exist ({max_bql_acq_us}us max) but are too "
                   f"small to account for the {prod_p99_us}us producer p99 gap.")
        out.append("  The producer's stall is happening somewhere else — most "
                   "likely waiting for the audio callback (consumer) to drain "
                   "the FIFO. Check the consumer gap p95/p99 and "
                   "backend-gated count.")
    elif has_dlock_stall:
        out.append("PRIMARY CAUSE: d->lock contention.")
        out.append("  apu_thread is waiting on d->lock — something else holds it.")
        out.append("  Audit: voice_lock paths, monitor_sink_cb, MMIO write handlers.")
    elif has_se_slow:
        out.append("PRIMARY CAUSE: se_frame execution exceeds 5ms occasionally.")
        out.append("  Voice processing or DSP frame work is the actual slow path.")
    elif underruns and throt:
        out.append("Producer is FALLING BEHIND. Throttle snaps coincide with "
                   "FIFO drainage but no lock-stall events were captured. "
                   "Either the stalls are sub-1ms accumulating (try lower "
                   "threshold) or stalls live outside the apu_thread "
                   "lock-acquisition sites we instrument.")
    elif underruns and not throt:
        out.append("Underruns WITHOUT producer slip or BQL stall. Either the "
                   "consumer is draining the FIFO faster than the producer can "
                   "fill it (check AAudio burst size vs producer rate), or the "
                   "FIFO is being initialized smaller than required for "
                   "callback cadence.")
    elif throt and not underruns:
        out.append("Producer is snapping forward but the FIFO never drains "
                   "fully. Audio plays back fine but you lose ~43ms of "
                   "content per snap. Not the user's stutter cause.")

    return "\n".join(out)


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
    if kind in ("all", "shader_dump"):
        # Per-pipeline-create SPIR-V/GLSL dump (Android-only, see draw.c).
        # External path so no run-as needed.
        count = _sh(f"ls {SHADER_DUMP_DIR} 2>/dev/null | wc -l").strip()
        size  = _sh(f"du -sh {SHADER_DUMP_DIR} 2>/dev/null").strip()
        parts.append(f"=== shader_dump ===\n{count} hashes\n{size}")
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
    if kind == "shader_dump":
        # External path under sdcard — wipe directly, no run-as needed.
        out = _sh(f"rm -rf {SHADER_DUMP_DIR}/* 2>&1; echo done")
        return f"Cleared shader_dump: {out}"
    if kind not in targets:
        return f"Unknown kind '{kind}'. Options: {', '.join(list(targets) + ['shader_dump'])}"
    cmds = " ; ".join(f"rm -rf {t}" for t in targets[kind])
    out = _sh(f"run-as {PKG} sh -c '{cmds} 2>&1; echo done'")
    return f"Cleared {kind}: {out}"


# ---------------------------------------------------------------------------
# Shader-dump audit (Android pipeline-create SPIR-V/GLSL archive + malioc)
# ---------------------------------------------------------------------------
#
# Pairs with the dumper at hw/xbox/nv2a/pgraph/vk/draw.c, which writes
# <SHADER_DUMP_DIR>/<hash>/killer_{vsh,psh,geom}.{spv,glsl} on every unique
# pipeline-create (disk-backed dedupe — keeps building across launches).

def _mali_gpu() -> str:
    """Detect the Mali GPU model from SurfaceFlinger; fall back to a sensible
    default. Cached after first call to avoid repeated dumpsys cost."""
    cached = getattr(_mali_gpu, "_cached", None)
    if cached: return cached
    out = _sh("dumpsys SurfaceFlinger | grep -m1 'GLES:'")
    m = re.search(r"Mali-[A-Za-z0-9]+", out)
    gpu = m.group(0) if m else "Mali-G715"
    _mali_gpu._cached = gpu  # type: ignore[attr-defined]
    return gpu

@mcp.tool()
def shader_dump_list(hash_prefix: str = "") -> str:
    """List shader hashes currently in <files>/x1box/shader_dump/.

    Each entry shows the hash + sizes of vsh/psh/geom (spv/glsl). Optionally
    filter by a hash prefix (e.g. "8c7" or "daf11ef5"). The dumper at
    draw.c is disk-backed-dedupe, so this archive grows with playtime."""
    if not _device(): return "No device connected."
    prefix = (hash_prefix or "").strip().lower()
    cmd = (
        f"ls {SHADER_DUMP_DIR} 2>/dev/null"
        + (f" | grep -E '^{re.escape(prefix)}'" if prefix else "")
    )
    hashes = [h for h in _sh(cmd).splitlines() if h.strip()]
    if not hashes:
        return f"No shaders in {SHADER_DUMP_DIR}" + (f" matching '{prefix}'" if prefix else "")
    rows = [f"{len(hashes)} hash{'es' if len(hashes) != 1 else ''} in {SHADER_DUMP_DIR}:", ""]
    rows.append(f"{'hash':<17}  {'vsh.spv':>8} {'psh.spv':>8} {'geom.spv':>9} {'vsh.glsl':>9} {'psh.glsl':>9} {'geom.glsl':>10}")
    for h in hashes:
        sizes = _sh(
            f"stat -c '%n %s' {SHADER_DUMP_DIR}/{h}/killer_vsh.spv "
            f"{SHADER_DUMP_DIR}/{h}/killer_psh.spv "
            f"{SHADER_DUMP_DIR}/{h}/killer_geom.spv "
            f"{SHADER_DUMP_DIR}/{h}/killer_vsh.glsl "
            f"{SHADER_DUMP_DIR}/{h}/killer_psh.glsl "
            f"{SHADER_DUMP_DIR}/{h}/killer_geom.glsl 2>/dev/null"
        )
        by_file: dict[str, str] = {}
        for line in sizes.splitlines():
            parts = line.split()
            if len(parts) == 2:
                by_file[parts[0].rsplit('/', 1)[-1]] = parts[1]
        def s(name: str) -> str: return by_file.get(name, "-")
        rows.append(
            f"{h:<17}  {s('killer_vsh.spv'):>8} {s('killer_psh.spv'):>8} "
            f"{s('killer_geom.spv'):>9} {s('killer_vsh.glsl'):>9} "
            f"{s('killer_psh.glsl'):>9} {s('killer_geom.glsl'):>10}"
        )
    return "\n".join(rows)

@mcp.tool()
def shader_dump_session_hashes(lines: int = 4000) -> str:
    """Return hashes seen at pipeline-create in recent logcat, in order.

    Useful for slicing the cumulative dump by play-area: clear logcat
    (`clear_log logcat`), play the area, then call this to get the exact
    hash list that became active. Pair with shader_dump_audit() on the
    resulting set."""
    if not _device(): return "No device connected."
    raw = _sh(f"logcat -d -t {int(lines)} -s {SHADER_DUMP_LOG_TAG}:W")
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in raw.splitlines():
        m = re.search(r"shader 0x([0-9a-f]+) seen at pipeline-create", line)
        if not m: continue
        h = m.group(1)
        if h in seen_set: continue
        seen.append(h); seen_set.add(h)
    if not seen:
        return "No shader-dump events in recent logcat. Play the target area, then retry."
    header = f"{len(seen)} unique shaders dumped this session:"
    return header + "\n" + "\n".join(seen)

@mcp.tool()
def shader_dump_audit(stage: str = "psh", gpu: str = "",
                      hash_prefix: str = "", details: bool = False) -> str:
    """Run Mali Offline Compiler over every dumped <stage>.spv and report.

    stage:        'psh' (default), 'vsh', or 'geom'
    gpu:          Mali GPU core (e.g. 'Mali-G715'). Auto-detected if empty.
    hash_prefix:  optional hash-prefix filter, same as shader_dump_list
    details:      if True, include the full malioc report per shader

    Pulls each <stage>.spv via adb to /tmp, runs malioc --vulkan against
    the device GPU, and returns a summary table flagging warnings/errors
    + Mali's late-ZS / coverage flags (the early-Z killers). Requires
    Arm Performance Studio installed at $ARM_PERFORMANCE_STUDIO_HOME."""
    if not _device(): return "No device connected."
    if stage not in ("vsh", "psh", "geom"):
        return f"Unknown stage '{stage}' (vsh|psh|geom)"
    if not os.path.exists(MALIOC_BIN):
        return (f"malioc not found at {MALIOC_BIN}. "
                f"Set $ARM_PERFORMANCE_STUDIO_HOME or install Arm Performance Studio.")
    gpu = gpu or _mali_gpu()
    prefix = (hash_prefix or "").strip().lower()
    cmd = (
        f"ls {SHADER_DUMP_DIR} 2>/dev/null"
        + (f" | grep -E '^{re.escape(prefix)}'" if prefix else "")
    )
    hashes = [h for h in _sh(cmd).splitlines() if h.strip()]
    if not hashes:
        return f"No shaders in {SHADER_DUMP_DIR}" + (f" matching '{prefix}'" if prefix else "")

    with tempfile.TemporaryDirectory(prefix="x1box-malioc-") as td:
        rows = [f"malioc audit on {gpu} — {len(hashes)} {stage} shader(s):", ""]
        rows.append(f"{'hash':<17}  {'warn':>4} {'err':>3} {'mod_cov':>7} {'late_zs':>7} {'side_fx':>7}  {'flags'}")
        suspects: list[str] = []
        for h in hashes:
            dev_path = f"{SHADER_DUMP_DIR}/{h}/killer_{stage}.spv"
            local = os.path.join(td, f"{h}.spv")
            r = _adb("pull", dev_path, local)
            if r.returncode != 0 or not os.path.exists(local):
                rows.append(f"{h:<17}  pull-failed")
                continue
            mr = subprocess.run(
                [MALIOC_BIN, "--vulkan", f"--{({'vsh':'vertex','psh':'fragment','geom':'geometry'})[stage]}",
                 "-c", gpu, "-d", "--format", "text", local],
                capture_output=True, text=True, check=False,
            )
            txt = mr.stdout + mr.stderr
            warn = len(re.findall(r"(?i)\bwarning\b", txt))
            err  = len(re.findall(r"(?i)\berror\b|\bfatal\b", txt))
            def flag(label: str) -> str:
                m = re.search(rf"{re.escape(label)}\s*:\s*(true|false)", txt)
                return m.group(1) if m else "?"
            mod_cov = flag("Modifies coverage")
            late_t  = flag("Uses late ZS test")
            late_u  = flag("Uses late ZS update")
            side    = flag("Has side-effects")
            flags_short = []
            if mod_cov == "true": flags_short.append("discard")
            if late_t == "true" or late_u == "true": flags_short.append("late-ZS")
            if side == "true": flags_short.append("side-fx")
            rows.append(
                f"{h:<17}  {warn:>4} {err:>3} {mod_cov:>7} "
                f"{(late_t if late_t == late_u else f'{late_t}/{late_u}'):>7} {side:>7}  "
                f"{','.join(flags_short) if flags_short else '-'}"
            )
            if warn or err or details:
                suspects.append(f"\n--- {h} ({stage}) ---\n{txt}")
        out = "\n".join(rows)
        if suspects and (details or any("warn" in r.lower() or " err " in r for r in suspects)):
            out += "\n\nFull reports for shaders with warnings/errors:\n" + "\n".join(suspects)
        return out


# ---------------------------------------------------------------------------
# Generic GPU probe runner — compile GLSL → SPIR-V → push → restart → pull
# ---------------------------------------------------------------------------
#
# Pairs with hw/xbox/nv2a/pgraph/vk/probe_runner.{c,h}. xemu watches for
# `<ext>/x1box/gpu_probe/probe_req.bin` at pgraph_vk_init time; if present
# it runs the user-supplied compute or fragment shader, writes a binary
# result to probe_out.bin, drops probe_done.flag as a sentinel, and
# renames the request to .consumed. We never touch logcat here — that's
# the whole reason this tool exists.

def _glslang_bin() -> str | None:
    """Locate a SPIR-V compiler on the host."""
    for c in ("glslangValidator", "glslc"):
        p = subprocess.run(["which", c], capture_output=True, text=True, check=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    return None

def _compile_glsl_to_spirv(glsl: str, stage: str) -> tuple[bytes | None, str]:
    """Compile a GLSL source string to SPIR-V bytes. Returns (spv, err)."""
    bin_ = _glslang_bin()
    if not bin_:
        return None, "No SPIR-V compiler in PATH (glslangValidator / glslc)."
    s_map = {"compute": "comp", "fragment": "frag"}
    s = s_map.get(stage)
    if not s:
        return None, f"stage must be 'compute' or 'fragment', got '{stage}'."
    with tempfile.TemporaryDirectory(prefix="x1box-glsl-") as td:
        src = os.path.join(td, f"probe.{s}")
        out = os.path.join(td, "probe.spv")
        with open(src, "w") as f:
            f.write(glsl)
        if "glslang" in bin_:
            cmd = [bin_, "-V", "-S", s, src, "-o", out]
        else:  # glslc
            cmd = [bin_, f"-fshader-stage={s}", src, "-o", out]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0 or not os.path.exists(out):
            return None, (p.stdout + p.stderr).strip() or "glsl compile failed"
        with open(out, "rb") as f:
            return f.read(), ""

def _build_probe_request(stage_id: int, n_outputs: int,
                         push_floats: list[float], spv: bytes) -> bytes:
    """Serialize struct x1b_probe_req + SPIR-V payload."""
    if len(spv) % 4 != 0:
        spv = spv + b"\x00" * (4 - (len(spv) % 4))
    n_push = len(push_floats)
    if n_push > 16:
        raise ValueError("push_floats must have at most 16 entries")
    pad = push_floats + [0.0] * (16 - n_push)
    # <4sIIIIIII = magic, version, stage, n_outputs, n_push_floats, spv_size, reserved[2]
    header = struct.pack("<4sIIIIIII", b"XPRB", GPU_PROBE_VERSION, stage_id,
                          n_outputs, n_push, len(spv), 0, 0)
    pcs = struct.pack("<16f", *pad)
    return header + pcs + spv

# struct x1b_probe_out (probe_runner.h):
#   char     magic[4];              // "XPRO"            4
#   uint32_t version;                                    4
#   uint32_t status;                                     4
#   uint32_t n_outputs;                                  4
#   char     gpu_name[256];                              256
#   uint32_t vendor_id, device_id, driver_version;       12
#   uint32_t fp32_szinfnan, denorm_preserve, denorm_ftz,
#            rounding_rte, rounding_rtz;                 20
#   uint32_t error_msg_size;                             4
#   uint32_t reserved[3];                                12
# = 320 bytes total.
_OUT_HDR_FMT = "<4sIII256sIIIIIIIII3I"
assert struct.calcsize(_OUT_HDR_FMT) == 320, struct.calcsize(_OUT_HDR_FMT)

def _classify_bits(v: int) -> str:
    """Match probe_runner.c's classify_bits()."""
    sign = (v >> 31) & 1
    exp  = (v >> 23) & 0xFF
    mant = v & 0x7FFFFF
    if exp == 0xFF and mant != 0:
        return "NaN"
    if exp == 0xFF and mant == 0:
        return "-Inf" if sign else "+Inf"
    if exp == 0   and mant == 0:
        return "-0" if sign else "+0"
    if exp == 0   and mant != 0:
        return "-denorm" if sign else "+denorm"
    return "finite"

def _decode_probe_output(blob: bytes) -> dict:
    """Unpack probe_out.bin into a dict. Raises ValueError on bad data."""
    if len(blob) < struct.calcsize(_OUT_HDR_FMT):
        raise ValueError(f"output too short ({len(blob)} bytes)")
    fields = struct.unpack_from(_OUT_HDR_FMT, blob, 0)
    (magic, version, status, n_outputs, gpu_raw,
     vendor_id, device_id, driver_version,
     fp32_szinf, fp32_denp, fp32_dftz, fp32_rte, fp32_rtz,
     err_size, _r0, _r1, _r2) = fields
    if magic != b"XPRO":
        raise ValueError(f"bad output magic: {magic!r}")
    gpu_name = gpu_raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")
    off = struct.calcsize(_OUT_HDR_FMT)
    outputs = list(struct.unpack_from(f"<{n_outputs}I", blob, off))
    off += n_outputs * 4
    err_msg = ""
    if err_size and off + err_size <= len(blob):
        err_msg = blob[off:off + err_size].decode("ascii", errors="replace")
    return {
        "version": version, "status": status, "n_outputs": n_outputs,
        "gpu_name": gpu_name, "vendor_id": vendor_id, "device_id": device_id,
        "driver_version": driver_version,
        "float_controls": {
            "fp32_signed_zero_inf_nan_preserve": bool(fp32_szinf),
            "fp32_denorm_preserve": bool(fp32_denp),
            "fp32_denorm_flush_to_zero": bool(fp32_dftz),
            "fp32_rounding_mode_rte": bool(fp32_rte),
            "fp32_rounding_mode_rtz": bool(fp32_rtz),
        },
        "outputs": outputs,
        "error": err_msg,
    }

def _format_probe_results(decoded: dict, labels: list[str] | None) -> str:
    rows = [
        f"GPU: {decoded['gpu_name']}"
        f"  (vendor=0x{decoded['vendor_id']:04x},"
        f" device=0x{decoded['device_id']:08x},"
        f" driver={decoded['driver_version']})",
        "FloatControls.fp32: "
        + ", ".join(f"{k.split('fp32_')[1]}={v}"
                    for k, v in decoded["float_controls"].items()),
        f"status: {decoded['status']}"
        + (f" ({decoded['error']})" if decoded['error'] else ""),
        "",
        f"{'slot':<5} {'label':<30} {'hex':>10}  {'class':<8}  decoded",
    ]
    for i, v in enumerate(decoded["outputs"]):
        f = struct.unpack("<f", struct.pack("<I", v))[0]
        label = labels[i] if labels and i < len(labels) else ""
        rows.append(f"{i:<5} {label:<30} 0x{v:08x}  "
                    f"{_classify_bits(v):<8}  {f!r}")
    return "\n".join(rows)

@mcp.tool()
def gpu_probe(glsl: str, stage: str = "compute", n_outputs: int = 24,
              push_floats: str = "", labels: str = "",
              auto_restart: bool = True, game_name: str = "",
              timeout: float = 30.0) -> str:
    """Run a one-shot Vulkan probe on the device GPU and return raw outputs.

    Compiles GLSL → SPIR-V on the host via glslangValidator, pushes the
    request file to the device, optionally stops + relaunches the emulator
    to trigger pgraph_vk_init (where probe_runner picks up the request),
    waits for `probe_done.flag`, pulls the result, and decodes it.

    The probe runs with ZERO logcat traffic — results come back as a
    binary file, not log lines, so dispatch overhead is one disk-write
    on each side.

    Compute shader expectations:
      #version 450
      layout(local_size_x = 1) in;
      layout(std430, set=0, binding=0) buffer Out { uint probe[N]; };
      // optional: layout(push_constant) uniform Push { float p[<= 16]; };
      void main() { probe[i] = floatBitsToUint(<expr>); ... }

    Fragment shader expectations:
      #version 450
      layout(location = 0) out uint result;     // R32_UINT attachment
      // optional: layout(push_constant) uniform Push { float p[<= 16]; };
      void main() { int slot = int(gl_FragCoord.x); result = ...; }
      // (the runner supplies a fullscreen-triangle vertex shader)

    Args:
      glsl:         the GLSL source. Must match the layout above.
      stage:        'compute' (default) or 'fragment'.
      n_outputs:    number of uint32 result slots (1..1024).
      push_floats:  comma-separated floats for the push-constant block.
      labels:       comma-separated per-slot labels for the result table.
      auto_restart: stop the app + relaunch a title to trigger the probe.
      game_name:    title to launch (case-insensitive substring match).
                    If empty, uses the first available from list_games.
      timeout:      seconds to wait for probe_done.flag.

    Returns: formatted table of GPU info + per-slot hex/decoded values."""
    if not _device(): return "No device connected."
    if stage not in ("compute", "fragment"):
        return f"stage must be 'compute' or 'fragment', got '{stage}'"
    if n_outputs < 1 or n_outputs > 1024:
        return "n_outputs must be in [1, 1024]"

    floats: list[float] = []
    if push_floats.strip():
        try:
            floats = [float(x) for x in push_floats.split(",")]
        except ValueError as e:
            return f"push_floats parse error: {e}"
    if len(floats) > 16:
        return "at most 16 push_floats supported"

    label_list = [x.strip() for x in labels.split(",")] if labels.strip() else None

    # Compile GLSL → SPIR-V locally.
    spv, err = _compile_glsl_to_spirv(glsl, stage)
    if not spv:
        return f"GLSL compile failed:\n{err}"

    stage_id = GPU_PROBE_STAGE_COMPUTE if stage == "compute" else GPU_PROBE_STAGE_FRAGMENT
    req_blob = _build_probe_request(stage_id, n_outputs, floats, spv)

    # Clean any stale done/out from a prior run, then push the request.
    _sh(f"mkdir -p {GPU_PROBE_DIR}; rm -f {GPU_PROBE_DONE} {GPU_PROBE_OUT} {GPU_PROBE_REQ_USED}")
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        tmp.write(req_blob); tmp.flush()
        p = _adb("push", tmp.name, GPU_PROBE_REQ)
        if p.returncode != 0:
            return f"adb push failed: {p.stderr.strip() or p.stdout.strip()}"

    if auto_restart:
        # Force-stop, then launch a game so pgraph_vk_init runs.
        _sh(f"am force-stop {PKG}")
        target = game_name
        if not target:
            # First listed game wins. list_games returns "title\tpath" lines.
            try:
                head = list_games().splitlines()  # type: ignore[name-defined]
                for line in head[1:]:
                    if "\t" in line:
                        target = line.split("\t", 1)[0]; break
            except Exception:
                target = ""
        if target:
            launch_game(target)  # type: ignore[name-defined]
        else:
            return ("Request pushed to " + GPU_PROBE_REQ +
                    " but no game available to launch; restart the app "
                    "manually and re-run gpu_probe with auto_restart=False "
                    "to just wait for the result.")

    # Poll for the done sentinel.
    deadline = time.monotonic() + max(1.0, float(timeout))
    poll_interval = 0.4
    while time.monotonic() < deadline:
        if _sh(f"[ -f {GPU_PROBE_DONE} ] && echo yes").strip() == "yes":
            break
        time.sleep(poll_interval)
    else:
        return (f"Timed out waiting for {GPU_PROBE_DONE} after {timeout:.1f}s. "
                f"Is xemu actually starting? (Check `logcat -d -s hakuX-gpu-probe:*`.)")

    # Pull the binary output.
    with tempfile.TemporaryDirectory(prefix="x1box-probe-") as td:
        local = os.path.join(td, "probe_out.bin")
        p = _adb("pull", GPU_PROBE_OUT, local)
        if p.returncode != 0 or not os.path.exists(local):
            return f"adb pull {GPU_PROBE_OUT} failed: {p.stderr.strip()}"
        with open(local, "rb") as f:
            blob = f.read()
    try:
        decoded = _decode_probe_output(blob)
    except ValueError as e:
        return f"output decode failed: {e}"

    return _format_probe_results(decoded, label_list)


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
# Extended profiling: multi-device, hw counters, call-graph, core affinity
# ---------------------------------------------------------------------------
#
# Built for cross-device A/B comparisons (e.g. Pixel 10a Mali vs Zenfone 10
# Adreno). The existing `profile` tool above is single-device, cpu-clock-only,
# and capped at head -60. These three tools generalise that:
#
#   profile_native           — simpleperf record+report with selectable
#                              events (cpu-clock:u, cpu-cycles, instructions,
#                              cache-misses, dTLB-load-misses, ...), flat or
#                              call-graph output, and a `device` parameter so
#                              both phones can be polled from one session.
#   core_affinity_snapshot   — per-thread last_cpu sampling over a window so
#                              you can see whether hot threads land on big
#                              (Cortex-X) or little (A520) cores. No code on
#                              device; purely reads /proc.
#   _adb_dev / _sh_dev       — `adb -s <serial>` shims used by the above so
#                              we don't trample the existing single-device
#                              flow.

def _adb_dev(serial: str | None, *args: str) -> subprocess.CompletedProcess:
    """adb with optional -s <serial>. None => default device."""
    if serial:
        return subprocess.run(
            ["adb", "-s", serial, *args],
            capture_output=True, text=True, check=False,
        )
    return _adb(*args)

def _sh_dev(serial: str | None, *args: str) -> str:
    r = _adb_dev(serial, "shell", *args)
    return (r.stdout + r.stderr).strip()

def _emu_pid_dev(serial: str | None, package: str | None = None) -> str | None:
    """Like _emu_pid() but targets a specific device and (optionally) a
    specific package. `package` defaults to PKG (the canonical debug
    install); pass e.g. "com.izzy2lost.x1box.perftest" to target the
    side-by-side perftest variant.

    /proc/<pid>/comm is truncated to 15 chars by the kernel so `pidof
    <pkg>:xemu` is empty even when the process runs. Match against
    /proc/<pid>/cmdline (full name, not truncated) AND verify argv[0]
    equals the expected emu-process name exactly — a substring grep
    matches transient helper shells that have the string in their args
    (e.g. our own `sh -c grep ... :xemu` invocation).

    Returns None if the :xemu process for `package` is not running on
    the target device. Does NOT fall back to the UI pid: tools that want
    emu time will silently profile the wrong process if we conflate the
    two. Use a UI-specific path when you want the launcher.
    """
    pkg = package or PKG
    emu_proc = f"{pkg}:xemu"
    rc = _adb_dev(
        serial, "shell",
        f"grep -l -F '{emu_proc}' /proc/[0-9]*/cmdline 2>/dev/null"
    )
    if rc.returncode != 0:
        return None
    candidates = []
    for line in (rc.stdout or "").splitlines():
        parts = line.strip().split("/")
        if len(parts) >= 3 and parts[2].isdigit():
            candidates.append(parts[2])
    for pid in candidates:
        rc2 = _adb_dev(
            serial, "shell",
            f"tr '\\0' '\\n' < /proc/{pid}/cmdline 2>/dev/null | head -1"
        )
        if rc2.returncode == 0 and rc2.stdout.strip() == emu_proc:
            return pid
    return None

def _list_devices() -> list[str]:
    lines = _adb("devices").stdout.strip().splitlines()[1:]
    return [l.split()[0] for l in lines if l.strip() and "device" in l]


@mcp.tool()
def profile_native(duration: int = 5,
                   events: str = "cpu-clock:u",
                   mode: str = "flat",
                   sort: str = "comm,dso,symbol",
                   top_n: int = 60,
                   tid: str = "",
                   device: str = "",
                   target: str = "emu",
                   package: str = "") -> str:
    """
    Sample-based profiler with selectable hardware events and multi-device
    support. Wraps `simpleperf record --app <pkg>` (which uses run-as
    internally so it works on debuggable Android builds without root) and
    `simpleperf report`. Returns a ranked symbol table or a call-graph view.

    Built for cross-device A/B (Pixel vs Zenfone) and for going beyond
    cpu-clock:u when you need to know *why* a hot symbol is hot — IPC,
    cache misses, dTLB misses, etc.

    Args:
      duration:  recording window in seconds (default 5).
      events:    comma-separated simpleperf events. Defaults to wall-clock
                 sampling. Useful alternatives:
                 - 'cpu-clock:u'                     where time was spent
                 - 'cpu-cycles,instructions'         per-symbol IPC
                 - 'cache-misses,cache-references'   memory pressure
                 - 'dTLB-load-misses'                page-walk pressure
                 - 'branch-misses'                   predictor whiff
                 Run `adb shell simpleperf list hw` to see what the device
                 supports. `:u` suffix restricts to userspace (required when
                 not root).
      mode:      'flat' (default) for a ranked symbol table; 'callgraph' for
                 the inclusive-time tree (sort defaults to comm,symbol then
                 walks callers).
      sort:      simpleperf --sort columns. Default 'comm,dso,symbol' attributes
                 each symbol to its thread + dso so cross-thread hotspots are
                 obvious. 'comm,symbol' is more compact.
      top_n:     max rows in the flat report (default 60). Ignored for callgraph.
      tid:       optional thread id (numeric string) to restrict sampling to a
                 single thread. Use get_threads() to find the TID; or just
                 leave empty for process-wide.
      device:    adb serial. Empty = use the default device (first one
                 connected). Use `list_devices` to discover serials when both
                 Pixel and Zenfone are plugged in.
      target:    'emu' (default :xemu emulation process), 'ui' (launcher).
      package:   Android package id to profile. Empty = the canonical debug
                 install (`com.izzy2lost.x1box`). Pass
                 `com.izzy2lost.x1box.perftest` to profile the side-by-side
                 debug-off perftest variant (lets us A/B debug-log overhead
                 vs no-log steady state across two separately-installed APKs
                 without changing tool plumbing).

    The recorded perf.data is left at /data/local/tmp/perf_native.data so you
    can pull and post-process with simpleperf if needed (e.g. flame graphs).

    Examples:
      # Wall-clock profile, both phones — same arguments, just change device.
      profile_native(device='61291JEA311201')
      profile_native(device='RBAIB70003585LA')
      # Profile the side-by-side perftest install (debug-off variant).
      profile_native(device='61291JEA311201',
                     package='com.izzy2lost.x1box.perftest')
      # IPC profile: tells you where the CPU is memory-stalled vs compute-bound.
      profile_native(events='cpu-cycles,instructions', top_n=40)
      # Why is tlb_reset_dirty slow? Cache miss profile.
      profile_native(events='cache-misses,cache-references', top_n=40)
      # Call-graph attribution: who's calling the hot leaf?
      profile_native(mode='callgraph', sort='comm,symbol')
    """
    if device and device not in _list_devices():
        return f"device '{device}' not in adb devices. Available: {_list_devices()}"
    if not device and not _list_devices():
        return "No device connected."

    pkg = package or PKG
    pid = _emu_pid_dev(device, package=pkg) if target != "ui" else (
        _sh_dev(device, f"pidof {pkg}").strip() or None)
    if not pid:
        return f"{pkg}:{'ui' if target == 'ui' else 'xemu'} not running on device '{device or '(default)'}'"

    duration = max(1, min(int(duration), 60))
    top_n = max(5, min(int(top_n), 500))
    perf_data = "/data/local/tmp/perf_native.data"

    record_args = [
        "simpleperf", "record",
        "--app", pkg,
        "-e", events,
        "-g",
        "--duration", str(duration),
        "-o", perf_data,
    ]
    if tid:
        record_args.extend(["-t", tid])

    try:
        rec = subprocess.run(
            ["adb", *(["-s", device] if device else []), "shell", *record_args],
            capture_output=True, text=True, check=False,
            timeout=duration + 15,
        )
    except subprocess.TimeoutExpired:
        return f"simpleperf record timed out after {duration + 15}s"

    rec_err = rec.stderr.strip()
    if rec.returncode != 0:
        return f"simpleperf record failed (rc={rec.returncode}):\n{rec_err}"

    if mode == "callgraph":
        report_args = [
            "simpleperf", "report", "-i", perf_data,
            "--sort", sort,
            "--children", "-g", "caller",
            "--full-callgraph", "--max-stack", "20",
        ]
    else:
        report_args = [
            "simpleperf", "report", "-i", perf_data,
            "--sort", sort,
            "-n", "--print-event-count",
        ]

    try:
        rep = subprocess.run(
            ["adb", *(["-s", device] if device else []), "shell", *report_args],
            capture_output=True, text=True, check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "simpleperf report timed out"

    # report has unbounded output for callgraph mode — cap it.
    out = rep.stdout
    if mode == "callgraph":
        # callgraph mode prints one tree per symbol; cap by lines.
        lines = out.splitlines()
        if len(lines) > 600:
            out = "\n".join(lines[:600]) + f"\n... ({len(lines) - 600} more lines truncated)"
    else:
        # flat mode: header (5 lines) + top_n symbol rows + tail entries.
        lines = out.splitlines()
        if len(lines) > top_n + 10:
            out = "\n".join(lines[:top_n + 10])

    header = (
        f"device={device or '(default)'}  pid={pid}  events={events}  "
        f"duration={duration}s  mode={mode}\n"
        f"perf.data => {perf_data}\n"
    )
    if rec_err:
        header += f"record stderr:\n{rec_err}\n"
    return header + "\n" + out


_AFFINITY_SCRIPT = r"""
PID="$1"
DUR_MS="$2"
INT_MS="$3"

if [ -z "$PID" ] || [ ! -d /proc/$PID/task ]; then
    echo "ERR no_pid"; exit 1
fi

# Convert ms to fractional seconds for `sleep` once. Doing it per-iteration
# forks awk every loop and on Android fork+exec is ~50-100ms, which
# dominated our sampling window before this rewrite.
SLEEP_SEC=$(awk "BEGIN{printf \"%.3f\", $INT_MS / 1000}")
DUR_SEC=$(awk "BEGIN{printf \"%d\", ($DUR_MS + 999) / 1000}")

# Snapshot all thread names + comm at the start.
# Single fork: one awk that walks every comm file. Per-thread forks would
# cost ~5s on a 50-thread process.
awk 'BEGIN{
    cmd = "ls /proc/'"$PID"'/task"
    while ((cmd | getline tid) > 0) {
        getline c < ("/proc/'"$PID"'/task/" tid "/comm")
        close("/proc/'"$PID"'/task/" tid "/comm")
        printf "N|%s|%s\n", tid, c
    }
    close(cmd)
}'

START_SEC=$(cut -d. -f1 /proc/uptime)
END_SEC=$((START_SEC + DUR_SEC))

while :; do
    NOW_SEC=$(cut -d. -f1 /proc/uptime)
    [ "$NOW_SEC" -ge "$END_SEC" ] && break
    # ONE awk pass over all /proc/<pid>/task/*/stat files. FILENAME gives
    # us the tid (4th path component). Walk comm-paren from end so names
    # with spaces ("(CPU 0/TCG)") don't shift field indices.
    awk '
    {
        for (i = NF; i >= 1; i--) if ($i ~ /\)/) { rest = i + 1; break }
        n = split(FILENAME, parts, "/")
        printf "S|%s|%s\n", parts[n-1], $(rest+36)
    }
    ' /proc/$PID/task/*/stat 2>/dev/null
    sleep $SLEEP_SEC
done
echo "DONE"
"""


@mcp.tool()
def core_affinity_snapshot(duration: int = 5,
                           interval_ms: int = 100,
                           device: str = "",
                           target: str = "emu",
                           thread_filter: str = "",
                           package: str = "") -> str:
    """
    Sample which CPU core each thread of the emulator is running on, over a
    window, and report per-thread histograms.

    Why this matters on big.LITTLE: Tensor G4 has 1 Cortex-X4 (CPU 7),
    3 Cortex-A720 (CPUs 4-6), 4 Cortex-A520 (CPUs 0-3). Snapdragon 8 Gen 2
    has 1 Cortex-X3 (CPU 7), 2 A715 + 2 A710 (CPUs 3-6), 3 A510 (CPUs 0-2).
    If `CPU 0/TCG` or `nv2a.pfifo_thre` lands on the little cluster, the
    same code runs 3-5x slower than on the big core — that alone explains
    cross-device perf gaps that look like cache or memory issues.

    Method: every `interval_ms`, read /proc/<pid>/task/<tid>/stat and pull
    field 39 (the last CPU the thread ran on). Build a histogram per thread.
    Read-only, zero perturbation to the running emulator. Costs roughly
    `duration / interval_ms` adb-shell stat reads, all done in a single
    on-device script so we don't round-trip per sample.

    Args:
      duration:      window length in seconds (default 5).
      interval_ms:   sample period in milliseconds (default 100 = 10 Hz).
                     50ms gives finer resolution at higher cost.
      device:        adb serial. Empty = default device. Use list_devices
                     to discover when multiple are connected.
      target:        'emu' (default) or 'ui'.
      thread_filter: only report threads whose comm matches this substring
                     (e.g. 'TCG', 'pfifo', 'vk.rende', 'mcpx'). Empty = all
                     threads with samples.
      package:       Android package id to target. Empty = the canonical
                     debug install (`com.izzy2lost.x1box`). Pass
                     `com.izzy2lost.x1box.perftest` to sample the perftest
                     variant when both are installed side-by-side.

    Returns a table where each row is a thread (name + tid) and the columns
    are CPU 0..N-1 with the % of samples observed on each.
    """
    if device and device not in _list_devices():
        return f"device '{device}' not in adb devices. Available: {_list_devices()}"
    if not device and not _list_devices():
        return "No device connected."

    pkg = package or PKG
    pid = _emu_pid_dev(device, package=pkg) if target != "ui" else (
        _sh_dev(device, f"pidof {pkg}").strip() or None)
    if not pid:
        return f"{pkg}:{'ui' if target == 'ui' else 'xemu'} not running"

    duration = max(1, int(duration))
    interval_ms = max(20, int(interval_ms))
    dur_ms = duration * 1000

    script = f"set -- {pid} {dur_ms} {interval_ms}\n{_AFFINITY_SCRIPT}"
    try:
        r = subprocess.run(
            ["adb", *(["-s", device] if device else []),
             "exec-out", "sh", "-c", script],
            capture_output=True, text=True, check=False,
            timeout=duration + 15,
        )
    except subprocess.TimeoutExpired:
        return f"core_affinity_snapshot timed out after {duration + 15}s"

    raw = r.stdout
    if "ERR no_pid" in raw:
        return f"profile failed: pid {pid} disappeared"
    if not raw.strip():
        return f"affinity sampler returned no output (stderr: {r.stderr.strip()})"

    # Parse: N|tid|comm establish names; S|tid|cpu accumulate.
    comm: dict[str, str] = {}
    samples: dict[str, dict[int, int]] = {}
    cpus_seen: set[int] = set()
    for line in raw.splitlines():
        if line.startswith("N|"):
            _, tid, c = line.split("|", 2)
            comm[tid] = c
        elif line.startswith("S|"):
            try:
                _, tid, cpu = line.split("|", 2)
                ci = int(cpu)
            except ValueError:
                continue
            cpus_seen.add(ci)
            samples.setdefault(tid, {})
            samples[tid][ci] = samples[tid].get(ci, 0) + 1

    if not samples:
        return f"no samples collected. raw head:\n{raw[:400]}"

    cpus = sorted(cpus_seen)
    rows = []
    for tid, hist in samples.items():
        name = comm.get(tid, "?")
        if thread_filter and thread_filter not in name:
            continue
        total = sum(hist.values())
        if total == 0:
            continue
        row = {"tid": tid, "comm": name, "total": total, "hist": hist}
        rows.append(row)

    # Sort by total samples descending — busiest threads first.
    rows.sort(key=lambda r: -r["total"])

    # Build output table.
    lines = []
    lines.append(
        f"device={device or '(default)'}  pid={pid}  "
        f"window={duration}s  interval={interval_ms}ms  "
        f"threads_with_samples={len(rows)}"
    )
    lines.append("")
    header = f"{'TID':>6}  {'COMM':<20}  {'N':>4}  "
    header += "  ".join(f"cpu{c:>2}" for c in cpus)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = [f"{r['tid']:>6}", f"{r['comm'][:20]:<20}", f"{r['total']:>4}"]
        for c in cpus:
            n = r["hist"].get(c, 0)
            pct = n * 100 / r["total"]
            cells.append(f"{pct:>5.0f}" if n else "    .")
        lines.append("  ".join(cells))

    return "\n".join(lines)


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
