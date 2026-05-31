# xpacks — x1-box patch packs

Cemu-style mod packs for x1-box. Four patch kinds: static `bytes` (xbe
code/data), `pattern_bytes` (tag/map data via signature scan), `cave`
(inject new x86 + JMP rel32 trampoline), and `shader` (Vulkan SPIR-V
replacement). Full spec: [SPEC.md](SPEC.md).

Drop a directory under
`<sdcard>/Android/data/com.izzy2lost.x1box/files/xpacks/<TITLE_ID>/<pack>/`
on the device, enable it in **Settings → Manage Mods (xpacks)**, and relaunch
the game. Bundled packs ship in the APK at `assets/xpacks/` and auto-install
on first launch (user edits in the writable root are preserved).

## Layout

```
xpacks/
  enabled.txt                            # newline list of enabled "<title_id>/<pack_name>"
  <TITLE_ID_HEX>/                        # e.g. 4D530064 (Halo 2 NTSC)
    <pack_name>/
      pack.toml                          # manifest
      shaders/<glsl_hash>.spv            # optional SPIR-V overrides
      <anything-else-referenced>
```

## Manifest, in 30 seconds

```toml
schema = 1
name = "Pack name"

[match]
title_id      = 0x4D530064               # required — xbe cert title id
xbe_code_sha1 = "abc...40hex"            # optional — restrict to specific xbe builds

# static patch (xbe code/data lives at fixed virtual address)
[[patch]]
kind        = "bytes"
description = "human-readable"
address     = 0x0045DC6C
expected    = "89 88 08 3D"
replace     = "89 88 88 3C"

# signature-anchored patch (tag/map data lives at non-deterministic address)
[[patch]]
kind           = "pattern_bytes"
description    = "..."
search_pattern = "DE AD ?? ?? BE EF"     # '??' wildcards allowed
search_start   = 0x00010000              # optional defaults
search_end     = 0x10000000
search_offset  = 0x10                    # bytes from match site to patch site
expected       = "00 00 80 3F"           # optional pre-image guard
replace        = "00 00 00 40"

# code cave + JMP rel32 trampoline (inject new x86 at a free guest VA)
[[patch]]
kind            = "cave"
description     = "..."
trampoline_at   = 0x00123110             # required — JMP rel32 install site
expected        = "55 8B EC 83 EC"       # optional — pre-image at trampoline
trampoline_size = 5                      # optional — bytes displaced (5..15)
cave_at         = 0x002B0000             # required — free region for cave body
cave_expected   = "00 00 00 ..."         # optional — verify cave site is free
cave_bytes      = "C2 08 00"             # required — x86 body (here: ret 8)
return_in_cave  = true                   # optional — cave does its own return

# pgraph_vk SPIR-V override keyed by GLSL hash
[[patch]]
kind      = "shader"
glsl_hash = "0xfeedfacecafebabe"
file      = "shaders/feedfacecafebabe.spv"
```

`expected` is the load-bearing safety on byte patches — refused on mismatch.
`xbe_code_sha1` is the build fingerprint (SHA1 over the 256-byte xbe digsig);
on mismatch the entire pack is skipped, which is the safe stop for pattern
patches that can't pre-image-check at a fixed address.

## Identifying a game

| Game | Title ID |
|------|----------|
| Halo 2 (NTSC) | `0x4D530064` |
| Halo CE (NTSC) | `0x4D530002` |

Title ID is at offset 8 of the xbe certificate (xbe virtual `0x1018C` for
Halo 2's image base of `0x10000`).

## Built-in sample

`xpacks/4D530064/halo2-60hz-tick-demo/` flips the simulation `seconds_per_tick`
constant from `1/30` to `1/60`. The game runs at 2x speed because everything
scales off that constant — it's a demo of the patch mechanism, not a usable
mod. Proper 60 Hz would require interpolation between sim ticks (an engine
rewrite, not a binary patch).

## Hook timing

- **Bytes patches** apply when `xemu_get_xbe_info()` first sees a valid xbe in
  guest memory. Idempotent per title id per boot.
- **Pattern patches** first attempt at xbe-detect, then retry every guest
  frame via `nv2a_profile_flip_stall()` until each signature is resident.
  Throttled internally to one scan per second.
- **Cave patches** apply once at xbe-detect, same timing as bytes. The cave
  body + (optional) displaced bytes + return-jump are written to `cave_at`,
  then a 5-byte JMP rel32 (NOP-padded to `trampoline_size`) is installed at
  `trampoline_at`. Both `expected` and `cave_expected` are load-bearing
  safety: a mismatch refuses the pack.
- **Shader overrides** apply during pgraph_vk's GLSL → SPIR-V compile. The
  override bypasses both the on-disk SPV cache and the glslang compiler.

Changes take effect on game relaunch.
