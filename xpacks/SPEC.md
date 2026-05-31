# xpack specification — v1

Canonical schema for x1-box runtime patch packs. Pack authors target this
spec; the loader (`hw/xbox/xpacks.c`) and the Android manager
(`XPackManager.kt`) implement it. Three patch kinds, all live.

## On-disk layout

```
<sdcard>/Android/data/com.izzy2lost.x1box/files/xpacks/
  enabled.txt
  <TITLE_ID_HEX>/<pack_name>/
    pack.toml                 # required — manifest
    shaders/<glsl_hash>.spv   # optional — SPIR-V replacements
    ...                       # any other assets the manifest references
```

- `enabled.txt` is a newline-delimited list of `<title_id>/<pack_name>` ids
  that the loader should apply. Written by the UI, consumed by the C loader.
- `<TITLE_ID_HEX>` is the xbe certificate title id as 8 uppercase hex chars
  (e.g. `4D530064` for Halo 2 NTSC). Mismatched packs are silently skipped.

Bundled packs ship from `assets/xpacks/<title_id>/<pack_name>/`. The Android
manager auto-copies them into the writable root on launch. User-edited copies
in the writable root are never overwritten.

## Manifest format

INI-style (a subset of TOML). Parsed by both C (`xpacks.c`) and Kotlin
(`XPackManager.kt`) — both implementations must agree on the surface below.

```toml
schema      = 1            # required — parser version
name        = "string"     # required — UI label
author      = "string"     # optional
description = "string"     # optional — UI subtext
version     = "string"     # optional — informational only

[match]
title_id      = 0x4D530064            # required — must match xbe cert
xbe_code_sha1 = "abc...40hex"         # optional — restrict to specific builds
xbe_code_sha1 = "abc...40hex, def..." # comma-separated for multi-build packs

[[patch]]
kind = "bytes" | "pattern_bytes" | "cave" | "shader"
# kind-specific fields follow
```

`schema = 1` is the version handshake. Future spec changes will bump this;
the loader refuses unknown kinds with a warning rather than failing the pack.

### `match.xbe_code_sha1` — build fingerprint

SHA1 over the **256-byte xbe digital-signature field** (`xbe_header.m_digsig`,
offset 4 in the xbe). Each retail build is signed independently, so this is
a uniquely deterministic per-build fingerprint that doesn't require knowing
section-table internals.

To capture it once:

```sh
# bytes 4..260 of the .xbe (first 4 bytes are the XBEH magic)
dd if=halo2-default.xbe bs=1 skip=4 count=256 2>/dev/null | sha1sum
```

If `xbe_code_sha1` is absent, the pack applies to any build matching the
title id. If present, the loader logs the running digsig SHA1 on mismatch
so you can copy it into your manifest.

## Patch kinds

### `kind = "bytes"`

Writes raw bytes into guest physical memory once, at the moment the xbe is
first detected post-load. Used for static patches in the xbe's code/data
sections, which always land at the xbe's declared image base on Xbox (no
ASLR).

```toml
[[patch]]
kind        = "bytes"
description = "human-readable"
address     = 0x0045DC6C       # required — guest virtual address
expected    = "89 88 08 3D"    # optional but recommended — pre-image
replace     = "89 88 88 3C"    # required — replacement bytes
```

- `expected` is the safety check. The loader refuses the patch if the
  current bytes don't match. Always include this.
- `replace` may be any length; sizes need not match `expected`.
  Overlong replacements clobber adjacent bytes.
- Image base for Halo 2 is `0x00010000`; the address you read in Ghidra
  *is* the guest virtual address.

### `kind = "pattern_bytes"`

Scans guest memory for a byte pattern with wildcards, then writes at a
fixed offset from the match site. Used for tag-data and map overlays —
content that lives at non-deterministic addresses (each level load lands
it differently) but contains a stable internal signature.

```toml
[[patch]]
kind           = "pattern_bytes"
description    = "human-readable"
search_pattern = "DE AD ?? ?? BE EF"     # required — bytes with '??' wildcards
search_start   = 0x00010000              # optional — default 0x00010000
search_end     = 0x10000000              # optional — default 0x10000000
search_offset  = 0x10                    # optional — bytes from match to patch site
expected       = "00 00 80 3F"           # optional — pre-image at patch site
replace        = "00 00 00 40"           # required — replacement bytes
```

Application timing:

- First pass runs in `xpacks_apply_for_xbe()` at xbe-detect. Most pattern
  patches will miss this pass because the target data hasn't loaded yet.
- `xpacks_tick()` is called from `nv2a_profile_flip_stall()` — every
  guest frame at ~30 Hz. It throttles internally (one scan per second by
  default), re-scanning until each pending pattern lands.
- Once applied, a pattern's `applied` flag is set and it's never re-scanned.
  A patch that "succeeds matching but fails its `expected` check" sets
  `applied=true` too — it disqualifies that specific signature, prevents
  pathological retry loops.

The scan reads guest memory in 64 KB chunks with overlap, skips unmapped
pages, and uses a first-byte filter before the full compare. Cost per scan
of a 64 MB range is well below a frame.

### `kind = "cave"`

Injects new x86 code at a free guest-virtual address (the "cave") and
installs a 5-byte `JMP rel32` trampoline at the site you want to intercept.
Used when `bytes` isn't expressive enough — `bytes` can overwrite existing
instructions but can't *add* logic, so things like "return early from this
function", "skip this loop", or "run a host-style fast path" need a cave.

```toml
[[patch]]
kind            = "cave"
description     = "human-readable"
trampoline_at   = 0x00123110            # required — JMP rel32 install site
trampoline_size = 5                     # optional — bytes displaced (5..15, default 5)
expected        = "55 8B EC 83 EC"      # optional — pre-image at trampoline_at
cave_at         = 0x002B0000            # required — where the cave body lives
cave_expected   = "00 00 00 00"         # optional — verify cave site is free
cave_bytes      = "C2 08 00"            # required — x86 body (here: `ret 8`)
return_in_cave  = true                  # optional — cave handles its own return
```

Cave layout when `return_in_cave = false` (default):

```
cave_at:   [cave_bytes]
           [original displaced bytes]      trampoline_size B
           [JMP rel32 back to trampoline_at + trampoline_size]   5 B
```

Trampoline at `trampoline_at`:

```
[JMP rel32 -> cave_at]                    5 B
[NOP * (trampoline_size - 5)]             0..10 B
```

When `return_in_cave = true`, `cave_bytes` is written verbatim with no
appended tail — the author is responsible for control flow exiting the
cave (typically `ret`, `ret imm16`, `jmp`, etc.).

Sharp edges:

- **`trampoline_size` must cover whole x86 instructions.** The five-byte
  `JMP rel32` overwrites complete instructions only — if the original site
  has instructions whose total length crosses 5 bytes (e.g. `push ebp`
  (1) + `mov ebp, esp` (2) + `sub esp, 8` (3) = 6 B), set
  `trampoline_size = 6`. Including a correct `expected` is the practical
  safeguard.
- **Cave site must be free.** The loader doesn't allocate guest memory. Pick
  unused padding in the xbe (post-section padding, .bss-adjacent slack,
  known-zero `.rdata` tail) and verify with `cave_expected`. If the site is
  actually live code/data, the pack will be refused at apply time.
- **Stack discipline / flag preservation** is the author's responsibility.
  If the displaced bytes were inside a function with active locals, the cave
  body must preserve `eax/ecx/edx`-style scratch convention or push/pop
  what it clobbers.
- **No relocation of displaced instructions.** The displaced bytes are
  copied verbatim. If any of them are PC-relative (`call`, conditional
  jumps, `jmp short/near`), the copy will branch to the wrong place. Avoid
  trampolining onto PC-relative instructions; pick a spot a few bytes later
  or earlier.

### `kind = "shader"`

Replaces a Vulkan SPIR-V module keyed by the GLSL source hash xemu computes
in `pgraph_vk_create_shader_module_from_glsl()`.

```toml
[[patch]]
kind      = "shader"
glsl_hash = "0xfeedfacecafebabe"           # required — uint64 hex
file      = "shaders/feedfacecafebabe.spv" # required — relative to pack dir
```

- The hash is `fast_hash(glsl_source, strlen(glsl_source))`. Capture it by
  enabling debug logging once and watching pgraph compile output.
- The override bypasses both the on-disk SPV cache and the glslang
  compiler. The SPIR-V file must be a complete, valid module for the
  stage it's replacing.

## Identifying targets

### Title id

In the xbe cert at offset 8 from `certificate_addr` (xbe header offset
`0x118`). For Halo 2: image base `0x00010000`, cert at virtual `0x00010184`,
title id at `0x0001018C` (little-endian; manifest uses host-order
`0x4D530064`).

### Guest virtual address

Equal to xbe file offset plus image base. The address shown in Ghidra when
the xbe is mapped at its declared base.

### GLSL hash

`fast_hash` from `qemu/fast-hash.h` over the full GLSL source string.
Capture via xemu's debug logs (Settings → enable Debug Logging, then grep
pgraph shader compile messages).

### Pack id

`<TITLE_ID_HEX>/<pack_dir_name>`. Case-sensitive on disk, but title id
matching is case-insensitive.

## Apply timing

| Kind | When | Hook |
|---|---|---|
| `bytes` | First `xemu_get_xbe_info()` success per title id | [xemu-xbe.c](xemu-xbe.c) |
| `pattern_bytes` | First attempt at xbe-detect + retry on every guest frame via `nv2a_profile_flip_stall()` until found | [hw/xbox/nv2a/pgraph/profile.c](hw/xbox/nv2a/pgraph/profile.c) |
| `cave` | First `xemu_get_xbe_info()` success per title id (same as `bytes`) | [xemu-xbe.c](xemu-xbe.c) |
| `shader` | On GLSL→SPIR-V compile | [hw/xbox/nv2a/pgraph/vk/glsl.c](hw/xbox/nv2a/pgraph/vk/glsl.c) |

Pack enable/disable changes take effect on the **next game launch**.
Toggling a switch does not re-patch a running guest.

## Safety

- `expected` pre-image matching is the load-bearing safety on `bytes` and
  `pattern_bytes`. The loader refuses to write if the pre-image doesn't
  match.
- `match.xbe_code_sha1` disqualifies entire packs at a build mismatch —
  the safer stop for pattern patches that don't have address-anchored
  pre-images.
- Title-id matching prevents cross-game contamination.
- Bundled packs are copy-once on launch; user-edited copies are preserved.

## Spec versioning

`schema = 1` is the current value. Bumps:
- `schema = 2` would be required if patch-kind semantics change or new
  required fields land in existing kinds.
- New optional kinds (e.g. `uniform`, `texture_replace`) do **not** bump
  the schema — the loader logs and ignores unknown kinds.

## Out of scope

- **Cave memory allocator** — the loader does not grow guest memory.
  `kind = "cave"` requires the author to nominate a free region. A future
  extension could carve a dedicated cave heap out of an unused
  physical-RAM range and hand out offsets, removing the "find a hole"
  burden from authors.
- **Displaced-bytes relocator** — caves copy the displaced trampoline
  bytes verbatim. PC-relative instructions (`call rel32`, conditional
  jumps, `jmp short/near`) at the trampoline site will branch wrong; the
  author must pick a non-PC-relative window. A relocator that rewrites
  `E8/E9` displacements at copy time could lift that restriction.
- **File overlay** — replacing files inside the disc image. xemu reads at
  the LBA/sector level via ATAPI, with no file-level hook; would require
  either pre-launch ISO staging or an XDVDFS-aware sector intercept.
  `pattern_bytes` already covers most cases where you'd want this (the
  guest reads the file into memory; once it's there, pattern matches).
