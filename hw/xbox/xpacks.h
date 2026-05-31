/*
 * xemu xpacks — runtime patch-pack system
 *
 * Modeled on Cemu's graphic packs. Loads "packs" from <base>/xpacks/<title_id>/<pack>/
 * and applies three patch primitives at runtime:
 *
 *   kind = "bytes"          fixed guest-virtual address patch
 *                           (xbe code/data — apply once at xbe-detect)
 *   kind = "pattern_bytes"  pattern-anchored guest memory patch
 *                           (tag-data / map overlays — retries on every tick
 *                            until the pattern appears, since maps load late)
 *   kind = "cave"           code cave + JMP rel32 trampoline
 *                           (inject new x86 at a free address, redirect a
 *                            site to it; loader appends displaced bytes +
 *                            return jump unless return_in_cave is set)
 *   kind = "shader"         SPIR-V replacement keyed by GLSL hash
 *                           (intercepted in pgraph_vk compile path)
 *
 * Application timing:
 *   - bytes patches   : applied once when xemu_get_xbe_info() first succeeds.
 *   - pattern patches : retried on every xpacks_tick() call until found.
 *   - shader patches  : looked up at GLSL→SPIR-V compile time.
 *
 * Optional [match].xbe_code_sha1 entries gate the entire pack on a SHA1 of the
 * xbe code section, so a pack authored against one build is refused on others.
 *
 * Toggling a pack on/off requires a guest restart.
 */

#ifndef HW_XBOX_XPACKS_H
#define HW_XBOX_XPACKS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct xbe; /* forward; see xemu-xbe.h */

/* Apply all enabled byte-patches for the running xbe.
 * Idempotent: returns immediately on subsequent calls for the same title.
 * Returns the number of bytes patches successfully applied.
 *
 * Pattern patches are NOT applied here — call xpacks_tick() periodically.
 */
int xpacks_apply_for_xbe(const struct xbe *xbe);

/* Re-scan guest memory for pending pattern_bytes patches and apply any whose
 * signature is now resident. Safe to call on every render frame — early-exits
 * if no pending patterns remain. Returns count newly applied this call. */
int xpacks_tick(void);

/* Look up a SPIR-V override by GLSL hash. Returns NULL if no override.
 * Caller takes ownership of the returned buffer and must g_free() it.
 * out_len receives the SPIR-V byte length.
 */
void *xpacks_lookup_spirv(uint64_t glsl_hash, size_t *out_len);

/* Reset internal state (test/debug). */
void xpacks_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* HW_XBOX_XPACKS_H */
