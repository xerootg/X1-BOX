/*
 * Per-frame burst diagnostics for x1-box.
 *
 * Lightweight counters bumped from hot vCPU paths (TB dispatch, softfloat)
 * and snapshotted at each NV097_FLIP_STALL — the user-visible frame
 * boundary. Gated at runtime by X1BOX_BURST_DIAG / debug.x1box.burst so
 * production builds pay only a single predicted-not-taken branch per
 * bump site.
 *
 * See [[project_halo2_audio_sync_bottleneck]] follow-up + the FPS
 * definition in [[feedback_user_visible_fps_definition]].
 */
#ifndef QEMU_BURST_DIAG_H
#define QEMU_BURST_DIAG_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XBOX

#define BURST_DIAG_PC_RING_LOG2 8
#define BURST_DIAG_PC_RING_SIZE (1u << BURST_DIAG_PC_RING_LOG2)
#define BURST_DIAG_PC_SAMPLE_MASK 0xFFu  /* sample 1 in 256 TB lookups */

extern bool burst_diag_active;
extern __thread uint64_t burst_diag_tb_lookups_tls;
extern __thread uint64_t burst_diag_softfloat_ops_tls;
extern __thread uint32_t burst_diag_pc_ring[BURST_DIAG_PC_RING_SIZE];
extern __thread uint32_t burst_diag_pc_ring_idx;

void burst_diag_on_flip_stall(void);

/*
 * Bump the per-thread TB-lookup counter and (1-in-256) sample the
 * next-TB guest PC into a per-thread ring. The sample arm is the bottom
 * 8 bits of the counter post-increment — drift-free, no extra branch.
 * On a spike frame the FLIP_STALL handler folds the ring into a
 * 4 KiB-page histogram and logs the top buckets.
 */
#define BURST_DIAG_BUMP_TB_LOOKUP_PC(_pc)                                  \
    do {                                                                   \
        if (__builtin_expect(burst_diag_active, 0)) {                      \
            uint64_t _bd_cnt = ++burst_diag_tb_lookups_tls;                \
            if ((_bd_cnt & BURST_DIAG_PC_SAMPLE_MASK) == 0) {              \
                burst_diag_pc_ring[burst_diag_pc_ring_idx &                \
                                   (BURST_DIAG_PC_RING_SIZE - 1)] =        \
                    (uint32_t)(_pc);                                       \
                ++burst_diag_pc_ring_idx;                                  \
            }                                                              \
        }                                                                  \
    } while (0)

#define BURST_DIAG_BUMP_TB_LOOKUP()                                        \
    do {                                                                   \
        if (__builtin_expect(burst_diag_active, 0)) {                      \
            ++burst_diag_tb_lookups_tls;                                   \
        }                                                                  \
    } while (0)

#define BURST_DIAG_BUMP_SOFTFLOAT()                                        \
    do {                                                                   \
        if (__builtin_expect(burst_diag_active, 0)) {                      \
            ++burst_diag_softfloat_ops_tls;                                \
        }                                                                  \
    } while (0)

#else /* !XBOX */

#define BURST_DIAG_BUMP_TB_LOOKUP() ((void)0)
#define BURST_DIAG_BUMP_TB_LOOKUP_PC(_pc) ((void)0)
#define BURST_DIAG_BUMP_SOFTFLOAT() ((void)0)
static inline void burst_diag_on_flip_stall(void) {}

#endif /* XBOX */

#ifdef __cplusplus
}
#endif

#endif /* QEMU_BURST_DIAG_H */
