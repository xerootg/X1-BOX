/*
 * Per-frame burst diagnostics implementation. See include/qemu/burst_diag.h.
 *
 * On each FLIP_STALL we read the per-thread counter snapshots from the vCPU
 * (FLIP_STALL is guest MMIO so we're guaranteed to be on the vCPU thread),
 * compute deltas vs the previous frame, maintain an EWMA, and log:
 *
 *   - every 60 frames (a baseline pulse, ~2 s at 30 fps)
 *   - on any frame where dtb or dsf exceeds 1.6 × EWMA (the burst marker)
 *
 * Activation:
 *   X1BOX_BURST_DIAG=1   (env)            — primary
 *   debug.x1box.burst=1  (Android prop)   — runtime toggle via `setprop`
 */
#include "qemu/osdep.h"

#ifdef XBOX

#include "qemu/burst_diag.h"
#include <stdlib.h>
#include <string.h>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#define BD_LOG(...) __android_log_print(ANDROID_LOG_INFO, "x1-burst", __VA_ARGS__)
#else
#include <stdio.h>
#define BD_LOG(fmt, ...) fprintf(stderr, "[x1-burst] " fmt "\n", ##__VA_ARGS__)
#endif

bool burst_diag_active = false;
__thread uint64_t burst_diag_tb_lookups_tls = 0;
__thread uint64_t burst_diag_softfloat_ops_tls = 0;
__thread uint32_t burst_diag_pc_ring[BURST_DIAG_PC_RING_SIZE];
__thread uint32_t burst_diag_pc_ring_idx = 0;

static void burst_diag_init_once(void)
{
    static bool inited = false;
    if (inited) {
        return;
    }
    inited = true;

    const char *e = getenv("X1BOX_BURST_DIAG");
    bool on = (e && e[0] && e[0] != '0');

#ifdef __ANDROID__
    if (!on) {
        char buf[PROP_VALUE_MAX] = {0};
        __system_property_get("debug.x1box.burst", buf);
        if (buf[0] && buf[0] != '0') {
            on = true;
        }
    }
#endif

    burst_diag_active = on;
    BD_LOG("burst_diag init: active=%d (env=%s)", (int)on, e ? e : "(unset)");
}

void burst_diag_on_flip_stall(void)
{
    burst_diag_init_once();
    if (!burst_diag_active) {
        return;
    }

    /*
     * State is single-threaded — FLIP_STALL fires from guest MMIO, which
     * always runs on the vCPU thread. Plain statics are fine here.
     */
    static uint64_t last_tb = 0;
    static uint64_t last_sf = 0;
    static uint32_t frame_idx = 0;
    static double   ewma_tb  = 0.0;
    static double   ewma_sf  = 0.0;
    static uint64_t total_spikes = 0;

    uint64_t cur_tb = burst_diag_tb_lookups_tls;
    uint64_t cur_sf = burst_diag_softfloat_ops_tls;
    uint64_t dtb = cur_tb - last_tb;
    uint64_t dsf = cur_sf - last_sf;
    last_tb = cur_tb;
    last_sf = cur_sf;
    frame_idx++;

    const double alpha = 0.05;
    bool warm = frame_idx > 60;
    bool spike = warm && ((double)dtb > ewma_tb * 1.6 ||
                          (double)dsf > ewma_sf * 1.6);

    ewma_tb = ewma_tb * (1.0 - alpha) + (double)dtb * alpha;
    ewma_sf = ewma_sf * (1.0 - alpha) + (double)dsf * alpha;

    if (spike) {
        total_spikes++;
    }

    bool periodic = (frame_idx % 60u) == 0u;
    if (!(periodic || spike)) {
        return;
    }

    BD_LOG("f=%u dtb=%llu dsf=%llu ewma_tb=%.0f ewma_sf=%.0f spikes=%llu %s",
           (unsigned)frame_idx,
           (unsigned long long)dtb,
           (unsigned long long)dsf,
           ewma_tb,
           ewma_sf,
           (unsigned long long)total_spikes,
           spike ? "SPIKE" : "");

    /*
     * On a spike frame, fold the PC ring into a 4 KiB-page histogram
     * and log the top 3 buckets. The ring is a wraparound of the last
     * BURST_DIAG_PC_RING_SIZE samples (sampling 1 in 256 TB lookups),
     * so it captures roughly the last ring_size * 256 = ~65 k lookups
     * — long enough to span a typical bursty frame.
     */
    if (!spike) {
        return;
    }

    enum { TOP_K = 3, HBUCKETS = BURST_DIAG_PC_RING_SIZE };
    struct {
        uint32_t page;
        uint32_t count;
    } hist[HBUCKETS];
    int hist_n = 0;

    for (uint32_t i = 0; i < BURST_DIAG_PC_RING_SIZE; i++) {
        uint32_t pc = burst_diag_pc_ring[i];
        if (pc == 0) {
            continue;
        }
        uint32_t page = pc & ~0xFFFu;
        int j;
        for (j = 0; j < hist_n; j++) {
            if (hist[j].page == page) {
                hist[j].count++;
                break;
            }
        }
        if (j == hist_n && hist_n < HBUCKETS) {
            hist[hist_n].page = page;
            hist[hist_n].count = 1;
            hist_n++;
        }
    }

    uint32_t top_page[TOP_K] = {0};
    uint32_t top_cnt[TOP_K] = {0};
    for (int j = 0; j < hist_n; j++) {
        for (int k = 0; k < TOP_K; k++) {
            if (hist[j].count > top_cnt[k]) {
                for (int m = TOP_K - 1; m > k; m--) {
                    top_page[m] = top_page[m - 1];
                    top_cnt[m]  = top_cnt[m - 1];
                }
                top_page[k] = hist[j].page;
                top_cnt[k]  = hist[j].count;
                break;
            }
        }
    }

    BD_LOG("  spike top pages: 0x%08x:%u 0x%08x:%u 0x%08x:%u (ring=%u uniq=%d)",
           top_page[0], top_cnt[0],
           top_page[1], top_cnt[1],
           top_page[2], top_cnt[2],
           (unsigned)burst_diag_pc_ring_idx,
           hist_n);
}

#endif /* XBOX */
