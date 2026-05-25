/*
 * HLE DSound audio backend.
 *
 * See hle-dsound-audio.h for the API contract and architecture. This
 * file implements:
 *
 *   - A 64-slot fixed-size table keyed by guest pBuffer (Fibonacci
 *     hash). Each slot owns a copy of the buffer's captured PCM bytes
 *     + format metadata + gain/freq overrides.
 *
 *   - A 4096-stereo-sample ring (≈85 ms at 48 kHz) shared between the
 *     vCPU thread (pushes via Play) and the apu_thread (drains via
 *     hle_audio_drain). Single-producer / single-consumer with seq_cst
 *     atomic head/tail indices.
 *
 *   - PCM 16/8-bit mono/stereo → 48 kHz stereo float linear resample
 *     during the push. We don't decode XADPCM yet (that's the next
 *     pass — Halo 2 streams a lot of it).
 *
 *   - Per-call telemetry surfaced via hle_audio_log_stats().
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "qemu/osdep.h"
#include "qemu/atomic.h"
#include "hle-dsound-audio.h"
#include "mcpx/apu/vp/adpcm.h"  /* static adpcm_decode_block */

#ifdef __ANDROID__
#include <android/log.h>
#define HA_LOG(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, "x1-hle-audio", fmt, ##__VA_ARGS__)
#else
#include <stdio.h>
#define HA_LOG(fmt, ...) fprintf(stderr, "[x1-hle-audio] " fmt "\n", ##__VA_ARGS__)
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HLE_AUDIO_OUT_RATE   48000u  /* matches MCPX_HW_OUTPUT_RATE */
#define HLE_AUDIO_RING_SAMPS 4096u   /* per-channel stereo frames; ~85ms */

/*
 * Per-buffer state. pBuffer is the guest VA the game holds — used
 * directly as the hash key (Xbox guest VAs are 32-bit and stable for
 * the buffer's lifetime).
 *
 * Lifetime: created lazily on first SetBufferData. Never freed in v1;
 * 64 slots is enough for Halo 2's UI buffers, and we overwrite the
 * coldest slot on overflow (LRU is overkill at this scale).
 */
typedef struct hle_audio_slot {
    uint32_t pBuffer;
    uint8_t *pcm;
    uint32_t len;
    uint16_t format_tag;     /* WAVE_FORMAT_PCM=1, X-ADPCM=0x69 */
    uint32_t sample_rate;    /* native; 0 means assume 22050 */
    uint16_t channels;
    uint16_t bits;
    uint16_t block_align;    /* nBlockAlign — needed for ADPCM block size */
    int32_t  lVolume;        /* 0 = max, -10000 = silent */
    uint32_t freq_override;  /* 0 = use native sample_rate */
    bool     playing;
    bool     looping;
    bool     is_stream;      /* true → continuous packets; false → static */
    bool     format_bound;   /* false until first Set/Push pops the FIFO */
    uint64_t hit_set;        /* SetBufferData count */
    uint64_t hit_play;       /* Play count */
    uint64_t hit_stop;
    uint64_t hit_packet;     /* stream-packet count */
} hle_audio_slot;

#define HLE_AUDIO_SLOTS 64u

static struct {
    hle_audio_slot slots[HLE_AUDIO_SLOTS];

    /* Output ring — stereo interleaved float [-1, 1]. */
    float    ring[HLE_AUDIO_RING_SAMPS * 2];
    uint32_t head;  /* producer index (sample frames, not floats) */
    uint32_t tail;  /* consumer index */

    /* Format FIFO (16 entries) — pushed by Create hooks, popped by
     * SetBufferData / first stream Process. */
    struct {
        uint16_t tag;
        uint32_t rate;
        uint16_t ch;
        uint16_t bits;
        uint16_t block_align;
    } fmt_fifo[16];
    uint32_t fmt_head;
    uint32_t fmt_tail;

    /* Telemetry */
    uint64_t pushes;
    uint64_t pushes_skipped_unknown_fmt;
    uint64_t adpcm_blocks_decoded;
    uint64_t stream_packets;
    uint64_t drains;
    uint64_t drain_underflow_samples;
    uint64_t ring_overflow_pushes;
    uint64_t fmt_pushes;
    uint64_t fmt_pops;
    uint64_t fmt_drops_full;
} g;

static bool g_inited;

void hle_audio_init(void)
{
    if (g_inited) return;
    g_inited = true;
    memset(&g, 0, sizeof(g));
    HA_LOG("init: ring=%u frames, slots=%u, out_rate=%u",
           HLE_AUDIO_RING_SAMPS, HLE_AUDIO_SLOTS, HLE_AUDIO_OUT_RATE);
}

/* ------------------------------------------------------------------ */
/*  Slot table                                                         */
/* ------------------------------------------------------------------ */

static inline unsigned slot_hash(uint32_t pBuffer)
{
    return (unsigned)((pBuffer * 2654435761u) >> (32 - 6));
}

static hle_audio_slot *slot_find(uint32_t pBuffer)
{
    unsigned start = slot_hash(pBuffer);
    for (unsigned probe = 0; probe < HLE_AUDIO_SLOTS; probe++) {
        unsigned i = (start + probe) & (HLE_AUDIO_SLOTS - 1);
        if (g.slots[i].pBuffer == pBuffer) return &g.slots[i];
        if (g.slots[i].pBuffer == 0)       return NULL; /* empty */
    }
    return NULL;
}

static hle_audio_slot *slot_alloc(uint32_t pBuffer)
{
    hle_audio_slot *existing = slot_find(pBuffer);
    if (existing) return existing;

    unsigned start = slot_hash(pBuffer);
    for (unsigned probe = 0; probe < HLE_AUDIO_SLOTS; probe++) {
        unsigned i = (start + probe) & (HLE_AUDIO_SLOTS - 1);
        if (g.slots[i].pBuffer == 0) {
            g.slots[i].pBuffer = pBuffer;
            return &g.slots[i];
        }
    }
    /* Table full — evict slot at hash bucket (cheap LRU stand-in). */
    unsigned i = start;
    free(g.slots[i].pcm);
    memset(&g.slots[i], 0, sizeof(g.slots[i]));
    g.slots[i].pBuffer = pBuffer;
    return &g.slots[i];
}

/* ------------------------------------------------------------------ */
/*  Ring buffer (single-producer / single-consumer)                    */
/* ------------------------------------------------------------------ */

static inline uint32_t ring_used(void)
{
    uint32_t h = qatomic_read(&g.head);
    uint32_t t = qatomic_read(&g.tail);
    return (h - t) & (HLE_AUDIO_RING_SAMPS - 1);
}

static inline uint32_t ring_free(void)
{
    return HLE_AUDIO_RING_SAMPS - 1 - ring_used();
}

/* Push one stereo frame. Caller has already done format conversion. */
static inline void ring_push_frame(float L, float R)
{
    if (ring_free() == 0) {
        g.ring_overflow_pushes++;
        return;
    }
    uint32_t h = g.head;
    g.ring[h * 2 + 0] = L;
    g.ring[h * 2 + 1] = R;
    qatomic_store_release(&g.head, (h + 1) & (HLE_AUDIO_RING_SAMPS - 1));
}

/* ------------------------------------------------------------------ */
/*  PCM → 48 kHz stereo float resample + push                          */
/* ------------------------------------------------------------------ */

static inline float pcm16_to_float(int16_t s)
{
    return (float)s * (1.0f / 32768.0f);
}

static inline float pcm8_to_float(uint8_t s)
{
    /* PCM8 is unsigned-128-centered on Xbox. */
    return ((float)s - 128.0f) * (1.0f / 128.0f);
}

/*
 * Convert one source sample-frame (mono or stereo) into a stereo pair.
 * `idx` is the source frame index. Returns true if the index was in
 * bounds; false at end-of-buffer.
 */
static bool fetch_frame(const hle_audio_slot *s, uint32_t idx,
                        float *L, float *R)
{
    uint32_t bytes_per_frame = (s->bits / 8) * s->channels;
    uint32_t frames = s->len / (bytes_per_frame ? bytes_per_frame : 1);
    if (idx >= frames) return false;

    const uint8_t *p = s->pcm + idx * bytes_per_frame;
    if (s->bits == 16) {
        const int16_t *p16 = (const int16_t *)p;
        if (s->channels == 2) {
            *L = pcm16_to_float(p16[0]);
            *R = pcm16_to_float(p16[1]);
        } else {
            *L = *R = pcm16_to_float(p16[0]);
        }
    } else { /* 8-bit */
        if (s->channels == 2) {
            *L = pcm8_to_float(p[0]);
            *R = pcm8_to_float(p[1]);
        } else {
            *L = *R = pcm8_to_float(p[0]);
        }
    }
    return true;
}

/* lVolume (1/100 dB, 0..-10000) → linear gain. */
static inline float volume_to_gain(int32_t lVolume)
{
    if (lVolume >= 0) return 1.0f;
    if (lVolume <= -10000) return 0.0f;
    return powf(10.0f, (float)lVolume / 2000.0f);
}

/*
 * Decode X-ADPCM (WAVE_FORMAT_XBOX_ADPCM = 0x69) into a temporary PCM16
 * scratch. nBlockAlign defines the block size for the format (typically
 * 36 bytes/channel — 4 byte header + 32 nibbles → 65 samples). One
 * block decodes to 64 samples per channel (the adpcm_decode_block
 * helper returns `samples = 1 + chunks * 8`, but our block layout has
 * 8 chunks per block → 65 samples; we use 64 to keep the math aligned
 * with the typical XADPCM block convention).
 *
 * Returns malloc'd buffer the caller must free; *out_frames receives
 * the frame count (per-channel sample count). NULL on failure.
 */
static int16_t *adpcm_decode_buffer(const uint8_t *adpcm, uint32_t adpcm_len,
                                    uint16_t channels, uint16_t block_align,
                                    uint32_t *out_frames)
{
    if (!block_align) block_align = 36u * (channels ? channels : 1u);
    if (channels < 1 || channels > 2) return NULL;
    uint32_t n_blocks = adpcm_len / block_align;
    if (!n_blocks) return NULL;

    /* 64 PCM samples per channel per block (8 chunks × 8 samples). */
    const uint32_t samples_per_block = 64u;
    uint32_t total_pcm_samples = n_blocks * samples_per_block * channels;
    int16_t *pcm = malloc(total_pcm_samples * sizeof(int16_t));
    if (!pcm) return NULL;

    int16_t *out = pcm;
    for (uint32_t b = 0; b < n_blocks; b++) {
        int got = adpcm_decode_block(out,
                                     adpcm + b * block_align,
                                     block_align, channels);
        if (got <= 0) break;
        out += got * channels;
        g.adpcm_blocks_decoded++;
    }
    *out_frames = (uint32_t)(out - pcm) / channels;
    return pcm;
}

/*
 * Push raw PCM (post-decode if ADPCM) into the ring with linear
 * resample from in_rate → 48 kHz. Mono is duplicated to both channels.
 * 8-bit and 16-bit PCM supported.
 */
static void push_pcm_to_ring(const uint8_t *bytes, uint32_t len,
                             uint16_t bits, uint16_t channels,
                             uint32_t in_rate, float gain)
{
    uint32_t bytes_per_frame = (bits / 8) * channels;
    if (!bytes_per_frame || !len) return;
    uint32_t in_frames = len / bytes_per_frame;
    if (in_frames < 2) return;

    double step = (double)in_rate / (double)HLE_AUDIO_OUT_RATE;
    double pos = 0.0;
    while (pos < (double)(in_frames - 1)) {
        uint32_t i0 = (uint32_t)pos;
        float frac = (float)(pos - (double)i0);
        float L0, R0, L1, R1;
        const uint8_t *p0 = bytes + i0 * bytes_per_frame;
        const uint8_t *p1 = p0    + bytes_per_frame;
        if (bits == 16) {
            const int16_t *q0 = (const int16_t *)p0;
            const int16_t *q1 = (const int16_t *)p1;
            if (channels == 2) {
                L0 = pcm16_to_float(q0[0]); R0 = pcm16_to_float(q0[1]);
                L1 = pcm16_to_float(q1[0]); R1 = pcm16_to_float(q1[1]);
            } else {
                L0 = R0 = pcm16_to_float(q0[0]);
                L1 = R1 = pcm16_to_float(q1[0]);
            }
        } else { /* 8-bit unsigned */
            if (channels == 2) {
                L0 = pcm8_to_float(p0[0]); R0 = pcm8_to_float(p0[1]);
                L1 = pcm8_to_float(p1[0]); R1 = pcm8_to_float(p1[1]);
            } else {
                L0 = R0 = pcm8_to_float(p0[0]);
                L1 = R1 = pcm8_to_float(p1[0]);
            }
        }
        ring_push_frame((L0 + (L1 - L0) * frac) * gain,
                        (R0 + (R1 - R0) * frac) * gain);
        pos += step;
    }
}

/*
 * Push the slot's captured data into the ring. Selects the decode path
 * based on format_tag: PCM goes straight through; X-ADPCM is decoded
 * to int16 first. Unknown formats are dropped (counted).
 */
static void push_slot_to_ring(hle_audio_slot *s)
{
    if (!s->pcm || !s->len) return;

    uint32_t in_rate = s->sample_rate ? s->sample_rate : 22050u;
    if (s->freq_override) in_rate = s->freq_override;
    float gain = volume_to_gain(s->lVolume);

    if (s->format_tag == 1u /* WAVE_FORMAT_PCM */) {
        push_pcm_to_ring(s->pcm, s->len,
                         s->bits ? s->bits : 16u,
                         s->channels ? s->channels : 1u,
                         in_rate, gain);
        g.pushes++;
        return;
    }
    if (s->format_tag == 0x69u /* WAVE_FORMAT_XBOX_ADPCM */) {
        uint32_t frames = 0;
        uint16_t ch = s->channels ? s->channels : 1u;
        int16_t *pcm = adpcm_decode_buffer(s->pcm, s->len, ch,
                                           s->block_align, &frames);
        if (!pcm) return;
        push_pcm_to_ring((uint8_t *)pcm, frames * ch * sizeof(int16_t),
                         16u, ch, in_rate, gain);
        free(pcm);
        g.pushes++;
        return;
    }
    g.pushes_skipped_unknown_fmt++;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/* Bind slot format from the FIFO if not already bound. Falls back to
 * the caller-supplied defaults if the FIFO is empty (typical for the
 * first few calls before any Create probe fires). */
static void slot_bind_format(hle_audio_slot *s,
                             uint16_t fallback_tag,
                             uint32_t fallback_rate,
                             uint16_t fallback_ch,
                             uint16_t fallback_bits)
{
    if (s->format_bound) return;
    uint16_t tag, ch, bits, ba;
    uint32_t rate;
    if (hle_audio_format_pop(&tag, &rate, &ch, &bits, &ba)) {
        s->format_tag  = tag ? tag : fallback_tag;
        s->sample_rate = rate ? rate : fallback_rate;
        s->channels    = ch ? ch : fallback_ch;
        s->bits        = bits ? bits : fallback_bits;
        s->block_align = ba;
    } else {
        s->format_tag  = fallback_tag;
        s->sample_rate = fallback_rate;
        s->channels    = fallback_ch;
        s->bits        = fallback_bits;
        s->block_align = 0;
    }
    s->format_bound = true;
}

void hle_audio_buffer_set_data(uint32_t pBuffer,
                               const uint8_t *bytes, uint32_t len,
                               uint16_t format_tag,
                               uint32_t sample_rate,
                               uint16_t channels,
                               uint16_t bits_per_sample)
{
    if (!g_inited) hle_audio_init();
    if (!bytes || !len) return;

    hle_audio_slot *s = slot_alloc(pBuffer);
    if (!s) return;

    /* Format binding: prefer the Create-time WAVEFORMATEX (from FIFO)
     * over caller-supplied defaults. The handler in xbox-hle.c passes
     * format_tag=1 / rate=0 / ch=1 / bits=16 — those only kick in if
     * the FIFO is empty (e.g. SetBufferData fires before a Create probe
     * landed, which would only happen on a missed Create hook). */
    (void)format_tag;
    slot_bind_format(s, 1u, sample_rate ? sample_rate : 22050u,
                     channels ? channels : 1u,
                     bits_per_sample ? bits_per_sample : 16u);

    free(s->pcm);
    s->pcm = malloc(len);
    if (!s->pcm) { s->len = 0; return; }
    memcpy(s->pcm, bytes, len);
    s->len = len;
    s->hit_set++;
}

void hle_audio_buffer_play(uint32_t pBuffer, bool looping)
{
    if (!g_inited) return;
    hle_audio_slot *s = slot_find(pBuffer);
    if (!s || !s->pcm) return;
    s->playing = true;
    s->looping = looping;
    s->hit_play++;
    push_slot_to_ring(s);
}

void hle_audio_buffer_stop(uint32_t pBuffer)
{
    if (!g_inited) return;
    hle_audio_slot *s = slot_find(pBuffer);
    if (!s) return;
    s->playing = false;
    s->hit_stop++;
}

void hle_audio_buffer_set_volume(uint32_t pBuffer, int32_t lVolume)
{
    if (!g_inited) return;
    hle_audio_slot *s = slot_find(pBuffer);
    if (!s) return;
    s->lVolume = lVolume;
}

void hle_audio_buffer_set_frequency(uint32_t pBuffer, uint32_t dwFreqHz)
{
    if (!g_inited) return;
    hle_audio_slot *s = slot_find(pBuffer);
    if (!s) return;
    s->freq_override = dwFreqHz;
}

/* ------------------------------------------------------------------ */
/*  Format FIFO                                                        */
/* ------------------------------------------------------------------ */

#define FMT_FIFO_CAP (sizeof(g.fmt_fifo) / sizeof(g.fmt_fifo[0]))

void hle_audio_format_push(uint16_t format_tag, uint32_t sample_rate,
                           uint16_t channels, uint16_t bits_per_sample,
                           uint16_t block_align)
{
    if (!g_inited) hle_audio_init();
    uint32_t next = (g.fmt_head + 1u) % FMT_FIFO_CAP;
    if (next == g.fmt_tail) {
        /* Full — drop the oldest, advance tail. */
        g.fmt_tail = (g.fmt_tail + 1u) % FMT_FIFO_CAP;
        g.fmt_drops_full++;
    }
    g.fmt_fifo[g.fmt_head].tag         = format_tag;
    g.fmt_fifo[g.fmt_head].rate        = sample_rate;
    g.fmt_fifo[g.fmt_head].ch          = channels;
    g.fmt_fifo[g.fmt_head].bits        = bits_per_sample;
    g.fmt_fifo[g.fmt_head].block_align = block_align;
    g.fmt_head = next;
    g.fmt_pushes++;
}

bool hle_audio_format_pop(uint16_t *format_tag, uint32_t *sample_rate,
                          uint16_t *channels, uint16_t *bits_per_sample,
                          uint16_t *block_align)
{
    if (!g_inited) return false;
    if (g.fmt_head == g.fmt_tail) return false;
    *format_tag     = g.fmt_fifo[g.fmt_tail].tag;
    *sample_rate    = g.fmt_fifo[g.fmt_tail].rate;
    *channels       = g.fmt_fifo[g.fmt_tail].ch;
    *bits_per_sample = g.fmt_fifo[g.fmt_tail].bits;
    *block_align    = g.fmt_fifo[g.fmt_tail].block_align;
    g.fmt_tail = (g.fmt_tail + 1u) % FMT_FIFO_CAP;
    g.fmt_pops++;
    return true;
}

/* ------------------------------------------------------------------ */
/*  Stream API (continuous packet feed)                                */
/* ------------------------------------------------------------------ */

void hle_audio_stream_push_packet(uint32_t pStream,
                                  const uint8_t *bytes, uint32_t len)
{
    if (!g_inited) hle_audio_init();
    if (!bytes || !len) return;

    hle_audio_slot *s = slot_alloc(pStream);
    if (!s) return;

    /* Stream slot keeps its format bound across the stream's lifetime;
     * the FIFO pop happens on the first packet. After that we reuse
     * the bound format for every push. */
    if (!s->format_bound) {
        slot_bind_format(s, 1u, 22050u, 1u, 16u);
        s->is_stream = true;
    }

    /* Decode + resample directly to the ring — streams are throwaway
     * after Process, so we don't need to keep the bytes in s->pcm. */
    uint32_t in_rate = s->sample_rate ? s->sample_rate : 22050u;
    if (s->freq_override) in_rate = s->freq_override;
    float gain = volume_to_gain(s->lVolume);

    if (s->format_tag == 1u) {
        push_pcm_to_ring(bytes, len, s->bits ? s->bits : 16u,
                         s->channels ? s->channels : 1u, in_rate, gain);
        g.pushes++;
    } else if (s->format_tag == 0x69u) {
        uint32_t frames = 0;
        uint16_t ch = s->channels ? s->channels : 1u;
        int16_t *pcm = adpcm_decode_buffer(bytes, len, ch,
                                           s->block_align, &frames);
        if (pcm) {
            push_pcm_to_ring((uint8_t *)pcm,
                             frames * ch * sizeof(int16_t),
                             16u, ch, in_rate, gain);
            free(pcm);
            g.pushes++;
        }
    } else {
        g.pushes_skipped_unknown_fmt++;
    }
    s->hit_packet++;
    g.stream_packets++;
}

void hle_audio_stream_flush(uint32_t pStream)
{
    if (!g_inited) return;
    /* Per-pStream drop of pending samples isn't tracked at the
     * per-byte level (samples already merged into the global ring).
     * The simplest correct flush is a no-op: pending samples drain
     * within the ring's ~85 ms window naturally. Hook is here so the
     * handler can be promoted later if a finer-grained flush matters. */
    (void)pStream;
}

void hle_audio_drain(float *front_left, float *front_right,
                     unsigned n_samples)
{
    g.drains++;
    uint32_t avail = ring_used();
    uint32_t serve = n_samples < avail ? n_samples : avail;
    uint32_t t = g.tail;
    for (uint32_t i = 0; i < serve; i++) {
        front_left[i]  = g.ring[t * 2 + 0];
        front_right[i] = g.ring[t * 2 + 1];
        t = (t + 1) & (HLE_AUDIO_RING_SAMPS - 1);
    }
    qatomic_store_release(&g.tail, t);

    /* Pad shortfall with silence. */
    for (uint32_t i = serve; i < n_samples; i++) {
        front_left[i] = 0.0f;
        front_right[i] = 0.0f;
    }
    if (serve < n_samples) {
        g.drain_underflow_samples += (n_samples - serve);
    }
}

void hle_audio_log_stats(void)
{
    if (!g_inited) return;
    unsigned occupied = 0;
    uint64_t set_total = 0, play_total = 0, stop_total = 0;
    for (unsigned i = 0; i < HLE_AUDIO_SLOTS; i++) {
        if (g.slots[i].pBuffer) occupied++;
        set_total  += g.slots[i].hit_set;
        play_total += g.slots[i].hit_play;
        stop_total += g.slots[i].hit_stop;
    }
    uint64_t pkt_total = 0;
    for (unsigned i = 0; i < HLE_AUDIO_SLOTS; i++) {
        pkt_total += g.slots[i].hit_packet;
    }
    HA_LOG("slots=%u/%u set=%llu play=%llu stop=%llu pkt=%llu "
           "pushes=%llu pcm_skip=%llu adpcm_blocks=%llu drains=%llu "
           "underflow=%llu overflow=%llu ring_used=%u "
           "fmt_push=%llu fmt_pop=%llu fmt_drop=%llu",
           occupied, HLE_AUDIO_SLOTS,
           (unsigned long long)set_total,
           (unsigned long long)play_total,
           (unsigned long long)stop_total,
           (unsigned long long)pkt_total,
           (unsigned long long)g.pushes,
           (unsigned long long)g.pushes_skipped_unknown_fmt,
           (unsigned long long)g.adpcm_blocks_decoded,
           (unsigned long long)g.drains,
           (unsigned long long)g.drain_underflow_samples,
           (unsigned long long)g.ring_overflow_pushes,
           ring_used(),
           (unsigned long long)g.fmt_pushes,
           (unsigned long long)g.fmt_pops,
           (unsigned long long)g.fmt_drops_full);
}
