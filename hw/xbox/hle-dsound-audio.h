/*
 * HLE DSound audio backend — host-side audio routing for the Xbox
 * DirectSound HLE port. Lives in MCPX's output mixbins so we reuse the
 * existing DSP+SDL output chain instead of opening a second audio
 * device.
 *
 * Flow:
 *   guest IDirectSoundBuffer_Play          (in xbox-hle.c handler)
 *     → hle_audio_buffer_play(pBuffer)
 *       → push the buffer's captured PCM into the host stereo ring
 *   apu_thread voice_work_dispatch         (in vp.c bypass branch)
 *     → hle_audio_drain(mixbins_L, mixbins_R, 32)
 *       → pop 32 samples, write into FRONT_LEFT/RIGHT mixbins
 *
 * No threading primitives — ring access is single-producer-single-
 * consumer (vCPU thread pushes from HLE handlers, apu_thread pops in
 * dispatch). Counters are plain seq_cst atomics.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef HW_XBOX_HLE_DSOUND_AUDIO_H
#define HW_XBOX_HLE_DSOUND_AUDIO_H

#include "qemu/osdep.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One-shot init. Cheap — allocates the ring + per-buffer slot table.
 * Safe to call repeatedly. Called from xbox_hle_init() when
 * X1BOX_HLE_DSOUND=1.
 */
void hle_audio_init(void);

/*
 * Capture audio bytes for a guest IDirectSoundBuffer pointer.
 *
 * The guest hands us a pvBufferData pointer + dwBufferBytes via the
 * SetBufferData call. We copy the bytes into a per-pBuffer host slot.
 * Subsequent Play(pBuffer) replays from this slot.
 *
 * format_tag:    WAVE_FORMAT_PCM (1) is the only thing v1 plays back;
 *                everything else is captured but skipped on Play.
 *                ADPCM is the common alternative — landing soon.
 * sample_rate:   in Hz (8000..48000 typical)
 * channels:      1 or 2
 * bits_per_sample: 8 or 16
 *
 * Set sample_rate=0 to indicate "format unknown" — the playback path
 * assumes 22050 mono 16-bit (Halo 2's most common UI default) and
 * resamples accordingly.
 */
void hle_audio_buffer_set_data(uint32_t pBuffer,
                               const uint8_t *bytes, uint32_t len,
                               uint16_t format_tag,
                               uint32_t sample_rate,
                               uint16_t channels,
                               uint16_t bits_per_sample);

/* Start playback for a captured buffer. looping=true keeps the ring
 * topped up; looping=false plays once and lets the ring drain. */
void hle_audio_buffer_play(uint32_t pBuffer, bool looping);

/* Stop playback for a buffer. Drops anything already queued FROM that
 * buffer; samples from other buffers stay in the ring. */
void hle_audio_buffer_stop(uint32_t pBuffer);

/* Apply gain to subsequent samples FROM this buffer. lVolume is the
 * Xbox attenuation in 1/100 dB units (0 = max, -10000 = silent). */
void hle_audio_buffer_set_volume(uint32_t pBuffer, int32_t lVolume);

/* Apply frequency scaling to subsequent samples FROM this buffer. 0
 * means "use the buffer's natural rate". */
void hle_audio_buffer_set_frequency(uint32_t pBuffer, uint32_t dwFreqHz);

/*
 * Phase 3: IDirectSoundBuffer::Unlock arrived. The game has just
 * written `len` bytes at the audio pointer the matching Lock returned
 * — i.e. raw PCM (or ADPCM) at the buffer's declared format. Push
 * directly into the HLE ring, lazily creating + format-binding the
 * slot on first call (via the format FIFO populated by Create probes).
 *
 * This is the load-bearing capture path for buffers that NEVER call
 * SetBufferData — Bink + any other custom-mixer / streaming-into-
 * static-buffer pattern. Without it, bypass=on produces silence for
 * those buffers because no Play/SetBufferData ever fires.
 *
 * Caller has already capped `len` (see xbox-hle.c). Safe to call
 * before init.
 */
void hle_audio_buffer_unlock_write(uint32_t pBuffer,
                                   const uint8_t *bytes, uint32_t len);

/*
 * Format-capture FIFO. Hook the various Create* entries as declining
 * probes — they snapshot the WAVEFORMATEX from the guest descriptor,
 * push it here, then return false so the real Create runs and gives
 * back a real guest pBuffer/pStream. The next SetBufferData (or first
 * stream Process) pops the FIFO and binds the format to its slot.
 *
 * Ring is 16 deep — enough that we don't lose formats across Halo 2's
 * batched Create-then-SetData patterns. Drops oldest on overflow.
 */
void hle_audio_format_push(uint16_t format_tag,
                           uint32_t sample_rate,
                           uint16_t channels,
                           uint16_t bits_per_sample,
                           uint16_t block_align);

/* Returns true if a format was popped; *out_* fields are set. */
bool hle_audio_format_pop(uint16_t *format_tag,
                          uint32_t *sample_rate,
                          uint16_t *channels,
                          uint16_t *bits_per_sample,
                          uint16_t *block_align);

/*
 * Stream API. Streams differ from buffers in that they're persistent
 * channels fed by repeated packets — game calls Create once, then
 * Process per packet for the life of the stream. We pop format from
 * the FIFO on the first push (lazy binding), then push every packet's
 * PCM/ADPCM through the same decode + resample pipeline as static
 * buffers.
 */
void hle_audio_stream_push_packet(uint32_t pStream,
                                  const uint8_t *bytes, uint32_t len);
void hle_audio_stream_flush(uint32_t pStream);

/*
 * Drain up to `n_samples` stereo float samples from the ring into the
 * caller-provided per-bin arrays. Pads with silence if the ring runs
 * dry. Always writes exactly `n_samples` to each output array.
 *
 * Called from vp.c's voice_work_dispatch bypass path once per audio
 * frame (NUM_SAMPLES_PER_FRAME = 32). MUST NOT BLOCK — runs in the
 * apu_thread realtime cadence.
 */
void hle_audio_drain(float *front_left, float *front_right,
                     unsigned n_samples);

/* Number of stereo sample frames currently queued in the ring.
 * Used by vp.c to decide whether to use HLE audio or fall back
 * to MCPX VP output when the ring is dry. */
unsigned hle_audio_ring_available(void);

/*
 * Telemetry: per-API call counts, ring fullness, slot occupancy.
 * Logged from xbox_hle_log_stats() on the existing 100K-TB cadence.
 */
void hle_audio_log_stats(void);

/*
 * Phase-2 parity counters. vp.c's per-voice bypass increments
 * voice_off_bypass when it fires voice_off(); the real voice_process
 * path increments voice_off_real. We want the bypass/real ratio
 * within [0.8, 1.2] over a measurement window — outside that means
 * the bypass's cursor advance is drifting from MCPX semantics.
 */
void hle_audio_count_voice_off_bypass(void);
void hle_audio_count_voice_off_real(void);

/*
 * Phase-1 voices-per-frame histogram. Called once per EP-frame from
 * vp.c with the number of voices the bypass path processed this
 * frame. Bucketing: 0, 1-2, 3-4, 5-8, 9-16, 17-32, 33-64, 64+.
 */
void hle_audio_record_bypass_active_voices(unsigned n);

/*
 * Phase-2 stream_process latency tracking. Caller stamps before and
 * after the snoop body; we maintain count/sum/min/max/over-budget.
 * "over_budget" counts calls that exceeded 5 ms — the realtime cap
 * documented in the plan.
 */
void hle_audio_record_stream_process_latency_ns(uint64_t ns);

#ifdef __cplusplus
}
#endif

#endif /* HW_XBOX_HLE_DSOUND_AUDIO_H */
