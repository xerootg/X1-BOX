/*
 * QEMU MCPX Audio Processing Unit implementation
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2018-2019 Jannik Vogel
 * Copyright (c) 2019-2025 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "apu_int.h"
#ifdef __ANDROID__
#include <android/log.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

MCPXAPUState *g_state; // Used via debug handlers

#include "audio_trace.h"

static void update_irq(MCPXAPUState *d)
{
    if (d->regs[NV_PAPU_FECTL] & NV_PAPU_FECTL_FEMETHMODE_TRAPPED) {
        qatomic_or(&d->regs[NV_PAPU_ISTS], NV_PAPU_ISTS_FETINTSTS);
    }
    if ((d->regs[NV_PAPU_IEN] & NV_PAPU_ISTS_GINTSTS) &&
        ((d->regs[NV_PAPU_ISTS] & ~NV_PAPU_ISTS_GINTSTS) &
         d->regs[NV_PAPU_IEN])) {
        qatomic_or(&d->regs[NV_PAPU_ISTS], NV_PAPU_ISTS_GINTSTS);
        // fprintf(stderr, "mcpx irq raise ien=%08x ists=%08x\n",
        //         d->regs[NV_PAPU_IEN], d->regs[NV_PAPU_ISTS]);
        pci_irq_assert(PCI_DEVICE(d));
    } else {
        qatomic_and(&d->regs[NV_PAPU_ISTS], ~NV_PAPU_ISTS_GINTSTS);
        // fprintf(stderr, "mcpx irq lower ien=%08x ists=%08x\n",
        //         d->regs[NV_PAPU_IEN], d->regs[NV_PAPU_ISTS]);
        pci_irq_deassert(PCI_DEVICE(d));
    }
}

static uint64_t mcpx_apu_read(void *opaque, hwaddr addr, unsigned int size)
{
    MCPXAPUState *d = opaque;

    uint64_t r = 0;
    switch (addr) {
    case NV_PAPU_XGSCNT:
        r = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) / 100; //???
        break;
    default:
        if (addr < 0x20000) {
            r = qatomic_read(&d->regs[addr]);
        }
        break;
    }

    trace_mcpx_apu_reg_read(addr, size, r);
    return r;
}

static void mcpx_apu_write(void *opaque, hwaddr addr, uint64_t val,
                           unsigned int size)
{
    MCPXAPUState *d = opaque;

    trace_mcpx_apu_reg_write(addr, size, val);

    switch (addr) {
    case NV_PAPU_ISTS:
        /* the bits of the interrupts to clear are written */
        qatomic_and(&d->regs[NV_PAPU_ISTS], ~val);
        update_irq(d);
        qemu_cond_broadcast(&d->cond);
        break;
    case NV_PAPU_FECTL:
    case NV_PAPU_SECTL:
        qatomic_set(&d->regs[addr], val);
        qemu_cond_broadcast(&d->cond);
        break;
    case NV_PAPU_FEMEMDATA: {
        /* 'magic write'
         * This value is expected to be written to FEMEMADDR on completion of
         * something to do with notifies. Just do it now :/ */
        hwaddr femem_addr = d->regs[NV_PAPU_FEMEMADDR];
        if (likely(femem_addr + 4 <= d->ram_size)) {
            stl_le_p(d->ram_ptr + femem_addr, val);
            memory_region_set_dirty(d->ram, femem_addr, 4);
        } else {
            stl_le_phys(&address_space_memory, femem_addr, val);
        }
        // fprintf(stderr, "MAGIC WRITE\n");
        qatomic_set(&d->regs[addr], val);
        break;
    }
    default:
        if (addr < 0x20000) {
            qatomic_set(&d->regs[addr], val);
        }
        break;
    }
}

static const MemoryRegionOps mcpx_apu_mmio_ops = {
    .read = mcpx_apu_read,
    .write = mcpx_apu_write,
};

static int monitor_num_used_bytes(MCPXAPUState *d)
{
    int queued_bytes;
    qemu_spin_lock(&d->monitor.fifo_lock);
    queued_bytes = (int)fifo8_num_used(&d->monitor.fifo);
    qemu_spin_unlock(&d->monitor.fifo_lock);
    return queued_bytes;
}

static void throttle(MCPXAPUState *d)
{
    if (d->ep_frame_div % 8) {
        return;
    }

    if (d->monitor.fifo_capacity_bytes <= 0) {
        return;
    }

    int64_t start_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int queued_bytes = monitor_num_used_bytes(d);

#ifdef __ANDROID__
    int64_t _ffwait_t0 = audio_trace_enabled() ? audio_trace_now_ns() : 0;
    int _ffwait_qb_entry = queued_bytes;
    int _ffwait_iters = 0;
#endif
    while (!qatomic_read(&d->exiting) &&
           queued_bytes >= d->monitor.queued_bytes_high) {
        qemu_cond_timedwait(&d->cond, &d->lock, EP_FRAME_US / 1000);
        if (qatomic_read(&d->exiting)) {
            break;
        }
        queued_bytes = monitor_num_used_bytes(d);
#ifdef __ANDROID__
        _ffwait_iters++;
#endif
    }
#ifdef __ANDROID__
    /* If we spent more than one EP_FRAME_US (5.33ms) in the FIFO-full
     * wait, the consumer is draining slower than the producer wants to
     * push. That's the upstream stall when audio backend gates the
     * callback. Log entry queue depth + iteration count + total time so
     * the analyzer can correlate with consumer-gap events. */
    if (audio_trace_enabled() && _ffwait_iters > 0) {
        int64_t wait_us = (audio_trace_now_ns() - _ffwait_t0) / 1000;
        if (wait_us > 5000) {
            __android_log_print(ANDROID_LOG_WARN, "hakuX-apu-bql",
                "t=%lld kind=ffwait wait_us=%lld qb_entry=%d "
                "qb_exit=%d iters=%d",
                (long long)audio_trace_now_ns(),
                (long long)wait_us, _ffwait_qb_entry,
                queued_bytes, _ffwait_iters);
        }
    }
#endif

#ifdef __ANDROID__
    /* Allow the deadline to slip by ~8 audio frames (~43 ms) before
     * snapping forward — Android's scheduler can easily oversleep by
     * several ms, and a too-tight catch-up window was the original
     * reason this whole branch was disabled here. */
    const int64_t catchup_limit_us = 8 * EP_FRAME_US;
#else
    /* Desktop schedulers track the period reliably, so reset the moment
     * we slip a single frame. */
    const int64_t catchup_limit_us = EP_FRAME_US;
#endif

    if (queued_bytes > d->monitor.queued_bytes_low) {
        int64_t now_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
        if (d->next_frame_time_us == 0 ||
            now_us - d->next_frame_time_us > catchup_limit_us) {
#ifdef __ANDROID__
            if (audio_trace_enabled() && d->next_frame_time_us != 0) {
                int64_t slip_us = now_us - d->next_frame_time_us;
                int64_t now_ns = audio_trace_now_ns();
                __android_log_print(ANDROID_LOG_WARN, "hakuX-apu-throt",
                    "t=%lld reason=deadline qb=%d slip_us=%lld limit_us=%lld",
                    (long long)now_ns, queued_bytes,
                    (long long)slip_us, (long long)catchup_limit_us);
            }
#endif
            d->next_frame_time_us = now_us;
        }
        while (!qatomic_read(&d->exiting)) {
            now_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
            int64_t remaining_us = d->next_frame_time_us - now_us;
            if (remaining_us <= 0) {
                break;
            }
            /* Round up to >= 1 ms so we don't busy-spin on hosts whose
             * cond_timedwait returns immediately for sub-ms timeouts. */
            int sleep_ms = (int)((remaining_us + 999) / 1000);
            qemu_cond_timedwait(&d->cond, &d->lock, sleep_ms);
        }
        d->next_frame_time_us += EP_FRAME_US;

        /* Nudge frame timing based on queue level to avoid drifting
         * toward one of the watermarks.
         */
        int mid = (d->monitor.queued_bytes_low + d->monitor.queued_bytes_high) / 2;
        d->next_frame_time_us += (queued_bytes > mid) - (queued_bytes < mid);
    } else {
        d->next_frame_time_us = start_us;
    }
}

static void se_frame(MCPXAPUState *d)
{
    mcpx_apu_update_dsp_preference(d);
    mcpx_debug_begin_frame();
    g_dbg.gp_realtime = d->gp.realtime;
    g_dbg.ep_realtime = d->ep.realtime;

    int64_t start_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int64_t elapsed_us = start_us - d->frame_count_time_us;
    if (elapsed_us >= 1000000) {
        /* A rudimentary calculation to determine approximately how taxed the APU
         * thread is, by measuring how much time we spend building frames.
         * =1: thread is not sleeping and likely falling behind realtime
         * <1: thread is able to complete work on time
         */
        g_dbg.utilization = (double)d->frame_work_acc_us / (double)elapsed_us;
        g_dbg.frames_processed = (int)(d->frame_count * 1000000.0 / elapsed_us + 0.5);
        d->frame_count_time_us = start_us;
        d->frame_count = 0;
        d->frame_work_acc_us = 0;
    }
    d->frame_count++;

    /* Buffer for all mixbins for this frame */
    float mixbins[NUM_MIXBINS][NUM_SAMPLES_PER_FRAME] = { 0 };

    mcpx_apu_vp_frame(d, mixbins);
    mcpx_apu_dsp_frame(d, mixbins);

    if ((d->ep_frame_div + 1) % 8 == 0) {
#if 0
        FILE *fd = fopen("ep.pcm", "a+");
        assert(fd != NULL);
        fwrite(d->apu_fifo_output, sizeof(d->apu_fifo_output), 1, fd);
        fclose(fd);
#endif

        if (0 <= g_config.audio.volume_limit && g_config.audio.volume_limit < 1) {
            float f = pow(g_config.audio.volume_limit, M_E);
            for (int i = 0; i < 256; i++) {
                d->monitor.frame_buf[i][0] *= f;
                d->monitor.frame_buf[i][1] *= f;
            }
        }

        if (d->monitor.fifo_capacity_bytes > 0) {
            /* Compute peak |sample| over the 256-frame buffer BEFORE the
             * push so the trace records the actual amplitude we're about
             * to enqueue. Same s16 values regardless of output format,
             * since float conversion is value-preserving. */
            int trace_peak = 0;
            if (audio_trace_enabled()) {
                for (int i = 0; i < 256; i++) {
                    int l = d->monitor.frame_buf[i][0];
                    int r = d->monitor.frame_buf[i][1];
                    int al = l < 0 ? -l : l;
                    int ar = r < 0 ? -r : r;
                    if (al > trace_peak) trace_peak = al;
                    if (ar > trace_peak) trace_peak = ar;
                }
            }
            int trace_q_pre = 0, trace_q_post = 0;
            if (d->monitor.output_is_float) {
                /* Convert s16 → float32 in normalized [-1, 1] range to match
                 * AAUDIO_FORMAT_PCM_FLOAT semantics. */
                float fbuf[256][2];
                for (int i = 0; i < 256; i++) {
                    fbuf[i][0] = (float)d->monitor.frame_buf[i][0] / 32768.0f;
                    fbuf[i][1] = (float)d->monitor.frame_buf[i][1] / 32768.0f;
                }
                qemu_spin_lock(&d->monitor.fifo_lock);
                if (audio_trace_enabled()) {
                    trace_q_pre = (int)fifo8_num_used(&d->monitor.fifo);
                }
                int num_bytes_free = (int)fifo8_num_free(&d->monitor.fifo);
                assert(num_bytes_free >= sizeof(fbuf));
                fifo8_push_all(&d->monitor.fifo, (uint8_t *)fbuf, sizeof(fbuf));
                if (audio_trace_enabled()) {
                    trace_q_post = (int)fifo8_num_used(&d->monitor.fifo);
                }
                qemu_spin_unlock(&d->monitor.fifo_lock);
            } else {
                qemu_spin_lock(&d->monitor.fifo_lock);
                if (audio_trace_enabled()) {
                    trace_q_pre = (int)fifo8_num_used(&d->monitor.fifo);
                }
                int num_bytes_free = (int)fifo8_num_free(&d->monitor.fifo);
                assert(num_bytes_free >= sizeof(d->monitor.frame_buf));
                fifo8_push_all(&d->monitor.fifo,
                               (uint8_t *)d->monitor.frame_buf,
                               sizeof(d->monitor.frame_buf));
                if (audio_trace_enabled()) {
                    trace_q_post = (int)fifo8_num_used(&d->monitor.fifo);
                }
                qemu_spin_unlock(&d->monitor.fifo_lock);
            }
#ifdef __ANDROID__
            if (audio_trace_enabled()) {
                int64_t now_ns = audio_trace_now_ns();
                int64_t work_us = qemu_clock_get_us(QEMU_CLOCK_REALTIME) - start_us;
                __android_log_print(ANDROID_LOG_INFO, "hakuX-apu-prod",
                    "t=%lld q_pre=%d q_post=%d peak=%d work_us=%lld div=%d",
                    (long long)now_ns, trace_q_pre, trace_q_post,
                    trace_peak, (long long)work_us, d->ep_frame_div);
            }
#endif
        }
        memset(d->monitor.frame_buf, 0, sizeof(d->monitor.frame_buf));
    }

    d->ep_frame_div++;
    d->frame_work_acc_us += qemu_clock_get_us(QEMU_CLOCK_REALTIME) - start_us;

    mcpx_debug_end_frame();
}

/* Note: only supports millisecond resolution on Windows */
static void sleep_ns(int64_t ns)
{
#ifndef _WIN32
        struct timespec sleep_delay, rem_delay;
        sleep_delay.tv_sec = ns / 1000000000LL;
        sleep_delay.tv_nsec = ns % 1000000000LL;
        nanosleep(&sleep_delay, &rem_delay);
#else
        Sleep(ns / SCALE_MS);
#endif
}

static int getenv_int_clamped(const char *name, int min_value, int max_value,
                              int fallback)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }

    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }

    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return (int)parsed;
}

static void monitor_sink_cb(void *opaque, uint8_t *stream, int free_b)
{
    MCPXAPUState *s = MCPX_APU_DEVICE(opaque);
#ifdef __ANDROID__
    int64_t _cb_entry_ns = audio_trace_enabled() ? audio_trace_now_ns() : 0;
#endif

    if (!runstate_is_running()) {
        memset(stream, 0, free_b);
        return;
    }

    /* Audio output callbacks must return promptly. AAudio on Android uses
     * "frames per burst" of ~96–192 frames (2–4 ms at 48 kHz) and treats
     * any callback that misses its burst deadline as a glitch. The previous
     * 10 × 0.5 ms spin-wait for the FIFO to fill was longer than AAudio's
     * burst period, which caused fast/chunky playback. Take whatever's in
     * the FIFO right now; the underrun-fill path below smooths gaps. */
    int avail;
    qemu_spin_lock(&s->monitor.fifo_lock);
    avail = fifo8_num_used(&s->monitor.fifo);
    qemu_spin_unlock(&s->monitor.fifo_lock);

    int copied = 0;
    int to_copy = MIN(free_b, avail);
    while (copied < to_copy) {
        uint32_t chunk_len = 0;
        qemu_spin_lock(&s->monitor.fifo_lock);
        chunk_len = fifo8_pop_buf(&s->monitor.fifo, stream + copied,
                                  to_copy - copied);
        qemu_spin_unlock(&s->monitor.fifo_lock);
        if (!chunk_len) {
            break;
        }
        copied += chunk_len;
    }

    /* Pad any shortfall with silence — matches Cemu's CubebAPI data_cb
     * pattern. We used to fade-out to last_sample then fade-in on
     * resume, but with AAudio's ~500 Hz callback rate any periodic
     * underrun makes those fades audible as amplitude-modulated pulses.
     * Hardware audio devices already have anti-click filtering at the
     * output stage; a clean silence gap is less perceptible than a
     * sub-ms ramp imposed on every callback boundary. */
    bool trace_underrun = copied < free_b;
    if (trace_underrun) {
        memset(stream + copied, 0, free_b - copied);
    }

#ifdef __ANDROID__
    if (audio_trace_enabled()) {
        int64_t now_ns = audio_trace_now_ns();
        /* dur_us tells us how long OUR code held the callback. If the
         * inter-callback gap is long but dur_us is small, the audio
         * backend isn't calling us. If dur_us itself is large, our
         * own code (spin_lock, broadcast) is the slow part. */
        int64_t dur_us = (now_ns - _cb_entry_ns) / 1000;
        __android_log_print(ANDROID_LOG_INFO, "hakuX-apu-cons",
            "t=%lld req=%d avail=%d copied=%d underrun=%d dur_us=%lld",
            (long long)now_ns, free_b, avail, copied,
            trace_underrun ? 1 : 0, (long long)dur_us);
    }
#endif

    qemu_cond_broadcast(&s->cond);
}

static void monitor_init(MCPXAPUState *d)
{
    qemu_spin_init(&d->monitor.fifo_lock);
    d->monitor.fifo_capacity_bytes = 0;
    d->monitor.device_buffer_bytes = 0;
    d->monitor.queued_bytes_low = 0;
    d->monitor.queued_bytes_high = 0;
    d->monitor.output_is_float = false;

    int fifo_frames = 3;
    int audio_samples = 512;
#ifdef __ANDROID__
    /*
     * SDL's OpenSL ES backend on Tensor (Pixel 10a) drains the audio device
     * in tight callback BURSTS rather than one-at-a-time at the per-buffer
     * rate. Observed pattern (traced via hakuX-apu-cons, 2026-05-21): a
     * burst of N callbacks fires ~100 µs apart, drains everything in the
     * FIFO, returns silence for the remaining N - satisfied callbacks,
     * then sleeps ~340 ms before the next burst. The user-perceived gap
     * (100 ms per 400 ms cycle, with small pops at the boundary) is the
     * difference: chunks-we-could-fill vs chunks-the-device-asked-for.
     *
     * The previous bump from (16, 512) → (24, 2048) increased the per-
     * callback request from 2 KiB → 8 KiB (4×) but only grew the FIFO 1.5×,
     * so the FIFO-to-burst ratio went from ~8 down to ~3 — exactly the
     * ratio of audible-to-silent chunks in the trace.
     *
     * Fix: keep audio_samples=2048 for the 43 ms stall tolerance it
     * provides, but size the FIFO to hold a full 8-callback burst (8 ×
     * 8192 = 64 KiB ≈ 64 frames of 1024 B each, or ~1.3 s headroom).
     * Raise the env clamp ceiling so users can override further if their
     * device's OpenSL ES queue is even deeper.
     */
    fifo_frames = 96;
    audio_samples = 2048;
    fifo_frames = getenv_int_clamped("XEMU_ANDROID_AUDIO_FIFO_FRAMES", 3, 256,
                                     fifo_frames);
    audio_samples = getenv_int_clamped("XEMU_ANDROID_AUDIO_SAMPLES", 256, 4096,
                                       audio_samples);
#endif
    struct SDL_AudioSpec sdl_audio_spec = {
        .freq = 48000,
        .format = AUDIO_S16LSB,
        .channels = 2,
        .samples = audio_samples,
        .callback = monitor_sink_cb,
        .userdata = d,
    };

    if (SDL_Init(SDL_INIT_AUDIO) < 0)  {
        fprintf(stderr, "WARNING: Failed to initialize SDL audio subsystem: %s\n",
                SDL_GetError());
        return;
    }

    SDL_AudioDeviceID sdl_audio_dev;
    SDL_AudioSpec obtained_audio_spec;
    sdl_audio_dev = SDL_OpenAudioDevice(NULL, 0, &sdl_audio_spec,
                                        &obtained_audio_spec, 0);
    if (sdl_audio_dev == 0) {
        fprintf(stderr, "WARNING: SDL_OpenAudioDevice failed: %s\n",
                SDL_GetError());
        return;
    }

    /* SDL's AAudio backend silently swaps our requested PCM_I16 stream for
     * PCM_FLOAT when the hardware is float-native (common on Tensor / modern
     * Pixels). It updates obtained.format but does NOT convert samples, so
     * raw s16 bytes get reinterpreted as float32 by the device — audible as
     * varying-intensity pops. Detect the mismatch and convert at push time
     * so AAudio works correctly without forcing OpenSL ES. */
    d->monitor.output_is_float = SDL_AUDIO_ISFLOAT(obtained_audio_spec.format);

    int bytes_per_sample = SDL_AUDIO_BITSIZE(obtained_audio_spec.format) / 8;
    if (bytes_per_sample <= 0) {
        bytes_per_sample = SDL_AUDIO_BITSIZE(sdl_audio_spec.format) / 8;
    }
    if (bytes_per_sample <= 0) {
        bytes_per_sample = 2;
    }
    int device_buffer_bytes = obtained_audio_spec.samples *
                              obtained_audio_spec.channels *
                              bytes_per_sample;
    if (device_buffer_bytes <= 0) {
        device_buffer_bytes = audio_samples * sdl_audio_spec.channels *
                              bytes_per_sample;
    }

    /* frame_bytes is the size of one push to the FIFO in OUTPUT-format
     * bytes (which is what the consumer's callback drains). When the
     * device wants float32, each push doubles in size vs. the s16 case. */
    int frame_bytes = sizeof(d->monitor.frame_buf);
    if (d->monitor.output_is_float) {
        frame_bytes *= sizeof(float) / sizeof(int16_t);  // ×2
    }
    int fifo_capacity_bytes = fifo_frames * frame_bytes;
    fifo8_create(&d->monitor.fifo, fifo_capacity_bytes);
    int drain_bytes = MAX(device_buffer_bytes, frame_bytes);
#ifdef __ANDROID__
    /* AAudio's "frames per burst" can be as small as 96 frames (~384 B at
     * 48 kHz stereo s16), and its callbacks fire at ~500 Hz. The producer
     * pushes 1 KiB every ~5.3 ms (~187 Hz). If queued_bytes_low collapses
     * to the burst size, the FIFO routinely dips below a single callback's
     * size between producer pushes, every dip triggers the underrun
     * fade-in/fade-out, and the device output sounds like a stream of
     * pops at the callback rate. Force enough headroom that the FIFO
     * stays above the burst size even when one producer push slips. */
    drain_bytes = MAX(drain_bytes, 4 * frame_bytes);
#endif
    int max_high = MAX(fifo_capacity_bytes - frame_bytes, frame_bytes);
    d->monitor.fifo_capacity_bytes = fifo_capacity_bytes;
    d->monitor.device_buffer_bytes = device_buffer_bytes;
#ifdef __ANDROID__
    /*
     * OpenSL ES on Tensor (Pixel 10a) drains the device in a tight burst
     * of N callbacks (observed N≈8, each req=8 KiB at audio_samples=2048),
     * spaced ~340 ms apart. The producer THROTTLE WATERMARK caps the FIFO
     * fill level — at 3×drain_bytes the producer stops at 24 KiB, so only
     * ~3 of the 8 callbacks in each burst find samples; the remaining 5
     * memset to silence (audible as the 100 ms gap per 400 ms cycle).
     *
     * Bump the high watermark to 8×drain_bytes so the FIFO can hold an
     * entire burst worth (≈64 KiB at audio_samples=2048). max_high already
     * clamps us to fifo_capacity − frame_bytes, so this is safe even on
     * smaller-FIFO configurations. Once a burst is fully serviced, the
     * device's internal queue spans the 340 ms gap; the producer refills
     * during that quiet period and the next burst lands on a full FIFO.
     *
     * If audio is still stuttering on a device with a larger OpenSL ES
     * queue, the next step is the adaptive PI controller — keep the
     * capacity at 96 frames but learn N from the trace and tune the
     * watermark to N×drain_bytes at runtime.
     */
    d->monitor.queued_bytes_high = MIN(8 * drain_bytes, max_high);
#else
    d->monitor.queued_bytes_high = MIN(3 * drain_bytes, max_high);
#endif
    d->monitor.queued_bytes_low = MIN(drain_bytes, d->monitor.queued_bytes_high);

    SDL_PauseAudioDevice(sdl_audio_dev, 0);
}

static void mcpx_apu_realize(PCIDevice *dev, Error **errp)
{
    MCPXAPUState *d = MCPX_APU_DEVICE(dev);

    dev->config[PCI_INTERRUPT_PIN] = 0x01;

    memory_region_init_io(&d->mmio, OBJECT(dev), &mcpx_apu_mmio_ops, d,
                          "mcpx-apu-mmio", 0x80000);

    memory_region_init_io(&d->vp.mmio, OBJECT(dev), &vp_ops, d,
                          "mcpx-apu-vp", 0x10000);
    memory_region_add_subregion(&d->mmio, 0x20000, &d->vp.mmio);

    memory_region_init_io(&d->gp.mmio, OBJECT(dev), &gp_ops, d,
                          "mcpx-apu-gp", 0x10000);
    memory_region_add_subregion(&d->mmio, 0x30000, &d->gp.mmio);

    memory_region_init_io(&d->ep.mmio, OBJECT(dev), &ep_ops, d,
                          "mcpx-apu-ep", 0x10000);
    memory_region_add_subregion(&d->mmio, 0x50000, &d->ep.mmio);

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);
}

static void mcpx_apu_exitfn(PCIDevice *dev)
{
    MCPXAPUState *d = MCPX_APU_DEVICE(dev);
    d->exiting = true;
    qemu_cond_broadcast(&d->cond);
    qemu_thread_join(&d->apu_thread);
    mcpx_apu_vp_finalize(d);
}

static void mcpx_apu_reset(MCPXAPUState *d)
{
    qemu_mutex_lock(&d->lock); // FIXME: Can fail if thread is pegged, add flag
    memset(d->regs, 0, sizeof(d->regs));

    mcpx_apu_vp_reset(d);

    // FIXME: Reset DSP state
    dsp_invalidate_opcache(d->gp.dsp);
    dsp_invalidate_opcache(d->ep.dsp);
    d->set_irq = false;
    d->next_frame_time_us = 0;
    qemu_cond_signal(&d->cond);
    qemu_mutex_unlock(&d->lock);
}

// Note: This is handled as a VM state change and not as a `pre_save` callback
// because we want to halt the FIFO before any VM state is saved/restored to
// avoid corruption.
static void mcpx_apu_vm_state_change(void *opaque, bool running, RunState state)
{
    MCPXAPUState *d = opaque;

    if (state == RUN_STATE_SAVE_VM) {
        qemu_mutex_lock(&d->lock);
        if (d->gp.dsp) {
            dsp_sync_to_vm(d->gp.dsp);
        }
        if (d->ep.dsp) {
            dsp_sync_to_vm(d->ep.dsp);
        }
    }
}

static int mcpx_apu_post_save(void *opaque)
{
    MCPXAPUState *d = opaque;
    qemu_cond_signal(&d->cond);
    qemu_mutex_unlock(&d->lock);
    return 0;
}

static int mcpx_apu_pre_load(void *opaque)
{
    MCPXAPUState *d = opaque;
    mcpx_apu_reset(d);
    qemu_mutex_lock(&d->lock);
    return 0;
}

static int mcpx_apu_post_load(void *opaque, int version_id)
{
    MCPXAPUState *d = opaque;
    if (d->gp.dsp) {
        dsp_sync_from_vm(d->gp.dsp);
    }
    if (d->ep.dsp) {
        dsp_sync_from_vm(d->ep.dsp);
    }
    qemu_cond_signal(&d->cond);
    qemu_mutex_unlock(&d->lock);
    return 0;
}

static void mcpx_apu_reset_hold(Object *obj, ResetType type)
{
    MCPXAPUState *d = MCPX_APU_DEVICE(obj);
    mcpx_apu_reset(d);
}

static bool vp_dsp_dma_read_count_needed(void *opaque)
{
    DSPDMAState *s = opaque;
    return s->dma_read_count != 0;
}

static const VMStateDescription vmstate_vp_dsp_dma_read_count = {
    .name = "mcpx-apu/dsp-state/dma/dma_read_count",
    .version_id = 1,
    .minimum_version_id = 1,
    .needed = vp_dsp_dma_read_count_needed,
    .fields = (VMStateField[]) {
        VMSTATE_UINT32(dma_read_count, DSPDMAState),
        VMSTATE_END_OF_LIST()
    }
};

const VMStateDescription vmstate_vp_dsp_dma_state = {
    .name = "mcpx-apu/dsp-state/dma",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields      = (VMStateField[]) {
        VMSTATE_UINT32(configuration, DSPDMAState),
        VMSTATE_UINT32(control, DSPDMAState),
        VMSTATE_UINT32(start_block, DSPDMAState),
        VMSTATE_UINT32(next_block, DSPDMAState),
        VMSTATE_BOOL(error, DSPDMAState),
        VMSTATE_BOOL(eol, DSPDMAState),
        VMSTATE_END_OF_LIST()
    },
    .subsections = (const VMStateDescription * const []) {
        &vmstate_vp_dsp_dma_read_count,
        NULL
    }
};

const VMStateDescription vmstate_vp_dsp_core_state = {
    .name = "mcpx-apu/dsp-state/core",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields      = (VMStateField[]) {
        VMSTATE_UINT16(instr_cycle, DspCoreState),
        VMSTATE_UINT32(pc, DspCoreState),
        VMSTATE_UINT32_ARRAY(registers, DspCoreState, DSP_REG_MAX),
        VMSTATE_UINT32_2DARRAY(stack, DspCoreState, 2, 16),
        VMSTATE_UINT32_ARRAY(xram, DspCoreState, DSP_XRAM_SIZE),
        VMSTATE_UINT32_ARRAY(yram, DspCoreState, DSP_YRAM_SIZE),
        VMSTATE_UINT32_ARRAY(pram, DspCoreState, DSP_PRAM_SIZE),
        VMSTATE_UINT32_ARRAY(mixbuffer, DspCoreState, DSP_MIXBUFFER_SIZE),
        VMSTATE_UINT32_ARRAY(periph, DspCoreState, DSP_PERIPH_SIZE),
        VMSTATE_UINT32(loop_rep, DspCoreState),
        VMSTATE_UINT32(pc_on_rep, DspCoreState),
        VMSTATE_UINT16(interrupt_state, DspCoreState),
        VMSTATE_UINT16(interrupt_instr_fetch, DspCoreState),
        VMSTATE_UINT16(interrupt_save_pc, DspCoreState),
        VMSTATE_UINT16(interrupt_counter, DspCoreState),
        VMSTATE_UINT16(interrupt_ipl_to_raise, DspCoreState),
        VMSTATE_UINT16(interrupt_pipeline_count, DspCoreState),
        VMSTATE_INT16_ARRAY(interrupt_ipl, DspCoreState, 12),
        VMSTATE_UINT16_ARRAY(interrupt_is_pending, DspCoreState, 12),
        VMSTATE_UNUSED(4),   /* was num_inst */
        VMSTATE_UINT32(cur_inst_len, DspCoreState),
        VMSTATE_UINT32(cur_inst, DspCoreState),
        VMSTATE_UNUSED(273), /* was: unused(1) + disasm_memory_ptr(4) +
                              * exception_debugging(1) + disasm_prev_inst_pc(4) +
                              * disasm_is_looping(1) + disasm_cur_inst(4) +
                              * disasm_cur_inst_len(2) +
                              * disasm_registers_save(256) */
        VMSTATE_END_OF_LIST()
    }
};

const VMStateDescription vmstate_vp_dsp_state = {
    .name = "mcpx-apu/dsp-state",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (VMStateField[]) {
        VMSTATE_STRUCT(core, DSPState, 1, vmstate_vp_dsp_core_state, DspCoreState),
        VMSTATE_STRUCT(dma, DSPState, 1, vmstate_vp_dsp_dma_state, DSPDMAState),
        VMSTATE_INT32(save_cycles, DSPState),
        VMSTATE_UINT32(interrupts, DSPState),
        VMSTATE_END_OF_LIST()
    }
};


const VMStateDescription vmstate_vp_ssl_data = {
    .name = "mcpx_apu_voice_data",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (VMStateField[]) {
        VMSTATE_UINT32_ARRAY(base, MCPXAPUVPSSLData, MCPX_HW_SSLS_PER_VOICE),
        VMSTATE_UINT8_ARRAY(count, MCPXAPUVPSSLData, MCPX_HW_SSLS_PER_VOICE),
        VMSTATE_INT32(ssl_index, MCPXAPUVPSSLData),
        VMSTATE_INT32(ssl_seg, MCPXAPUVPSSLData),
        VMSTATE_END_OF_LIST()
    }
};

static const VMStateDescription vmstate_mcpx_apu = {
    .name = "mcpx-apu",
    .version_id = 1,
    .minimum_version_id = 1,
    .post_save = mcpx_apu_post_save,
    .pre_load = mcpx_apu_pre_load,
    .post_load = mcpx_apu_post_load,
    .fields = (VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, MCPXAPUState),
        VMSTATE_STRUCT_POINTER(gp.dsp, MCPXAPUState, vmstate_vp_dsp_state,
                               DSPState),
        VMSTATE_UINT32_ARRAY(gp.regs, MCPXAPUState, 0x10000),
        VMSTATE_STRUCT_POINTER(ep.dsp, MCPXAPUState, vmstate_vp_dsp_state,
                               DSPState),
        VMSTATE_UINT32_ARRAY(ep.regs, MCPXAPUState, 0x10000),
        VMSTATE_UINT32_ARRAY(regs, MCPXAPUState, 0x20000),
        VMSTATE_UINT32(vp.inbuf_sge_handle, MCPXAPUState),
        VMSTATE_UINT32(vp.outbuf_sge_handle, MCPXAPUState),
        VMSTATE_STRUCT_ARRAY(vp.ssl, MCPXAPUState, MCPX_HW_MAX_VOICES, 1,
                             vmstate_vp_ssl_data, MCPXAPUVPSSLData),
        VMSTATE_INT32(vp.ssl_base_page, MCPXAPUState),
        VMSTATE_UINT8_ARRAY(vp.hrtf_submix, MCPXAPUState, 4),
        VMSTATE_UINT8(vp.hrtf_headroom, MCPXAPUState),
        VMSTATE_UINT8_ARRAY(vp.submix_headroom, MCPXAPUState, NUM_MIXBINS),
        VMSTATE_UINT64_ARRAY(vp.voice_locked, MCPXAPUState, 4),
        VMSTATE_END_OF_LIST()
    },
};

static void mcpx_apu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->vendor_id = PCI_VENDOR_ID_NVIDIA;
    k->device_id = PCI_DEVICE_ID_NVIDIA_MCPX_APU;
    k->revision = 177;
    k->class_id = PCI_CLASS_MULTIMEDIA_AUDIO;
    k->realize = mcpx_apu_realize;
    k->exit = mcpx_apu_exitfn;

    rc->phases.hold = mcpx_apu_reset_hold;

    dc->desc = "MCPX Audio Processing Unit";
    dc->vmsd = &vmstate_mcpx_apu;
}

static const TypeInfo mcpx_apu_info = {
    .name = "mcpx-apu",
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(MCPXAPUState),
    .class_init = mcpx_apu_class_init,
    .interfaces =
        (InterfaceInfo[]){
            { INTERFACE_CONVENTIONAL_PCI_DEVICE },
            {},
        },
};

static void mcpx_apu_register(void)
{
    type_register_static(&mcpx_apu_info);
}
type_init(mcpx_apu_register);

static void *mcpx_apu_frame_thread(void *arg)
{
    MCPXAPUState *d = MCPX_APU_DEVICE(arg);
#ifdef __ANDROID__
    prctl(PR_SET_NAME, (unsigned long)"mcpx.apu_thread", 0, 0, 0);
    __android_log_print(ANDROID_LOG_INFO, "x1box-thread",
                        "mcpx_apu_frame_thread entered: tid=%d",
                        (int)syscall(__NR_gettid));
#endif
    qemu_mutex_lock(&d->lock);
    while (!qatomic_read(&d->exiting)) {
#ifdef __ANDROID__
        /* Re-pin name each iteration. The APU thread calls bql_lock /
         * update_irq paths that can attach to the JVM (PCI IRQ raise
         * goes through chipset → ioapic → KVM-style routing, some of
         * which on Android touches JNI for SystemServer signalling).
         * One prctl per audio frame is cheap. */
        prctl(PR_SET_NAME, (unsigned long)"mcpx.apu_thread", 0, 0, 0);
#endif
        int xcntmode = GET_MASK(qatomic_read(&d->regs[NV_PAPU_SECTL]),
                                NV_PAPU_SECTL_XCNTMODE);
        uint32_t fectl = qatomic_read(&d->regs[NV_PAPU_FECTL]);
        if (xcntmode == NV_PAPU_SECTL_XCNTMODE_OFF ||
            (fectl & NV_PAPU_FECTL_FEMETHMODE_TRAPPED) ||
            (fectl & NV_PAPU_FECTL_FEMETHMODE_HALTED)) {
            d->set_irq = true;
        }

        if (d->set_irq) {
            qemu_mutex_unlock(&d->lock);
#ifdef __ANDROID__
            /* Time bql_lock to detect BQL contention. The vCPU often
             * holds BQL during MMIO emulation or DMA dispatch; if it
             * hangs onto it for >10ms our apu_thread blocks here. */
            int64_t _bql_t0 = audio_trace_enabled()
                              ? audio_trace_now_ns() : 0;
#endif
            bql_lock();
#ifdef __ANDROID__
            if (audio_trace_enabled()) {
                int64_t bql_acq_ns = audio_trace_now_ns() - _bql_t0;
                /* Only log slow acquisitions to keep logcat sane —
                 * sub-ms is the steady-state. */
                if (bql_acq_ns > 1000000LL) {
                    __android_log_print(ANDROID_LOG_WARN, "hakuX-apu-bql",
                        "t=%lld kind=bql_lock acq_us=%lld",
                        (long long)audio_trace_now_ns(),
                        (long long)(bql_acq_ns / 1000));
                }
            }
#endif
            update_irq(d);
            bql_unlock();
#ifdef __ANDROID__
            int64_t _dlock_t0 = audio_trace_enabled()
                                ? audio_trace_now_ns() : 0;
#endif
            qemu_mutex_lock(&d->lock);
#ifdef __ANDROID__
            if (audio_trace_enabled()) {
                int64_t dlock_acq_ns = audio_trace_now_ns() - _dlock_t0;
                if (dlock_acq_ns > 1000000LL) {
                    __android_log_print(ANDROID_LOG_WARN, "hakuX-apu-bql",
                        "t=%lld kind=dlock_post_irq acq_us=%lld",
                        (long long)audio_trace_now_ns(),
                        (long long)(dlock_acq_ns / 1000));
                }
            }
#endif
            d->set_irq = false;
        }

        xcntmode = GET_MASK(qatomic_read(&d->regs[NV_PAPU_SECTL]),
                            NV_PAPU_SECTL_XCNTMODE);
        fectl = qatomic_read(&d->regs[NV_PAPU_FECTL]);
        if (xcntmode == NV_PAPU_SECTL_XCNTMODE_OFF ||
            (fectl & NV_PAPU_FECTL_FEMETHMODE_TRAPPED) ||
            (fectl & NV_PAPU_FECTL_FEMETHMODE_HALTED)) {
            qemu_cond_wait(&d->cond, &d->lock);
            continue;
        }
        throttle(d);
        /*
         * se_frame is the bulk of the per-frame work — voice fanout
         * (already lock-aware: it drops d->lock around the workers'
         * cond_wait) plus mcpx_apu_dsp_frame (now lock-aware: it grabs
         * d->gp.dsp_lock / d->ep.dsp_lock only around dsp_run, not the
         * whole frame).
         *
         * Holding d->lock across the entire frame used to pin the vCPU
         * on futex_wait for hundreds of µs at every GP/EP MMIO write
         * (~15% of vCPU time on Halo 2's title screen). Drop the lock
         * here so the vCPU can update DSP memory concurrently with the
         * apu_thread, then re-acquire for the loop-top register check.
         *
         * Nothing else in se_frame's reach reads d->-protected state:
         *   - per-apu-thread fields (ep_frame_div, frame_count_*,
         *     next_frame_time_us) are touched only here
         *   - register reads use qatomic
         *   - monitor.fifo / monitor.frame_buf have their own locking
         *   - voice-worker coordination uses vwd->lock
         *   - DSP state is guarded by the per-DSP locks
         */
        qemu_mutex_unlock(&d->lock);
#ifdef __ANDROID__
        int64_t _se_t0 = audio_trace_enabled() ? audio_trace_now_ns() : 0;
#endif
        se_frame((void *)d);
#ifdef __ANDROID__
        int64_t _dlock_t1 = audio_trace_enabled() ? audio_trace_now_ns() : 0;
#endif
        qemu_mutex_lock(&d->lock);
#ifdef __ANDROID__
        if (audio_trace_enabled()) {
            int64_t now_ns = audio_trace_now_ns();
            int64_t se_ns = _dlock_t1 - _se_t0;
            int64_t dlock_acq_ns = now_ns - _dlock_t1;
            /* Only emit when slow — keeps the log signal clean. >5ms is
             * already past the audio frame budget so anything over that
             * is interesting. */
            if (se_ns > 5000000LL || dlock_acq_ns > 1000000LL) {
                __android_log_print(ANDROID_LOG_WARN, "hakuX-apu-bql",
                    "t=%lld kind=se_or_dlock se_us=%lld dlock_us=%lld",
                    (long long)now_ns,
                    (long long)(se_ns / 1000),
                    (long long)(dlock_acq_ns / 1000));
            }
        }
#endif
    }
    qemu_mutex_unlock(&d->lock);
    return NULL;
}

void mcpx_apu_init(PCIBus *bus, int devfn, MemoryRegion *ram)
{
    PCIDevice *dev = pci_create_simple(bus, devfn, "mcpx-apu");
    MCPXAPUState *d = MCPX_APU_DEVICE(dev);

    g_state = d;

    d->ram = ram;
    d->ram_ptr = memory_region_get_ram_ptr(d->ram);
    d->ram_size = memory_region_size(d->ram);

    mcpx_apu_dsp_init(d);

    d->set_irq = false;
    d->exiting = false;

    qemu_mutex_init(&d->lock);
    qemu_cond_init(&d->cond);
    qemu_add_vm_change_state_handler(mcpx_apu_vm_state_change, d);

    mcpx_apu_vp_init(d);
    monitor_init(d);
    qemu_thread_create(&d->apu_thread, "mcpx.apu_thread", mcpx_apu_frame_thread,
                       d, QEMU_THREAD_JOINABLE);
}
