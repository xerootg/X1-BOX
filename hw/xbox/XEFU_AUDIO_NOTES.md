# XEFU Audio Emulation — Reverse Engineering Notes

Findings from Ghidra analysis of `xefu2021c.xex.0` (Xbox 360 backward-compatibility
emulator). Documents the full audio emulation pipeline as a reference for xemu's
`hw/xbox/hle-dsound-audio.c` HLE implementation.

## Architecture: HLE + 360 hardware passthrough

XEFU does **not** implement a DSP56300 interpreter or JIT for the MCPX APU. It
intercepts the Xbox 1 DirectSound-for-Xbox (FSDX) API and translates every call
into Xbox 360 hardware operations. The MCPX DSP56300 is replaced entirely.

```
Xbox 1 game
    │
    ▼ FSDX (DirectSound for Xbox) API
    │  FSDX_DispatchIRP / FSDX_IRP_DeviceControl_*
    │
    ├──[VP path]─────────────────────────────────────────────────────────────┐
    │   CPU-side ADPCM → PCM16 decode + resample to 48 kHz                  │
    │   VP_CmdBuild_* → XmaVoiceCtx (0x60-byte XEFU wrapper)                │
    │   XMACreateContext kernel syscall → 64-byte XMA_CONTEXT_DATA slot     │
    │   XMA_CONTEXT_DATA.input_buffer_0_ptr → physical address of PCM16     │
    │   XMA hardware: PCM passthrough mode (no XMA decode) → output ring    │
    │                                                                         │
    ├──[GP path]─────────────────────────────────────────────────────────────┤
    │   I3DL2 reverb/chorus/flange preset → 360 GP DSP handle lookup        │
    │   GP_CmdBuild_* → opcode 0x0a "load script by handle"                 │
    │   No microcode interpretation; .scr names are keys, not code          │
    │                                                                         │
    └──[EP path]─────────────────────────────────────────────────────────────┘
        EP_XAudioRenderDriverCallback → XAudio2 render client
        256 samples × 6 channels × float32 @ 48 kHz = 0x1800 bytes/frame
        EP_CmdBuild_MixerMatrix7x2 → 7 virtual buses → 2 XAudio channels
```

## VP (Voice Processor) — key facts

### XmaVoiceCtx struct (0x60 bytes, XEFU-internal)

Separate from the 64-byte `XMA_CONTEXT_DATA` hardware struct. The hardware
context object is allocated via `XMACreateContext` kernel syscall; XEFU's
per-voice wrapper holds a handle to it plus double-buffered PCM output regions.

```c
struct XmaVoiceCtx {          // stride 0x60
    +0x00  uint32  word0;     // block_count[26:22], subframe_bits[21:16]
    +0x04  uint32  word1;     // active[31], sample_rate_code[28:27], channels[29], subframe_count[23:20]
    +0x08  uint32  word2;
    +0x0C  uint32  word3;     // loop_start[25:0]
    +0x10  uint32  word4;     // play flag[31]
    +0x14  uint32  word5;     // decoded sample count
    +0x18  uint32  word6;
    +0x1C  uint32  phys_pcm;  // MmGetPhysicalAddress(pcm_buf)
    +0x20  uint32  phys_pcm2;
    ...
    +0x40  ptr     xma_ctx_handle;   // XMACreateContext() result
    +0x44  uint32  pcm_buf_pa;
    +0x48  uint32  pcm_buf2_pa;
    +0x4C  uint32  pos_flags;        // bits[4:0]
    +0x50  uint16  context_index;    // hw slot = (MmGetPhysAddr(handle) - base_pa) >> 6
    +0x52  uint16  reset_flag;
    +0x54  uint32  pcm_buf_va;
    +0x58  uint32  pcm_buf2_va;
};
```

### XMA_CONTEXT_DATA (64 bytes — canonical hardware struct)

Canonical layout from xenia (`xenia/apu/xma_context.h`). Each field is a
packed bitfield in big-endian (PowerPC) memory order. Lives in the hardware
DMA region addressed by `0x7FEA1800`.

```
DWORD 0:  input_buffer_0_packet_count[11:0] | loop_count[19:12] |
          input_buffer_0_valid[20] | input_buffer_1_valid[21] |
          output_buffer_block_count[26:22] | output_buffer_write_offset[31:27]
DWORD 1:  input_buffer_1_packet_count[11:0] | loop_subframe_start[13:12] |
          loop_subframe_end[16:14] | loop_subframe_skip[19:17] |
          subframe_decode_count[23:20] | subframe_skip_count[26:24] |
          sample_rate[28:27] | is_stereo[29] | output_buffer_valid[31]
DWORD 2:  input_buffer_read_offset[25:0]   (XMAGetInputBufferReadOffset, in bits)
DWORD 3:  loop_start[25:0]                 (bit offset of loop start frame)
DWORD 4:  loop_end[25:0] | packet_metadata[30:26] | current_buffer[31]
DWORD 5:  input_buffer_0_ptr               (physical address)
DWORD 6:  input_buffer_1_ptr               (physical address)
DWORD 7:  output_buffer_ptr                (physical address)
DWORD 8:  work_buffer_ptr
DWORD 9:  output_buffer_read_offset[4:0] | stop_when_done[30] | interrupt_when_done[31]
DWORDS 10-15: reserved
```

### PCM passthrough mode — the critical finding

Xbox 1 games use ADPCM and PCM audio, not XMA. The XMA hardware on Xbox 360
normally requires XMA-compressed input packets. XEFU exploits an undocumented
**PCM passthrough mode**:

- `XMA_CONTEXT_DATA.word0 bits[26:22]` = block count set
- `XMA_CONTEXT_DATA.word0 bits[31:27]` = **never set** (no XMA2 frame header)
- Without a valid XMA frame header, hardware treats the input buffer as raw
  16-bit PCM and DMA-copies it straight to the output buffer

The "decode type" at `iVar3+4` in XEFU's voice context lookup: `0=PCM, 2=XMA`.
All Xbox 1 BC voices use type 0 (PCM). `input_buffer_0_ptr` points at decoded
PCM16, not XMA packets. `XMASetInputBuffer0/1` kernel calls are never made.

### Decode happens upstream

ADPCM decode happens **before** the VP/XMA layer, in the FSDX HLE shim:

1. FSDX IRP arrives with WAVEFORMATEX + ADPCM data pointer
2. Shim software-decodes ADPCM → PCM16 and resamples to 48 kHz
3. `XMA_ConfigureVoicePool` stores `output_sample_rate = 48000` (hardcoded)
4. `VP_SetVoiceSampleBuffer` points `input_buffer_0_ptr` at the decoded PCM
5. XMA hardware DMA-copies PCM to output ring buffer (PCM passthrough)

## XMA2 MMIO register map (base 0x7FEA0000)

All reads use `lwbrx` (load word byte-reversed) — registers are big-endian.

| Address       | Register   | Description |
|---------------|------------|-------------|
| `0x7FEA1800`  | `ContextArrayAddress` | Physical base of 64-byte context DMA region |
| `0x7FEA1804`  | *(unnamed)* | Engine command: write 0 (clear), write `0x03000000` (reset pulse, bits[25:24]) |
| `0x7FEA1818`  | `CurrentContextIndex` | Hardware writes current-processing context slot index here |
| `0x7FEA181C`  | `NextContextIndex`  | |
| `0x1FFA8690`  | *(scheduler)* | Per-frame context activation bitmask (written before each decode pass) |
| `0x1FFA86A0`  | *(scheduler)* | Persistent context registration bitmask (written at XMACreateContext time) |

*Note: the kick/lock/clear registers seen in xenia (`Context0Kick` at `0x7FEA1940`,
`Context0Lock` at `0x7FEA1A40`) are not directly written by XEFU — it uses
the separate scheduler bitmaps at `0x1FFA86xx` instead.*

### Per-frame sequence (XMA_RenderFrameUpdate @ 0x82137E48)

```
1.  KeRaiseIrqlToDpcLevel + spinlock
2.  FUN_821344d8: write activation bitmask → 0x1FFA8690 (eieio)
3.  XMA_WriteVoiceRegistersAndReset:
      a. Poll 0x7FEA1818 (byte-reversed): hardware writes active context_index here
         XOR with 0x200 = "done token"; scan all voice.context_index fields
         Spin (125ms timeout) until all contexts idle
      b. dcbz PCM output buffers (zero + cache-flush per voice)
      c. Write 0x00000000 → 0x7FEA1804 (eieio)
      d. Write 0x03000000 → 0x7FEA1804 (eieio)   ← reset pulse
      e. Re-poll 0x7FEA1818 for idle confirmation
4.  XMA_ResetVoiceRegisterBlock per voice: re-init SW state, set active bit
5.  KfLowerIrql + release spinlock
6.  Dispatch to XAudio EP mixer chain
```

### Completion detection

`0x7FEA1818` is **not** a bitmask of done contexts. The hardware writes the
slot index of the context it is currently processing. XEFU XORs this with
`0x200` (frame-parity toggle bit) and scans all `voice.context_index` fields.
If none match, all contexts are idle.

## GP (Global Processor) — effects

XEFU maps I3DL2 reverb/chorus/flange to 360 GP DSP program handles. The `.scr`
files (reverb.scr, chorus.scr, etc.) are **lookup keys only** — not microcode.

- `GP_CmdBuild_SelectEffectPreset` (0x82090F00): preset name → GP handle lookup
  in context at `+0x644..+0x650`, then emits opcode `0x0a` ("load script by handle")
- `GP_CmdBuild_ReverbEffectRoute` (0x820A3690): 37-way effect I/O routing table
- Bus slot allocation: `GP_OptimizeEffectBusSlotAlloc`, `GP_TryMergeEffectBusNodes`
- No script is interpreted; the entire GP "emulation" is a preset-name→handle map

## EP (Encode Processor) — output

- `EP_XAudioRenderDriverCallback` (0x8212FD68): registered as XAudio2 render driver
- Frame: 256 samples × 6 channels × float32 @ 48 kHz = `0x1800` bytes
- `EP_CmdBuild_MixerMatrix7x2` (0x820A5F78): 7 virtual bus outputs → 2 XAudio channels
  (opcodes `0x71`, fixed coefficients; routes Xbox 1 2.0/4.0/5.1 → 360 stereo)
- SRC for non-48kHz: `48000 / rate` integer ratio in `EP_SetSampleRateConverter`

## Implications for xemu HLE-DSound

| XEFU behavior | xemu equivalent |
|---|---|
| ADPCM decoded in FSDX HLE shim before VP layer | Decode in `hle_audio_buffer_set_data` at ingestion time, not at play time |
| Output always 48000 Hz (hardcoded) | `HLE_AUDIO_OUT_RATE = 48000` — correct |
| XMA hardware used as PCM passthrough mixer | Our audio ring + vp.c bypass drain is the equivalent |
| PCM mode: no XMA frame header, just block_count in word0 | N/A for xemu — we have no XMA hardware to configure |
| No `XMASetInputBuffer0/1` calls | N/A — xemu doesn't implement those kernel calls |
| `input_buffer_read_offset` is meaningless for PCM | `XMA_CONTEXT_DATA` loop fields irrelevant for Xbox 1 emulation |
| Double-buffered PCM output (ping-pong) | Our ring buffer provides equivalent producer/consumer decoupling |

## Named functions (selected)

```
VP layer:
  VP_CmdBuild_ActivateVoice       0x820A0xxx   voice activation command
  VP_CmdBuild_VoiceFormat0-5      0x820A0xxx   voice format configuration (6 variants)
  VP_CmdBuild_VoiceMixCoeff       0x820Axxxx   envelope/mix coefficients
  VP_CmdBuild_VoicePanUpdate      0x820Axxxx   pan position
  VP_CmdBuild_VoiceFreqShift      0x820Axxxx   pitch / frequency shift
  VP_SetVoiceSampleBuffer         0x820B0270   set input_buffer_0/1_ptr + valid bits
  VP_SetVoiceSampleData           0x820B0518   set format type byte (+0xb in voice node)
  XMA_CreateVoicePool             0x82133730   allocate XmaVoiceCtx array + PCM bufs
  XMA_CreateContextsForPool       0x82134408   XMACreateContext per voice, assign slots
  XMA_RenderFrameUpdate           0x82137E48   per-frame decode+reset pump
  XMA_WriteVoiceRegistersAndReset 0x82134558   wait idle → MMIO reset → SW state reset
  XMA_ResetVoiceRegisterBlock     0x82133F78   per-voice SW state reset

GP layer:
  GP_CmdBuild_ReverbEffectRoute   0x820A3690
  GP_CmdBuild_ChorusFlangeRoute   0x820A5998
  GP_CmdBuild_EffectIORouting     0x820B20A0   37-way routing switch
  GP_UpdateDirtyEffectChain       0x820B2758
  GP_OptimizeEffectBusSlotAlloc   (unnamed)

EP layer:
  EP_XAudioRenderDriverCallback   0x8212FD68
  EP_XAudioRenderDriverFramePump  0x8212FBD0
  EP_CmdBuild_MixerMatrix7x2      0x820A5F78
  EP_SetSampleRateConverter       0x8212F8C8

APU core:
  APU_AdvanceCmdWritePtr          0x8205FE68   ring packet advance (stride 0x6C)
  APU_AllocCmdPacket              0x820BE508   packet allocator (NtAllocateVirtualMemory)

FSDX IRP layer:
  FSDX_DispatchIRP                0x82031DF8   7-volume state machine
  FSDX_IRP_Create                 0x8203xxxx
  FSDX_IRP_DeviceControl_*        0x8203xxxx   various IRP codes
```
