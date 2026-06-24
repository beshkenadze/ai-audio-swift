# Sortformer Bounded Long-Form Streaming — Design

**Status:** validated (brainstorming + Codex design review), ready for implementation
**Date:** 2026-06-24
**Component:** `Sources/MLXAudioVAD/Models/Sortformer/`

## Problem

`SortformerModel.generateStream(audio: MLXArray, …)` looks like a streaming API but is **not memory-bounded** for long files. On a ~50-min 16 kHz mono FLAC (model `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`, `use_aosc=true`) it reached only 64 chunks / 63.5 % after 839.8 s; per-chunk wall time grew from ~2 s to ~55–60 s; peak memory ~28 GB. MLX/Metal is hard-capped at 18 GB on this machine; exceeding ~25 GB can OOM-reboot it.

## Root cause — two independent defects

### D1. Unbounded FIFO when the chunk is larger than `spkcacheUpdatePeriod`

`maybeCompressState` ([Sortformer.swift:1018](../../Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift)) pops at most `spkcacheUpdatePeriod` (=188) frames from the FIFO **per call**, and it is called **once per chunk**:

```
popLen = state.fifoLen - fifoMax
if useAosc { popLen = min(popLen, spkcacheUpdatePeriod) }   // <= 188
```

If a chunk emits more than 188 diar (embedding) frames, the FIFO grows by `chunkEmbLen − 188` *every* chunk, unbounded. `streamingStep` then runs the Conformer + Transformer encoders over `[spkcache ++ fifo ++ leftCtx ++ chunk ++ rightCtx]`, so attention cost grows ~quadratically with the FIFO. **This is the dominant cause of the 2 s → 55 s/chunk blowup.**

- `chunkEmbLen > 188` ⟺ `chunkDuration > ~15 s` (188 emb frames ÷ 12.5 emb-frames/s = 15.04 s).
- Corroborated by project memory: *"Large `--chunk-duration` (≥20 s) … OOM."* The downstream Tish bench ran ~30 s chunks (64 chunks ≈ 63.5 % of 50 min ⇒ ~30 s/chunk ⇒ ~375 emb frames/chunk).
- The default `chunkDuration = 5.0 s` (≈62 emb frames) does **not** trigger D1 — which is why short-clip tests never exposed it.

### D2. Whole-file precompute proportional to total duration

`generateStream` computes `extractMelFeatures` over the **entire** waveform ([Sortformer.swift:869](../../Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift)) and, for the v2.1/AOSC path, a full-file `fcEncoder.preEncode(features, length: totalMelFrames)` → `allPreEmbs` ([Sortformer.swift:894](../../Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift)). The latter materialises huge Conv2d intermediates. `allPreEmbs` exists **only** to supply right-context embeddings — and `chunkRightContext = 1`, i.e. a **single** embedding frame (8 mel frames = 1280 samples ≈ 0.08 s) per chunk. Pre-encoding the whole file to slice a 1-frame lookahead is the waste.

Both defects must be fixed; D1 is necessary even before any bounded-source work.

## Constants (v2.1 / 4-spk config)

| Symbol | Value | Meaning |
|---|---|---|
| `sr` | 16000 | sample rate |
| `hop` | 160 | STFT hop |
| `nFft` | 512 | FFT size (STFT center-pads `nFft/2 = 256` both sides, `.constant`) |
| `winLength` | 400 | Hann window |
| `nMels` | 80 | mel bins |
| `subsamplingFactor` | 8 | ConvSubsampling total stride |
| conv | k=3, s=2, 3 stages | receptive field **15 mel frames (radius 7)** |
| `chunkLen` | 188 | emb frames per reference step (≈15.04 s) |
| `spkcacheLen` / `spkcacheUpdatePeriod` | 188 | spkcache cap / pop period |
| `chunkLeftContext` / `chunkRightContext` | 1 / 1 | emb frames; left from FIFO tail, right = lookahead |
| diar `frameDuration` | `hop·8/sr` = 0.08 s | per emb frame |
| mel frames/s | `sr/hop` = 100 | |
| emb frames/s | 12.5 | |

## Design — three clearly-separated inference paths

After this work the model exposes three paths with documented contracts:

1. **`generate(audio:)`** — offline, single forward over the whole file. Lowest latency for short clips; OOMs on long audio (kept as-is, documented as short-only).
2. **`generateStream(audio:)`** — *precompute* streaming: still takes the full `MLXArray`, still does whole-file mel + (AOSC) full-file `preEncode`. Kept for backward compatibility and short/medium clips; documented as **not memory-bounded**. Benefits from the D1 fix.
3. **`generateStreamBounded(audioSource:)`** — *new*, the memory-bounded long-form path. Consumes raw PCM incrementally; no whole-file tensor ever exists.

### D1 fix (shared, benefits all paths)

Make compression a true cap: loop the pop/compress until `fifoLen ≤ fifoMax`. Each iteration pops ≤ `spkcacheUpdatePeriod` and folds into the spkcache (preserving AOSC per-period semantics, so a big chunk is equivalent to several reference steps). Pure refactor of `maybeCompressState`; existing output is unchanged whenever the FIFO was already within bounds.

### `streamingStep` refactor (DRY)

Extract everything **after** `preEncode` ([Sortformer.swift:688–743](../../Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift)) into:

```swift
func streamingStepFromEmbeddings(
    chunkEmbs: MLXArray,            // (1, chunkEmbLen, embDim) — CHUNK FRAMES ONLY
    chunkDiarLen: Int,
    state: StreamingState,
    rightContextEmbs: MLXArray?     // (1, rc, embDim) or nil
) -> (MLXArray, StreamingState)
```

`streamingStep` becomes `preEncode(chunk)` → `streamingStepFromEmbeddings(...)` (behaviour identical; `feed`/`generateStream` untouched). The bounded path calls `streamingStepFromEmbeddings` directly with embeddings it already computed. **Invariant:** `chunkEmbs` must contain chunk frames only — never left-halo or right-context frames — because `chunkStart = spkcacheLen + fifoLen + leftCtxLen` ([Sortformer.swift:727](../../Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift)) depends on it.

### The windowing recipe (heart of the bounded path)

Process the stream in fixed steps of `C = chunkLen·8 = 1504` **new** mel frames (≈15 s — the reference step size; keeps FIFO net-zero growth under the D1 cap and matches NeMo's one-spkcache-update-per-step semantics).

To compute mel frames for a step **exactly equal to the full-file mel** (no boundary drift), exploit two facts:

- STFT center-pads each window with `nFft/2` zeros, so a sub-window starting at original sample `A` has **frame `j` centered at original sample `A + j·hop`**.
- Therefore, if `A = a·hop` is **hop-aligned**, sub-window frame `j` ≡ full-file frame `a + j` exactly. Frames with `j ≥ ceil(nFft/2 / hop) = 2` (and symmetric on the right) are free of zero-pad contamination.

**Halos** (both multiples of `subsamplingFactor = 8` so the embedding grid stays aligned):
- left halo `H_L = 16` mel frames (= 2 emb frames; covers conv radius 7 and STFT margin 2)
- right halo / lookahead `H_R = 16` mel frames (= 2 emb frames; covers `rc·8 = 8` plus conv radius 7)

**Per step `i`** (let `g0` = global mel index of the first new frame):
1. Window mel range `[g0 − H_L, g0 + C + H_R)` (clamp left at 0 for the first step; clamp/​shrink right at EOF).
2. Raw samples needed: `[(g0 − H_L)·hop − nFft/2, (g0 + C + H_R)·hop + nFft/2)` — pulled from the accumulator (absolute-sample addressed). Window start is hop-aligned by construction.
3. `extractMelFeatures(window, normalize: nil, padTo: 0)` → slice the aligned interior so the result is exactly global frames `[g0 − H_L, g0 + C + H_R)`.
4. `preEncode(melWindow)` once → emb frames. Discard the first `H_L/8 = 2` (left-halo) emb frames. The next `chunkLen` emb frames are **chunk embeddings**; the following `rc` emb frames are **right context**; discard the rest.
5. `streamingStepFromEmbeddings(chunkEmbs, chunkLen, state, rightContextEmbs: rcEmbs)`.
6. `predsToSegments`, offset by a global diar-frame counter; `yield`.
7. `maybeCompressState` (now a true cap).
8. Advance `g0 += C`; drop consumed samples from the accumulator, retaining only the tail needed for the next step's left halo + STFT margin.

**First step:** window starts at sample 0; STFT/Conv2d zero-pad the left naturally (matches `generate`/`generateStream` behaviour). **Do not** prepend zero *samples* to fake left context — silence samples become non-zero log-mel values, not feature-zero padding.
**EOF:** `H_R` shrinks to what remains; `rc = min(rc, availableFutureEmbFrames)`; the final partial chunk processes whatever new frames remain. Right zero-padding appears only here.

### Memory bound

Every per-step tensor is sized to `C·hop + ~0.5 s` of samples and `C + 32` mel frames — **independent of total duration**. `spkcache`/`fifo` are capped (D1). Accumulator retains ≤ one step + halo of PCM. Result: flat memory, flat per-chunk latency.

### `generateStreamBounded` API

```swift
public func generateStreamBounded(
    audioSource: @escaping () async throws -> [Float]?,  // 16 kHz mono PCM; nil = EOF
    sampleRate: Int = 16000,
    chunkDuration: Float = 15.0,        // default = chunkLen (reference step); any value safe under D1
    threshold: Float = 0.5,
    minDuration: Float = 0.0,
    mergeGap: Float = 0.0,
    spkcacheMax: Int = 188,
    fifoMax: Int = 188,
    verbose: Bool = false
) -> AsyncThrowingStream<DiarizationOutput, Error>
```

- **Pull closure returning `[Float]?`** (CPU PCM), not `MLXArray` — Sendable-clean; the `MLXArray` is built inside the producing task (Codex P2). `nil` signals EOF. Closure chunk size is arbitrary; the internal accumulator re-buffers to step size.
- `continuation.onTermination` cancels/stops the producer (the current `generateStream` lacks this).
- Diarization core stays format-agnostic; resampling/downmix is the caller's job.

### CLI tool `mlx-audio-swift-diar` (benchmark host)

New executable target under `Sources/Tools/mlx-audio-swift-diar`, mirroring `mlx-audio-swift-lid`'s `@main`/manual-arg pattern:
- Sets `GPU.set(memoryLimit: 18 * 1024 * 1024 * 1024, relaxed: false)`.
- Windowed reader: `AVAudioFile` + persistent `AVAudioConverter` (48 kHz stereo → 16 kHz mono), reading native windows and pulling resampled mono `[Float]` blocks → the `audioSource` closure. Bounded decode/resample memory.
- Streams `generateStreamBounded`, writes RTTM, and reports: total wall time, per-chunk latency series (+ min/median/max/trend), peak process RSS (`task_info`/`mach_task_basic_info`), MLX peak memory (`GPU.snapshot()` / peak API — verify exact symbol at impl time), segment count, RTF.
- Flags: `--input <path>`, `--repo`/default model id, `--chunk-duration`, `--threshold`, `--rttm-out`, `--verbose`.

## Parity / correctness strategy (baseline = NeMo/CUDA v2.1, not current `generateStream`)

- **Self-consistency gate (CI-able, no GPU truth needed):** on a short clip (≤ a few min) where both paths fit in memory, the bounded path's RTTM must match `generateStream`'s within a tiny DER tolerance. The windowing is *more* faithful to the reference than the full-file shortcut, so this is a regression guard, not a correctness proof.
- **Reference DER (manual, user-run):** the user supplies the long file (e.g. `…/Tish/Recordings/2B165E75-…/mix_stereo.flac`, 67.6 min, 48 kHz stereo); DER vs NeMo/CUDA v2.1 reference RTTM computed externally (NeMo env on `pc.lan`).

## Error handling

- `audioSource` throws → propagated via `AsyncThrowingStream` finish(throwing:); producer stopped via `onTermination`.
- Empty/zero-length source → finish with no segments.
- Source shorter than one step → single partial step.
- Cancellation (`Task.checkCancellation`) honoured each step.

## Testing

- **Unit (TDD, no model weights):** D1 cap loop (FIFO bounded after a large synthetic chunk); windowing frame-accounting math (hop-aligned mapping, no ±1 drift, halo discard counts); accumulator re-buffering; EOF/first-step edge cases.
- **Integration (needs weights + Metal, runs from `.build`):** bounded vs `generateStream` self-consistency on a short clip; bounded path runs end-to-end on a long clip with flat per-chunk latency and bounded peak memory.
- Run integration via the built executable (per project memory: `swift test` can't load the metallib; `.build` CLI binaries do run Metal).

## Risks / open questions

- Exact MLX-Swift peak-memory symbol (`GPU.snapshot()` vs `Memory.peakMemory`) — verify at impl.
- `AVAudioConverter` resampling state continuity across windowed pulls — the converter is stateful; feed via its input block, never recreate per window.
- `mix_stereo` channel semantics (mic vs playback per channel) — downmix to mono (mean) to capture both speakers.

## Non-goals (from task)

No fallback to another diarization engine; no per-chunk independent diarization (state continuity preserved); no batching of independent chunks.
