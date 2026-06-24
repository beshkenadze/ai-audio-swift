# mlx-audio-swift-diar

Memory-bounded long-form streaming speaker diarization benchmark host for the Sortformer
model (`SortformerModel.generateStreamBounded`).

`SortformerModel` exposes three inference paths (see the doc comments on the methods in
`Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift`):

- `generate(audio:)` — offline single forward pass; lowest latency for short clips; OOMs on long audio.
- `generateStream(audio:)` — precompute streaming; takes the whole `MLXArray`, not memory-bounded.
- `generateStreamBounded(audioSource:)` — the memory-bounded long-form path this tool drives.

This tool decodes and resamples the input **incrementally** — one window at a time via
`AVAudioFile` + a persistent `AVAudioConverter` (downmix + resample to 16 kHz mono Float32) — and
feeds raw PCM blocks into the bounded streaming path through a pull closure. Peak memory and
per-chunk latency stay flat regardless of file duration.

The process caps MLX/Metal memory at 18 GiB as its first action.

## Usage

```bash
swift build --product mlx-audio-swift-diar

.build/debug/mlx-audio-swift-diar --input recording.flac --rttm-out out.rttm --verbose

# Self-consistency regression guard (bounded vs precompute) on a short clip:
.build/debug/mlx-audio-swift-diar --self-check --input clip.flac --max-seconds 150 --chunk-duration 15 -v
```

> Run the `.build` binary directly — the SwiftPM CLI target ships the Metal `default.metallib`
> next to the executable, so no Xcode/DerivedData is required.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `-i <path>` | (required) | Input audio file, any format AVFoundation can read |
| `--repo <id>` | `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | Hugging Face repo id |
| `--chunk-duration <sec>` | `15` | Streaming step size in seconds |
| `--threshold <f>` | `0.5` | Speaker-activity threshold in `[0, 1]` |
| `--rttm-out <path>` | — | Optional RTTM output path |
| `--max-seconds <n>` | — | Stop feeding after N seconds of audio (quick smokes) |
| `--self-check` | off | Regression guard: load a short clip fully and compare bounded vs precompute (`generateStreamBounded` vs `generateStream`) for frame agreement. Pair with `--max-seconds`. |
| `--verbose`, `-v` | off | Per-chunk progress |

## Metrics reported

- Audio duration, total wall time, RTF (`wall / audio`)
- Chunks processed; per-chunk latency: min / median / max and first-vs-last trend
- Total segment count, distinct speaker count
- Peak process RSS (`task_info` / `MACH_TASK_BASIC_INFO`)
- MLX peak memory (`MLX.Memory.peakMemory`)

On a ~68-minute recording the bounded path runs at flat memory (well under 1 GB MLX peak, not
scaling with duration) and flat per-chunk latency, and `--self-check` confirms it is bit-faithful
to the precompute path.
