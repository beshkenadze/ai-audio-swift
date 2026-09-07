# VibeVoice ASR × omni-bench

Two independent omni-bench hosts for VibeVoice-ASR-Streaming, so the Swift MLX
port can be measured against the PyTorch reference:

| arm | where | code | backend |
|---|---|---|---|
| A — reference | CUDA box (RTX 4090) | `tools/omni_bench_host_vibevoice/vibevoice_host.py` | `pytorch-cuda` |
| B — port | Mac (Apple silicon) | `tools/omni-bench-host/` (SwiftPM) | `mlx` |

Neither arm calls the other. Each runs inference locally, so its Result's
`hardware` axis describes the machine that actually ran the model and the
measured latency is inference rather than transport. They are compared with
`omni-bench parity`.

Both implement both `audio_transcription.v1` seams — `Transcriber` (batch) and
`StreamingTranscriber` (paced streaming).

## What is pinned so the comparison means something

A parity delta is only attributable to the model if everything around the model
is identical. Three things are specified explicitly rather than left to
whichever library was convenient on each platform:

**Resampling.** Prepared omni-bench audio is canonically 16 kHz mono; VibeVoice
consumes 24 kHz. librosa on one side and `AVAudioConverter` on the other would
feed the two arms different waveforms. Both implement the same
bandlimited-sinc formula (Blackman-windowed, 32 zero crossings), and the
agreement is *proved*, not assumed:

```bash
# regenerate the golden vector the Swift test checks against
cd tools/omni_bench_host_vibevoice
PYTHONPATH=<omni-bench>/python/src python3 dump_resampler_golden.py \
  --out ../omni-bench-host/Tests/OmniBenchVibeVoiceHostTests/Fixtures/vibevoice_resampler_golden.json

# each arm's streaming resampler must equal its own whole-signal resampler
PYTHONPATH=<omni-bench>/python/src python3 test_resampler_equivalence.py
cd ../omni-bench-host && swift test
```

The streaming path uses a stateful resampler that keeps the kernel's context
and stays on the global output grid. Resampling each model window on its own —
the obvious implementation — restarts the grid phase per window and treats both
window edges as silence, which quietly desynchronizes streaming from batch; the
equivalence tests above exist to catch exactly that.

**Determinism.** The released acoustic tokenizer config has
`std_dist_type="gaussian"`, so the reference samples its VAE and its transcript
changes between runs. The Swift port takes the mean. The Python host forces the
same (`std_dist_type="none"`) and **asserts two `encode_speech` calls now agree
bit-exactly** before serving anything. Decoding is greedy on both arms.

**Hypothesis cleanup.** The streaming prompt asks for `speaker, content` keys,
so chunks arrive framed as `" \n Speaker 0:text"`. Both arms strip that framing
— it is protocol, not transcription, and would inflate WER against raw
orthographic references. Bracketed events like `[Laughter]` are left alone; the
Scoring Profile owns normalization.

## Running arm A (CUDA)

One-time setup on the box:

```bash
uv pip install --python /mnt/d/Projects/vibevoice-official/.venv-parity/bin/python \
  -e /mnt/d/Projects/omni-bench/python
scp tools/omni_bench_host_vibevoice/*.py <box>:/mnt/d/Projects/vibevoice-official/
```

The Registry must be the *same* on both machines or the prepared manifest is
rejected (`application.registry_ref_mismatch`). Sync it whenever omni-bench's
registry changes:

```bash
cd <omni-bench> && tar czf /tmp/ob-sync.tgz fixtures/consumer-bundle python/src schemas
scp /tmp/ob-sync.tgz <box>:/mnt/d/Projects/ && ssh <box> \
  'cd /mnt/d/Projects/omni-bench && tar xzf /mnt/d/Projects/ob-sync.tgz'
```

Then run — the host is started explicitly, there is no daemon and nothing is
left running afterwards:

```bash
cd /mnt/d/Projects/vibevoice-official
export OMNI_BENCH_REGISTRY_BUNDLE=/mnt/d/Projects/omni-bench/fixtures/consumer-bundle
export VIBEVOICE_MODEL_PATH=/mnt/d/Projects/vibevoice-test/streaming-1.5b
export PYTHONPATH=/mnt/d/Projects/vibevoice-official

./.venv-parity/bin/omni-bench run \
  --adapter vibevoice_host:make \
  --manifest <prepared>/asr.synthetic.en.v1/manifest.json \
  --measurement-profile audio_transcription.batch_single.v1 \
  --run-profile '{"delivery":"batch","chunk_ms":null,"warmup_samples":0,"concurrency":1,"family_parameters":{}}' \
  --implementation python \
  --hardware '{"soc":"host","accelerator":"NVIDIA RTX 4090","mem_gb":24}' \
  --os-identity '{"name":"Linux","version":"6.6"}' \
  --out vv-cuda.jsonl
```

Tunables (all optional): `VIBEVOICE_DTYPE` (default `bfloat16`),
`VIBEVOICE_CHUNK_DURATION` (2.0), `VIBEVOICE_TEXT_AUDIO_DELAY` (0.5),
`VIBEVOICE_MAX_NEW_TOKENS` (256), `VIBEVOICE_TEMPERATURE` (0.0),
`VIBEVOICE_MODEL_ID`, `VIBEVOICE_ARTIFACT_SHA256`.

## Running arm B (Swift / MLX)

The Swift host drives `AudioTranscriptionProducer` directly — the Python CLI
cannot import a Swift adapter. Identity is emitted by omni-bench's own resolver
so the Registry digests cannot drift out of sync:

```bash
cd <omni-bench>
PYTHONPATH=python/src uv run --project python python \
  <repo>/tools/omni_bench_host_vibevoice/emit_swift_identity.py \
  --manifest data/asr.synthetic.en.v1/manifest.json \
  --measurement-profile audio_transcription.batch_single.v1 \
  --run-profile '{"delivery":"batch","chunk_ms":null,"warmup_samples":0,"concurrency":1,"family_parameters":{}}' \
  --model '{"base_model_id":"microsoft/VibeVoice-ASR-Streaming-1.5B","artifact_sha256":null,"quantization":null}' \
  --backend '{"id":"mlx","version":"0.31.4"}' \
  --implementation swift \
  --out /tmp/vv-identity-swift.json

cd <repo>/tools/omni-bench-host
swift build -c release
# MLX ships its Metal kernels in a resource bundle; a nested SwiftPM package
# does not always get one copied next to the binary. See AGENTS.md.
cp -R ../../.build/arm64-apple-macosx/release/mlx-swift_Cmlx.bundle \
      .build/arm64-apple-macosx/release/

./.build/release/vibevoice-omni-bench-run \
  --manifest <omni-bench>/data/asr.synthetic.en.v1/manifest.json \
  --identity /tmp/vv-identity-swift.json \
  --model /Volumes/DATA/models/vibevoice-asr-streaming-1.5b \
  --out vv-swift.jsonl
```

## Scoring and comparing

Scoring is Python-only; hosts emit Evidence and never quality scores.

```bash
cd <omni-bench>
uv run --project python omni-bench score --manifest <manifest> \
  --artifact vv-cuda.jsonl  --out vv-cuda.result.json
uv run --project python omni-bench score --manifest <manifest> \
  --artifact vv-swift.jsonl --out vv-swift.result.json

uv run --project python omni-bench parity --manifest <manifest> \
  --result-a vv-cuda.result.json  --artifact-a vv-cuda.jsonl \
  --result-b vv-swift.result.json --artifact-b vv-swift.jsonl \
  --out vv-parity.json
```

### Results so far

**`asr.fleurs.en.quick.v1`, arm B (Swift/MLX, 64 real-speech samples)** — 64/64
OK, no sample errors:

| metric | value |
|---|---|
| `quality.wer_norm.v1` | 0.581 |
| `quality.wer_ortho.v1` | 0.703 |
| `quality.cer.v1` | 0.458 |
| `latency.request_completion_s.v1` p50 | 2.01 s |
| `throughput.audio_rtfx_wall.v1` | 4.77 |
| `resources.peak_process_rss_gb.v1` | 4.63 |

The absolute WER is high for a 1.5B streaming diarization model asked to
transcribe short read sentences in 2 s windows; it is reported here as a
baseline for the arm comparison, not as a quality claim.

The matching CUDA run is still outstanding, so **no cross-arm quality
comparison on real speech has been made yet.**

### Known limitations

**Do not read the synthetic-task parity as model parity.** On
`asr.synthetic.en.v1` both arms emit `[Silence]` for all three samples and
score WER = CER = 1.0, because the samples are synthetic tones with no speech
in them. The zero WER delta and `hypothesis_identity = 1.0` therefore only say
the two arms are *identically wrong*: they demonstrate that the harness,
preprocessing contract and artifact plumbing agree end to end, which is what
that Task is for (contract conformance), and nothing about recognition quality.
A real-speech Task is required for the latter, which is why
`asr.fleurs.en.quick.v1` is run above.

**Parity across operating systems returns `fail` by construction.** On
`asr.synthetic.en.v1` both quality gates pass —
`quality.wer_norm.v1` delta `0.0` (threshold `0.005`) and
`parity.hypothesis_identity.v1` `1.0` (both vacuously, see above) — yet the
overall verdict is `fail` with `reason.detail = identity_scope_mismatch`, and
it would stay `fail` on a real-speech Task for the same reason.

The cause is structural, not a defect in either host. These metrics declare
`allowed_difference_axes = [model, backend, implementation, hardware]`; `os` is
neither allowed to differ nor required to be equal, so any OS difference trips
the compatibility check. The MLX arm can only run on macOS and the CUDA arm only
on Linux, so that axis can never be equalized honestly.

Options, none of which this integration takes unilaterally because they are
Registry changes and therefore outside a host's remit:

1. Add `os` to `allowed_difference_axes` for the ASR quality/parity metrics.
2. Treat the cross-platform question as a leaderboard/compare question rather
   than a parity question, and reserve `parity` for same-platform regressions
   (e.g. Swift vs Swift across a code change), where it works today.

Until then, read `gate_evaluations` in the parity report — the substantive
comparison is computed and meaningful even when the verdict is `fail`.

**The Swift arm cannot run `streaming_single` yet.** Both hosts implement
`StreamingTranscriber`, but omni-bench's Swift side has no streaming audio
producer: `AudioTranscriptionProducer` validates
`delivery == "batch"` and `measurement_profile == audio_transcription.batch_single.v1`,
and only `TextGenerationProducer` handles streaming. Producing streaming
Evidence from Swift needs an `AudioTranscription` streaming producer in
omni-bench core (mirroring the Python one), which is framework work rather than
host work. The Python arm can run `streaming_single` today.
