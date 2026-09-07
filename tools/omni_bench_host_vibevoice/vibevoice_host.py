"""omni-bench host adapter for VibeVoice-ASR-Streaming (PyTorch / CUDA reference arm).

This is the *reference* arm. The other arm is the Swift MLX port in this same
repository (`Sources/OmniBenchVibeVoiceHost`). Both run inference locally on
their own hardware and are compared with `omni-bench parity`; neither calls the
other over a network, so each Result's `hardware` axis is honest and the
measured latency is inference, not transport.

Implements both `audio_transcription.v1` seams:

  * ``Transcriber``          -> ``audio_transcription.batch_single.v1``
  * ``StreamingTranscriber`` -> ``audio_transcription.streaming_single.v1``

Three things must match the Swift arm exactly, or parity would measure the
mismatch instead of the port. All three are specified here rather than
delegated to a library:

1. **Resampling.** Prepared omni-bench audio is canonically 16 kHz mono;
   VibeVoice consumes 24 kHz. Using librosa here and AVAudioConverter there
   would feed the two arms different waveforms. `resample_16k_to_24k` below is
   written from an explicit formula that the Swift side reimplements verbatim,
   and `tools/omni_bench_host_vibevoice/dump_resampler_golden.py` emits a
   golden vector that the Swift test checks against.

2. **Determinism.** The released acoustic tokenizer config has
   ``std_dist_type='gaussian'``, so the reference's `encode_speech` samples the
   VAE and its transcript changes run to run (measured earlier: enough to
   reorder the top-5). The Swift port takes the VAE mean. This host forces the
   same by setting ``std_dist_type='none'`` and asserts the switch took effect
   before serving anything.

3. **Hypothesis cleanup.** The model is prompted to emit
   ``speaker, content`` keys, so chunks arrive as ``" \\n Speaker 0:text"``.
   That prefix is protocol framing, not transcription, and would wreck WER
   against raw orthographic references. It is stripped. Everything else --
   punctuation, casing, bracketed events like ``[Laughter]`` -- is left for the
   Scoring Profile, which owns normalization.

Usage on the CUDA box:

    export VIBEVOICE_MODEL_PATH=/mnt/d/Projects/vibevoice-test/streaming-1.5b
    export OMNI_BENCH_REGISTRY_BUNDLE=/mnt/d/Projects/omni-bench/fixtures/consumer-bundle
    PYTHONPATH=/mnt/d/Projects/vibevoice-official:/mnt/d/Projects/mlx-audio-swift/tools/omni_bench_host_vibevoice \\
    omni-bench run --adapter vibevoice_host:make ...
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from omni_bench.core.adapter import (
    AudioInput,
    Capabilities,
    Transcript,
    TranscriptionError,
)
from omni_bench.core.streaming import AudioChunk

# --------------------------------------------------------------------------
# Shared preprocessing contract -- keep byte-for-byte in step with
# Sources/OmniBenchVibeVoiceHost/VibeVoiceResampler.swift
# --------------------------------------------------------------------------

MODEL_SAMPLE_RATE = 24_000

#: Zero crossings each side of the sinc kernel. Larger = sharper transition and
#: more work; 32 puts the reconstruction error far below the bf16 checkpoint's
#: own noise floor, so it cannot be what a parity delta is measuring.
RESAMPLE_ZEROS = 32


def _blackman(u: np.ndarray) -> np.ndarray:
    """Blackman window over u in [0, 1]. Written out rather than np.blackman so
    the Swift side can reproduce the same coefficients from the same formula."""
    return 0.42 - 0.5 * np.cos(2.0 * np.pi * u) + 0.08 * np.cos(4.0 * np.pi * u)


def _resample_core(
    buffer: np.ndarray,
    *,
    buffer_offset: int,
    out_indices: np.ndarray,
    step: float,
    input_available: int,
) -> np.ndarray:
    """Evaluate output samples `out_indices` on the *global* 24 kHz grid.

    Both the whole-signal and the incremental paths route through this, so
    they cannot drift apart. Indices are global: `buffer` holds input samples
    starting at `buffer_offset`, and any tap outside either the buffer or
    `[0, input_available)` contributes zero -- the same edge convention the
    whole-signal path has.
    """
    half = RESAMPLE_ZEROS
    t = out_indices.astype(np.float64) * step
    base = np.floor(t).astype(np.int64)

    offsets = np.arange(-half + 1, half + 1, dtype=np.int64)
    idx = base[:, None] + offsets[None, :]          # global input indices
    dist = t[:, None] - idx.astype(np.float64)

    kernel = np.sinc(dist) * _blackman((dist + half) / (2.0 * half))
    kernel[np.abs(dist) >= half] = 0.0

    local = idx - buffer_offset
    valid = (idx >= 0) & (idx < input_available) & (local >= 0) & (local < buffer.shape[0])
    gathered = np.take(buffer, np.clip(local, 0, max(buffer.shape[0] - 1, 0)), mode="clip")
    gathered = np.where(valid, gathered, 0.0)

    return np.asarray((gathered * kernel).sum(axis=1), dtype=np.float32)


def resample_16k_to_24k(samples: np.ndarray, *, source_rate: int) -> np.ndarray:
    """Bandlimited sinc interpolation from `source_rate` to 24 kHz.

    16 kHz -> 24 kHz is upsampling, so no anti-alias narrowing is needed: the
    kernel keeps the full input band (cutoff at input Nyquist, i.e. 0.5
    cycles/input-sample) and simply interpolates. Output sample ``n`` reads
    input position ``t = n * source_rate / 24000`` and convolves the
    ``RESAMPLE_ZEROS`` input samples either side with ``sinc(t-k) *
    blackman(...)``.
    """
    if source_rate == MODEL_SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    if source_rate > MODEL_SAMPLE_RATE:
        raise TranscriptionError(
            f"downsampling {source_rate} -> {MODEL_SAMPLE_RATE} is not part of the "
            "shared contract; prepared omni-bench audio is 16 kHz")

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    n_in = x.shape[0]
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)

    step = source_rate / MODEL_SAMPLE_RATE  # input samples per output sample
    n_out = int(math.floor(n_in / step))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    return _resample_core(
        x, buffer_offset=0, out_indices=np.arange(n_out, dtype=np.int64),
        step=step, input_available=n_in)


class ResamplerStream:
    """Incremental resampler whose concatenated output is identical to
    resampling the whole signal at once.

    Resampling each model window independently -- which is what an obvious
    implementation does -- restarts the output grid's phase at every window and
    treats both window edges as silence, so the streaming seam would feed the
    model a different waveform than the batch seam, and this arm would differ
    from the Swift arm for a reason that has nothing to do with either model.
    This keeps the `RESAMPLE_ZEROS` samples of context the kernel needs, stays
    on the global output grid, and emits a sample only once every tap it
    depends on has arrived. Mirrors `VibeVoiceResampler.Stream` in Swift.
    """

    def __init__(self, source_rate: int) -> None:
        if source_rate > MODEL_SAMPLE_RATE:
            raise TranscriptionError(
                f"downsampling {source_rate} -> {MODEL_SAMPLE_RATE} is not part of "
                "the shared contract; prepared omni-bench audio is 16 kHz")
        self._source_rate = source_rate
        self._step = source_rate / MODEL_SAMPLE_RATE
        self._pending = np.zeros(0, dtype=np.float64)
        self._dropped_input = 0
        self._next_output = 0

    @property
    def is_passthrough(self) -> bool:
        return self._source_rate == MODEL_SAMPLE_RATE

    def push(self, samples: np.ndarray) -> np.ndarray:
        if self.is_passthrough:
            return np.asarray(samples, dtype=np.float32)
        self._pending = np.concatenate(
            [self._pending, np.asarray(samples, dtype=np.float64).reshape(-1)])
        return self._drain(final=False)

    def finish(self) -> np.ndarray:
        if self.is_passthrough:
            return np.zeros(0, dtype=np.float32)
        out = self._drain(final=True)
        self._pending = np.zeros(0, dtype=np.float64)
        return out

    def _drain(self, *, final: bool) -> np.ndarray:
        half = RESAMPLE_ZEROS
        available = self._dropped_input + self._pending.shape[0]

        if final:
            limit = int(math.floor(available / self._step))
        else:
            # Produce only outputs whose rightmost tap (base + half) has
            # arrived; base grows monotonically with the output index.
            limit = self._next_output
            while True:
                base = int(math.floor(limit * self._step))
                if base + half >= available:
                    break
                limit += 1

        if limit <= self._next_output:
            return np.zeros(0, dtype=np.float32)

        out_indices = np.arange(self._next_output, limit, dtype=np.int64)
        out = _resample_core(
            self._pending, buffer_offset=self._dropped_input,
            out_indices=out_indices, step=self._step, input_available=available)
        self._next_output = limit

        if not final:
            # Everything below the leftmost tap of the next output is dead.
            next_base = int(math.floor(self._next_output * self._step))
            keep_from = max(0, next_base - half + 1)
            drop = min(max(0, keep_from - self._dropped_input), self._pending.shape[0])
            if drop > 0:
                self._pending = self._pending[drop:]
                self._dropped_input += drop
        return out


#: ``" \n Speaker 0:text"`` -> ``"text"``. Anchored at a line start so a literal
#: mention of a speaker inside the transcription is not eaten.
_SPEAKER_LABEL = re.compile(r"(?m)^\s*Speaker\s+\d+\s*:\s*")


def strip_speaker_labels(text: str) -> str:
    """Remove the diarization scaffolding the streaming prompt asks for.

    The model is instructed to transcribe "with these keys: speaker, content",
    so every chunk is framed as ``Speaker N:``. That framing is protocol, not
    hypothesis -- leaving it in would inflate WER against raw references for a
    reason that has nothing to do with recognition quality. Bracketed events
    such as ``[Laughter]`` are the model's transcription choice and are left
    alone; the Scoring Profile's normalizer decides what to do with them.
    """
    return " ".join(_SPEAKER_LABEL.sub(" ", text).split())


# --------------------------------------------------------------------------
# Windowing -- mirrors VibeVoiceWindowBuffer on the Swift side and
# streaming_generate's split_then_encode branch in the official reference.
# --------------------------------------------------------------------------

TOKENIZER_HOP = 3200  # samples per acoustic latent frame at 24 kHz


@dataclass(frozen=True)
class WindowPlan:
    chunk_samples: int
    lookahead_samples: int

    @property
    def window_samples(self) -> int:
        return self.chunk_samples + self.lookahead_samples


def plan_windows(chunk_duration: float, text_audio_delay: float) -> WindowPlan:
    """Chunk/lookahead sizes in samples at 24 kHz.

    The lookahead is snapped down to a whole tokenizer frame exactly as the
    reference does (``round(delay / frame_dur) * frame_dur``): the encoder only
    produces latents on frame boundaries, so a partial frame of lookahead would
    silently round somewhere else and desync the two arms.
    """
    frame_dur = TOKENIZER_HOP / MODEL_SAMPLE_RATE
    lookahead_sec = round(text_audio_delay / frame_dur) * frame_dur
    return WindowPlan(
        chunk_samples=int(chunk_duration * MODEL_SAMPLE_RATE),
        lookahead_samples=int(lookahead_sec * MODEL_SAMPLE_RATE))


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------


class VibeVoiceHost:
    """Both ASR seams over one loaded reference model."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        chunk_duration: float,
        text_audio_delay: float,
        max_new_tokens_per_chunk: int,
        temperature: float,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._plan = plan_windows(chunk_duration, text_audio_delay)
        self._max_new_tokens = max_new_tokens_per_chunk
        self._temperature = temperature

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_timestamps=False, supports_streaming=True, max_concurrency=1)

    # ---- batch ----------------------------------------------------------

    def transcribe(self, audio: AudioInput, *, language: str, task: dict) -> Transcript:
        del language, task  # the model is multilingual and takes no language hint
        try:
            waveform = resample_16k_to_24k(
                audio.samples(), source_rate=audio.sample_rate_hz)
        except Exception as exc:  # noqa: BLE001 - one bad sample is not fatal
            raise TranscriptionError(f"{audio.path}: {exc}") from exc
        if waveform.size == 0:
            raise TranscriptionError(f"{audio.path}: empty after resampling")

        texts = [text for _, text in self._run_windows(waveform)]
        return Transcript(text=strip_speaker_labels(" ".join(texts)))

    # ---- streaming ------------------------------------------------------

    def transcribe_stream(
        self,
        stream: Iterator[AudioChunk],
        *,
        language: str,
        task: dict,
        emit: Callable[[str], None],
    ) -> Transcript:
        del language, task
        state = self._model.init_streaming_state(self._tokenizer)
        texts: list[str] = []
        resampler: ResamplerStream | None = None
        source_rate: int | None = None
        # Buffer in the *model's* 24 kHz domain and window there, exactly as
        # the batch path does. Resampling is continuous underneath, so the
        # samples this buffer sees are the same ones a whole-signal resample
        # would have produced.
        pending = np.zeros(0, dtype=np.float32)

        def consume(*, final: bool) -> None:
            nonlocal pending
            window = self._plan.window_samples
            stride = self._plan.chunk_samples
            while pending.size >= window or (final and pending.size > 0):
                segment = pending[:window]
                texts.append(self._step(self._pad(segment), state))
                partial = strip_speaker_labels(" ".join(texts))
                if partial:
                    emit(partial)
                pending = pending[stride:]

        # The producer paces chunks at chunk_ms (typically 100 ms) while the
        # model wants ~2 s windows. Every chunk must be pulled: returning
        # early is a per-sample error, so this loop always drains.
        for chunk in stream:
            if source_rate is None:
                source_rate = chunk.sample_rate_hz
                resampler = ResamplerStream(source_rate)
            elif chunk.sample_rate_hz != source_rate:
                raise TranscriptionError(
                    f"sample rate changed mid-stream: {source_rate} -> {chunk.sample_rate_hz}")
            assert resampler is not None
            produced = resampler.push(np.asarray(chunk.samples, dtype=np.float32))
            if produced.size:
                pending = np.concatenate([pending, produced])
                consume(final=False)

        if resampler is not None:
            tail = resampler.finish()
            if tail.size:
                pending = np.concatenate([pending, tail])
        # The reference advances one stride per window until the cursor passes
        # the end, so a leftover longer than one stride still yields more than
        # one window; stopping at a single padded window would drop audio the
        # batch arm transcribes.
        consume(final=True)

        return Transcript(text=strip_speaker_labels(" ".join(texts)))

    # ---- internals ------------------------------------------------------

    def _pad(self, window: np.ndarray) -> np.ndarray:
        target = self._plan.window_samples
        if window.size >= target:
            return window[:target]
        return np.concatenate([window, np.zeros(target - window.size, dtype=np.float32)])

    def _run_windows(self, waveform: np.ndarray):
        """Offline equivalent of the streaming loop, so batch and streaming
        cannot drift apart in how they cut windows."""
        state = self._model.init_streaming_state(self._tokenizer)
        start = 0
        index = 0
        total = waveform.size
        while start < total:
            end = min(start + self._plan.window_samples, total)
            if end > start:
                yield index, self._step(self._pad(waveform[start:end]), state)
                index += 1
            start += self._plan.chunk_samples

    def warmup(self) -> None:
        """Run one silent window so the first scored sample does not pay kernel
        autotune and allocator growth."""
        self._step(
            np.zeros(self._plan.window_samples, dtype=np.float32),
            self._model.init_streaming_state(self._tokenizer))

    def _step(self, window: np.ndarray, state: dict) -> str:
        import torch

        device = next(self._model.parameters()).device
        tensor = torch.from_numpy(np.ascontiguousarray(window))[None, :].to(device)
        features = self._model.encode_speech(tensor)
        # `streaming_generate_step` mutates `state` in place (it reassigns
        # past_key_values) and returns that same dict, so the running KV cache
        # carries across chunks without any copying here.
        text, _ = self._model.streaming_generate_step(
            features, state, self._tokenizer,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature)
        return text


def _force_mean_latents(model) -> None:
    """Disable the acoustic VAE's sampling and prove it took effect.

    Without this the reference is stochastic and any parity delta against the
    Swift arm is dominated by RNG rather than by the port.
    """
    import torch

    tokenizer_module = model.model.acoustic_tokenizer
    tokenizer_module.std_dist_type = "none"

    probe = torch.zeros(
        1, plan_windows(2.0, 0.5).window_samples,
        device=next(model.parameters()).device)
    with torch.no_grad():
        a = model.encode_speech(probe)
        b = model.encode_speech(probe)
    deviation = (a - b).abs().max().item()
    if deviation != 0.0:
        raise RuntimeError(
            "acoustic tokenizer still samples after forcing std_dist_type='none' "
            f"(max|d| = {deviation:.3e}); the CUDA arm would be non-deterministic")


def make():
    """omni-bench adapter factory: ``(adapter, model_identity, backend)``."""
    import torch

    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.modular.modular_vibevoice_text_tokenizer import (
        VibeVoiceASRTextTokenizerFast,
    )

    model_path = os.environ.get("VIBEVOICE_MODEL_PATH")
    if not model_path:
        raise RuntimeError("VIBEVOICE_MODEL_PATH is required")

    dtype_name = os.environ.get("VIBEVOICE_DTYPE", "bfloat16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[dtype_name]

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map=None)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(model_path)
    if tokenizer.text_chunk_end_id is None:
        raise RuntimeError(
            f"{model_path}: tokenizer has no <|text_chunk_end|>; chunks could never terminate")

    _force_mean_latents(model)

    adapter = VibeVoiceHost(
        model, tokenizer,
        chunk_duration=float(os.environ.get("VIBEVOICE_CHUNK_DURATION", "2.0")),
        text_audio_delay=float(os.environ.get("VIBEVOICE_TEXT_AUDIO_DELAY", "0.5")),
        max_new_tokens_per_chunk=int(os.environ.get("VIBEVOICE_MAX_NEW_TOKENS", "256")),
        temperature=float(os.environ.get("VIBEVOICE_TEMPERATURE", "0.0")))

    adapter.warmup()

    model_identity = {
        "base_model_id": os.environ.get(
            "VIBEVOICE_MODEL_ID", "microsoft/VibeVoice-ASR-Streaming-1.5B"),
        "artifact_sha256": os.environ.get("VIBEVOICE_ARTIFACT_SHA256") or None,
        "quantization": None,
    }
    backend = {"id": "pytorch-cuda", "version": torch.__version__}
    return adapter, model_identity, backend
