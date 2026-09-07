#!/usr/bin/env python3
"""End-to-end streaming transcript reference for VibeVoice-ASR-Streaming.

Runs the official PyTorch `streaming_generate` over multi-chunk audio with
every stochastic source removed, so any Swift/CUDA divergence is a real bug
rather than RNG:

  * greedy decode (temperature=0 -> argmax)
  * the acoustic VAE's `sample()` forced to return its mean

The mean switch is applied at the source rather than by patching call sites:
`VibeVoiceGaussianDistribution.sample(dist_type)` returns `self.mean`
untouched for any dist_type that is neither 'fix' nor 'gaussian', so setting
`acoustic_tokenizer.std_dist_type = 'none'` makes `encode_speech` -- which
reads that attribute -- deterministic without touching the model graph. That
is exactly what the Swift port does (`VibeVoiceAcousticTokenizerEncoder`).

Emits two files:
  <out>.safetensors  input_audio + per-chunk first-step logits (for bisecting)
  <out>.json         per-chunk token ids and text, plus the run's parameters

Usage (on the CUDA box):
  PYTHONPATH=/mnt/d/Projects/vibevoice-official \
  .venv-parity/bin/python vibevoice_parity_stream.py \
      --model /mnt/d/Projects/vibevoice-test/streaming-1.5b \
      --audio /path/to/audio.wav --seconds 7.3 \
      --out /mnt/d/Projects/vibevoice-test/parity_stream
"""

import argparse
import json
import sys

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True, help="path prefix (no extension)")
    parser.add_argument("--seconds", type=float, default=7.3,
                        help="clip length; pick a non-multiple of chunk_duration so the "
                             "final window is partial and must be zero-padded")
    parser.add_argument("--offset", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-duration", type=float, default=2.0)
    parser.add_argument("--text-audio-delay", type=float, default=0.5)
    parser.add_argument("--max-new-tokens-per-chunk", type=int, default=256)
    args = parser.parse_args()

    import librosa
    from safetensors.torch import save_file
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)  # cuDNN conv autotune is fine; no RNG involved
    torch.backends.cudnn.benchmark = False

    wav, _ = librosa.load(
        args.audio, sr=args.sample_rate, mono=True,
        offset=args.offset, duration=args.seconds)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    audio = torch.from_numpy(wav)[None, :]
    print(f"slice: {wav.shape[0]} samples "
          f"({wav.shape[0] / args.sample_rate:.3f}s), rms={float(np.sqrt((wav**2).mean())):.4f}",
          flush=True)

    print("loading model (fp32)...", flush=True)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    audio = audio.to(device)

    # `streaming_generate` reads speech_start_id / speech_end_id /
    # text_chunk_end_id off the tokenizer, which plain AutoTokenizer does not
    # expose -- it needs the ASR wrapper that maps the repurposed Qwen2.5
    # tokens (<|object_ref_start|> etc.) onto those names.
    tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(args.model)
    print(f"tokenizer: {type(tokenizer).__name__} "
          f"sp_start={tokenizer.speech_start_id} sp_end={tokenizer.speech_end_id} "
          f"tce={tokenizer.text_chunk_end_id} eos={tokenizer.eos_token_id}", flush=True)
    assert tokenizer.text_chunk_end_id is not None, \
        "tokenizer lacks <|text_chunk_end|>; no chunk could ever terminate"

    # ---- kill the acoustic VAE noise -------------------------------
    before = model.model.acoustic_tokenizer.std_dist_type
    model.model.acoustic_tokenizer.std_dist_type = "none"
    print(f"acoustic std_dist_type: {before!r} -> "
          f"{model.model.acoustic_tokenizer.std_dist_type!r} (mean, deterministic)", flush=True)

    # Prove the switch took effect before trusting a single transcript: two
    # encode_speech calls must now agree exactly.
    with torch.no_grad():
        a = model.encode_speech(audio)
        b = model.encode_speech(audio)
    repeat_dev = (a - b).abs().max().item()
    print(f"encode_speech determinism check: max|d| between two calls = {repeat_dev:.3e}",
          flush=True)
    assert repeat_dev == 0.0, "acoustic sampling still stochastic -- mean switch did not apply"

    # ---- capture per-chunk logits and true generated ids ------------
    # `streaming_generate` yields text only. Re-encoding that text would be
    # circular -- two different token sequences can decode to the same
    # string, which is exactly the compensating error this comparison has to
    # catch. So hook lm_head and recover the ids the model actually emitted.
    #
    # Per chunk the reference performs, in order:
    #   A            forward of [sp_start, features, sp_end]
    #   B_1..B_m     one forward per accepted token
    #   C            forward of <|text_chunk_end|>
    # and greedy token j+1 is argmax of capture j. With len(captures)=m+2 for
    # a chunk, the accepted ids are the argmaxes of the first m captures --
    # the (m+1)-th argmax is the terminator that ended the chunk.
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output[:, -1, :].detach().float().cpu())

    handle = model.lm_head.register_forward_hook(hook)

    chunks = []
    bounds = []
    with torch.no_grad():
        # Capture 0 belongs to the prompt prefill forward that
        # `streaming_generate` runs before the first chunk, not to chunk 0.
        # (Its argmax is the model's guess with no audio seen yet.) Getting
        # this wrong prepends a phantom token, which is what the decode
        # self-check below exists to catch.
        cursor = 1
        for chunk_idx, total_chunks, chunk_text in model.streaming_generate(
            audio_tensor=audio,
            tokenizer=tokenizer,
            chunk_duration=args.chunk_duration,
            text_audio_delay=args.text_audio_delay,
            sample_rate=args.sample_rate,
            max_new_tokens_per_chunk=args.max_new_tokens_per_chunk,
            temperature=0.0,
            encode_mode="split_then_encode",
            pad_last_chunk=True,
        ):
            bounds.append((cursor, len(captured)))
            chunks.append({
                "index": chunk_idx,
                "total_chunks": total_chunks,
                "text": chunk_text,
            })
            cursor = len(captured)
            print(f"chunk {chunk_idx + 1}/{total_chunks}: {chunk_text!r}", flush=True)
    handle.remove()

    tensors = {"input_audio": audio.detach().cpu().float()}
    for i, (start, end) in enumerate(bounds):
        cs = captured[start:end]
        n_tokens = max(0, len(cs) - 2)
        token_ids = [int(cs[j].argmax(dim=-1).item()) for j in range(n_tokens)]
        chunks[i]["token_ids"] = token_ids
        chunks[i]["terminator_id"] = (
            int(cs[n_tokens].argmax(dim=-1).item()) if n_tokens < len(cs) else None)
        if cs:
            tensors[f"chunk_{i}_first_logits"] = cs[0].contiguous()

        # Self-check: if the recovered ids were mis-indexed they would not
        # decode back to the text the reference yielded.
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        for st in ["<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                   "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>"]:
            decoded = decoded.replace(st, "")
        assert decoded == chunks[i]["text"], (
            f"chunk {i}: recovered ids decode to {decoded!r} but reference yielded "
            f"{chunks[i]['text']!r} -- capture indexing is wrong")

    save_file(tensors, args.out + ".safetensors")

    meta = {
        "chunks": chunks,
        "transcript": " ".join(c["text"].strip() for c in chunks if c["text"].strip()),
        "params": {
            "chunk_duration": args.chunk_duration,
            "text_audio_delay": args.text_audio_delay,
            "sample_rate": args.sample_rate,
            "max_new_tokens_per_chunk": args.max_new_tokens_per_chunk,
            "temperature": 0.0,
            "pad_last_chunk": True,
            "encode_mode": "split_then_encode",
            "acoustic_std_dist_type": "none",
            "dtype": "float32",
            "samples": int(wav.shape[0]),
        },
        "special_ids": {
            "speech_start": tokenizer.convert_tokens_to_ids("<|object_ref_start|>"),
            "speech_end": tokenizer.convert_tokens_to_ids("<|object_ref_end|>"),
            "text_chunk_end": tokenizer.convert_tokens_to_ids("<|text_chunk_end|>"),
            "eos": tokenizer.eos_token_id,
        },
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nchunks: {len(chunks)}", flush=True)
    print(f"transcript: {meta['transcript']!r}", flush=True)
    print(f"wrote {args.out}.safetensors / {args.out}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
