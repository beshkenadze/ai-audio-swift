#!/usr/bin/env python3
"""Dump VibeVoice-ASR-Streaming reference activations for Swift parity checks.

Runs the official PyTorch implementation on a fixed audio slice and writes the
comparison points to a single safetensors file that MLX Swift loads directly.

IMPORTANT -- the reference inference path is stochastic
-------------------------------------------------------
`VibeVoiceASRForConditionalGeneration.encode_speech` does NOT use the acoustic
VAE mean. It calls

    encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)

and for the released 1.5B checkpoint that config is `std_dist_type="gaussian"`,
`fix_std=0.5`, i.e.

    std = randn(batch) * (fix_std / 0.8)
    x   = mean + std * randn_like(mean)

so every call injects fresh Gaussian noise into the acoustic latent. (The
semantic tokenizer has `fix_std=0`, `std_dist_type="none"`, whose `sample()`
branch returns the mean unchanged -- that path is deterministic.)

Bit-exact parity against that is impossible by construction. So this dump
reports two things:

  * DETERMINISTIC tensors (`*_mean`, `connector_sum`, `first_logits`) built
    from the acoustic VAE mean. These are what the Swift port computes, and
    they are the actual parity targets -- they isolate the ported arithmetic
    from the RNG.
  * NOISE tensors (`sampled_*`), so the magnitude of the reference's own
    run-to-run variation can be compared against the Swift-vs-CUDA gap. A
    port that matches the mean path more closely than the reference matches
    itself across two runs is as good as parity can get here.

Everything runs and is stored in float32: bf16 accumulation differs between
CUDA and Metal, and that drift would mask real porting bugs.

First-step logits use the streaming prefix the model was trained on
(`modeling_vibevoice_asr.py:519-539`):

    prompt -> [sp_start, features, sp_end] -> logits[:, -1]

Usage (on the CUDA box):
  PYTHONPATH=/mnt/d/Projects/vibevoice-official \
  .venv-parity/bin/python vibevoice_parity_dump.py \
      --model /mnt/d/Projects/vibevoice-test/streaming-1.5b \
      --audio /path/to/audio.wav \
      --out /mnt/d/Projects/vibevoice-test/parity_ref.safetensors
"""

import argparse
import sys

import numpy as np
import torch


def stats(name, a, b):
    d = (a - b).abs()
    denom = b.abs().max().item()
    rel = d.max().item() / denom if denom > 0 else float("nan")
    print(f"  {name:<26} max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} "
          f"rel={rel:.3e} (ref max|x|={denom:.3e})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=2.5333333333333332,
                        help="window length: chunk(2.0s) + lookahead(4 frames = 0.5333s)")
    parser.add_argument("--offset", type=float, default=10.0,
                        help="seconds into the file to start, to skip leading silence")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    import librosa
    import soundfile as sf
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

    # ---- audio -------------------------------------------------------
    # Resample once, here, and ship the resulting waveform to the Swift side
    # too: a resampler mismatch would otherwise surface as a model bug at the
    # very first comparison point.
    info = sf.info(args.audio)
    print(f"audio: {info.samplerate} Hz, {info.channels} ch, {info.duration:.1f} s", flush=True)

    wav, _ = librosa.load(
        args.audio, sr=args.sample_rate, mono=True,
        offset=args.offset, duration=args.seconds)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    print(f"slice: {wav.shape[0]} samples, rms={float(np.sqrt((wav**2).mean())):.4f}", flush=True)

    audio = torch.from_numpy(wav)[None, :]  # [1, T]

    # ---- model -------------------------------------------------------
    print("loading model (fp32)...", flush=True)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=None)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    audio = audio.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    acoustic_tok = model.model.acoustic_tokenizer
    semantic_tok = model.model.semantic_tokenizer
    def describe(tok, label):
        # The semantic tokenizer defines neither attribute: its `sampling()`
        # hardcodes dist_type='none', which returns the mean untouched.
        dist = getattr(tok, "std_dist_type", "none (hardcoded)")
        std = getattr(tok, "fix_std", None)
        std_s = f"{float(std):.3f}" if std is not None else "n/a"
        print(f"{label}: std_dist_type={dist!r} fix_std={std_s}", flush=True)

    describe(acoustic_tok, "acoustic")
    describe(semantic_tok, "semantic")

    out = {"input_audio": audio.detach().cpu().float()}

    with torch.no_grad():
        # ---- point 1: tokenizer encoder means ------------------------
        # Encoders take [B, 1, T] (torch NCL). The Swift port takes
        # [B, T, 1] (MLX NLC) -- same tensor, different layout convention.
        audio_nc = audio.unsqueeze(1)
        acoustic_out = acoustic_tok.encode(audio_nc)
        semantic_out = semantic_tok.encode(audio_nc)

        acoustic_mean = acoustic_out.mean
        semantic_mean = semantic_out.mean
        out["acoustic_latent"] = acoustic_mean.detach().cpu().float()
        out["semantic_latent"] = semantic_mean.detach().cpu().float()
        print(f"acoustic_latent {tuple(acoustic_mean.shape)}", flush=True)
        print(f"semantic_latent {tuple(semantic_mean.shape)}", flush=True)

        # ---- point 2: connector sum (deterministic) ------------------
        acoustic_feat = model.model.acoustic_connector(acoustic_mean)
        semantic_feat = model.model.semantic_connector(semantic_mean)
        connector_sum = acoustic_feat + semantic_feat

        out["acoustic_features"] = acoustic_feat.detach().cpu().float()
        out["semantic_features"] = semantic_feat.detach().cpu().float()
        out["connector_sum"] = connector_sum.detach().cpu().float()
        print(f"connector_sum {tuple(connector_sum.shape)}", flush=True)

        # ---- how much does the reference disagree with itself? -------
        # Two independent draws of the model's own inference path bound how
        # much of any Swift-vs-CUDA gap is RNG rather than a porting error.
        torch.manual_seed(0)
        sampled_a = model.encode_speech(audio)
        torch.manual_seed(1)
        sampled_b = model.encode_speech(audio)
        out["sampled_connector_sum_seed0"] = sampled_a.detach().cpu().float()
        out["sampled_connector_sum_seed1"] = sampled_b.detach().cpu().float()

        print("\nreference stochasticity (encode_speech uses .sample(), not the mean):", flush=True)
        stats("sampled(seed0) vs mean", sampled_a, connector_sum)
        stats("sampled(seed1) vs mean", sampled_b, connector_sum)
        stats("sampled(seed0) vs seed1", sampled_a, sampled_b)

        # ---- point 3: first-step logits ------------------------------
        embed_tokens = model.get_input_embeddings()

        prompt_text = (
            "You are a helpful assistant that transcribes audio input into text output. "
            "Please transcribe the following audios streamingly with these keys: speaker, content\n"
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        out["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.int32)
        print(f"\nprompt: {len(prompt_ids)} tokens", flush=True)

        sp_start_id = tokenizer.convert_tokens_to_ids("<|object_ref_start|>")
        sp_end_id = tokenizer.convert_tokens_to_ids("<|object_ref_end|>")
        tce_id = tokenizer.convert_tokens_to_ids("<|text_chunk_end|>")
        print(f"sp_start={sp_start_id} sp_end={sp_end_id} text_chunk_end={tce_id} "
              f"eos={tokenizer.eos_token_id}", flush=True)
        out["special_ids"] = torch.tensor(
            [sp_start_id, sp_end_id, tce_id, tokenizer.eos_token_id], dtype=torch.int32)

        def first_logits_for(features):
            prompt_embeds = embed_tokens(
                torch.tensor([prompt_ids], dtype=torch.long, device=device))
            outputs = model(inputs_embeds=prompt_embeds, use_cache=True, return_dict=True)
            prompt_last = outputs.logits[:, -1, :]
            audio_embeds = torch.cat(
                [
                    embed_tokens(torch.tensor([[sp_start_id]], device=device)),
                    features,
                    embed_tokens(torch.tensor([[sp_end_id]], device=device)),
                ], dim=1)
            outputs = model(
                inputs_embeds=audio_embeds, past_key_values=outputs.past_key_values,
                use_cache=True, return_dict=True)
            return prompt_last, outputs.logits[:, -1, :]

        prompt_last, first_logits = first_logits_for(connector_sum)
        out["prompt_last_logits"] = prompt_last.detach().cpu().float()
        out["first_logits"] = first_logits.detach().cpu().float()

        # Same logits under the reference's own noisy features: this is the
        # bar the Swift port has to beat, not 0.
        _, noisy_logits = first_logits_for(sampled_a)
        out["first_logits_sampled_seed0"] = noisy_logits.detach().cpu().float()
        print("\nlogit impact of the reference's own sampling noise:", flush=True)
        stats("first_logits sampled vs mean", noisy_logits, first_logits)

        def top5(logits, label):
            t = torch.topk(logits[0], 5)
            print(f"{label}:", flush=True)
            for score, idx in zip(t.values.tolist(), t.indices.tolist()):
                print(f"  {idx:>7} {score:+.5f}  {tokenizer.decode([idx])!r}", flush=True)
            return t.indices.tolist()

        det_top = top5(first_logits, "\nfirst-step top-5 (deterministic / mean)")
        noisy_top = top5(noisy_logits, "first-step top-5 (reference sampled)")
        print(f"argmax same under noise: {det_top[0] == noisy_top[0]}", flush=True)

    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    print(f"\nwrote {args.out}", flush=True)
    for k, v in out.items():
        print(f"  {k:<30} {tuple(v.shape)} {v.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
