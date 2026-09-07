#!/usr/bin/env python3
"""Emit a golden vector proving the two arms resample identically.

The Python host and the Swift host each implement the 16 kHz -> 24 kHz
bandlimited-sinc resampler from the same written formula. "From the same
formula" is an intention, not evidence -- an off-by-one in the tap range or a
different window convention would still produce plausible audio while feeding
the two arms different waveforms, and the resulting parity delta would be
blamed on the MLX port.

This script writes the Python side's exact output for a fixed input so the
Swift test can assert agreement. The input is deterministic and adversarial by
construction: a chirp sweeping to near Nyquist (where interpolation error is
largest), an impulse (exposes tap-range and centering mistakes), a DC step
(exposes gain and window normalization), and silence.

    python3 dump_resampler_golden.py --out Tests/Fixtures/vibevoice_resampler_golden.json
"""

import argparse
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))


def build_input(n: int, sample_rate: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Chirp 200 Hz -> 7.6 kHz (just under the 8 kHz Nyquist of 16 kHz input).
    f0, f1 = 200.0, 7600.0
    duration = n / sample_rate
    phase = 2.0 * math.pi * (f0 * t + (f1 - f0) / (2.0 * duration) * t * t)
    x = 0.6 * np.sin(phase)
    # Impulse: catches a shifted or truncated tap window.
    x[n // 4] += 1.0
    # DC step: catches kernel-gain / window-normalization errors.
    x[n // 2: n // 2 + 400] += 0.35
    # Trailing silence: catches edge handling past the end of the signal.
    x[-200:] = 0.0
    return np.asarray(x, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--source-rate", type=int, default=16000)
    args = parser.parse_args()

    from vibevoice_host import MODEL_SAMPLE_RATE, RESAMPLE_ZEROS, resample_16k_to_24k

    x = build_input(args.samples, args.source_rate)
    y = resample_16k_to_24k(x, source_rate=args.source_rate)

    expected = int(math.floor(x.size * MODEL_SAMPLE_RATE / args.source_rate))
    assert y.size == expected, f"length {y.size} != {expected}"

    payload = {
        "note": "golden vector for the shared VibeVoice 16k->24k resampler; "
                "regenerate with tools/omni_bench_host_vibevoice/dump_resampler_golden.py",
        "source_rate_hz": args.source_rate,
        "target_rate_hz": MODEL_SAMPLE_RATE,
        "resample_zeros": RESAMPLE_ZEROS,
        "input": [float(v) for v in x],
        "output": [float(v) for v in y],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"in={x.size} out={y.size} peak={float(np.abs(y).max()):.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
