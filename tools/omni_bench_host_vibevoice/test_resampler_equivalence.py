#!/usr/bin/env python3
"""Prove the Python host's incremental resampler equals its whole-signal one.

This is the Python twin of `streamingMatchesBatch` in the Swift host's test
suite. It exists because an earlier version of this host resampled every model
window independently: that restarted the output grid's phase at each window and
treated both window edges as silence, so the streaming seam quietly fed the
model a different waveform than the batch seam -- and the CUDA arm diverged
from the Swift arm for a reason that had nothing to do with either model. The
transcripts still looked plausible, so only an equivalence check like this one
catches it.

    PYTHONPATH=<omni-bench>/python/src python3 test_resampler_equivalence.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    from dump_resampler_golden import build_input
    from vibevoice_host import MODEL_SAMPLE_RATE, ResamplerStream, resample_16k_to_24k

    source_rate = 16_000
    x = build_input(4000, source_rate)
    batch = resample_16k_to_24k(x, source_rate=source_rate)

    failures = 0
    # A producer paces on wall-clock, so chunk lengths need not divide the
    # kernel stride; 1 and 7 are deliberately pathological.
    for push in (1, 7, 160, 1600, 4096):
        stream = ResamplerStream(source_rate)
        parts = []
        for start in range(0, x.size, push):
            parts.append(stream.push(x[start:start + push]))
        parts.append(stream.finish())
        produced = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        if produced.size != batch.size:
            print(f"FAIL push={push}: length {produced.size} != {batch.size}")
            failures += 1
            continue
        delta = float(np.abs(produced - batch).max()) if batch.size else 0.0
        status = "ok  " if delta == 0.0 else "FAIL"
        if delta != 0.0:
            failures += 1
        print(f"[{status}] push={push:5d}  n={produced.size}  max|delta|={delta:.3e}")

    expected = x.size * MODEL_SAMPLE_RATE // source_rate
    if batch.size != expected:
        print(f"FAIL length ratio: {batch.size} != {expected}")
        failures += 1

    print("PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
