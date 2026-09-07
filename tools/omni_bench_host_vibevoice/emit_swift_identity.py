#!/usr/bin/env python3
"""Emit the closed 0.6.0 identity document a Swift host needs.

The Python CLI assembles identity itself inside `FoundationApplication.run`;
a Swift host calls the producer directly and has to be handed the same
document. Hand-copying the Registry digests into Swift would rot the first
time the Registry changes and would fail loudly but confusingly, so this
resolves them through omni-bench's own resolver instead -- the digests are
authoritative by construction.

    python3 emit_swift_identity.py \\
        --manifest /path/to/data/asr.synthetic.en.v1/manifest.json \\
        --measurement-profile audio_transcription.batch_single.v1 \\
        --run-profile '{"delivery":"batch","chunk_ms":null,"warmup_samples":0,
                        "concurrency":1,"family_parameters":{}}' \\
        --model '{"base_model_id":"microsoft/VibeVoice-ASR-Streaming-1.5B",
                  "artifact_sha256":null,"quantization":null}' \\
        --backend '{"id":"mlx","version":"0.31.4"}' \\
        --implementation swift \\
        --out identity.json
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--measurement-profile", required=True)
    parser.add_argument("--run-profile", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--implementation", default="swift")
    parser.add_argument("--environment-label", default=None)
    parser.add_argument(
        "--hardware", default=None,
        help="explicit hardware identity JSON; required when the running machine "
             "is not the one being described (e.g. emitting on the Mac for the Mac)")
    parser.add_argument("--os-identity", default=None)
    parser.add_argument("--bundle", default=None, help="consumer bundle root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from omni_bench.application import FoundationApplication
    from omni_bench.core.hardware import collect_os

    app = FoundationApplication.from_bundle(
        Path(args.bundle) if args.bundle else None)

    manifest = json.loads(Path(args.manifest).read_text())
    task = app.task_for_manifest(manifest)
    measurement_ref = app._measurement_ref(task, args.measurement_profile)  # noqa: SLF001

    hardware = json.loads(args.hardware) if args.hardware else app._hardware()  # noqa: SLF001
    os_identity = json.loads(args.os_identity) if args.os_identity else collect_os()

    identity = {
        "model": json.loads(args.model),
        "backend": json.loads(args.backend),
        "hardware": hardware,
        "os": os_identity,
        "implementation": args.implementation,
        "definition_ref": dict(task.task_ref),
        "interface_family_ref": dict(task.task["interface_family_ref"]),
        "measurement_profile_ref": dict(measurement_ref),
        "dataset_content_sha256": task.task["dataset_content_sha256"],
        "protocol_sha256": task.task["protocol_sha256"],
        "run_profile": json.loads(args.run_profile),
        "measurement_environment": {
            "clock": "monotonic",
            "rss_sampling_ms": 10,
            "environment_label": args.environment_label,
        },
    }

    Path(args.out).write_text(json.dumps(identity, indent=2))
    print(f"task     : {task.task_ref['id']}")
    print(f"profile  : {measurement_ref['id']}")
    print(f"hardware : {hardware}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
