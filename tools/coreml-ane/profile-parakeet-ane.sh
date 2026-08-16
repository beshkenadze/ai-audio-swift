#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <encoder.mlpackage|encoder.mlmodelc> [report.json]" >&2
    exit 64
fi

model=$1
report=${2:-parakeet-ane-report.json}
min_ane_percent=${MIN_ANE_PERCENT:-95}
max_gpu_ops=${MAX_GPU_OPS:-0}
max_inference_ms=${MAX_INFERENCE_MS:-1000}

if ! command -v anemll-profile >/dev/null 2>&1; then
    echo "anemll-profile not found; install it with: brew install anemll/tap/anemll-profile" >&2
    exit 69
fi
if [[ ! -e "$model" ]]; then
    echo "Core ML model not found: $model" >&2
    exit 66
fi

anemll-profile -a -j "$report" "$model"

jq -e \
    --argjson minANE "$min_ane_percent" \
    --argjson maxGPU "$max_gpu_ops" \
    --argjson maxMs "$max_inference_ms" \
    '
    (.summary.ane_cost | type) == "number"
    and (.summary.gpu_ops | type) == "number"
    and (.measured_prediction.steady_ms | type) == "number"
    and .measured_prediction.status == "ok"
    and (.summary.ane_cost * 100) >= $minANE
    and .summary.gpu_ops <= $maxGPU
    and .measured_prediction.steady_ms <= $maxMs
    ' "$report" >/dev/null

jq '{
    ane_cost_percent: (.summary.ane_cost * 100),
    ane_ops: .summary.ane_ops,
    gpu_ops: .summary.gpu_ops,
    cpu_ops: .summary.cpu_ops,
    steady_ms: .measured_prediction.steady_ms,
    status: .measured_prediction.status
}' "$report"
