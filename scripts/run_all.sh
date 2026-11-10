#!/usr/bin/env bash
# Full pipeline.  Usage:  bash scripts/run_all.sh [configs/default.yaml] [extra --set overrides...]
set -euo pipefail
CFG="${1:-configs/default.yaml}"; shift || true
OVR=()
if [ "$#" -gt 0 ]; then OVR=(--set "$@"); fi
run() { echo "=== spinenet-mo $1 ==="; python -m spinenet_mo.cli "$1" --config "$CFG" "${OVR[@]}"; }

run prepare        # stratified split + class summary
run search         # NSGA-III multi-objective search  (Section 3.4 / Table 1)
run train          # final training of the selected architecture
run evaluate       # test-set metrics, confidence intervals, ROC data
run efficiency     # parameters, GFLOPs, latency
run ablation       # Table 6
run seeds          # Table 3
run baselines      # Tables 4-5
run crossdomain    # Table 7 (needs cross_domain.data_root)
run gradcam
run figures
