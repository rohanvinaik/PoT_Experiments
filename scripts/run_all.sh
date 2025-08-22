#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-manifests/neurips_demo.yaml}"
OUTDIR="${2:-}"

echo "== PoT Runner (batch) =="
if [[ -z "${OUTDIR}" ]]; then
  python tools/pot_runner.py batch --manifest "${MANIFEST}"
else
  python tools/pot_runner.py batch --manifest "${MANIFEST}" --out "${OUTDIR}"
fi

# Optional: quick bootstrap power on the batch directory
# Since we don't use yq, hardcoding RUNROOT
RUNROOT="runs/neurips_demo"
if [[ -d "${RUNROOT}" ]]; then
  echo -e "\n== Bootstrap power (CI proxy) =="
  python tools/pot_runner.py power --run "${RUNROOT}" --B 1000 || true
fi

echo -e "\nDone."