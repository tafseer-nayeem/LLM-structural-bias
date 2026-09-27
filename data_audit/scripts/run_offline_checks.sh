#!/usr/bin/env bash
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m compileall -q "$PKG/src" "$PKG/scripts"

python3 "$PKG/scripts/run_data_audit.py" \
  --config "$PKG/configs/example_data.json" \
  --variant-file "$PKG/../resources/example_AmE_BrE_variations.json" \
  --output-dir "$PKG/results/example_check" \
  --force

echo "Offline reproducibility checks completed."
