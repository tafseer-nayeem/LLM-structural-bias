#!/usr/bin/env bash
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m compileall -q "$PKG/src" "$PKG/scripts"

python3 "$PKG/scripts/run_tokenizer_analysis.py" run \
  --config "$PKG/configs/example_tokenizer.json" \
  --variant-file "$PKG/../../resources/example_AmE_BrE_variations.json" \
  --output-dir "$PKG/results/example_check"

echo "Representation/tokenization example checks completed."
