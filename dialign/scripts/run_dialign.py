#!/usr/bin/env python3
"""Score text with DiAlign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialign.google_ngrams import GoogleNgramFrequencies
from dialign.io import load_markers
from dialign.scoring import DiAlignConfig, DiAlignScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--variant-file",
        type=Path,
        default=REPO_ROOT / "resources" / "AmE_BrE_variations.json",
    )
    parser.add_argument(
        "--frequency-store",
        type=Path,
        default=PACKAGE_ROOT / ".cache" / "ngram_frequencies.sqlite3",
    )
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--marker-boost", type=float, default=1.5)
    args = parser.parse_args()

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    config = DiAlignConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        marker_boost=args.marker_boost,
    )
    with GoogleNgramFrequencies(args.frequency_store) as frequencies:
        scorer = DiAlignScorer(frequencies, load_markers(args.variant_file), config=config)
        result = scorer.score(text)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
