#!/usr/bin/env python3
"""Download the AmE--BrE variant resource with Hugging Face Datasets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo, for example USERNAME/ame-bre-structural-bias.",
    )
    parser.add_argument("--data-files", default="data/AmE_BrE_variations.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("resources/AmE_BrE_variations.json"))
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        args.repo_id,
        data_files=args.data_files,
        split=args.split,
    )
    rows = [dict(row) for row in dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json

    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
