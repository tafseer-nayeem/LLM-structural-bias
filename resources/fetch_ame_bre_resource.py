#!/usr/bin/env python3
"""Download the AmE--BrE variant resource from a Hugging Face dataset repo."""

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
    parser.add_argument("--filename", default="data/AmE_BrE_variations.json")
    parser.add_argument("--output", type=Path, default=Path("resources/AmE_BrE_variations.json"))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(downloaded.read_bytes())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
