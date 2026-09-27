#!/usr/bin/env python3
"""Run document-level DiAlign analysis on pretraining corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for source_dir in (PACKAGE_ROOT / "src", REPO_ROOT / "dialign" / "src"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from dialect_data_audit.document_dialign import score_corpus, write_summary_csv
from dialign.google_ngrams import GoogleNgramFrequencies
from dialign.io import load_markers
from dialign.scoring import DiAlignConfig, DiAlignScorer


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "pretraining_documents.json",
    )
    parser.add_argument("--documents-per-corpus", type=int, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--include-contributors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--variant-file",
        type=Path,
        default=REPO_ROOT / "resources" / "AmE_BrE_variations.json",
    )
    parser.add_argument(
        "--frequency-store",
        type=Path,
        default=REPO_ROOT / "dialign" / ".cache" / "ngram_frequencies.sqlite3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "pretraining_dialign",
    )
    args = parser.parse_args()

    if args.documents_per_corpus <= 0:
        parser.error("--documents-per-corpus must be greater than zero")
    if args.max_words is not None and args.max_words <= 0:
        parser.error("--max-words must be greater than zero")

    config_path = args.config.resolve()
    entries = load_json(config_path)
    selected = set(args.only or [])
    if selected:
        unknown = selected - {entry["id"] for entry in entries}
        if unknown:
            parser.error(f"Unknown corpus id(s): {', '.join(sorted(unknown))}")
        entries = [entry for entry in entries if entry["id"] in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    scoring_config = DiAlignConfig()
    with GoogleNgramFrequencies(args.frequency_store) as frequencies:
        scorer = DiAlignScorer(
            frequencies,
            load_markers(args.variant_file),
            config=scoring_config,
        )
        for entry in entries:
            document_path = args.output_dir / f"{entry['id']}_documents.jsonl"
            summary_path = args.output_dir / f"{entry['id']}_summary.json"
            if document_path.exists() or summary_path.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"Output already exists for {entry['id']}; use --overwrite to replace it"
                    )
            summary = score_corpus(
                entry=entry,
                scorer=scorer,
                document_limit=args.documents_per_corpus,
                repo_root=REPO_ROOT,
                config_dir=config_path.parent,
                output_path=document_path,
                max_words=args.max_words,
                include_contributors=args.include_contributors,
            )
            summary.update(
                {
                    "dataset_id": entry.get("dataset_id"),
                    "dataset_config": entry.get("config"),
                    "dataset_revision": entry.get("revision"),
                    "hf_url": entry.get("hf_url"),
                    "source_type": entry.get("source_type", "hf_dataset"),
                    "split": entry.get("split", "train"),
                    "text_fields": entry["text_fields"],
                    "shuffle_seed": entry.get("shuffle_seed"),
                    "shuffle_buffer": entry.get("shuffle_buffer"),
                    "max_words": args.max_words,
                    "start_year": scoring_config.start_year,
                    "end_year": scoring_config.end_year,
                    "min_n": scoring_config.min_n,
                    "max_n": scoring_config.max_n,
                    "ame_corpus_id": scoring_config.ame_corpus_id,
                    "ref_bre_corpus_id": scoring_config.ref_bre_corpus_id,
                    "smoothing": scoring_config.smoothing,
                    "marker_boost": scoring_config.marker_boost,
                    "min_frequency": scoring_config.min_frequency,
                    "unique_ngrams_per_document": True,
                    "corpus_aggregation": "document_macro_average",
                }
            )
            with summary_path.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=True)
            summaries.append(summary)

    with (args.output_dir / "corpus_summaries.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2, ensure_ascii=True)
    write_summary_csv(args.output_dir / "corpus_summaries.csv", summaries)


if __name__ == "__main__":
    main()
