"""Command-line data audit for pretraining and post-training corpora."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from tqdm import tqdm

from .text import extract_text, normalize_text, passes_filters
from .variants import (
    VariantCounter,
    analysis_rows,
    enrich_variant_rows,
    excluded_pair_set,
    load_variant_records,
    safe_ratio,
    summarize_group,
    wilcoxon_summary,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VARIANTS = DEFAULT_REPO_ROOT / "resources" / "AmE_BrE_variations.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value: str | Path, repo_root: Path, config_dir: Path) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(raw)
    if path.is_absolute():
        return path
    config_candidate = config_dir / path
    if config_candidate.exists():
        return config_candidate
    package_candidate = PACKAGE_ROOT / path
    if package_candidate.exists():
        return package_candidate
    return repo_root / path


def hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def iter_hf_rows(entry: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    token = hf_token()
    split = entry.get("split", "train")
    streaming = bool(entry.get("streaming", True))
    load_kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
    if token:
        load_kwargs["token"] = token
    if entry.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True

    if entry.get("configs") == "__all__":
        config_kwargs: Dict[str, Any] = {}
        if token:
            config_kwargs["token"] = token
        if entry.get("trust_remote_code"):
            config_kwargs["trust_remote_code"] = True
        configs: List[Optional[str]] = get_dataset_config_names(entry["dataset_id"], **config_kwargs)
    elif isinstance(entry.get("configs"), list):
        configs = entry["configs"]
    elif entry.get("config"):
        configs = [entry["config"]]
    else:
        configs = [None]

    max_rows_per_config = entry.get("max_rows_per_config")
    for config in configs:
        stream = (
            load_dataset(entry["dataset_id"], config, **load_kwargs)
            if config
            else load_dataset(entry["dataset_id"], **load_kwargs)
        )
        seen_for_config = 0
        for row in stream:
            if max_rows_per_config is not None and seen_for_config >= max_rows_per_config:
                break
            seen_for_config += 1
            yield row


def iter_local_jsonl_rows(entry: Dict[str, Any], repo_root: Path, config_dir: Path) -> Iterator[Dict[str, Any]]:
    pattern = resolve_path(entry["path_glob"], repo_root, config_dir)
    files = sorted(pattern.parent.glob(pattern.name))
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def count_text_source(
    entry: Dict[str, Any],
    variant_records: List[Dict[str, Any]],
    repo_root: Path,
    config_dir: Path,
    limit: Optional[int],
) -> Dict[str, Any]:
    source_type = entry.get("source_type", "hf_dataset")
    iterator: Iterable[Dict[str, Any]]
    if source_type == "hf_dataset":
        iterator = iter_hf_rows(entry)
    elif source_type == "local_jsonl_glob":
        iterator = iter_local_jsonl_rows(entry, repo_root, config_dir)
    else:
        raise ValueError(f"Unsupported text source_type for counting: {source_type}")

    counter = VariantCounter(variant_records)
    variant_counts: Counter[str] = Counter()
    rows_seen = 0
    rows_audited = 0
    rows_with_variant = 0
    token_count = 0
    char_count = 0
    desc = entry["id"]
    for row in tqdm(iterator, desc=desc, unit="rows"):
        rows_seen += 1
        if limit is not None and rows_audited >= limit:
            break
        if not passes_filters(row, entry.get("filters")):
            continue
        text = extract_text(row, entry["text_fields"])
        normalized = normalize_text(text)
        rows_audited += 1
        if not normalized:
            continue
        row_counts = counter.count(normalized)
        if row_counts:
            rows_with_variant += 1
            variant_counts.update(row_counts)
        token_count += len(normalized.split())
        char_count += len(text)
    return {
        "variant_counts": variant_counts,
        "rows_seen": rows_seen,
        "rows_audited": rows_audited,
        "rows_with_variant": rows_with_variant,
        "estimated_tokens": token_count,
        "characters": char_count,
    }


def count_word_frequency_source(
    entry: Dict[str, Any],
    variant_records: List[Dict[str, Any]],
    repo_root: Path,
    config_dir: Path,
) -> Dict[str, Any]:
    path = resolve_path(entry["word_frequency_path"], repo_root, config_dir)
    word_freq = load_json(path)
    variant_counts: Counter[str] = Counter()
    for record in variant_records:
        if record["ame_norm"]:
            variant_counts[record["ame_norm"]] += int(word_freq.get(record["ame_norm"], 0))
        if record["bre_norm"]:
            variant_counts[record["bre_norm"]] += int(word_freq.get(record["bre_norm"], 0))
    token_count = sum(int(v) for v in word_freq.values() if isinstance(v, int))
    return {
        "variant_counts": variant_counts,
        "rows_seen": entry.get("rows_seen"),
        "rows_audited": entry.get("rows_audited"),
        "rows_with_variant": None,
        "estimated_tokens": token_count,
        "characters": None,
    }


def summarize_dataset(
    entry: Dict[str, Any],
    variant_records: List[Dict[str, Any]],
    counts: Dict[str, Any],
    limit: Optional[int],
    runtime_seconds: float,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = enrich_variant_rows(
        variant_records,
        counts["variant_counts"],
        excluded_pairs=excluded_pair_set(entry),
    )
    included = analysis_rows(rows)
    valid = [row for row in included if row["total"] > 0]
    ame_total = sum(row["ame_freq"] for row in included)
    bre_total = sum(row["bre_freq"] for row in included)
    total_hits = ame_total + bre_total
    category_rows = summarize_group(rows, "category", entry["id"])
    type_rows = summarize_group(rows, "type", entry["id"])
    category_lookup = {row["category"]: row for row in category_rows}
    orth = category_lookup.get("Orthographic/Spelling", {})
    vocab = category_lookup.get("Vocabulary", {})
    wilcoxon = wilcoxon_summary(rows)
    summary = {
        "id": entry["id"],
        "name": entry["name"],
        "stage": entry.get("stage"),
        "source_type": entry.get("source_type", "hf_dataset"),
        "dataset_id": entry.get("dataset_id"),
        "config": entry.get("config"),
        "split": entry.get("split"),
        "hf_url": entry.get("hf_url"),
        "citation_key": entry.get("citation_key"),
        "text_fields": entry.get("text_fields"),
        "filters": entry.get("filters"),
        "rows_seen": counts.get("rows_seen"),
        "rows_audited": counts.get("rows_audited"),
        "rows_with_variant": counts.get("rows_with_variant"),
        "estimated_tokens": counts.get("estimated_tokens"),
        "characters": counts.get("characters"),
        "variant_pairs": len(included),
        "nonzero_pairs": len(valid),
        "zero_pairs": len(included) - len(valid),
        "excluded_variant_pairs": entry.get("exclude_variant_pairs", []),
        "ame_variant_hits": ame_total,
        "bre_variant_hits": bre_total,
        "total_variant_hits": total_hits,
        "weighted_p_ame": safe_ratio(ame_total, total_hits),
        "weighted_p_bre": safe_ratio(bre_total, total_hits),
        "orthographic_mean_p_ame": orth.get("mean_p_ame"),
        "orthographic_mean_p_bre": orth.get("mean_p_bre"),
        "vocabulary_mean_p_ame": vocab.get("mean_p_ame"),
        "vocabulary_mean_p_bre": vocab.get("mean_p_bre"),
        "wilcoxon_statistic": wilcoxon["statistic"],
        "wilcoxon_p_value": wilcoxon["p_value"],
        "wilcoxon_n_pairs": wilcoxon["n_pairs"],
        "limit": limit,
        "runtime_seconds": round(runtime_seconds, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selection_note": entry.get("selection_note"),
        "schema_note": entry.get("schema_note"),
    }
    return summary, rows, category_rows, type_rows


def audit_entry(
    entry: Dict[str, Any],
    variant_records: List[Dict[str, Any]],
    repo_root: Path,
    config_dir: Path,
    output_dir: Path,
    limit: Optional[int],
    force: bool,
) -> Dict[str, Any]:
    dataset_id = entry["id"]
    summary_path = output_dir / f"{dataset_id}_summary.json"
    pair_path = output_dir / f"{dataset_id}_ame_bre_probabilities.json"
    count_path = output_dir / f"{dataset_id}_variant_counts.json"
    if not force and summary_path.exists() and pair_path.exists() and count_path.exists():
        return load_json(summary_path)

    start = time.time()
    source_type = entry.get("source_type", "hf_dataset")
    if source_type == "word_frequency":
        counts = count_word_frequency_source(entry, variant_records, repo_root, config_dir)
    else:
        counts = count_text_source(entry, variant_records, repo_root, config_dir, limit)

    summary, pair_rows, category_rows, type_rows = summarize_dataset(
        entry,
        variant_records,
        counts,
        limit,
        time.time() - start,
    )
    write_json(pair_path, pair_rows)
    write_json(count_path, dict(sorted(counts["variant_counts"].items())))
    write_json(summary_path, summary)
    write_json(output_dir / f"{dataset_id}_category_summary.json", category_rows)
    write_json(output_dir / f"{dataset_id}_type_summary.json", type_rows)
    return summary


def run_config(
    config_path: Path,
    output_dir: Path,
    variant_file: Path,
    repo_root: Path,
    only: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> None:
    datasets = load_json(config_path)
    selected = set(only or [])
    if selected:
        datasets = [entry for entry in datasets if entry["id"] in selected]
    datasets = [entry for entry in datasets if not entry.get("skip")]
    variant_records = load_variant_records(str(variant_file))
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    type_rows: List[Dict[str, Any]] = []
    config_dir = config_path.parent
    for entry in datasets:
        print(f"\n=== Auditing {entry['id']} ===")
        summary = audit_entry(
            entry,
            variant_records,
            repo_root,
            config_dir,
            output_dir,
            limit,
            force,
        )
        summaries.append(summary)
        category_rows.extend(load_json(output_dir / f"{entry['id']}_category_summary.json"))
        type_rows.extend(load_json(output_dir / f"{entry['id']}_type_summary.json"))

    write_json(output_dir / "dataset_summaries.json", summaries)
    write_json(output_dir / "category_summaries.json", category_rows)
    write_json(output_dir / "type_summaries.json", type_rows)
    write_csv(output_dir / "dataset_summaries.csv", summaries)
    write_csv(output_dir / "category_summaries.csv", category_rows)
    write_csv(output_dir / "type_summaries.csv", type_rows)
    print(f"\nDone. Wrote {output_dir / 'dataset_summaries.csv'}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "audit")
    parser.add_argument("--variant-file", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum audited rows per text dataset.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    run_config(
        config_path=args.config.resolve(),
        output_dir=args.output_dir,
        variant_file=args.variant_file.resolve(),
        repo_root=args.repo_root.resolve(),
        only=args.only,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
