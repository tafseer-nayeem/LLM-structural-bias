"""Tokenizer fertility and token-length analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .grouping import classify
from .tokenizers import TokenizerAdapter, load_tokenizer


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fertility(text: str, tokenizer: TokenizerAdapter) -> float:
    words = text.split()
    return tokenizer.token_count(text) / len(words) if words else 0.0


def token_bin(n_tokens: int) -> str:
    return str(n_tokens) if n_tokens < 3 else "3+"


def percent_delta(ame_value: float, bre_value: float) -> Optional[float]:
    if ame_value == 0:
        return None
    return ((bre_value - ame_value) / ame_value) * 100.0


def wilcoxon(ame_values: List[int], bre_values: List[int]) -> tuple[Optional[float], Optional[float]]:
    if not ame_values or len(ame_values) != len(bre_values) or all(a == b for a, b in zip(ame_values, bre_values)):
        return None, None
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        stat, p_value = scipy_wilcoxon(ame_values, bre_values, zero_method="wilcox", correction=True)
        return float(stat), float(p_value)
    except Exception:
        return None, None


def analyze_tokenizer(
    entry: Dict[str, Any],
    variants: List[Dict[str, Any]],
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.time()
    adapter = load_tokenizer(entry, hf_token=hf_token)

    pair_rows: List[Dict[str, Any]] = []
    bins: Dict[str, Dict[str, int]] = {"AmE": defaultdict(int), "BrE": defaultdict(int)}
    by_category: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"AmE": [], "BrE": []})
    by_category_lengths: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"AmE": [], "BrE": []})

    for item in variants:
        ame_word = item.get("ame", item.get("us"))
        bre_word = item.get("bre", item.get("uk"))
        if ame_word is None or bre_word is None:
            raise KeyError("Each variant row must contain 'ame' and 'bre' fields.")
        group, diff_type, category = classify(ame_word, bre_word)

        ame_len = adapter.token_count(ame_word)
        bre_len = adapter.token_count(bre_word)
        ame_fertility = fertility(ame_word, adapter)
        bre_fertility = fertility(bre_word, adapter)

        bins["AmE"][token_bin(ame_len)] += 1
        bins["BrE"][token_bin(bre_len)] += 1
        by_category[category]["AmE"].append(ame_fertility)
        by_category[category]["BrE"].append(bre_fertility)
        by_category_lengths[category]["AmE"].append(ame_len)
        by_category_lengths[category]["BrE"].append(bre_len)

        pair_rows.append(
            {
                "model": entry["name"],
                "variant_id": item.get("_id"),
                "ame": ame_word,
                "bre": bre_word,
                "group": group,
                "type": diff_type,
                "category": category,
                "ame_token_count": ame_len,
                "bre_token_count": bre_len,
                "ame_fertility": ame_fertility,
                "bre_fertility": bre_fertility,
            }
        )

    category_rows: List[Dict[str, Any]] = []
    for category, values in sorted(by_category.items()):
        ame = values["AmE"]
        bre = values["BrE"]
        ame_avg = sum(ame) / len(ame) if ame else 0.0
        bre_avg = sum(bre) / len(bre) if bre else 0.0
        stat, p_value = wilcoxon(by_category_lengths[category]["AmE"], by_category_lengths[category]["BrE"])
        category_rows.append(
            {
                "model": entry["name"],
                "category": category,
                "pair_count": len(ame),
                "ame_fertility": ame_avg,
                "bre_fertility": bre_avg,
                "delta_percent": percent_delta(ame_avg, bre_avg),
                "wilcoxon_statistic": stat,
                "wilcoxon_p_value": p_value,
            }
        )

    bin_rows: List[Dict[str, Any]] = []
    for variant_label in ("AmE", "BrE"):
        for bin_label in sorted(bins[variant_label], key=lambda x: 999 if x == "3+" else int(x)):
            bin_rows.append(
                {
                    "model": entry["name"],
                    "variant": variant_label,
                    "token_bin": bin_label,
                    "count": bins[variant_label][bin_label],
                }
            )

    summary = {
        "model": entry["name"],
        "backend": entry["backend"],
        "repo_id": entry.get("repo_id"),
        "encoding": entry.get("model") or entry.get("encoding_name"),
        "origin_country": entry.get("origin_country"),
        "model_access": entry.get("model_access"),
        "vocab_size": adapter.vocab_size,
        "total_vocab_size": adapter.total_vocab_size,
        "variant_pairs": len(variants),
        "runtime_seconds": round(time.time() - started, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "summary": summary,
        "pairs": pair_rows,
        "categories": category_rows,
        "bins": bin_rows,
    }


def run_analysis(
    config_path: Path,
    output_dir: Path,
    variant_file: Path,
    only: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    skip_unavailable: bool = True,
) -> None:
    tokenizers = load_json(config_path)
    selected = set(only or [])
    if selected:
        tokenizers = [entry for entry in tokenizers if entry["name"] in selected or entry["id"] in selected]
    variants = load_json(variant_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for entry in tokenizers:
        print(f"\n=== Tokenizer analysis: {entry['name']} ===")
        try:
            result = analyze_tokenizer(entry, variants, hf_token=hf_token)
        except Exception as exc:
            if not skip_unavailable:
                raise
            print(f"Skipping {entry['name']}: {exc}", file=sys.stderr)
            failures.append({"model": entry["name"], "error": str(exc)})
            continue
        summary_rows.append(result["summary"])
        pair_rows.extend(result["pairs"])
        category_rows.extend(result["categories"])
        bin_rows.extend(result["bins"])

    write_csv(output_dir / "tokenizer_summary.csv", summary_rows)
    write_csv(output_dir / "tokenizer_pair_lengths.csv", pair_rows)
    write_csv(output_dir / "tokenizer_category_fertility.csv", category_rows)
    write_csv(output_dir / "tokenizer_length_bins.csv", bin_rows)
    write_json(output_dir / "tokenizer_failures.json", failures)

    print(f"\nDone. Wrote {output_dir}")


def resolve_hf_token(token_arg: Optional[str], token_file: Optional[Path]) -> Optional[str]:
    if token_arg:
        return token_arg.strip()
    if token_file:
        return token_file.read_text(encoding="utf-8").strip()
    return None


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyze tokenizers and write CSV/JSON outputs.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--variant-file", type=Path, default=DEFAULT_VARIANTS)
    run.add_argument("--only", nargs="*", default=None)
    run.add_argument("--hf-token", default=None, help="Hugging Face access token for gated tokenizers.")
    run.add_argument("--hf-token-file", type=Path, default=None, help="Path to a file containing a Hugging Face token.")
    run.add_argument("--fail-on-unavailable", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_analysis(
            config_path=args.config,
            output_dir=args.output_dir,
            variant_file=args.variant_file,
            only=args.only,
            hf_token=resolve_hf_token(args.hf_token, args.hf_token_file),
            skip_unavailable=not args.fail_on_unavailable,
        )


if __name__ == "__main__":
    main()
