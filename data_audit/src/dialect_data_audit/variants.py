"""Variant inventory loading, counting, and summary statistics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .grouping import classify
from .text import normalize_text


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def load_variant_records(path: str) -> List[Dict[str, Any]]:
    """Load AmE/BrE variants and attach normalized forms plus category labels."""

    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[Dict[str, Any]] = []
    for i, entry in enumerate(data, start=1):
        ame = entry.get("ame", entry.get("us"))
        bre = entry.get("bre", entry.get("uk"))
        if ame is None or bre is None:
            raise KeyError("Each variant row must contain 'ame' and 'bre' fields.")
        group, diff_type, category = classify(ame, bre)
        records.append(
            {
                "_id": entry.get("_id", i),
                "ame": ame,
                "bre": bre,
                "ame_norm": normalize_text(ame),
                "bre_norm": normalize_text(bre),
                "group": group,
                "type": diff_type,
                "category": category,
            }
        )
    return records


class VariantCounter:
    """Efficient exact counting of normalized variant words and phrases."""

    def __init__(self, variant_records: List[Dict[str, Any]]) -> None:
        variants = {record["ame_norm"] for record in variant_records}
        variants.update(record["bre_norm"] for record in variant_records)
        variants.discard("")
        self.single_variants = {variant for variant in variants if " " not in variant}
        self.phrase_patterns = {
            phrase: re.compile(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])")
            for phrase in sorted(variant for variant in variants if " " in variant)
        }

    def count(self, normalized_text: str) -> Counter[str]:
        tokens = normalized_text.split()
        row_counts = Counter(token for token in tokens if token in self.single_variants)
        for phrase, pattern in self.phrase_patterns.items():
            count = len(pattern.findall(normalized_text))
            if count:
                row_counts[phrase] += count
        return row_counts


def excluded_pair_set(entry: Dict[str, Any]) -> set[Tuple[str, str]]:
    pairs = set()
    for item in entry.get("exclude_variant_pairs", []):
        ame = item.get("ame", item.get("us"))
        bre = item.get("bre", item.get("uk"))
        if ame is not None and bre is not None:
            pairs.add((normalize_text(ame), normalize_text(bre)))
    return pairs


def enrich_variant_rows(
    variant_records: List[Dict[str, Any]],
    variant_counts: Counter[str],
    excluded_pairs: set[Tuple[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """Attach frequencies and probabilities to every variant pair."""

    excluded_pairs = excluded_pairs or set()
    rows: List[Dict[str, Any]] = []
    for record in variant_records:
        ame_freq = int(variant_counts.get(record["ame_norm"], 0))
        bre_freq = int(variant_counts.get(record["bre_norm"], 0))
        total = ame_freq + bre_freq
        excluded = (record["ame_norm"], record["bre_norm"]) in excluded_pairs
        rows.append(
            {
                "_id": record["_id"],
                "ame": record["ame"],
                "bre": record["bre"],
                "ame_norm": record["ame_norm"],
                "bre_norm": record["bre_norm"],
                "ame_freq": ame_freq,
                "bre_freq": bre_freq,
                "total": total,
                "p_ame": safe_ratio(ame_freq, total),
                "p_bre": safe_ratio(bre_freq, total),
                "group": record["group"],
                "type": record["type"],
                "category": record["category"],
                "excluded_from_summary": excluded,
            }
        )
    return rows


def analysis_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if not row.get("excluded_from_summary")]


def summarize_group(rows: List[Dict[str, Any]], key: str, dataset_id: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows(rows):
        grouped[row[key]].append(row)
    summaries: List[Dict[str, Any]] = []
    for group_value, group_rows in sorted(grouped.items()):
        valid = [row for row in group_rows if row["total"] > 0]
        ame_sum = sum(row["ame_freq"] for row in group_rows)
        bre_sum = sum(row["bre_freq"] for row in group_rows)
        total_sum = ame_sum + bre_sum
        summaries.append(
            {
                "dataset_id": dataset_id,
                key: group_value,
                "pairs": len(group_rows),
                "nonzero_pairs": len(valid),
                "mean_p_ame": (
                    sum(row["p_ame"] for row in valid if row["p_ame"] is not None) / len(valid)
                    if valid
                    else None
                ),
                "mean_p_bre": (
                    sum(row["p_bre"] for row in valid if row["p_bre"] is not None) / len(valid)
                    if valid
                    else None
                ),
                "ame_freq_sum": ame_sum,
                "bre_freq_sum": bre_sum,
                "total_freq_sum": total_sum,
                "weighted_p_ame": safe_ratio(ame_sum, total_sum),
                "weighted_p_bre": safe_ratio(bre_sum, total_sum),
            }
        )
    return summaries


def wilcoxon_summary(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    valid = [row for row in analysis_rows(rows) if row["total"] > 0]
    if not valid or all(row["ame_freq"] == row["bre_freq"] for row in valid):
        return {"statistic": None, "p_value": None, "n_pairs": len(valid)}
    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(
            [row["ame_freq"] for row in valid],
            [row["bre_freq"] for row in valid],
        )
        return {"statistic": float(stat), "p_value": float(p_value), "n_pairs": len(valid)}
    except Exception:
        return {"statistic": None, "p_value": None, "n_pairs": len(valid)}
