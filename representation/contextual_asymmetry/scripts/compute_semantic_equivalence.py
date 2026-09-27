#!/usr/bin/env python3
"""Measure contextual semantic equivalence for AmE/BrE counterfactual pairs.

The loss result is only interpretable if the paired sentences are meaning
preserving. This script tests that premise directly by comparing contextual
representations of AmE/BrE sentence pairs against non-equivalent controls.
It also exports small PCA coordinates for an illustrative connected-pair plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import compute_prediction_cost as loss_utils


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def perturb_span(sentence: str, span: str) -> str:
    idx = sentence.find(span)
    if idx < 0 or not span:
        return sentence + " zxqv"
    chars = list(span)
    for pos, ch in enumerate(chars):
        if ch.isalpha():
            chars[pos] = "x" if ch.lower() != "x" else "z"
            break
    else:
        chars.append("x")
    return sentence[:idx] + "".join(chars) + sentence[idx + len(span) :]


def prepare_rows(rows: list[dict[str, str]], max_rows: int | None, seed: int) -> list[dict[str, str]]:
    rows = [r for r in rows if r.get("quality_flag") == "core"]
    rows = [r for r in rows if r.get("variant_type") == "spelling"]
    rows = [r for r in rows if r.get("template_type") in {"minimal_mention", "orthographic_context"}]
    if max_rows and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
        rows = rows[:max_rows]
    return rows


def cosine(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item())


def selected_layer_indices(num_hidden_states: int) -> list[tuple[str, int]]:
    # hidden_states includes embedding output at index 0 and one entry per layer.
    last = num_hidden_states - 1
    mid = last // 2
    picks = [("embedding", 0), ("middle", mid), ("final", last)]
    out: list[tuple[str, int]] = []
    seen = set()
    for name, idx in picks:
        if idx not in seen:
            out.append((name, idx))
            seen.add(idx)
    return out


def sentence_representations(tokenizer, model, texts: list[str], batch_size: int, add_special_tokens: bool):
    import torch

    device = next((p.device for p in model.parameters() if p.device.type != "meta"), None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reps: list[dict[str, Any]] = []
    layer_names: list[str] | None = None
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            add_special_tokens=add_special_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = output.hidden_states
        selected = selected_layer_indices(len(hidden_states))
        if layer_names is None:
            layer_names = [name for name, _ in selected]
        mask = attention_mask.unsqueeze(-1).to(hidden_states[0].dtype)
        denom = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        for item_idx in range(len(batch)):
            item = {}
            for name, layer_idx in selected:
                pooled = (hidden_states[layer_idx][item_idx] * mask[item_idx]).sum(dim=0) / denom[item_idx]
                item[name] = pooled.detach().cpu()
            reps.append(item)
        del output
        del hidden_states
    return reps, (layer_names or [])


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    values = [float(v) for v in values if not math.isnan(float(v))]
    if not values:
        return math.nan, math.nan
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    means.sort()
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def summarize(records: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["model_display_name"], row["layer"], row["comparison"])].append(row)
    out = []
    for (model, layer, comparison), rows in sorted(grouped.items()):
        vals = [float(r["cosine_similarity"]) for r in rows]
        lo, hi = bootstrap_ci(vals, bootstrap_samples, seed)
        out.append(
            {
                "model_display_name": model,
                "layer": layer,
                "comparison": comparison,
                "rows": len(rows),
                "unique_pairs": len({r["pair_id"] for r in rows}),
                "mean_cosine_similarity": round(statistics.mean(vals), 6),
                "ci95_cosine_similarity_low": round(lo, 6),
                "ci95_cosine_similarity_high": round(hi, 6),
            }
        )
    return out


def pca_2d(vectors):
    import numpy as np

    x = np.stack([v.float().numpy() for v in vectors]).astype("float32")
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    return coords


def flush_outputs(
    output_root: Path,
    records: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
    skips: list[dict[str, str]],
    models: list[dict[str, str]],
    completed_models: list[str],
    bootstrap_samples: int,
    seed: int,
    status: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "rq2_semantic_equivalence_records.csv"
    summary_path = output_root / "rq2_semantic_equivalence_summary.csv"
    pca_path = output_root / "rq2_semantic_equivalence_pca_coords.csv"
    skips_path = output_root / "rq2_semantic_equivalence_skips.jsonl"
    run_info_path = output_root / "rq2_semantic_equivalence_run_info.json"
    write_csv(records_path, records)
    write_csv(summary_path, summarize(records, bootstrap_samples, seed) if records else [])
    write_csv(pca_path, pca_rows)
    write_jsonl(skips_path, skips)
    run_info = {
        "status": status,
        "completed_models": completed_models,
        "completed_model_count": len(completed_models),
        "requested_model_count": len(models),
        "skipped_models": skips,
        "similarity_records": len(records),
        "pca_rows": len(pca_rows),
        "records": str(records_path),
        "summary": str(summary_path),
        "pca_coords": str(pca_path),
        "skips": str(skips_path),
    }
    run_info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("representation/contextual_asymmetry/results/sentence_pairs/ame_bre_sentence_pairs.csv"),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("representation/contextual_asymmetry/configs/semantic_equivalence_10model.yaml"),
    )
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("representation/contextual_asymmetry/results/semantic_equivalence"),
    )
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--pca-pairs", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = prepare_rows(loss_utils.read_csv(args.input), args.max_rows, args.seed)
    if len(rows) < 3:
        raise SystemExit("Need at least three rows for shuffled controls.")
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if any(a["pair_id"] == b["pair_id"] for a, b in zip(rows, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]

    config = loss_utils.load_yaml_like(args.models)
    models = loss_utils.select_models(config, args.model)
    all_records: list[dict[str, Any]] = []
    all_pca_rows: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    completed_models: list[str] = []
    add_special_tokens = not args.no_special_tokens

    for idx, model_spec in enumerate(models, start=1):
        repo = model_spec["repo"]
        display = model_spec.get("display_name", repo)
        print(f"[{idx}/{len(models)}] loading model: {repo}", flush=True)
        try:
            tokenizer, model = loss_utils.load_model(repo, args.dtype, args.device_map, args.local_files_only)
        except Exception as exc:
            skips.append({"model": repo, "reason": type(exc).__name__, "message": str(exc)})
            print(f"  skipped {repo}: {type(exc).__name__}: {exc}", flush=True)
            flush_outputs(
                args.output_root,
                all_records,
                all_pca_rows,
                skips,
                models,
                completed_models,
                args.bootstrap_samples,
                args.seed,
                "partial",
            )
            continue

        texts: list[str] = []
        for row, ctrl in zip(rows, shuffled):
            texts.extend(
                [
                    row["ame_sentence"],
                    row["bre_sentence"],
                    ctrl["bre_sentence"],
                    perturb_span(row["ame_sentence"], row["changed_span_ame"]),
                ]
            )
        reps, layer_names = sentence_representations(tokenizer, model, texts, args.batch_size, add_special_tokens)
        model_record_start = len(all_records)
        for i, (row, ctrl) in enumerate(zip(rows, shuffled)):
            ame = reps[4 * i]
            bre = reps[4 * i + 1]
            unrelated = reps[4 * i + 2]
            perturbed = reps[4 * i + 3]
            for layer in layer_names:
                comparisons = {
                    "ame_bre_counterfactual": bre,
                    "unrelated_bre_control": unrelated,
                    "character_perturbation_control": perturbed,
                }
                for comparison, other in comparisons.items():
                    all_records.append(
                        {
                            "model_display_name": display,
                            "model_repo": repo,
                            "model_slug": loss_utils.model_slug(repo),
                            "pair_id": row["pair_id"],
                            "variant_type": row["variant_type"],
                            "template_type": row["template_type"],
                            "ame_variant": row["ame_variant"],
                            "bre_variant": row["bre_variant"],
                            "comparison": comparison,
                            "layer": layer,
                            "cosine_similarity": round(cosine(ame[layer], other[layer]), 6),
                            "control_pair_id": ctrl["pair_id"] if comparison == "unrelated_bre_control" else "",
                        }
                    )

        pca_n = min(args.pca_pairs, len(rows))
        final_vectors = []
        pca_meta = []
        for i, row in enumerate(rows[:pca_n]):
            ame = reps[4 * i]["final"]
            bre = reps[4 * i + 1]["final"]
            final_vectors.extend([ame, bre])
            pca_meta.extend(
                [
                    {"model_display_name": display, "pair_id": row["pair_id"], "dialect": "AmE", "variant": row["ame_variant"]},
                    {"model_display_name": display, "pair_id": row["pair_id"], "dialect": "BrE", "variant": row["bre_variant"]},
                ]
            )
        if final_vectors:
            coords = pca_2d(final_vectors)
            for meta, xy in zip(pca_meta, coords):
                out = dict(meta)
                out.update({"x": round(float(xy[0]), 6), "y": round(float(xy[1]), 6), "layer": "final"})
                all_pca_rows.append(out)

        completed_models.append(repo)
        print(f"  wrote semantic similarity rows: {len(all_records) - model_record_start}", flush=True)

        try:
            import gc
            import torch

            del model
            del tokenizer
            del reps
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        flush_outputs(
            args.output_root,
            all_records,
            all_pca_rows,
            skips,
            models,
            completed_models,
            args.bootstrap_samples,
            args.seed,
            "partial",
        )

    flush_outputs(
        args.output_root,
        all_records,
        all_pca_rows,
        skips,
        models,
        completed_models,
        args.bootstrap_samples,
        args.seed,
        "complete",
    )
    print(
        {
            "similarity_records": len(all_records),
            "pca_rows": len(all_pca_rows),
            "skipped_models": len(skips),
            "output_root": str(args.output_root),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
