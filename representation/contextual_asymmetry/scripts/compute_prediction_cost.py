#!/usr/bin/env python3
"""Compute paired AmE/BrE prediction-cost gaps for RQ2.

For each controlled sentence pair, this script estimates the model negative
log-likelihood of the AmE and BrE sentence and reports BrE-minus-AmE loss gaps.
It also exports tokenizer-controlled slices so the paper can distinguish
segmentation effects from learned prediction-cost differences.
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


def load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        data: dict[str, list[dict[str, str]]] = {"models": []}
        section = None
        current = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1]
                data.setdefault(section, [])
                current = None
            elif line.strip().startswith("- "):
                if current is not None and section:
                    data[section].append(current)
                current = {}
                rest = line.strip()[2:]
                if rest and ":" in rest:
                    k, v = rest.split(":", 1)
                    current[k.strip()] = v.strip()
            elif current is not None and ":" in line:
                k, v = line.strip().split(":", 1)
                current[k.strip()] = v.strip()
        if current is not None and section:
            data[section].append(current)
        return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def flush_partial_outputs(
    output_root: Path,
    all_records: list[dict[str, Any]],
    skips: list[dict[str, str]],
    completed_models: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> None:
    """Persist cumulative outputs so long multi-model runs are inspectable."""
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "rq2_loss_gap_records.csv"
    summary_path = output_root / "rq2_loss_gap_summary.csv"
    skips_path = output_root / "rq2_loss_gap_skips.jsonl"
    run_info_path = output_root / "rq2_loss_gap_run_info.json"

    if all_records:
        write_csv(records_path, all_records)
        write_csv(summary_path, summarize(all_records, bootstrap_samples, seed))
    else:
        records_path.touch()
        summary_path.touch()

    write_jsonl(skips_path, skips)
    run_info = {
        "status": "partial",
        "completed_models": completed_models,
        "skipped_models": skips,
        "loss_records": len(all_records),
    }
    run_info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def select_models(config: dict[str, Any], requested: list[str] | None) -> list[dict[str, str]]:
    models = [
        m
        for m in config.get("models", [])
        if m.get("repo") and m.get("role", "confirmatory_loss") != "tokenizer_only"
    ]
    if not requested:
        return models
    wanted = set(requested)
    out = []
    for model in models:
        keys = {model.get("repo", ""), model.get("display_name", ""), model_slug(model.get("repo", ""))}
        if keys & wanted:
            out.append(model)
    missing = wanted - {m.get("repo", "") for m in out} - {m.get("display_name", "") for m in out} - {model_slug(m.get("repo", "")) for m in out}
    if missing:
        raise SystemExit(f"Requested model(s) not found in config: {sorted(missing)}")
    return out


def prepare_rows(rows: list[dict[str, str]], max_rows: int | None, seed: int) -> list[dict[str, str]]:
    rows = [r for r in rows if r.get("quality_flag") == "core"]
    rows = [r for r in rows if r.get("template_type") in {"minimal_mention", "orthographic_context"}]
    if max_rows and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = rows[:]
        rng.shuffle(rows)
        rows = rows[:max_rows]
    return rows


def load_model(repo: str, dtype: str, device_map: str, local_files_only: bool):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    if device_map != "none":
        model_kwargs["device_map"] = device_map
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, local_files_only=local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            repo,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_files_only,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(repo, trust_remote_code=True, local_files_only=local_files_only)
    model_cls = AutoModelForImageTextToText if getattr(config, "model_type", "") == "mistral3" else AutoModelForCausalLM
    model = model_cls.from_pretrained(repo, **model_kwargs)
    model.eval()
    if device_map == "none":
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def sequence_nll(tokenizer, model, texts: list[str], batch_size: int, add_special_tokens: bool) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F

    outputs: list[dict[str, float]] = []
    device = next((p.device for p in model.parameters() if p.device.type != "meta"), None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        losses = losses * shift_mask
        nll_sum = losses.sum(dim=1)
        token_count = shift_mask.sum(dim=1)
        for nll, n_tok in zip(nll_sum.tolist(), token_count.tolist()):
            mean = float(nll / n_tok) if n_tok else math.nan
            outputs.append({"nll_sum": float(nll), "nll_per_token": mean, "loss_token_count": int(n_tok)})
    return outputs


def tokenizer_counts(tokenizer, row: dict[str, str], add_special_tokens: bool) -> dict[str, int | bool]:
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

    ame_variant = row["ame_variant"]
    bre_variant = row["bre_variant"]
    ame_var_len = count(ame_variant)
    bre_var_len = count(bre_variant)
    ame_pref_len = count(" " + ame_variant)
    bre_pref_len = count(" " + bre_variant)
    ame_sent_len = count(row["ame_sentence"])
    bre_sent_len = count(row["bre_sentence"])
    return {
        "ame_variant_token_count": ame_var_len,
        "bre_variant_token_count": bre_var_len,
        "ame_preheld constant_variant_token_count": ame_pref_len,
        "bre_preheld constant_variant_token_count": bre_pref_len,
        "ame_sentence_token_count": ame_sent_len,
        "bre_sentence_token_count": bre_sent_len,
        "variant_token_count_gap_bre_minus_ame": bre_var_len - ame_var_len,
        "preheld constant_variant_token_count_gap_bre_minus_ame": bre_pref_len - ame_pref_len,
        "sentence_token_count_gap_bre_minus_ame": bre_sent_len - ame_sent_len,
        "same_sentence_token_count": ame_sent_len == bre_sent_len,
        "same_preheld constant_variant_token_count": ame_pref_len == bre_pref_len,
        "both_preheld constant_variants_single_token": ame_pref_len == 1 and bre_pref_len == 1,
    }


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1 or samples <= 0:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def summarize(records: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    slices = {
        "all_core": lambda r: True,
        "same_sentence_token_count": lambda r: r["same_sentence_token_count"],
        "same_preheld constant_variant_token_count": lambda r: r["same_preheld constant_variant_token_count"],
        "both_preheld constant_variants_single_token": lambda r: r["both_preheld constant_variants_single_token"],
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        for slice_name, keep in slices.items():
            if keep(row):
                key = (row["model_display_name"], row["variant_type"], row["template_type"], slice_name)
                grouped[key].append(row)

    summary = []
    for (model, variant_type, template_type, slice_name), rows in sorted(grouped.items()):
        gaps_sum = [float(r["nll_sum_gap_bre_minus_ame"]) for r in rows]
        gaps_mean = [float(r["nll_per_token_gap_bre_minus_ame"]) for r in rows]
        lo_sum, hi_sum = bootstrap_ci(gaps_sum, bootstrap_samples, seed)
        lo_mean, hi_mean = bootstrap_ci(gaps_mean, bootstrap_samples, seed + 17)
        summary.append(
            {
                "model_display_name": model,
                "variant_type": variant_type,
                "template_type": template_type,
                "slice": slice_name,
                "rows": len(rows),
                "unique_pairs": len({r["pair_id"] for r in rows}),
                "mean_nll_sum_gap_bre_minus_ame": round(statistics.mean(gaps_sum), 6),
                "ci95_nll_sum_gap_low": round(lo_sum, 6),
                "ci95_nll_sum_gap_high": round(hi_sum, 6),
                "mean_nll_per_token_gap_bre_minus_ame": round(statistics.mean(gaps_mean), 6),
                "ci95_nll_per_token_gap_low": round(lo_mean, 6),
                "ci95_nll_per_token_gap_high": round(hi_mean, 6),
                "pct_bre_higher_nll_sum": round(100 * sum(g > 0 for g in gaps_sum) / len(gaps_sum), 2),
                "mean_sentence_token_count_gap_bre_minus_ame": round(
                    statistics.mean(float(r["sentence_token_count_gap_bre_minus_ame"]) for r in rows), 6
                ),
                "mean_preheld constant_variant_token_count_gap_bre_minus_ame": round(
                    statistics.mean(float(r["preheld constant_variant_token_count_gap_bre_minus_ame"]) for r in rows), 6
                ),
            }
        )
    return summary


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
        default=Path("representation/contextual_asymmetry/configs/prediction_cost_models.yaml"),
    )
    parser.add_argument("--model", action="append", default=None, help="Repo, display name, or slug. Repeatable.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("representation/contextual_asymmetry/results/prediction_cost"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Use 'none' to place the model on one device.")
    parser.add_argument("--local-files-only", action="store_true", help="Load models/tokenizers only from the local HF cache.")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    add_special_tokens = not args.no_special_tokens
    rows = prepare_rows(read_csv(args.input), args.max_rows, args.seed)
    config = load_yaml_like(args.models)
    models = select_models(config, args.model)
    if not models:
        raise SystemExit("No models selected.")

    all_records: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    completed_models: list[dict[str, Any]] = []
    for idx, model_spec in enumerate(models, start=1):
        repo = model_spec["repo"]
        display = model_spec.get("display_name", repo)
        print(f"[{idx}/{len(models)}] loading model: {repo}", flush=True)
        try:
            tokenizer, model = load_model(repo, args.dtype, args.device_map, args.local_files_only)
        except Exception as exc:
            skips.append({"model": repo, "reason": type(exc).__name__, "message": str(exc)})
            print(f"  skipped {repo}: {type(exc).__name__}: {exc}", flush=True)
            flush_partial_outputs(args.output_root, all_records, skips, completed_models, args.bootstrap_samples, args.seed)
            continue

        model_record_start = len(all_records)
        texts: list[str] = []
        for row in rows:
            texts.extend([row["ame_sentence"], row["bre_sentence"]])
        losses = sequence_nll(tokenizer, model, texts, args.batch_size, add_special_tokens)

        for i, row in enumerate(rows):
            ame_loss = losses[2 * i]
            bre_loss = losses[2 * i + 1]
            counts = tokenizer_counts(tokenizer, row, add_special_tokens)
            out: dict[str, Any] = dict(row)
            out.update(counts)
            out.update(
                {
                    "model_display_name": display,
                    "model_repo": repo,
                    "model_slug": model_slug(repo),
                    "ame_nll_sum": ame_loss["nll_sum"],
                    "bre_nll_sum": bre_loss["nll_sum"],
                    "ame_nll_per_token": ame_loss["nll_per_token"],
                    "bre_nll_per_token": bre_loss["nll_per_token"],
                    "ame_loss_token_count": ame_loss["loss_token_count"],
                    "bre_loss_token_count": bre_loss["loss_token_count"],
                    "nll_sum_gap_bre_minus_ame": bre_loss["nll_sum"] - ame_loss["nll_sum"],
                    "nll_per_token_gap_bre_minus_ame": bre_loss["nll_per_token"] - ame_loss["nll_per_token"],
                }
            )
            all_records.append(out)

        model_records = len(all_records) - model_record_start
        completed_models.append(
            {
                "index": idx,
                "total_models": len(models),
                "display_name": display,
                "repo": repo,
                "records": model_records,
            }
        )
        flush_partial_outputs(args.output_root, all_records, skips, completed_models, args.bootstrap_samples, args.seed)
        print(f"  wrote partial outputs after {display}: {model_records} records", flush=True)

        # Release GPU memory before the next model.
        try:
            import gc
            import torch

            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not all_records:
        raise SystemExit("No loss records were produced.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    records_path = args.output_root / "rq2_loss_gap_records.csv"
    summary_path = args.output_root / "rq2_loss_gap_summary.csv"
    skips_path = args.output_root / "rq2_loss_gap_skips.jsonl"
    write_csv(records_path, all_records)
    summary = summarize(all_records, args.bootstrap_samples, args.seed)
    write_csv(summary_path, summary)
    with skips_path.open("w", encoding="utf-8") as f:
        for skip in skips:
            f.write(json.dumps(skip, ensure_ascii=False) + "\n")
    run_info_path = args.output_root / "rq2_loss_gap_run_info.json"
    run_info_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_models": completed_models,
                "skipped_models": skips,
                "loss_records": len(all_records),
                "summary_rows": len(summary),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        {
            "loss_records": len(all_records),
            "summary_rows": len(summary),
            "skipped_models": len(skips),
            "records": str(records_path),
            "summary": str(summary_path),
            "run_info": str(run_info_path),
        }
    )


if __name__ == "__main__":
    main()
