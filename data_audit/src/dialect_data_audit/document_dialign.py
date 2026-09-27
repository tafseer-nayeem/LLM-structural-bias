"""Document-level DiAlign analysis for pretraining corpora."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from .text import extract_text, passes_filters, values_for_path


class Scorer(Protocol):
    def score(self, text: str) -> Any: ...


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HF_ACCESS_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if value := os.environ.get(key):
            return value
    return None


def resolve_path(value: str, repo_root: Path, config_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if path.is_absolute():
        return path
    for parent in (config_dir, repo_root / "data_audit", repo_root):
        candidate = parent / path
        if candidate.exists():
            return candidate
    return repo_root / "data_audit" / path


def iter_hf_documents(entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": entry.get("split", "train"),
        "streaming": bool(entry.get("streaming", True)),
    }
    if token := hf_token():
        kwargs["token"] = token
    if entry.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    if entry.get("revision"):
        kwargs["revision"] = entry["revision"]

    config = entry.get("config")
    dataset = (
        load_dataset(entry["dataset_id"], config, **kwargs)
        if config
        else load_dataset(entry["dataset_id"], **kwargs)
    )
    seed = int(entry.get("shuffle_seed", 42))
    if kwargs["streaming"]:
        dataset = dataset.shuffle(seed=seed, buffer_size=int(entry.get("shuffle_buffer", 10_000)))
    else:
        dataset = dataset.shuffle(seed=seed)
    yield from dataset


def iter_local_documents(
    entry: dict[str, Any], repo_root: Path, config_dir: Path
) -> Iterator[dict[str, Any]]:
    pattern = resolve_path(entry["path_glob"], repo_root, config_dir)
    paths = sorted(pattern.parent.glob(pattern.name))
    if not paths:
        raise FileNotFoundError(f"No input files match {pattern}")

    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name}, line {line_number}") from exc
                if isinstance(row, dict):
                    yield row


def iter_local_hf_documents(
    entry: dict[str, Any], repo_root: Path, config_dir: Path
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    dataset_path = resolve_path(entry["dataset_path"], repo_root, config_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    dataset = load_dataset(str(dataset_path), split=entry.get("split", "train"))
    dataset = dataset.shuffle(seed=int(entry.get("shuffle_seed", 42)))
    yield from dataset


def buffered_shuffle(
    rows: Iterable[dict[str, Any]], seed: int, buffer_size: int
) -> Iterator[dict[str, Any]]:
    """Shuffle a stream within a bounded buffer using a reproducible seed."""

    if buffer_size <= 0:
        raise ValueError("shuffle_buffer must be greater than zero")
    generator = random.Random(seed)
    iterator = iter(rows)
    buffer = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    for row in iterator:
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    generator.shuffle(buffer)
    yield from buffer


def iter_documents(
    entry: dict[str, Any], repo_root: Path, config_dir: Path
) -> Iterator[dict[str, Any]]:
    source_type = entry.get("source_type", "hf_dataset")
    if source_type == "hf_dataset":
        yield from iter_hf_documents(entry)
    elif source_type == "local_hf_dataset":
        yield from iter_local_hf_documents(entry, repo_root, config_dir)
    elif source_type == "local_jsonl_glob":
        rows = iter_local_documents(entry, repo_root, config_dir)
        yield from buffered_shuffle(
            rows,
            seed=int(entry.get("shuffle_seed", 42)),
            buffer_size=int(entry.get("shuffle_buffer", 10_000)),
        )
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")


def document_identifier(row: dict[str, Any], id_fields: Iterable[str]) -> str | None:
    for field_name in id_fields:
        values = values_for_path(row, field_name)
        if values and values[0] not in (None, ""):
            return str(values[0])
    return None


def alignment_label(ame_score: float, ref_bre_score: float) -> str:
    if ame_score > ref_bre_score:
        return "AmE"
    if ref_bre_score > ame_score:
        return "Ref-BrE"
    return "no_evidence" if ame_score == 0.0 else "tie"


@dataclass
class CorpusSummary:
    corpus_id: str
    corpus_name: str
    requested_documents: int
    rows_seen: int = 0
    documents_scored: int = 0
    documents_with_evidence: int = 0
    ame_documents: int = 0
    ref_bre_documents: int = 0
    tied_documents: int = 0
    no_evidence_documents: int = 0
    total_candidate_ngrams: int = 0
    total_evidence_ngrams: int = 0
    _ame_score_sum: float = field(default=0.0, repr=False)
    _ref_bre_score_sum: float = field(default=0.0, repr=False)

    def add(self, record: dict[str, Any]) -> None:
        self.documents_scored += 1
        self.total_candidate_ngrams += record["candidate_ngram_count"]
        self.total_evidence_ngrams += record["evidence_ngram_count"]
        label = record["alignment_label"]
        if label == "no_evidence":
            self.no_evidence_documents += 1
            return
        self.documents_with_evidence += 1
        self._ame_score_sum += record["ame_alignment_score"]
        self._ref_bre_score_sum += record["ref_bre_alignment_score"]
        if label == "AmE":
            self.ame_documents += 1
        elif label == "Ref-BrE":
            self.ref_bre_documents += 1
        else:
            self.tied_documents += 1

    def as_dict(self) -> dict[str, Any]:
        evidence_n = self.documents_with_evidence
        scored_n = self.documents_scored
        return {
            "corpus_id": self.corpus_id,
            "corpus_name": self.corpus_name,
            "requested_documents": self.requested_documents,
            "rows_seen": self.rows_seen,
            "documents_scored": scored_n,
            "documents_with_evidence": evidence_n,
            "evidence_coverage": evidence_n / scored_n if scored_n else 0.0,
            "ame_documents": self.ame_documents,
            "ref_bre_documents": self.ref_bre_documents,
            "tied_documents": self.tied_documents,
            "no_evidence_documents": self.no_evidence_documents,
            "ame_document_share": self.ame_documents / evidence_n if evidence_n else 0.0,
            "ref_bre_document_share": self.ref_bre_documents / evidence_n if evidence_n else 0.0,
            "tie_share": self.tied_documents / evidence_n if evidence_n else 0.0,
            "mean_ame_alignment": self._ame_score_sum / evidence_n if evidence_n else 0.0,
            "mean_ref_bre_alignment": self._ref_bre_score_sum / evidence_n if evidence_n else 0.0,
            "total_candidate_ngrams": self.total_candidate_ngrams,
            "total_evidence_ngrams": self.total_evidence_ngrams,
        }


def score_corpus(
    entry: dict[str, Any],
    scorer: Scorer,
    document_limit: int,
    repo_root: Path,
    config_dir: Path,
    output_path: Path,
    max_words: int | None = None,
    include_contributors: bool = False,
) -> dict[str, Any]:
    summary = CorpusSummary(entry["id"], entry["name"], document_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for row in iter_documents(entry, repo_root, config_dir):
            summary.rows_seen += 1
            if not passes_filters(row, entry.get("filters")):
                continue
            text = extract_text(row, entry["text_fields"]).strip()
            if not text:
                continue

            source_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            original_word_count = len(text.split())
            if max_words is not None and original_word_count > max_words:
                text = " ".join(text.split()[:max_words])
            result = scorer.score(text)
            record: dict[str, Any] = {
                "corpus_id": entry["id"],
                "document_index": summary.documents_scored + 1,
                "source_document_id": document_identifier(
                    row, entry.get("id_fields", ["id", "url"])
                ),
                "source_text_sha256": source_text_hash,
                "scored_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "original_word_count": original_word_count,
                "scored_word_count": len(text.split()),
                "text_truncated": len(text.split()) < original_word_count,
                "ame_alignment_score": result.ame_score,
                "ref_bre_alignment_score": result.ref_bre_score,
                "alignment_label": alignment_label(result.ame_score, result.ref_bre_score),
                "candidate_ngram_count": result.candidate_ngram_count,
                "evidence_ngram_count": result.evidence_ngram_count,
            }
            if include_contributors:
                record["top_contributors"] = [item.as_dict() for item in result.top_contributors]
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
            summary.add(record)
            if summary.documents_scored >= document_limit:
                break

    return summary.as_dict()


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
