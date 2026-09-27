"""Core DiAlign weighting and scoring."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Callable, Iterable, Protocol


TOKEN_RE = re.compile(r"\b[\w'-]+\b")


class FrequencyProvider(Protocol):
    def frequency(
        self,
        ngram: str,
        corpus_id: int,
        start_year: int,
        end_year: int,
        smoothing: int,
    ) -> float: ...


@dataclass(frozen=True)
class DiAlignConfig:
    min_n: int = 2
    max_n: int = 5
    start_year: int = 1950
    end_year: int = 2022
    smoothing: int = 0
    ame_corpus_id: int = 17
    ref_bre_corpus_id: int = 6
    marker_boost: float = 1.5
    min_frequency: float = 0.0
    top_k: int = 15


@dataclass(frozen=True)
class NgramEvidence:
    ngram: str
    ame_frequency: float
    ref_bre_frequency: float
    log_ratio: float
    divergence: float
    marker_boost: float
    weight: float
    contribution: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    ame_score: float
    ref_bre_score: float
    top_contributors: tuple[NgramEvidence, ...]
    candidate_ngram_count: int = 0
    evidence_ngram_count: int = 0

    def as_dict(self) -> dict:
        return {
            "ame_alignment_score": self.ame_score,
            "ref_bre_alignment_score": self.ref_bre_score,
            "candidate_ngram_count": self.candidate_ngram_count,
            "evidence_ngram_count": self.evidence_ngram_count,
            "top_contributors": [item.as_dict() for item in self.top_contributors],
        }


@lru_cache(maxsize=100_000)
def nltk_named_entity(ngram: str) -> bool:
    from nltk import ne_chunk, pos_tag
    from nltk.tree import Tree

    tree = ne_chunk(pos_tag(ngram.split()))
    return any(
        isinstance(node, Tree) and node.label() in {"PERSON", "GPE", "ORGANIZATION"}
        for node in tree
    )


def english_stopwords() -> set[str]:
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


class DiAlignScorer:
    def __init__(
        self,
        frequencies: FrequencyProvider,
        dialect_markers: Iterable[str],
        config: DiAlignConfig | None = None,
        stopwords: set[str] | None = None,
        named_entity_check: Callable[[str], bool] | None = None,
    ) -> None:
        self.frequencies = frequencies
        self.markers = {item.lower() for item in dialect_markers}
        self.config = config or DiAlignConfig()
        self.stopwords = english_stopwords() if stopwords is None else stopwords
        self.named_entity_check = named_entity_check or nltk_named_entity

    @staticmethod
    def preprocess(text: str) -> str:
        return re.sub(r"[^\w\s'-]", "", text).lower().strip()

    def _ngrams(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(self.preprocess(text))
        unique: dict[str, None] = {}
        for n in range(self.config.min_n, self.config.max_n + 1):
            for start in range(len(tokens) - n + 1):
                words = tokens[start : start + n]
                if all(word in self.stopwords for word in words):
                    continue
                ngram = " ".join(words)
                if self.named_entity_check(ngram):
                    continue
                unique.setdefault(ngram, None)
        return list(unique)

    def _evidence(self, ngram: str) -> NgramEvidence | None:
        cfg = self.config
        ame_frequency = self.frequencies.frequency(
            ngram, cfg.ame_corpus_id, cfg.start_year, cfg.end_year, cfg.smoothing
        )
        ref_bre_frequency = self.frequencies.frequency(
            ngram, cfg.ref_bre_corpus_id, cfg.start_year, cfg.end_year, cfg.smoothing
        )
        if ame_frequency <= 0.0 or ref_bre_frequency <= 0.0:
            return None
        if ame_frequency < cfg.min_frequency and ref_bre_frequency < cfg.min_frequency:
            return None

        log_ratio = math.log2(ame_frequency / ref_bre_frequency)
        divergence = abs(ame_frequency - ref_bre_frequency) / (ame_frequency + ref_bre_frequency)
        marker_boost = cfg.marker_boost if any(token in self.markers for token in ngram.split()) else 1.0
        weight = divergence * marker_boost
        contribution = abs(log_ratio) * weight
        return NgramEvidence(
            ngram=ngram,
            ame_frequency=ame_frequency,
            ref_bre_frequency=ref_bre_frequency,
            log_ratio=log_ratio,
            divergence=divergence,
            marker_boost=marker_boost,
            weight=weight,
            contribution=contribution,
        )

    def score(self, text: str) -> AlignmentResult:
        ngrams = self._ngrams(text)
        evidence = [item for ngram in ngrams if (item := self._evidence(ngram)) is not None]
        directional_evidence = [item for item in evidence if item.contribution > 0.0]
        ame_total = sum(item.contribution for item in directional_evidence if item.log_ratio > 0.0)
        ref_bre_total = sum(item.contribution for item in directional_evidence if item.log_ratio < 0.0)
        total = ame_total + ref_bre_total
        if total == 0.0:
            return AlignmentResult(
                0.0,
                0.0,
                (),
                candidate_ngram_count=len(ngrams),
                evidence_ngram_count=len(directional_evidence),
            )

        contributors = tuple(
            sorted(directional_evidence, key=lambda item: item.contribution, reverse=True)[
                : self.config.top_k
            ]
        )
        return AlignmentResult(
            ame_score=round(ame_total / total, 4),
            ref_bre_score=round(ref_bre_total / total, 4),
            top_contributors=contributors,
            candidate_ngram_count=len(ngrams),
            evidence_ngram_count=len(directional_evidence),
        )
