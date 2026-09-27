from __future__ import annotations

import unittest

from dialign.scoring import DiAlignConfig, DiAlignScorer


class Frequencies:
    def __init__(self, values: dict[tuple[str, int], float]) -> None:
        self.values = values

    def frequency(self, ngram, corpus_id, start_year, end_year, smoothing):
        return self.values.get((ngram, corpus_id), 0.0)


class ScoringTest(unittest.TestCase):
    def scorer(self) -> DiAlignScorer:
        values = {
            ("color center", 17): 8.0,
            ("color center", 6): 2.0,
            ("colour centre", 17): 1.0,
            ("colour centre", 6): 4.0,
        }
        return DiAlignScorer(
            Frequencies(values),
            {"color", "colour"},
            config=DiAlignConfig(min_n=2, max_n=2),
            stopwords=set(),
            named_entity_check=lambda _: False,
        )

    def test_marker_weighting(self) -> None:
        result = self.scorer().score("color center")
        self.assertEqual(result.ame_score, 1.0)
        self.assertEqual(result.ref_bre_score, 0.0)
        self.assertEqual(result.top_contributors[0].marker_boost, 1.5)
        self.assertAlmostEqual(result.top_contributors[0].weight, 0.9)

    def test_repeated_ngrams_are_scored_once(self) -> None:
        scorer = self.scorer()
        once = scorer.score("color center")
        repeated = scorer.score("color center color center")
        self.assertEqual(once.ame_score, repeated.ame_score)
        self.assertEqual(once.ref_bre_score, repeated.ref_bre_score)

    def test_opposing_evidence_is_normalized(self) -> None:
        result = self.scorer().score("color center colour centre")
        self.assertAlmostEqual(result.ame_score + result.ref_bre_score, 1.0)


if __name__ == "__main__":
    unittest.main()
