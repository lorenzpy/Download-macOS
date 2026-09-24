"""Tests for the passage-selection logic, independent of the real
cross-encoder model and any network access (a fake score_fn is injected).
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answer import select_answer  # noqa: E402
from web_search import Passage  # noqa: E402


def logit(p: float) -> float:
    """Inverse sigmoid, so tests can specify scores as plain probabilities."""
    return math.log(p / (1 - p))


def make_passages(n: int) -> list[Passage]:
    return [Passage(text=f"passage {i}", source_title=f"Article {i}", source_url=f"https://x/{i}") for i in range(n)]


class SelectAnswerTest(unittest.TestCase):
    def test_single_confident_passage_wins_alone(self):
        passages = make_passages(3)
        scores = [logit(0.95), logit(0.5), logit(0.3)]
        result = select_answer("q", passages, lambda q, ps: scores)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].passage.text, "passage 0")

    def test_low_confidence_strings_multiple_passages_together(self):
        passages = make_passages(4)
        scores = [logit(0.6), logit(0.55), logit(0.5), logit(0.1)]
        result = select_answer("q", passages, lambda q, ps: scores)
        texts = [r.passage.text for r in result]
        self.assertEqual(texts, ["passage 0", "passage 1", "passage 2"])

    def test_no_passages_returns_empty(self):
        result = select_answer("q", [], lambda q, ps: [])
        self.assertEqual(result, [])

    def test_all_low_relevance_still_returns_best_one(self):
        passages = make_passages(2)
        scores = [logit(0.2), logit(0.1)]
        result = select_answer("q", passages, lambda q, ps: scores)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].passage.text, "passage 0")


if __name__ == "__main__":
    unittest.main()
