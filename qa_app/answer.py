"""Picks the passage(s) that best answer a question, combining more than
one when a single passage does not confidently cover it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from web_search import Passage

ScoreFn = Callable[[str, Sequence[str]], list[float]]

HIGH_CONFIDENCE = 0.85  # sigmoid-normalized score: one passage is enough
MIN_RELEVANCE = 0.4  # below this, a passage is not included at all
MAX_PASSAGES = 6


@dataclass
class RankedPassage:
    passage: Passage
    score: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rank_passages(question: str, passages: Sequence[Passage], score_fn: ScoreFn) -> list[RankedPassage]:
    if not passages:
        return []
    raw_scores = score_fn(question, [p.text for p in passages])
    ranked = [RankedPassage(passage=p, score=_sigmoid(s)) for p, s in zip(passages, raw_scores)]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def select_answer(question: str, passages: Sequence[Passage], score_fn: ScoreFn) -> list[RankedPassage]:
    """Return the passage(s) that best answer the question.

    If the top match is confident enough, only that one passage comes
    back. Otherwise further relevant passages are strung together, most
    relevant first, up to MAX_PASSAGES.
    """
    ranked = rank_passages(question, passages, score_fn)
    if not ranked:
        return []
    if ranked[0].score >= HIGH_CONFIDENCE:
        return ranked[:1]
    selected = [r for r in ranked if r.score >= MIN_RELEVANCE][:MAX_PASSAGES]
    return selected or ranked[:1]
