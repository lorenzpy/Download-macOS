"""Relevance scoring for (question, passage) pairs.

This is the "decision" half of the app: a cross-encoder classification
model that scores how well a passage answers a question. It never writes
new text -- it only ranks the human-written candidates it is given.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(_MODEL_NAME)


def score_passages(question: str, passages: Sequence[str]) -> list[float]:
    """Score each passage's relevance to the question (higher = better)."""
    if not passages:
        return []
    model = _load_model()
    pairs = [(question, passage) for passage in passages]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
