"""Fetches candidate human-written passages from Wikipedia for a question."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Passage:
    text: str
    source_title: str
    source_url: str


def _split_into_passages(content: str, min_len: int = 200, max_len: int = 600) -> list[str]:
    """Split article text into paragraph-sized, citable chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    passages: list[str] = []
    for para in paragraphs:
        if para.startswith("=="):  # Wikipedia section heading
            continue
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            passages.append(para)
        else:
            for i in range(0, len(para), max_len):
                chunk = para[i : i + max_len].strip()
                if len(chunk) >= min_len:
                    passages.append(chunk)
    return passages


def fetch_candidate_passages(question: str, max_articles: int = 3) -> list[Passage]:
    """Search Wikipedia for the question and return candidate passages."""
    import wikipedia

    passages: list[Passage] = []
    titles = wikipedia.search(question, results=max_articles)
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue
        for text in _split_into_passages(page.content):
            passages.append(Passage(text=text, source_title=page.title, source_url=page.url))
    return passages
