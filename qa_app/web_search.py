"""Searches the general web (DuckDuckGo) for articles/tutorials and extracts
their main, human-written text as citable passages. This replaces the
earlier Wikipedia-only retrieval so answers can also come from tutorials
and other websites, not just encyclopedia entries.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Passage:
    text: str
    source_title: str
    source_url: str


def _split_into_passages(text: str, min_len: int = 200, max_len: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    passages: list[str] = []
    for para in paragraphs:
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


def _extract_article(url: str) -> tuple[str, str] | None:
    """Returns (title, main_text) for a URL, or None if extraction fails."""
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(downloaded, favor_recall=True)
    if not text:
        return None
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else url
    return title, text


def fetch_candidate_passages(
    queries: list[str], max_results_per_query: int = 3, max_pages: int = 6
) -> list[Passage]:
    """Runs each search query through DuckDuckGo and extracts text from the
    result pages (articles, tutorials, or any other website)."""
    from ddgs import DDGS

    urls_seen: set[str] = set()
    passages: list[Passage] = []

    with DDGS() as ddgs:
        for query in queries:
            if len(urls_seen) >= max_pages:
                break
            for result in ddgs.text(query, max_results=max_results_per_query):
                url = result.get("href")
                if not url or url in urls_seen:
                    continue
                urls_seen.add(url)
                extracted = _extract_article(url)
                if not extracted:
                    continue
                title, text = extracted
                for chunk in _split_into_passages(text):
                    passages.append(Passage(text=chunk, source_title=title, source_url=url))
                if len(urls_seen) >= max_pages:
                    break
    return passages
