"""Small local conversational AI step (via Ollama).

It only does two things: turn the user's question into good search
queries, or ask ONE clarifying question when the request is too vague to
search well. It never writes the final answer -- that still comes
verbatim from retrieved web text (web_search.py + reranker.py).

Requires a running local Ollama server (`ollama serve`) with a model
pulled, e.g. `ollama pull llama3.2:3b`.
"""
from __future__ import annotations

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

_PROMPT = """Du hilfst dabei, aus einer Nutzerfrage gute Suchbegriffe fuer eine \
Websuche abzuleiten.

Frage: {question}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, ohne weiteren Text, in einer \
der beiden Formen:
- Frage ist klar genug: {{"queries": ["suchbegriff 1", "suchbegriff 2"]}} \
(1 bis 3 praezise Suchbegriffe, keine ganzen Saetze)
- Frage ist zu vage/mehrdeutig, um sinnvoll zu suchen: \
{{"clarify": "eine kurze Rueckfrage an den Nutzer"}}
"""


def _call_ollama(prompt: str) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False, "format": "json"}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["response"]


def understand_question(question: str) -> dict:
    """Returns {"queries": [...]}, or {"clarify": "..."} for one clarifying
    question. Falls back to using the raw question as the only query if the
    model's output can't be parsed."""
    raw = _call_ollama(_PROMPT.format(question=question))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"queries": [question]}
    if data.get("clarify"):
        return {"clarify": str(data["clarify"])}
    queries = data.get("queries") or [question]
    return {"queries": [str(q) for q in queries][:3]}
