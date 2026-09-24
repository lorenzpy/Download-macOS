"""Web app: ask a question, get back real human-written text as the answer.

Three ingredients are combined:
1. Understanding (query_ai.py) -- a small local LLM (Ollama) turns the
   question into good search queries, or asks ONE clarifying question if
   the request is too vague. It never writes the final answer.
2. Retrieval (web_search.py)   -- DuckDuckGo is searched for the queries,
   result pages (articles, tutorials, any website) are fetched and split
   into candidate passages.
3. Decision (reranker.py)      -- a cross-encoder scores/ranks each
   (question, passage) pair; it never writes new text, it only picks.
answer.py uses those scores to return one passage, or string several
together when one alone does not confidently answer the question.
"""
from __future__ import annotations

from flask import Flask, render_template, request

from answer import select_answer
from query_ai import understand_question
from reranker import score_passages
from web_search import fetch_candidate_passages

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    clarify_question = None
    original_question = None
    result = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        prior_question = request.form.get("original_question", "").strip()
        clarification_answer = request.form.get("clarification_answer", "").strip()

        effective_question = (
            f"{prior_question} ({clarification_answer})"
            if prior_question and clarification_answer
            else question
        )

        if not effective_question:
            error = "Bitte eine Frage eingeben."
        else:
            try:
                understanding = understand_question(effective_question)
                if "clarify" in understanding:
                    clarify_question = understanding["clarify"]
                    original_question = effective_question
                    question = effective_question
                else:
                    passages = fetch_candidate_passages(understanding["queries"])
                    result = select_answer(effective_question, passages, score_passages)
                    if not result:
                        error = "Keine passende Textstelle gefunden."
            except Exception as exc:  # network / Ollama / Wikipedia errors surfaced to the user
                error = f"Fehler: {exc}"

    return render_template(
        "index.html",
        question=question,
        clarify_question=clarify_question,
        original_question=original_question,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
