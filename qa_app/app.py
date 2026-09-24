"""Web app: ask a question, get back real human-written text as the answer.

Two ingredients are combined:
1. Retrieval  -- Wikipedia is searched for candidate passages (retrieval.py).
2. Decision   -- a cross-encoder model scores/ranks (question, passage)
   pairs; it never writes new text, it only picks (reranker.py).
answer.py uses those scores to return one passage, or string several
together when one alone does not confidently answer the question.
"""
from __future__ import annotations

from flask import Flask, render_template, request

from answer import select_answer
from reranker import score_passages
from retrieval import fetch_candidate_passages

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    result = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Bitte eine Frage eingeben."
        else:
            try:
                passages = fetch_candidate_passages(question)
                result = select_answer(question, passages, score_passages)
                if not result:
                    error = "Keine passende Textstelle gefunden."
            except Exception as exc:  # network / Wikipedia errors surfaced to the user
                error = f"Fehler bei der Suche: {exc}"

    return render_template("index.html", question=question, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
