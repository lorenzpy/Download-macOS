# Frage → Textantwort

Web-App, die eine Frage entgegennimmt und als Antwort **keinen von einer KI
geschriebenen Text**, sondern vorhandene, menschlich geschriebene
Textstellen mit Quellenangabe liefert.

## Funktionsweise

Zwei Bausteine werden kombiniert:

1. **Retrieval** (`retrieval.py`): Wikipedia wird nach der Frage durchsucht,
   passende Artikel werden geladen und in zitierbare Absätze zerlegt.
2. **Entscheidung** (`reranker.py`): Ein Cross-Encoder-Modell
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) bewertet jedes
   Frage/Absatz-Paar mit einem Relevanz-Score. Das Modell **generiert
   keinen Text**, es entscheidet nur zwischen den vorhandenen Kandidaten.

`answer.py` nutzt diese Scores, um zu entscheiden, ob eine einzelne
Textstelle sicher genug die Frage beantwortet oder ob mehrere relevante
Absätze aneinandergereiht werden müssen.

## Installation

```bash
pip install -r requirements.txt
```

Beim ersten Start lädt `sentence-transformers` das Cross-Encoder-Modell
(~90 MB) herunter; dafür ist einmalig eine Internetverbindung nötig.

## Start

```bash
python app.py
```

Danach im Browser `http://127.0.0.1:5000` öffnen, Frage eingeben.

## Tests

Die Auswahllogik (`answer.py`) ist ohne Netzwerk und ohne das reale
Modell testbar, da die Score-Funktion injiziert wird:

```bash
python -m unittest tests.test_answer -v
```

## Schwellenwerte

In `answer.py`:

- `HIGH_CONFIDENCE = 0.85` – ab diesem (sigmoid-normalisierten) Score wird
  nur eine einzelne Textstelle zurückgegeben.
- `MIN_RELEVANCE = 0.4` – Textstellen darunter werden verworfen.
- `MAX_PASSAGES = 5` – maximale Anzahl kombinierter Textstellen.

Diese Werte sind heuristisch und können je nach gewünschtem Verhalten
angepasst werden.
