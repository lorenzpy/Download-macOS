# Frage → Textantwort

Web-App, die eine Frage entgegennimmt und als Antwort **keinen von einer KI
geschriebenen Text**, sondern vorhandene, menschlich geschriebene
Textstellen aus dem Web mit Quellenangabe liefert.

## Funktionsweise

Drei Bausteine werden kombiniert:

1. **Verständnis** (`query_ai.py`): Ein kleines lokales Sprachmodell
   (via [Ollama](https://ollama.com)) formt aus der Frage gute
   Suchbegriffe, oder stellt **eine** kurze Rückfrage, falls die Frage zu
   unklar ist. Dieses Modell schreibt nie die eigentliche Antwort.
2. **Suche** (`web_search.py`): DuckDuckGo wird mit den Suchbegriffen
   durchsucht; Ergebnisseiten (Artikel, Tutorials, beliebige Webseiten)
   werden geladen und ihr Haupttext extrahiert und in zitierbare Absätze
   zerlegt.
3. **Entscheidung** (`reranker.py`): Ein Cross-Encoder-Modell
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) bewertet jedes
   Frage/Absatz-Paar mit einem Relevanz-Score. Auch dieses Modell
   **generiert keinen Text**, es entscheidet nur zwischen den vorhandenen
   Kandidaten.

`answer.py` nutzt diese Scores, um zu entscheiden, ob eine einzelne
Textstelle sicher genug die Frage beantwortet oder ob mehrere relevante
Absätze aneinandergereiht werden müssen (für lange, ausführliche
Antworten).

## Installation

### 1. Ollama (für die Verständnis-Komponente)

- Installieren: https://ollama.com/download
- Modell laden: `ollama pull llama3.2:3b`
- Server starten (läuft meist automatisch nach der Installation im
  Hintergrund, sonst manuell): `ollama serve`

Ein 3B-Modell läuft auch ohne dedizierte GPU auf modernen Laptops (z. B.
Lenovo Yoga Gen 7) in vertretbarer Zeit. Falls es zu langsam ist, kann mit
`ollama pull llama3.2:1b` und der Umgebungsvariable
`OLLAMA_MODEL=llama3.2:1b` auf ein kleineres Modell gewechselt werden.

### 2. Python-Abhängigkeiten

```bash
pip install -r requirements.txt
```

Beim ersten Start lädt `sentence-transformers` zusätzlich das
Cross-Encoder-Modell (~90 MB) herunter; dafür ist einmalig eine
Internetverbindung nötig.

## Start

```bash
python app.py
```

Danach im Browser `http://127.0.0.1:5000` öffnen, Frage eingeben. Falls
das Modell nachfragt, die Rückfrage beantworten – danach wird gesucht und
die Antwort mit Quellen angezeigt.

## Tests

Die Auswahllogik (`answer.py`) ist ohne Netzwerk, ohne Ollama und ohne das
reale Reranker-Modell testbar, da die Score-Funktion injiziert wird:

```bash
python -m unittest tests.test_answer -v
```

## Konfiguration

- `OLLAMA_URL` (Standard `http://localhost:11434/api/generate`)
- `OLLAMA_MODEL` (Standard `llama3.2:3b`)

In `answer.py`:

- `HIGH_CONFIDENCE = 0.85` – ab diesem (sigmoid-normalisierten) Score wird
  nur eine einzelne Textstelle zurückgegeben.
- `MIN_RELEVANCE = 0.4` – Textstellen darunter werden verworfen.
- `MAX_PASSAGES = 6` – maximale Anzahl kombinierter Textstellen.

Diese Werte sind heuristisch und können je nach gewünschtem Verhalten
angepasst werden.
