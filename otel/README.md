# otel/ — RAG-Anwendung, Scorer und Seeder

Dieser Ordner enthält die RAG-Referenzapplikation und ihre
Instrumentierungsvarianten, den Korpus-Seeder sowie den Trace-Scorer.

## Inhalt

| Datei | Rolle |
|-------|-------|
| `rag-mlflow.py` | **Variante A** — RAG-Pipeline mit MLflow-nativem Tracing (`@mlflow.trace`, MLflow-SDK). |
| `rag-otel.py` | **Variante B** — dieselbe Pipeline mit reinem OpenTelemetry-SDK: Spans via OTLP/HTTP an die MLflow-OTLP-Ingress (`/v1/traces`), plus GenAI-Metrics. Nutzt kein MLflow-SDK. |
| `scorer.py` | Bewertet die Trace-Qualität (`trace_schema_health` via `mlflow.genai.evaluate`); ein gemeinsamer Scorer für beide Varianten. |
| `seed_weather.py` | Baut den Korpus auf: ruft Open-Meteo-Archivdaten ab, erstellt Wochenzusammenfassungen und lädt sie in ChromaDB. |
| `run.sh` | Wrapper, der Standard-Umgebungsvariablen setzt und eine App-Datei startet. |

## Einrichtung

Die Abhängigkeiten sind im Repository-Root gepinnt:

```bash
cd ..                      # Repository-Root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Die Infrastruktur (MLflow + ChromaDB) und das Seeden des Korpus übernimmt der
Compose-Stack — siehe [`../docker/README.md`](../docker/README.md).

## Ausführung

`run.sh` benötigt vier Positionsargumente und exportiert die gängigen
Einstellungen vor dem Start der App:

```bash
# Verwendung: ./run.sh <app.py> <experiment id> <user id> <session id>
./run.sh rag-mlflow.py 1 alice session-42   # Variante A (MLflow-nativ)
./run.sh rag-otel.py   1 alice session-42   # Variante B (OpenTelemetry + OTLP)
```

Um den Scorer über die erfassten Traces auszuführen:

```bash
python scorer.py
```

## Konfiguration

Alle Einstellungen sind Umgebungsvariablen mit Standardwerten. `run.sh` setzt die
gängigen; eine Überschreibung erfolgt durch Export vor dem Lauf.

| Variable | Standard | Zweck |
|----------|----------|-------|
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | MLflow-Tracking-Server. |
| `MLFLOW_EXPERIMENT` | `weather_rag_v1` | Name des Experiments. |
| `MLFLOW_EXPERIMENT_ID` | *(aus run.sh-Argument)* | Explizite Experiment-ID (vom Scorer genutzt). |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB-Host. |
| `CHROMA_PORT` | `7000` | ChromaDB-Host-Port (Container nutzt 8000; Compose mappt auf 7000). |
| `CHROMA_COLLECTION` | `weather_rag_docs` | Name der Vektor-Collection. |
| `OPENAI_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-kompatibler LLM-Endpoint (z. B. LM Studio). |
| `OPENAI_API_KEY` | `lm-studio` | API-Key für den Endpoint. |
| `OPENAI_MODEL` | *(automatisch erkannt)* | Modell-ID; falls leer, wird das erste Modell aus `/models` verwendet. |
| `TOP_K` | `3` | Anzahl der abgerufenen Dokumente. |
| `OTEL_SERVICE_NAME` | `weather-rag-app` | OpenTelemetry-`service.name` (Variante B). |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `<MLFLOW_TRACKING_URI>/v1/traces` | OTLP-Ziel für Spans (Variante B). |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | *(leer)* | OTLP-Ziel für Metrics; ohne Wert werden Metrics aufgezeichnet, aber nicht exportiert (No-op, da MLflow keine Metrics-Ingress hat). |
| `OTEL_CAPTURE_CONTENT` | `false` | Ob Prompt-/Antworttexte in Span-Attribute aufgenommen werden. |
| `TRACE_USER_ID` / `TRACE_SESSION_ID` | *(aus run.sh-Argumenten)* | An die Traces gehängte Nutzer-/Session-Metadaten. |

Der Korpus-Seeder (`seed_weather.py`) berücksichtigt zusätzlich `START_DATE`,
`END_DATE`, `TIMEZONE` und `RESET_COLLECTION`.

## Fehlerbehebung

### Seeden funktioniert nicht aus dem Chroma-Container heraus

Den Seeder stattdessen vom Host gegen den veröffentlichten Port ausführen:

```bash
export CHROMA_HOST=127.0.0.1
export CHROMA_PORT=7000
export CHROMA_COLLECTION=weather_rag_docs
python seed_weather.py
```

## TODO / Offene Punkte

- **App als Container (AP 3.1.3):** Die Anwendung wird derzeit auf dem Host im
  venv über `run.sh` ausgeführt. Ein `Dockerfile` (Basis `python:3.13-slim` +
  `requirements.txt`) plus ein App-Service in `docker/compose.yaml` würde die
  Reproduzierbarkeit verbessern und AP 3.1.3 vervollständigen. **Noch nicht
  umgesetzt.** Zu beachten:
  - Die App ist eine interaktive REPL (`input()`-Schleife) — ein Container
    bräuchte `stdin_open: true` + `tty: true`; ein Batch-/`--query`-Modus wäre
    container-freundlicher.
  - Der LLM-Endpoint läuft auf dem Host (`127.0.0.1:1234`); aus dem Container
    heraus über `host.docker.internal` bzw. Host-Networking ansprechen.
