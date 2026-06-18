# Weather-RAG Observability Reference

Eine kleine Retrieval-Augmented-Generation-(RAG-)Anwendung („Wetter-Assistent"),
die als Referenzapplikation dient, um zwei Ansätze für Tracing/Observability von
GenAI-Pipelines zu vergleichen:

- **Variante A** — MLflow-natives Tracing (`otel/rag-mlflow.py`)
- **Variante B** — reines OpenTelemetry-SDK (GenAI-Semantic-Conventions), Export
  via OTLP an die MLflow-Ingress (`otel/rag-otel.py`)

Die Anwendung beantwortet Wetterfragen ausschließlich auf Basis eines abgerufenen
Korpus aus wöchentlichen Wetterzusammenfassungen (Open-Meteo-Archivdaten). Jeder Pipeline-Schritt wird getract. Ein Scorer bewertet die
*Qualität der Traces* selbst.

## Architektur

```
            ┌─────────────┐      retrieval      ┌──────────────┐
   Frage  ─▶│  RAG-App    ├────────────────────▶│  ChromaDB    │  (Vektorspeicher)
            │ (otel/*.py) │                     │  :7000       │
            │             │      chat            └──────────────┘
            │             ├────────────────────▶┌──────────────┐
            └──────┬──────┘                     │ LLM-Endpoint │  (OpenAI-kompatibel,
                   │ Traces / Spans             │ :1234/v1     │   z. B. LM Studio)
                   ▼                            └──────────────┘
            ┌─────────────┐
            │  MLflow     │  (Tracking + Trace-UI + Evaluation)
            │  :5000      │
            └─────────────┘
```

## Repository-Struktur

| Pfad | Zweck |
|------|-------|
| `otel/` | Die RAG-Anwendung, der Trace-Scorer, der Korpus-Seeder und das Run-Skript. Siehe [`otel/README.md`](otel/README.md). |
| `docker/` | Compose-Stack für MLflow + ChromaDB + einen einmaligen Korpus-Seeder. Siehe [`docker/README.md`](docker/README.md). |
| `requirements.txt` | Gepinnte Python-Abhängigkeiten für App, Seeder und Scorer. |
| `AP_progress.md` | Fortschritts-/Abdeckungsbewertung gegenüber dem Projektplan. |

## Voraussetzungen

- Python 3.13
- Docker + Docker Compose
- Ein laufender OpenAI-kompatibler LLM-Endpoint (Standard: LM Studio unter
  `http://127.0.0.1:1234/v1`). Jeder OpenAI-kompatible Server funktioniert;
  konfigurierbar über `OPENAI_BASE_URL` / `OPENAI_API_KEY`.

## Quickstart

```bash
# 0. Repository klonen
git clone git@github.com:Novanz/Software-Projekt-OTEL.git
cd Software-Projekt-OTEL

# 1. Python-Umgebung
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Infrastruktur starten (MLflow + ChromaDB) und Korpus seeden
cd docker
docker compose up -d mlflow chromadb
docker compose run --rm seed_chroma      # einmalig: lädt die Wetter-Dokumente in Chroma
cd ..

# 3. Eine RAG-Variante ausführen (Argumente: <app.py> <experiment_id> <user_id> <session_id>)
cd otel
./run.sh rag-otel.py   1 alice session-42  # Variante B (OpenTelemetry + OTLP)
# ./run.sh rag-mlflow.py 1 alice session-42  # Variante A (MLflow-nativ)

# 4. Die erfassten Traces bewerten
python scorer.py
```

Die MLflow-UI ist unter <http://127.0.0.1:5000> erreichbar, um Traces zu
inspizieren.

## Konfiguration

Alle Laufzeiteinstellungen sind Umgebungsvariablen mit sinnvollen Standardwerten;
siehe die Konfigurationstabelle in [`otel/README.md`](otel/README.md). Das
Wrapper-Skript `run.sh` setzt die gängigen Variablen automatisch.
