# docker/ — Infrastruktur-Stack

Compose-Stack, der die Backend-Dienste für die Weather-RAG-App bereitstellt: einen
MLflow-Tracking-Server, einen ChromaDB-Vektorspeicher und einen einmaligen Job,
der den Korpus seedet.

## Dienste

| Dienst | Image | Adresse | Rolle |
|--------|-------|---------|-------|
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.11.1-full` | `127.0.0.1:5000` | Tracking-Server + Trace-UI; SQLite-Backend, lokaler Artefakt-Store. |
| `chromadb` | `chromadb/chroma:1.5.8` | `127.0.0.1:7000` → Container `8000` | Persistenter Vektorspeicher. |
| `seed_chroma` | `python:3.13-slim` | — | Einmaliger Job, der `../otel/seed_weather.py` ausführt, um den Korpus zu laden. |

## Verwendung

```bash
# Die dauerhaft laufenden Dienste starten
docker compose up -d mlflow chromadb

# Den Korpus seeden (einmalig; erneut ausführen, um neu zu laden)
docker compose run --rm seed_chroma

# Herunterfahren
docker compose down
```

- MLflow-UI: <http://127.0.0.1:5000>
- ChromaDB ist vom Host aus über Port **7000** erreichbar (der Container lauscht
  auf 8000; das Mapping ist beabsichtigt, um Konflikte mit anderen lokalen
  Diensten zu vermeiden).

## Persistenz

Die Daten werden per Bind-Mount in diesen Ordner eingehängt und sind
**gitignored**:

- `./mlflow/` — MLflow-SQLite-DB (`mlflow.db`) und Artefakte
- `./chroma/` — persistente ChromaDB-Daten

Diese Verzeichnisse löschen, um mit einem sauberen Zustand zu starten.

## Hinweise

- Die Volume-Mounts verwenden das SELinux-Label `:Z` (Fedora/RHEL). Unter
  Debian/Ubuntu das Suffix `:Z` aus den Volume-Definitionen in `compose.yaml`
  entfernen.
- Wird Host-basiertes Seeden gegenüber dem `seed_chroma`-Job bevorzugt, siehe den
  Abschnitt zur Fehlerbehebung in [`../otel/README.md`](../otel/README.md).
- Der LLM-Endpoint ist **nicht** Teil dieses Stacks — einen OpenAI-kompatiblen
  Server (z. B. LM Studio) separat betreiben und die App über `OPENAI_BASE_URL`
  darauf verweisen.
