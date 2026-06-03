# Prerequisits

## Create venv

```bash
cd <prokect_dir>
python -m venv .venv
source .venv/bin/activate
```

## install deps

```bash
pip install "mlflow>=3.6.0" chromadb requests opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

# Troubleshooting

## Seed does not work in chroma container

temp fix:

```bash
export CHROMA_HOST=127.0.0.1
export CHROMA_PORT=7000
export CHROMA_COLLECTION=weather_rag_docs
python seed_weather.py
```
