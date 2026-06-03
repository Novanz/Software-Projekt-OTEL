#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <python_app_file>" >&2
    echo "Example: $0 rag_weather_v1.py" >&2
    exit 1
fi

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-7000}"
CHROMA_COLLECTION="${CHROMA_COLLECTION:-weather_rag_docs}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:1234/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-lm-studio}"
APP_FILE="$1"

export MLFLOW_TRACKING_URI
export CHROMA_HOST
export CHROMA_PORT
export CHROMA_COLLECTION
export OPENAI_BASE_URL
export OPENAI_API_KEY

if [[ ! -f "$APP_FILE" ]]; then
    echo "Error: app file '$APP_FILE' was not found in $(pwd)" >&2
    echo "Usage: $0 <python_app_file>" >&2
    echo "Example: $0 rag_weather_v1.py" >&2
    exit 1
fi

echo "Starting weather RAG app with:"
echo "  APP_FILE=$APP_FILE"
echo "  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "  CHROMA_HOST=$CHROMA_HOST"
echo "  CHROMA_PORT=$CHROMA_PORT"
echo "  CHROMA_COLLECTION=$CHROMA_COLLECTION"
echo "  OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo

python "$APP_FILE"
