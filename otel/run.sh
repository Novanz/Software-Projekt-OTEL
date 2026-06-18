#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <app.py> <experiment id> <user id> <session id>" >&2
    echo "Example: $0 rag-otel.py 1 alice session-42" >&2
    exit 1
fi

APP_FILE="$1"
MLFLOW_EXPERIMENT_ID_ARG="$2"
TRACE_USER_ID_ARG="$3"
TRACE_SESSION_ID_ARG="$4"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-evaluation_v1}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-7000}"
CHROMA_COLLECTION="${CHROMA_COLLECTION:-weather_rag_docs}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:1234/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-lm-studio}"
ENABLE_OTEL="${ENABLE_OTEL:-true}"

export MLFLOW_TRACKING_URI
export MLFLOW_EXPERIMENT
export MLFLOW_EXPERIMENT_ID="$MLFLOW_EXPERIMENT_ID_ARG"
export TRACE_USER_ID="$TRACE_USER_ID_ARG"
export TRACE_SESSION_ID="$TRACE_SESSION_ID_ARG"
export CHROMA_HOST
export CHROMA_PORT
export CHROMA_COLLECTION
export OPENAI_BASE_URL
export OPENAI_API_KEY
export ENABLE_OTEL

if [[ ! -f "$APP_FILE" ]]; then
    echo "Error: app file '$APP_FILE' was not found in $(pwd)" >&2
    echo "Usage: $0 <app.py> <experiment id> <user id> <session id>" >&2
    echo "Example: $0 rag-otel.py 1 alice session-42" >&2
    exit 1
fi

echo "Starting weather RAG app with:"
echo " APP_FILE=$APP_FILE"
echo " MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo " MLFLOW_EXPERIMENT=$MLFLOW_EXPERIMENT"
echo " MLFLOW_EXPERIMENT_ID=$MLFLOW_EXPERIMENT_ID"
echo " TRACE_USER_ID=$TRACE_USER_ID"
echo " TRACE_SESSION_ID=$TRACE_SESSION_ID"
echo " CHROMA_HOST=$CHROMA_HOST"
echo " CHROMA_PORT=$CHROMA_PORT"
echo " CHROMA_COLLECTION=$CHROMA_COLLECTION"
echo " OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo " ENABLE_OTEL=$ENABLE_OTEL"
echo

python "$APP_FILE"
