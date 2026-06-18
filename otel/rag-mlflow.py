import json
import os
import time

import chromadb
import mlflow
import requests

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "weather_rag_v1")

# Trace metadata consumed by scorer.py (shared with the OTel variant so the same
# scorer evaluates both). app_version differs to tell the variants apart.
DEFAULT_USER_ID = os.getenv("TRACE_USER_ID", "demo-user")
DEFAULT_SESSION_ID = os.getenv("TRACE_SESSION_ID", "session-1")
TRACE_USE_CASE = os.getenv("TRACE_USE_CASE", "rag_eval_demo")
TRACE_APP_VERSION = os.getenv("TRACE_APP_VERSION", "v1-mlflow")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "weather_rag_docs")
TOP_K = int(os.getenv("TOP_K", "3"))

session = requests.Session()

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION)

_ACTIVE_MODEL = None

def openai_headers():
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

def get_model_name() -> str:
    global _ACTIVE_MODEL
    if _ACTIVE_MODEL:
        return _ACTIVE_MODEL

    if OPENAI_MODEL:
        _ACTIVE_MODEL = OPENAI_MODEL
        return _ACTIVE_MODEL

    response = session.get(
        f"{OPENAI_BASE_URL.rstrip('/')}/models",
        headers=openai_headers(),
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        raise RuntimeError("No models exposed by the OpenAI-compatible endpoint.")
    _ACTIVE_MODEL = data[0]["id"]
    return _ACTIVE_MODEL

@mlflow.trace
def retrieve(query: str, top_k: int = TOP_K):
    with mlflow.start_span("chroma_retrieval") as span:
        results = chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        ranked = []
        for doc_id, doc_text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            doc_text = doc_text or ""

            title = metadata.get("title", doc_id)

            if "\n" in doc_text:
                first_line, rest = doc_text.split("\n", 1)
                if not metadata.get("title"):
                    title = first_line.strip() or doc_id
                text = rest.strip()
            else:
                text = doc_text.strip()

            ranked.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "score": round(float(distance), 4),
                    "city": metadata.get("city"),
                    "week_start": metadata.get("week_start"),
                    "week_end": metadata.get("week_end"),
                    "source": metadata.get("source"),
                }
            )

        span.set_inputs({"query": query, "top_k": top_k})
        span.set_outputs(
            {
                "hits": [
                    {
                        "id": d["id"],
                        "title": d["title"],
                        "score": d["score"],
                        "city": d["city"],
                        "week_start": d["week_start"],
                        "week_end": d["week_end"],
                    }
                    for d in ranked
                ]
            }
        )
        span.set_attribute("retrieval.backend", "chromadb")
        span.set_attribute("retrieval.collection", CHROMA_COLLECTION)
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.hit_count", len(ranked))
        span.set_attribute("retrieval.hit_ids", json.dumps([d["id"] for d in ranked]))

        return ranked

@mlflow.trace
def build_prompt(query: str, docs: list[dict]):
    with mlflow.start_span("prompt_assembly") as span:
        context = "\n\n".join(
            [
                f"[{d['id']}] {d['title']}\n{d['text']}"
                for d in docs
            ]
        )

        prompt = f"""You are a weather assistant.
Answer only from the provided context.
Be precise with city names, dates, temperatures, precipitation, and wind values.
If the retrieved context is weak or incomplete, say so clearly.

Question:
{query}

Context:
{context}

Answer:"""

        span.set_inputs({"query": query, "doc_ids": [d["id"] for d in docs]})
        span.set_outputs({"prompt_preview": prompt[:500]})
        span.set_attribute("prompt.doc_count", len(docs))
        return prompt

@mlflow.trace
def ask_openai_compatible(prompt: str):
    with mlflow.start_span("lmstudio_inference") as span:
        model_name = get_model_name()
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer weather questions strictly from retrieved context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "stream": False,
        }

        t0 = time.perf_counter()
        response = session.post(
            f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers=openai_headers(),
            json=payload,
            timeout=(30,620),
        )
        if not response.ok:
            print("LM Studio error status:", response.status_code)
            print("LM Studio error body:", response.text[:4000])
        response.raise_for_status()

        data = response.json()
        latency = time.perf_counter() - t0

        message = data["choices"][0]["message"]

        answer = (message.get("content") or "").strip()
        if not answer:
            answer = (message.get("reasoning_content") or "").strip()
        # answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        span.set_inputs({"model": model_name, "prompt_preview": prompt[:500]})
        span.set_outputs({"answer_preview": answer[:500]})

        span.set_attribute("inference.latency_sec", round(latency, 3))
        span.set_attribute("openai.prompt_tokens", input_tokens)
        span.set_attribute("openai.completion_tokens", output_tokens)
        span.set_attribute("openai.total_tokens", total_tokens)
        span.set_attribute("llm.provider", "lmstudio")
        span.set_attribute("llm.api", "openai-compatible")

        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

        return answer, model_name

@mlflow.trace
def rag_mock(query: str, user_id: str = DEFAULT_USER_ID, session_id: str = DEFAULT_SESSION_ID):
    root_span = mlflow.get_current_active_span()

    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "trace.use_case": TRACE_USE_CASE,
            "trace.app_version": TRACE_APP_VERSION,
        }
    )

    try:
        docs = retrieve(query, top_k=TOP_K)
        prompt = build_prompt(query, docs)
        answer, model_name = ask_openai_compatible(prompt)

        result = {
            "query": query,
            "model": model_name,
            "retrieved_docs": [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "score": d["score"],
                    "city": d["city"],
                    "week_start": d["week_start"],
                    "week_end": d["week_end"],
                }
                for d in docs
            ],
            "answer": answer,
        }

        if root_span is not None:
            root_span.set_attribute("output.model", model_name)
            root_span.set_attribute("output.answer_length", len(answer))
            root_span.set_attribute("output.status", "ok")

        return result

    except Exception:
        if root_span is not None:
            root_span.set_attribute("output.status", "error")
        raise

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Weather RAG started.")
    print("Type your question and press Enter.")
    print("Type 'quit' to stop.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        if not user_query:
            print("Please enter a question.\n")
            continue

        result = rag_mock(user_query)

        print("\nModel:")
        print(result["model"])
        print("\nAssistant:")
        print(result["answer"])
        print("\nRetrieved docs:")
        for doc in result["retrieved_docs"]:
            print(
                f"- {doc['id']} | {doc['title']} | "
                f"city={doc['city']} | {doc['week_start']}..{doc['week_end']} | score={doc['score']}"
            )
        print()
