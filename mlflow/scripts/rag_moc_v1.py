import os
import time

import chromadb
import mlflow
import requests

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://192.168.56.103:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "chroma_v1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.56.103:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

CHROMA_HOST = os.getenv("CHROMA_HOST", "192.168.56.103")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8008"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION)


@mlflow.trace
def retrieve(query: str, top_k: int = 2):
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
                    }
                    for d in ranked
                ]
            }
        )
        span.set_attribute("retrieval.backend", "chromadb")
        span.set_attribute("retrieval.collection", CHROMA_COLLECTION)

        return ranked


@mlflow.trace
def build_prompt(query: str, docs: list[dict]):
    with mlflow.start_span("prompt_assembly") as span:
        context = "\n\n".join(
            [f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs]
        )

        prompt = f"""You are a helpful assistant.
Answer the question using the provided context.
If the context is weak, say so clearly.

Question:
{query}

Context:
{context}

Answer:"""

        span.set_inputs({"query": query, "doc_ids": [d["id"] for d in docs]})
        span.set_outputs({"prompt_preview": prompt[:500]})
        return prompt


@mlflow.trace
def ask_ollama(prompt: str):
    with mlflow.start_span("ollama_inference") as span:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        t0 = time.perf_counter()
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        latency = time.perf_counter() - t0

        answer = data.get("response", "")
        span.set_inputs({"model": OLLAMA_MODEL})
        span.set_outputs({"answer_preview": answer[:500]})
        span.set_attribute("inference.latency_sec", round(latency, 3))
        return answer


@mlflow.trace
def rag_mock(query: str, user_id: str = "demo-user", session_id: str = "session-1"):
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )

    docs = retrieve(query, top_k=2)
    prompt = build_prompt(query, docs)
    answer = ask_ollama(prompt)

    return {
        "query": query,
        "retrieved_docs": [
            {"id": d["id"], "title": d["title"], "score": d["score"]}
            for d in docs
        ],
        "answer": answer,
    }


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("RAG mock started.")
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

        print("\nAssistant:")
        print(result["answer"])
        print("\nRetrieved docs:")
        for doc in result["retrieved_docs"]:
            print(f"- {doc['id']} | {doc['title']} | score={doc['score']}")
        print()
