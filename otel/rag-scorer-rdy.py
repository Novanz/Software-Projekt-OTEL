import os

# Required when combining explicit OpenTelemetry spans with MLflow tracing
# through a shared tracer provider.
os.environ.setdefault("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")

import json
import time
from contextlib import nullcontext
from urllib.parse import urlparse

import chromadb
import mlflow
import requests
from mlflow.entities.trace_location import MlflowExperimentLocation

OTEL_AVAILABLE = True
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind, Status, StatusCode
except Exception:
    OTEL_AVAILABLE = False
    otel_trace = None
    TracerProvider = None
    SpanKind = None
    Status = None
    StatusCode = None


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "weather_rag_v1")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "weather_rag_docs")
TOP_K = int(os.getenv("TOP_K", "3"))

ENABLE_OTEL = os.getenv("ENABLE_OTEL", "true").lower() == "true"
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "weather-rag-app")
OTEL_CAPTURE_CONTENT = os.getenv("OTEL_CAPTURE_CONTENT", "false").lower() == "true"

DEFAULT_USER_ID = os.getenv("TRACE_USER_ID", "demo-user")
DEFAULT_SESSION_ID = os.getenv("TRACE_SESSION_ID", "session-1")
TRACE_USE_CASE = os.getenv("TRACE_USE_CASE", "rag_eval_demo")
TRACE_APP_VERSION = os.getenv("TRACE_APP_VERSION", "v2-scorer-ready")
TRACE_CAPTURE_QUERY = os.getenv("TRACE_CAPTURE_QUERY", "true").lower() == "true"
TRACE_CAPTURE_OUTPUT = os.getenv("TRACE_CAPTURE_OUTPUT", "true").lower() == "true"
TRACE_TEXT_LIMIT = int(os.getenv("TRACE_TEXT_LIMIT", "4000"))

session = requests.Session()

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION)

_ACTIVE_MODEL = None
_OTEL_TRACER = None
_EXPERIMENT_ID = None


def clip_text(value: str | None, limit: int = TRACE_TEXT_LIMIT) -> str:
    if not value:
        return ""
    value = str(value)
    return value[:limit]


def json_str(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def setup_tracing():
    global _OTEL_TRACER, _EXPERIMENT_ID

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    _EXPERIMENT_ID = exp.experiment_id

    if not ENABLE_OTEL:
        print("[otel] disabled by ENABLE_OTEL=false")
        return _EXPERIMENT_ID

    if not OTEL_AVAILABLE:
        print("[otel] OpenTelemetry packages are not installed; running with MLflow tracing only")
        return _EXPERIMENT_ID

    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)

    mlflow.tracing.set_destination(MlflowExperimentLocation(_EXPERIMENT_ID))
    _OTEL_TRACER = otel_trace.get_tracer(OTEL_SERVICE_NAME)

    print("[otel] enabled via shared tracer provider")
    print(f"[otel] service.name={OTEL_SERVICE_NAME}")
    print(f"[otel] mlflow_experiment_id={_EXPERIMENT_ID}")

    return _EXPERIMENT_ID


def get_otel_tracer():
    return _OTEL_TRACER


def start_otel_span(name: str, kind=None, attributes: dict | None = None):
    tracer = get_otel_tracer()
    if tracer is None:
        return nullcontext()
    return tracer.start_as_current_span(name, kind=kind, attributes=attributes or {})


def set_otel_ok(span, extra_attrs: dict | None = None):
    if span is None or not OTEL_AVAILABLE:
        return

    if extra_attrs:
        for k, v in extra_attrs.items():
            if v is not None:
                span.set_attribute(k, v)

    span.set_status(Status(StatusCode.OK))


def set_otel_error(span, exc: Exception):
    if span is None or not OTEL_AVAILABLE:
        return

    span.record_exception(exc)
    span.set_attribute("error.type", exc.__class__.__name__)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def openai_headers():
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def parse_server_address_and_port(url: str):
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port

    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80

    return host, port


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
    otel_attrs = {
        "gen_ai.operation.name": "retrieval",
        "gen_ai.data_source.id": CHROMA_COLLECTION,
        "server.address": CHROMA_HOST,
        "server.port": CHROMA_PORT,
        "retrieval.top_k": int(top_k),
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
    }

    kind = SpanKind.CLIENT if OTEL_AVAILABLE else None

    with start_otel_span("chroma_retrieval.otel", kind=kind, attributes=otel_attrs) as otel_span:
        with mlflow.start_span("chroma_retrieval") as span:
            try:
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

                if otel_span is not None:
                    hit_ids = [d["id"] for d in ranked]
                    hit_cities = sorted({d["city"] for d in ranked if d["city"]})
                    hit_ranges = [
                        {
                            "id": d["id"],
                            "city": d["city"],
                            "week_start": d["week_start"],
                            "week_end": d["week_end"],
                            "score": d["score"],
                        }
                        for d in ranked
                    ]

                    otel_span.set_attribute("db.system", "chroma")
                    otel_span.set_attribute("retrieval.backend", "chromadb")
                    otel_span.set_attribute("retrieval.collection", CHROMA_COLLECTION)
                    otel_span.set_attribute("retrieval.hit_count", int(len(ranked)))
                    otel_span.set_attribute("retrieval.hit_ids", json_str(hit_ids))
                    otel_span.set_attribute("retrieval.cities", json_str(hit_cities))
                    otel_span.set_attribute("retrieval.time_ranges", json_str(hit_ranges))

                    if TRACE_CAPTURE_QUERY:
                        otel_span.set_attribute("input.query", clip_text(query))

                    if OTEL_CAPTURE_CONTENT:
                        otel_span.set_attribute(
                            "retrieval.doc_titles",
                            json_str([d["title"] for d in ranked]),
                        )

                    for idx, d in enumerate(ranked, start=1):
                        otel_span.add_event(
                            "retrieval.hit",
                            {
                                "hit.rank": int(idx),
                                "hit.id": str(d["id"]),
                                "hit.score": float(d["score"]),
                                "hit.city": str(d["city"] or ""),
                                "hit.week_start": str(d["week_start"] or ""),
                                "hit.week_end": str(d["week_end"] or ""),
                            },
                        )

                    set_otel_ok(otel_span)

                return ranked

            except Exception as exc:
                set_otel_error(otel_span, exc)
                raise


@mlflow.trace
def build_prompt(query: str, docs: list[dict]):
    kind = SpanKind.INTERNAL if OTEL_AVAILABLE else None

    with start_otel_span(
        "prompt_assembly.otel",
        kind=kind,
        attributes={
            "prompt.doc_count": int(len(docs)),
            "prompt.template_version": "weather-rag-v1",
            "trace.use_case": TRACE_USE_CASE,
            "trace.app_version": TRACE_APP_VERSION,
        },
    ) as otel_span:
        with mlflow.start_span("prompt_assembly") as span:
            try:
                context = "\n\n".join(
                    [f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs]
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

                if otel_span is not None:
                    otel_span.set_attribute("prompt.doc_count", int(len(docs)))
                    otel_span.set_attribute("prompt.doc_ids", json_str([d["id"] for d in docs]))
                    otel_span.set_attribute("prompt.template_version", "weather-rag-v1")
                    otel_span.set_attribute(
                        "prompt.preview",
                        clip_text(prompt, 1200) if OTEL_CAPTURE_CONTENT else "[redacted]",
                    )
                    set_otel_ok(otel_span)

                return prompt

            except Exception as exc:
                set_otel_error(otel_span, exc)
                raise


@mlflow.trace
def ask_openai_compatible(prompt: str):
    model_name = get_model_name()
    server_address, server_port = parse_server_address_and_port(OPENAI_BASE_URL)

    otel_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": model_name,
        "gen_ai.request.temperature": 0.2,
        "gen_ai.request.stream": False,
        "server.address": str(server_address or ""),
        "server.port": int(server_port or 0),
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
    }

    kind = SpanKind.CLIENT if OTEL_AVAILABLE else None

    with start_otel_span("lmstudio_inference.otel", kind=kind, attributes=otel_attrs) as otel_span:
        with mlflow.start_span("lmstudio_inference") as span:
            try:
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You answer weather questions strictly from retrieved context.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.2,
                    "stream": False,
                }

                t0 = time.perf_counter()
                response = session.post(
                    f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                    headers=openai_headers(),
                    json=payload,
                    timeout=(30, 620),
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

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

                finish_reason = data["choices"][0].get("finish_reason")
                response_id = data.get("id")
                response_model = data.get("model", model_name)

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

                if otel_span is not None:
                    otel_span.set_attribute("gen_ai.response.model", str(response_model))
                    if response_id:
                        otel_span.set_attribute("gen_ai.response.id", str(response_id))
                    otel_span.set_attribute("gen_ai.response.finish_reason", str(finish_reason or "unknown"))
                    otel_span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
                    otel_span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
                    otel_span.set_attribute("openai.total_tokens", int(total_tokens))
                    otel_span.set_attribute("llm.provider", "lmstudio")
                    otel_span.set_attribute("llm.api", "openai-compatible")
                    otel_span.set_attribute("inference.latency_sec", float(round(latency, 3)))
                    otel_span.set_attribute("output.answer_length", int(len(answer)))

                    otel_span.add_event(
                        "gen_ai.request.summary",
                        {
                            "message_count": 2,
                            "prompt_preview": clip_text(prompt, 300) if OTEL_CAPTURE_CONTENT else "[redacted]",
                        },
                    )
                    otel_span.add_event(
                        "gen_ai.response.summary",
                        {
                            "answer_preview": clip_text(answer, 300) if OTEL_CAPTURE_CONTENT else "[redacted]",
                            "finish_reason": str(finish_reason or "unknown"),
                        },
                    )

                    set_otel_ok(otel_span)

                return answer, model_name

            except Exception as exc:
                set_otel_error(otel_span, exc)
                raise


@mlflow.trace
def rag_mock(query: str, user_id: str = DEFAULT_USER_ID, session_id: str = DEFAULT_SESSION_ID):
    kind = SpanKind.INTERNAL if OTEL_AVAILABLE else None

    root_attrs = {
        "app.user_id": user_id,
        "app.session_id": session_id,
        "gen_ai.conversation.id": session_id,
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
        "evaluation.enabled": False,
        "evaluation.version": "pending",
    }

    if TRACE_CAPTURE_QUERY:
        root_attrs["input.query"] = clip_text(query)

    with start_otel_span("rag_request.otel", kind=kind, attributes=root_attrs) as otel_span:
        try:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.user": user_id,
                    "mlflow.trace.session": session_id,
                    "trace.use_case": TRACE_USE_CASE,
                    "trace.app_version": TRACE_APP_VERSION,
                }
            )

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

            if otel_span is not None:
                otel_span.set_attribute("rag.retrieved_doc_count", int(len(docs)))
                otel_span.set_attribute("rag.model", str(model_name))
                otel_span.set_attribute("output.model", str(model_name))
                otel_span.set_attribute("output.answer_length", int(len(answer)))
                otel_span.set_attribute("output.status", "ok")

                if TRACE_CAPTURE_OUTPUT:
                    otel_span.set_attribute("output.answer", clip_text(answer))

                otel_span.set_attribute(
                    "retrieval.hit_ids",
                    json_str([d["id"] for d in docs]),
                )
                otel_span.set_attribute(
                    "retrieval.cities",
                    json_str(sorted({d["city"] for d in docs if d["city"]})),
                )

                set_otel_ok(otel_span)

            return result

        except Exception as exc:
            if otel_span is not None:
                otel_span.set_attribute("output.status", "error")
            set_otel_error(otel_span, exc)
            raise


if __name__ == "__main__":
    experiment_id = setup_tracing()

    print("Weather RAG started.")
    print("Type your question and press Enter.")
    print("Type 'quit' to stop.\n")

    print("Runtime configuration:")
    print(f" MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}")
    print(f" MLFLOW_EXPERIMENT={MLFLOW_EXPERIMENT}")
    print(f" MLFLOW_EXPERIMENT_ID={experiment_id}")
    print(f" CHROMA_HOST={CHROMA_HOST}")
    print(f" CHROMA_PORT={CHROMA_PORT}")
    print(f" CHROMA_COLLECTION={CHROMA_COLLECTION}")
    print(f" OPENAI_BASE_URL={OPENAI_BASE_URL}")
    print(f" TOP_K={TOP_K}")
    print(f" ENABLE_OTEL={ENABLE_OTEL}")
    print(f" OTEL_SERVICE_NAME={OTEL_SERVICE_NAME}")
    print(f" OTEL_CAPTURE_CONTENT={OTEL_CAPTURE_CONTENT}")
    print(f" TRACE_USE_CASE={TRACE_USE_CASE}")
    print(f" TRACE_APP_VERSION={TRACE_APP_VERSION}")
    print(f" TRACE_CAPTURE_QUERY={TRACE_CAPTURE_QUERY}")
    print(f" TRACE_CAPTURE_OUTPUT={TRACE_CAPTURE_OUTPUT}")
    print(f" TRACE_TEXT_LIMIT={TRACE_TEXT_LIMIT}")
    print(f" MLFLOW_USE_DEFAULT_TRACER_PROVIDER={os.getenv('MLFLOW_USE_DEFAULT_TRACER_PROVIDER')}")
    print()

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        if not user_query:
            print("Please enter a question.\n")
            continue

        result = rag_mock(user_query, user_id=DEFAULT_USER_ID, session_id=DEFAULT_SESSION_ID)

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
