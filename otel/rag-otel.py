import json
import os
import time
from urllib.parse import urlparse

import chromadb
import requests

# --- Pure OpenTelemetry SDK (Variant B) -------------------------------------
# This variant uses NO MLflow SDK. Spans are exported via OTLP/HTTP to MLflow's
# OTLP ingress (POST /v1/traces), and metrics via the OTel Metrics signal.
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "weather_rag_v1").strip()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "weather_rag_docs")
TOP_K = int(os.getenv("TOP_K", "3"))

OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "weather-rag-app")
OTEL_CAPTURE_CONTENT = os.getenv("OTEL_CAPTURE_CONTENT", "false").lower() == "true"

# OTLP export targets. Traces default to MLflow's OTLP ingress; metrics have no
# MLflow ingress, so they fall back to console unless an endpoint is provided.
OTLP_TRACES_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    f"{MLFLOW_TRACKING_URI.rstrip('/')}/v1/traces",
)
OTLP_METRICS_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "").strip()
METRIC_EXPORT_INTERVAL_MS = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"))

# MLflow OTLP ingress maps a trace to an experiment via this request header.
MLFLOW_EXPERIMENT_ID_HEADER = "x-mlflow-experiment-id"

DEFAULT_USER_ID = os.getenv("TRACE_USER_ID", "demo-user")
DEFAULT_SESSION_ID = os.getenv("TRACE_SESSION_ID", "session-1")
TRACE_USE_CASE = os.getenv("TRACE_USE_CASE", "rag_eval_demo")
TRACE_APP_VERSION = os.getenv("TRACE_APP_VERSION", "v2-otel-otlp")
TRACE_CAPTURE_QUERY = os.getenv("TRACE_CAPTURE_QUERY", "true").lower() == "true"
TRACE_CAPTURE_OUTPUT = os.getenv("TRACE_CAPTURE_OUTPUT", "true").lower() == "true"
TRACE_TEXT_LIMIT = int(os.getenv("TRACE_TEXT_LIMIT", "4000"))

session = requests.Session()

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_collection = chroma_client.get_collection(name=CHROMA_COLLECTION)

_ACTIVE_MODEL = None
_TRACER = None
_TOKEN_USAGE = None
_OP_DURATION = None


def clip_text(value: str | None, limit: int = TRACE_TEXT_LIMIT) -> str:
    if not value:
        return ""
    return str(value)[:limit]


def json_str(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def resolve_experiment_id() -> str:
    """Resolve the MLflow experiment id via the REST API (no MLflow SDK).

    Uses MLFLOW_EXPERIMENT_ID if provided, otherwise looks the experiment up by
    name and creates it if missing.
    """
    if MLFLOW_EXPERIMENT_ID:
        return MLFLOW_EXPERIMENT_ID

    base = MLFLOW_TRACKING_URI.rstrip("/")
    resp = session.get(
        f"{base}/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": MLFLOW_EXPERIMENT},
        timeout=30,
    )
    if resp.status_code == 200:
        return str(resp.json()["experiment"]["experiment_id"])

    create = session.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": MLFLOW_EXPERIMENT},
        timeout=30,
    )
    create.raise_for_status()
    return str(create.json()["experiment_id"])


def setup_telemetry():
    """Configure the OTel TracerProvider + MeterProvider and OTLP exporters."""
    global _TRACER, _TOKEN_USAGE, _OP_DURATION

    experiment_id = resolve_experiment_id()
    resource = Resource.create(
        {
            "service.name": OTEL_SERVICE_NAME,
            "service.version": TRACE_APP_VERSION,
        }
    )

    # Traces -> MLflow OTLP ingress
    span_exporter = OTLPSpanExporter(
        endpoint=OTLP_TRACES_ENDPOINT,
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    otel_trace.set_tracer_provider(tracer_provider)
    _TRACER = otel_trace.get_tracer(OTEL_SERVICE_NAME)

    # Metrics signal.
    #
    # The GenAI metric instruments below are always recorded, but there is no
    # metrics backend yet: MLflow only ingests *traces* (POST /v1/traces); it has
    # no metrics ingress or metrics UI. So by default we attach NO metric reader,
    # which makes the MeterProvider a no-op sink (instruments record without error,
    # nothing is collected or exported, no console spam).
    #
    # TODO (AP 5.1.3 / 5.2.5 export target): wire the metrics to a real backend.
    #   Set OTEL_EXPORTER_OTLP_METRICS_ENDPOINT to an OTLP collector
    #   (e.g. an OpenTelemetry Collector -> Prometheus/Grafana) and the OTLP
    #   reader below activates automatically. This is where p95 latency,
    #   tokens/minute, and error-rate dashboards would come from.
    #   For quick local debugging instead, swap in a console reader:
    #       from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
    #       metric_readers = [PeriodicExportingMetricReader(
    #           ConsoleMetricExporter(),
    #           export_interval_millis=METRIC_EXPORT_INTERVAL_MS)]
    metric_readers = []
    if OTLP_METRICS_ENDPOINT:
        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=OTLP_METRICS_ENDPOINT,
                    headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
                ),
                export_interval_millis=METRIC_EXPORT_INTERVAL_MS,
            )
        )
        metric_target = OTLP_METRICS_ENDPOINT
    else:
        metric_target = "disabled (no reader - metrics recorded but not exported)"

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    otel_metrics.set_meter_provider(meter_provider)

    meter = otel_metrics.get_meter(OTEL_SERVICE_NAME)
    # GenAI metric conventions: token usage and operation duration histograms.
    # Errors are recorded as the error.type dimension on the duration histogram.
    _TOKEN_USAGE = meter.create_histogram(
        "gen_ai.client.token.usage",
        unit="{token}",
        description="Number of tokens used per GenAI request",
    )
    _OP_DURATION = meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="s",
        description="Duration of GenAI operations",
    )

    print("[otel] pure OpenTelemetry SDK variant")
    print(f"[otel] service.name={OTEL_SERVICE_NAME} service.version={TRACE_APP_VERSION}")
    print(f"[otel] traces -> {OTLP_TRACES_ENDPOINT} (experiment {experiment_id})")
    print(f"[otel] metrics -> {metric_target}")

    return experiment_id, tracer_provider, meter_provider


def set_ok(span, extra_attrs: dict | None = None):
    if extra_attrs:
        for k, v in extra_attrs.items():
            if v is not None:
                span.set_attribute(k, v)
    span.set_status(Status(StatusCode.OK))


def set_error(span, exc: Exception):
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


def retrieve(query: str, top_k: int = TOP_K):
    attrs = {
        "gen_ai.operation.name": "retrieval",
        "gen_ai.data_source.id": CHROMA_COLLECTION,
        "db.system": "chroma",
        "server.address": CHROMA_HOST,
        "server.port": CHROMA_PORT,
        "retrieval.top_k": int(top_k),
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
    }
    op_attrs = {"gen_ai.operation.name": "retrieval"}

    with _TRACER.start_as_current_span(
        "chroma_retrieval.otel", kind=SpanKind.CLIENT, attributes=attrs
    ) as span:
        t0 = time.perf_counter()
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

            elapsed = time.perf_counter() - t0

            span.set_attribute("retrieval.backend", "chromadb")
            span.set_attribute("retrieval.collection", CHROMA_COLLECTION)
            span.set_attribute("retrieval.hit_count", int(len(ranked)))
            span.set_attribute("retrieval.hit_ids", json_str([d["id"] for d in ranked]))
            span.set_attribute(
                "retrieval.cities",
                json_str(sorted({d["city"] for d in ranked if d["city"]})),
            )

            # MLflow-reserved keys so the trace UI renders span inputs/outputs.
            span.set_attribute(
                "mlflow.spanInputs",
                json_str({"query": clip_text(query) if TRACE_CAPTURE_QUERY else "[redacted]", "top_k": int(top_k)}),
            )
            span.set_attribute(
                "mlflow.spanOutputs",
                json_str([{"id": d["id"], "title": d["title"], "score": d["score"], "city": d["city"]} for d in ranked]),
            )

            if TRACE_CAPTURE_QUERY:
                span.set_attribute("input.query", clip_text(query))

            if OTEL_CAPTURE_CONTENT:
                span.set_attribute(
                    "retrieval.doc_titles",
                    json_str([d["title"] for d in ranked]),
                )

            for idx, d in enumerate(ranked, start=1):
                span.add_event(
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

            _OP_DURATION.record(elapsed, op_attrs)
            set_ok(span)
            return ranked

        except Exception as exc:
            _OP_DURATION.record(
                time.perf_counter() - t0,
                {**op_attrs, "error.type": exc.__class__.__name__},
            )
            set_error(span, exc)
            raise


def build_prompt(query: str, docs: list[dict]):
    attrs = {
        "prompt.doc_count": int(len(docs)),
        "prompt.template_version": "weather-rag-v1",
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
    }

    with _TRACER.start_as_current_span(
        "prompt_assembly.otel", kind=SpanKind.INTERNAL, attributes=attrs
    ) as span:
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

            span.set_attribute("prompt.doc_count", int(len(docs)))
            span.set_attribute("prompt.doc_ids", json_str([d["id"] for d in docs]))
            span.set_attribute(
                "prompt.preview",
                clip_text(prompt, 1200) if OTEL_CAPTURE_CONTENT else "[redacted]",
            )
            span.set_attribute(
                "mlflow.spanInputs",
                json_str({"query": clip_text(query) if TRACE_CAPTURE_QUERY else "[redacted]", "doc_ids": [d["id"] for d in docs]}),
            )
            span.set_attribute(
                "mlflow.spanOutputs",
                json_str({"prompt_preview": clip_text(prompt, 500) if OTEL_CAPTURE_CONTENT else "[redacted]"}),
            )
            set_ok(span)
            return prompt

        except Exception as exc:
            set_error(span, exc)
            raise


def ask_openai_compatible(prompt: str):
    model_name = get_model_name()
    server_address, server_port = parse_server_address_and_port(OPENAI_BASE_URL)

    attrs = {
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
    metric_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": model_name,
    }

    with _TRACER.start_as_current_span(
        "lmstudio_inference.otel", kind=SpanKind.CLIENT, attributes=attrs
    ) as span:
        t0 = time.perf_counter()
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

            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.response.model", str(response_model))
            if response_id:
                span.set_attribute("gen_ai.response.id", str(response_id))
            span.set_attribute("gen_ai.response.finish_reason", str(finish_reason or "unknown"))
            span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
            span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
            span.set_attribute("openai.total_tokens", int(total_tokens))
            span.set_attribute("llm.provider", "lmstudio")
            span.set_attribute("llm.api", "openai-compatible")
            span.set_attribute("inference.latency_sec", float(round(latency, 3)))
            span.set_attribute("output.answer_length", int(len(answer)))

            # MLflow-reserved keys: token usage (-> Tokens column) and span I/O.
            span.set_attribute(
                "mlflow.chat.tokenUsage",
                json_str({
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "total_tokens": int(total_tokens),
                }),
            )
            span.set_attribute(
                "mlflow.spanInputs",
                json_str({"model": model_name, "prompt_preview": clip_text(prompt, 500) if OTEL_CAPTURE_CONTENT else "[redacted]"}),
            )
            span.set_attribute(
                "mlflow.spanOutputs",
                json_str({"answer": clip_text(answer, 500) if OTEL_CAPTURE_CONTENT else "[redacted]", "finish_reason": str(finish_reason or "unknown")}),
            )

            span.add_event(
                "gen_ai.response.summary",
                {
                    "answer_preview": clip_text(answer, 300) if OTEL_CAPTURE_CONTENT else "[redacted]",
                    "finish_reason": str(finish_reason or "unknown"),
                },
            )

            # GenAI metrics: token usage (by type) and operation duration.
            _TOKEN_USAGE.record(int(input_tokens), {**metric_attrs, "gen_ai.token.type": "input"})
            _TOKEN_USAGE.record(int(output_tokens), {**metric_attrs, "gen_ai.token.type": "output"})
            _OP_DURATION.record(latency, metric_attrs)

            set_ok(span)
            return answer, model_name

        except Exception as exc:
            _OP_DURATION.record(
                time.perf_counter() - t0,
                {**metric_attrs, "error.type": exc.__class__.__name__},
            )
            set_error(span, exc)
            raise


def rag_mock(query: str, user_id: str = DEFAULT_USER_ID, session_id: str = DEFAULT_SESSION_ID):
    root_attrs = {
        "app.user_id": user_id,
        "app.session_id": session_id,
        "gen_ai.conversation.id": session_id,
        # Mirror the scorer's trace-metadata keys as root-span attributes, since
        # this variant has no MLflow SDK to call update_current_trace().
        "mlflow.trace.user": user_id,
        "mlflow.trace.session": session_id,
        "trace.use_case": TRACE_USE_CASE,
        "trace.app_version": TRACE_APP_VERSION,
    }

    if TRACE_CAPTURE_QUERY:
        root_attrs["input.query"] = clip_text(query)

    # Root span inputs drive the trace-level Request preview in the MLflow UI.
    root_attrs["mlflow.spanInputs"] = json_str(
        {"query": clip_text(query) if TRACE_CAPTURE_QUERY else "[redacted]"}
    )

    with _TRACER.start_as_current_span(
        "rag_request.otel", kind=SpanKind.INTERNAL, attributes=root_attrs
    ) as span:
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

            span.set_attribute("rag.retrieved_doc_count", int(len(docs)))
            span.set_attribute("rag.model", str(model_name))
            span.set_attribute("output.model", str(model_name))
            span.set_attribute("output.answer_length", int(len(answer)))
            span.set_attribute("output.status", "ok")
            span.set_attribute("retrieval.hit_ids", json_str([d["id"] for d in docs]))

            # Root span outputs drive the trace-level Response preview.
            span.set_attribute(
                "mlflow.spanOutputs",
                json_str({
                    "model": model_name,
                    "answer": clip_text(answer) if TRACE_CAPTURE_OUTPUT else "[redacted]",
                    "retrieved_doc_count": len(docs),
                }),
            )

            if TRACE_CAPTURE_OUTPUT:
                span.set_attribute("output.answer", clip_text(answer))

            set_ok(span)
            return result

        except Exception as exc:
            span.set_attribute("output.status", "error")
            set_error(span, exc)
            raise


if __name__ == "__main__":
    experiment_id, tracer_provider, meter_provider = setup_telemetry()

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
    print(f" OTEL_SERVICE_NAME={OTEL_SERVICE_NAME}")
    print(f" OTEL_CAPTURE_CONTENT={OTEL_CAPTURE_CONTENT}")
    print(f" TRACE_USE_CASE={TRACE_USE_CASE}")
    print(f" TRACE_APP_VERSION={TRACE_APP_VERSION}")
    print(f" TRACE_CAPTURE_QUERY={TRACE_CAPTURE_QUERY}")
    print(f" TRACE_CAPTURE_OUTPUT={TRACE_CAPTURE_OUTPUT}")
    print()

    try:
        while True:
            user_query = input("You: ").strip()

            if user_query.lower() in ["quit", "exit", "q"]:
                print("Goodbye.")
                break

            if not user_query:
                print("Please enter a question.\n")
                continue

            try:
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
            except Exception as exc:
                # A failed request (e.g. LLM down) should not kill the REPL.
                print(f"\nRequest failed: {exc}\n")
            finally:
                # Export this request's spans immediately instead of waiting for
                # the BatchSpanProcessor timer or a clean shutdown.
                tracer_provider.force_flush()
    finally:
        # Flush spans and metrics before exit (BatchSpanProcessor + periodic reader).
        tracer_provider.shutdown()
        meter_provider.shutdown()
