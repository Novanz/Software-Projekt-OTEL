import os
from typing import Any, Optional

import mlflow
from mlflow.entities import Feedback, AssessmentSource
from mlflow.genai.scorers import scorer

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", "").strip()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "evaluation_v1").strip()


def _safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _trace_metadata(trace) -> dict:
    if trace is None:
        return {}
    info = _safe_getattr(trace, "info", None)
    if info is None:
        return {}
    meta = _safe_getattr(info, "trace_metadata", None)
    if meta is None:
        meta = _safe_getattr(info, "metadata", None)
    return meta or {}


def _all_spans(trace) -> list:
    if trace is None:
        return []
    try:
        spans = trace.search_spans()
        return list(spans or [])
    except Exception:
        pass

    data = _safe_getattr(trace, "data", None)
    spans = _safe_getattr(data, "spans", None)
    return list(spans or [])


def _span_name(span) -> str:
    return str(_safe_getattr(span, "name", "") or "")


def _span_attrs(span) -> dict:
    attrs = _safe_getattr(span, "attributes", None)
    return attrs or {}


def _find_first_span(trace, *candidate_names: str):
    names = set(candidate_names)
    for span in _all_spans(trace):
        if _span_name(span) in names:
            return span
    return None


def _get_query(inputs: Optional[dict], trace) -> str:
    if isinstance(inputs, dict):
        query = inputs.get("query") or inputs.get("input.query")
        if query:
            return str(query)

    meta = _trace_metadata(trace)
    for key in ("input.query", "query"):
        if meta.get(key):
            return str(meta[key])

    root = _find_first_span(trace, "rag_request.otel", "rag_mock")
    if root:
        attrs = _span_attrs(root)
        if attrs.get("input.query"):
            return str(attrs["input.query"])

    return ""


def _get_answer(outputs: Any, trace) -> str:
    if isinstance(outputs, dict):
        answer = outputs.get("answer") or outputs.get("output.answer")
        if answer:
            return str(answer)

    if isinstance(outputs, str):
        return outputs

    root = _find_first_span(trace, "rag_request.otel", "rag_mock")
    if root:
        attrs = _span_attrs(root)
        if attrs.get("output.answer"):
            return str(attrs["output.answer"])

    return ""


def _has_any_attr(span, keys: list[str]) -> bool:
    if span is None:
        return False
    attrs = _span_attrs(span)
    return any(k in attrs for k in keys)


def _source() -> AssessmentSource:
    return AssessmentSource(source_type="CODE", source_id="trace_schema_health_v1")


@scorer
def trace_schema_health(
    *,
    inputs: Optional[dict[str, Any]] = None,
    outputs: Optional[Any] = None,
    expectations: Optional[dict[str, Any]] = None,
    trace=None,
):
    spans = _all_spans(trace)
    span_names = {_span_name(s) for s in spans}
    meta = _trace_metadata(trace)

    query = _get_query(inputs, trace)
    answer = _get_answer(outputs, trace)

    retrieval_span = _find_first_span(trace, "chroma_retrieval", "chroma_retrieval.otel")
    prompt_span = _find_first_span(trace, "prompt_assembly", "prompt_assembly.otel")
    inference_span = _find_first_span(trace, "lmstudio_inference", "lmstudio_inference.otel")
    root_span = _find_first_span(trace, "rag_request.otel", "rag_mock")

    has_query = bool(query)
    has_retrieval_span = retrieval_span is not None or any("chroma_retrieval" in n for n in span_names)
    has_prompt_span = prompt_span is not None or any("prompt_assembly" in n for n in span_names)
    has_inference_span = inference_span is not None or any("lmstudio_inference" in n for n in span_names)
    has_output_answer = bool(answer)

    # The MLflow variant sets these via update_current_trace (-> trace_metadata).
    # The pure-OTel variant exports over OTLP, where MLflow does not promote span
    # attributes to trace metadata, so fall back to the root-span attributes.
    root_attrs = _span_attrs(root_span)

    def _meta(key: str):
        return meta.get(key) or root_attrs.get(key)

    metadata_complete = all(
        [
            bool(_meta("mlflow.trace.user")),
            bool(_meta("mlflow.trace.session")),
            bool(_meta("trace.use_case")),
            bool(_meta("trace.app_version")),
        ]
    )

    retrieval_runtime_complete = (
        _has_any_attr(retrieval_span, ["retrieval.hit_ids"])
        and _has_any_attr(retrieval_span, ["retrieval.hit_count"])
        and _has_any_attr(retrieval_span, ["retrieval.collection"])
    )

    inference_runtime_complete = (
        _has_any_attr(inference_span, ["gen_ai.request.model"])
        and _has_any_attr(inference_span, ["gen_ai.usage.input_tokens"])
        and _has_any_attr(inference_span, ["gen_ai.usage.output_tokens"])
        and _has_any_attr(inference_span, ["inference.latency_sec"])
    )

    root_runtime_complete = (
        _has_any_attr(root_span, ["output.model"])
        and _has_any_attr(root_span, ["output.answer_length"])
        and _has_any_attr(root_span, ["output.status"])
    )

    runtime_complete = retrieval_runtime_complete and inference_runtime_complete and root_runtime_complete

    ready = all(
        [
            has_query,
            has_retrieval_span,
            has_prompt_span,
            has_inference_span,
            has_output_answer,
            metadata_complete,
            runtime_complete,
        ]
    )

    return [
        Feedback(
            name="trace_has_query",
            value=has_query,
            rationale="Query found in inputs or root trace attributes." if has_query else "No query found in inputs or root trace attributes.",
            source=_source(),
        ),
        Feedback(
            name="trace_has_retrieval_span",
            value=has_retrieval_span,
            rationale="Retrieval span is present." if has_retrieval_span else "Retrieval span is missing.",
            source=_source(),
        ),
        Feedback(
            name="trace_has_prompt_span",
            value=has_prompt_span,
            rationale="Prompt assembly span is present." if has_prompt_span else "Prompt assembly span is missing.",
            source=_source(),
        ),
        Feedback(
            name="trace_has_inference_span",
            value=has_inference_span,
            rationale="Inference span is present." if has_inference_span else "Inference span is missing.",
            source=_source(),
        ),
        Feedback(
            name="trace_has_output_answer",
            value=has_output_answer,
            rationale="Final answer found in outputs or root trace attributes." if has_output_answer else "Final answer is missing.",
            source=_source(),
        ),
        Feedback(
            name="trace_metadata_complete",
            value=metadata_complete,
            rationale=(
                "Trace metadata contains user, session, use case, and app version."
                if metadata_complete
                else "Trace metadata is missing one or more of: user, session, use case, app version."
            ),
            source=_source(),
        ),
        Feedback(
            name="trace_runtime_complete",
            value=runtime_complete,
            rationale=(
                "Retrieval, inference, and root runtime attributes are present."
                if runtime_complete
                else "One or more runtime attribute groups are missing."
            ),
            source=_source(),
        ),
        Feedback(
            name="trace_observability_ready",
            value=ready,
            rationale=(
                "Trace contains the minimum structure and attributes needed for trace-based evaluation."
                if ready
                else "Trace is not yet complete enough for reliable trace-based evaluation."
            ),
            source=_source(),
        ),
    ]


def _resolve_experiment_id() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_EXPERIMENT_ID:
        exp = mlflow.get_experiment(MLFLOW_EXPERIMENT_ID)
        if exp is None:
            raise RuntimeError(
                f"MLFLOW_EXPERIMENT_ID={MLFLOW_EXPERIMENT_ID} was not found on {MLFLOW_TRACKING_URI}"
            )
        return str(exp.experiment_id)

    exp = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return str(exp.experiment_id)


def main():
    experiment_id = _resolve_experiment_id()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Using experiment_id={experiment_id}")

    # Replace this block with your trace-loading call if your local MLflow version
    # exposes a different retrieval API.
    traces = mlflow.search_traces(experiment_ids=[experiment_id])

    result = mlflow.genai.evaluate(
        data=traces,
        scorers=[trace_schema_health],
    )

    print("Evaluation finished.")
    print(result)


if __name__ == "__main__":
    main()
