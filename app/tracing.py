import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv
from opentelemetry import trace
from phoenix.otel import register


_TRACING_INITIALIZED = False


def setup_tracing() -> None:
    """
    Initialize Phoenix/OpenTelemetry tracing.
    """
    global _TRACING_INITIALIZED

    if _TRACING_INITIALIZED:
        return

    load_dotenv()

    phoenix_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        "http://localhost:6006/v1/traces",
    )

    project_name = os.getenv(
        "PHOENIX_PROJECT_NAME",
        "finprompt-coach",
    )

    register(
        endpoint=phoenix_endpoint,
        project_name=project_name,
        protocol="http/protobuf",
    )

    _TRACING_INITIALIZED = True


def get_tracer():
    return trace.get_tracer("finprompt-coach")


def _clean_span_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value

    if value is None:
        return ""

    if isinstance(value, Mapping):
        return str(dict(value))

    if isinstance(value, (list, tuple, set)):
        return str(list(value))

    return str(value)


def set_span_attributes(span: Any, attributes: dict[str, Any] | None = None) -> None:
    if not attributes:
        return

    for key, value in attributes.items():
        if value is None:
            continue

        span.set_attribute(key, _clean_span_value(value))


def current_trace_id() -> str | None:
    span = trace.get_current_span()
    span_context = span.get_span_context()

    if not span_context or not span_context.trace_id:
        return None

    return format(span_context.trace_id, "032x")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """
    Small helper for manual tracing.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        set_span_attributes(span, attributes)

        yield span
