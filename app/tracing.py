import os
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


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """
    Small helper for manual tracing.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue

                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                else:
                    span.set_attribute(key, str(value))

        yield span