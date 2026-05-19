import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.tracing import set_span_attributes, trace_span


load_dotenv()


def get_gemini_client() -> genai.Client:
    """
    Creates a Gemini client through Vertex AI using Application Default Credentials.

    No GOOGLE_API_KEY is used here.
    Authentication is handled by ADC:
    - local: gcloud auth application-default login
    - cloud: attached service account
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is missing. Add it to your .env file."
        )

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )


def _extract_usage_metadata(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    for attr_name in ["prompt_token_count", "promptTokenCount"]:
        value = getattr(usage, attr_name, None) if usage is not None else None
        if value is not None:
            prompt_tokens = int(value)
            break

    for attr_name in ["candidates_token_count", "candidatesTokenCount"]:
        value = getattr(usage, attr_name, None) if usage is not None else None
        if value is not None:
            completion_tokens = int(value)
            break

    for attr_name in ["total_token_count", "totalTokenCount"]:
        value = getattr(usage, attr_name, None) if usage is not None else None
        if value is not None:
            total_tokens = int(value)
            break

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _safe_float_from_env(env_name: str) -> float | None:
    value = os.getenv(env_name)
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def estimate_gemini_cost(
    usage: dict[str, int | None],
    *,
    model_name: str,
) -> dict[str, Any]:
    prompt_rate = _safe_float_from_env("GEMINI_INPUT_COST_PER_1M_TOKENS")
    completion_rate = _safe_float_from_env("GEMINI_OUTPUT_COST_PER_1M_TOKENS")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    if prompt_rate is None or completion_rate is None or prompt_tokens is None or completion_tokens is None:
        return {
            "available": False,
            "model": model_name,
            "input_cost_per_1m_tokens": prompt_rate,
            "output_cost_per_1m_tokens": completion_rate,
            "prompt_cost_usd": None,
            "completion_cost_usd": None,
            "total_cost_usd": None,
            "note": "Phoenix can still derive costs when token counts and model information are traced.",
        }

    prompt_cost = round((prompt_tokens / 1_000_000) * prompt_rate, 8)
    completion_cost = round((completion_tokens / 1_000_000) * completion_rate, 8)

    return {
        "available": True,
        "model": model_name,
        "input_cost_per_1m_tokens": prompt_rate,
        "output_cost_per_1m_tokens": completion_rate,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": round(prompt_cost + completion_cost, 8),
        "note": "Estimated locally from configured per-million-token pricing.",
    }


def generate_gemini_response(
    prompt: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generate Gemini text plus usage and cost metadata.
    """
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = get_gemini_client()

    with trace_span(
        "gemini.generate_content",
        {
            "llm.provider": "google",
            "llm.model_name": selected_model,
            "llm.model": selected_model,
            "llm.platform": "vertex_ai",
            "cloud.project": project,
            "cloud.location": location,
            "auth.method": "application_default_credentials",
        },
    ) as span:
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
        )

        text = response.text or ""
        usage = _extract_usage_metadata(response)
        cost_summary = estimate_gemini_cost(usage, model_name=selected_model)
        response_id = getattr(response, "response_id", None) or getattr(response, "responseId", None)
        model_version = getattr(response, "model_version", None) or getattr(response, "modelVersion", None)

        set_span_attributes(
            span,
            {
                "llm.response.length": len(text),
                "llm.token_count.prompt": usage.get("prompt_tokens"),
                "llm.token_count.completion": usage.get("completion_tokens"),
                "llm.token_count.total": usage.get("total_tokens"),
                "llm.usage.prompt_tokens": usage.get("prompt_tokens"),
                "llm.usage.completion_tokens": usage.get("completion_tokens"),
                "llm.usage.total_tokens": usage.get("total_tokens"),
                "llm.response_id": response_id,
                "llm.model_version": model_version,
                "llm.cost.usd": cost_summary.get("total_cost_usd"),
            },
        )

        return {
            "text": text,
            "model": selected_model,
            "model_version": model_version,
            "response_id": response_id,
            "usage": usage,
            "cost_summary": cost_summary,
        }


def generate_gemini_text(
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Generate text with Gemini on Vertex AI using ADC.
    Also creates a Phoenix span around the model call.
    """
    return generate_gemini_response(prompt, model=model)["text"]
