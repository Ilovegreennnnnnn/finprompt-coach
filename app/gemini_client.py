import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.tracing import trace_span


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


def generate_gemini_text(
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Generate text with Gemini on Vertex AI using ADC.
    Also creates a Phoenix span around the model call.
    """
    selected_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = get_gemini_client()

    with trace_span(
        "gemini.generate_content",
        {
            "llm.provider": "google",
            "llm.platform": "vertex_ai",
            "llm.model": selected_model,
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

        span.set_attribute("llm.response.length", len(text))

        return text