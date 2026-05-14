from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="FinPrompt Coach API",
    description="Arize-powered prompt optimization backend for financial agents.",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    project: str
    purpose: str


@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        project="FinPrompt Coach",
        purpose="Prompt evaluation and optimization for financial agents. Not financial advice.",
    )


@app.get("/health")
def health():
    return {"status": "ok"}