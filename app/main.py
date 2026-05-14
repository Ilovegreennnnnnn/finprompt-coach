from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.dataset import get_case_by_id, load_cases

from app.tools import run_tool


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


@app.get("/cases")
def list_cases() -> dict[str, Any]:
    cases = load_cases()

    return {
        "count": len(cases),
        "cases": cases,
    }


@app.get("/cases/{case_id}")
def read_case(case_id: str) -> dict[str, Any]:
    case = get_case_by_id(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    return case


@app.post("/tools/{tool_name}")
def test_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    provided_context = payload.get("provided_context", {})

    return run_tool(
        tool_name=tool_name,
        provided_context=provided_context,
    )