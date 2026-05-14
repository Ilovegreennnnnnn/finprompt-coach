from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.dataset import get_case_by_id, load_cases

from app.tools import run_tool
from app.evaluators import evaluate_response
from app.agent import run_weak_agent
from app.runner import compare_prompt_versions, run_evaluation_suite
from app.prompt_coach import improve_prompt
from app.tracing import setup_tracing, trace_span
from app.demo import build_demo_summary



app = FastAPI(
    title="FinPrompt Coach API",
    description="Arize-powered prompt optimization backend for financial agents.",
    version="0.1.0",
)

setup_tracing()


class HealthResponse(BaseModel):
    status: str
    project: str
    purpose: str

class EvaluationRequest(BaseModel):
    response_text: str
    tool_used: str | None = None

class RunCaseRequest(BaseModel):
    prompt: str | None = None


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

@app.post("/evaluate/{case_id}")
def evaluate_case(case_id: str, payload: EvaluationRequest) -> dict[str, Any]:
    case = get_case_by_id(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    return evaluate_response(
        case=case,
        response_text=payload.response_text,
        tool_used=payload.tool_used,
    )


@app.post("/run-case/{case_id}")
def run_case(case_id: str, payload: RunCaseRequest) -> dict[str, Any]:
    case = get_case_by_id(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    agent_run = run_weak_agent(
        case=case,
        prompt=payload.prompt,
    )

    evaluation = evaluate_response(
        case=case,
        response_text=agent_run["response_text"],
        tool_used=agent_run["tool_used"],
    )

    return {
        "case_id": case_id,
        "agent_version": "weak_simulated_v1",
        "case": case,
        "agent_run": agent_run,
        "evaluation": evaluation,
    }


@app.post("/run-suite")
def run_suite(payload: RunCaseRequest) -> dict[str, Any]:
    cases = load_cases()

    suite_result = run_evaluation_suite(
        cases=cases,
        prompt=payload.prompt,
    )

    return {
        "agent_version": "weak_simulated_v1",
        "prompt": payload.prompt,
        "suite_result": suite_result,
    }


@app.post("/improve-prompt")
def improve_prompt_endpoint(payload: RunCaseRequest) -> dict[str, Any]:
    cases = load_cases()

    suite_result = run_evaluation_suite(
        cases=cases,
        prompt=payload.prompt,
    )

    coach_result = improve_prompt(
        original_prompt=payload.prompt,
        suite_result=suite_result,
    )

    return {
        "agent_version": "weak_simulated_v1",
        "suite_result": suite_result,
        "coach_result": coach_result,
    }


@app.post("/experiment")
def run_experiment(payload: RunCaseRequest) -> dict[str, Any]:
    with trace_span(
        "experiment.prompt_v1_vs_prompt_v2",
        {
            "experiment.name": "prompt_v1_vs_prompt_v2",
            "project.name": "finprompt-coach",
        },
    ):
        cases = load_cases()

        prompt_v1 = payload.prompt or "You are a helpful finance assistant. Answer clearly."

        with trace_span(
            "experiment.run_v1_suite",
            {
                "agent.version": "weak",
                "prompt.version": "v1",
                "dataset.size": len(cases),
            },
        ):
            v1_suite_result = run_evaluation_suite(
                cases=cases,
                prompt=prompt_v1,
                agent_version="weak",
            )

        with trace_span(
            "experiment.prompt_coach",
            {
                "prompt.version.input": "v1",
                "prompt.version.output": "v2",
                "v1.overall_score": v1_suite_result["overall_score"],
                "v1.passed_cases": v1_suite_result["passed_cases"],
                "v1.failed_cases": v1_suite_result["failed_cases"],
            },
        ):
            coach_result = improve_prompt(
                original_prompt=prompt_v1,
                suite_result=v1_suite_result,
            )

        prompt_v2 = coach_result["improved_prompt"]

        with trace_span(
            "experiment.compare_v1_v2",
            {
                "prompt.version.a": "v1",
                "prompt.version.b": "v2",
            },
        ):
            experiment_result = compare_prompt_versions(
                cases=cases,
                prompt_v1=prompt_v1,
                prompt_v2=prompt_v2,
            )

        return {
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "coach_result": coach_result,
            "experiment_result": experiment_result,
        }

@app.post("/demo-summary")
def demo_summary(payload: RunCaseRequest) -> dict[str, Any]:
    with trace_span(
        "demo.summary",
        {
            "project.name": "finprompt-coach",
            "demo.type": "compact_summary",
        },
    ):
        cases = load_cases()

        prompt_v1 = payload.prompt or "You are a helpful finance assistant. Answer clearly."

        v1_suite_result = run_evaluation_suite(
            cases=cases,
            prompt=prompt_v1,
            agent_version="weak",
        )

        coach_result = improve_prompt(
            original_prompt=prompt_v1,
            suite_result=v1_suite_result,
        )

        prompt_v2 = coach_result["improved_prompt"]

        experiment_result = compare_prompt_versions(
            cases=cases,
            prompt_v1=prompt_v1,
            prompt_v2=prompt_v2,
        )

        return build_demo_summary(
            prompt_v1=prompt_v1,
            prompt_v2=prompt_v2,
            coach_result=coach_result,
            experiment_result=experiment_result,
        )