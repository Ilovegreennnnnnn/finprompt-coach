from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.dataset import get_case_by_id, load_cases

from app.tools import run_tool
from app.evaluators import evaluate_response
from app.agent import run_gemini_agent, run_weak_agent
from app.runner import (
    compare_gemini_prompt_versions,
    compare_prompt_versions,
    run_evaluation_suite,
)
from app.prompt_coach import improve_prompt
from app.tracing import setup_tracing, trace_span
from app.demo import build_demo_summary
from fastapi.staticfiles import StaticFiles
from app.gemini_client import generate_gemini_text




app = FastAPI(
    title="FinPrompt Coach API",
    description="Arize-powered prompt optimization backend for financial agents.",
    version="0.1.0",
)

setup_tracing()
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")

class HealthResponse(BaseModel):
    status: str
    project: str
    purpose: str

class EvaluationRequest(BaseModel):
    response_text: str
    tool_used: str | None = None

class RunCaseRequest(BaseModel):
    prompt: str | None = None

class GeminiTestRequest(BaseModel):
    prompt: str

class PromptRequest(BaseModel):
    prompt: str

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

@app.post("/gemini-test")
def gemini_test(payload: GeminiTestRequest) -> dict[str, Any]:
    with trace_span(
        "gemini.test",
        {
            "project.name": "finprompt-coach",
            "auth.method": "application_default_credentials",
        },
    ):
        response_text = generate_gemini_text(payload.prompt)

        return {
            "provider": "google",
            "platform": "vertex_ai",
            "auth": "application_default_credentials",
            "response_text": response_text,
        }

@app.post("/run-gemini-case/{case_id}")
def run_gemini_case(case_id: str, payload: RunCaseRequest) -> dict[str, Any]:
    with trace_span(
        "case.gemini.run_and_evaluate",
        {
            "case.id": case_id,
            "agent.version": "gemini",
        },
    ):
        case = get_case_by_id(case_id)

        if case is None:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

        agent_run = run_gemini_agent(
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
            "agent_version": "gemini",
            "case": case,
            "agent_run": agent_run,
            "evaluation": evaluation,
        }


@app.post("/run-gemini-suite")
def run_gemini_suite(payload: RunCaseRequest) -> dict[str, Any]:
    with trace_span(
        "suite.gemini.run_and_evaluate",
        {
            "agent.version": "gemini",
            "project.name": "finprompt-coach",
        },
    ):
        cases = load_cases()

        suite_result = run_evaluation_suite(
            cases=cases,
            prompt=payload.prompt,
            agent_version="gemini",
        )

        return {
            "agent_version": "gemini",
            "prompt": payload.prompt,
            "suite_result": suite_result,
        }
def summarize_suite(result: dict) -> dict:
    return {
        "agent_version": result.get("agent_version"),
        "total_cases": result.get("total_cases"),
        "passed_cases": result.get("passed_cases"),
        "failed_cases": result.get("failed_cases"),
        "overall_score": result.get("overall_score"),
        "failed_case_summaries": [
            {
                "case_id": item.get("case_id"),
                "title": item.get("title"),
                "overall_score": item.get("evaluation", {}).get("overall_score"),
                "failed_evaluators": [
                    evaluator.get("name")
                    for evaluator in item.get("evaluation", {}).get("evaluations", [])
                    if not evaluator.get("passed")
                ],
            }
            for item in result.get("results", [])
            if not item.get("evaluation", {}).get("passed")
        ],
    }


@app.post("/gemini-experiment")
def run_gemini_experiment(payload: PromptRequest) -> dict[str, Any]:
    with trace_span(
        "experiment.gemini_prompt_v1_vs_prompt_v2",
        {
            "experiment.name": "gemini_prompt_v1_vs_prompt_v2",
            "project.name": "finprompt-coach",
            "agent.version": "gemini",
        },
    ):
        cases = load_cases()

        prompt_v1 = payload.prompt or "You are a helpful finance assistant. Answer clearly."

        v1_suite_result = run_evaluation_suite(
            cases=cases,
            prompt=prompt_v1,
            agent_version="gemini",
        )

        coach_result = improve_prompt(
            original_prompt=prompt_v1,
            suite_result=v1_suite_result,
        )

        prompt_v2 = coach_result["improved_prompt"]

        v2_suite_result = run_evaluation_suite(
            cases=cases,
            prompt=prompt_v2,
            agent_version="gemini",
        )

        experiment_result = {
            "experiment_name": "gemini_prompt_v1_vs_prompt_v2",
            "agent_version": "gemini",
            "prompt_v1_score": v1_suite_result["overall_score"],
            "prompt_v2_score": v2_suite_result["overall_score"],
            "improvement": round(
                v2_suite_result["overall_score"] - v1_suite_result["overall_score"],
                2,
            ),
            "v1_summary": summarize_suite(v1_suite_result),
            "v2_summary": summarize_suite(v2_suite_result),
        }

        return {
            "agent_version": "gemini",
            "v1_summary": summarize_suite(v1_suite_result),
            "v2_summary": summarize_suite(v2_suite_result),
            "coach_result": coach_result,
            "experiment_result": experiment_result,
        }