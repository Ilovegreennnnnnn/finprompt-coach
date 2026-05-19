from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.dataset import get_case_by_id, load_cases
from app.dataset import get_benchmark_case_by_id
from app.dataset import list_benchmarks
from app.tools import run_tool
from app.evaluators import evaluate_response
from app.agent import run_gemini_agent, run_weak_agent
from app.runner import (
    compare_prompt_versions,
    run_evaluation_suite,
)
from app.prompt_coach import improve_prompt
from app.tracing import setup_tracing, trace_span
from app.demo import build_demo_summary
from app.gemini_client import generate_gemini_text
from app.explorer_service import (
    create_watchlist,
    get_analysis_dossier_detail,
    get_audit_graph_detail,
    get_lab_queue_items,
    get_market_research_run_detail,
    get_phoenix_overview,
    get_phoenix_prompt_state,
    list_market_research_runs,
    list_watchlists,
    manual_prompt_rollback,
    perform_market_research_run,
    rerun_introspection,
)
from app.lab_service import run_prompt_lab
from app.live_service import get_trace_summary, perform_live_analysis, promote_trace
from app.live_tools import SUPPORTED_LIVE_TOOLS, run_live_tool
from app.research_scheduler import phoenix_daily_scheduler
from app.research_state import (
    export_universal_data_bundle,
    get_universal_collection,
    get_universal_record,
    list_universal_collections,
)


app = FastAPI(
    title="FinPrompt Coach API",
    description="Arize-powered prompt optimization backend for financial agents.",
    version="0.1.0",
)

setup_tracing()
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


@app.on_event("startup")
def startup_event() -> None:
    phoenix_daily_scheduler.start()


@app.on_event("shutdown")
def shutdown_event() -> None:
    phoenix_daily_scheduler.stop()

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


class LiveAnalyzeRequest(BaseModel):
    message: str
    tickers: list[str] | None = None
    conversation_id: str | None = None


class LabRunRequest(BaseModel):
    prompt: str
    benchmark_id: str | None = None
    max_rounds: int = 3


class PromoteTraceRequest(BaseModel):
    trace_id: str


class WatchlistCreateRequest(BaseModel):
    name: str
    tickers: list[str]
    description: str | None = None
    schedule_enabled: bool = True


class PromptRollbackRequest(BaseModel):
    reason: str | None = None


class DataExportRequest(BaseModel):
    collections: list[str] | None = None
    export_format: str = "json"


def summarize_suite(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_version": result.get("agent_version"),
        "total_cases": result.get("total_cases"),
        "passed_cases": result.get("passed_cases"),
        "failed_cases": result.get("failed_cases"),
        "overall_score": result.get("overall_score"),
        "average_latency_ms": result.get("average_latency_ms"),
        "average_cost_usd": result.get("average_cost_usd"),
        "tool_usage_breakdown": result.get("tool_usage_breakdown", {}),
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


def summarize_coach_result(coach_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "failure_counts": coach_result.get("failure_counts", {}),
        "failed_cases": coach_result.get("failed_cases", []),
        "prompt_patch": coach_result.get("prompt_patch", []),
    }

@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/", status_code=307)


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
        coach_summary = summarize_coach_result(coach_result)

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
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "coach_summary": coach_summary,
            "coach_result": coach_result,
            "experiment_result": experiment_result,
        }


@app.post("/live/analyze")
def live_analyze(payload: LiveAnalyzeRequest) -> dict[str, Any]:
    with trace_span(
        "live.api.analyze",
        {
            "project.name": "finprompt-coach",
            "api.route": "/live/analyze",
        },
    ):
        return perform_live_analysis(
            message=payload.message,
            explicit_tickers=payload.tickers,
            conversation_id=payload.conversation_id,
        )


@app.post("/live/tools/{tool_name}")
def live_tool_debug(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in SUPPORTED_LIVE_TOOLS:
        raise HTTPException(status_code=404, detail=f"Unsupported live tool: {tool_name}")

    with trace_span(
        "live.api.tool_debug",
        {
            "project.name": "finprompt-coach",
            "api.route": "/live/tools/{tool_name}",
            "tool.name": tool_name,
        },
    ):
        return run_live_tool(tool_name, payload)


@app.get("/live/traces/{trace_id}/summary")
def live_trace_summary(trace_id: str) -> dict[str, Any]:
    summary = get_trace_summary(trace_id)

    if summary is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

    return summary


@app.get("/phoenix/overview")
def phoenix_overview() -> dict[str, Any]:
    overview = get_phoenix_overview()
    overview["scheduler"] = phoenix_daily_scheduler.status()
    return overview


@app.get("/phoenix/runs/{run_id}")
def phoenix_run(run_id: str) -> dict[str, Any]:
    detail = get_market_research_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Research run not found: {run_id}")
    return detail


@app.get("/phoenix/prompts")
def phoenix_prompts() -> dict[str, Any]:
    return get_phoenix_prompt_state()


@app.post("/phoenix/prompts/rollback")
def phoenix_prompt_rollback(payload: PromptRollbackRequest) -> dict[str, Any]:
    result = manual_prompt_rollback(payload.reason)
    if result is None:
        raise HTTPException(status_code=409, detail="No previous prompt is available for rollback.")
    return {
        "rolled_back_to": result,
    }


@app.get("/explorer/watchlists")
def explorer_watchlists() -> dict[str, Any]:
    return {
        "watchlists": list_watchlists(),
        "scheduler": phoenix_daily_scheduler.status(),
    }


@app.post("/explorer/watchlists")
def explorer_create_watchlist(payload: WatchlistCreateRequest) -> dict[str, Any]:
    try:
        watchlist = create_watchlist(
            name=payload.name,
            tickers=payload.tickers,
            description=payload.description,
            schedule_enabled=payload.schedule_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "watchlist": watchlist,
    }


@app.post("/explorer/watchlists/{watchlist_id}/run")
def explorer_run_watchlist(watchlist_id: str) -> dict[str, Any]:
    with trace_span(
        "explorer.api.run_watchlist",
        {
            "project.name": "finprompt-coach",
            "api.route": "/explorer/watchlists/{watchlist_id}/run",
            "explorer.watchlist_id": watchlist_id,
        },
    ):
        try:
            return perform_market_research_run(watchlist_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/explorer/runs")
def explorer_runs() -> dict[str, Any]:
    return {
        "runs": list_market_research_runs(),
    }


@app.get("/explorer/runs/{run_id}")
def explorer_run_detail(run_id: str) -> dict[str, Any]:
    detail = get_market_research_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Research run not found: {run_id}")
    return detail


@app.get("/explorer/runs/{run_id}/tickers/{ticker}/dossier")
def explorer_ticker_dossier(run_id: str, ticker: str) -> dict[str, Any]:
    detail = get_analysis_dossier_detail(run_id, ticker)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Analysis dossier not found for {ticker} in run {run_id}")
    return detail


@app.get("/explorer/runs/{run_id}/tickers/{ticker}/audit-graph")
def explorer_ticker_audit_graph(run_id: str, ticker: str) -> dict[str, Any]:
    detail = get_audit_graph_detail(run_id, ticker)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Audit graph not found for {ticker} in run {run_id}")
    return detail


@app.post("/explorer/runs/{run_id}/introspect")
def explorer_run_introspect(run_id: str) -> dict[str, Any]:
    with trace_span(
        "explorer.api.introspect_run",
        {
            "project.name": "finprompt-coach",
            "api.route": "/explorer/runs/{run_id}/introspect",
            "explorer.run_id": run_id,
        },
    ):
        try:
            return rerun_introspection(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/lab/benchmarks")
def lab_benchmarks() -> dict[str, Any]:
    return {
        "benchmarks": list_benchmarks(),
    }


@app.get("/lab/cases/{case_id}")
def lab_case(case_id: str) -> dict[str, Any]:
    case = get_benchmark_case_by_id(case_id)

    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    return case


@app.post("/lab/promote-trace")
def lab_promote_trace(payload: PromoteTraceRequest) -> dict[str, Any]:
    result = promote_trace(payload.trace_id)

    if result is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {payload.trace_id}")

    return result


@app.post("/lab/run")
def lab_run(payload: LabRunRequest) -> dict[str, Any]:
    with trace_span(
        "lab.run",
        {
            "project.name": "finprompt-coach",
            "api.route": "/lab/run",
            "benchmark.id": payload.benchmark_id or "all_equity_research",
            "lab.max_rounds": max(1, min(payload.max_rounds, 3)),
        },
    ):
        return run_prompt_lab(
            prompt=payload.prompt,
            benchmark_id=payload.benchmark_id,
            max_rounds=payload.max_rounds,
        )


@app.get("/lab/queue")
def lab_queue() -> dict[str, Any]:
    return {
        "queue": get_lab_queue_items(),
    }


@app.get("/data/collections")
def data_collections() -> dict[str, Any]:
    return {
        "collections": list_universal_collections(),
    }


@app.get("/data/{collection_name}")
def data_collection(collection_name: str, limit: int | None = None) -> dict[str, Any]:
    payload = get_universal_collection(collection_name, limit=limit)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown collection: {collection_name}")
    return payload


@app.get("/data/{collection_name}/{record_id}")
def data_record(collection_name: str, record_id: str) -> dict[str, Any]:
    payload = get_universal_record(collection_name, record_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Record not found in {collection_name}: {record_id}")
    return {
        "collection": collection_name,
        "record": payload,
    }


@app.post("/data/export")
def data_export(payload: DataExportRequest) -> dict[str, Any]:
    try:
        return export_universal_data_bundle(
            collections=payload.collections,
            export_format=payload.export_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
