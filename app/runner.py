from typing import Any

from app.agent import run_weak_agent
from app.evaluators import evaluate_response


def run_evaluation_suite(
    cases: list[dict[str, Any]],
    prompt: str | None = None,
) -> dict[str, Any]:
    """
    Runs the weak simulated agent on every case and evaluates each response.
    """
    results = []

    for case in cases:
        agent_run = run_weak_agent(
            case=case,
            prompt=prompt,
        )

        evaluation = evaluate_response(
            case=case,
            response_text=agent_run["response_text"],
            tool_used=agent_run["tool_used"],
        )

        results.append(
            {
                "case_id": case.get("id"),
                "title": case.get("title"),
                "agent_run": agent_run,
                "evaluation": evaluation,
            }
        )

    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["evaluation"]["passed"])
    failed_cases = total_cases - passed_cases

    if total_cases == 0:
        overall_score = 0.0
    else:
        overall_score = sum(
            result["evaluation"]["overall_score"] for result in results
        ) / total_cases

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "overall_score": round(overall_score, 2),
        "results": results,
    }