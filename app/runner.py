from typing import Any

from app.agent import run_improved_agent, run_weak_agent
from app.evaluators import evaluate_response


def run_evaluation_suite(
    cases: list[dict[str, Any]],
    prompt: str | None = None,
    agent_version: str = "weak",
) -> dict[str, Any]:
    """
    Runs an agent on every case and evaluates each response.
    """
    results = []

    for case in cases:
        if agent_version == "improved":
            agent_run = run_improved_agent(
                case=case,
                prompt=prompt,
            )
        else:
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
        "agent_version": agent_version,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "overall_score": round(overall_score, 2),
        "results": results,
    }


def compare_prompt_versions(
    cases: list[dict[str, Any]],
    prompt_v1: str,
    prompt_v2: str,
) -> dict[str, Any]:
    """
    Runs an experiment comparing weak v1 behavior against improved v2 behavior.
    """
    v1_result = run_evaluation_suite(
        cases=cases,
        prompt=prompt_v1,
        agent_version="weak",
    )

    v2_result = run_evaluation_suite(
        cases=cases,
        prompt=prompt_v2,
        agent_version="improved",
    )

    improvement = round(
        v2_result["overall_score"] - v1_result["overall_score"],
        2,
    )

    return {
        "experiment_name": "prompt_v1_vs_prompt_v2",
        "prompt_v1_score": v1_result["overall_score"],
        "prompt_v2_score": v2_result["overall_score"],
        "improvement": improvement,
        "v1_result": v1_result,
        "v2_result": v2_result,
    }