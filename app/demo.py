from typing import Any


def summarize_failure_counts(failure_counts: dict[str, int]) -> list[dict[str, Any]]:
    """
    Converts failure counts into a sorted list for UI/demo display.
    """
    return [
        {
            "evaluator": name,
            "failures": count,
        }
        for name, count in sorted(
            failure_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def find_sample_failure(v1_result: dict[str, Any]) -> dict[str, Any] | None:
    """
    Finds one clear failed v1 case for the demo.
    """
    for result in v1_result.get("results", []):
        evaluation = result.get("evaluation", {})

        if not evaluation.get("passed", True):
            failed_evaluators = [
                item["name"]
                for item in evaluation.get("evaluations", [])
                if not item.get("passed", False)
            ]

            return {
                "case_id": result.get("case_id"),
                "title": result.get("title"),
                "response_text": result.get("agent_run", {}).get("response_text"),
                "tool_used": result.get("agent_run", {}).get("tool_used"),
                "overall_score": evaluation.get("overall_score"),
                "failed_evaluators": failed_evaluators,
            }

    return None


def find_matching_v2_case(
    v2_result: dict[str, Any],
    case_id: str | None,
) -> dict[str, Any] | None:
    """
    Finds the improved v2 response for the same case as the sample v1 failure.
    """
    if case_id is None:
        return None

    for result in v2_result.get("results", []):
        if result.get("case_id") == case_id:
            evaluation = result.get("evaluation", {})

            return {
                "case_id": result.get("case_id"),
                "title": result.get("title"),
                "response_text": result.get("agent_run", {}).get("response_text"),
                "tool_used": result.get("agent_run", {}).get("tool_used"),
                "overall_score": evaluation.get("overall_score"),
                "passed": evaluation.get("passed"),
            }

    return None


def build_demo_summary(
    prompt_v1: str,
    prompt_v2: str,
    coach_result: dict[str, Any],
    experiment_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Builds a compact demo-friendly summary from the full experiment result.
    """
    v1_result = experiment_result["v1_result"]
    v2_result = experiment_result["v2_result"]

    sample_failure = find_sample_failure(v1_result)

    sample_improved_response = find_matching_v2_case(
        v2_result=v2_result,
        case_id=sample_failure["case_id"] if sample_failure else None,
    )

    return {
        "project": "FinPrompt Coach",
        "positioning": "Arize-powered prompt optimization for financial agents. Not financial advice.",
        "core_loop": [
            "Prompt",
            "Run",
            "Trace",
            "Evaluate",
            "Detect Failure",
            "Improve Prompt",
            "Experiment",
            "Compare",
        ],
        "prompt_v1": prompt_v1,
        "prompt_v2": prompt_v2,
        "scores": {
            "prompt_v1_score": experiment_result["prompt_v1_score"],
            "prompt_v2_score": experiment_result["prompt_v2_score"],
            "improvement": experiment_result["improvement"],
            "v1_passed_cases": v1_result["passed_cases"],
            "v1_failed_cases": v1_result["failed_cases"],
            "v2_passed_cases": v2_result["passed_cases"],
            "v2_failed_cases": v2_result["failed_cases"],
            "total_cases": v1_result["total_cases"],
        },
        "failure_analysis": summarize_failure_counts(
            coach_result.get("failure_counts", {})
        ),
        "prompt_patch": coach_result.get("prompt_patch", []),
        "sample_failure": sample_failure,
        "sample_improved_response": sample_improved_response,
        "demo_message": "FinPrompt Coach turns financial agent failures into measurable prompt improvements backed by Phoenix traces, evals, and experiments.",
    }