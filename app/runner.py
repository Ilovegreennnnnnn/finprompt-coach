from typing import Any

from opentelemetry import trace

from app.agent import run_gemini_agent, run_improved_agent, run_weak_agent
from app.evaluators import evaluate_response
from app.tracing import set_span_attributes, trace_span


def run_evaluation_suite(
    cases: list[dict[str, Any]],
    prompt: str | None = None,
    agent_version: str = "weak",
) -> dict[str, Any]:
    """
    Runs an agent on every case and evaluates each response.
    Adds Phoenix/OpenTelemetry spans for the suite, each case, and each evaluation.
    """
    with trace_span(
        "suite.run",
        {
            "agent.version": agent_version,
            "dataset.size": len(cases),
        },
    ):
        results = []

        for case in cases:
            case_id = case.get("id")
            title = case.get("title")

            with trace_span(
                "case.run",
                {
                    "case.id": case_id,
                    "case.title": title,
                    "agent.version": agent_version,
                    "expected.tool": case.get("expected_tool"),
                    "expected.behavior": case.get("expected_behavior"),
                },
            ):
                if agent_version == "gemini":
                    agent_run = run_gemini_agent(
                        case=case,
                        prompt=prompt,
                    )
                elif agent_version == "improved":
                    agent_run = run_improved_agent(
                        case=case,
                        prompt=prompt,
                    )
                else:
                    agent_run = run_weak_agent(
                        case=case,
                        prompt=prompt,
                    )

                with trace_span(
                    "case.evaluate",
                    {
                        "case.id": case_id,
                        "case.title": title,
                        "tool.used": agent_run["tool_used"],
                        "agent.version": agent_version,
                    },
                ) as span:
                    evaluation = evaluate_response(
                        case=case,
                        response_text=agent_run["response_text"],
                        tool_used=agent_run["tool_used"],
                    )

                    span.set_attribute(
                        "evaluation.overall_score",
                        evaluation["overall_score"],
                    )
                    span.set_attribute(
                        "evaluation.passed",
                        evaluation["passed"],
                    )

                    for item in evaluation["evaluations"]:
                        evaluator_name = item["name"]

                        span.set_attribute(
                            f"evaluation.{evaluator_name}.score",
                            item["score"],
                        )
                        span.set_attribute(
                            f"evaluation.{evaluator_name}.passed",
                            item["passed"],
                        )

                results.append(
                    {
                        "case_id": case_id,
                        "title": title,
                        "agent_run": agent_run,
                        "evaluation": evaluation,
                    }
                )

        total_cases = len(results)
        passed_cases = sum(
            1 for result in results if result["evaluation"]["passed"]
        )
        failed_cases = total_cases - passed_cases
        latency_values = [
            float(result.get("agent_run", {}).get("latency_ms", 0.0) or 0.0)
            for result in results
        ]
        cost_values = [
            result.get("agent_run", {}).get("cost_summary", {}).get("total_cost_usd")
            for result in results
            if result.get("agent_run", {}).get("cost_summary", {}).get("total_cost_usd") is not None
        ]
        tool_breakdown: dict[str, int] = {}

        for result in results:
            tool_name = result.get("agent_run", {}).get("tool_used")
            if not tool_name:
                continue

            tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1

        if total_cases == 0:
            overall_score = 0.0
        else:
            overall_score = sum(
                result["evaluation"]["overall_score"]
                for result in results
            ) / total_cases

        suite_result = {
            "agent_version": agent_version,
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "overall_score": round(overall_score, 2),
            "average_latency_ms": round(sum(latency_values) / total_cases, 2) if total_cases else 0.0,
            "average_cost_usd": round(sum(cost_values) / len(cost_values), 8) if cost_values else None,
            "tool_usage_breakdown": tool_breakdown,
            "results": results,
        }

        current_span = trace.get_current_span()
        set_span_attributes(
            current_span,
            {
                "suite.total_cases": total_cases,
                "suite.passed_cases": passed_cases,
                "suite.failed_cases": failed_cases,
                "suite.overall_score": suite_result["overall_score"],
                "suite.average_latency_ms": suite_result["average_latency_ms"],
                "suite.average_cost_usd": suite_result["average_cost_usd"],
                "suite.tool_usage_breakdown": tool_breakdown,
            },
        )

        return suite_result


def compare_prompt_versions(
    cases: list[dict[str, Any]],
    prompt_v1: str,
    prompt_v2: str,
) -> dict[str, Any]:
    """
    Runs an experiment comparing weak v1 behavior against improved v2 behavior.
    """
    with trace_span(
        "experiment.compare_prompt_versions",
        {
            "experiment.name": "prompt_v1_vs_prompt_v2",
            "prompt.v1.length": len(prompt_v1),
            "prompt.v2.length": len(prompt_v2),
        },
    ):
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

        comparison_result = {
            "experiment_name": "prompt_v1_vs_prompt_v2",
            "prompt_v1_score": v1_result["overall_score"],
            "prompt_v2_score": v2_result["overall_score"],
            "improvement": improvement,
            "v1_result": v1_result,
            "v2_result": v2_result,
        }

        current_span = trace.get_current_span()
        current_span.set_attribute(
            "experiment.prompt_v1_score",
            comparison_result["prompt_v1_score"],
        )
        current_span.set_attribute(
            "experiment.prompt_v2_score",
            comparison_result["prompt_v2_score"],
        )
        current_span.set_attribute(
            "experiment.improvement",
            comparison_result["improvement"],
        )
        current_span.set_attribute(
            "experiment.v1_passed_cases",
            v1_result["passed_cases"],
        )
        current_span.set_attribute(
            "experiment.v2_passed_cases",
            v2_result["passed_cases"],
        )

        return comparison_result


def compare_gemini_prompt_versions(
    cases: list[dict[str, Any]],
    prompt_v1: str,
    prompt_v2: str,
) -> dict[str, Any]:
    """
    Runs a Gemini experiment comparing prompt v1 against prompt v2.
    Both versions use the real Gemini agent path.
    """
    with trace_span(
        "experiment.gemini_compare_prompt_versions",
        {
            "experiment.name": "gemini_prompt_v1_vs_prompt_v2",
            "agent.version": "gemini",
            "prompt.v1.length": len(prompt_v1),
            "prompt.v2.length": len(prompt_v2),
        },
    ):
        v1_result = run_evaluation_suite(
            cases=cases,
            prompt=prompt_v1,
            agent_version="gemini",
        )

        v2_result = run_evaluation_suite(
            cases=cases,
            prompt=prompt_v2,
            agent_version="gemini",
        )

        improvement = round(
            v2_result["overall_score"] - v1_result["overall_score"],
            2,
        )

        comparison_result = {
            "experiment_name": "gemini_prompt_v1_vs_prompt_v2",
            "agent_version": "gemini",
            "prompt_v1_score": v1_result["overall_score"],
            "prompt_v2_score": v2_result["overall_score"],
            "improvement": improvement,
            "v1_result": v1_result,
            "v2_result": v2_result,
        }

        current_span = trace.get_current_span()
        current_span.set_attribute(
            "experiment.prompt_v1_score",
            comparison_result["prompt_v1_score"],
        )
        current_span.set_attribute(
            "experiment.prompt_v2_score",
            comparison_result["prompt_v2_score"],
        )
        current_span.set_attribute(
            "experiment.improvement",
            comparison_result["improvement"],
        )
        current_span.set_attribute(
            "experiment.v1_passed_cases",
            v1_result["passed_cases"],
        )
        current_span.set_attribute(
            "experiment.v2_passed_cases",
            v2_result["passed_cases"],
        )

        return comparison_result
