from typing import Any

from opentelemetry import trace

from app.agent import run_gemini_agent, run_improved_agent, run_weak_agent
from app.evaluators import evaluate_response
from app.tracing import trace_span


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
            "results": results,
        }

        current_span = trace.get_current_span()
        current_span.set_attribute("suite.total_cases", total_cases)
        current_span.set_attribute("suite.passed_cases", passed_cases)
        current_span.set_attribute("suite.failed_cases", failed_cases)
        current_span.set_attribute(
            "suite.overall_score",
            suite_result["overall_score"],
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