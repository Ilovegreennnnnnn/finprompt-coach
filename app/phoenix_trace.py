from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace


tracer = trace.get_tracer("finprompt-coach")
MAX_ATTR_LEN = 7000


def _clean_attr(value: Any) -> str | int | float | bool:
    if value is None:
        return ""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        return value[:MAX_ATTR_LEN]

    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:MAX_ATTR_LEN]
    except Exception:
        return str(value)[:MAX_ATTR_LEN]


def set_span_attrs(span: trace.Span, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        try:
            span.set_attribute(key, _clean_attr(value))
        except Exception:
            span.set_attribute(key, str(value)[:MAX_ATTR_LEN])


def _evaluation_scores(evaluation: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}

    for item in evaluation.get("evaluations", []):
        name = item.get("name", "unknown")
        scores[f"eval.{name}.passed"] = item.get("passed", False)
        scores[f"eval.{name}.score"] = item.get("score", 0)
        scores[f"eval.{name}.details"] = item.get("details", {})

    return scores


def trace_experiment_result(payload: dict[str, Any]) -> None:
    experiment = payload.get("experiment_result", payload)

    with tracer.start_as_current_span("finprompt.experiment.compare_prompts") as experiment_span:
        set_span_attrs(
            experiment_span,
            {
                "project.name": "FinPrompt Coach",
                "experiment.name": experiment.get("experiment_name"),
                "agent.version": experiment.get("agent_version"),
                "prompt.v1.score": experiment.get("prompt_v1_score"),
                "prompt.v2.score": experiment.get("prompt_v2_score"),
                "prompt.improvement": experiment.get("improvement"),
                "prompt.v1.preview": payload.get("prompt_v1", "")[:1000],
                "prompt.v2.preview": payload.get("prompt_v2", "")[:1000],
            },
        )

        for suite_key in ["v1_result", "v2_result"]:
            suite = experiment.get(suite_key, {})
            prompt_version = "v1" if suite_key == "v1_result" else "v2"

            with tracer.start_as_current_span(f"finprompt.suite.{prompt_version}") as suite_span:
                set_span_attrs(
                    suite_span,
                    {
                        "suite.prompt_version": prompt_version,
                        "suite.agent_version": suite.get("agent_version"),
                        "suite.total_cases": suite.get("total_cases"),
                        "suite.passed_cases": suite.get("passed_cases"),
                        "suite.failed_cases": suite.get("failed_cases"),
                        "suite.overall_score": suite.get("overall_score"),
                    },
                )

                for result in suite.get("results", []):
                    case_id = result.get("case_id")
                    title = result.get("title")
                    agent_run = result.get("agent_run", {})
                    evaluation = result.get("evaluation", {})

                    failed_evaluators = [
                        item.get("name")
                        for item in evaluation.get("evaluations", [])
                        if not item.get("passed", False)
                    ]

                    with tracer.start_as_current_span(
                        f"finprompt.case.{prompt_version}.{case_id}"
                    ) as case_span:
                        set_span_attrs(
                            case_span,
                            {
                                "case.id": case_id,
                                "case.title": title,
                                "case.prompt_version": prompt_version,
                                "case.overall_score": evaluation.get("overall_score"),
                                "case.passed": evaluation.get("passed"),
                                "case.failed_evaluators": failed_evaluators,
                                "tool.used": agent_run.get("tool_used"),
                                "response.preview": agent_run.get("response_text", "")[:2500],
                            },
                        )

                        set_span_attrs(case_span, _evaluation_scores(evaluation))

                        with tracer.start_as_current_span(
                            f"finprompt.tool.{agent_run.get('tool_used', 'unknown')}"
                        ) as tool_span:
                            set_span_attrs(
                                tool_span,
                                {
                                    "tool.name": agent_run.get("tool_used"),
                                    "tool.output": agent_run.get("tool_output"),
                                    "case.id": case_id,
                                    "case.prompt_version": prompt_version,
                                },
                            )

                        for evaluator in evaluation.get("evaluations", []):
                            evaluator_name = evaluator.get("name", "unknown")

                            with tracer.start_as_current_span(
                                f"finprompt.eval.{evaluator_name}"
                            ) as eval_span:
                                set_span_attrs(
                                    eval_span,
                                    {
                                        "case.id": case_id,
                                        "case.prompt_version": prompt_version,
                                        "eval.name": evaluator_name,
                                        "eval.passed": evaluator.get("passed"),
                                        "eval.score": evaluator.get("score"),
                                        "eval.details": evaluator.get("details"),
                                    },
                                )

    try:
        trace.get_tracer_provider().force_flush()
    except Exception:
        pass