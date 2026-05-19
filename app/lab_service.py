from __future__ import annotations

from typing import Any

from app.dataset import ALL_BENCHMARK_ID, get_benchmark_case_by_id, list_benchmarks, load_benchmark_cases
from app.prompt_coach import optimize_prompt_loop


def get_lab_benchmarks() -> list[dict[str, Any]]:
    return list_benchmarks()


def get_lab_case(case_id: str) -> dict[str, Any] | None:
    return get_benchmark_case_by_id(case_id)


def run_prompt_lab(
    *,
    prompt: str,
    benchmark_id: str | None = None,
    max_rounds: int = 3,
) -> dict[str, Any]:
    selected_benchmark_id = benchmark_id or ALL_BENCHMARK_ID
    cases = load_benchmark_cases(selected_benchmark_id)
    optimization_result = optimize_prompt_loop(
        original_prompt=prompt,
        cases=cases,
        max_rounds=max_rounds,
        agent_version="gemini",
    )

    return {
        "benchmark_id": selected_benchmark_id,
        "benchmark_case_count": len(cases),
        **optimization_result,
    }
