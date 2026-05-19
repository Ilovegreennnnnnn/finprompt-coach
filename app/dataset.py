import json
from pathlib import Path
from threading import Lock
from typing import Any


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.json"
PROMOTED_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "promoted_cases.json"
DATASET_LOCK = Lock()
CORE_BENCHMARK_ID = "core_equity_research"
PROMOTED_BENCHMARK_ID = "promoted_live_cases"
ALL_BENCHMARK_ID = "all_equity_research"


def load_cases() -> list[dict[str, Any]]:
    """
    Load synthetic financial evaluation cases from data/cases.json.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Dataset must be a list of cases")

    return cases


def _ensure_promoted_cases_file() -> None:
    if PROMOTED_DATASET_PATH.exists():
        return

    PROMOTED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMOTED_DATASET_PATH.write_text("[]\n", encoding="utf-8")


def load_promoted_cases() -> list[dict[str, Any]]:
    _ensure_promoted_cases_file()

    with PROMOTED_DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Promoted dataset must be a list of cases")

    return cases


def save_promoted_cases(cases: list[dict[str, Any]]) -> None:
    _ensure_promoted_cases_file()

    with DATASET_LOCK:
        with PROMOTED_DATASET_PATH.open("w", encoding="utf-8") as file:
            json.dump(cases, file, indent=2)
            file.write("\n")


def upsert_promoted_case(case: dict[str, Any]) -> dict[str, Any]:
    cases = load_promoted_cases()
    source_trace_id = case.get("benchmark_metadata", {}).get("source_trace_id")
    case_id = case.get("id")

    updated = False
    for index, existing in enumerate(cases):
        existing_trace_id = existing.get("benchmark_metadata", {}).get("source_trace_id")

        if (source_trace_id and existing_trace_id == source_trace_id) or (
            case_id and existing.get("id") == case_id
        ):
            cases[index] = case
            updated = True
            break

    if not updated:
        cases.append(case)

    save_promoted_cases(cases)
    return case


def load_benchmark_cases(benchmark_id: str | None = None) -> list[dict[str, Any]]:
    selected_id = benchmark_id or ALL_BENCHMARK_ID
    core_cases = load_cases()
    promoted_cases = load_promoted_cases()

    if selected_id == CORE_BENCHMARK_ID:
        return core_cases

    if selected_id == PROMOTED_BENCHMARK_ID:
        return promoted_cases

    if selected_id == ALL_BENCHMARK_ID:
        return core_cases + promoted_cases

    raise ValueError(f"Unknown benchmark id: {selected_id}")


def list_benchmarks() -> list[dict[str, Any]]:
    core_cases = load_cases()
    promoted_cases = load_promoted_cases()

    return [
        {
            "benchmark_id": CORE_BENCHMARK_ID,
            "title": "Core Equity Research Benchmark",
            "description": "Synthetic benchmark covering safety, risk, format, and tool-selection cases.",
            "case_count": len(core_cases),
        },
        {
            "benchmark_id": PROMOTED_BENCHMARK_ID,
            "title": "Promoted Live Cases",
            "description": "Frozen benchmark cases promoted automatically from live traced analyst sessions.",
            "case_count": len(promoted_cases),
        },
        {
            "benchmark_id": ALL_BENCHMARK_ID,
            "title": "All Equity Research Cases",
            "description": "Union of the core benchmark and promoted live cases.",
            "case_count": len(core_cases) + len(promoted_cases),
        },
    ]


def get_benchmark_case_by_id(case_id: str) -> dict[str, Any] | None:
    for case in load_cases():
        if case.get("id") == case_id:
            return case

    for case in load_promoted_cases():
        if case.get("id") == case_id:
            return case

    return None


def get_case_by_id(case_id: str) -> dict[str, Any] | None:
    """
    Return a single case by ID.
    """
    cases = load_cases()

    for case in cases:
        if case.get("id") == case_id:
            return case

    return None
