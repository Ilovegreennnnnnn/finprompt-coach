import json
from pathlib import Path
from typing import Any


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.json"


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


def get_case_by_id(case_id: str) -> dict[str, Any] | None:
    """
    Return a single case by ID.
    """
    cases = load_cases()

    for case in cases:
        if case.get("id") == case_id:
            return case

    return None