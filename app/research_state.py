from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WATCHLISTS_PATH = DATA_DIR / "watchlists.json"
RESEARCH_RUNS_PATH = DATA_DIR / "market_research_runs.json"
IDEA_CARDS_PATH = DATA_DIR / "idea_cards.json"
INTROSPECTIONS_PATH = DATA_DIR / "introspection_reports.json"
PROMPT_REGISTRY_PATH = DATA_DIR / "prompt_registry.json"
LAB_QUEUE_PATH = DATA_DIR / "lab_queue.json"
ANALYSIS_DOSSIERS_PATH = DATA_DIR / "analysis_dossiers.json"
AUDIT_GRAPHS_PATH = DATA_DIR / "audit_graphs.json"

_STATE_LOCK = Lock()
_STATE_CACHE: dict[str, dict[str, Any]] = {}

DEFAULT_LIVE_PROMPT_TEXT = """
You are FinPrompt Coach operating inside a Phoenix-observed equity research loop.

Your role:
- Provide educational equity-research analysis only.
- Do not provide personalized financial advice.
- Do not recommend buying, selling, holding, or guaranteeing future performance.
- Use only the provided tool outputs and mark any missing or suspicious data clearly.
- If a metric looks implausible, stale, or unit-conflicted, label it as an anomaly instead of presenting it as a trusted fact.
- Keep the answer concise, grounded, and sectioned.
- Never reveal chain-of-thought or internal self-corrections.

Required structure:
1. Key facts
2. Positive factors
3. Risks
4. Missing information
5. Educational conclusion
6. Not financial advice

Always end with:
This is not financial advice.
""".strip()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_watchlists() -> list[dict[str, Any]]:
    now = utcnow_iso()
    return [
        {
            "id": "watchlist_platform_leaders",
            "name": "Platform Leaders",
            "description": "Default Phoenix watchlist for daily opportunity scans.",
            "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
            "schedule_enabled": True,
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": "watchlist_fr_sbf_120_core",
            "name": "France - SBF 120 Core",
            "description": (
                "Editable French-market seed watchlist built around liquid SBF 120 names for exploration on Euronext Paris. "
                "Use it as a starting universe, then add or remove names as needed."
            ),
            "tickers": [
                "AI.PA",
                "AIR.PA",
                "ALO.PA",
                "BN.PA",
                "BNP.PA",
                "BVI.PA",
                "CAP.PA",
                "CA.PA",
                "CS.PA",
                "DG.PA",
                "DSY.PA",
                "EDEN.PA",
                "EL.PA",
                "EN.PA",
                "ENGI.PA",
                "ENX.PA",
                "ERF.PA",
                "FGR.PA",
                "GLE.PA",
                "HO.PA",
                "KER.PA",
                "LI.PA",
                "LR.PA",
                "MC.PA",
                "ML.PA",
                "ORA.PA",
                "OR.PA",
                "PUB.PA",
                "RI.PA",
                "RMS.PA",
                "RNO.PA",
                "RXL.PA",
                "SAF.PA",
                "SAN.PA",
                "SGO.PA",
                "SU.PA",
                "TEP.PA",
                "TTE.PA",
                "URW.PA",
                "VIE.PA",
                "VIV.PA",
                "WLN.PA",
            ],
            "schedule_enabled": True,
            "created_at": now,
            "updated_at": now,
        }
    ]


def _default_prompt_registry() -> dict[str, Any]:
    now = utcnow_iso()
    version_id = "live_prompt_2026_05_15"
    return {
        "current_version_id": version_id,
        "candidate_version_id": None,
        "previous_version_id": None,
        "prompts": [
            {
                "version_id": version_id,
                "status": "current",
                "prompt_text": DEFAULT_LIVE_PROMPT_TEXT,
                "patch_source": ["Seed Phoenix-first live analyst prompt."],
                "origin_run_id": None,
                "activation_timestamp": now,
                "validation_metrics": {
                    "validation_status": "seeded",
                    "baseline_score": None,
                    "candidate_score": None,
                    "score_delta": None,
                    "latency_delta_ms": None,
                    "cost_delta_usd": None,
                },
                "observation_history": [],
                "rollback_reason": None,
            }
        ],
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_json_file(path: Path, default_data: Any) -> None:
    if path.exists():
        return

    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(default_data, file, indent=2)
        file.write("\n")


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    path: Path
    default_data: Any
    id_field: str = "id"
    sort_field: str | None = None
    sort_reverse: bool = True


def _load_json(path: Path, default_data: Any) -> Any:
    _ensure_json_file(path, default_data)

    stat = path.stat()
    cache_key = str(path)
    cached = _STATE_CACHE.get(cache_key)
    if cached and cached.get("mtime_ns") == stat.st_mtime_ns:
        return deepcopy(cached["data"])

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    _STATE_CACHE[cache_key] = {
        "mtime_ns": stat.st_mtime_ns,
        "data": deepcopy(data),
    }
    return deepcopy(data)


def _save_json(path: Path, data: Any) -> None:
    _ensure_parent(path)
    with _STATE_LOCK:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    try:
        stat = path.stat()
        _STATE_CACHE[str(path)] = {
            "mtime_ns": stat.st_mtime_ns,
            "data": deepcopy(data),
        }
    except OSError:
        _STATE_CACHE.pop(str(path), None)


COLLECTION_SPECS: dict[str, CollectionSpec] = {
    "watchlists": CollectionSpec(
        name="watchlists",
        path=WATCHLISTS_PATH,
        default_data=_default_watchlists(),
        id_field="id",
        sort_field="updated_at",
    ),
    "market_research_runs": CollectionSpec(
        name="market_research_runs",
        path=RESEARCH_RUNS_PATH,
        default_data=[],
        id_field="id",
        sort_field="created_at",
    ),
    "idea_cards": CollectionSpec(
        name="idea_cards",
        path=IDEA_CARDS_PATH,
        default_data=[],
        id_field="id",
        sort_field="created_at",
    ),
    "introspection_reports": CollectionSpec(
        name="introspection_reports",
        path=INTROSPECTIONS_PATH,
        default_data=[],
        id_field="id",
        sort_field="created_at",
    ),
    "lab_queue": CollectionSpec(
        name="lab_queue",
        path=LAB_QUEUE_PATH,
        default_data=[],
        id_field="id",
        sort_field="created_at",
    ),
    "analysis_dossiers": CollectionSpec(
        name="analysis_dossiers",
        path=ANALYSIS_DOSSIERS_PATH,
        default_data=[],
        id_field="id",
        sort_field="id",
    ),
    "audit_graphs": CollectionSpec(
        name="audit_graphs",
        path=AUDIT_GRAPHS_PATH,
        default_data=[],
        id_field="id",
        sort_field="id",
    ),
}


def _load_collection(spec: CollectionSpec) -> list[dict[str, Any]]:
    data = _load_json(spec.path, spec.default_data)
    return deepcopy(data if isinstance(data, list) else deepcopy(spec.default_data))


def _save_collection(spec: CollectionSpec, records: list[dict[str, Any]]) -> None:
    items = deepcopy(records)
    if spec.sort_field:
        items.sort(key=lambda item: item.get(spec.sort_field, ""), reverse=spec.sort_reverse)
    _save_json(spec.path, items)


def _upsert_many(spec: CollectionSpec, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = _load_collection(spec)
    records_by_id: dict[Any, dict[str, Any]] = {
        item.get(spec.id_field): item
        for item in existing
        if isinstance(item, dict) and item.get(spec.id_field) is not None
    }
    ordered_new_ids: list[Any] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get(spec.id_field)
        if record_id is None:
            continue
        if record_id not in records_by_id:
            ordered_new_ids.append(record_id)
        records_by_id[record_id] = record

    merged = list(records_by_id.values())
    _save_collection(spec, merged)
    return [records_by_id[record_id] for record_id in ordered_new_ids] or records


def _get_record(spec: CollectionSpec, record_id: Any) -> dict[str, Any] | None:
    for record in _load_collection(spec):
        if record.get(spec.id_field) == record_id:
            return record
    return None


def _list_records_by_field(spec: CollectionSpec, field_name: str, value: Any) -> list[dict[str, Any]]:
    return [
        record
        for record in _load_collection(spec)
        if record.get(field_name) == value
    ]


def load_watchlists() -> list[dict[str, Any]]:
    data = _load_json(WATCHLISTS_PATH, _default_watchlists())
    watchlists = data if isinstance(data, list) else _default_watchlists()
    known_ids = {item.get("id") for item in watchlists if isinstance(item, dict)}
    mutated = False

    for default_watchlist in _default_watchlists():
        if default_watchlist.get("id") not in known_ids:
            watchlists.append(default_watchlist)
            mutated = True

    if mutated:
        save_watchlists(watchlists)

    return deepcopy(watchlists)


def save_watchlists(watchlists: list[dict[str, Any]]) -> None:
    _save_json(WATCHLISTS_PATH, watchlists)


def upsert_watchlist(watchlist: dict[str, Any]) -> dict[str, Any]:
    watchlists = load_watchlists()
    updated = False

    for index, existing in enumerate(watchlists):
        if existing.get("id") == watchlist.get("id"):
            watchlists[index] = watchlist
            updated = True
            break

    if not updated:
        watchlists.append(watchlist)

    save_watchlists(watchlists)
    return watchlist


def get_watchlist(watchlist_id: str) -> dict[str, Any] | None:
    for watchlist in load_watchlists():
        if watchlist.get("id") == watchlist_id:
            return watchlist
    return None


def load_market_research_runs() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["market_research_runs"])


def save_market_research_runs(runs: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["market_research_runs"], runs)


def save_market_research_run(run_record: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["market_research_runs"], [run_record])
    return run_record


def get_market_research_run(run_id: str) -> dict[str, Any] | None:
    return _get_record(COLLECTION_SPECS["market_research_runs"], run_id)


def load_idea_cards() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["idea_cards"])


def save_idea_cards(cards: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["idea_cards"], cards)


def save_idea_card(card: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["idea_cards"], [card])
    return card


def save_idea_cards_bulk(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _upsert_many(COLLECTION_SPECS["idea_cards"], cards)


def list_idea_cards_for_run(run_id: str) -> list[dict[str, Any]]:
    return _list_records_by_field(COLLECTION_SPECS["idea_cards"], "run_id", run_id)


def load_introspection_reports() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["introspection_reports"])


def save_introspection_reports(reports: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["introspection_reports"], reports)


def save_introspection_report(report: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["introspection_reports"], [report])
    return report


def get_introspection_report(report_id: str) -> dict[str, Any] | None:
    for report in load_introspection_reports():
        if report.get("id") == report_id:
            return report
    return None


def load_lab_queue() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["lab_queue"])


def save_lab_queue(queue_items: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["lab_queue"], queue_items)


def save_lab_queue_item(queue_item: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["lab_queue"], [queue_item])
    return queue_item


def load_analysis_dossiers() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["analysis_dossiers"])


def save_analysis_dossiers(dossiers: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["analysis_dossiers"], dossiers)


def save_analysis_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["analysis_dossiers"], [dossier])
    return dossier


def save_analysis_dossiers_bulk(dossiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _upsert_many(COLLECTION_SPECS["analysis_dossiers"], dossiers)


def list_analysis_dossiers_for_run(run_id: str) -> list[dict[str, Any]]:
    return _list_records_by_field(COLLECTION_SPECS["analysis_dossiers"], "run_id", run_id)


def get_analysis_dossier(run_id: str, ticker: str) -> dict[str, Any] | None:
    ticker_upper = ticker.upper()
    for dossier in load_analysis_dossiers():
        if dossier.get("run_id") == run_id and str(dossier.get("ticker", "")).upper() == ticker_upper:
            return dossier
    return None


def load_audit_graphs() -> list[dict[str, Any]]:
    return _load_collection(COLLECTION_SPECS["audit_graphs"])


def save_audit_graphs(graphs: list[dict[str, Any]]) -> None:
    _save_collection(COLLECTION_SPECS["audit_graphs"], graphs)


def save_audit_graph(graph: dict[str, Any]) -> dict[str, Any]:
    _upsert_many(COLLECTION_SPECS["audit_graphs"], [graph])
    return graph


def save_audit_graphs_bulk(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _upsert_many(COLLECTION_SPECS["audit_graphs"], graphs)


def list_audit_graphs_for_run(run_id: str) -> list[dict[str, Any]]:
    return _list_records_by_field(COLLECTION_SPECS["audit_graphs"], "run_id", run_id)


def get_audit_graph(run_id: str, ticker: str) -> dict[str, Any] | None:
    ticker_upper = ticker.upper()
    for graph in load_audit_graphs():
        entity = graph.get("entity", {})
        if graph.get("run_id") == run_id and str(entity.get("ticker", "")).upper() == ticker_upper:
            return graph
    return None


def load_prompt_registry() -> dict[str, Any]:
    data = _load_json(PROMPT_REGISTRY_PATH, _default_prompt_registry())
    if not isinstance(data, dict):
        data = _default_prompt_registry()
        _save_json(PROMPT_REGISTRY_PATH, data)
    return deepcopy(data)


def save_prompt_registry(registry: dict[str, Any]) -> None:
    _save_json(PROMPT_REGISTRY_PATH, registry)


def _get_prompt_entry(registry: dict[str, Any], version_id: str | None) -> dict[str, Any] | None:
    if not version_id:
        return None

    for entry in registry.get("prompts", []):
        if entry.get("version_id") == version_id:
            return entry
    return None


def get_current_prompt_entry() -> dict[str, Any]:
    registry = load_prompt_registry()
    entry = _get_prompt_entry(registry, registry.get("current_version_id"))
    if entry:
        return entry
    seeded = _default_prompt_registry()
    save_prompt_registry(seeded)
    return seeded["prompts"][0]


def get_prompt_registry_state() -> dict[str, Any]:
    registry = load_prompt_registry()
    current_entry = _get_prompt_entry(registry, registry.get("current_version_id"))
    candidate_entry = _get_prompt_entry(registry, registry.get("candidate_version_id"))
    previous_entry = _get_prompt_entry(registry, registry.get("previous_version_id"))
    return {
        "current": deepcopy(current_entry),
        "candidate": deepcopy(candidate_entry),
        "previous": deepcopy(previous_entry),
        "registry": registry,
    }


def register_prompt_candidate(
    *,
    prompt_text: str,
    prompt_patch: list[str],
    origin_run_id: str | None,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    registry = load_prompt_registry()
    candidate_version_id = f"live_candidate_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    existing_candidate = _get_prompt_entry(registry, registry.get("candidate_version_id"))

    if existing_candidate:
        existing_candidate["status"] = "rejected"
        existing_candidate["rejection_reason"] = "Replaced by a newer candidate."

    candidate_entry = {
        "version_id": candidate_version_id,
        "status": "candidate",
        "prompt_text": prompt_text,
        "patch_source": list(prompt_patch),
        "origin_run_id": origin_run_id,
        "activation_timestamp": None,
        "validation_metrics": validation_metrics,
        "observation_history": [],
        "rollback_reason": None,
    }
    registry.setdefault("prompts", []).append(candidate_entry)
    registry["candidate_version_id"] = candidate_version_id
    save_prompt_registry(registry)
    return candidate_entry


def activate_prompt_candidate(*, reason: str) -> dict[str, Any] | None:
    registry = load_prompt_registry()
    current_entry = _get_prompt_entry(registry, registry.get("current_version_id"))
    candidate_entry = _get_prompt_entry(registry, registry.get("candidate_version_id"))
    previous_entry = _get_prompt_entry(registry, registry.get("previous_version_id"))

    if candidate_entry is None:
        return None

    if previous_entry:
        previous_entry["status"] = "rejected"
        previous_entry["rejection_reason"] = "Superseded by a newer rollback baseline."

    if current_entry:
        current_entry["status"] = "previous"

    candidate_entry["status"] = "current"
    candidate_entry["activation_timestamp"] = utcnow_iso()
    candidate_entry["activation_reason"] = reason

    registry["previous_version_id"] = registry.get("current_version_id")
    registry["current_version_id"] = candidate_entry.get("version_id")
    registry["candidate_version_id"] = None
    save_prompt_registry(registry)
    return candidate_entry


def reject_prompt_candidate(version_id: str, reason: str) -> dict[str, Any] | None:
    registry = load_prompt_registry()
    candidate_entry = _get_prompt_entry(registry, version_id)
    if candidate_entry is None:
        return None

    candidate_entry["status"] = "rejected"
    candidate_entry["rejection_reason"] = reason
    if registry.get("candidate_version_id") == version_id:
        registry["candidate_version_id"] = None
    save_prompt_registry(registry)
    return candidate_entry


def record_prompt_observation(
    *,
    version_id: str,
    run_id: str,
    regression: bool,
    anomaly_count: int,
    latency_ms: float | None,
) -> dict[str, Any] | None:
    registry = load_prompt_registry()
    entry = _get_prompt_entry(registry, version_id)
    if entry is None:
        return None

    observation = {
        "run_id": run_id,
        "recorded_at": utcnow_iso(),
        "regression": regression,
        "anomaly_count": anomaly_count,
        "latency_ms": latency_ms,
    }
    history = entry.setdefault("observation_history", [])
    history.append(observation)
    entry["observation_history"] = history[-10:]
    save_prompt_registry(registry)
    return entry


def rollback_to_previous_prompt(*, reason: str) -> dict[str, Any] | None:
    registry = load_prompt_registry()
    current_entry = _get_prompt_entry(registry, registry.get("current_version_id"))
    previous_entry = _get_prompt_entry(registry, registry.get("previous_version_id"))

    if current_entry is None or previous_entry is None:
        return None

    current_entry["status"] = "previous"
    current_entry["rollback_reason"] = reason
    previous_entry["status"] = "current"
    previous_entry["activation_timestamp"] = utcnow_iso()
    previous_entry["activation_reason"] = f"Rollback: {reason}"

    registry["current_version_id"], registry["previous_version_id"] = (
        registry["previous_version_id"],
        registry["current_version_id"],
    )
    save_prompt_registry(registry)
    return previous_entry


def list_universal_collections() -> list[dict[str, Any]]:
    collections: list[dict[str, Any]] = []
    for spec in COLLECTION_SPECS.values():
        records = _load_collection(spec)
        collections.append(
            {
                "name": spec.name,
                "format": "json",
                "encoding": "utf-8",
                "path": str(spec.path),
                "record_count": len(records),
                "id_field": spec.id_field,
                "sort_field": spec.sort_field,
            }
        )

    collections.append(
        {
            "name": "prompt_registry",
            "format": "json",
            "encoding": "utf-8",
            "path": str(PROMPT_REGISTRY_PATH),
            "record_count": len(load_prompt_registry().get("prompts", [])),
            "id_field": "version_id",
            "sort_field": "activation_timestamp",
        }
    )
    return collections


def get_universal_collection(
    collection_name: str,
    *,
    limit: int | None = None,
) -> dict[str, Any] | None:
    if collection_name == "prompt_registry":
        registry = load_prompt_registry()
        prompts = registry.get("prompts", [])
        if isinstance(limit, int) and limit > 0:
            prompts = prompts[:limit]
        return {
            "collection": collection_name,
            "format": "json",
            "encoding": "utf-8",
            "records": prompts,
            "registry_state": {
                "current_version_id": registry.get("current_version_id"),
                "candidate_version_id": registry.get("candidate_version_id"),
                "previous_version_id": registry.get("previous_version_id"),
            },
        }

    spec = COLLECTION_SPECS.get(collection_name)
    if spec is None:
        return None

    records = _load_collection(spec)
    if isinstance(limit, int) and limit > 0:
        records = records[:limit]
    return {
        "collection": spec.name,
        "format": "json",
        "encoding": "utf-8",
        "records": records,
    }


def get_universal_record(collection_name: str, record_id: str) -> dict[str, Any] | None:
    if collection_name == "prompt_registry":
        registry = load_prompt_registry()
        for prompt in registry.get("prompts", []):
            if prompt.get("version_id") == record_id:
                return prompt
        return None

    spec = COLLECTION_SPECS.get(collection_name)
    if spec is None:
        return None
    return _get_record(spec, record_id)


def export_universal_data_bundle(
    *,
    collections: list[str] | None = None,
    export_format: str = "json",
) -> dict[str, Any]:
    selected = collections or list(COLLECTION_SPECS.keys()) + ["prompt_registry"]
    normalized_format = export_format.lower().strip()
    if normalized_format not in {"json", "jsonl"}:
        raise ValueError("Unsupported export format. Use 'json' or 'jsonl'.")

    manifest = {
        "generated_at": utcnow_iso(),
        "format": normalized_format,
        "collections": [],
    }
    payloads: dict[str, Any] = {}

    for collection_name in selected:
        collection_payload = get_universal_collection(collection_name)
        if collection_payload is None:
            continue

        records = collection_payload.get("records", [])
        manifest["collections"].append(
            {
                "name": collection_name,
                "record_count": len(records) if isinstance(records, list) else 0,
                "encoding": "utf-8",
            }
        )

        if normalized_format == "jsonl":
            lines = [
                json.dumps(record, ensure_ascii=True)
                for record in records
                if isinstance(record, dict)
            ]
            payloads[collection_name] = "\n".join(lines)
        else:
            payloads[collection_name] = records

    return {
        "manifest": manifest,
        "payloads": payloads,
    }
