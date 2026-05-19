from __future__ import annotations

from typing import Any

from app.live_tools import normalize_for_json
from app.research_state import utcnow_iso


def _snapshot_timestamp(payload: dict[str, Any]) -> str:
    timestamp = payload.get("timestamp_utc")
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    return utcnow_iso().replace("+00:00", "Z")


def _snapshot_confidence(base: float, warnings: list[str], extra_penalty: float = 0.0) -> float:
    confidence = float(base) - float(extra_penalty) - len(warnings) * 4.0
    return round(max(25.0, min(98.0, confidence)), 2)


def _normalize_filing_items(raw_items: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []

    filings: list[dict[str, Any]] = []
    for item in raw_items[:5]:
        if not isinstance(item, dict):
            continue
        filings.append(
            {
                "title": item.get("title") or item.get("type") or item.get("headline"),
                "date": item.get("date") or item.get("filingDate") or item.get("published"),
                "type": item.get("type") or item.get("formType"),
                "summary": item.get("summary") or item.get("description"),
            }
        )
    return filings


def build_source_snapshots(
    *,
    ticker: str,
    fundamentals_payload: dict[str, Any],
    price_payload: dict[str, Any],
    market_context_payload: dict[str, Any],
    peer_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    fundamentals = fundamentals_payload.get("data", {}).get(ticker, {})
    price_history = price_payload.get("data", {}).get(ticker, {})
    market_context = market_context_payload.get("data", {}).get(ticker, {})
    data_quality = fundamentals.get("data_quality", {})
    filings = _normalize_filing_items(
        fundamentals.get("sec_filings")
        or market_context.get("sec_filings")
        or []
    )

    fundamentals_warnings = list(fundamentals_payload.get("warnings", []))
    price_warnings = list(price_payload.get("warnings", []))
    market_warnings = list(market_context_payload.get("warnings", []))
    filing_warnings = [] if filings else [f"No structured filing snapshot was available for {ticker}."]

    snapshots = [
        {
            "source_name": "yahoo_fundamentals",
            "snapshot_timestamp": _snapshot_timestamp(fundamentals_payload),
            "field_map": normalize_for_json(fundamentals, max_records=40),
            "confidence": _snapshot_confidence(
                data_quality.get("confidence_score", 72),
                fundamentals_warnings,
            ),
            "staleness": "near_real_time",
            "warnings": fundamentals_warnings,
        },
        {
            "source_name": "yahoo_price_history",
            "snapshot_timestamp": _snapshot_timestamp(price_payload),
            "field_map": normalize_for_json(price_history, max_records=25),
            "confidence": _snapshot_confidence(78, price_warnings),
            "staleness": "same_session",
            "warnings": price_warnings,
        },
        {
            "source_name": "yahoo_news_context",
            "snapshot_timestamp": _snapshot_timestamp(market_context_payload),
            "field_map": normalize_for_json(market_context, max_records=10),
            "confidence": _snapshot_confidence(
                68,
                market_warnings,
                extra_penalty=0.0 if market_context.get("recent_news") else 12.0,
            ),
            "staleness": "same_session",
            "warnings": market_warnings,
        },
        {
            "source_name": "yahoo_filings_snapshot",
            "snapshot_timestamp": _snapshot_timestamp(fundamentals_payload),
            "field_map": normalize_for_json({"recent_filings": filings}, max_records=8),
            "confidence": _snapshot_confidence(64 if filings else 38, filing_warnings),
            "staleness": "unknown",
            "warnings": filing_warnings,
        },
    ]

    if isinstance(peer_row, dict) and peer_row:
        snapshots.append(
            {
                "source_name": "peer_comparison_snapshot",
                "snapshot_timestamp": _snapshot_timestamp(fundamentals_payload),
                "field_map": normalize_for_json(peer_row, max_records=30),
                "confidence": _snapshot_confidence(74, []),
                "staleness": "same_session",
                "warnings": [],
            }
        )

    return snapshots
