from __future__ import annotations

import math
from typing import Any

from app.live_tools import CRITICAL_FUNDAMENTAL_FIELDS


METRIC_SOURCES = {
    "market_cap": "yahoo_fundamentals",
    "trailing_pe": "yahoo_fundamentals",
    "forward_pe": "yahoo_fundamentals",
    "price_to_book": "yahoo_fundamentals",
    "operating_margin": "yahoo_fundamentals",
    "profit_margin": "yahoo_fundamentals",
    "gross_margin": "yahoo_fundamentals",
    "return_on_equity": "yahoo_fundamentals",
    "revenue_growth": "yahoo_fundamentals",
    "earnings_growth": "yahoo_fundamentals",
    "debt_to_equity": "yahoo_fundamentals",
    "current_ratio": "yahoo_fundamentals",
    "quick_ratio": "yahoo_fundamentals",
    "free_cash_flow": "yahoo_fundamentals",
    "operating_cash_flow": "yahoo_fundamentals",
    "total_cash": "yahoo_fundamentals",
    "total_debt": "yahoo_fundamentals",
    "dividend_yield": "yahoo_fundamentals",
    "payout_ratio": "yahoo_fundamentals",
    "period_return_pct": "yahoo_price_history",
    "annualized_volatility_pct": "yahoo_price_history",
    "recent_news_count": "yahoo_news_context",
}


def safe_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _get_snapshot_map(snapshots: list[dict[str, Any]], source_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for snapshot in snapshots:
        if snapshot.get("source_name") == source_name:
            field_map = snapshot.get("field_map")
            if isinstance(field_map, dict):
                return snapshot, field_map
            return snapshot, {}
    return {}, {}


def _metric_warning(metric_name: str, value: float) -> list[str]:
    warnings: list[str] = []
    if metric_name in {"operating_margin", "profit_margin", "gross_margin", "return_on_equity"}:
        if value < -75 or value > 150:
            warnings.append("metric_anomaly")
    elif metric_name == "dividend_yield":
        if value < 0 or value > 25:
            warnings.extend(["metric_anomaly", "unit_conflict"])
    elif metric_name == "payout_ratio":
        if value < 0 or value > 130:
            warnings.append("metric_anomaly")
    elif metric_name == "price_to_book":
        if value < 0 or value > 80:
            warnings.append("metric_anomaly")
    elif metric_name == "trailing_pe":
        if value <= 0 or value > 150:
            warnings.append("metric_anomaly")
    elif metric_name == "debt_to_equity":
        if value < 0 or value > 400:
            warnings.append("metric_anomaly")
    elif metric_name == "current_ratio":
        if value < 0 or value > 20:
            warnings.append("metric_anomaly")
    elif metric_name == "annualized_volatility_pct":
        if value < 0 or value > 250:
            warnings.append("metric_anomaly")
    return warnings


def _build_conflict_node(
    *,
    ticker: str,
    metric_name: str,
    source_name: str,
    value: Any,
    snapshot_timestamp: str,
    conflict_type: str,
    details: str,
) -> dict[str, Any]:
    return {
        "conflict_id": f"{ticker.lower()}_{metric_name}_{conflict_type}",
        "ticker": ticker,
        "metric_name": metric_name,
        "source_name": source_name,
        "snapshot_timestamp": snapshot_timestamp,
        "observed_value": value,
        "conflict_type": conflict_type,
        "details": details,
    }


def normalize_source_snapshots(
    *,
    ticker: str,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    fundamentals_snapshot, fundamentals_map = _get_snapshot_map(snapshots, "yahoo_fundamentals")
    price_snapshot, price_map = _get_snapshot_map(snapshots, "yahoo_price_history")
    news_snapshot, news_map = _get_snapshot_map(snapshots, "yahoo_news_context")
    peer_snapshot, peer_map = _get_snapshot_map(snapshots, "peer_comparison_snapshot")

    summary = price_map.get("summary", {}) if isinstance(price_map.get("summary"), dict) else {}
    data_quality = fundamentals_map.get("data_quality", {})
    quality_flags = set(data_quality.get("quality_flags", []))
    missing_fields = set(data_quality.get("missing_fields", []))
    uncertainty_flags = set()
    conflicts: list[dict[str, Any]] = []
    metric_observations: list[dict[str, Any]] = []

    normalized_metrics: dict[str, Any] = {
        "ticker": ticker,
        "sector": fundamentals_map.get("sector"),
        "industry": fundamentals_map.get("industry"),
        "company_name": fundamentals_map.get("company_name"),
        "currency": fundamentals_map.get("currency"),
        "recent_news_count": len(news_map.get("recent_news", [])) if isinstance(news_map.get("recent_news"), list) else 0,
        "filing_count": len(
            (((_get_snapshot_map(snapshots, "yahoo_filings_snapshot")[1]).get("recent_filings")) or [])
            if isinstance((_get_snapshot_map(snapshots, "yahoo_filings_snapshot")[1]).get("recent_filings"), list)
            else []
        ),
    }

    metrics_to_extract = [
        "market_cap",
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "operating_margin",
        "profit_margin",
        "gross_margin",
        "return_on_equity",
        "revenue_growth",
        "earnings_growth",
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "free_cash_flow",
        "operating_cash_flow",
        "total_cash",
        "total_debt",
        "dividend_yield",
        "payout_ratio",
    ]

    for metric_name in metrics_to_extract:
        value = safe_number(fundamentals_map.get(metric_name))
        normalized_metrics[metric_name] = value
        if value is None:
            if metric_name in CRITICAL_FUNDAMENTAL_FIELDS:
                quality_flags.add("missing_critical_field")
                missing_fields.add(metric_name)
            continue

        warnings = _metric_warning(metric_name, value)
        for warning in warnings:
            quality_flags.add(warning)
        if warnings:
            conflicts.append(
                _build_conflict_node(
                    ticker=ticker,
                    metric_name=metric_name,
                    source_name="yahoo_fundamentals",
                    value=value,
                    snapshot_timestamp=fundamentals_snapshot.get("snapshot_timestamp", ""),
                    conflict_type=warnings[0],
                    details=f"{metric_name} was outside the configured plausibility range.",
                )
            )

        metric_observations.append(
            {
                "metric_name": metric_name,
                "value": value,
                "source_name": METRIC_SOURCES[metric_name],
                "snapshot_timestamp": fundamentals_snapshot.get("snapshot_timestamp"),
                "confidence": fundamentals_snapshot.get("confidence"),
                "warnings": warnings,
            }
        )

    for metric_name in ("period_return_pct", "annualized_volatility_pct"):
        value = safe_number(summary.get(metric_name))
        normalized_metrics[metric_name] = value
        warnings = _metric_warning(metric_name, value) if value is not None else []
        for warning in warnings:
            quality_flags.add(warning)
        if value is not None:
            metric_observations.append(
                {
                    "metric_name": metric_name,
                    "value": value,
                    "source_name": METRIC_SOURCES[metric_name],
                    "snapshot_timestamp": price_snapshot.get("snapshot_timestamp"),
                    "confidence": price_snapshot.get("confidence"),
                    "warnings": warnings,
                }
            )

    if not news_map.get("recent_news"):
        uncertainty_flags.add("limited_news_context")

    if peer_map:
        valuation_delta = safe_number(peer_map.get("trailing_pe"))
        if valuation_delta is not None and normalized_metrics.get("trailing_pe") is not None:
            if abs(valuation_delta - normalized_metrics["trailing_pe"]) > max(5.0, normalized_metrics["trailing_pe"] * 0.15):
                conflicts.append(
                    _build_conflict_node(
                        ticker=ticker,
                        metric_name="trailing_pe",
                        source_name="peer_comparison_snapshot",
                        value=valuation_delta,
                        snapshot_timestamp=peer_snapshot.get("snapshot_timestamp", ""),
                        conflict_type="peer_snapshot_drift",
                        details="Peer snapshot materially diverged from the direct fundamentals snapshot.",
                    )
                )

    if (
        normalized_metrics.get("free_cash_flow") is not None
        and normalized_metrics.get("operating_cash_flow") is not None
        and normalized_metrics["free_cash_flow"] > normalized_metrics["operating_cash_flow"] * 1.1
    ):
        conflicts.append(
            _build_conflict_node(
                ticker=ticker,
                metric_name="free_cash_flow",
                source_name="yahoo_fundamentals",
                value=normalized_metrics["free_cash_flow"],
                snapshot_timestamp=fundamentals_snapshot.get("snapshot_timestamp", ""),
                conflict_type="cash_flow_conflict",
                details="Free cash flow exceeded operating cash flow by an unusually wide margin.",
            )
        )

    if (
        normalized_metrics.get("revenue_growth") is not None
        and normalized_metrics.get("period_return_pct") is not None
        and normalized_metrics["revenue_growth"] < 0
        and normalized_metrics["period_return_pct"] > 20
    ):
        conflicts.append(
            _build_conflict_node(
                ticker=ticker,
                metric_name="revenue_growth",
                source_name="yahoo_news_context",
                value=normalized_metrics["period_return_pct"],
                snapshot_timestamp=news_snapshot.get("snapshot_timestamp", ""),
                conflict_type="market_vs_fundamental_signal",
                details="Price momentum looked strong despite negative revenue growth.",
            )
        )

    confidence_score = float(data_quality.get("confidence_score", fundamentals_snapshot.get("confidence", 68)))
    confidence_score -= len(conflicts) * 6.0
    confidence_score -= len(missing_fields) * 3.0
    if uncertainty_flags:
        confidence_score -= 4.0

    missing_rate = round(len(missing_fields) / max(len(metrics_to_extract), 1), 4)

    return {
        "normalized_metrics": normalized_metrics,
        "metric_observations": metric_observations,
        "quality_flags": sorted(quality_flags),
        "uncertainty_flags": sorted(uncertainty_flags),
        "missing_fields": sorted(missing_fields),
        "conflicts": conflicts,
        "missing_rate": missing_rate,
        "data_reliability_inputs": {
            "snapshot_confidence": fundamentals_snapshot.get("confidence"),
            "base_confidence_score": round(max(0.0, min(100.0, confidence_score)), 2),
            "conflict_count": len(conflicts),
            "news_available": bool(news_map.get("recent_news")),
        },
    }
