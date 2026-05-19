from __future__ import annotations

from typing import Any


def _average(values: list[float | None], fallback: float = 50.0) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return round(fallback, 2)
    return round(sum(usable) / len(usable), 2)


def _score_entry(value: float, explanation: str, evidence_metrics: list[str]) -> dict[str, Any]:
    return {
        "value": round(max(0.0, min(100.0, value)), 2),
        "explanation": explanation,
        "evidence_metrics": evidence_metrics,
    }


def score_dossier(
    *,
    normalized_bundle: dict[str, Any],
    derived_metrics: dict[str, dict[str, Any]],
    peer_baseline: dict[str, Any],
    risk_register: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    peer_metrics = peer_baseline.get("metrics", {})
    quality_flags = normalized_bundle.get("quality_flags", [])
    conflicts = normalized_bundle.get("conflicts", [])
    missing_fields = normalized_bundle.get("missing_fields", [])
    data_inputs = normalized_bundle.get("data_reliability_inputs", {})

    quality_score = _average(
        [
            peer_metrics.get("operating_margin", {}).get("percentile"),
            peer_metrics.get("profit_margin", {}).get("percentile"),
            peer_metrics.get("return_on_equity", {}).get("percentile"),
        ],
        fallback=52.0,
    )
    valuation_score = _average(
        [
            peer_metrics.get("trailing_pe", {}).get("percentile"),
            peer_metrics.get("price_to_book", {}).get("percentile"),
            peer_metrics.get("free_cash_flow_yield_pct", {}).get("percentile"),
        ],
        fallback=48.0,
    )
    cash_flow_score = _average(
        [
            peer_metrics.get("free_cash_flow_yield_pct", {}).get("percentile"),
            peer_metrics.get("cash_conversion_ratio", {}).get("percentile"),
        ],
        fallback=50.0,
    )
    balance_sheet_score = _average(
        [
            peer_metrics.get("debt_to_equity", {}).get("percentile"),
            peer_metrics.get("current_ratio", {}).get("percentile"),
            peer_metrics.get("cash_to_debt_ratio", {}).get("percentile"),
        ],
        fallback=50.0,
    )
    growth_quality_score = _average(
        [
            peer_metrics.get("revenue_growth", {}).get("percentile"),
            peer_metrics.get("valuation_to_growth", {}).get("percentile"),
        ],
        fallback=50.0,
    )
    market_context_score = _average(
        [
            peer_metrics.get("volatility_adjusted_return", {}).get("percentile"),
            peer_metrics.get("annualized_volatility_pct", {}).get("percentile"),
        ],
        fallback=48.0,
    )

    data_reliability_score = float(data_inputs.get("base_confidence_score", 62.0))
    data_reliability_score -= len(conflicts) * 7.0
    data_reliability_score -= len(missing_fields) * 3.0
    if "missing_critical_field" in quality_flags:
        data_reliability_score -= 10.0
    if "unit_conflict" in quality_flags:
        data_reliability_score -= 8.0
    if "metric_anomaly" in quality_flags:
        data_reliability_score -= 8.0
    data_reliability_score = round(max(5.0, min(100.0, data_reliability_score)), 2)

    opportunity_score = round(
        quality_score * 0.22
        + valuation_score * 0.18
        + cash_flow_score * 0.18
        + balance_sheet_score * 0.14
        + growth_quality_score * 0.12
        + market_context_score * 0.08
        + data_reliability_score * 0.08,
        2,
    )

    severe_risk_penalty = sum(6 for risk in risk_register if risk.get("severity") == "high")
    medium_risk_penalty = sum(3 for risk in risk_register if risk.get("severity") == "medium")
    contradiction_penalty = sum(
        4
        for claim in claims
        if claim.get("contradicting_evidence")
    )
    reasoning_confidence = round(
        max(
            5.0,
            min(
                98.0,
                opportunity_score * 0.55
                + data_reliability_score * 0.45
                - severe_risk_penalty
                - medium_risk_penalty
                - contradiction_penalty,
            ),
        ),
        2,
    )

    return {
        "quality_score": _score_entry(
            quality_score,
            "Peer-relative profitability and return-on-capital indicators anchor the quality score.",
            ["operating_margin", "profit_margin", "return_on_equity"],
        ),
        "valuation_score": _score_entry(
            valuation_score,
            "Lower peer-relative multiples and stronger free-cash-flow yield improve valuation attractiveness.",
            ["trailing_pe", "price_to_book", "free_cash_flow_yield_pct"],
        ),
        "cash_flow_score": _score_entry(
            cash_flow_score,
            "Cash-flow quality relies on both free-cash-flow yield and conversion from operating cash flow.",
            ["free_cash_flow_yield_pct", "cash_conversion_ratio"],
        ),
        "balance_sheet_score": _score_entry(
            balance_sheet_score,
            "Leverage, liquidity, and cash-to-debt coverage shape the balance-sheet score.",
            ["debt_to_equity", "current_ratio", "cash_to_debt_ratio"],
        ),
        "growth_quality_score": _score_entry(
            growth_quality_score,
            "Growth quality compares current growth support with how expensive that growth already looks.",
            ["revenue_growth", "valuation_to_growth"],
        ),
        "market_context_score": _score_entry(
            market_context_score,
            "Market context rewards cleaner momentum and penalizes unstable volatility regimes.",
            ["volatility_adjusted_return", "annualized_volatility_pct"],
        ),
        "data_reliability_score": _score_entry(
            data_reliability_score,
            "Reliability falls when critical fields are missing, conflicts appear, or anomaly flags accumulate.",
            ["quality_flags", "conflicts", "missing_fields"],
        ),
        "opportunity_score": round(opportunity_score, 2),
        "reasoning_confidence": reasoning_confidence,
    }
