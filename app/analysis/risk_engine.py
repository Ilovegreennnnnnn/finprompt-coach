from __future__ import annotations

from typing import Any

from .normalization import safe_number


def _risk_node(name: str, severity: str, evidence: list[str], missing_confirmation: list[str]) -> dict[str, Any]:
    return {
        "risk_name": name,
        "severity": severity,
        "evidence": evidence,
        "missing_confirmation": missing_confirmation,
    }


def build_risk_register(
    *,
    normalized_bundle: dict[str, Any],
    derived_metrics: dict[str, dict[str, Any]],
    peer_baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = normalized_bundle.get("normalized_metrics", {})
    quality_flags = set(normalized_bundle.get("quality_flags", []))
    missing_fields = normalized_bundle.get("missing_fields", [])
    conflicts = normalized_bundle.get("conflicts", [])
    peer_metrics = peer_baseline.get("metrics", {})

    risks: list[dict[str, Any]] = []

    if quality_flags or conflicts:
        risks.append(
            _risk_node(
                "data_reliability_risk",
                "high" if "missing_critical_field" in quality_flags or conflicts else "medium",
                [
                    f"Quality flags: {sorted(quality_flags)}" if quality_flags else "No explicit quality flags.",
                    f"Conflicts detected: {len(conflicts)}.",
                ],
                ["Independent confirmation from filing or cleaner source snapshots."],
            )
        )

    valuation_percentile = peer_metrics.get("trailing_pe", {}).get("percentile")
    price_to_book_percentile = peer_metrics.get("price_to_book", {}).get("percentile")
    valuation_to_growth = safe_number(derived_metrics.get("valuation_to_growth", {}).get("value"))
    if (
        (valuation_percentile is not None and valuation_percentile <= 35)
        or (price_to_book_percentile is not None and price_to_book_percentile <= 35)
        or (valuation_to_growth is not None and valuation_to_growth >= 2.2)
    ):
        risks.append(
            _risk_node(
                "valuation_risk",
                "medium",
                [
                    f"Peer-relative valuation percentile: {valuation_percentile}.",
                    f"Price-to-book percentile: {price_to_book_percentile}.",
                    f"Valuation-to-growth ratio: {valuation_to_growth}.",
                ],
                ["More complete growth durability evidence from management commentary or filings."],
            )
        )

    cash_conversion = safe_number(derived_metrics.get("cash_conversion_ratio", {}).get("value"))
    if cash_conversion is not None and cash_conversion < 0.25:
        risks.append(
            _risk_node(
                "cash_flow_quality_risk",
                "medium",
                [f"Cash conversion ratio was {cash_conversion}, indicating weaker conversion into free cash flow."],
                ["Multi-period capex and operating cash flow trend confirmation."],
            )
        )

    debt_to_equity = safe_number(metrics.get("debt_to_equity"))
    current_ratio = safe_number(metrics.get("current_ratio"))
    cash_to_debt = safe_number(derived_metrics.get("cash_to_debt_ratio", {}).get("value"))
    if (
        (debt_to_equity is not None and debt_to_equity >= 120)
        or (current_ratio is not None and current_ratio < 1.0)
        or (cash_to_debt is not None and cash_to_debt < 0.5)
    ):
        risks.append(
            _risk_node(
                "balance_sheet_risk",
                "high" if debt_to_equity is not None and debt_to_equity >= 180 else "medium",
                [
                    f"Debt to equity: {debt_to_equity}.",
                    f"Current ratio: {current_ratio}.",
                    f"Cash to debt ratio: {cash_to_debt}.",
                ],
                ["Debt maturity profile and liquidity backup facilities."],
            )
        )

    volatility = safe_number(metrics.get("annualized_volatility_pct"))
    revenue_growth = safe_number(metrics.get("revenue_growth"))
    if (volatility is not None and volatility >= 45) or (revenue_growth is not None and revenue_growth < 0):
        risks.append(
            _risk_node(
                "execution_and_market_risk",
                "medium",
                [
                    f"Annualized volatility: {volatility}.",
                    f"Revenue growth: {revenue_growth}.",
                ],
                ["A cleaner catalyst path and better forward growth evidence."],
            )
        )

    dividend_support = safe_number(derived_metrics.get("dividend_support_indicator", {}).get("value"))
    if "unit_conflict" in quality_flags or (dividend_support is not None and dividend_support < 35):
        risks.append(
            _risk_node(
                "capital_return_risk",
                "medium",
                [
                    f"Dividend support indicator: {dividend_support}.",
                    "Reported dividend metrics may be noisy or unsupported.",
                ],
                ["Verified dividend policy and payout sustainability detail."],
            )
        )

    if missing_fields:
        risks.append(
            _risk_node(
                "coverage_gap_risk",
                "low",
                [f"Missing fields: {', '.join(missing_fields[:5])}."],
                ["A more complete snapshot with critical missing fields filled."],
            )
        )

    if not risks:
        risks.append(
            _risk_node(
                "baseline_research_risk",
                "low",
                ["Even stronger dossiers still depend on future execution, market context, and refreshed evidence."],
                ["Longer trend history and management commentary."],
            )
        )

    return risks
