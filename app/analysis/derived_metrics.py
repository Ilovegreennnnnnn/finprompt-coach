from __future__ import annotations

from typing import Any

from .normalization import safe_number


def _metric_record(name: str, value: float | None, explanation: str, unit: str | None = None) -> dict[str, Any]:
    return {
        "metric_name": name,
        "value": None if value is None else round(value, 4),
        "unit": unit,
        "explanation": explanation,
    }


def compute_derived_metrics(normalized_bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = normalized_bundle.get("normalized_metrics", {})
    market_cap = safe_number(metrics.get("market_cap"))
    free_cash_flow = safe_number(metrics.get("free_cash_flow"))
    operating_cash_flow = safe_number(metrics.get("operating_cash_flow"))
    total_cash = safe_number(metrics.get("total_cash"))
    total_debt = safe_number(metrics.get("total_debt"))
    trailing_pe = safe_number(metrics.get("trailing_pe"))
    revenue_growth = safe_number(metrics.get("revenue_growth"))
    period_return = safe_number(metrics.get("period_return_pct"))
    volatility = safe_number(metrics.get("annualized_volatility_pct"))
    payout_ratio = safe_number(metrics.get("payout_ratio"))
    dividend_yield = safe_number(metrics.get("dividend_yield"))

    fcf_yield = None
    if market_cap and market_cap > 0 and free_cash_flow is not None:
        fcf_yield = free_cash_flow / market_cap * 100.0

    cash_conversion = None
    if operating_cash_flow not in (None, 0) and free_cash_flow is not None:
        cash_conversion = free_cash_flow / operating_cash_flow

    cash_to_debt = None
    if total_debt not in (None, 0) and total_cash is not None:
        cash_to_debt = total_cash / total_debt

    net_cash = None
    if total_cash is not None or total_debt is not None:
        net_cash = (total_cash or 0.0) - (total_debt or 0.0)

    valuation_to_growth = None
    if trailing_pe is not None and revenue_growth not in (None, 0) and revenue_growth > 0:
        valuation_to_growth = trailing_pe / revenue_growth

    volatility_adjusted_return = None
    if period_return is not None and volatility not in (None, 0):
        volatility_adjusted_return = period_return / max(volatility, 1.0)

    dividend_support = None
    if dividend_yield is not None and payout_ratio is not None:
        dividend_support = max(0.0, 100.0 - payout_ratio) - max(0.0, dividend_yield - 8.0) * 3.0

    return {
        "free_cash_flow_yield_pct": _metric_record(
            "free_cash_flow_yield_pct",
            fcf_yield,
            "Free cash flow scaled by market capitalization to avoid size-biased absolute comparisons.",
            "%",
        ),
        "cash_conversion_ratio": _metric_record(
            "cash_conversion_ratio",
            cash_conversion,
            "Measures how much operating cash flow is retained after capital intensity.",
            "ratio",
        ),
        "cash_to_debt_ratio": _metric_record(
            "cash_to_debt_ratio",
            cash_to_debt,
            "Higher values imply more balance-sheet flexibility versus outstanding debt.",
            "ratio",
        ),
        "net_cash_position": _metric_record(
            "net_cash_position",
            net_cash,
            "Cash minus debt, useful for spotting balance-sheet fragility or optionality.",
            "USD",
        ),
        "valuation_to_growth": _metric_record(
            "valuation_to_growth",
            valuation_to_growth,
            "Compares earnings multiple to current revenue growth as a rough relative tension gauge.",
            "ratio",
        ),
        "volatility_adjusted_return": _metric_record(
            "volatility_adjusted_return",
            volatility_adjusted_return,
            "Recent return scaled by realized volatility to highlight cleaner market traction.",
            "ratio",
        ),
        "dividend_support_indicator": _metric_record(
            "dividend_support_indicator",
            dividend_support,
            "Heuristic indicator that penalizes stretched payout ratios and suspicious headline yields.",
            "score",
        ),
    }
