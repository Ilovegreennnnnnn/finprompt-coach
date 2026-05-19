from __future__ import annotations

from typing import Any

from .normalization import safe_number


PEER_METRIC_CONFIG = {
    "trailing_pe": "lower_better",
    "price_to_book": "lower_better",
    "operating_margin": "higher_better",
    "profit_margin": "higher_better",
    "return_on_equity": "higher_better",
    "revenue_growth": "higher_better",
    "debt_to_equity": "lower_better",
    "current_ratio": "higher_better",
    "period_return_pct": "higher_better",
    "annualized_volatility_pct": "lower_better",
    "free_cash_flow_yield_pct": "higher_better",
    "cash_conversion_ratio": "higher_better",
    "cash_to_debt_ratio": "higher_better",
    "valuation_to_growth": "lower_better",
    "volatility_adjusted_return": "higher_better",
}


def _rank_metric(values: list[tuple[str, float]], ticker: str, direction: str) -> dict[str, Any]:
    if not values:
        return {
            "value": None,
            "rank": None,
            "sample_size": 0,
            "percentile": None,
            "peer_median": None,
            "peer_average": None,
            "direction": direction,
        }

    reverse = direction == "higher_better"
    ordered = sorted(values, key=lambda item: item[1], reverse=reverse)
    sample_size = len(ordered)
    target_value = None
    rank = None

    for index, (symbol, value) in enumerate(ordered, start=1):
        if symbol == ticker:
            target_value = value
            rank = index
            break

    if target_value is None or rank is None:
        return {
            "value": None,
            "rank": None,
            "sample_size": sample_size,
            "percentile": None,
            "peer_median": None,
            "peer_average": None,
            "direction": direction,
        }

    percentile = 100.0 if sample_size == 1 else round((sample_size - rank) / (sample_size - 1) * 100.0, 2)
    numeric_values = [value for _, value in ordered]
    midpoint = sample_size // 2
    if sample_size % 2 == 1:
        median = numeric_values[midpoint]
    else:
        median = (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2

    return {
        "value": round(target_value, 4),
        "rank": rank,
        "sample_size": sample_size,
        "percentile": percentile,
        "peer_median": round(median, 4),
        "peer_average": round(sum(numeric_values) / sample_size, 4),
        "direction": direction,
    }


def _peer_group(
    *,
    ticker: str,
    normalized_by_ticker: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = normalized_by_ticker.get(ticker, {})
    industry = normalized.get("industry")
    sector = normalized.get("sector")

    industry_members = [
        symbol
        for symbol, item in normalized_by_ticker.items()
        if item.get("industry") and item.get("industry") == industry
    ]
    if len(industry_members) >= 2:
        return {
            "scope": "industry",
            "members": industry_members,
            "sector": sector,
            "industry": industry,
            "sample_size": len(industry_members),
        }

    sector_members = [
        symbol
        for symbol, item in normalized_by_ticker.items()
        if item.get("sector") and item.get("sector") == sector
    ]
    if len(sector_members) >= 2:
        return {
            "scope": "sector",
            "members": sector_members,
            "sector": sector,
            "industry": industry,
            "sample_size": len(sector_members),
        }

    return {
        "scope": "watchlist",
        "members": sorted(normalized_by_ticker.keys()),
        "sector": sector,
        "industry": industry,
        "sample_size": len(normalized_by_ticker),
    }


def build_peer_baselines(
    *,
    normalized_by_ticker: dict[str, dict[str, Any]],
    derived_by_ticker: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for ticker in normalized_by_ticker:
        peer_group = _peer_group(ticker=ticker, normalized_by_ticker=normalized_by_ticker)
        members = peer_group["members"]
        metrics: dict[str, Any] = {}

        for metric_name, direction in PEER_METRIC_CONFIG.items():
            values: list[tuple[str, float]] = []
            for symbol in members:
                normalized = normalized_by_ticker.get(symbol, {})
                if metric_name in derived_by_ticker.get(symbol, {}):
                    candidate = safe_number(derived_by_ticker[symbol][metric_name].get("value"))
                else:
                    candidate = safe_number(normalized.get(metric_name))
                if candidate is not None:
                    values.append((symbol, candidate))

            metrics[metric_name] = _rank_metric(values, ticker, direction)

        results[ticker] = {
            "peer_group": peer_group,
            "metrics": metrics,
        }

    return results
