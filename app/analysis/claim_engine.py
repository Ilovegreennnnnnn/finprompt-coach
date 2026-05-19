from __future__ import annotations

from typing import Any

from .normalization import safe_number


def _evidence_node(
    *,
    evidence_id: str,
    label: str,
    metric_name: str,
    value: Any,
    source_name: str,
    snapshot_timestamp: str | None,
    note: str,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "label": label,
        "metric_name": metric_name,
        "value": value,
        "source_name": source_name,
        "snapshot_timestamp": snapshot_timestamp,
        "note": note,
    }


def _claim_node(
    *,
    claim_id: str,
    claim: str,
    stance: str,
    confidence: float,
    supporting_evidence: list[dict[str, Any]],
    contradicting_evidence: list[dict[str, Any]],
    open_questions: list[str],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "stance": stance,
        "confidence": round(max(10.0, min(95.0, confidence)), 2),
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "open_questions": open_questions,
    }


def _find_observation(metric_observations: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    for observation in metric_observations:
        if observation.get("metric_name") == metric_name:
            return observation
    return {}


def build_claims(
    *,
    ticker: str,
    normalized_bundle: dict[str, Any],
    derived_metrics: dict[str, dict[str, Any]],
    peer_baseline: dict[str, Any],
    risk_register: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = normalized_bundle.get("normalized_metrics", {})
    metric_observations = normalized_bundle.get("metric_observations", [])
    peer_metrics = peer_baseline.get("metrics", {})
    quality_flags = set(normalized_bundle.get("quality_flags", []))
    conflicts = normalized_bundle.get("conflicts", [])
    missing_fields = normalized_bundle.get("missing_fields", [])

    claims: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    operating_margin = peer_metrics.get("operating_margin", {}).get("percentile")
    profit_margin = peer_metrics.get("profit_margin", {}).get("percentile")
    if operating_margin is not None or profit_margin is not None:
        supporting = []
        for metric_name in ("operating_margin", "profit_margin"):
            observation = _find_observation(metric_observations, metric_name)
            if observation:
                evidence_node = _evidence_node(
                    evidence_id=f"{ticker.lower()}_{metric_name}_quality",
                    label=f"{metric_name.replace('_', ' ').title()} supports quality",
                    metric_name=metric_name,
                    value=observation.get("value"),
                    source_name=observation.get("source_name"),
                    snapshot_timestamp=observation.get("snapshot_timestamp"),
                    note=f"Peer percentile: {peer_metrics.get(metric_name, {}).get('percentile')}.",
                )
                evidence.append(evidence_node)
                supporting.append(evidence_node)

        confidence = 68.0 + ((operating_margin or 50.0) + (profit_margin or 50.0)) / 8.0 - len(conflicts) * 3.0
        claims.append(
            _claim_node(
                claim_id=f"{ticker.lower()}_quality_relative",
                claim="Quality is above peers." if (operating_margin or 0) >= 60 or (profit_margin or 0) >= 60 else "Quality looks mixed versus peers.",
                stance="positive" if (operating_margin or 0) >= 60 or (profit_margin or 0) >= 60 else "neutral",
                confidence=confidence,
                supporting_evidence=supporting,
                contradicting_evidence=[],
                open_questions=["Are the margin advantages durable across the next cycle?"],
            )
        )

    valuation_percentile = peer_metrics.get("trailing_pe", {}).get("percentile")
    price_to_book_percentile = peer_metrics.get("price_to_book", {}).get("percentile")
    valuation_support = []
    for metric_name in ("trailing_pe", "price_to_book"):
        observation = _find_observation(metric_observations, metric_name)
        if observation:
            evidence_node = _evidence_node(
                evidence_id=f"{ticker.lower()}_{metric_name}_valuation",
                label=f"{metric_name.replace('_', ' ').title()} peer anchor",
                metric_name=metric_name,
                value=observation.get("value"),
                source_name=observation.get("source_name"),
                snapshot_timestamp=observation.get("snapshot_timestamp"),
                note=f"Peer percentile: {peer_metrics.get(metric_name, {}).get('percentile')}.",
            )
            evidence.append(evidence_node)
            valuation_support.append(evidence_node)

    valuation_claim = "Valuation remains demanding versus the peer set."
    stance = "negative"
    if (valuation_percentile or 0) >= 60 or (price_to_book_percentile or 0) >= 60:
        valuation_claim = "Valuation looks more manageable than much of the peer set."
        stance = "positive"
    claims.append(
        _claim_node(
            claim_id=f"{ticker.lower()}_valuation_relative",
            claim=valuation_claim,
            stance=stance,
            confidence=72.0 - len(conflicts) * 4.0,
            supporting_evidence=valuation_support,
            contradicting_evidence=[],
            open_questions=["How much of the current multiple already prices in forward growth?"],
        )
    )

    cash_conversion = safe_number(derived_metrics.get("cash_conversion_ratio", {}).get("value"))
    fcf_yield = safe_number(derived_metrics.get("free_cash_flow_yield_pct", {}).get("value"))
    supporting = []
    for metric_name in ("free_cash_flow", "operating_cash_flow"):
        observation = _find_observation(metric_observations, metric_name)
        if observation:
            evidence_node = _evidence_node(
                evidence_id=f"{ticker.lower()}_{metric_name}_cashflow",
                label=f"{metric_name.replace('_', ' ').title()} cash-flow anchor",
                metric_name=metric_name,
                value=observation.get("value"),
                source_name=observation.get("source_name"),
                snapshot_timestamp=observation.get("snapshot_timestamp"),
                note=f"Derived cash conversion ratio: {cash_conversion}.",
            )
            evidence.append(evidence_node)
            supporting.append(evidence_node)
    claim_text = "Cash-flow support is strong."
    stance = "positive"
    if cash_conversion is not None and cash_conversion < 0.25:
        claim_text = "Cash-flow support is present but conversion quality is weak."
        stance = "neutral"
    claims.append(
        _claim_node(
            claim_id=f"{ticker.lower()}_cashflow_support",
            claim=claim_text,
            stance=stance,
            confidence=(70.0 if fcf_yield is not None else 55.0) - len(conflicts) * 4.0,
            supporting_evidence=supporting,
            contradicting_evidence=[],
            open_questions=["Does free cash flow remain resilient after a more normal capex cycle?"],
        )
    )

    news_count = int(metrics.get("recent_news_count") or 0)
    open_questions = []
    if news_count == 0:
        open_questions.append("Recent catalyst coverage is thin and should be verified externally.")
    if missing_fields:
        open_questions.append("Some missing fields limit the certainty of the current thesis.")
    if quality_flags:
        open_questions.append("One or more quality flags require explicit caution in any narrative.")
    claims.append(
        _claim_node(
            claim_id=f"{ticker.lower()}_confidence_boundary",
            claim="Confidence should remain qualified and evidence-led.",
            stance="neutral",
            confidence=55.0 - len(risk_register) * 2.0 + max(0, 3 - len(conflicts)) * 3.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            open_questions=open_questions or ["What additional source would most improve confidence?"],
        )
    )

    return claims, evidence
