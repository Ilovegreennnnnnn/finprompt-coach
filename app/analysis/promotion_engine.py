from __future__ import annotations

from typing import Any


def determine_promotion_decision(
    *,
    score_breakdown: dict[str, Any],
    normalized_bundle: dict[str, Any],
    risk_register: list[dict[str, Any]],
    external_warnings: list[str],
) -> dict[str, Any]:
    opportunity_score = float(score_breakdown.get("opportunity_score", 0.0) or 0.0)
    data_reliability_score = float(
        (score_breakdown.get("data_reliability_score", {}) or {}).get("value", 0.0) or 0.0
    )
    quality_flags = set(normalized_bundle.get("quality_flags", []))
    conflicts = normalized_bundle.get("conflicts", [])
    high_risk_count = sum(1 for risk in risk_register if risk.get("severity") == "high")

    reasons: list[str] = []

    if {"metric_anomaly", "unit_conflict", "missing_critical_field"} & quality_flags and data_reliability_score < 45:
        reasons.append("Critical quality issues materially reduced data reliability.")
        return {
            "decision": "discard_noisy_case",
            "reasons": reasons,
        }

    if conflicts:
        reasons.append(f"{len(conflicts)} explicit source or plausibility conflict(s) were detected.")
    if external_warnings:
        reasons.append("Live collection surfaced warnings that deserve replay and evaluation.")
    if high_risk_count:
        reasons.append("The dossier contains at least one high-severity risk that should be observed carefully.")

    if opportunity_score >= 68 or conflicts or external_warnings:
        if data_reliability_score >= 45:
            reasons.append("The case is useful for the Prompt Lab because it is either attractive, ambiguous, or noisy in an informative way.")
            return {
                "decision": "promote_to_lab",
                "reasons": reasons,
            }

    reasons.append("The case is worth keeping in view, but it does not currently justify benchmark promotion.")
    return {
        "decision": "observe_only",
        "reasons": reasons,
    }
