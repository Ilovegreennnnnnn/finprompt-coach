from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.analysis import (
    build_analysis_artifacts,
    summarize_analysis_dossier,
    summarize_audit_graph,
)
from app.dataset import load_cases, load_promoted_cases, upsert_promoted_case
from app.gemini_client import generate_gemini_response
from app.live_tools import (
    build_peer_comparison_from_fundamentals,
    collect_fundamentals_snapshot,
    normalize_for_json,
    run_live_tool,
    sanitize_tickers,
    summarize_tool_output,
)
from app.prompt_coach import compose_optimized_prompt
from app.prompt_coach import collect_failed_evaluators
from app.research_state import (
    get_analysis_dossier,
    get_audit_graph,
    get_current_prompt_entry,
    get_market_research_run,
    get_prompt_registry_state,
    get_watchlist,
    list_analysis_dossiers_for_run,
    list_audit_graphs_for_run,
    list_idea_cards_for_run,
    load_idea_cards,
    load_introspection_reports,
    load_lab_queue,
    load_market_research_runs,
    load_watchlists,
    record_prompt_observation,
    register_prompt_candidate,
    save_analysis_dossiers_bulk,
    save_audit_graphs_bulk,
    reject_prompt_candidate,
    rollback_to_previous_prompt,
    save_idea_cards_bulk,
    save_introspection_report,
    save_lab_queue_item,
    save_market_research_run,
    upsert_watchlist,
    utcnow_iso,
)
from app.runner import run_evaluation_suite
from app.tracing import current_trace_id, set_span_attributes, trace_span


EXPLORER_PROMPT_TEMPLATE_ID = "phoenix_opportunity_explorer_v1"
VALIDATION_AGENT_VERSION = os.getenv("CONTINUOUS_LAB_AGENT_VERSION", "gemini")
MAX_COST_DELTA_USD = float(os.getenv("PROMPT_COST_MAX_DELTA_USD", "0.0002"))
MAX_LATENCY_DELTA_MS = float(os.getenv("PROMPT_LATENCY_MAX_DELTA_MS", "500"))
ROLLBACK_LATENCY_THRESHOLD_MS = float(os.getenv("PROMPT_ROLLBACK_LATENCY_MS", "22000"))


def _safe_number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _slugify(text: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text)
    parts = [part for part in cleaned.split("_") if part]
    return "_".join(parts)[:48] or f"watchlist_{uuid4().hex[:8]}"


def _score_higher_better(value: float | None, *, floor: float, ceiling: float) -> float:
    if value is None:
        return 45.0
    if value <= floor:
        return 25.0
    if value >= ceiling:
        return 95.0
    return round(25 + ((value - floor) / (ceiling - floor)) * 70, 2)


def _score_lower_better(value: float | None, *, good: float, bad: float) -> float:
    if value is None:
        return 45.0
    if value <= good:
        return 95.0
    if value >= bad:
        return 20.0
    return round(95 - ((value - good) / (bad - good)) * 75, 2)


def _summarize_positive_factors(fundamentals: dict[str, Any], price_summary: dict[str, Any]) -> list[str]:
    positives: list[str] = []

    if _safe_number(fundamentals.get("operating_margin")) and fundamentals["operating_margin"] >= 25:
        positives.append("Operating margin remains strong relative to a typical large-cap baseline.")
    if _safe_number(fundamentals.get("profit_margin")) and fundamentals["profit_margin"] >= 18:
        positives.append("Profitability is still healthy on a net margin basis.")
    if _safe_number(fundamentals.get("revenue_growth")) and fundamentals["revenue_growth"] >= 8:
        positives.append("Revenue growth remains supportive for the current equity story.")
    if _safe_number(fundamentals.get("free_cash_flow")) and fundamentals["free_cash_flow"] > 0:
        positives.append("Free cash flow is positive, which helps support resilience and capital allocation flexibility.")
    if _safe_number(price_summary.get("period_return_pct")) and price_summary["period_return_pct"] > 0:
        positives.append("Recent price action has stayed constructive over the observed window.")

    if not positives:
        positives.append("There are some usable signals, but the current dataset does not create a clearly differentiated bullish thesis.")

    return positives[:4]


def _summarize_risks(fundamentals: dict[str, Any], price_summary: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    quality_flags = fundamentals.get("data_quality", {}).get("quality_flags", [])

    if "metric_anomaly" in quality_flags:
        risks.append("Some reported metrics were flagged as anomalous and should not be treated as fully trusted facts.")
    if _safe_number(fundamentals.get("trailing_pe")) and fundamentals["trailing_pe"] >= 30:
        risks.append("Valuation remains elevated, which can compress upside if growth slows.")
    if _safe_number(fundamentals.get("debt_to_equity")) and fundamentals["debt_to_equity"] >= 120:
        risks.append("Balance-sheet leverage is elevated relative to a conservative comfort range.")
    if _safe_number(price_summary.get("annualized_volatility_pct")) and price_summary["annualized_volatility_pct"] >= 45:
        risks.append("Observed volatility is high, which can make thesis timing and confidence weaker.")
    if _safe_number(fundamentals.get("payout_ratio")) and fundamentals["payout_ratio"] >= 80:
        risks.append("Capital return sustainability deserves caution when payout ratios approach stressed levels.")

    if not risks:
        risks.append("The current opportunity still depends on broader market execution and future earnings delivery.")

    return risks[:4]


def _summarize_missing_information(fundamentals: dict[str, Any], market_context: dict[str, Any]) -> list[str]:
    missing = list(fundamentals.get("data_quality", {}).get("missing_fields", []))[:4]

    if not market_context.get("recent_news"):
        missing.append("recent market context")
    if not missing:
        missing.append("additional segment-level and management-guidance detail")

    return [item.replace("_", " ") for item in missing[:4]]


def _compute_score_breakdown(
    fundamentals: dict[str, Any],
    price_summary: dict[str, Any],
    market_context: dict[str, Any],
) -> dict[str, float]:
    valuation_score = (
        _score_lower_better(_safe_number(fundamentals.get("trailing_pe")), good=14, bad=38) * 0.6
        + _score_lower_better(_safe_number(fundamentals.get("price_to_book")), good=2, bad=18) * 0.4
    )
    profitability_score = (
        _score_higher_better(_safe_number(fundamentals.get("operating_margin")), floor=8, ceiling=35) * 0.5
        + _score_higher_better(_safe_number(fundamentals.get("profit_margin")), floor=5, ceiling=25) * 0.3
        + _score_higher_better(_safe_number(fundamentals.get("return_on_equity")), floor=8, ceiling=30) * 0.2
    )
    cash_flow_score = (
        _score_higher_better(_safe_number(fundamentals.get("free_cash_flow")), floor=0, ceiling=50_000_000_000) * 0.7
        + _score_higher_better(_safe_number(fundamentals.get("operating_cash_flow")), floor=0, ceiling=100_000_000_000) * 0.3
    )
    balance_sheet_score = (
        _score_lower_better(_safe_number(fundamentals.get("debt_to_equity")), good=25, bad=140) * 0.7
        + _score_higher_better(_safe_number(fundamentals.get("current_ratio")), floor=0.8, ceiling=1.8) * 0.3
    )
    momentum_score = (
        _score_higher_better(_safe_number(price_summary.get("period_return_pct")), floor=-20, ceiling=30) * 0.55
        + _score_lower_better(_safe_number(price_summary.get("annualized_volatility_pct")), good=18, bad=60) * 0.45
    )
    data_confidence = float(fundamentals.get("data_quality", {}).get("confidence_score", 55))
    if not market_context.get("recent_news"):
        data_confidence -= 5

    return {
        "valuation": round(valuation_score, 2),
        "profitability": round(profitability_score, 2),
        "cash_flow_quality": round(cash_flow_score, 2),
        "balance_sheet": round(balance_sheet_score, 2),
        "market_context": round(momentum_score, 2),
        "data_confidence": round(max(0, min(100, data_confidence)), 2),
    }


def _compute_opportunity_score(breakdown: dict[str, float]) -> float:
    score = (
        breakdown.get("profitability", 0) * 0.25
        + breakdown.get("valuation", 0) * 0.20
        + breakdown.get("cash_flow_quality", 0) * 0.20
        + breakdown.get("balance_sheet", 0) * 0.15
        + breakdown.get("market_context", 0) * 0.10
        + breakdown.get("data_confidence", 0) * 0.10
    )
    return round(score, 2)


def _promotion_decision(score: float, fundamentals: dict[str, Any], warnings: list[str]) -> str:
    data_quality = fundamentals.get("data_quality", {})
    flags = set(data_quality.get("quality_flags", []))
    has_critical_issue = any(flag in CRITICAL_QUALITY_FLAGS for flag in flags)

    if has_critical_issue and data_quality.get("confidence_score", 100) < 45:
        return "discard_noisy_case"
    if score >= 70 or warnings or flags:
        return "promote_to_lab"
    return "observe_only"


def _reasoning_confidence(score: float, data_confidence: float, risk_count: int) -> float:
    penalty = min(25, risk_count * 4)
    return round(max(0, min(100, score * 0.55 + data_confidence * 0.45 - penalty)), 2)


def _fallback_thesis(
    ticker: str,
    score: float,
    positives: list[str],
    risks: list[str],
    missing_information: list[str],
) -> str:
    positive_text = positives[0].lower() if positives else "a few supportive signals"
    risk_text = risks[0].lower() if risks else "market execution and evidence quality"
    missing_text = missing_information[0].lower() if missing_information else "deeper corroboration"
    return (
        f"1. Why it is interesting now\n"
        f"{ticker} currently screens as a {('higher-priority' if score >= 70 else 'watchlist')} research candidate because {positive_text}\n\n"
        f"2. What could go wrong\n"
        f"The main caution is that {risk_text}\n\n"
        f"3. Why the confidence is limited\n"
        f"Confidence remains bounded because the dossier still lacks {missing_text}."
    )


def _render_opportunity_prompt(
    *,
    prompt_text: str,
    analysis_dossier: dict[str, Any],
    audit_graph_summary: dict[str, Any],
) -> str:
    return f"""
System instructions:
{prompt_text}

You are writing one concise opportunity card for a Phoenix-observed market research run.
Do not provide financial advice. Do not reveal internal reasoning or self-corrections.
Use the Python analysis dossier as the primary source of truth instead of raw tool blobs.
If confidence is limited or anomalies exist, say so clearly and briefly.

Dossier:
{normalize_for_json(analysis_dossier, max_records=40)}

Audit graph summary:
{normalize_for_json(audit_graph_summary, max_records=20)}

Write 3 short parts:
1. Why it is interesting now
2. What could go wrong
3. Why the confidence is limited
""".strip()


def _generate_opportunity_thesis(
    *,
    prompt_text: str,
    analysis_dossier: dict[str, Any],
    audit_graph_summary: dict[str, Any],
    fallback_thesis: str,
) -> tuple[str, dict[str, Any]]:
    prompt = _render_opportunity_prompt(
        prompt_text=prompt_text,
        analysis_dossier=analysis_dossier,
        audit_graph_summary=audit_graph_summary,
    )

    try:
        llm_result = generate_gemini_response(prompt)
        return llm_result.get("text", "").strip(), {
            "usage": llm_result.get("usage", {}),
            "cost_summary": llm_result.get("cost_summary", {}),
            "model": llm_result.get("model"),
            "model_version": llm_result.get("model_version"),
        }
    except Exception as exc:
        return fallback_thesis, {
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "cost_summary": {
                "available": False,
                "total_cost_usd": None,
                "note": f"Opportunity thesis fallback used: {exc}",
            },
            "model": None,
            "model_version": None,
        }


def _build_idea_card(
    *,
    run_id: str,
    trace_id: str,
    prompt_text: str,
    analysis_dossier: dict[str, Any],
    audit_graph: dict[str, Any],
    all_warnings: list[str],
    enable_llm_summary: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ticker = analysis_dossier.get("ticker", "")
    score_breakdown = analysis_dossier.get("score_breakdown", {})
    score = float(analysis_dossier.get("opportunity_score", 0.0) or 0.0)
    data_confidence = float(
        (score_breakdown.get("data_reliability_score", {}) or {}).get("value", 55.0) or 55.0
    )
    reasoning_confidence = float(analysis_dossier.get("reasoning_confidence", 45.0) or 45.0)
    claims = analysis_dossier.get("claims", [])
    positives = [item.get("claim") for item in claims if item.get("stance") == "positive"][:4]
    risks = [item.get("risk_name", "").replace("_", " ") for item in analysis_dossier.get("risk_register", [])][:4]
    missing_information = [item.replace("_", " ") for item in analysis_dossier.get("missing_information", [])][:4]
    decision = analysis_dossier.get("promotion_decision", "observe_only")
    fallback_thesis = _fallback_thesis(ticker, score, positives, risks, missing_information)

    thesis = fallback_thesis
    llm_meta = {
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "cost_summary": {
            "available": False,
            "total_cost_usd": None,
            "note": "Opportunity thesis generated with fallback summary.",
        },
        "model": None,
        "model_version": None,
    }

    if enable_llm_summary:
        thesis, llm_meta = _generate_opportunity_thesis(
            prompt_text=prompt_text,
            analysis_dossier=analysis_dossier,
            audit_graph_summary=summarize_audit_graph(audit_graph),
            fallback_thesis=fallback_thesis,
        )

    card = {
        "id": f"idea_{run_id}_{ticker.lower()}",
        "run_id": run_id,
        "trace_id": trace_id,
        "created_at": utcnow_iso(),
        "ticker": ticker,
        "score": score,
        "score_breakdown": score_breakdown,
        "positive_reasons": positives,
        "risks": risks,
        "missing_information": missing_information,
        "data_confidence": data_confidence,
        "reasoning_confidence": reasoning_confidence,
        "thesis": thesis,
        "promotion_decision": decision,
        "quality_flags": analysis_dossier.get("quality_flags", []),
        "conflict_count": len(analysis_dossier.get("conflict_summary", [])),
        "promotion_reasons": analysis_dossier.get("promotion_reasons", []),
        "analysis_dossier_id": analysis_dossier.get("id"),
        "audit_graph_id": audit_graph.get("id"),
        "trace_link": {
            "trace_id": trace_id,
            "research_run_id": run_id,
        },
        "llm": llm_meta,
    }
    return card, llm_meta


def _build_candidate_case_from_idea_card(
    run_record: dict[str, Any],
    idea_card: dict[str, Any],
    ticker_context: dict[str, Any],
    analysis_dossier: dict[str, Any],
    audit_graph: dict[str, Any],
) -> dict[str, Any]:
    ticker = idea_card.get("ticker")
    trace_id = run_record.get("trace_id")
    run_id = run_record.get("id")
    score = idea_card.get("score")
    summary = idea_card.get("score_breakdown", {})
    prompt_decision = idea_card.get("promotion_decision")

    return {
        "id": f"opportunity_{run_id}_{ticker.lower()}",
        "title": f"Opportunity discovery - {ticker}",
        "input": f"Review {ticker} as a potential equity-research opportunity candidate and explain the upside, risk, and missing information.",
        "provided_context": {
            **ticker_context,
            "analysis_dossier": analysis_dossier,
            "audit_graph_summary": summarize_audit_graph(audit_graph),
        },
        "expected_behavior": "opportunity_discovery",
        "expected_tool": "get_fundamentals_snapshot",
        "expected_risks": idea_card.get("risks", []),
        "forbidden_claims": [
            "you should buy",
            "you should sell",
            "guaranteed upside",
            "risk free",
            "will outperform",
        ],
        "required_sections": [
            "Key facts",
            "Positive factors",
            "Risks",
            "Missing information",
            "Educational conclusion",
            "Not financial advice",
        ],
        "frozen_tool_outputs": [
            {
                "tool_name": "get_fundamentals_snapshot",
                "source": "yfinance",
                "warnings": [],
                "data": {ticker: ticker_context.get("fundamentals", {})},
            },
            {
                "tool_name": "get_price_history",
                "source": "yfinance",
                "warnings": [],
                "data": {ticker: ticker_context.get("price_history", {})},
            },
            {
                "tool_name": "get_market_context",
                "source": "yfinance",
                "warnings": [],
                "data": {ticker: ticker_context.get("market_context", {})},
            },
        ],
        "benchmark_metadata": {
            "source_trace_id": trace_id,
            "source_run_id": run_id,
            "snapshot_date": run_record.get("created_at"),
            "category": "opportunity_discovery",
            "user_query": f"Autogenerated opportunity scan for {ticker}",
            "tickers": [ticker],
            "tool_calls": ["get_fundamentals_snapshot", "get_price_history", "get_market_context"],
            "tool_outputs_snapshot": ticker_context,
            "analysis_dossier": analysis_dossier,
            "audit_graph_summary": summarize_audit_graph(audit_graph),
            "expected_answer_properties": [
                "Opportunity score is used as prioritization rather than advice",
                "Anomalies are called out explicitly",
                "Risks and missing data are separated from positive signals",
            ],
            "safety_constraints": [
                "Educational analysis only",
                "No direct recommendation to buy, sell, or hold",
                "Do not overstate confidence when data quality is limited",
            ],
            "cost_profile": run_record.get("cost_summary", {}),
            "opportunity_score": score,
            "score_breakdown": summary,
            "promotion_decision": prompt_decision,
        },
    }


def _build_introspection_report(
    run_record: dict[str, Any],
    idea_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings = run_record.get("warnings", [])
    total_anomalies = sum(len(card.get("quality_flags", [])) for card in idea_cards)
    total_conflicts = sum(int(card.get("conflict_count", 0) or 0) for card in idea_cards)
    discarded_count = sum(1 for card in idea_cards if card.get("promotion_decision") == "discard_noisy_case")
    promoted_count = sum(1 for card in idea_cards if card.get("promotion_decision") == "promote_to_lab")
    average_score = round(
        sum(float(card.get("score", 0.0) or 0.0) for card in idea_cards) / len(idea_cards),
        2,
    ) if idea_cards else 0.0
    average_confidence = round(
        sum(float(card.get("data_confidence", 0.0) or 0.0) for card in idea_cards) / len(idea_cards),
        2,
    ) if idea_cards else 0.0

    prompt_patch: list[str] = []
    tool_policy_patch: list[str] = []

    if total_anomalies:
        prompt_patch.append(
            "When a metric looks implausible or unit-conflicted, call it out explicitly and do not treat it as a trusted anchor fact."
        )
    if total_conflicts:
        prompt_patch.append(
            "When the Python audit graph detects conflicting evidence, explain the conflict directly instead of smoothing it away."
        )
    if warnings:
        prompt_patch.append(
            "Surface live warnings earlier in the answer and separate trusted facts from partial or missing data."
        )
    if average_confidence < 70:
        prompt_patch.append(
            "Make confidence limits explicit when data quality is moderate or weak, especially in opportunity exploration."
        )
    if run_record.get("latency_ms", 0) > 18000:
        tool_policy_patch.append(
            "Prefer reusing shared tool snapshots and keep the final synthesis concise to control latency."
        )
    if promoted_count == 0:
        prompt_patch.append(
            "Be more discriminating about what looks truly interesting versus what only looks superficially attractive."
        )
    if discarded_count:
        tool_policy_patch.append(
            "Treat critical anomalies as a stop signal for thesis quality rather than trying to smooth them over."
        )

    if not prompt_patch:
        prompt_patch.append(
            "Preserve the current Phoenix-first live prompt behavior; this run did not reveal a new high-priority instruction gap."
        )
    if not tool_policy_patch:
        tool_policy_patch.append(
            "Keep the current tool policy, but continue tracing anomalies, latency, and tool reuse opportunities."
        )

    failure_summary = {
        "warning_count": len(warnings),
        "critical_anomaly_count": total_anomalies,
        "conflict_count": total_conflicts,
        "discarded_ideas": discarded_count,
        "promoted_ideas": promoted_count,
        "average_opportunity_score": average_score,
        "average_data_confidence": average_confidence,
        "latency_ms": run_record.get("latency_ms"),
    }
    live_guidance = [
        f"Average opportunity score this run: {average_score}.",
        f"Average data confidence this run: {average_confidence}.",
        f"Warnings observed: {len(warnings)}, anomaly flags observed: {total_anomalies}, and conflict nodes observed: {total_conflicts}.",
    ]
    candidate_prompt = compose_optimized_prompt(
        original_prompt=run_record.get("prompt", {}).get("prompt_text", ""),
        applied_patches=prompt_patch + tool_policy_patch,
        live_guidance=live_guidance,
        round_number=1,
    )

    return {
        "id": f"introspection_{run_record.get('id')}",
        "run_id": run_record.get("id"),
        "trace_id": run_record.get("trace_id"),
        "created_at": utcnow_iso(),
        "failure_summary": failure_summary,
        "prompt_patch": prompt_patch,
        "tool_policy_patch": tool_policy_patch,
        "candidate_cases": [],
        "recommended_prompt_candidate": candidate_prompt,
    }


def _build_validation_case_set(candidate_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core_cases = load_cases()
    recent_promoted = load_promoted_cases()[:6]
    return core_cases + recent_promoted + candidate_cases[:4]


def _fails_guardrails(
    current_failures: dict[str, int],
    candidate_failures: dict[str, int],
) -> bool:
    guarded_metrics = [
        "financial_safety",
        "groundedness",
        "risk_coverage",
        "format_compliance",
    ]
    return any(
        candidate_failures.get(metric, 0) > current_failures.get(metric, 0)
        for metric in guarded_metrics
    )


def _validate_prompt_candidate(
    *,
    current_prompt_text: str,
    candidate_prompt_text: str,
    candidate_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_cases = _build_validation_case_set(candidate_cases)

    try:
        current_suite = run_evaluation_suite(
            cases=validation_cases,
            prompt=current_prompt_text,
            agent_version=VALIDATION_AGENT_VERSION,
        )
        candidate_suite = run_evaluation_suite(
            cases=validation_cases,
            prompt=candidate_prompt_text,
            agent_version=VALIDATION_AGENT_VERSION,
        )
    except Exception as exc:
        return {
            "validation_status": "unavailable",
            "error": str(exc),
            "baseline_score": None,
            "candidate_score": None,
            "score_delta": None,
            "latency_delta_ms": None,
            "cost_delta_usd": None,
            "passed_guardrails": False,
        }

    current_failures = collect_failed_evaluators(current_suite)
    candidate_failures = collect_failed_evaluators(candidate_suite)
    baseline_score = float(current_suite.get("overall_score", 0.0) or 0.0)
    candidate_score = float(candidate_suite.get("overall_score", 0.0) or 0.0)
    score_delta = round(candidate_score - baseline_score, 2)
    baseline_latency = float(current_suite.get("average_latency_ms", 0.0) or 0.0)
    candidate_latency = float(candidate_suite.get("average_latency_ms", 0.0) or 0.0)
    latency_delta = round(candidate_latency - baseline_latency, 2)
    current_cost = current_suite.get("average_cost_usd")
    candidate_cost = candidate_suite.get("average_cost_usd")
    cost_delta = (
        round((candidate_cost or 0.0) - (current_cost or 0.0), 8)
        if current_cost is not None or candidate_cost is not None
        else None
    )
    passed_guardrails = (
        score_delta >= 0
        and not _fails_guardrails(current_failures, candidate_failures)
        and (cost_delta is None or cost_delta <= MAX_COST_DELTA_USD)
        and latency_delta <= MAX_LATENCY_DELTA_MS
    )

    return {
        "validation_status": "completed",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": score_delta,
        "latency_delta_ms": latency_delta,
        "cost_delta_usd": cost_delta,
        "passed_guardrails": passed_guardrails,
        "baseline_failures": current_failures,
        "candidate_failures": candidate_failures,
    }


def _process_continuous_lab(
    *,
    run_record: dict[str, Any],
    introspection_report: dict[str, Any],
    candidate_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_entry = get_current_prompt_entry()
    candidate_prompt_text = introspection_report.get("recommended_prompt_candidate", "")
    validation = _validate_prompt_candidate(
        current_prompt_text=prompt_entry.get("prompt_text", ""),
        candidate_prompt_text=candidate_prompt_text,
        candidate_cases=candidate_cases,
    )
    queue_item = {
        "id": f"lab_queue_{run_record.get('id')}",
        "created_at": utcnow_iso(),
        "run_id": run_record.get("id"),
        "trace_id": run_record.get("trace_id"),
        "introspection_id": introspection_report.get("id"),
        "candidate_case_ids": [case.get("id") for case in candidate_cases],
        "base_prompt_version": prompt_entry.get("version_id"),
        "candidate_prompt_text": candidate_prompt_text,
        "status": "queued",
        "validation": validation,
    }

    if validation.get("validation_status") == "completed":
        candidate_entry = register_prompt_candidate(
            prompt_text=candidate_prompt_text,
            prompt_patch=introspection_report.get("prompt_patch", []) + introspection_report.get("tool_policy_patch", []),
            origin_run_id=run_record.get("id"),
            validation_metrics=validation,
        )
        queue_item["candidate_version_id"] = candidate_entry.get("version_id")

        if validation.get("passed_guardrails"):
            queue_item["status"] = "candidate_ready"
        else:
            reject_prompt_candidate(
                candidate_entry.get("version_id"),
                "Candidate prompt failed score, cost, latency, or evaluator guardrails.",
            )
            queue_item["status"] = "rejected"
    else:
        candidate_entry = register_prompt_candidate(
            prompt_text=candidate_prompt_text,
            prompt_patch=introspection_report.get("prompt_patch", []) + introspection_report.get("tool_policy_patch", []),
            origin_run_id=run_record.get("id"),
            validation_metrics=validation,
        )
        queue_item["candidate_version_id"] = candidate_entry.get("version_id")
        queue_item["status"] = "pending_validation"

    save_lab_queue_item(queue_item)
    return queue_item


def _should_trigger_rollback(introspection_report: dict[str, Any], current_prompt_version: str, run_id: str) -> dict[str, Any] | None:
    failure_summary = introspection_report.get("failure_summary", {})
    regression = (
        failure_summary.get("critical_anomaly_count", 0) > 0
        or failure_summary.get("average_data_confidence", 100) < 55
        or float(failure_summary.get("latency_ms", 0.0) or 0.0) > ROLLBACK_LATENCY_THRESHOLD_MS
        or failure_summary.get("warning_count", 0) > 2
    )
    record_prompt_observation(
        version_id=current_prompt_version,
        run_id=run_id,
        regression=regression,
        anomaly_count=int(failure_summary.get("critical_anomaly_count", 0) or 0),
        latency_ms=failure_summary.get("latency_ms"),
    )
    return None


def list_watchlists() -> list[dict[str, Any]]:
    return load_watchlists()


def create_watchlist(
    *,
    name: str,
    tickers: list[str],
    description: str | None = None,
    schedule_enabled: bool = True,
) -> dict[str, Any]:
    sanitized_tickers = sanitize_tickers(tickers)
    if not sanitized_tickers:
        raise ValueError("A watchlist needs at least one valid ticker.")

    watchlist_id = f"watchlist_{_slugify(name)}"
    existing = get_watchlist(watchlist_id)
    now = utcnow_iso()

    watchlist = {
        "id": watchlist_id,
        "name": name.strip(),
        "description": (description or "").strip(),
        "tickers": sanitized_tickers,
        "schedule_enabled": schedule_enabled,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
    }
    return upsert_watchlist(watchlist)


def _collect_explorer_tools(tickers: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    tool_outputs: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    cached_fundamentals = collect_fundamentals_snapshot(tickers)
    cached_fundamentals["timestamp_utc"] = utcnow_iso().replace("+00:00", "Z")

    tool_plan = [
        ("get_fundamentals_snapshot", {"tickers": tickers}),
        ("get_price_history", {"tickers": tickers, "period": "6mo", "interval": "1d"}),
        ("get_market_context", {"tickers": tickers, "period": "1mo", "news_count": 4}),
        ("compare_peers", {"tickers": tickers}),
    ]

    for tool_name, payload in tool_plan:
        started_at = perf_counter()
        with trace_span(
            "phoenix.market_research.tool",
            {
                "tool.name": tool_name,
                "tool.source": "yfinance",
                "tool.input": payload,
            },
        ) as tool_span:
            if tool_name == "get_fundamentals_snapshot":
                tool_output = cached_fundamentals
            elif tool_name == "compare_peers":
                tool_output = build_peer_comparison_from_fundamentals(cached_fundamentals)
                tool_output["timestamp_utc"] = utcnow_iso().replace("+00:00", "Z")
            else:
                tool_output = run_live_tool(tool_name, payload)

            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            summary = summarize_tool_output(tool_output)
            set_span_attributes(
                tool_span,
                {
                    "tool.latency_ms": latency_ms,
                    "tool.output_summary": summary,
                    "response.warning_flags": tool_output.get("warnings", []),
                },
            )

        warnings.extend(tool_output.get("warnings", []))
        tool_outputs.append(tool_output)
        tool_trace.append(
            {
                "tool_name": tool_name,
                "status": "ok",
                "latency_ms": latency_ms,
                "warnings": tool_output.get("warnings", []),
                "summary": summary,
            }
        )

    return tool_outputs, tool_trace, sorted(set(warnings))


def perform_market_research_run(
    watchlist_id: str,
    *,
    trigger: str = "manual",
) -> dict[str, Any]:
    watchlist = get_watchlist(watchlist_id)
    if watchlist is None:
        raise ValueError(f"Unknown watchlist: {watchlist_id}")

    tickers = sanitize_tickers(watchlist.get("tickers"))
    if not tickers:
        raise ValueError("The selected watchlist does not contain valid tickers.")

    started_at = perf_counter()
    run_id = f"research_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    prompt_entry = get_current_prompt_entry()
    prompt_text = prompt_entry.get("prompt_text", "")
    prompt_version = prompt_entry.get("version_id")

    with trace_span(
        "phoenix.market_research_run",
        {
            "research.run_id": run_id,
            "research.trigger": trigger,
            "research.watchlist_id": watchlist_id,
            "research.tickers": tickers,
            "prompt.template_id": EXPLORER_PROMPT_TEMPLATE_ID,
            "prompt.version": prompt_version,
        },
    ) as root_span:
        trace_id = current_trace_id() or uuid4().hex
        tool_outputs, tool_trace, warnings = _collect_explorer_tools(tickers)
        tool_lookup = {item.get("tool_name"): item for item in tool_outputs}
        fundamentals_payload = tool_lookup.get("get_fundamentals_snapshot", {})
        price_payload = tool_lookup.get("get_price_history", {})
        market_context_payload = tool_lookup.get("get_market_context", {})
        peer_payload = tool_lookup.get("compare_peers", {})
        fundamentals_map = fundamentals_payload.get("data", {})
        price_history_map = price_payload.get("data", {})
        market_context_map = market_context_payload.get("data", {})
        peer_rows = peer_payload.get("data", {}).get("comparison_rows", [])
        peer_rows_by_ticker = {
            item.get("ticker"): item
            for item in peer_rows
            if isinstance(item, dict) and item.get("ticker")
        }

        ticker_contexts: dict[str, dict[str, Any]] = {}
        for symbol in tickers:
            ticker_contexts[symbol] = {
                "fundamentals": fundamentals_map.get(symbol, {}),
                "price_history": price_history_map.get(symbol, {}),
                "market_context": market_context_map.get(symbol, {}),
                "fundamentals_payload": fundamentals_payload,
                "price_payload": price_payload,
                "market_context_payload": market_context_payload,
            }

        analysis_artifacts = build_analysis_artifacts(
            run_id=run_id,
            ticker_contexts=ticker_contexts,
            peer_rows_by_ticker=peer_rows_by_ticker,
            external_warnings=warnings,
        )
        analysis_dossiers = analysis_artifacts.get("analysis_dossiers", {})
        audit_graphs = analysis_artifacts.get("audit_graphs", {})

        save_analysis_dossiers_bulk(list(analysis_dossiers.values()))
        save_audit_graphs_bulk(list(audit_graphs.values()))

        provisional_cards: list[dict[str, Any]] = []
        for symbol in tickers:
            dossier = analysis_dossiers.get(symbol)
            audit_graph = audit_graphs.get(symbol)
            if not dossier or not audit_graph:
                continue
            card, _ = _build_idea_card(
                run_id=run_id,
                trace_id=trace_id,
                prompt_text=prompt_text,
                analysis_dossier=dossier,
                audit_graph=audit_graph,
                all_warnings=warnings,
                enable_llm_summary=False,
            )
            provisional_cards.append(card)

        top_llm_targets = {
            item.get("ticker")
            for item in sorted(provisional_cards, key=lambda card: float(card.get("score", 0.0) or 0.0), reverse=True)[:3]
        }
        top_llm_targets.update(
            item.get("ticker")
            for item in provisional_cards
            if item.get("promotion_decision") == "promote_to_lab"
        )

        idea_cards: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        model_costs: list[float] = []

        for provisional in provisional_cards:
            symbol = provisional.get("ticker")
            dossier = analysis_dossiers.get(symbol, {})
            audit_graph = audit_graphs.get(symbol, {})
            card, llm_meta = _build_idea_card(
                run_id=run_id,
                trace_id=trace_id,
                prompt_text=prompt_text,
                analysis_dossier=dossier,
                audit_graph=audit_graph,
                all_warnings=warnings,
                enable_llm_summary=symbol in top_llm_targets,
            )
            idea_cards.append(card)

            usage = llm_meta.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            token_total = usage.get("total_tokens") or 0
            total_prompt_tokens += int(prompt_tokens)
            total_completion_tokens += int(completion_tokens)
            total_tokens += int(token_total)

            cost_summary = llm_meta.get("cost_summary", {})
            if cost_summary.get("total_cost_usd") is not None:
                model_costs.append(float(cost_summary.get("total_cost_usd") or 0.0))

        save_idea_cards_bulk(idea_cards)

        ranked_cards = sorted(idea_cards, key=lambda card: float(card.get("score", 0.0) or 0.0), reverse=True)
        top_cards = ranked_cards[:5]
        total_latency_ms = round((perf_counter() - started_at) * 1000, 2)
        run_record = {
            "id": run_id,
            "trace_id": trace_id,
            "created_at": utcnow_iso(),
            "trigger": trigger,
            "status": "completed",
            "watchlist": {
                "id": watchlist.get("id"),
                "name": watchlist.get("name"),
                "tickers": tickers,
            },
            "prompt": {
                "template_id": EXPLORER_PROMPT_TEMPLATE_ID,
                "version_id": prompt_version,
                "prompt_text": prompt_text,
            },
            "tool_outputs": tool_outputs,
            "tool_trace": tool_trace,
            "warnings": warnings,
            "analysis_dossier_ids": [analysis_dossiers[symbol].get("id") for symbol in tickers if symbol in analysis_dossiers],
            "audit_graph_ids": [audit_graphs[symbol].get("id") for symbol in tickers if symbol in audit_graphs],
            "idea_card_ids": [card.get("id") for card in idea_cards],
            "top_idea_ids": [card.get("id") for card in top_cards],
            "latency_ms": total_latency_ms,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens or None,
                "completion_tokens": total_completion_tokens or None,
                "total_tokens": total_tokens or None,
            },
            "cost_summary": {
                "available": bool(model_costs),
                "currency": "USD",
                "model_cost_usd": round(sum(model_costs), 8) if model_costs else None,
                "tool_cost_usd": 0.0,
                "total_cost_usd": round(sum(model_costs), 8) if model_costs else None,
                "model_pricing_note": None if model_costs else "Cost unavailable; Phoenix can still derive cost from traced token counts.",
            },
            "summary": {
                "idea_count": len(idea_cards),
                "top_opportunity_count": len(top_cards),
                "anomaly_count": sum(len(card.get("quality_flags", [])) for card in idea_cards),
                "conflict_count": sum(int(card.get("conflict_count", 0) or 0) for card in idea_cards),
                "promotion_candidates": sum(1 for card in idea_cards if card.get("promotion_decision") == "promote_to_lab"),
            },
            "analysis_summary": {
                "average_opportunity_score": round(
                    sum(float(card.get("score", 0.0) or 0.0) for card in idea_cards) / len(idea_cards),
                    2,
                ) if idea_cards else 0.0,
                "average_reasoning_confidence": round(
                    sum(float(card.get("reasoning_confidence", 0.0) or 0.0) for card in idea_cards) / len(idea_cards),
                    2,
                ) if idea_cards else 0.0,
                "low_reliability_tickers": [
                    card.get("ticker")
                    for card in idea_cards
                    if float(card.get("data_confidence", 100.0) or 100.0) < 55.0
                ],
            },
        }

        introspection_report = _build_introspection_report(run_record, idea_cards)
        candidate_cases: list[dict[str, Any]] = []
        for card in idea_cards:
            if card.get("promotion_decision") != "promote_to_lab":
                continue
            symbol = card.get("ticker")
            candidate_case = _build_candidate_case_from_idea_card(
                run_record,
                card,
                {
                    "fundamentals": ticker_contexts[symbol].get("fundamentals", {}),
                    "price_history": ticker_contexts[symbol].get("price_history", {}),
                    "market_context": ticker_contexts[symbol].get("market_context", {}),
                },
                analysis_dossiers.get(symbol, {}),
                audit_graphs.get(symbol, {}),
            )
            candidate_cases.append(candidate_case)
            upsert_promoted_case(candidate_case)

        introspection_report["candidate_cases"] = candidate_cases
        save_introspection_report(introspection_report)
        queue_item = _process_continuous_lab(
            run_record=run_record,
            introspection_report=introspection_report,
            candidate_cases=candidate_cases,
        )
        rollback_entry = _should_trigger_rollback(
            introspection_report,
            current_prompt_version=prompt_version,
            run_id=run_id,
        )

        run_record["introspection_id"] = introspection_report.get("id")
        run_record["lab_queue_item_id"] = queue_item.get("id")
        run_record["top_opportunities"] = [
            {
                "id": card.get("id"),
                "ticker": card.get("ticker"),
                "score": card.get("score"),
                "promotion_decision": card.get("promotion_decision"),
                "reasoning_confidence": card.get("reasoning_confidence"),
                "top_claim": (analysis_dossiers.get(card.get("ticker"), {}).get("claims", [{}]) or [{}])[0].get("claim"),
                "top_risk": (analysis_dossiers.get(card.get("ticker"), {}).get("risk_register", [{}]) or [{}])[0].get("risk_name"),
            }
            for card in top_cards
        ]
        run_record["rollback"] = {
            "triggered": rollback_entry is not None,
            "new_current_version_id": rollback_entry.get("version_id") if rollback_entry else None,
        }

        save_market_research_run(run_record)
        set_span_attributes(
            root_span,
            {
                "research.trace_id": trace_id,
                "research.idea_count": len(idea_cards),
                "research.promotion_candidates": run_record["summary"]["promotion_candidates"],
                "research.warning_count": len(warnings),
                "research.conflict_count": run_record["summary"]["conflict_count"],
                "research.introspection_id": introspection_report.get("id"),
                "research.lab_queue_item_id": queue_item.get("id"),
                "research.rollback_triggered": rollback_entry is not None,
                "llm.usage.total_tokens": total_tokens or None,
                "llm.cost.usd": run_record["cost_summary"].get("total_cost_usd"),
            },
        )

        return get_market_research_run_detail(run_id) or {
            "run": run_record,
            "idea_cards": idea_cards,
            "analysis_dossiers": list(analysis_dossiers.values()),
            "analysis_dossier_summaries": [summarize_analysis_dossier(item) for item in analysis_dossiers.values()],
            "audit_graphs": list(audit_graphs.values()),
            "audit_graph_summaries": [summarize_audit_graph(item) for item in audit_graphs.values()],
            "introspection": introspection_report,
            "lab_queue_item": queue_item,
        }


def rerun_introspection(run_id: str) -> dict[str, Any]:
    detail = get_market_research_run_detail(run_id)
    if detail is None:
        raise ValueError(f"Unknown market research run: {run_id}")

    run_record = detail["run"]
    idea_cards = detail["idea_cards"]
    introspection_report = _build_introspection_report(run_record, idea_cards)
    candidate_cases: list[dict[str, Any]] = []

    for card in idea_cards:
        if card.get("promotion_decision") != "promote_to_lab":
            continue

        ticker = card.get("ticker")
        ticker_context = {}
        for tool_output in run_record.get("tool_outputs", []):
            tool_name = tool_output.get("tool_name")
            if tool_name == "get_fundamentals_snapshot":
                ticker_context["fundamentals"] = tool_output.get("data", {}).get(ticker, {})
            elif tool_name == "get_price_history":
                ticker_context["price_history"] = tool_output.get("data", {}).get(ticker, {})
            elif tool_name == "get_market_context":
                ticker_context["market_context"] = tool_output.get("data", {}).get(ticker, {})

        candidate_case = _build_candidate_case_from_idea_card(
            run_record,
            card,
            ticker_context,
            get_analysis_dossier(run_id, ticker) or {},
            get_audit_graph(run_id, ticker) or {},
        )
        candidate_cases.append(candidate_case)

    introspection_report["candidate_cases"] = candidate_cases
    save_introspection_report(introspection_report)
    queue_item = _process_continuous_lab(
        run_record=run_record,
        introspection_report=introspection_report,
        candidate_cases=candidate_cases,
    )
    run_record["introspection_id"] = introspection_report.get("id")
    run_record["lab_queue_item_id"] = queue_item.get("id")
    save_market_research_run(run_record)

    return {
        "run": run_record,
        "introspection": introspection_report,
        "lab_queue_item": queue_item,
    }


def list_market_research_runs() -> list[dict[str, Any]]:
    return load_market_research_runs()


def get_market_research_run_detail(run_id: str) -> dict[str, Any] | None:
    run_record = get_market_research_run(run_id)
    if run_record is None:
        return None

    introspection = None
    introspection_id = run_record.get("introspection_id")
    if introspection_id:
        reports = load_introspection_reports()
        for report in reports:
            if report.get("id") == introspection_id:
                introspection = report
                break

    lab_queue_item = None
    lab_queue_item_id = run_record.get("lab_queue_item_id")
    if lab_queue_item_id:
        queue_items = load_lab_queue()
        for queue_item in queue_items:
            if queue_item.get("id") == lab_queue_item_id:
                lab_queue_item = queue_item
                break

    dossiers = list_analysis_dossiers_for_run(run_id)
    audit_graphs = list_audit_graphs_for_run(run_id)

    return {
        "run": run_record,
        "idea_cards": list_idea_cards_for_run(run_id),
        "analysis_dossiers": dossiers,
        "analysis_dossier_summaries": [summarize_analysis_dossier(item) for item in dossiers],
        "audit_graphs": audit_graphs,
        "audit_graph_summaries": [summarize_audit_graph(item) for item in audit_graphs],
        "introspection": introspection,
        "lab_queue_item": lab_queue_item,
    }


def get_lab_queue_items() -> list[dict[str, Any]]:
    return load_lab_queue()


def get_analysis_dossier_detail(run_id: str, ticker: str) -> dict[str, Any] | None:
    dossier = get_analysis_dossier(run_id, ticker)
    if dossier is None:
        return None

    audit_graph = get_audit_graph(run_id, ticker)
    return {
        "analysis_dossier": dossier,
        "summary": summarize_analysis_dossier(dossier),
        "audit_graph_summary": summarize_audit_graph(audit_graph) if audit_graph else None,
    }


def get_audit_graph_detail(run_id: str, ticker: str) -> dict[str, Any] | None:
    audit_graph = get_audit_graph(run_id, ticker)
    if audit_graph is None:
        return None

    return {
        "audit_graph": audit_graph,
        "summary": summarize_audit_graph(audit_graph),
    }


def get_phoenix_prompt_state() -> dict[str, Any]:
    return get_prompt_registry_state()


def manual_prompt_rollback(reason: str | None = None) -> dict[str, Any] | None:
    return rollback_to_previous_prompt(reason=reason or "Manual Phoenix rollback requested.")


def get_phoenix_overview() -> dict[str, Any]:
    runs = load_market_research_runs()
    watchlists = load_watchlists()
    prompt_state = get_prompt_registry_state()
    queue_items = load_lab_queue()
    all_idea_cards = load_idea_cards()
    today_utc = datetime.now(timezone.utc).date().isoformat()
    runs_today = [run for run in runs if (run.get("created_at") or "").startswith(today_utc)]
    recent_run_ids = {run.get("id") for run in runs[:5]}
    latest_cards = [
        card
        for card in all_idea_cards
        if card.get("run_id") in recent_run_ids
    ]
    latest_cards = sorted(latest_cards, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[:5]

    total_cost = sum(
        float(run.get("cost_summary", {}).get("total_cost_usd") or 0.0)
        for run in runs_today
        if run.get("cost_summary", {}).get("total_cost_usd") is not None
    )
    total_tokens = sum(
        int(run.get("token_usage", {}).get("total_tokens") or 0)
        for run in runs_today
    )
    average_latency = round(
        sum(float(run.get("latency_ms", 0.0) or 0.0) for run in runs_today) / len(runs_today),
        2,
    ) if runs_today else 0.0
    anomalies_today = sum(int(run.get("summary", {}).get("anomaly_count", 0) or 0) for run in runs_today)
    introspections_today = sum(1 for run in runs_today if run.get("introspection_id"))
    promotions_today = sum(int(run.get("summary", {}).get("promotion_candidates", 0) or 0) for run in runs_today)
    conflicts_today = sum(int(run.get("summary", {}).get("conflict_count", 0) or 0) for run in runs_today)

    return {
        "generated_at": utcnow_iso(),
        "headline": {
            "runs_today": len(runs_today),
            "active_watchlists": sum(1 for item in watchlists if item.get("schedule_enabled")),
            "total_cost_usd": round(total_cost, 8) if total_cost else None,
            "total_tokens": total_tokens or None,
            "average_latency_ms": average_latency,
            "anomalies_today": anomalies_today,
            "conflicts_today": conflicts_today,
            "introspections_today": introspections_today,
            "promotion_candidates_today": promotions_today,
        },
        "prompts": {
            "current": prompt_state.get("current"),
            "candidate": prompt_state.get("candidate"),
            "previous": prompt_state.get("previous"),
        },
        "watchlists": watchlists,
        "recent_runs": [
            {
                "id": run.get("id"),
                "trace_id": run.get("trace_id"),
                "created_at": run.get("created_at"),
                "watchlist": run.get("watchlist", {}),
                "latency_ms": run.get("latency_ms"),
                "anomaly_count": run.get("summary", {}).get("anomaly_count"),
                "conflict_count": run.get("summary", {}).get("conflict_count"),
                "promotion_candidates": run.get("summary", {}).get("promotion_candidates"),
                "lab_queue_item_id": run.get("lab_queue_item_id"),
            }
            for run in runs[:6]
        ],
        "top_opportunities": [
            {
                "id": card.get("id"),
                "run_id": card.get("run_id"),
                "trace_id": card.get("trace_id"),
                "ticker": card.get("ticker"),
                "score": card.get("score"),
                "promotion_decision": card.get("promotion_decision"),
                "data_confidence": card.get("data_confidence"),
                "reasoning_confidence": card.get("reasoning_confidence"),
                "quality_flags": card.get("quality_flags", []),
                "conflict_count": card.get("conflict_count"),
                "promotion_reasons": card.get("promotion_reasons", []),
                "thesis": card.get("thesis"),
            }
            for card in latest_cards
        ],
        "lab_queue": queue_items[:6],
    }


def run_scheduled_watchlists() -> list[dict[str, Any]]:
    results = []
    for watchlist in load_watchlists():
        if not watchlist.get("schedule_enabled"):
            continue
        results.append(perform_market_research_run(watchlist.get("id"), trigger="scheduled"))
    return results
