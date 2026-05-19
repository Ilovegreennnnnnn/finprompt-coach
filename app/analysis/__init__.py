from __future__ import annotations

from typing import Any

from app.live_tools import normalize_for_json
from app.tracing import set_span_attributes, trace_span

from .claim_engine import build_claims
from .derived_metrics import compute_derived_metrics
from .normalization import normalize_source_snapshots
from .peer_baselines import build_peer_baselines
from .promotion_engine import determine_promotion_decision
from .risk_engine import build_risk_register
from .score_engine import score_dossier
from .source_adapters import build_source_snapshots


def summarize_audit_graph(audit_graph: dict[str, Any]) -> dict[str, Any]:
    score_breakdown = audit_graph.get("score_breakdown", {})
    promotion = audit_graph.get("promotion_decision", {})
    return {
        "ticker": audit_graph.get("entity", {}).get("ticker"),
        "opportunity_score": score_breakdown.get("opportunity_score"),
        "reasoning_confidence": score_breakdown.get("reasoning_confidence"),
        "conflict_count": len(audit_graph.get("conflicts", [])),
        "risk_count": len(audit_graph.get("risk_register", [])),
        "quality_flags": audit_graph.get("uncertainty_flags", []),
        "promotion_decision": promotion.get("decision"),
        "promotion_reasons": promotion.get("reasons", []),
    }


def summarize_analysis_dossier(dossier: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": dossier.get("ticker"),
        "company_name": dossier.get("company_name"),
        "sector": dossier.get("sector"),
        "industry": dossier.get("industry"),
        "opportunity_score": dossier.get("opportunity_score"),
        "reasoning_confidence": dossier.get("reasoning_confidence"),
        "data_reliability_score": dossier.get("score_breakdown", {}).get("data_reliability_score", {}).get("value"),
        "promotion_decision": dossier.get("promotion_decision"),
        "top_claims": dossier.get("claims", [])[:3],
        "top_risks": dossier.get("risk_register", [])[:3],
        "quality_flags": dossier.get("quality_flags", []),
        "conflict_count": len(dossier.get("conflict_summary", [])),
    }


def build_analysis_artifacts(
    *,
    run_id: str,
    ticker_contexts: dict[str, dict[str, Any]],
    peer_rows_by_ticker: dict[str, dict[str, Any]],
    external_warnings: list[str],
) -> dict[str, Any]:
    source_snapshots_by_ticker: dict[str, list[dict[str, Any]]] = {}
    normalized_bundles: dict[str, dict[str, Any]] = {}
    normalized_by_ticker: dict[str, dict[str, Any]] = {}
    derived_by_ticker: dict[str, dict[str, dict[str, Any]]] = {}
    audit_graphs: dict[str, dict[str, Any]] = {}
    analysis_dossiers: dict[str, dict[str, Any]] = {}

    with trace_span(
        "phoenix.analysis.collect_sources",
        {
            "research.run_id": run_id,
            "pipeline.stage": "collect_sources",
            "pipeline.input_size": len(ticker_contexts),
        },
    ) as span:
        for ticker, context in ticker_contexts.items():
            source_snapshots_by_ticker[ticker] = build_source_snapshots(
                ticker=ticker,
                fundamentals_payload=context.get("fundamentals_payload", {}),
                price_payload=context.get("price_payload", {}),
                market_context_payload=context.get("market_context_payload", {}),
                peer_row=peer_rows_by_ticker.get(ticker),
            )
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(source_snapshots_by_ticker),
                "pipeline.warning_count": sum(
                    len(snapshot.get("warnings", []))
                    for snapshots in source_snapshots_by_ticker.values()
                    for snapshot in snapshots
                ),
                "pipeline.artifact_type": "source_snapshots",
            },
        )

    with trace_span(
        "phoenix.analysis.normalize_sources",
        {
            "research.run_id": run_id,
            "pipeline.stage": "normalize_sources",
            "pipeline.input_size": len(source_snapshots_by_ticker),
        },
    ) as span:
        for ticker, snapshots in source_snapshots_by_ticker.items():
            bundle = normalize_source_snapshots(ticker=ticker, snapshots=snapshots)
            normalized_bundles[ticker] = bundle
            normalized_by_ticker[ticker] = bundle.get("normalized_metrics", {})
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(normalized_bundles),
                "pipeline.warning_count": sum(len(item.get("quality_flags", [])) for item in normalized_bundles.values()),
                "pipeline.artifact_type": "normalized_metrics",
            },
        )

    with trace_span(
        "phoenix.analysis.derive_metrics",
        {
            "research.run_id": run_id,
            "pipeline.stage": "derive_metrics",
            "pipeline.input_size": len(normalized_bundles),
        },
    ) as span:
        for ticker, bundle in normalized_bundles.items():
            derived_by_ticker[ticker] = compute_derived_metrics(bundle)
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(derived_by_ticker),
                "pipeline.artifact_type": "derived_metrics",
            },
        )

    with trace_span(
        "phoenix.analysis.build_peer_baselines",
        {
            "research.run_id": run_id,
            "pipeline.stage": "build_peer_baselines",
            "pipeline.input_size": len(normalized_by_ticker),
        },
    ) as span:
        peer_baselines = build_peer_baselines(
            normalized_by_ticker=normalized_by_ticker,
            derived_by_ticker=derived_by_ticker,
        )
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(peer_baselines),
                "pipeline.artifact_type": "peer_baselines",
            },
        )

    with trace_span(
        "phoenix.analysis.build_audit_graph",
        {
            "research.run_id": run_id,
            "pipeline.stage": "build_audit_graph",
            "pipeline.input_size": len(peer_baselines),
        },
    ) as span:
        for ticker, normalized_bundle in normalized_bundles.items():
            risk_register = build_risk_register(
                normalized_bundle=normalized_bundle,
                derived_metrics=derived_by_ticker[ticker],
                peer_baseline=peer_baselines[ticker],
            )
            claims, evidence = build_claims(
                ticker=ticker,
                normalized_bundle=normalized_bundle,
                derived_metrics=derived_by_ticker[ticker],
                peer_baseline=peer_baselines[ticker],
                risk_register=risk_register,
            )
            score_breakdown = score_dossier(
                normalized_bundle=normalized_bundle,
                derived_metrics=derived_by_ticker[ticker],
                peer_baseline=peer_baselines[ticker],
                risk_register=risk_register,
                claims=claims,
            )
            promotion = determine_promotion_decision(
                score_breakdown=score_breakdown,
                normalized_bundle=normalized_bundle,
                risk_register=risk_register,
                external_warnings=external_warnings,
            )
            audit_graphs[ticker] = {
                "id": f"audit_{run_id}_{ticker.lower()}",
                "run_id": run_id,
                "entity": {
                    "ticker": ticker,
                    "company_name": normalized_bundle.get("normalized_metrics", {}).get("company_name"),
                    "sector": normalized_bundle.get("normalized_metrics", {}).get("sector"),
                    "industry": normalized_bundle.get("normalized_metrics", {}).get("industry"),
                },
                "source_snapshots": source_snapshots_by_ticker[ticker],
                "normalized_metrics": normalized_bundle.get("normalized_metrics", {}),
                "derived_metrics": derived_by_ticker[ticker],
                "peer_comparison": peer_baselines[ticker],
                "claims": claims,
                "evidence": evidence,
                "conflicts": normalized_bundle.get("conflicts", []),
                "uncertainty_flags": sorted(
                    set(normalized_bundle.get("quality_flags", []))
                    | set(normalized_bundle.get("uncertainty_flags", []))
                ),
                "risk_register": risk_register,
                "score_breakdown": score_breakdown,
                "promotion_decision": promotion,
            }
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(audit_graphs),
                "pipeline.warning_count": sum(len(graph.get("conflicts", [])) for graph in audit_graphs.values()),
                "pipeline.artifact_type": "audit_graph",
            },
        )

    with trace_span(
        "phoenix.analysis.score_dossier",
        {
            "research.run_id": run_id,
            "pipeline.stage": "score_dossier",
            "pipeline.input_size": len(audit_graphs),
        },
    ) as span:
        for ticker, audit_graph in audit_graphs.items():
            analysis_dossiers[ticker] = {
                "id": f"dossier_{run_id}_{ticker.lower()}",
                "run_id": run_id,
                "ticker": ticker,
                "company_name": audit_graph.get("entity", {}).get("company_name"),
                "sector": audit_graph.get("entity", {}).get("sector"),
                "industry": audit_graph.get("entity", {}).get("industry"),
                "facts": {
                    metric_name: audit_graph.get("normalized_metrics", {}).get(metric_name)
                    for metric_name in [
                        "market_cap",
                        "trailing_pe",
                        "price_to_book",
                        "operating_margin",
                        "profit_margin",
                        "revenue_growth",
                        "free_cash_flow",
                        "debt_to_equity",
                        "period_return_pct",
                        "annualized_volatility_pct",
                    ]
                },
                "normalized_metrics": audit_graph.get("normalized_metrics", {}),
                "derived_metrics": audit_graph.get("derived_metrics", {}),
                "score_breakdown": audit_graph.get("score_breakdown", {}),
                "opportunity_score": audit_graph.get("score_breakdown", {}).get("opportunity_score"),
                "reasoning_confidence": audit_graph.get("score_breakdown", {}).get("reasoning_confidence"),
                "quality_flags": normalized_bundles[ticker].get("quality_flags", []),
                "conflict_summary": audit_graph.get("conflicts", []),
                "claims": audit_graph.get("claims", []),
                "risk_register": audit_graph.get("risk_register", []),
                "missing_information": normalized_bundles[ticker].get("missing_fields", []),
                "promotion_decision": audit_graph.get("promotion_decision", {}).get("decision"),
                "promotion_reasons": audit_graph.get("promotion_decision", {}).get("reasons", []),
                "audit_graph_summary": summarize_audit_graph(audit_graph),
            }
        set_span_attributes(
            span,
            {
                "pipeline.output_size": len(analysis_dossiers),
                "pipeline.artifact_type": "analysis_dossier",
                "research.score_distribution": normalize_for_json(
                    [item.get("opportunity_score") for item in analysis_dossiers.values()],
                    max_records=20,
                ),
            },
        )

    return {
        "source_snapshots": source_snapshots_by_ticker,
        "normalized_bundles": normalized_bundles,
        "derived_metrics": derived_by_ticker,
        "peer_baselines": peer_baselines,
        "audit_graphs": audit_graphs,
        "analysis_dossiers": analysis_dossiers,
    }
