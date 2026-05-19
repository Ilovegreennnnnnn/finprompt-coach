from __future__ import annotations

import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.dataset import PROMOTED_BENCHMARK_ID, upsert_promoted_case
from app.gemini_client import generate_gemini_response
from app.live_store import compact_trace_summary, live_trace_store
from app.live_tools import (
    build_peer_comparison_from_fundamentals,
    build_promoted_case_from_trace,
    classify_analysis_mode,
    collect_fundamentals_snapshot,
    resolve_tickers,
    run_live_tool,
    summarize_tool_output,
)
from app.research_state import get_current_prompt_entry
from app.tracing import current_trace_id, set_span_attributes, trace_span


LIVE_PROMPT_TEMPLATE_ID = "live_analyst_v1"


def build_live_tool_payload(
    tool_name: str,
    *,
    tickers: list[str],
    analysis_mode: str,
) -> dict[str, Any]:
    if tool_name == "get_price_history":
        return {
            "tickers": tickers,
            "period": "1mo" if analysis_mode == "market_context" else "6mo",
            "interval": "1d",
        }

    if tool_name == "get_market_context":
        return {
            "tickers": tickers,
            "period": "1mo",
            "news_count": 6,
        }

    return {
        "tickers": tickers,
    }


def render_live_analyst_prompt(
    *,
    prompt_text: str,
    message: str,
    tickers: list[str],
    analysis_mode: str,
    tool_trace: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
) -> str:
    compact_outputs = json.dumps(tool_outputs, ensure_ascii=True, indent=2)
    compact_trace = json.dumps(tool_trace, ensure_ascii=True, indent=2)

    return f"""
System instructions:
{prompt_text}

Detected tickers:
{tickers}

Analysis mode:
{analysis_mode}

Tool trace summary:
{compact_trace}

Tool outputs:
{compact_outputs}

User message:
{message}

Write a concise but specific answer grounded only in the provided tool outputs.
Do not show self-corrections, internal reasoning, or draft thoughts.
""".strip()


def _fallback_live_answer(
    *,
    tickers: list[str],
    warnings: list[str],
) -> str:
    if not tickers:
        return (
            "1. Key facts\n"
            "- No ticker was detected in the request.\n\n"
            "2. Positive factors\n"
            "- The request can still be analyzed once a ticker is provided.\n\n"
            "3. Risks\n"
            "- Without a ticker, any financial analysis would be speculative.\n\n"
            "4. Missing information\n"
            "- Provide at least one ticker symbol such as AAPL or MSFT.\n\n"
            "5. Educational conclusion\n"
            "- I need a ticker before I can run live equity research tools.\n\n"
            "6. Not financial advice\n"
            "This is not financial advice."
        )

    warning_lines = "\n".join(f"- {item}" for item in warnings[:4]) if warnings else "- Live tools returned incomplete data."
    return (
        "1. Key facts\n"
        f"{warning_lines}\n\n"
        "2. Positive factors\n"
        "- Some live context may still be usable once the data source is available again.\n\n"
        "3. Risks\n"
        "- The current answer is incomplete because the live data or model response was unavailable.\n\n"
        "4. Missing information\n"
        "- Retry the request or provide a clearer ticker set and question.\n\n"
        "5. Educational conclusion\n"
        "- I cannot provide a grounded live analysis until the underlying tools and model call succeed.\n\n"
        "6. Not financial advice\n"
        "This is not financial advice."
    )


def _build_cost_summary(llm_result: dict[str, Any], tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
    llm_cost = llm_result.get("cost_summary", {})
    tool_costs = [
        {
            "tool_name": item.get("tool_name"),
            "cost_usd": 0.0,
        }
        for item in tool_trace
    ]

    total_cost = llm_cost.get("total_cost_usd")
    return {
        "available": llm_cost.get("available", False),
        "currency": "USD",
        "model_cost_usd": total_cost,
        "tool_cost_usd": 0.0,
        "total_cost_usd": total_cost,
        "model_pricing_note": llm_cost.get("note"),
        "by_component": {
            "llm": llm_cost,
            "tools": tool_costs,
        },
    }


def perform_live_analysis(
    *,
    message: str,
    explicit_tickers: list[str] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    ticker_warnings: list[str] = []
    tool_outputs: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    llm_result: dict[str, Any] = {
        "text": "",
        "model": None,
        "model_version": None,
        "response_id": None,
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "cost_summary": {
            "available": False,
            "total_cost_usd": None,
            "note": "No LLM cost data was recorded.",
        },
    }

    with trace_span(
        "live.analyze",
        {
            "session.id": conversation_id,
            "live.request.message": message,
        },
    ) as root_span:
        trace_id = current_trace_id() or uuid4().hex
        prompt_entry = get_current_prompt_entry()
        prompt_text = prompt_entry.get("prompt_text", "")
        prompt_version = prompt_entry.get("version_id", "live_prompt_unknown")
        tickers, ticker_warnings = resolve_tickers(message, explicit_tickers)
        warnings = list(ticker_warnings)
        analysis_plan = classify_analysis_mode(message, tickers)

        set_span_attributes(
            root_span,
            {
                "live.request.tickers": tickers,
                "live.analysis_mode": analysis_plan["analysis_mode"],
                "live.primary_tool": analysis_plan["primary_tool"],
                "live.tool_sequence": analysis_plan["tool_sequence"],
                "prompt.template_id": LIVE_PROMPT_TEMPLATE_ID,
                "prompt.version": prompt_version,
            },
        )

        if tickers:
            cached_fundamentals: dict[str, Any] | None = None
            for tool_name in analysis_plan["tool_sequence"]:
                payload = build_live_tool_payload(tool_name, tickers=tickers, analysis_mode=analysis_plan["analysis_mode"])
                tool_started_at = perf_counter()

                try:
                    with trace_span(
                        "live.tool.call",
                        {
                            "tool.name": tool_name,
                            "tool.input": payload,
                            "tool.selection_reason": analysis_plan["reason"],
                            "tool.source": "yfinance",
                        },
                    ) as tool_span:
                        if tool_name == "compare_peers":
                            cached_fundamentals = collect_fundamentals_snapshot(tickers)
                            tool_output = build_peer_comparison_from_fundamentals(cached_fundamentals)
                            tool_output["timestamp_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                        elif tool_name == "get_fundamentals_snapshot" and cached_fundamentals is not None:
                            tool_output = cached_fundamentals
                            tool_output.setdefault(
                                "timestamp_utc",
                                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            )
                        else:
                            tool_output = run_live_tool(tool_name, payload)
                        latency_ms = round((perf_counter() - tool_started_at) * 1000, 2)
                        summary = summarize_tool_output(tool_output)
                        set_span_attributes(
                            tool_span,
                            {
                                "tool.latency_ms": latency_ms,
                                "tool.output_summary": summary,
                                "response.warning_flags": tool_output.get("warnings", []),
                            },
                        )

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
                    warnings.extend(tool_output.get("warnings", []))
                except Exception as exc:
                    latency_ms = round((perf_counter() - tool_started_at) * 1000, 2)
                    warning = f"{tool_name} failed: {exc}"
                    warnings.append(warning)
                    tool_trace.append(
                        {
                            "tool_name": tool_name,
                            "status": "error",
                            "latency_ms": latency_ms,
                            "warnings": [warning],
                            "summary": {
                                "tool_name": tool_name,
                                "source": "yfinance",
                                "warnings": [warning],
                                "ticker_count": len(tickers),
                                "preview": {},
                            },
                        }
                    )

        rendered_prompt = None
        answer = ""
        outcome = "ok"

        if tickers and tool_outputs:
            rendered_prompt = render_live_analyst_prompt(
                prompt_text=prompt_text,
                message=message,
                tickers=tickers,
                analysis_mode=analysis_plan["analysis_mode"],
                tool_trace=tool_trace,
                tool_outputs=tool_outputs,
            )
            set_span_attributes(
                root_span,
                {
                    "prompt.rendered_text": rendered_prompt,
                },
            )

            try:
                llm_started_at = perf_counter()
                llm_result = generate_gemini_response(rendered_prompt)
                answer = llm_result["text"]
                set_span_attributes(
                    root_span,
                    {
                        "live.llm.latency_ms": round((perf_counter() - llm_started_at) * 1000, 2),
                    },
                )
            except Exception as exc:
                warning = f"Gemini live analysis failed: {exc}"
                warnings.append(warning)
                answer = _fallback_live_answer(tickers=tickers, warnings=warnings)
                outcome = "llm_error"
        else:
            answer = _fallback_live_answer(tickers=tickers, warnings=warnings)
            outcome = "missing_ticker" if not tickers else "tooling_unavailable"

        total_latency_ms = round((perf_counter() - started_at) * 1000, 2)
        cost_summary = _build_cost_summary(llm_result, tool_trace)
        trace_record = {
            "trace_id": trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conversation_id": conversation_id,
            "request": {
                "message": message,
                "tickers": tickers,
                "analysis_mode": analysis_plan["analysis_mode"],
                "primary_tool": analysis_plan["primary_tool"],
                "tool_sequence": analysis_plan["tool_sequence"],
                "selection_reason": analysis_plan["reason"],
            },
            "prompt": {
                "template_id": LIVE_PROMPT_TEMPLATE_ID,
                "version": prompt_version,
                "rendered_text": rendered_prompt,
            },
            "tool_outputs": tool_outputs,
            "tool_trace": tool_trace,
            "llm": {
                "model": llm_result.get("model"),
                "model_version": llm_result.get("model_version"),
                "response_id": llm_result.get("response_id"),
                "usage": llm_result.get("usage", {}),
            },
            "cost_summary": cost_summary,
            "response": {
                "answer": answer,
                "warnings": warnings,
                "outcome": outcome,
            },
            "latency_ms": total_latency_ms,
            "promotion": {},
        }
        live_trace_store.save(trace_record)

        promoted_case = None
        if trace_id and tickers and tool_outputs:
            promoted_case = upsert_promoted_case(build_promoted_case_from_trace(trace_record))
            trace_record["promotion"] = {
                "auto_promoted": True,
                "benchmark_id": PROMOTED_BENCHMARK_ID,
                "promoted_case_id": promoted_case.get("id"),
            }
            live_trace_store.update(trace_id, {"promotion": trace_record["promotion"]})

        set_span_attributes(
            root_span,
            {
                "response.outcome": outcome,
                "response.warning_flags": warnings,
                "live.trace_id": trace_id,
                "live.total_latency_ms": total_latency_ms,
                "live.promoted_case_id": trace_record["promotion"].get("promoted_case_id"),
            },
        )

    return {
        "answer": answer,
        "analysis_mode": analysis_plan["analysis_mode"],
        "tickers": tickers,
        "warnings": warnings,
        "tool_trace": tool_trace,
        "trace_metadata": {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "prompt_template_id": LIVE_PROMPT_TEMPLATE_ID,
            "prompt_version": prompt_version,
            "llm_model": llm_result.get("model"),
            "llm_model_version": llm_result.get("model_version"),
            "latency_ms": total_latency_ms,
            "promoted_case_id": trace_record["promotion"].get("promoted_case_id"),
        },
        "cost_summary": cost_summary,
        "token_usage": llm_result.get("usage", {}),
        "promotion": trace_record["promotion"],
        "prompt_registry": {
            "current_version_id": prompt_version,
            "template_id": LIVE_PROMPT_TEMPLATE_ID,
        },
    }


def get_trace_summary(trace_id: str) -> dict[str, Any] | None:
    record = live_trace_store.get(trace_id)
    if not record:
        return None

    return compact_trace_summary(record)


def promote_trace(trace_id: str) -> dict[str, Any] | None:
    record = live_trace_store.get(trace_id)
    if not record:
        return None

    case = upsert_promoted_case(build_promoted_case_from_trace(record))
    promotion = {
        "auto_promoted": True,
        "benchmark_id": PROMOTED_BENCHMARK_ID,
        "promoted_case_id": case.get("id"),
    }
    live_trace_store.update(trace_id, {"promotion": promotion})
    return {
        "trace_id": trace_id,
        "promotion": promotion,
        "case": case,
    }
