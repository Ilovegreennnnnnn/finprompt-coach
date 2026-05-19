import unittest
from unittest.mock import patch

from app.analysis import build_analysis_artifacts
from app.dataset import PROMOTED_DATASET_PATH
from app.explorer_service import create_watchlist, get_phoenix_overview, perform_market_research_run
from app.live_service import get_trace_summary, perform_live_analysis
from app.live_store import live_trace_store
from app.prompt_coach import optimize_prompt_loop
from app.research_state import (
    ANALYSIS_DOSSIERS_PATH,
    AUDIT_GRAPHS_PATH,
    IDEA_CARDS_PATH,
    INTROSPECTIONS_PATH,
    LAB_QUEUE_PATH,
    PROMPT_REGISTRY_PATH,
    RESEARCH_RUNS_PATH,
    WATCHLISTS_PATH,
    export_universal_data_bundle,
    get_prompt_registry_state,
    list_universal_collections,
)


class LiveAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        live_trace_store._records.clear()

    def tearDown(self) -> None:
        live_trace_store._records.clear()

    @patch("app.live_service.upsert_promoted_case")
    @patch("app.live_service.generate_gemini_response")
    @patch("app.live_service.run_live_tool")
    def test_perform_live_analysis_promotes_trace_and_returns_metadata(
        self,
        mock_run_live_tool,
        mock_generate_gemini_response,
        mock_upsert_promoted_case,
    ) -> None:
        mock_run_live_tool.side_effect = [
            {
                "tool_name": "compare_peers",
                "source": "yfinance",
                "input": {"tickers": ["MSFT", "AAPL"]},
                "warnings": [],
                "data": {
                    "comparison_rows": [
                        {"ticker": "MSFT", "trailing_pe": 31.2},
                        {"ticker": "AAPL", "trailing_pe": 28.4},
                    ]
                },
                "timestamp_utc": "2026-05-15T10:00:00Z",
            },
            {
                "tool_name": "get_fundamentals_snapshot",
                "source": "yfinance",
                "input": {"tickers": ["MSFT", "AAPL"]},
                "warnings": [],
                "data": {
                    "MSFT": {"free_cash_flow": 123.0},
                    "AAPL": {"free_cash_flow": 98.0},
                },
                "timestamp_utc": "2026-05-15T10:00:01Z",
            },
        ]
        mock_generate_gemini_response.return_value = {
            "text": "1. Key facts\n- Synthetic answer.\n\n6. Not financial advice\nThis is not financial advice.",
            "model": "gemini-2.5-flash",
            "model_version": "gemini-2.5-flash-001",
            "response_id": "resp-123",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
            },
            "cost_summary": {
                "available": False,
                "total_cost_usd": None,
                "note": "Phoenix derives cost from traced token counts.",
            },
        }
        mock_upsert_promoted_case.return_value = {"id": "live_case_001"}

        result = perform_live_analysis(
            message="Compare MSFT and AAPL on valuation and cash flow quality.",
            conversation_id="conv-001",
        )

        self.assertEqual(result["analysis_mode"], "peer_comparison")
        self.assertEqual(result["tickers"], ["MSFT", "AAPL"])
        self.assertEqual(result["promotion"]["promoted_case_id"], "live_case_001")
        self.assertEqual(result["token_usage"]["total_tokens"], 200)
        self.assertEqual(len(result["tool_trace"]), 2)

        trace_id = result["trace_metadata"]["trace_id"]
        self.assertIsNotNone(trace_id)
        summary = get_trace_summary(trace_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["trace_id"], trace_id)
        self.assertEqual(summary["request"]["analysis_mode"], "peer_comparison")


class PromptLabTests(unittest.TestCase):
    @patch("app.prompt_coach.load_promoted_cases", return_value=[])
    @patch("app.runner.run_evaluation_suite")
    def test_optimize_prompt_loop_keeps_best_nonfinal_round(
        self,
        mock_run_evaluation_suite,
        _mock_load_promoted_cases,
    ) -> None:
        mock_run_evaluation_suite.side_effect = [
            {
                "agent_version": "gemini",
                "total_cases": 1,
                "passed_cases": 0,
                "failed_cases": 1,
                "overall_score": 0.42,
                "average_latency_ms": 1100.0,
                "average_cost_usd": 0.00032,
                "tool_usage_breakdown": {"get_fundamentals_snapshot": 1},
                "results": [
                    {
                        "case_id": "case_001",
                        "title": "Case 001",
                        "evaluation": {
                            "overall_score": 0.42,
                            "passed": False,
                            "evaluations": [
                                {"name": "financial_safety", "passed": False},
                            ],
                        },
                    }
                ],
            },
            {
                "agent_version": "gemini",
                "total_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "overall_score": 0.67,
                "average_latency_ms": 950.0,
                "average_cost_usd": 0.00029,
                "tool_usage_breakdown": {"get_fundamentals_snapshot": 1},
                "results": [],
            },
            {
                "agent_version": "gemini",
                "total_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "overall_score": 0.61,
                "average_latency_ms": 980.0,
                "average_cost_usd": 0.00028,
                "tool_usage_breakdown": {"get_fundamentals_snapshot": 1},
                "results": [],
            },
        ]

        result = optimize_prompt_loop(
            original_prompt="You are a helpful finance assistant. Answer clearly.",
            cases=[{"id": "case_001", "title": "Case 001"}],
            max_rounds=3,
            agent_version="gemini",
        )

        self.assertEqual(result["best_round"], 2)
        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual(result["improvement_summary"]["score_delta"], 0.25)
        self.assertEqual(
            result["best_prompt"],
            result["rounds"][1]["prompt"],
        )
        self.assertEqual(result["improvement_summary"]["stop_reason"], "score_plateau")


class PhoenixLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._state_paths = [
            WATCHLISTS_PATH,
            RESEARCH_RUNS_PATH,
            IDEA_CARDS_PATH,
            INTROSPECTIONS_PATH,
            PROMPT_REGISTRY_PATH,
            LAB_QUEUE_PATH,
            ANALYSIS_DOSSIERS_PATH,
            AUDIT_GRAPHS_PATH,
            PROMOTED_DATASET_PATH,
        ]
        self._backups: dict[str, str | None] = {}

        for path in self._state_paths:
            self._backups[str(path)] = path.read_text(encoding="utf-8") if path.exists() else None

    def tearDown(self) -> None:
        for path in self._state_paths:
            original = self._backups.get(str(path))
            if original is None:
                if path.exists():
                    try:
                        path.unlink()
                    except PermissionError:
                        if path == PROMPT_REGISTRY_PATH:
                            path.write_text(
                                (
                                    '{\n'
                                    '  "current_version_id": "live_prompt_2026_05_15",\n'
                                    '  "candidate_version_id": null,\n'
                                    '  "previous_version_id": null,\n'
                                    '  "prompts": [\n'
                                    '    {\n'
                                    '      "version_id": "live_prompt_2026_05_15",\n'
                                    '      "status": "current",\n'
                                    '      "prompt_text": "You are FinPrompt Coach operating inside a Phoenix-observed equity research loop.",\n'
                                    '      "patch_source": ["Seed Phoenix-first live analyst prompt."],\n'
                                    '      "origin_run_id": null,\n'
                                    '      "activation_timestamp": null,\n'
                                    '      "validation_metrics": {},\n'
                                    '      "observation_history": [],\n'
                                    '      "rollback_reason": null\n'
                                    '    }\n'
                                    '  ]\n'
                                    '}\n'
                                ),
                                encoding="utf-8",
                            )
                        else:
                            path.write_text("[]\n", encoding="utf-8")
            else:
                path.write_text(original, encoding="utf-8")

    @patch("app.explorer_service.run_evaluation_suite")
    @patch("app.explorer_service.generate_gemini_response")
    @patch("app.explorer_service.run_live_tool")
    @patch("app.explorer_service.collect_fundamentals_snapshot")
    def test_market_research_run_creates_introspection_and_registers_candidate_prompt(
        self,
        mock_collect_fundamentals_snapshot,
        mock_run_live_tool,
        mock_generate_gemini_response,
        mock_run_evaluation_suite,
    ) -> None:
        current_before = get_prompt_registry_state().get("current", {}).get("version_id")
        watchlist = create_watchlist(
            name="Phoenix Test Watchlist",
            tickers=["MSFT", "AAPL"],
            description="Synthetic watchlist for continuous lab coverage.",
            schedule_enabled=True,
        )

        mock_collect_fundamentals_snapshot.return_value = {
            "tool_name": "get_fundamentals_snapshot",
            "source": "yfinance",
            "input": {"tickers": ["MSFT", "AAPL"]},
            "warnings": ["AAPL included anomalous financial metrics that were flagged before analysis."],
            "data": {
                "MSFT": {
                    "ticker": "MSFT",
                    "market_cap": 3000000000000,
                    "trailing_pe": 25.0,
                    "price_to_book": 7.0,
                    "operating_margin": 45.0,
                    "profit_margin": 38.0,
                    "return_on_equity": 34.0,
                    "revenue_growth": 17.0,
                    "free_cash_flow": 47000000000,
                    "operating_cash_flow": 122000000000,
                    "debt_to_equity": 28.0,
                    "current_ratio": 1.2,
                    "data_quality": {
                        "quality_flags": [],
                        "missing_fields": [],
                        "confidence_score": 94,
                    },
                },
                "AAPL": {
                    "ticker": "AAPL",
                    "market_cap": 4200000000000,
                    "trailing_pe": 34.0,
                    "price_to_book": 41.0,
                    "operating_margin": 32.0,
                    "profit_margin": 26.0,
                    "return_on_equity": 142.0,
                    "revenue_growth": 15.0,
                    "free_cash_flow": 99000000000,
                    "operating_cash_flow": 145000000000,
                    "debt_to_equity": 82.0,
                    "current_ratio": 1.0,
                    "dividend_yield": 36.0,
                    "data_quality": {
                        "quality_flags": ["metric_anomaly", "unit_conflict"],
                        "missing_fields": [],
                        "confidence_score": 52,
                    },
                },
            },
        }
        mock_run_live_tool.side_effect = [
            {
                "tool_name": "get_price_history",
                "source": "yfinance",
                "warnings": [],
                "timestamp_utc": "2026-05-15T10:00:00Z",
                "data": {
                    "MSFT": {
                        "summary": {
                            "period_return_pct": 12.0,
                            "annualized_volatility_pct": 22.0,
                        }
                    },
                    "AAPL": {
                        "summary": {
                            "period_return_pct": 8.0,
                            "annualized_volatility_pct": 30.0,
                        }
                    },
                },
            },
            {
                "tool_name": "get_market_context",
                "source": "yfinance",
                "warnings": [],
                "timestamp_utc": "2026-05-15T10:00:05Z",
                "data": {
                    "MSFT": {"recent_news": [{"title": "Cloud demand remains healthy"}]},
                    "AAPL": {"recent_news": [{"title": "Device cycle concerns remain mixed"}]},
                },
            },
        ]
        mock_generate_gemini_response.side_effect = [
            {
                "text": "1. Why it is interesting now\nMSFT still screens well.\n2. What could go wrong\nExecution or valuation could soften.\n3. Why the confidence is limited\nFuture demand still matters.",
                "model": "gemini-2.5-flash",
                "model_version": "gemini-2.5-flash-001",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 60,
                    "total_tokens": 180,
                },
                "cost_summary": {
                    "available": True,
                    "total_cost_usd": 0.00012,
                },
            },
            {
                "text": "1. Why it is interesting now\nAAPL remains important but noisier.\n2. What could go wrong\nData anomalies reduce trust.\n3. Why the confidence is limited\nSome metrics need verification.",
                "model": "gemini-2.5-flash",
                "model_version": "gemini-2.5-flash-001",
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 55,
                    "total_tokens": 165,
                },
                "cost_summary": {
                    "available": True,
                    "total_cost_usd": 0.00011,
                },
            },
        ]
        mock_run_evaluation_suite.side_effect = [
            {
                "agent_version": "gemini",
                "total_cases": 3,
                "passed_cases": 2,
                "failed_cases": 1,
                "overall_score": 0.64,
                "average_latency_ms": 1200.0,
                "average_cost_usd": 0.00034,
                "results": [
                    {
                        "case_id": "case_001",
                        "title": "Case 001",
                        "evaluation": {
                            "passed": False,
                            "overall_score": 0.42,
                            "evaluations": [
                                {"name": "groundedness", "passed": False},
                            ],
                        },
                    }
                ],
            },
            {
                "agent_version": "gemini",
                "total_cases": 3,
                "passed_cases": 3,
                "failed_cases": 0,
                "overall_score": 0.71,
                "average_latency_ms": 1260.0,
                "average_cost_usd": 0.00035,
                "results": [],
            },
        ]

        result = perform_market_research_run(watchlist["id"])

        self.assertEqual(result["run"]["watchlist"]["id"], watchlist["id"])
        self.assertTrue(result["idea_cards"])
        self.assertTrue(result["analysis_dossiers"])
        self.assertTrue(result["audit_graphs"])
        self.assertIsNotNone(result["introspection"])
        self.assertEqual(result["lab_queue_item"]["status"], "candidate_ready")
        self.assertGreaterEqual(len(result["introspection"]["candidate_cases"]), 1)

        overview = get_phoenix_overview()
        self.assertGreaterEqual(overview["headline"]["runs_today"], 1)
        self.assertTrue(overview["top_opportunities"])
        self.assertIsNotNone(overview["prompts"]["current"])
        self.assertEqual(
            overview["prompts"]["current"]["version_id"],
            current_before,
        )
        self.assertIsNotNone(overview["prompts"]["candidate"])


class PythonAnalysisEngineTests(unittest.TestCase):
    def test_build_analysis_artifacts_detects_conflicts_and_peer_relative_scores(self) -> None:
        ticker_contexts = {
            "MSFT": {
                "fundamentals_payload": {
                    "timestamp_utc": "2026-05-19T10:00:00Z",
                    "warnings": [],
                    "data": {
                        "MSFT": {
                            "ticker": "MSFT",
                            "company_name": "Microsoft Corporation",
                            "sector": "Technology",
                            "industry": "Software - Infrastructure",
                            "market_cap": 3000000000000,
                            "trailing_pe": 25.0,
                            "price_to_book": 7.0,
                            "operating_margin": 45.0,
                            "profit_margin": 38.0,
                            "return_on_equity": 34.0,
                            "revenue_growth": 17.0,
                            "free_cash_flow": 47000000000,
                            "operating_cash_flow": 122000000000,
                            "total_cash": 78000000000,
                            "total_debt": 125000000000,
                            "debt_to_equity": 28.0,
                            "current_ratio": 1.2,
                            "data_quality": {
                                "quality_flags": [],
                                "missing_fields": [],
                                "confidence_score": 94,
                            },
                        }
                    },
                },
                "price_payload": {
                    "timestamp_utc": "2026-05-19T10:00:10Z",
                    "warnings": [],
                    "data": {
                        "MSFT": {
                            "summary": {
                                "period_return_pct": 12.0,
                                "annualized_volatility_pct": 22.0,
                            }
                        }
                    },
                },
                "market_context_payload": {
                    "timestamp_utc": "2026-05-19T10:00:20Z",
                    "warnings": [],
                    "data": {
                        "MSFT": {
                            "recent_news": [{"title": "Cloud demand remains healthy"}]
                        }
                    },
                },
            },
            "AAPL": {
                "fundamentals_payload": {
                    "timestamp_utc": "2026-05-19T10:00:00Z",
                    "warnings": [],
                    "data": {
                        "AAPL": {
                            "ticker": "AAPL",
                            "company_name": "Apple Inc.",
                            "sector": "Technology",
                            "industry": "Consumer Electronics",
                            "market_cap": 4200000000000,
                            "trailing_pe": 34.0,
                            "price_to_book": 41.0,
                            "operating_margin": 32.0,
                            "profit_margin": 26.0,
                            "return_on_equity": 142.0,
                            "revenue_growth": 15.0,
                            "free_cash_flow": 99000000000,
                            "operating_cash_flow": 145000000000,
                            "total_cash": 68000000000,
                            "total_debt": 85000000000,
                            "debt_to_equity": 82.0,
                            "current_ratio": 1.0,
                            "dividend_yield": 36.0,
                            "payout_ratio": 12.0,
                            "data_quality": {
                                "quality_flags": ["metric_anomaly", "unit_conflict"],
                                "missing_fields": [],
                                "confidence_score": 52,
                            },
                        }
                    },
                },
                "price_payload": {
                    "timestamp_utc": "2026-05-19T10:00:10Z",
                    "warnings": [],
                    "data": {
                        "AAPL": {
                            "summary": {
                                "period_return_pct": 8.0,
                                "annualized_volatility_pct": 30.0,
                            }
                        }
                    },
                },
                "market_context_payload": {
                    "timestamp_utc": "2026-05-19T10:00:20Z",
                    "warnings": [],
                    "data": {
                        "AAPL": {
                            "recent_news": [{"title": "Device cycle concerns remain mixed"}]
                        }
                    },
                },
            },
        }
        peer_rows_by_ticker = {
            "MSFT": {"ticker": "MSFT", "trailing_pe": 25.0, "price_to_book": 7.0},
            "AAPL": {"ticker": "AAPL", "trailing_pe": 34.0, "price_to_book": 41.0},
        }

        result = build_analysis_artifacts(
            run_id="research_test",
            ticker_contexts=ticker_contexts,
            peer_rows_by_ticker=peer_rows_by_ticker,
            external_warnings=[],
        )

        msft_dossier = result["analysis_dossiers"]["MSFT"]
        aapl_dossier = result["analysis_dossiers"]["AAPL"]
        aapl_graph = result["audit_graphs"]["AAPL"]

        self.assertGreater(msft_dossier["opportunity_score"], aapl_dossier["opportunity_score"])
        self.assertIn("unit_conflict", aapl_dossier["quality_flags"])
        self.assertGreaterEqual(len(aapl_graph["conflicts"]), 1)
        self.assertIn(
            aapl_dossier["promotion_decision"],
            {"promote_to_lab", "discard_noisy_case"},
        )
        self.assertTrue(aapl_dossier["claims"])
        self.assertTrue(aapl_dossier["risk_register"])

    def test_universal_data_bundle_exposes_portable_jsonl_exports(self) -> None:
        collections = list_universal_collections()
        self.assertTrue(any(item["name"] == "watchlists" for item in collections))

        export = export_universal_data_bundle(
            collections=["watchlists", "prompt_registry"],
            export_format="jsonl",
        )

        self.assertEqual(export["manifest"]["format"], "jsonl")
        self.assertIn("watchlists", export["payloads"])
        self.assertIn("prompt_registry", export["payloads"])
        self.assertIsInstance(export["payloads"]["watchlists"], str)


if __name__ == "__main__":
    unittest.main()
