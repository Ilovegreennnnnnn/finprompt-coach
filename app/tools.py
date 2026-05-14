from typing import Any


def get_company_metrics(provided_context: dict[str, Any]) -> dict[str, Any]:
    """
    Simulated tool for retrieving company-level financial metrics.
    """
    return {
        "tool_name": "get_company_metrics",
        "metrics": provided_context,
        "note": "Synthetic company metrics provided by the evaluation dataset.",
    }


def compare_peer_metrics(provided_context: dict[str, Any]) -> dict[str, Any]:
    """
    Simulated tool for comparing two companies using provided peer metrics.
    """
    return {
        "tool_name": "compare_peer_metrics",
        "comparison_data": provided_context,
        "note": "Synthetic peer comparison data provided by the evaluation dataset.",
    }


def search_financial_filing(provided_context: dict[str, Any]) -> dict[str, Any]:
    """
    Simulated tool for searching financial filings or filing excerpts.
    """
    return {
        "tool_name": "search_financial_filing",
        "filing_data": provided_context,
        "note": "Synthetic filing excerpt provided by the evaluation dataset.",
    }


def get_market_snapshot(provided_context: dict[str, Any]) -> dict[str, Any]:
    """
    Simulated tool for retrieving market movement or news context.
    """
    return {
        "tool_name": "get_market_snapshot",
        "market_data": provided_context,
        "note": "Synthetic market snapshot provided by the evaluation dataset.",
    }


AVAILABLE_TOOLS = {
    "get_company_metrics": get_company_metrics,
    "compare_peer_metrics": compare_peer_metrics,
    "search_financial_filing": search_financial_filing,
    "get_market_snapshot": get_market_snapshot,
}


def run_tool(tool_name: str, provided_context: dict[str, Any]) -> dict[str, Any]:
    """
    Run a simulated financial tool by name.
    """
    tool = AVAILABLE_TOOLS.get(tool_name)

    if tool is None:
        return {
            "tool_name": tool_name,
            "error": f"Unknown tool: {tool_name}",
        }

    return tool(provided_context)