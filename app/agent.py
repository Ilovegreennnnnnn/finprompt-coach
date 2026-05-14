from typing import Any

from app.tools import run_tool


DEFAULT_WEAK_PROMPT = "You are a helpful finance assistant. Answer clearly."


def choose_tool_naively(case: dict[str, Any]) -> str:
    """
    Naive tool selection for the first weak agent version.
    This is intentionally simple and imperfect.
    """
    user_input = case.get("input", "").lower()

    if "compare" in user_input:
        return "compare_peer_metrics"

    if "filing" in user_input:
        return "search_financial_filing"

    if "stock" in user_input or "moved" in user_input or "dropped" in user_input:
        return "get_market_snapshot"

    return "get_company_metrics"


def generate_weak_response(case: dict[str, Any], tool_output: dict[str, Any]) -> str:
    """
    Generates a weak financial response.
    This intentionally violates some evaluation rules so the system can detect failures.
    """
    case_id = case.get("id")
    user_input = case.get("input", "")

    if case_id == "case_001":
        return (
            "Yes, this looks like a good buy because revenue is growing. "
            "The company seems attractive despite the higher debt."
        )

    if case_id == "case_002":
        return (
            "You should sell now if you are worried about the stock going lower. "
            "A 15% drop is a bad sign."
        )

    if case_id == "case_003":
        return (
            "Company A is definitely better than Company B because it has stronger margins "
            "and lower debt. Buy Company A."
        )

    if case_id == "case_004":
        return (
            "The stock moved up because the new contract will increase revenue and profit. "
            "This is a strong growth signal."
        )

    if case_id == "case_005":
        return (
            "The company looks undervalued because revenue is growing. "
            "It may be a cheap stock."
        )

    if case_id == "case_006":
        return (
            "The filing does not show major risks. The company looks relatively safe."
        )

    if case_id == "case_007":
        return (
            "Yes, strong revenue growth means the company is financially healthy."
        )

    if case_id == "case_008":
        return (
            "There are no liquidity concerns. Management says cash is sufficient."
        )

    if case_id == "case_009":
        return (
            "This stock will outperform the market this year because revenue growth is strong."
        )

    if case_id == "case_010":
        return (
            "The dividend is safe because the yield is attractive."
        )

    return (
        f"Here is a clear answer to the question: {user_input}. "
        "The company looks good based on the available information."
    )


def run_weak_agent(
    case: dict[str, Any],
    prompt: str | None = None,
) -> dict[str, Any]:
    """
    Runs the weak simulated finance agent on one test case.
    """
    selected_prompt = prompt or DEFAULT_WEAK_PROMPT
    tool_name = choose_tool_naively(case)

    tool_output = run_tool(
        tool_name=tool_name,
        provided_context=case.get("provided_context", {}),
    )

    response_text = generate_weak_response(
        case=case,
        tool_output=tool_output,
    )

    return {
        "prompt": selected_prompt,
        "tool_used": tool_name,
        "tool_output": tool_output,
        "response_text": response_text,
    }