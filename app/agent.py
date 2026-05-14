from typing import Any
from app.prompt_coach import BASE_IMPROVED_PROMPT
from app.tools import run_tool
from app.tracing import trace_span


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

    with trace_span(
        "tool.call",
        {
            "tool.name": tool_name,
            "agent.version": "weak",
            "case.id": case.get("id"),
        },
    ):
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


def choose_tool_with_policy(case: dict[str, Any]) -> str:
    """
    Better tool selection policy for the improved simulated agent.
    This approximates what the stronger Gemini prompt should encourage.
    """
    user_input = case.get("input", "").lower()

    if "compare" in user_input or "company a" in user_input or "company b" in user_input:
        return "compare_peer_metrics"

    if "filing" in user_input or "liquidity" in user_input or "risk" in user_input:
        return "search_financial_filing"

    if "stock moved" in user_input or "moved today" in user_input or "dropped" in user_input:
        return "get_market_snapshot"

    return "get_company_metrics"


def format_context_facts(provided_context: dict[str, Any]) -> str:
    """
    Converts the provided context into readable facts without inventing data.
    """
    facts = []

    for key, value in provided_context.items():
        if value is None:
            continue

        if isinstance(value, dict):
            nested_facts = []
            for nested_key, nested_value in value.items():
                nested_facts.append(f"{nested_key}: {nested_value}")
            facts.append(f"{key}: " + ", ".join(nested_facts))
        else:
            facts.append(f"{key}: {value}")

    if not facts:
        return "No concrete financial facts were provided."

    return "\n".join(f"- {fact}" for fact in facts)


def infer_positive_factors(case: dict[str, Any]) -> list[str]:
    context = case.get("provided_context", {})
    positives = []

    context_text = str(context).lower()

    if "revenue_growth" in context and context.get("revenue_growth") is not None:
        positives.append("Revenue growth may be a positive factor.")

    if "operating_margin" in context_text:
        positives.append("Operating margin data can help assess profitability.")

    if "contract" in context_text:
        positives.append("A new contract may be a positive business signal, but its impact is unclear without financial terms.")

    if "current cash is sufficient" in context_text:
        positives.append("Management indicates current cash may be sufficient for the next twelve months.")

    if not positives:
        positives.append("The provided information may contain useful context, but it is not enough for a full investment view.")

    return positives


def infer_risks(case: dict[str, Any]) -> list[str]:
    """
    Simple rule-based risk extraction for the improved simulated agent.
    """
    text = f"{case.get('input', '')} {case.get('provided_context', {})}".lower()
    risks = []

    if "debt" in text:
        risks.append("debt risk")

    if "valuation" in text or "undervalued" in text:
        risks.append("valuation risk")

    if "cash flow" in text or "free_cash_flow" in text:
        risks.append("cash flow risk")

    if "margin" in text or "earnings" in text:
        risks.append("earnings risk")

    if "liquidity" in text or "additional financing" in text:
        risks.append("liquidity risk")

    if "financing" in text:
        risks.append("financing risk")

    if "interest rates" in text or "borrowing costs" in text:
        risks.append("interest rate risk")

    if "large customers" in text:
        risks.append("customer concentration risk")

    if "stock" in text or "market" in text or "outperform" in text:
        risks.append("market risk")

    if "contract" in text:
        risks.append("news uncertainty")
        risks.append("missing contract value")

    if "dividend" in text or "payout_ratio" in text:
        risks.append("dividend sustainability risk")

    if "negative" in text or "operating losses" in text:
        risks.append("profitability risk")

    if "forecast" in text or "this year" in text:
        risks.append("forecasting uncertainty")

    if not risks:
        risks.append("incomplete information")

    return sorted(set(risks))


def infer_missing_information(case: dict[str, Any]) -> list[str]:
    context = case.get("provided_context", {})
    missing = []

    for key, value in context.items():
        if value is None:
            missing.append(key.replace("_", " "))

    text = f"{case.get('input', '')} {context}".lower()

    if "undervalued" in text and "valuation data" not in missing:
        missing.append("valuation data")

    if "outperform" in text:
        missing.append("market forecast")
        missing.append("valuation data")

    if "contract" in text:
        missing.append("contract value")
        missing.append("financial terms")

    if "compare" in text:
        missing.append("growth data")
        missing.append("cash flow data")

    if not missing:
        missing.append("More financial context may be needed for a complete analysis.")

    return sorted(set(missing))


def generate_improved_response(case: dict[str, Any], tool_output: dict[str, Any]) -> str:
    """
    Generates a safer, better-structured response for the improved simulated agent.
    """
    context = case.get("provided_context", {})

    facts = format_context_facts(context)
    positives = infer_positive_factors(case)
    risks = infer_risks(case)
    missing = infer_missing_information(case)

    positive_text = "\n".join(f"- {item}" for item in positives)
    risk_text = "\n".join(f"- {item}" for item in risks)
    missing_text = "\n".join(f"- {item}" for item in missing)

    return f"""
1. Key facts
{facts}

2. Positive factors
{positive_text}

3. Risks
{risk_text}

4. Missing information
{missing_text}

5. Educational conclusion
Based only on the provided information, the situation should be interpreted cautiously. The data may show some positive signals, but the risks and missing information prevent a confident investment conclusion. I cannot say whether the user should buy, sell, hold, or trade the asset.

6. Not financial advice
This is not financial advice.
""".strip()


def run_improved_agent(
    case: dict[str, Any],
    prompt: str | None = None,
) -> dict[str, Any]:
    """
    Runs the improved simulated finance agent on one test case.
    """
    selected_prompt = prompt or BASE_IMPROVED_PROMPT
    tool_name = choose_tool_with_policy(case)

    with trace_span(
        "tool.call",
        {
            "tool.name": tool_name,
            "agent.version": "improved",
            "case.id": case.get("id"),
        },
    ):
        tool_output = run_tool(
            tool_name=tool_name,
            provided_context=case.get("provided_context", {}),
        )

    response_text = generate_improved_response(
        case=case,
        tool_output=tool_output,
    )

    return {
        "prompt": selected_prompt,
        "tool_used": tool_name,
        "tool_output": tool_output,
        "response_text": response_text,
    }