"""Prompt engineering: PromptTemplate, ChatPromptTemplate and the JSON schema.

Python has already done the maths, so the prompts tell the model to INTERPRET the numbers,
never to recompute them, and to stay inside the educational safety rules.
"""
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from .config import CATEGORY_LABELS, GOALS
from .utils import fmt_money

# ---- System prompt (role + safety) ----------------------------------------
SYSTEM_PROMPT = """You are FinWise AI, an educational personal-finance assistant inside a classroom FinTech prototype.

SAFETY RULES (always follow):
- Everything you say is for informational and educational purposes only.
- Never guarantee any financial outcome, return, or saving amount.
- Never recommend specific stocks, funds, crypto, loans or products, and give no tax or legal advice.
- You cannot access bank accounts or move money; never ask for sensitive personal data.
- Use ONLY the numbers provided. Python already calculated them; do not recompute or contradict them.
- Be supportive and non-judgemental. Remind the user to consult a qualified financial professional.
Write all text values in {language}."""

# ---- Required JSON schema --------------------------------------------------
JSON_SCHEMA = {
    "financial_summary": "",
    "financial_health_score": 0,
    "spending_analysis": [{"category": "", "observation": "", "recommendation": ""}],
    "risk_level": "",
    "top_priorities": [],
    "budget_recommendations": [],
    "savings_strategy": [],
    "next_month_action_plan": [],
}
# braces are doubled so LangChain does not treat the schema as template variables
SCHEMA_TEXT = json.dumps(JSON_SCHEMA, indent=2).replace("{", "{{").replace("}", "}}")

JSON_RULES = f"""Return ONLY one valid JSON object (no markdown, no extra text) with exactly this schema:
{SCHEMA_TEXT}
Rules: financial_health_score is an integer 0-100 (80-100 strong, 60-79 generally healthy, 40-59 needs improvement,
below 40 high attention). risk_level is exactly LOW, MEDIUM or HIGH. spending_analysis covers the 3-5 most important
categories. Each list has 3-5 short, specific strings. Stay consistent with the Python numbers."""

# ---- PromptTemplate (single string, all required variables) ----------------
DATA_TEMPLATE = """FINANCIAL DATA (calculated by Python, currency {currency}):
- monthly_income: {currency}{monthly_income}
- total_expenses: {currency}{total_expenses}
- remaining_income: {currency}{remaining_income}
- current monthly savings: {currency}{savings}
- savings_ratio: {savings_ratio}% of income
- expense_ratio: {expense_ratio}% of income
- debt payments: {debt_ratio}% of income
- financial_goal: {financial_goal}
- Python preliminary score: {preliminary_score}/100 (risk: {preliminary_risk})
- Python warning flags: {flags}

EXPENSE BREAKDOWN:
{expense_breakdown}"""

DATA_VARIABLES = [
    "currency", "monthly_income", "total_expenses", "remaining_income", "savings", "savings_ratio",
    "expense_ratio", "debt_ratio", "financial_goal", "preliminary_score", "preliminary_risk",
    "flags", "expense_breakdown",
]
FINANCIAL_PROMPT = PromptTemplate(input_variables=DATA_VARIABLES, template=DATA_TEMPLATE)

# ---- ChatPromptTemplates (System + Human) ---------------------------------
ANALYSIS_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", DATA_TEMPLATE + "\n\nTASK: Analyse this situation and produce the structured report.\n" + JSON_RULES),
])

NARRATIVE_INSTRUCTIONS = """TASK: Write a warm, clear recommendation in {language} (about 220-280 words, markdown).
Use these bold section titles: **Where you stand**, **Biggest opportunities**, **Your next 30 days**, **Friendly reminder**.
Refer to real numbers and the user's goal. End the reminder with the educational disclaimer."""
NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", DATA_TEMPLATE + "\n\n" + NARRATIVE_INSTRUCTIONS),
])

CHAT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nThe user's calculated finances for context:\n" + DATA_TEMPLATE
     + "\n\nAnswer follow-up questions briefly (under 150 words) using this context."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def build_inputs(calc: dict, symbol: str, language: str = "English") -> dict:
    """Turn the Python calculation dict into the prompt variables."""
    amounts, shares = calc["category_amounts"], calc["category_shares"]
    breakdown = "\n".join(
        f"- {CATEGORY_LABELS.get(k, k)}: {fmt_money(v, symbol)} ({shares[k]:.1f}% of income)"
        for k, v in amounts.items())
    return {
        "currency": symbol,
        "monthly_income": f"{calc['monthly_income']:,.2f}",
        "total_expenses": f"{calc['total_expenses']:,.2f}",
        "remaining_income": f"{calc['remaining_income']:,.2f}",
        "savings": f"{calc['savings']:,.2f}",
        "savings_ratio": f"{calc['savings_ratio']:.1f}",
        "expense_ratio": f"{calc['expense_ratio']:.1f}",
        "debt_ratio": f"{calc['debt_ratio']:.1f}",
        "financial_goal": GOALS.get(calc["financial_goal"], calc["financial_goal"]),
        "preliminary_score": f"{calc['preliminary_score']:.0f}",
        "preliminary_risk": calc["preliminary_risk"],
        "flags": "; ".join(calc["flags"]) or "none",
        "expense_breakdown": breakdown,
        "language": language,
    }
