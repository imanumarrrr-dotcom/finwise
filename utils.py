"""Helpers: money formatting, SAFE JSON parsing, rule-based fallback and report export."""
import json
import re
from datetime import datetime

from config import CATEGORY_LABELS, DISCLAIMER, GOALS

LIST_KEYS = ["top_priorities", "budget_recommendations", "savings_strategy", "next_month_action_plan"]


def fmt_money(value: float, symbol: str, decimals: int = 0) -> str:
    return f"{'-' if value < 0 else ''}{symbol}{abs(value):,.{decimals}f}"


def extract_json(text: str):
    """Pull a JSON object out of model text (handles ```json fences and extra chatter)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the model reply.")
    return json.loads(text[start:end + 1])


def _as_list(value) -> list:
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value] if isinstance(value, list) else []


def validate_analysis(data: dict, calc: dict) -> dict:
    """Coerce the parsed JSON into the exact schema, filling safe defaults."""
    out = {"financial_summary": str(data.get("financial_summary", "")).strip() or "No summary returned."}
    try:
        score = int(round(float(data.get("financial_health_score", calc["preliminary_score"]))))
    except (TypeError, ValueError):
        score = int(calc["preliminary_score"])
    out["financial_health_score"] = max(0, min(100, score))
    risk = str(data.get("risk_level", "")).strip().upper()
    out["risk_level"] = risk if risk in ("LOW", "MEDIUM", "HIGH") else calc["preliminary_risk"]
    spending = []
    rows = data.get("spending_analysis")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            spending.append({k: str(row.get(k, "")) for k in ("category", "observation", "recommendation")})
    out["spending_analysis"] = spending
    for key in LIST_KEYS:
        out[key] = _as_list(data.get(key))
    return out


def fallback_analysis(calc: dict) -> dict:
    """Rule-based report used when the LLM reply cannot be parsed (the app never crashes)."""
    spending = [{
        "category": CATEGORY_LABELS.get(k, k),
        "observation": f"{share:.0f}% of your income goes here.",
        "recommendation": "Review this line for savings you can make without hurting essentials.",
    } for k, _, share in calc["top_categories"]]
    return {
        "financial_summary": (f"Rule-based summary: expenses are {calc['expense_ratio']:.0f}% of income and "
                              f"savings are {calc['savings_ratio']:.0f}%. Score {calc['preliminary_score']:.0f}/100."),
        "financial_health_score": int(calc["preliminary_score"]),
        "spending_analysis": spending,
        "risk_level": calc["preliminary_risk"] if calc["preliminary_risk"] != "N/A" else "HIGH",
        "top_priorities": calc["flags"] or ["Keep tracking your spending every month."],
        "budget_recommendations": ["Aim toward the 50/30/20 guideline: needs / wants / savings and debt."],
        "savings_strategy": ["Automate a small saving on payday and increase it gradually."],
        "next_month_action_plan": ["Track every expense for 30 days.", "Trim your largest flexible category by 5-10%."],
    }


def parse_analysis(raw: str, calc: dict):
    """Return (analysis_dict, warning_or_None). Never raises."""
    try:
        return validate_analysis(extract_json(raw), calc), None
    except Exception as exc:  # invalid JSON, wrong types, etc.
        return fallback_analysis(calc), f"The AI reply was not valid JSON ({exc}). Showing a rule-based report instead."


def report_markdown(calc: dict, analysis: dict, narrative: str, symbol: str) -> str:
    """Plain-text financial report for download."""
    L = [f"# FinWise AI — Monthly Financial Report ({datetime.now():%Y-%m-%d})", "", f"> {DISCLAIMER}", "",
         "## Python calculations (deterministic)",
         f"- Income: {fmt_money(calc['monthly_income'], symbol)}",
         f"- Expenses: {fmt_money(calc['total_expenses'], symbol)} ({calc['expense_ratio']:.1f}%)",
         f"- Remaining: {fmt_money(calc['remaining_income'], symbol)}",
         f"- Savings: {fmt_money(calc['savings'], symbol)} ({calc['savings_ratio']:.1f}%)",
         f"- Goal: {GOALS.get(calc['financial_goal'], calc['financial_goal'])}",
         f"- Preliminary score: {calc['preliminary_score']:.0f}/100 · risk {calc['preliminary_risk']}", "",
         "## AI insights", analysis["financial_summary"], "",
         f"**AI health score:** {analysis['financial_health_score']}/100 · **Risk:** {analysis['risk_level']}", ""]
    for title, key in [("Top priorities", "top_priorities"), ("Budget recommendations", "budget_recommendations"),
                       ("Savings strategy", "savings_strategy"), ("Next-month action plan", "next_month_action_plan")]:
        L += [f"### {title}"] + [f"- {x}" for x in analysis[key]] + [""]
    if narrative:
        L += ["## Written recommendation", narrative]
    return "\n".join(L)
