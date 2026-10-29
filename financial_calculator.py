"""Deterministic financial maths — NO AI in this file.

The same inputs always produce the same outputs. The LLM only *interprets* these numbers.
"""
from __future__ import annotations

from typing import Dict

from .config import DEBT_KEY, NEEDS_KEYS, WANTS_KEYS


def safe_div(numerator: float, denominator: float, default=0.0):
    """Divide without crashing when the denominator is 0 (e.g. income = 0)."""
    return numerator / denominator if denominator else default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_band(score: float) -> str:
    """Educational health bands from the assignment."""
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Generally healthy"
    if score >= 40:
        return "Needs improvement"
    return "High attention"


def preliminary_score(savings_ratio: float, remaining_ratio: float,
                      expense_ratio: float, debt_ratio: float) -> float:
    """Weighted 0-100 heuristic. All ratios are percentages of income.

    savings 35 pts (full at >= 20%) | leftover 25 pts (full at >= 20%)
    expense ratio 25 pts (full at <= 60%, zero at >= 100%) | debt burden 15 pts (full <= 15%, zero >= 40%)
    """
    pts = 35 * clamp(savings_ratio / 20)
    pts += 25 * clamp(remaining_ratio / 20)
    pts += 25 * clamp((100 - expense_ratio) / 40)
    pts += 15 * clamp((40 - debt_ratio) / 25)
    return round(pts, 1)


def preliminary_risk(score: float, remaining: float, expense_ratio: float,
                     debt_ratio: float, remaining_ratio: float) -> str:
    if remaining < 0 or expense_ratio >= 95 or debt_ratio >= 40 or score < 40:
        return "HIGH"
    if score < 60 or debt_ratio >= 25 or remaining_ratio < 10:
        return "MEDIUM"
    return "LOW"


def calculate_financials(income: float, expenses: Dict[str, float], savings: float,
                         goal: str = "save_money") -> dict:
    income = max(float(income or 0), 0.0)
    savings = max(float(savings or 0), 0.0)
    expenses = {k: max(float(v or 0), 0.0) for k, v in expenses.items()}

    total = sum(expenses.values())
    remaining = income - total
    savings_ratio = safe_div(savings, income) * 100
    expense_ratio = safe_div(total, income) * 100
    remaining_ratio = safe_div(remaining, income) * 100
    debt_ratio = safe_div(expenses.get(DEBT_KEY, 0.0), income) * 100

    if income <= 0:
        score, risk = 0.0, ("HIGH" if total > 0 else "N/A")
    else:
        score = preliminary_score(savings_ratio, remaining_ratio, expense_ratio, debt_ratio)
        risk = preliminary_risk(score, remaining, expense_ratio, debt_ratio, remaining_ratio)

    shares = {k: safe_div(v, income) * 100 for k, v in expenses.items()}
    top = sorted(((k, v, shares[k]) for k, v in expenses.items() if v > 0),
                 key=lambda t: t[1], reverse=True)[:3]

    # 50/30/20 benchmark (needs / wants / savings + debt) as % of income
    needs = sum(expenses.get(k, 0) for k in NEEDS_KEYS)
    wants = sum(expenses.get(k, 0) for k in WANTS_KEYS)
    save_debt = expenses.get(DEBT_KEY, 0) + savings
    buckets = {
        "Needs": {"amount": needs, "pct": safe_div(needs, income) * 100, "target": 50},
        "Wants": {"amount": wants, "pct": safe_div(wants, income) * 100, "target": 30},
        "Savings & debt": {"amount": save_debt, "pct": safe_div(save_debt, income) * 100, "target": 20},
    }

    emergency_target = total * 3
    months_to_emergency = safe_div(emergency_target, savings, default=None)
    discretionary = expenses.get("entertainment", 0) + expenses.get("other", 0)
    savings_potential = max(remaining, 0.0) + 0.25 * discretionary

    flags = []
    if income <= 0:
        flags.append("No income entered, so ratios cannot be calculated.")
    else:
        if remaining < 0:
            flags.append("Spending is higher than income this month (a deficit).")
        elif remaining == 0:
            flags.append("Income and spending are exactly equal, so there is no buffer for surprises.")
        if debt_ratio >= 30:
            flags.append(f"Debt payments take {debt_ratio:.0f}% of income, which is a heavy load.")
        if savings_ratio < 10:
            flags.append(f"Savings are {savings_ratio:.0f}% of income, below the common 10-20% guideline.")
        if top and top[0][2] >= 40:
            flags.append(f"One category ({top[0][0].replace('_', ' ')}) uses {top[0][2]:.0f}% of income.")

    return {
        "monthly_income": income, "total_expenses": total, "remaining_income": remaining,
        "savings": savings, "savings_ratio": savings_ratio, "expense_ratio": expense_ratio,
        "remaining_ratio": remaining_ratio, "debt_ratio": debt_ratio,
        "preliminary_score": score, "score_band": score_band(score), "preliminary_risk": risk,
        "category_amounts": expenses, "category_shares": shares, "top_categories": top,
        "buckets": buckets, "emergency_target": emergency_target,
        "months_to_emergency": months_to_emergency, "savings_potential": savings_potential,
        "flags": flags, "financial_goal": goal,
    }
