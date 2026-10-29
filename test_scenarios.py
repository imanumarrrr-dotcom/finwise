"""Checks the 5 assignment scenarios against the deterministic calculator (no API key needed)."""
from src.config import SCENARIOS
from src.financial_calculator import calculate_financials
from src.utils import parse_analysis


def calc(name):
    s = SCENARIOS[name]
    return calculate_financials(s["income"], s["expenses"], s["savings"], s["goal"])


def test_scenarios():
    c = {n[0]: calc(n) for n in SCENARIOS}
    assert c["1"]["remaining_income"] > 5000 and c["1"]["preliminary_risk"] == "LOW"
    assert c["2"]["remaining_income"] < 0 and c["2"]["expense_ratio"] > 100 and c["2"]["preliminary_risk"] == "HIGH"
    assert c["3"]["preliminary_risk"] in ("MEDIUM", "HIGH") and c["3"]["debt_ratio"] == 50
    assert round(c["4"]["savings_ratio"]) == 30 and c["4"]["preliminary_risk"] == "LOW"
    assert c["5"]["remaining_income"] == 0 and c["5"]["preliminary_risk"] in ("MEDIUM", "HIGH")


def test_zero_income_and_bad_json():
    c = calculate_financials(0, {"housing": 100}, 0)
    assert c["savings_ratio"] == 0 and c["preliminary_score"] == 0
    data, warn = parse_analysis("not json at all", c)
    assert warn and data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    data, warn = parse_analysis('```json\n{"financial_health_score": 150, "risk_level": "low"}\n```', c)
    assert warn is None and data["financial_health_score"] == 100 and data["risk_level"] == "LOW"
