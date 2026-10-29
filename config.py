"""Settings, form options, colour palettes and demo scenarios for FinWise AI."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if one exists (never committed)

APP_NAME = "FinWise AI"
APP_ICON = "💎"
TAGLINE = "Personal financial analysis and a smarter monthly budget"

# ---- Model settings -------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
LANGUAGES = ["English", "Urdu", "Hindi", "Arabic", "Spanish", "French"]

# ---- Caching --------------------------------------------------------------
CACHE_OPTIONS = ["In-memory", "SQLite", "Off"]
SQLITE_PATH = os.path.join("data", "finwise_cache.db")

# ---- Form options ---------------------------------------------------------
CURRENCIES = {
    "USD": "$", "EUR": "€", "GBP": "£", "PKR": "Rs ", "INR": "₹",
    "AED": "AED ", "SAR": "SAR ", "CAD": "C$", "AUD": "A$",
}

# key, label, icon
EXPENSE_CATEGORIES = [
    ("housing", "Housing / Rent", "🏠"),
    ("food", "Food & Groceries", "🍽️"),
    ("transportation", "Transportation", "🚌"),
    ("utilities", "Utilities", "💡"),
    ("education", "Education", "🎓"),
    ("healthcare", "Healthcare", "🩺"),
    ("entertainment", "Entertainment", "🎬"),
    ("loan_debt", "Loan / Debt payments", "💳"),
    ("insurance", "Insurance", "🛡️"),
    ("other", "Other", "📦"),
]
CATEGORY_LABELS = {k: label for k, label, _ in EXPENSE_CATEGORIES}
CATEGORY_ICONS = {k: icon for k, _, icon in EXPENSE_CATEGORIES}

# 50/30/20 benchmark buckets
NEEDS_KEYS = ["housing", "food", "transportation", "utilities", "education", "healthcare", "insurance"]
WANTS_KEYS = ["entertainment", "other"]
DEBT_KEY = "loan_debt"

GOALS = {
    "save_money": "Save money",
    "emergency_fund": "Build an emergency fund",
    "pay_off_debt": "Pay off debt",
    "vacation": "Save for a vacation",
    "start_business": "Start a business",
    "improve_budgeting": "Improve budgeting",
}

# ---- Colour accents (the light/dark base is handled in theme.py) -----------
ACCENTS = {
    "Aurora":  ("#8b7bff", "#2dd4bf", "#f472b6"),
    "Sunset":  ("#ff7a59", "#ffc233", "#f43f5e"),
    "Emerald": ("#10b981", "#a3e635", "#22d3ee"),
    "Ocean":   ("#3b82f6", "#06b6d4", "#a78bfa"),
    "Royal":   ("#d4a017", "#f6d77a", "#9b6bff"),
}

DISCLAIMER = (
    "For informational and educational purposes only. FinWise AI is not financial advice, "
    "does not guarantee any outcome, and never connects to bank accounts or moves money. "
    "Please speak with a qualified financial professional before making real money decisions."
)


# ---- Demo scenarios (the five test cases from the assignment) --------------
def _exp(**kw):
    base = {k: 0.0 for k, _, _ in EXPENSE_CATEGORIES}
    base.update({k: float(v) for k, v in kw.items()})
    return base


SCENARIOS = {
    "1. Wealth builder (income 8000, spend ~2000)": dict(
        income=8000.0, savings=1500.0, goal="save_money",
        expenses=_exp(housing=900, food=350, transportation=200, utilities=150,
                      healthcare=100, entertainment=150, insurance=80, other=70)),
    "2. Overspending (income 2000, spend ~2600)": dict(
        income=2000.0, savings=0.0, goal="improve_budgeting",
        expenses=_exp(housing=900, food=450, transportation=250, utilities=180, education=100,
                      healthcare=80, entertainment=220, loan_debt=250, insurance=70, other=100)),
    "3. Debt heavy (income 5000, debt 2500)": dict(
        income=5000.0, savings=100.0, goal="pay_off_debt",
        expenses=_exp(housing=1000, food=400, transportation=250, utilities=180, healthcare=100,
                      entertainment=150, loan_debt=2500, insurance=80, other=80)),
    "4. Strong saver (income 4000, saves 1200)": dict(
        income=4000.0, savings=1200.0, goal="emergency_fund",
        expenses=_exp(housing=1000, food=400, transportation=200, utilities=150,
                      healthcare=80, entertainment=150, insurance=70, other=50)),
    "5. Break-even (income 3000, spend 3000)": dict(
        income=3000.0, savings=0.0, goal="save_money",
        expenses=_exp(housing=1100, food=500, transportation=250, utilities=200, education=150,
                      healthcare=100, entertainment=250, loan_debt=200, insurance=100, other=150)),
}


def env_api_key() -> str:
    """Key from .env — only offered in the UI if FINWISE_ALLOW_ENV_KEY=1 (local use)."""
    if os.getenv("FINWISE_ALLOW_ENV_KEY", "0") != "1":
        return ""
    return os.getenv("OPENAI_API_KEY", "").strip()
