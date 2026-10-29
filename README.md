# 💎 FinWise AI — AI-Powered Personal Financial Analysis & Smart Budget Assistant

A **LangChain + OpenAI + Streamlit** FinTech prototype. Users enter income, expenses, savings and a goal;
**Python** calculates the numbers and **the LLM** turns them into structured, educational budgeting insights.

> ⚠️ **Educational use only.** Not financial advice. No guaranteed outcomes, no bank connections, no transactions.
> Every screen shows a disclaimer and points users to a qualified financial professional.

## Features
- 🔐 **API-key gate** — the console unlocks only after a verified OpenAI key (kept in session memory, never saved)
- 🎨 Animated UI: dark/light mode, 5 accent palettes, colour-wipe theme transition, animated score rings
- 🧮 Deterministic calculations + 0-100 preliminary score and rule-based risk (`financial_calculator.py`)
- 🤖 `ChatOpenAI`, `PromptTemplate`, `ChatPromptTemplate`, `SystemMessage/HumanMessage/AIMessage`, reusable `LLMChain`
- 🧾 Forced JSON output, validated and parsed safely (rule-based fallback if the reply is invalid)
- ⌨️ Streaming recommendation via `llm.stream()` + `st.write_stream()`
- ⚡ `InMemoryCache` / `SQLiteCache` / Off, switchable in the sidebar
- Extras: 5 built-in test scenarios, charts (donut, 50/30/20), multi-currency & language, follow-up chat with memory,
  session history comparison, report download (.md / .json), developer console tab

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional: put OPENAI_API_KEY here
streamlit run app.py
```
Paste your `sk-...` key on the unlock screen. (Optional: set `FINWISE_ALLOW_ENV_KEY=1` in `.env` to get a
"Use key from .env" button.) `.env` is git-ignored — never commit a real key.

## Python vs AI
| Python (`financial_calculator.py`) | LLM (`chains.py`) |
|---|---|
| Totals, ratios, preliminary score, risk | Explains, prioritises, writes the plan |
| Same input → same output, free, instant | Wording varies, costs tokens |
The prompts tell the model to interpret Python's numbers, never recompute them.

## Caching
`set_llm_cache(...)` registers one global cache that LangChain checks before every call.
| | InMemoryCache | SQLiteCache |
|---|---|---|
| Stored in | RAM | `data/finwise_cache.db` |
| Survives restart | No | Yes |
| Best for | One session | Reuse across sessions |
Run the same analysis twice: the second is instant and makes no API call.

## Folder structure
```
app.py  requirements.txt  .env.example  README.md
src/ config.py prompts.py financial_calculator.py chains.py cache_manager.py utils.py theme.py
tests/test_scenarios.py   docs/FinTech_AI_Assignment.pdf
```
Run tests (no API key needed): `pytest tests`
