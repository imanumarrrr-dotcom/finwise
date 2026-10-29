"""FinWise AI — Streamlit UI.  Run with:  streamlit run app.py"""
import copy
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from cache_manager import cache_info, clear_cache, configure_cache
from src.chains import (build_analysis_chain, build_llm, message_demo, run_analysis, stream_chat,
                        stream_recommendations, validate_key)
from src.config import (ACCENTS, APP_ICON, APP_NAME, CACHE_OPTIONS, CATEGORY_ICONS, CATEGORY_LABELS, CURRENCIES,
                        DEFAULT_MODEL, DISCLAIMER, EXPENSE_CATEGORIES, GOALS, LANGUAGES, MODEL_OPTIONS,
                        SCENARIOS, TAGLINE, env_api_key)
from src.financial_calculator import calculate_financials
from src.prompts import FINANCIAL_PROMPT, build_inputs
from src.theme import inject_css, plotly_layout
from src.utils import fmt_money, parse_analysis, report_markdown

st.set_page_config(page_title=f"{APP_NAME} — AI Budget Assistant", page_icon=APP_ICON, layout="wide")

NAV = ["📝 Input", "📊 Dashboard", "💬 Ask FinWise", "🛠️ Console"]
INPUT_KEYS = ["inc", "sav", "goal", "cur"] + [f"exp_{k}" for k, _, _ in EXPENSE_CATEGORIES]
SETTING_KEYS = ["model", "temperature", "language", "cache_mode", "mode", "accent", "nav"]
DEFAULTS = dict(api_key="", key_ok=False, model=DEFAULT_MODEL, temperature=0.3, language="English",
                cache_mode="In-memory", mode="Dark", accent="Aurora", nav=NAV[0], run_requested=False,
                pending_stream=False, analysis=None, calc=None, inputs=None, symbol="$", narrative="",
                warning=None, history=[], chat=[], last_ms=None, cache_hit=False, raw_json="",
                inc=3000.0, sav=0.0, goal="save_money", cur="USD",
                **{f"exp_{k}": 0.0 for k, _, _ in EXPENSE_CATEGORIES})
S = st.session_state
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, copy.deepcopy(_v))
for _k in INPUT_KEYS + SETTING_KEYS:  # keep widget values alive while their page is not rendered
    S[_k] = S[_k]

st.markdown(inject_css(S.mode, S.accent), unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def current_calc() -> dict:
    return calculate_financials(S.inc, {k: S[f"exp_{k}"] for k, _, _ in EXPENSE_CATEGORIES}, S.sav, S.goal)


def score_color(s: float) -> str:
    return "#22c55e" if s >= 80 else "#84cc16" if s >= 60 else "#f59e0b" if s >= 40 else "#ef4444"


def badge(level: str) -> str:
    return f'<span class="badge {level.replace("/", "")}">{level if level != "N/A" else "N/A"}</span>'.replace(
        'class="badge N/A"', 'class="badge NA"')


def ring(score: float, label: str, caption: str) -> str:
    off, col = 339.3 * (1 - max(0, min(100, score)) / 100), score_color(score)
    return (f'<div class="ring-wrap"><svg viewBox="0 0 120 120" width="150" height="150">'
            f'<circle cx="60" cy="60" r="54" class="ring-bg"/>'
            f'<circle cx="60" cy="60" r="54" class="ring-fg" style="stroke:{col};stroke-dashoffset:{off:.1f}"/>'
            f'<text x="60" y="64" text-anchor="middle" class="ring-num">{score:.0f}</text>'
            f'<text x="60" y="80" text-anchor="middle" class="ring-sub">/ 100</text></svg>'
            f'<div class="ring-label">{label}</div><div class="ring-cap">{caption}</div></div>')


def disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)


def hero(title: str, sub: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------- callbacks
def load_scenario() -> None:
    sc = SCENARIOS[S.scn]
    S.inc, S.sav, S.goal = sc["income"], sc["savings"], sc["goal"]
    for k, _, _ in EXPENSE_CATEGORIES:
        S[f"exp_{k}"] = sc["expenses"].get(k, 0.0)


def request_run() -> None:
    S.run_requested, S.nav = True, NAV[1]


def reset_session() -> None:
    keep = {"api_key", "key_ok", "mode", "accent", "model"}
    for k in list(S.keys()):
        if k not in keep:
            del S[k]


def lock_console() -> None:
    S.api_key, S.key_ok = "", False


# ------------------------------------------------------------------ sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_NAME}")
        st.caption(TAGLINE)
        st.markdown("##### 🎨 Appearance")
        st.radio("Mode", ["Dark", "Light"], key="mode", horizontal=True)
        st.selectbox("Accent palette", list(ACCENTS), key="accent")
        if S.key_ok:
            with st.expander("🤖 Model settings", expanded=True):
                st.selectbox("OpenAI model", MODEL_OPTIONS, key="model")
                st.slider("Creativity (temperature)", 0.0, 1.0, step=0.1, key="temperature")
                st.selectbox("Response language", LANGUAGES, key="language")
            with st.expander("⚡ Caching"):
                st.radio("Cache type", CACHE_OPTIONS, key="cache_mode")
                st.caption("In-memory = fastest, lost on restart. SQLite = file on disk, survives restarts. "
                           "A cache hit makes no new API call.")
                st.button("🧹 Clear this cache", on_click=lambda: clear_cache(S.cache_mode))
            c1, c2 = st.columns(2)
            c1.button("🔄 Reset", on_click=reset_session, help="Clear all data and results")
            c2.button("🔒 Lock", on_click=lock_console, help="Forget the API key")
        with st.expander("ℹ️ Project info"):
            st.markdown("**AI-Powered Personal Financial Analysis & Smart Budget Assistant**  \n"
                        "LangChain + OpenAI + Streamlit. Python does the maths; the LLM explains it.")
        disclaimer()


# -------------------------------------------------------------- API key gate
def gate() -> None:
    hero(f"{APP_ICON} Welcome to {APP_NAME}", "Enter your OpenAI API key to unlock the console.")
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid, st.container(border=True):
        st.markdown("### 🔐 Unlock the console")
        st.caption("Your key stays in this browser session's memory only — it is never written to disk or logged.")
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", label_visibility="collapsed")
        use_env = False
        if env_api_key():
            use_env = st.button("Use key from .env", width="stretch")
        if st.button("🔓 Unlock console", width="stretch") or use_env:
            key = env_api_key() if use_env else key.strip()
            if not key.startswith("sk-"):
                st.error("An OpenAI key starts with `sk-`. Please paste a valid key.")
            else:
                with st.spinner("Verifying key with OpenAI..."):
                    ok, msg = validate_key(key, S.model)
                if ok:
                    S.api_key, S.key_ok = key, True
                    st.toast("Console unlocked 🎉")
                    st.rerun()
                st.error(msg)
        st.markdown("[Get a key at platform.openai.com](https://platform.openai.com/api-keys)")
    disclaimer()


# ---------------------------------------------------------------- page: input
def page_input() -> None:
    hero("Tell me about your month", "Enter income, expenses and a goal — Python calculates, AI explains.")
    with st.expander("⚡ Try one of the 5 assignment test scenarios"):
        a, b = st.columns([4, 1])
        a.selectbox("Scenario", list(SCENARIOS), key="scn", label_visibility="collapsed")
        b.button("Load", on_click=load_scenario, width="stretch")
    with st.form("finance_form"):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Monthly income", min_value=0.0, step=100.0, key="inc")
        c2.selectbox("Currency", list(CURRENCIES), key="cur")
        c3.selectbox("Financial goal", list(GOALS), format_func=GOALS.get, key="goal")
        with st.expander("🧾 Monthly expenses", expanded=True):
            cols = st.columns(2)
            for i, (k, label, icon) in enumerate(EXPENSE_CATEGORIES):
                cols[i % 2].number_input(f"{icon} {label}", min_value=0.0, step=50.0, key=f"exp_{k}")
        st.number_input("🏦 Current monthly savings", min_value=0.0, step=50.0, key="sav")
        st.form_submit_button("🚀 Analyse my finances", on_click=request_run)
    disclaimer()


# ------------------------------------------------------------ page: dashboard
def run_pipeline() -> None:
    S.run_requested = False
    calc = current_calc()
    symbol = CURRENCIES[S.cur]
    inputs = build_inputs(calc, symbol, S.language)
    configure_cache(S.cache_mode)
    t0, warning = time.perf_counter(), None
    try:
        with st.spinner("🧠 Python is calculating and the AI is analysing..."):
            chain = build_analysis_chain(build_llm(S.api_key, S.model, S.temperature, json_mode=True))
            raw = run_analysis(chain, inputs)
        analysis, warning = parse_analysis(raw, calc)
    except Exception as exc:
        raw = ""
        analysis, warning = parse_analysis("", calc)[0], f"AI call failed ({type(exc).__name__}): {exc}"
    ms = (time.perf_counter() - t0) * 1000
    S.update(calc=calc, symbol=symbol, inputs=inputs, analysis=analysis, warning=warning, raw_json=raw,
             last_ms=ms, cache_hit=(S.cache_mode != "Off" and ms < 400 and not warning),
             pending_stream=not (warning or "").startswith("AI call failed"), narrative="")
    S.history.append(dict(Time=time.strftime("%H:%M:%S"), Income=calc["monthly_income"],
                          Expenses=calc["total_expenses"], Remaining=calc["remaining_income"],
                          **{"Python score": calc["preliminary_score"],
                             "AI score": analysis["financial_health_score"], "Risk": analysis["risk_level"]}))


def charts(calc: dict, sym: str) -> None:
    lay = plotly_layout(S.mode)
    rows = [(CATEGORY_LABELS[k], v) for k, v in calc["category_amounts"].items() if v > 0]
    c1, c2 = st.columns(2)
    if rows:
        df = pd.DataFrame(rows, columns=["Category", "Amount"])
        fig = px.pie(df, names="Category", values="Amount", hole=.55, title="Where your money goes",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(**lay)
        c1.plotly_chart(fig, width="stretch")
    b = calc["buckets"]
    fig = go.Figure([go.Bar(name="Your %", x=list(b), y=[v["pct"] for v in b.values()], marker_color=ACCENTS[S.accent][0]),
                     go.Bar(name="50/30/20 target %", x=list(b), y=[v["target"] for v in b.values()],
                            marker_color=ACCENTS[S.accent][1])])
    fig.update_layout(barmode="group", title="50/30/20 rule check", **lay)
    c2.plotly_chart(fig, width="stretch")


def page_dashboard() -> None:
    if S.run_requested:
        run_pipeline()
    calc, an = S.calc, S.analysis
    if not calc:
        hero("Your dashboard", "No analysis yet.")
        st.info("Go to **📝 Input**, fill in your numbers and press **Analyse my finances**.")
        disclaimer()
        return
    sym = S.symbol
    hero("Your financial dashboard", "Python numbers on the left of every card, AI insight on the right.")
    disclaimer()
    if S.warning:
        st.warning(S.warning)
    if S.last_ms is not None:
        st.caption(f"⏱️ Analysis took {S.last_ms:.0f} ms" + (" — ⚡ cache hit, no API call made" if S.cache_hit else "")
                   + f" · cache: {S.cache_mode}")

    st.markdown("#### 💼 Financial overview")
    m = st.columns(4)
    m[0].metric("Monthly income", fmt_money(calc["monthly_income"], sym))
    m[1].metric("Total expenses", fmt_money(calc["total_expenses"], sym), f"{calc['expense_ratio']:.0f}% of income",
                delta_color="inverse")
    m[2].metric("Remaining balance", fmt_money(calc["remaining_income"], sym), f"{calc['remaining_ratio']:+.0f}% of income")
    m[3].metric("Current savings", fmt_money(calc["savings"], sym), f"{calc['savings_ratio']:.0f}% savings ratio")
    for f in calc["flags"]:
        st.warning(f)

    st.markdown("#### 🎯 Health scores — Python vs AI")
    r1, r2, r3 = st.columns([1, 1, 1.3])
    r1.markdown(ring(calc["preliminary_score"], "🐍 Python score", calc["score_band"]), unsafe_allow_html=True)
    r2.markdown(ring(an["financial_health_score"], "🤖 AI score", "LLM judgement"), unsafe_allow_html=True)
    with r3:
        st.markdown(f"**Risk level**  \n🐍 Python: {badge(calc['preliminary_risk'])} &nbsp; 🤖 AI: {badge(an['risk_level'])}",
                    unsafe_allow_html=True)
        st.progress(an["financial_health_score"] / 100, text=f"AI health score {an['financial_health_score']}/100")
        st.progress(min(calc["expense_ratio"], 100) / 100, text=f"Expense ratio {calc['expense_ratio']:.0f}%")
        st.progress(min(calc["savings_ratio"], 100) / 100, text=f"Savings ratio {calc['savings_ratio']:.0f}%")

    t1, t2, t3, t4 = st.tabs(["🧾 Spending analysis", "🎯 Priorities & budget", "💰 Savings & plan", "📈 Charts"])
    with t1:
        st.info(an["financial_summary"])
        for row in an["spending_analysis"]:
            with st.expander(f"📌 {row['category']}"):
                st.markdown(f"**Observation:** {row['observation']}")
                st.success(f"💡 {row['recommendation']}")
    with t2:
        a, b = st.columns(2)
        a.markdown("##### Top priorities")
        for i, x in enumerate(an["top_priorities"], 1):
            a.markdown(f"**{i}.** {x}")
        b.markdown("##### Budget recommendations")
        for x in an["budget_recommendations"]:
            b.markdown(f"- {x}")
    with t3:
        a, b = st.columns(2)
        a.markdown("##### Savings strategy")
        for x in an["savings_strategy"]:
            a.markdown(f"- {x}")
        b.markdown("##### Next-month action plan")
        for x in an["next_month_action_plan"]:
            b.markdown(f"☐ {x}")
        if calc["months_to_emergency"]:
            st.metric("Months to a 3-month emergency fund at current savings", f"{calc['months_to_emergency']:.0f}")
        st.metric("Estimated monthly savings potential", fmt_money(calc["savings_potential"], sym))
    with t4:
        charts(calc, sym)

    st.markdown("#### ✍️ Personalised recommendation")
    if S.pending_stream:
        S.pending_stream = False
        try:
            llm = build_llm(S.api_key, S.model, S.temperature)
            S.narrative = st.write_stream(stream_recommendations(llm, S.inputs))
        except Exception as exc:
            st.error(f"Streaming failed ({type(exc).__name__}). Please try again.")
    else:
        st.markdown(S.narrative or "_No written recommendation yet._")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download report (.md)", report_markdown(calc, an, S.narrative, sym), "finwise_report.md",
                       width="stretch")
    d2.download_button("⬇️ Download AI JSON", S.raw_json or "{}", "finwise_analysis.json", width="stretch")
    with st.expander("🕑 Session history & month-to-month comparison"):
        df = pd.DataFrame(S.history)
        st.dataframe(df, width="stretch")
        if len(df) > 1:
            st.line_chart(df.set_index("Time")[["Python score", "AI score"]])


# ----------------------------------------------------------------- page: chat
def page_chat() -> None:
    hero("Ask FinWise", "Follow-up questions about your budget — with conversation memory.")
    disclaimer()
    calc = S.calc or current_calc()
    inputs = S.inputs or build_inputs(calc, CURRENCIES[S.cur], S.language)
    for msg in S.chat:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)
    q = st.chat_input("e.g. How can I cut my food spending?")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                configure_cache(S.cache_mode)
                llm = build_llm(S.api_key, S.model, S.temperature)
                reply = st.write_stream(stream_chat(llm, inputs, list(S.chat), q))
                S.chat += [HumanMessage(content=q), AIMessage(content=reply)]
            except Exception as exc:
                st.error(f"Chat failed ({type(exc).__name__}).")
    if S.chat:
        st.button("🗑️ Clear conversation", on_click=lambda: S.chat.clear())


# --------------------------------------------------------------- page: console
def page_console() -> None:
    hero("Developer console", "See exactly how LangChain, prompts, messages and caching work.")
    calc = S.calc or current_calc()
    inputs = S.inputs or build_inputs(calc, CURRENCIES[S.cur], S.language)
    t1, t2, t3, t4 = st.tabs(["🐍 Python vs 🤖 AI", "💬 Messages & prompts", "⚡ Cache", "🧪 Raw output"])
    with t1:
        st.table(pd.DataFrame({
            "Python (deterministic)": ["Totals, ratios, 0-100 score, rule-based risk", "Same input → same output",
                                       "Free and instant", "Guards divide-by-zero"],
            "LLM (probabilistic)": ["Explains, prioritises, writes the plan", "Wording may vary",
                                    "Costs tokens (cache helps)", "Output validated & parsed safely"]}))
        st.json({k: v for k, v in calc.items() if k in ("total_expenses", "remaining_income", "savings_ratio",
                                                        "expense_ratio", "preliminary_score", "preliminary_risk")})
    with t2:
        st.markdown("**SystemMessage / HumanMessage / AIMessage demo**")
        for msg in message_demo(S.language):
            with st.expander(type(msg).__name__):
                st.code(msg.content, language="text")
        st.markdown("**PromptTemplate rendered with your data**")
        st.code(FINANCIAL_PROMPT.format(**{k: inputs[k] for k in FINANCIAL_PROMPT.input_variables}), language="text")
    with t3:
        info = cache_info(S.cache_mode)
        c = st.columns(3)
        c[0].metric("Cache type", info["mode"])
        c[1].metric("Entries", info["entries"])
        c[2].metric("Survives restart", "Yes" if info["survives_restart"] else "No")
        st.caption(f"Location: {info['location']}. Run the same analysis twice to see the second one return instantly.")
    with t4:
        st.code(S.raw_json or "Run an analysis first.", language="json")


# ---------------------------------------------------------------------- main
sidebar()
if not S.key_ok:
    gate()
else:
    st.radio("Navigate", NAV, key="nav", horizontal=True, label_visibility="collapsed")
    {NAV[0]: page_input, NAV[1]: page_dashboard, NAV[2]: page_chat, NAV[3]: page_console}[S.nav]()
