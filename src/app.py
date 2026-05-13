"""
Streamlit UI: multi-company IPO RAG and IPO briefs.
(Market Performance tab temporarily disabled — see bottom of file.)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agents.pipeline import run_brief_pipeline, run_qa_pipeline_streaming
from financials import COMPANIES


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINANCIALS_PATH = PROJECT_ROOT / "data" / "financials.json"

try:
    with open(FINANCIALS_PATH, encoding="utf-8") as _f:
        FINANCIALS: dict[str, Any] = json.load(_f)
except (OSError, json.JSONDecodeError):
    FINANCIALS = {}

COMPANY_MAP: dict[str, dict[str, Any]] = {c["name"]: c for c in COMPANIES}
COMPANY_NAMES: list[str] = [c["name"] for c in COMPANIES]

TICKER_COLORS: dict[str, str] = {
    "CART": "#16a34a",
    "RDDT": "#ea580c",
    "KVYO": "#9333ea",
    "ARM": "#2563eb",
    "COIN": "#0d9488",
    "DASH": "#dc2626",
    "SNOW": "#06b6d4",
}

INTENT_BADGES: dict[str, str] = {
    "single": "🔵 single",
    "comparative": "🟣 comparative",
    "open_ended": "🟢 open_ended",
    "verification": "🟠 verification",
}

RISK_CATEGORY_STYLE: dict[str, tuple[str, str]] = {
    "market": ("#1e40af", "Market"),
    "regulatory": ("#ea580c", "Regulatory"),
    "competition": ("#9333ea", "Competition"),
    "financial": ("#dc2626", "Financial"),
}


def _ticker_pill_html(ticker: str, label: str | None = None) -> str:
    bg = TICKER_COLORS.get(ticker.upper(), "#4b5563")
    text = label or ticker
    return (
        f"<span style='display:inline-block;margin:2px 4px 2px 0;padding:0.25rem 0.6rem;"
        f"border-radius:999px;background:{bg};color:white;font-size:0.78rem;font-weight:600;'>"
        f"{text}</span>"
    )


def _company_pills_row(company_keys: list[str]) -> str:
    parts: list[str] = []
    for name in company_keys:
        co = COMPANY_MAP.get(name)
        if not co:
            continue
        t = str(co.get("ticker", ""))
        parts.append(_ticker_pill_html(t, t))
    return "".join(parts) if parts else "<span>(none)</span>"


def _intent_badge_md(intent: str) -> str:
    label = INTENT_BADGES.get(intent, intent)
    return f"<span style='font-weight:600;'>{label}</span>"


def _confidence_row(score: Any) -> str:
    try:
        n = int(score)
    except (TypeError, ValueError):
        n = 10
    if n >= 8:
        emoji = "✅"
    elif n >= 5:
        emoji = "⚠️"
    else:
        emoji = "❌"
    return f"{emoji} **Confidence:** {n}/10"


# Plotly `add_vline` + `annotation_position` raised TypeError (mixed x-axis types in Plotly 6.x).
# yfinance / plotly imports and `get_market_data()` removed here; re-enable when fixing the chart.
# def _parse_dollar_amount(text: str) -> float | None: ...
# def _parse_first_int(text: str) -> int | None: ...
# def get_market_data(company_name: str) -> dict[str, Any]: ...


def _financial_source_caption(company_name: str) -> str:
    row = FINANCIALS.get(company_name, {}) if isinstance(FINANCIALS, dict) else {}
    src = str(row.get("source", "") or "").lower()
    if src == "xbrl":
        return "📊 Source: XBRL"
    if src == "pdfplumber":
        return "📋 Source: pdfplumber"
    return "📋 Source: unavailable / local JSON"


# --- Page setup ---
st.set_page_config(
    page_title="IPO Prospectus Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
    div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    .metric-hover {
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .metric-hover:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        border-radius: 10px;
    }
    .risk-card {
        font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .hl-box {
        border-radius: 10px;
        padding: 0.75rem;
        border: 1px solid rgba(148,163,184,0.25);
        min-height: 4rem;
        font-size: 0.95rem;
    }
    div[data-testid="stChatMessage"] { margin-bottom: 0.35rem; }
    div[data-testid="stExpander"] { margin-top: 0.15rem; margin-bottom: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## 📈 IPO Prospectus Analyzer")
st.markdown(
    "*Multi-company RAG across 7 IPO prospectuses · LangChain · Qdrant · GPT-4.1*"
)

pill_row = "".join(
    _ticker_pill_html(str(c["ticker"]), str(c["ticker"])) for c in COMPANIES
)
st.markdown(pill_row, unsafe_allow_html=True)
st.markdown("")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

tab_qa, tab_brief = st.tabs(["💬 Q&A", "📄 IPO Brief"])

# ----- Tab 1: Q&A -----
with tab_qa:
    user_question = st.chat_input("Ask anything about these 7 IPOs...")

    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry.get("question", ""))

        with st.chat_message("assistant"):
            st.write(entry.get("answer", ""))

        with st.expander("🔍 Agent details", expanded=False):
            intent = str(entry.get("intent", ""))
            st.markdown(
                f"**Intent:** {_intent_badge_md(intent)}",
                unsafe_allow_html=True,
            )
            companies = entry.get("companies_detected") or []
            st.markdown(
                "**Companies:** " + _company_pills_row(companies),
                unsafe_allow_html=True,
            )
            st.markdown(_confidence_row(entry.get("confidence")))
            retried = entry.get("retried", False)
            st.write(f"**Retrieval retried:** {'yes' if retried else 'no'}")
            st.write("**Sub-queries:**")
            for i, sq in enumerate(entry.get("sub_queries", []), start=1):
                st.write(f"{i}. {sq}")

        st.caption(
            "Pages referenced: "
            + (
                ", ".join(str(p) for p in (entry.get("pages") or []))
                or "None"
            )
        )

    if user_question:
        try:
            with st.chat_message("user"):
                st.write(user_question)

            tail = st.session_state["conversation_history"][-3:]
            with st.spinner("Retrieving answer..."):
                meta, answer_gen = run_qa_pipeline_streaming(
                    user_question, conversation_history=tail
                )

            meta_line = (
                "🔍 Searching "
                + _company_pills_row(meta.get("companies_detected") or [])
                + " · **Intent:** "
                + _intent_badge_md(str(meta.get("intent", "")))
            )
            st.markdown(meta_line, unsafe_allow_html=True)

            with st.chat_message("assistant"):
                _acc: list[str] = []

                def _tracked_stream():
                    for _chunk in answer_gen:
                        if _chunk:
                            _acc.append(_chunk)
                        yield _chunk

                streamed = st.write_stream(_tracked_stream())
                full_answer = streamed if isinstance(streamed, str) and streamed else "".join(
                    _acc
                )

            with st.expander("🔍 Agent details", expanded=False):
                intent = str(meta.get("intent", ""))
                st.markdown(
                    f"**Intent:** {_intent_badge_md(intent)}",
                    unsafe_allow_html=True,
                )
                companies = meta.get("companies_detected") or []
                st.markdown(
                    "**Companies:** " + _company_pills_row(companies),
                    unsafe_allow_html=True,
                )
                st.markdown(_confidence_row(meta.get("confidence")))
                retried = meta.get("retried", False)
                st.write(f"**Retrieval retried:** {'yes' if retried else 'no'}")
                st.write("**Sub-queries:**")
                for i, sq in enumerate(meta.get("sub_queries", []), start=1):
                    st.write(f"{i}. {sq}")

            st.caption(
                "Pages referenced: "
                + (
                    ", ".join(str(p) for p in (meta.get("pages_referenced") or []))
                    or "None"
                )
            )

            st.session_state["chat_history"].append(
                {
                    "question": user_question,
                    "answer": full_answer or "",
                    "intent": meta.get("intent"),
                    "companies_detected": meta.get("companies_detected", []),
                    "sub_queries": meta.get("sub_queries", []),
                    "confidence": meta.get("confidence"),
                    "pages": meta.get("pages_referenced", []),
                    "retried": meta.get("retried", False),
                }
            )
            st.session_state["conversation_history"].append(
                {"role": "user", "content": user_question}
            )
            st.session_state["conversation_history"].append(
                {"role": "assistant", "content": full_answer or ""}
            )
            st.session_state["conversation_history"] = st.session_state[
                "conversation_history"
            ][-3:]
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ----- Tab 2: IPO Brief -----
with tab_brief:
    pick = st.selectbox(
        "Select company",
        options=COMPANY_NAMES,
        format_func=lambda x: f"{x.title()} ({COMPANY_MAP[x]['ticker']})",
        key="brief_company",
    )
    if st.button("Generate IPO Brief", type="primary", key="gen_brief"):
        try:
            with st.spinner(f"Analyzing {pick.title()} S-1..."):
                st.session_state[f"brief_{pick}"] = run_brief_pipeline(pick)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    brief_key = f"brief_{pick}"
    brief_result = st.session_state.get(brief_key)
    if brief_result:
        brief = brief_result.get("brief", {}) or {}
        critique = brief_result.get("critique", {}) or {}
        fin_row = FINANCIALS.get(pick, {}) if isinstance(FINANCIALS, dict) else {}

        st.subheader("Key Financials")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Revenue", str(fin_row.get("revenue", "N/A")))
        with c2:
            st.metric("Gross profit", str(fin_row.get("gross_profit", "N/A")))
        with c3:
            st.metric("Net income", str(fin_row.get("net_income", "N/A")))
        with c4:
            st.metric("Operating expenses", str(fin_row.get("operating_expenses", "N/A")))
        st.caption(_financial_source_caption(pick))

        st.subheader("Business Overview")
        st.info(brief.get("business_overview") or "Not available.")

        st.subheader("Key Highlights")
        highlights = brief.get("key_highlights") or []
        hc1, hc2, hc3 = st.columns(3)
        for col, hl in zip((hc1, hc2, hc3), (highlights + ["", "", ""])[:3]):
            with col:
                st.markdown(
                    f"<div class='hl-box'>{html.escape(str(hl or '—'))}</div>",
                    unsafe_allow_html=True,
                )

        st.subheader("Top Risks by Category")
        for risk in brief.get("risks") or []:
            if not isinstance(risk, dict):
                continue
            cat = str(risk.get("category", "market")).lower()
            color, _label = RISK_CATEGORY_STYLE.get(cat, ("#6b7280", cat.title()))
            verified = bool(risk.get("verified"))
            vtext = "✅ Verified" if verified else "❌ Corrected"
            title = html.escape(str(risk.get("title", "Risk")))
            desc = html.escape(str(risk.get("description", "")))
            st.markdown(
                f"<div class='risk-card'>"
                f"<span style='background:{color};color:white;padding:0.15rem 0.5rem;"
                f"border-radius:999px;font-size:0.72rem;font-weight:600;'>{html.escape(cat.title())}</span>"
                f"<p style='margin:0.5rem 0 0.25rem;font-weight:700;font-size:1.05rem;'>{title}</p>"
                f"<p style='margin:0;color:#d1d5db;font-weight:400;'>{desc}</p>"
                f"<p style='margin:0.45rem 0 0;font-size:0.88rem;'>{vtext}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if critique.get("passed"):
            st.success(critique.get("verdict", "Brief passed critique."))
        else:
            st.error(critique.get("verdict", "Brief failed critique."))
            unsupported = critique.get("unsupported_claims") or []
            if unsupported:
                st.write("Unsupported claims:")
                for claim in unsupported:
                    st.write(f"- {claim}")

        st.caption(
            "Pages referenced: "
            + ", ".join(str(p) for p in brief_result.get("pages_referenced", []) or [])
            or "None"
        )

