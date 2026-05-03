import os

import streamlit as st
from dotenv import load_dotenv

from agents.pipeline import run_brief_pipeline, run_qa_pipeline


load_dotenv()

INSTACART_FINANCIALS = {
    "revenue": "$2.9B (FY2022)",
    "gross_margin": "29%",
    "ebitda": "$74M (Adjusted)",
    "ipo_valuation": "$9.3B",
}


def format_pages_caption(pages: list) -> str:
    if not pages:
        return "Pages referenced: None"
    pages_text = ", ".join(str(page) for page in pages)
    return f"Pages referenced: {pages_text}"


def risk_category_badge(category: str) -> str:
    category_colors = {
        "market": "#1e40af",
        "regulatory": "#c2410c",
        "competition": "#7e22ce",
        "financial": "#b91c1c",
    }
    color = category_colors.get(str(category).lower(), "#374151")
    label = str(category).title() if category else "Unknown"
    return (
        f"<span style='background-color:{color}; color:white; "
        "padding:0.2rem 0.55rem; border-radius:999px; font-size:0.78rem;'>"
        f"{label}</span>"
    )


st.set_page_config(page_title="IPO Prospectus Analyzer", layout="wide")

st.title("IPO Prospectus Analyzer")
st.subheader("Instacart (Maplebear Inc.) · S-1/A · Sep 15, 2023")
st.success("Document loaded ✅")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "brief_result" not in st.session_state:
    st.session_state["brief_result"] = None

tab_qa, tab_brief = st.tabs(["Q&A", "IPO Brief"])

with tab_qa:
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry.get("question", ""))

        with st.chat_message("assistant"):
            st.write(entry.get("answer", ""))
            with st.expander("🔍 Agent details", expanded=False):
                st.write("Sub-queries used:")
                for index, sub_query in enumerate(entry.get("sub_queries", []), start=1):
                    st.write(f"{index}. {sub_query}")
                st.write(f"Confidence score: {entry.get('confidence')}")
            st.caption(format_pages_caption(entry.get("pages", [])))

    user_question = st.chat_input("Ask a question about the S-1...")
    if user_question:
        try:
            with st.spinner("Running pipeline..."):
                result = run_qa_pipeline(user_question)

            st.session_state["chat_history"].append(
                {
                    "question": user_question,
                    "answer": result.get("answer", ""),
                    "sub_queries": result.get("sub_queries", []),
                    "confidence": result.get("confidence"),
                    "pages": result.get("pages_referenced", []),
                }
            )
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

with tab_brief:
    if st.button("Generate IPO Brief"):
        try:
            with st.spinner("Generating brief — this takes ~30 seconds..."):
                st.session_state["brief_result"] = run_brief_pipeline()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    brief_result = st.session_state.get("brief_result")
    if brief_result:
        brief = brief_result.get("brief", {})
        critique = brief_result.get("critique", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Revenue", INSTACART_FINANCIALS["revenue"])
        col2.metric("Gross Margin", INSTACART_FINANCIALS["gross_margin"])
        col3.metric("EBITDA", INSTACART_FINANCIALS["ebitda"])
        col4.metric("IPO Valuation", INSTACART_FINANCIALS["ipo_valuation"])

        st.subheader("Business Overview")
        st.write(brief.get("business_overview", "Not available."))

        st.subheader("Key Highlights")
        for highlight in brief.get("key_highlights", []) or []:
            st.info(highlight)

        st.subheader("Top Risks by Category")
        for risk in brief.get("risks", []) or []:
            category = risk.get("category", "unknown")
            verified = bool(risk.get("verified"))
            with st.container(border=True):
                st.markdown(risk_category_badge(category), unsafe_allow_html=True)
                st.markdown(f"**{risk.get('title', 'Untitled risk')}**")
                st.write(risk.get("description", "No description provided."))
                st.write(f"Verified: {'✅' if verified else '❌'}")

        if critique.get("passed"):
            st.success(critique.get("verdict", "Brief passed critique."))
        else:
            st.error(critique.get("verdict", "Brief failed critique."))
            unsupported_claims = critique.get("unsupported_claims", [])
            if unsupported_claims:
                st.write("Unsupported claims:")
                for claim in unsupported_claims:
                    st.write(f"- {claim}")

        st.caption(format_pages_caption(brief_result.get("pages_referenced", [])))
