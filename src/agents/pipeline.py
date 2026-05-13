import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from financials import COMPANIES


load_dotenv()

COMPANY_REGISTRY: dict[str, dict[str, Any]] = {c["name"]: c for c in COMPANIES}
ALL_COMPANY_NAMES: list[str] = list(COMPANY_REGISTRY.keys())

SYSTEM_PROMPT = (
    "You are a query expansion agent specialized in SEC filings and IPO prospectuses. "
    "Given a user question, rewrite it into exactly 3 specific sub-queries that would help "
    "retrieve the most relevant sections from an S-1 filing. Return only a valid JSON array "
    "of exactly 3 strings. No explanation, no markdown, just the JSON array."
)

DETECT_INTENT_SYSTEM = (
    "You are a query intent classifier for an IPO prospectus analysis system. "
    "Available companies are: instacart (CART), reddit (RDDT), klaviyo (KVYO), arm (ARM), "
    "coinbase (COIN), doordash (DASH), snowflake (SNOW).\n"
    "Rules:\n\n"
    "If the question explicitly names specific companies → intent is single (one company) or "
    "comparative (multiple), return ONLY those named companies\n"
    "If the question asks 'which company' or 'best/worst/highest/lowest across companies' with no "
    "specific names → intent is open_ended, return ALL company names\n"
    "If the question contains a claim to verify → intent is verification, return the relevant company\n"
    "NEVER return all companies when specific company names are mentioned in the question\n"
    "Return only valid JSON with keys: intent, companies, reasoning"
)

LLM = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Dedicated streaming client for Streamlit UI (does not affect synchronous agents).
LLM_STREAM = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

_vectorstore_cache: dict[str, QdrantVectorStore] = {}


def get_vectorstore(company_name: str) -> QdrantVectorStore:
    key = company_name.strip().lower()
    if key not in COMPANY_REGISTRY:
        raise ValueError(
            f"Unknown company {company_name!r}. Expected one of: {ALL_COMPANY_NAMES}"
        )
    if key in _vectorstore_cache:
        return _vectorstore_cache[key]
    collection_name = f"ipo_{key}"
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError(
            f"Cannot connect to Qdrant collection {collection_name!r}: QDRANT_URL is not set."
        )
    try:
        store = QdrantVectorStore.from_existing_collection(
            embedding=EMBEDDINGS,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open Qdrant collection {collection_name!r} for company {key!r}: {exc}"
        ) from exc
    _vectorstore_cache[key] = store
    return store


def _normalize_company_list(names: Any) -> list[str]:
    if not isinstance(names, list):
        return []
    out: list[str] = []
    for item in names:
        if not isinstance(item, str):
            continue
        raw = item.strip().lower()
        if raw in COMPANY_REGISTRY:
            out.append(raw)
            continue
        for reg_name in ALL_COMPANY_NAMES:
            if raw == reg_name.lower():
                out.append(reg_name)
                break
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def detect_companies_and_intent(
    question: str, conversation_history: list | None = None
) -> dict[str, Any]:
    history = conversation_history or []
    tail = history[-3:] if history else []
    history_lines: list[str] = []
    for i, msg in enumerate(tail):
        if isinstance(msg, dict):
            role = str(msg.get("role", "message"))
            content = str(msg.get("content", ""))
            history_lines.append(f"{i + 1}. [{role}] {content}")
        else:
            history_lines.append(f"{i + 1}. {msg}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DETECT_INTENT_SYSTEM),
                (
                    "human",
                    "Question:\n{question}\n\nRecent conversation (last up to 3 messages):\n"
                    "{history_block}",
                ),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"question": question, "history_block": history_block})
        if not isinstance(result, dict):
            raise ValueError("expected JSON object")
        intent = str(result.get("intent", "open_ended")).strip().lower()
        if intent not in ("single", "comparative", "open_ended", "verification"):
            intent = "open_ended"
        companies_raw = result.get("companies")
        companies = _normalize_company_list(companies_raw)
        if intent == "open_ended" or not companies:
            companies = list(ALL_COMPANY_NAMES)
        reasoning = str(result.get("reasoning", "")).strip() or "classified"
        return {"intent": intent, "companies": companies, "reasoning": reasoning}
    except Exception:
        return {
            "intent": "open_ended",
            "companies": list(ALL_COMPANY_NAMES),
            "reasoning": "fallback",
        }


def rewrite_query(user_question: str) -> list[str]:
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{user_question}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"user_question": user_question})
        if (
            isinstance(result, list)
            and len(result) == 3
            and all(isinstance(item, str) for item in result)
        ):
            return result
        return [user_question]
    except Exception:
        return [user_question]


def _retrieve_raw_results(
    sub_queries: list[str], companies: list[str], k: int
) -> list[tuple[Document, float, str, str]]:
    all_results: list[tuple[Document, float, str, str]] = []
    for company in companies:
        store = get_vectorstore(company)
        reg = COMPANY_REGISTRY[company]
        ticker = str(reg.get("ticker", ""))
        for sub_query in sub_queries:
            query_results = store.similarity_search_with_relevance_scores(sub_query, k=k)
            for doc, score in query_results:
                all_results.append((doc, float(score), company, ticker))
    return all_results


def retrieve_context(
    sub_queries: list[str], companies: list[str], k: int = 5
) -> list[dict]:
    if not companies:
        companies = list(ALL_COMPANY_NAMES)
    raw_results = _retrieve_raw_results(sub_queries=sub_queries, companies=companies, k=k)
    deduped: dict[str, dict] = {}

    for doc, score, company, ticker in raw_results:
        text = doc.page_content.strip()
        dedup_key = text[:100]
        page_number = doc.metadata.get("page_number", "N/A")

        candidate = {
            "text": text,
            "page_number": page_number,
            "score": float(score),
            "company": company,
            "ticker": ticker,
        }
        existing = deduped.get(dedup_key)
        if existing is None or candidate["score"] > existing["score"]:
            deduped[dedup_key] = candidate

    sorted_chunks = sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
    return sorted_chunks


def format_context(chunks: list[dict]) -> str:
    formatted_chunks = [
        f"[{str(chunk['company']).upper()} - Page {chunk['page_number']}]: {chunk['text']}"
        for chunk in chunks
    ]
    return "\n\n---\n\n".join(formatted_chunks)


def check_confidence(question: str, context: str) -> dict:
    system_prompt = (
        "You are a confidence evaluation agent. Given a user question and a retrieved context "
        "from an SEC S-1 filing, evaluate whether the context contains enough information to "
        "answer the question accurately. Return only a valid JSON object with two keys: "
        "confidence (integer 1-10) and reason (one sentence explanation). No markdown, no extra text."
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"question": question, "context": context})
        confidence = int(result.get("confidence"))
        reason = str(result.get("reason", "")).strip()
        if not reason:
            reason = "no reason provided"
        return {"confidence": confidence, "reason": reason}
    except Exception:
        return {"confidence": 10, "reason": "evaluation failed, proceeding"}


def retrieve_with_confidence(
    question: str, companies: list[str], k: int = 5
) -> dict:
    sub_queries = rewrite_query(question)
    chunks = retrieve_context(sub_queries=sub_queries, companies=companies, k=k)
    context = format_context(chunks)
    confidence_result = check_confidence(question=question, context=context)

    confidence = int(confidence_result.get("confidence", 10))
    reason = str(confidence_result.get("reason", "evaluation failed, proceeding"))
    retried = False

    if confidence < 7:
        retried = True
        refined_question = f"{question} Please provide specific details and data."
        sub_queries = rewrite_query(refined_question)
        chunks = retrieve_context(sub_queries=sub_queries, companies=companies, k=k)
        context = format_context(chunks)

    return {
        "sub_queries": sub_queries,
        "chunks": chunks,
        "context": context,
        "confidence": confidence,
        "reason": reason,
        "retried": retried,
    }


_ANSWER_SOURCES_RULE = (
    "Never repeat the sources or references section. Include sources only once at the very end."
)

_COMPARATIVE_FINANCIAL_CAVEAT = (
    "Note that revenue figures come from each company's S-1 filing which covers different time "
    "periods — always mention this caveat when comparing financial figures across companies."
)


def _system_prompt_for_intent(intent: str) -> str:
    if intent == "single":
        return (
            "You are a financial analyst. Answer the question using only the provided context. "
            "Cite page numbers as (Page X). "
            + _ANSWER_SOURCES_RULE
        )
    if intent in ("comparative", "open_ended"):
        return (
            "You are a financial analyst comparing multiple IPO companies. Answer the question by "
            "clearly attributing information to each company. Use format: CompanyName: followed "
            "by the relevant information. Cite page numbers. "
            + _COMPARATIVE_FINANCIAL_CAVEAT
            + " "
            + _ANSWER_SOURCES_RULE
        )
    if intent == "verification":
        return (
            "You are a fact-checker. Given the question which contains a claim, verify whether the "
            "claim is supported, contradicted, or not mentioned in the context. Return a clear verdict. "
            + _ANSWER_SOURCES_RULE
        )
    return (
        "You are a financial analyst. Answer the question using only the provided context. "
        "Cite page numbers as (Page X). "
        + _ANSWER_SOURCES_RULE
    )


def generate_answer(
    question: str, context: str, chunks: list[dict], intent: str
) -> dict:
    system_prompt = _system_prompt_for_intent(intent)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | LLM | parser
    answer = chain.invoke({"question": question, "context": context})

    pages_referenced: list[Any] = []
    companies_referenced: list[str] = []
    for chunk in chunks:
        page_number = chunk.get("page_number")
        if page_number not in pages_referenced:
            pages_referenced.append(page_number)
        co = chunk.get("company")
        if isinstance(co, str) and co and co not in companies_referenced:
            companies_referenced.append(co)

    return {
        "answer": answer,
        "question": question,
        "pages_referenced": pages_referenced,
        "companies_referenced": companies_referenced,
    }


def stream_answer(
    question: str, context: str, chunks: list[dict], intent: str
) -> Iterator[str]:
    """Token stream for the answer generator (Streamlit only). Yields string chunks."""
    _ = chunks  # same signature as generate_answer; context already encodes retrieval
    system_prompt = _system_prompt_for_intent(intent)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | LLM_STREAM | parser
    for chunk in chain.stream({"question": question, "context": context}):
        if isinstance(chunk, str) and chunk:
            yield chunk


def run_qa_pipeline_streaming(
    question: str, conversation_history: list | None = None
) -> tuple[dict[str, Any], Iterator[str]]:
    """
    Run detection, rewrite, and retrieval; return metadata plus a streaming generator for the answer.
    """
    detection = detect_companies_and_intent(question, conversation_history)
    intent = detection["intent"]
    companies = detection["companies"]
    sub_queries = rewrite_query(question)
    retrieval_result = retrieve_with_confidence(question, companies)

    pages_referenced: list[Any] = []
    for chunk in retrieval_result["chunks"]:
        page_number = chunk.get("page_number")
        if page_number not in pages_referenced:
            pages_referenced.append(page_number)

    metadata: dict[str, Any] = {
        "intent": intent,
        "companies_detected": companies,
        "sub_queries": sub_queries,
        "confidence": retrieval_result["confidence"],
        "retried": retrieval_result["retried"],
        "pages_referenced": pages_referenced,
        "chunks": retrieval_result["chunks"],
    }
    generator = stream_answer(
        question=question,
        context=retrieval_result["context"],
        chunks=retrieval_result["chunks"],
        intent=intent,
    )
    return metadata, generator


def run_qa_pipeline(question: str, conversation_history: list | None = None) -> dict:
    detection = detect_companies_and_intent(question, conversation_history)
    intent = detection["intent"]
    companies = detection["companies"]
    sub_queries = rewrite_query(question)
    retrieval_result = retrieve_with_confidence(question, companies)
    answer_result = generate_answer(
        question=question,
        context=retrieval_result["context"],
        chunks=retrieval_result["chunks"],
        intent=intent,
    )
    return {
        "question": question,
        "sub_queries": sub_queries,
        "confidence": retrieval_result["confidence"],
        "retried": retrieval_result["retried"],
        "answer": answer_result["answer"],
        "pages_referenced": answer_result["pages_referenced"],
        "intent": intent,
        "companies_detected": companies,
        "companies_referenced": answer_result["companies_referenced"],
    }


def generate_ipo_brief(context: str) -> dict:
    system_prompt = (
        "You are a senior financial analyst specialized in IPO prospectuses. Given context extracted "
        "from an S-1 filing, generate a structured IPO brief. Return only a valid JSON object with "
        "exactly these keys:\n\n"
        "company_name: string\n"
        "business_overview: 2-3 sentence summary of what the company does\n"
        "financials: dict with keys revenue, gross_margin, ebitda, ipo_valuation — use actual figures "
        "from context, use null if not found\n"
        "risks: list of exactly 4 dicts, each with keys title (short), description (one sentence), "
        "category (one of: market, regulatory, competition, financial)\n"
        "key_highlights: list of 3 strings, each a notable business highlight\n"
        "No markdown, no extra text, only valid JSON."
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Context:\n{context}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"context": context})
        return dict(result)
    except Exception:
        return {
            "company_name": None,
            "business_overview": None,
            "financials": {
                "revenue": None,
                "gross_margin": None,
                "ebitda": None,
                "ipo_valuation": None,
            },
            "risks": None,
            "key_highlights": None,
        }


def classify_and_verify_risks(risks: list[dict]) -> list[dict]:
    system_prompt = (
        "You are a risk classification verification agent for SEC S-1 filings. Given a list of risk "
        "factors each with a title, description and assigned category, verify each classification. "
        "Categories must be one of: market, regulatory, competition, financial. Return only a valid "
        "JSON array where each item has keys: title, description, category (corrected if needed), "
        "verified (boolean), reason (one sentence explaining verification decision). No markdown, "
        "no extra text.\n\n"
        "Few-shot examples:\n"
        "Title: Gig worker reclassification | Description: Changes in labor classification rules can "
        "increase fulfillment costs. | Category: regulatory -> Verified as regulatory.\n"
        "Title: Amazon and Walmart compete on delivery | Description: Large competitors can pressure "
        "pricing and market share. | Category: competition -> Verified as competition.\n"
        "Title: Revenue depends on transaction volume | Description: Lower order frequency reduces "
        "top-line revenue growth. | Category: financial -> Verified as financial.\n"
        "Title: Post-pandemic demand slowdown | Description: Normalizing consumer behavior can reduce "
        "online grocery demand. | Category: market -> Verified as market."
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Risks JSON:\n{risks_json}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"risks_json": json.dumps(risks)})
        if isinstance(result, list):
            return result
        return risks
    except Exception:
        return risks


def critique_brief(brief: dict, context: str) -> dict:
    system_prompt = (
        "You are a fact-checking agent for IPO briefs. Given an IPO brief and the source context "
        "from the S-1 filing, check whether each claim in the brief is supported by the context. "
        "Return only a valid JSON object with keys:\n\n"
        "passed (boolean — true if all major claims are supported)\n"
        "unsupported_claims (list of strings — claims not found in context, empty list if all passed)\n"
        "verdict (one sentence summary)"
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Brief JSON:\n{brief_json}\n\nContext:\n{context}"),
            ]
        )
        parser = JsonOutputParser()
        chain = prompt | LLM | parser
        result = chain.invoke({"brief_json": json.dumps(brief), "context": context})
        return dict(result)
    except Exception:
        return {
            "passed": False,
            "unsupported_claims": ["Critique generation failed."],
            "verdict": "Unable to verify claims against context.",
        }


def _load_financials_json_for_company(company_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "financials.json"
    key = company_name.strip().lower()
    na_block = {
        "revenue": "N/A",
        "gross_margin": "N/A",
        "ebitda": "N/A",
        "ipo_valuation": "N/A",
    }
    if not path.is_file():
        return na_block
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return na_block
    row = data.get(key) if isinstance(data, dict) else None
    if not isinstance(row, dict):
        row = {}
    revenue = row.get("revenue") or "N/A"
    gross = row.get("gross_profit") or "N/A"
    return {
        "revenue": revenue,
        "gross_margin": gross,
        "ebitda": "N/A",
        "ipo_valuation": "N/A",
    }


def run_brief_pipeline(company_name: str) -> dict:
    key = company_name.strip().lower()
    if key not in COMPANY_REGISTRY:
        raise ValueError(
            f"Unknown company {company_name!r}. Expected one of: {ALL_COMPANY_NAMES}"
        )
    verified_fin = _load_financials_json_for_company(key)

    display_name = key.title()
    comprehensive_question = (
        f"Give a comprehensive overview of {display_name}'s business model, revenue streams, "
        "financial performance, and key risk factors"
    )
    companies = [key]
    retrieval_result = retrieve_with_confidence(comprehensive_question, companies=companies)
    context = retrieval_result["context"]

    brief = generate_ipo_brief(context)
    brief["company_name"] = display_name
    brief["financials"] = {
        "revenue": verified_fin["revenue"],
        "gross_margin": verified_fin["gross_margin"],
        "ebitda": verified_fin["ebitda"],
        "ipo_valuation": verified_fin["ipo_valuation"],
    }

    risks = brief.get("risks")
    if isinstance(risks, list):
        brief["risks"] = classify_and_verify_risks(risks)

    critique = critique_brief(brief, context)

    return {
        "brief": brief,
        "critique": critique,
        "confidence": retrieval_result["confidence"],
        "pages_referenced": sorted(
            {
                chunk.get("page_number")
                for chunk in retrieval_result["chunks"]
                if chunk.get("page_number") is not None
            }
        ),
    }


if __name__ == "__main__":
    # Tests 2 (comparative) and 3 (open_ended) only
    flat_history: list[dict[str, str]] = []

    tests = [
        ("comparative", "Compare Reddit and Coinbase's regulatory risks"),
        ("open_ended", "Which company has the highest revenue?"),
    ]

    for _label, q in tests:
        print("=" * 60)
        print(f"Q: {q}")
        out = run_qa_pipeline(q, conversation_history=flat_history)
        print("Detected intent:", out.get("intent"))
        print("Companies detected:", out.get("companies_detected"))
        print("Answer:\n", out.get("answer"))
