import os
import json
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


load_dotenv()

COLLECTION_NAME = "instacart_s1a_lc"
SYSTEM_PROMPT = (
    "You are a query expansion agent specialized in SEC filings and IPO prospectuses. "
    "Given a user question, rewrite it into exactly 3 specific sub-queries that would help "
    "retrieve the most relevant sections from an S-1 filing. Return only a valid JSON array "
    "of exactly 3 strings. No explanation, no markdown, just the JSON array."
)

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

VECTOR_STORE = QdrantVectorStore.from_existing_collection(
    embedding=EMBEDDINGS,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=COLLECTION_NAME,
)


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


def _retrieve_raw_results(sub_queries: list[str], k: int) -> list[tuple[Any, float]]:
    all_results: list[tuple[Any, float]] = []
    for sub_query in sub_queries:
        query_results = VECTOR_STORE.similarity_search_with_relevance_scores(sub_query, k=k)
        all_results.extend(query_results)
    return all_results


def retrieve_context(sub_queries: list[str], k: int = 5) -> list[dict]:
    raw_results = _retrieve_raw_results(sub_queries=sub_queries, k=k)
    deduped: dict[str, dict] = {}

    for doc, score in raw_results:
        text = doc.page_content.strip()
        dedup_key = text[:100]
        page_number = doc.metadata.get("page_number", "N/A")

        candidate = {
            "text": text,
            "page_number": page_number,
            "score": float(score),
        }
        existing = deduped.get(dedup_key)
        if existing is None or candidate["score"] > existing["score"]:
            deduped[dedup_key] = candidate

    sorted_chunks = sorted(deduped.values(), key=lambda item: item["score"], reverse=True)
    return sorted_chunks


def format_context(chunks: list[dict]) -> str:
    formatted_chunks = [
        f"[Page {chunk['page_number']}]: {chunk['text']}"
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


def retrieve_with_confidence(question: str, k: int = 5) -> dict:
    sub_queries = rewrite_query(question)
    chunks = retrieve_context(sub_queries=sub_queries, k=k)
    context = format_context(chunks)
    confidence_result = check_confidence(question=question, context=context)

    confidence = int(confidence_result.get("confidence", 10))
    reason = str(confidence_result.get("reason", "evaluation failed, proceeding"))
    retried = False

    if confidence < 7:
        retried = True
        refined_question = f"{question} Please provide specific details and data."
        sub_queries = rewrite_query(refined_question)
        chunks = retrieve_context(sub_queries=sub_queries, k=k)
        context = format_context(chunks)

    return {
        "sub_queries": sub_queries,
        "chunks": chunks,
        "context": context,
        "confidence": confidence,
        "reason": reason,
        "retried": retried,
    }


def generate_answer(question: str, context: str, chunks: list[dict]) -> dict:
    system_prompt = (
        "You are a financial analyst assistant specialized in IPO prospectuses and SEC S-1 filings. "
        "Answer the user's question accurately and concisely using only the provided context from the "
        "S-1 filing. If the context does not contain enough information to answer, say so clearly. "
        "Do not make up information. Cite page numbers when possible using the format (Page X)."
    )
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
    for chunk in chunks:
        page_number = chunk.get("page_number")
        if page_number not in pages_referenced:
            pages_referenced.append(page_number)

    return {
        "answer": answer,
        "question": question,
        "pages_referenced": pages_referenced,
    }


def run_qa_pipeline(question: str) -> dict:
    sub_queries = rewrite_query(question)
    retrieval_result = retrieve_with_confidence(question)
    answer_result = generate_answer(
        question=question,
        context=retrieval_result["context"],
        chunks=retrieval_result["chunks"],
    )
    return {
        "question": question,
        "sub_queries": sub_queries,
        "confidence": retrieval_result["confidence"],
        "retried": retrieval_result["retried"],
        "answer": answer_result["answer"],
        "pages_referenced": answer_result["pages_referenced"],
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


def run_brief_pipeline() -> dict:
    comprehensive_question = (
        "Give a comprehensive overview of Instacart's business model, revenue streams, "
        "financial performance, and key risk factors"
    )
    _ = run_qa_pipeline(comprehensive_question)
    retrieval_result = retrieve_with_confidence(comprehensive_question)
    context = retrieval_result["context"]

    brief = generate_ipo_brief(context)

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
    result = run_brief_pipeline()
    brief = result.get("brief", {})
    critique = result.get("critique", {})
    financials = brief.get("financials", {}) if isinstance(brief, dict) else {}
    risks = brief.get("risks", []) if isinstance(brief, dict) else []

    print("Company name:")
    print(brief.get("company_name"))

    print("\nBusiness overview:")
    print(brief.get("business_overview"))

    print("\nFinancials:")
    print(f"Revenue: {financials.get('revenue')}")
    print(f"Gross margin: {financials.get('gross_margin')}")
    print(f"EBITDA: {financials.get('ebitda')}")
    print(f"IPO valuation: {financials.get('ipo_valuation')}")

    print("\nRisks:")
    for index, risk in enumerate(risks, start=1):
        print(
            f"{index}. {risk.get('title')} | Category: {risk.get('category')} | "
            f"Verified: {risk.get('verified')}"
        )
        print(f"   {risk.get('description')}")

    print("\nKey highlights:")
    for index, highlight in enumerate(brief.get("key_highlights", []), start=1):
        print(f"{index}. {highlight}")

    print("\nCritic verdict and pass status:")
    print(f"Passed: {critique.get('passed')}")
    print(f"Verdict: {critique.get('verdict')}")

    print("\nPages referenced:")
    print(result.get("pages_referenced"))
