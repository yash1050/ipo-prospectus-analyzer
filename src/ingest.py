"""
Multi-company PDF ingestion: chunk, embed, and upload each prospectus to its own Qdrant collection.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if __package__:
    from .financials import COMPANIES
else:
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from financials import COMPANIES

import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

EMBEDDING_MODEL = "text-embedding-3-small"
MIN_PAGE_CHARS = 100
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DELAY_SECONDS = 2


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def ingest_company(company: dict[str, Any], embeddings: OpenAIEmbeddings) -> int:
    """
    Ingest one company's PDF into Qdrant collection ipo_{name}.
    Returns number of chunks uploaded.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    name = str(company["name"])
    ticker = str(company.get("ticker", ""))
    pdf_rel = str(company["pdf"])
    collection_name = f"ipo_{name}"

    qdrant_url = _require_env("QDRANT_URL")
    qdrant_api_key = _require_env("QDRANT_API_KEY")

    pdf_path = Path(pdf_rel)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name=collection_name)
        print(f"🗑️ Deleted existing collection: {name}")

    documents: list[Document] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text").strip()
            if len(text) < MIN_PAGE_CHARS:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "company": name,
                        "ticker": ticker,
                        "page_number": i + 1,
                        "source": str(pdf_path.resolve()),
                    },
                )
            )
    finally:
        doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    print(f"📄 {name}: {len(chunks)} chunks created → uploading...")

    if not chunks:
        return 0

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )
    return len(chunks)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    _require_env("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    chunks_by_company: dict[str, int] = {}
    failures = 0

    for index, company in enumerate(COMPANIES):
        label = company.get("name", str(company))
        try:
            n = ingest_company(company, embeddings)
            chunks_by_company[label] = n
            print(f"✅ {label}: {n} chunks uploaded to ipo_{label}")
        except Exception as exc:
            failures += 1
            chunks_by_company[label] = 0
            print(f"❌ Failed: {label} — {exc}")

        if index < len(COMPANIES) - 1:
            time.sleep(DELAY_SECONDS)

    successful = sum(1 for c in chunks_by_company.values() if c > 0)
    total_chunks = sum(chunks_by_company.values())

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total companies ingested (with ≥1 chunk): {successful} / {len(COMPANIES)}")
    if failures:
        print(f"Failures: {failures}")
    print("Chunks per company:")
    for co_name, count in chunks_by_company.items():
        print(f"  {co_name}: {count}")
    print(f"Total chunks across all companies: {total_chunks}")


if __name__ == "__main__":
    main()
