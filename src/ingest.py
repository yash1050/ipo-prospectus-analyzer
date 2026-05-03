import os
from pathlib import Path
from typing import List

import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PDF_PATH = PROJECT_ROOT / "data" / "s1a_instacart.pdf"
COLLECTION_NAME = "instacart_s1a_lc"
EMBEDDING_MODEL = "text-embedding-3-small"
MIN_PAGE_CHARS = 100


def require_env_var(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def extract_pdf_pages_to_documents(pdf_path: Path) -> List[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    doc = fitz.open(pdf_path)
    extracted_documents: List[Document] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if len(text) < MIN_PAGE_CHARS:
                continue
            extracted_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": i + 1,
                        "source": str(pdf_path),
                    },
                )
            )
    finally:
        doc.close()

    return extracted_documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)


def upload_documents_to_qdrant(documents: List[Document], qdrant_url: str, qdrant_api_key: str) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=COLLECTION_NAME,
    )


def verify_retrieval(vector_store: QdrantVectorStore) -> None:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    query = "What is Instacart's primary revenue stream?"
    results = retriever.invoke(query)

    print("\nTop 3 retrieved chunks:")
    for index, doc in enumerate(results, start=1):
        page_number = doc.metadata.get("page_number", "N/A")
        preview = doc.page_content[:200].replace("\n", " ").strip()
        print(f"{index}. Page {page_number}: {preview}")


def main() -> None:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = require_env_var("OPENAI_API_KEY")
    qdrant_url = require_env_var("QDRANT_URL")
    qdrant_api_key = require_env_var("QDRANT_API_KEY")

    print("Extracting PDF text...")
    documents = extract_pdf_pages_to_documents(PDF_PATH)
    print(f"Extracted usable pages: {len(documents)}")

    print("Chunking text...")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created. Check PDF extraction and filtering thresholds.")

    print("Uploading chunks to Qdrant...")
    vector_store = upload_documents_to_qdrant(chunks, qdrant_url, qdrant_api_key)

    print("Running retrieval verification...")
    verify_retrieval(vector_store)


if __name__ == "__main__":
    main()
