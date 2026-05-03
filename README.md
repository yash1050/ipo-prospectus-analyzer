# IPO Prospectus Analyzer

A Python project that ingests an SEC **S-1/A** prospectus (PDF), indexes it in **Qdrant** with **OpenAI** embeddings, and exposes a **LangChain**-based multi-agent pipeline plus a **Streamlit** UI for question answering and structured **IPO brief** generation. The default document target is **Instacart (Maplebear Inc.)**, S-1/A dated **September 15, 2023**.

## Features

- **PDF ingestion**: Page-by-page text extraction with **PyMuPDF** (more reliable for dense SEC PDFs than generic loaders), chunking with LangChain’s `RecursiveCharacterTextSplitter`, embedding with `text-embedding-3-small`, and upload to Qdrant.
- **RAG Q&A**: Query rewriting (3 sub-queries), retrieval with deduplication and scoring, optional one-shot retrieval refinement when confidence is low, and grounded answers with page references.
- **IPO brief**: Generates a JSON-structured brief from retrieved context, verifies risk categories, and runs a fact-check critique against the same context.
- **Web app**: Streamlit interface with **Q&A** chat and **IPO Brief** tab (metrics, overview, highlights, risks, critique, pages).

## Repository layout

```
IPO-Prospectus-Analyzer/
├── .env                 # Not committed — API keys and Qdrant URL (see below)
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── ingest.py        # One-shot: PDF → chunks → Qdrant + smoke retrieval
    ├── app.py           # Streamlit entrypoint
    └── agents/
        ├── __init__.py
        └── pipeline.py  # All LangChain agents and orchestration
```

The `data/` directory is **gitignored**. Place your prospectus PDF there as:

`data/s1a_instacart.pdf`

(Paths are defined in `src/ingest.py`.)

## Prerequisites

- **Python 3.10+** (project tested with 3.13 in development)
- **OpenAI** API access (chat + embeddings)
- **Qdrant** instance (cloud or self-hosted) with HTTP URL and API key

## Environment variables

Create a `.env` file in the project root:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (chat + embeddings) |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |

`python-dotenv` loads these for `ingest.py`, `pipeline.py`, and `app.py`.

## Installation

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ingestion (index the PDF into Qdrant)

1. Add `data/s1a_instacart.pdf` (or adjust `PDF_PATH` in `src/ingest.py` if you use another filename).
2. Ensure `.env` contains `OPENAI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.
3. Run:

```powershell
python src/ingest.py
```

**Behavior summary**

- Skips pages with fewer than **100** characters.
- Each kept page becomes a LangChain `Document` with metadata: `page_number`, `source`.
- Chunks: **500** characters, **50** overlap, `length_function=len`.
- Collection name: **`instacart_s1a_lc`** (created/used by `QdrantVectorStore.from_documents`).
- Prints total chunks before upload and runs a short retrieval test after upload.

The agent pipeline (`src/agents/pipeline.py`) connects to the **same** collection name for retrieval.

## Streamlit app

Run from the **project root** so `src` resolves as a package for imports:

```powershell
streamlit run src/app.py
```

- **Q&A tab**: Chat history, `run_qa_pipeline()` per message, expander for sub-queries and confidence, captions for pages.
- **IPO Brief tab**: **Generate IPO Brief** runs `run_brief_pipeline()`. Summary metrics for Instacart use hardcoded display values in `app.py`; narrative content comes from the LLM and retrieved context.

Errors from the pipelines are caught and shown with `st.error(...)`.

## Agent pipeline (`src/agents/pipeline.py`)

Module-level singletons: `ChatOpenAI` (`gpt-4o-mini`), `OpenAIEmbeddings` (`text-embedding-3-small`), and `QdrantVectorStore` for collection `instacart_s1a_lc`.

| Step | Function | Role |
|------|----------|------|
| Query expansion | `rewrite_query` | 3 S-1-oriented sub-queries (JSON), fallback to original question |
| Retrieval | `retrieve_context` | Per sub-query similarity search, dedupe by first 100 chars, sort by score |
| Context string | `format_context` | `[Page N]: text` blocks joined by separators |
| Confidence | `check_confidence` | JSON score 1–10 + reason; safe fallback |
| Retrieval loop | `retrieve_with_confidence` | Optional single retry with refined question if confidence &lt; 7 |
| Answer | `generate_answer` | Grounded answer from context + `pages_referenced` from chunks |
| End-to-end Q&A | `run_qa_pipeline` | Rewrite → retrieve with confidence → generate answer |
| Brief | `generate_ipo_brief` | Structured JSON brief from context |
| Risks | `classify_and_verify_risks` | Single-call verification/correction of risk categories |
| Critique | `critique_brief` | Fact-check brief vs context |
| End-to-end brief | `run_brief_pipeline` | QA-style retrieval for overview question → brief → verify risks → critique |

## Dependencies (`requirements.txt`)

- **LangChain**: `langchain`, `langchain-community`, `langchain-openai`, `langchain-qdrant`, `langchain-text-splitters`
- **Vector DB**: `qdrant-client`
- **PDF**: `pymupdf`
- **App / config**: `streamlit`, `python-dotenv`

## Security notes

- Never commit `.env` or prospectus PDFs if they are confidential or subject to redistribution limits.
- `data/` is ignored by git by design.

## License

Specify a license in this repository if you plan to distribute the code publicly.
