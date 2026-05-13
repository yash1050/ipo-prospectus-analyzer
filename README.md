# IPO Prospectus Analyzer

A Python project that ingests **seven** SEC **S-1** prospectus PDFs, indexes each in **Qdrant** under its own collection with **OpenAI** embeddings, and exposes a **LangChain** multi-agent pipeline plus a **Streamlit** UI for multi-company **RAG Q&A** and per-company **IPO brief** generation. Chat uses **GPT-4.1-mini**; embeddings use **text-embedding-3-small**.

Companies covered: **Instacart (CART)**, **Reddit (RDDT)**, **Klaviyo (KVYO)**, **Arm (ARM)**, **Coinbase (COIN)**, **DoorDash (DASH)**, **Snowflake (SNOW)** — paths and CIKs are defined in `src/financials.py`.

## Features

- **Multi-company PDF ingestion**: Page-level text with **PyMuPDF**, chunking with LangChain `RecursiveCharacterTextSplitter`, embeddings, and upload to Qdrant collection **`ipo_{company_name}`** (one collection per issuer).
- **Financials cache**: `src/financials.py` can pull metrics (XBRL via SEC, PDF fallback) and write **`data/financials.json`** for the UI and brief pipeline.
- **RAG Q&A**: Intent and company detection, query rewriting (3 sub-queries), multi-collection retrieval with deduplication, confidence check with optional retry, and grounded answers with company and page attribution. The Streamlit UI uses **`run_qa_pipeline_streaming()`** so metadata appears first and the answer **streams** token-by-token.
- **IPO brief (per company)**: Retrieval from that company’s collection only, LLM-generated brief, verified risk categories, fact-check critique, and **financials overwritten from `data/financials.json`** when available.
- **Web app**: Streamlit with **Q&A** (chat history, agent details expander, page captions) and **IPO Brief** (company picker, metrics from JSON, overview, highlights, risks, critique). **Market Performance** (yfinance/plotly) is not active in the UI right now; dependencies remain in `requirements.txt` for future use.

## Repository layout

```
IPO-Prospectus-Analyzer/
├── .env                 # Not committed — API keys and Qdrant URL (see below)
├── .gitignore
├── README.md
├── requirements.txt
├── scripts/
│   └── check_openai_models.py   # Optional OpenAI model listing helper
└── src/
    ├── ingest.py        # Ingest all COMPANIES PDFs → ipo_{name} collections
    ├── financials.py    # XBRL/PDF financial extraction → data/financials.json
    ├── app.py           # Streamlit entrypoint (run from src/ or see below)
    └── agents/
        ├── __init__.py
        └── pipeline.py  # LangChain agents, Qdrant per company, Q&A + brief
```

The **`data/`** directory is **gitignored**. Place each company’s S-1 PDF under `data/` using the filenames expected in `src/financials.py` (for example `data/instacart s1.pdf`, `data/snowflake s1.pdf`, etc.).

## Prerequisites

- **Python 3.10+** (tested with 3.13 in development)
- **OpenAI** API access (chat + embeddings)
- **Qdrant** instance (cloud or self-hosted) with HTTP URL and API key

## Environment variables

Create a **`.env`** file in the **project root**:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (chat + embeddings) |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |

`python-dotenv` loads these for `ingest.py`, `pipeline.py`, `financials.py`, and `app.py`.

## Installation

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ingestion (index PDFs into Qdrant)

1. Add the PDFs under `data/` as configured in `src/financials.py` (`COMPANIES` list).
2. Ensure `.env` contains `OPENAI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.
3. Run from project root (so imports resolve):

```powershell
cd src
python ingest.py
```

Or:

```powershell
python -m ingest
```

(from `src` as cwd, with `PYTHONPATH` including `src` if needed).

**Behavior summary**

- Skips pages with fewer than **100** characters.
- Metadata on chunks includes **`company`**, **`ticker`**, **`page_number`**, **`source`**.
- Chunks: **500** characters, **50** overlap.
- **Deletes and recreates** each `ipo_{name}` collection before upload for a clean re-index.

The agent pipeline opens collections **on demand** via **`get_vectorstore(company_name)`** (cached), not a single global collection.

## Financials JSON (optional but recommended for the UI)

From the project root (with `src` on the path):

```powershell
cd src
python financials.py
```

Writes **`data/financials.json`** (ignored by git) used by the IPO Brief tab metrics and by **`run_brief_pipeline(company_name)`** to inject verified figures.

## Streamlit app

Run from the **`src`** directory (recommended for `agents` / `financials` imports):

```powershell
cd src
streamlit run app.py
```

From the repo root you can use:

```powershell
streamlit run src/app.py
```

if your environment resolves `src` imports (e.g. `PYTHONPATH` includes `src`).

- **Q&A tab**: Chat input, **`run_qa_pipeline_streaming()`** with **`st.spinner("Retrieving answer...")`** during retrieval, then streamed answer; agent details expander and page captions per turn.
- **IPO Brief tab**: Select company, **Generate IPO Brief** calls **`run_brief_pipeline(company_name)`**; key metrics from **`FINANCIALS`** / `financials.json`; narrative from RAG + LLM.

Errors are wrapped in **`try` / `except`** and shown with **`st.error(...)`** where applicable.

## Agent pipeline (`src/agents/pipeline.py`)

| Step | Function | Role |
|------|----------|------|
| Company / intent | `detect_companies_and_intent` | Classifies intent (single, comparative, open_ended, verification) and company list |
| Query expansion | `rewrite_query` | 3 S-1-oriented sub-queries (JSON), fallback to original question |
| Vector store | `get_vectorstore` | `QdrantVectorStore` for collection `ipo_{company_name}` (module-level cache) |
| Retrieval | `retrieve_context` | Per company × sub-query search; dedupe by first 100 chars; sort by score; chunk fields include `company`, `ticker` |
| Context string | `format_context` | `[COMPANY - Page N]: text` blocks |
| Confidence | `check_confidence` | JSON score 1–10 + reason; safe fallback |
| Retrieval loop | `retrieve_with_confidence` | Optional retry if confidence is below 7 |
| Answer | `generate_answer` | Intent-specific system prompt; `pages_referenced`, `companies_referenced` |
| Answer (stream) | `stream_answer` | Same prompts as `generate_answer`, **`LLM_STREAM`** + `.stream()` for the UI |
| End-to-end Q&A | `run_qa_pipeline` | Detect → rewrite → retrieve → **`generate_answer`** (non-streaming) |
| Streaming Q&A | `run_qa_pipeline_streaming` | Same through retrieval; returns **`(metadata_dict, stream_generator)`** |
| Brief | `generate_ipo_brief` | Structured JSON brief from context |
| Risks | `classify_and_verify_risks` | Verify/correct risk categories |
| Critique | `critique_brief` | Fact-check brief vs context |
| End-to-end brief | `run_brief_pipeline(company_name)` | Single-company retrieval → brief → risks → critique; injects **`financials.json`** financials |

Model: **`gpt-4.1-mini`** for chat agents; embeddings **`text-embedding-3-small`**.

## Dependencies (`requirements.txt`)

- **LangChain**: `langchain`, `langchain-community`, `langchain-openai`, `langchain-qdrant`, `langchain-text-splitters`
- **Vector DB**: `qdrant-client`
- **PDF**: `pymupdf`, `pdfplumber`
- **HTTP**: `requests`
- **App / config**: `streamlit`, `python-dotenv`
- **Optional / future UI**: `yfinance`, `plotly` (installed; market tab not wired in the current `app.py`)

## Security notes

- Never commit `.env` or prospectus PDFs if they are confidential or subject to redistribution limits.
- **`data/`** is ignored by git by design.

## License

Specify a license in this repository if you plan to distribute the code publicly.
