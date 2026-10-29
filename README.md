# Local RAG Web Application

A fully local, privacy-first Retrieval-Augmented Generation (RAG) web application with multi-format document ingestion, SQLite persistence, Okapi BM25 and Hybrid Dense Vector (RRF) retrieval, page-level grounded citations, and a responsive web UI.

---

## Architecture Overview

- **Core Engine (`ragcore.py`)**:
  - Ingestion & extraction: `.txt`, `.docx`, `.xlsx`, `.pdf` (via PyMuPDF).
  - Robust exception handling for corrupt files, encrypted PDFs, and scanned PDFs lacking a text layer (`IngestError`).
  - Sliding-window text chunking with exact page number preservation.
  - Okapi BM25 lexical search with content-token filtering.
  - Dense vector similarity and Reciprocal Rank Fusion (`RRF`).
  - Strict model tracking per chunk to prevent mixing embeddings from different models.
- **Persistence (`app/database.py`)**:
  - SQLite storage with foreign keys, indexing, and WAL mode.
  - Tables: `knowledge_bases`, `documents`, `chunks`, `conversations`, `messages`.
  - Filename sanitization against path traversal; SHA-256 deduplication.
  - Transactional status commit: documents transition to `indexed` only after chunks are committed.
- **API (`app/main.py`)**:
  - FastAPI endpoints for KB management, multi-file uploads, document management, chat, and citation inspection.
  - Cross-KB isolation and boundary enforcement.
- **LLM Integrations (`app/llm.py`)**:
  - `extractive`: Grounded direct extraction from top retrieved chunks (100% offline, zero external dependency).
  - `ollama`: Local LLMs (e.g. `llama3.2`) with retries and clean 502/503 HTTP status mappings on connection errors.
  - `openai_compatible`: Local vLLM, LM Studio, or OpenAI endpoints.
- **Web UI (`app/static/index.html`)**:
  - Plain modern HTML5/CSS/JavaScript (no build step or Node.js required).
  - Multi-file drag-and-drop upload, real-time status badges, chunk counts.
  - Interactive chat with clickable citations opening a side drawer with source excerpt and metadata.
  - Collapsible retrieval debug panel displaying BM25/RRF scores and latency breakdown.
  - Light and Dark theme toggle.

---

## Quickstart & Setup

### 1. macOS & Linux

```bash
# Navigate to the project directory
cd /Users/bhavishyatyagi/local-rag-app

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Windows (PowerShell)

```powershell
cd local-rag-app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the Application

### Start the Server

```bash
# Using the virtual environment python
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://127.0.0.1:8000`** to access the Web UI.
Access interactive Swagger API documentation at **`http://127.0.0.1:8000/docs`**.

---

## Running the Tests

To run the complete test suite (27 tests covering extraction, PDF edge cases, persistence, hybrid retrieval, and API endpoints):

```bash
.venv/bin/python run_tests.py
```

Or using pytest directly:

```bash
.venv/bin/pytest -v tests/
```

---

## Running the Retrieval Evaluation Benchmark

To compute **Recall@k**, **MRR (Mean Reciprocal Rank)**, **Citation Validity**, and **No-Answer Precision** across configurations:

```bash
.venv/bin/python -m eval.run
```

---

## Ollama Setup (Optional for Generative Mode)

By default, the application runs in **Extractive Mode** with zero external dependencies. To enable local generative LLMs via Ollama:

1. Install Ollama:
   - **macOS / Linux**: Download from [ollama.com](https://ollama.com) or run `brew install ollama` (macOS).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull the desired model:
   ```bash
   ollama pull llama3.2
   ```
4. (Optional) Pull embedding model for vector search:
   ```bash
   ollama pull nomic-embed-text
   ```
5. Set environment variables in `.env` (or pass to terminal):
   ```bash
   export LLM_PROVIDER=ollama
   export LLM_MODEL=llama3.2
   export LLM_BASE_URL=http://localhost:11434
   ```

---

## Configuration (`.env`)

Refer to `.env.example` for all configurable environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `extractive` | Provider: `extractive`, `ollama`, or `openai_compatible` |
| `LLM_MODEL` | `llama3.2` | Model identifier |
| `LLM_BASE_URL` | `http://localhost:11434` | Endpoint for Ollama or OpenAI-compatible server |
| `LLM_API_KEY` | *(empty)* | Optional API key for OpenAI-compatible endpoints |
| `EMBEDDING_PROVIDER` | `none` | Dense embedding provider (`none` or `ollama`) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model name |
| `MAX_UPLOAD_SIZE_MB` | `25` | Maximum upload size allowed per file |
| `CORS_ORIGINS` | `*` | Allowed CORS origins for external API clients |
| `DATABASE_PATH` | `data/rag.db` | SQLite database file location |
| `UPLOADS_DIR` | `data/uploads` | Directory for uploaded source files |

---

## Implementation Status

### Fully Implemented and Verified
- [x] Multi-format document parser (.txt, .docx, .xlsx, .pdf)
- [x] PyMuPDF-based PDF extraction with exact page-level mapping
- [x] Corrupt, encrypted, and scanned PDF error detection (`IngestError`)
- [x] Sliding-window text chunking preserving page numbers and token counts
- [x] SQLite schema with WAL mode, indexing, and cascade delete
- [x] Path-traversal sanitization and SHA-256 deduplication
- [x] Transactional status indexing (`status = 'indexed'` only after chunk commit)
- [x] Full REST API (FastAPI) covering KBs, documents, search, chat, and chunk viewer
- [x] Cross-KB isolation and boundary enforcement
- [x] Extractive QA mode (100% offline, grounded answers, no hallucinations)
- [x] External LLM client with retries, timeouts, and clean 502/503 errors
- [x] Hybrid BM25 + Dense vector retrieval with Reciprocal Rank Fusion (RRF)
- [x] Dynamic model tracking per chunk with automatic re-indexing upon model change
- [x] Responsive Web UI with drag-and-drop upload, chat, citations drawer, retrieval debug panel, theme toggle
- [x] Benchmark evaluation harness with 22 questions reporting Recall@k, MRR, citation validity

### Not Implemented / Out of Scope
- [ ] Optical Character Recognition (OCR) for image-only scanned PDFs (requires external Tesseract/OCR engine).
- [ ] Cloud object storage drivers (e.g. S3/GCS); uploads are safely stored in local `./data/uploads`.
- [ ] Audio/video transcription (Whisper).
