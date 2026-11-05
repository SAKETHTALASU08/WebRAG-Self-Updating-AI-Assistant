# WebRAG 🕸️🤖

A self-updating local AI assistant generated from a website's root URL.

WebRAG crawls a website starting from its root URL, extracts text from HTML pages and linked PDFs, splits content into overlapping chunks with SHA-256 hashing for incremental change detection, builds an isolated local FAISS vector index & SQLite database for each website, and answers user queries via local Ollama (Llama 3.1) with strict URL and section citations.

Everything runs **100% locally with zero external APIs**.

---

## 🛠️ Stack

- **Language & Runtime**: Python 3.12
- **Crawling & Ingestion**: Requests, BeautifulSoup4, PyMuPDF (fitz)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
- **Vector Search**: FAISS (`IndexFlatIP` with L2-normalized cosine similarity)
- **Metadata & State Storage**: SQLite (dedicated database per website)
- **Local LLM**: Ollama (`llama3.1` / `llama3.2`)
- **Backend API**: FastAPI + Uvicorn (streaming SSE support)
- **Frontend UI**: Responsive HTML5 / CSS3 / Vanilla JavaScript (zero CDN dependencies)
- **Containerization**: Docker & Docker Compose

---

## 🚀 Key Features & Rules

1. **Strict Knowledge Base Isolation**:
   - Each website receives its own isolated directory under `data/sites/{site_id}/`.
   - Separate SQLite database (`site.db`) and dedicated FAISS index (`faiss.index`).
   - Multiple sites never cross-contaminate or mix vectors.

2. **Incremental Updates (SHA-256 at Chunk Level)**:
   - Chunk-level SHA-256 fingerprinting: `hashlib.sha256(normalized_text)`.
   - On re-crawl, unchanged chunks are detected in `O(1)` from SQLite and skipped (no re-embedding).
   - Modified/new chunks are embedded and indexed.
   - Deleted/obsolete chunks are purged from both SQLite and FAISS.

3. **Domain-Scoped Crawler**:
   - Discovers HTML pages and linked PDFs.
   - Stays strictly within the same domain/host.
   - Respects `robots.txt` directives.
   - Enforces polite rate limiting between requests.
   - Normalizes and deduplicates URLs (handles relative paths, query parameter sorting, trailing slashes, fragments `#...`).
   - Configurable `max_depth` and `max_pages`.
   - Automatic HTTP retries with exponential backoff on transient errors (429, 500, 502, 503, 504) and request timeouts.
   - Captures and stores `ETag` and `Last-Modified` HTTP headers.
   - Directly outputs a list of `(url, content_type, raw_bytes)`.

4. **Accurate Citations**:
   - Every answer cites exact source URLs, chunk indices, and page numbers (for PDFs).
   - Interactive UI with source badges, preview tooltips, and chunk inspector.

---

## 📂 Project Structure

```
WebRag/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic BaseSettings configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic schemas (CrawlRequest, QueryResponse, Citations)
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── crawler.py            # WebCrawler (robots.txt, rate limits, retries, ETag/Last-Modified)
│   │   └── extractor.py          # HTML (BeautifulSoup4) & PDF (PyMuPDF) text extractor
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── site_manager.py       # Isolated per-site filesystem & storage manager
│   │   ├── database.py           # SQLite manager per site
│   │   └── vector_store.py       # FAISS vector store per site
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Word-level chunker with SHA-256 hashing
│   │   ├── embedder.py           # Local SentenceTransformers embedder
│   │   ├── ollama_client.py      # Ollama client with streaming token generator
│   │   └── pipeline.py           # Incremental indexing orchestrator & RAG retriever
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # REST API routes
│   ├── main.py                   # FastAPI application entry point
│   └── static/                   # Self-contained frontend UI
│       ├── index.html
│       ├── style.css
│       └── app.js
├── tests/
│   ├── conftest.py
│   ├── test_crawler_mocked.py    # Mocked crawler unit tests
│   ├── test_crawler_extractor.py # Extractor & normalization tests
│   ├── test_chunker_incremental.py # Incremental chunking & SHA-256 tests
│   ├── test_isolated_storage.py  # Isolated DB & FAISS tests
│   ├── test_rag_pipeline.py      # End-to-end RAG & citation tests
│   └── test_api_endpoints.py     # FastAPI endpoints tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🏃 Getting Started

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.ai/) installed locally:
  ```bash
  ollama pull llama3.1
  # or
  ollama pull llama3.2
  ```

### Local Setup

1. **Create and activate virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Start the WebRAG server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the Web UI**:
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🐳 Docker Setup

Run both WebRAG and Ollama in containers:

```bash
docker-compose up --build
```

Then pull the model into the containerized Ollama instance:
```bash
docker exec -it webrag_ollama ollama pull llama3.1
```

The WebRAG UI will be available at [http://localhost:8000](http://localhost:8000).

---

## 📡 API Endpoints

- `GET /api/health` - Check Ollama connection and available models
- `GET /api/sites` - List all website knowledge bases
- `POST /api/sites` - Register a website root URL and trigger background crawl
- `GET /api/sites/{site_id}` - Retrieve details and stats for a site
- `DELETE /api/sites/{site_id}` - Delete an isolated knowledge base and its vector index
- `POST /api/sites/{site_id}/crawl` - Trigger an incremental crawl / update
- `GET /api/sites/{site_id}/crawl/status` - Poll real-time crawl progress
- `POST /api/sites/{site_id}/query` - Ask a question (returns answer + citations)
- `POST /api/sites/{site_id}/query/stream` - SSE streaming query tokens and citations
- `GET /api/sites/{site_id}/chunks` - Inspect stored chunks and their SHA-256 hashes
