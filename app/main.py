from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated startup/shutdown events."""
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Data directory: {settings.DATA_DIR.resolve()}")
    logger.info(f"Local Ollama host: {settings.OLLAMA_HOST} (model: {settings.OLLAMA_MODEL})")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


tags_metadata = [
    {
        "name": "Sites",
        "description": "Create, list, inspect, and delete isolated per-website knowledge bases.",
    },
    {
        "name": "Crawl & Refresh",
        "description": "Real-time crawl status monitoring and incremental updates using ETag/Last-Modified and chunk diffing.",
    },
    {
        "name": "Chat & Query",
        "description": "Grounded question answering with local Ollama, numbered context citations [1][2], and SSE token streaming.",
    },
    {
        "name": "Inspection",
        "description": "Inspect stored chunk hashes and chunk text for auditability.",
    },
    {
        "name": "System",
        "description": "System health, connectivity checks, and device acceleration status.",
    },
]

app = FastAPI(
    title="WebRAG API",
    description=(
        "**WebRAG**: An autonomous, self-updating local AI assistant generated directly from any website's root URL.\n\n"
        "### Key Principles:\n"
        "- **100% Local**: Everything runs locally via `sentence-transformers`, `FAISS`, `SQLite`, and `Ollama (Llama 3.1)`.\n"
        "- **Isolated Knowledge Bases**: Every website gets its own directory, SQLite database, and FAISS `IndexIDMap`.\n"
        "- **Incremental Updates**: SHA-256 chunk hashing and HTTP conditional requests (`ETag` / `Last-Modified`) ensure zero wasted embeddings.\n"
        "- **Strict Grounding**: System prompt strictly mandates answers from context blocks and inline citations `[1]`, `[2]`."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for local cross-origin usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Structured Error Handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Input validation failed. Please check the request fields.",
            "details": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
        },
    )


# Dual route mount: supports both /sites and /api/sites seamlessly
app.include_router(api_router, prefix="/api")
app.include_router(api_router)

# Mount Static UI directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    """Serve the WebRAG single page UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "WebRAG API is running. UI index.html not yet built."}
