import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webrag")

app = FastAPI(
    title=settings.APP_NAME,
    description="WebRAG: Self-updating AI assistant generated from a website's root URL.",
    version="1.0.0",
)

# Enable CORS for local cross-origin usage if accessed from different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount Static UI directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    """Serve the WebRAG single page UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "WebRAG API is running. UI index.html not yet built."}


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Data directory: {settings.DATA_DIR.resolve()}")
    logger.info(f"Local Ollama host: {settings.OLLAMA_HOST} (model: {settings.OLLAMA_MODEL})")
