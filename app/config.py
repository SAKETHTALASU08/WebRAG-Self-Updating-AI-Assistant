import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str = "WebRAG"
    DATA_DIR: Path = Path("data")
    
    # Embedding Configuration (Local sentence-transformers)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Ollama Configuration (Local LLM)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    
    # Chunker Configuration
    CHUNK_SIZE: int = 500  # Target words per chunk
    CHUNK_OVERLAP: int = 60  # Overlap words
    
    # Crawler Configuration
    MAX_CRAWL_DEPTH: int = 3
    MAX_CRAWL_PAGES: int = 50
    CRAWL_DELAY_SECONDS: float = 0.2
    REQUEST_TIMEOUT: int = 15
    USER_AGENT: str = "WebRAG-Crawler/1.0 (+https://github.com/webrag)"
    
    # RAG Retrieval Configuration
    SIMILARITY_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.2
    
    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()

# Ensure base data directory exists
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
