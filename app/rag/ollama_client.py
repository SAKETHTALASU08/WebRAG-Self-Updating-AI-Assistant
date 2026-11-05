import json
import logging
from typing import Generator, List, Optional
import requests

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM service with streaming and model auto-discovery."""

    def __init__(self, host: str = settings.OLLAMA_HOST, default_model: str = settings.OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.default_model = default_model

    def check_health(self) -> bool:
        """Check if Ollama server is alive."""
        try:
            resp = requests.get(f"{self.host}/api/version", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all downloaded models in local Ollama."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models from {self.host}: {e}")
        return []

    def resolve_model(self, requested_model: Optional[str] = None) -> str:
        """Resolve model name, falling back to an available model if preferred model is missing."""
        if requested_model:
            return requested_model

        available = self.list_models()
        if not available:
            return self.default_model

        # Check exact or prefix match for default model
        for m in available:
            if m.startswith(self.default_model):
                return m

        # Fallbacks: llama3.2, llama3, or first available model
        for pref in ["llama3.2", "llama3", "mistral", "phi3"]:
            for m in available:
                if m.startswith(pref):
                    logger.info(f"Target model '{self.default_model}' not found, falling back to '{m}'")
                    return m

        fallback = available[0]
        logger.info(f"Using available model: {fallback}")
        return fallback

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Non-streaming response generation."""
        target_model = self.resolve_model(model)
        url = f"{self.host}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API returned HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            return data.get("response", "")
        except requests.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.host}. Is Ollama running?")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Streaming response generation yielding tokens as they arrive."""
        target_model = self.resolve_model(model)
        url = f"{self.host}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "system": system or "",
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=90) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama API returned HTTP {resp.status_code}: {resp.text}")
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except requests.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.host}. Is Ollama running?")
