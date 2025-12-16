"""
Ollama model handler for vision models like llava.
"""

import base64
import io
import logging
from typing import Optional

import requests
from PIL import Image

from .base import ModelHandler

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaLlavaHandler(ModelHandler):
    """Handler for Ollama llava models."""

    name = "llava:13b"
    requires_vram_gb = 8.0  # Ollama manages this

    def __init__(self, model_name: str = "llava:13b"):
        self._model_name = model_name
        self._loaded = False

    def load(self, device: str = "auto") -> None:
        """Check Ollama is running and model is available."""
        if self._loaded:
            logger.info(f"Model already loaded: {self._model_name}")
            return

        logger.info(f"Checking Ollama model: {self._model_name}")

        # Check Ollama is running
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if self._model_name not in model_names:
                # Try without tag
                base_name = self._model_name.split(":")[0]
                matching = [n for n in model_names if n.startswith(base_name)]
                if not matching:
                    raise RuntimeError(
                        f"Model {self._model_name} not found in Ollama. "
                        f"Available: {model_names}. "
                        f"Pull with: ollama pull {self._model_name}"
                    )
                logger.info(f"Found matching model: {matching[0]}")

            self._loaded = True
            logger.info(f"Ollama model ready: {self._model_name}")

        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Start it with: ollama serve"
            )

    def unload(self) -> None:
        """Ollama manages its own memory, just mark as unloaded."""
        if self._loaded:
            logger.info(f"Marking Ollama model as unloaded: {self._model_name}")
            self._loaded = False

    def extract(self, image: Image.Image, prompt: str) -> str:
        """Run extraction via Ollama API."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Convert image to base64
        buffer = io.BytesIO()
        # Convert to RGB if necessary (handles RGBA, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Call Ollama generate API with image
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            }
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=180,  # Vision inference can be slow
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        except requests.Timeout:
            raise RuntimeError("Ollama request timed out (>180s)")
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e.response.text}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> Optional[str]:
        return "Ollama" if self._loaded else None
