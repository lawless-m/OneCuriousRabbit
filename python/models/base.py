"""
Base model handler interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class ModelHandler(ABC):
    """Abstract base class for model handlers."""

    name: str = ""
    requires_vram_gb: float = 0

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load the model and processor."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def extract(self, image: Image.Image, prompt: str) -> str:
        """Run extraction and return raw text response."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @property
    @abstractmethod
    def device_info(self) -> Optional[str]:
        """Get device the model is loaded on."""
        pass
