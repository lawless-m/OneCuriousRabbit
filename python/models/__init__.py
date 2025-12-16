"""
Model handlers for Invoice OCR.

Provides a registry of available models and their handlers.
"""

from typing import Dict, Type, List
from .base import ModelHandler
from .qwen import Qwen2VL7BHandler, Qwen2VL2BHandler
from .florence import Florence2Handler
from .ollama import OllamaLlavaHandler


# Registry of available model handlers
MODEL_REGISTRY: Dict[str, Type[ModelHandler]] = {
    "Qwen/Qwen2-VL-7B-Instruct": Qwen2VL7BHandler,
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VL2BHandler,
    "microsoft/Florence-2-large": Florence2Handler,
    "llava:13b": OllamaLlavaHandler,
    # Aliases for convenience
    "qwen-7b": Qwen2VL7BHandler,
    "qwen-2b": Qwen2VL2BHandler,
    "florence-2": Florence2Handler,
    "llava": OllamaLlavaHandler,
}

# Default model
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def get_handler(model_name: str) -> ModelHandler:
    """Get a model handler instance by name."""
    if model_name not in MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}"
        )
    return MODEL_REGISTRY[model_name]()


def list_available_models() -> List[dict]:
    """List all available models with their info."""
    seen = set()
    models = []
    for name, handler_cls in MODEL_REGISTRY.items():
        # Skip aliases (they have the same handler class)
        if handler_cls.name in seen:
            continue
        seen.add(handler_cls.name)
        models.append({
            "name": handler_cls.name,
            "requires_vram_gb": handler_cls.requires_vram_gb,
        })
    return models


__all__ = [
    "ModelHandler",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "get_handler",
    "list_available_models",
    "Qwen2VL7BHandler",
    "Qwen2VL2BHandler",
    "Florence2Handler",
    "OllamaLlavaHandler",
]
