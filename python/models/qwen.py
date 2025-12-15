"""
Qwen2-VL model handlers.
"""

import gc
import logging
from typing import Optional

import torch
from PIL import Image

from .base import ModelHandler

logger = logging.getLogger(__name__)


def get_optimal_dtype() -> torch.dtype:
    """Get the optimal dtype for the current hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class Qwen2VL7BHandler(ModelHandler):
    """Handler for Qwen2-VL-7B-Instruct model."""

    name = "Qwen/Qwen2-VL-7B-Instruct"
    requires_vram_gb = 16.0

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    def load(self, device: str = "auto", max_retries: int = 3, retry_delay: int = 30) -> None:
        if self._model is not None:
            logger.info(f"Model already loaded: {self.name}")
            return

        logger.info(f"Loading model: {self.name}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        dtype = get_optimal_dtype()

        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"
                logger.warning("CUDA not available, using CPU (will be slow)")
        else:
            self._device = device

        # Retry on OOM - other services may be using GPU
        for attempt in range(max_retries):
            try:
                self._processor = AutoProcessor.from_pretrained(self.name)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.name,
                    torch_dtype=dtype,
                    device_map=self._device,
                )
                logger.info(f"Model loaded on {self._device}")
                return
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"OOM error on attempt {attempt + 1}/{max_retries}")
                        logger.info(f"GPU busy - waiting {retry_delay}s for other services to release memory...")
                        # Clean up any partial allocation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import time
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to load model after {max_retries} attempts")
                        raise
                else:
                    # Not an OOM error - raise immediately
                    raise

    def unload(self) -> None:
        if self._model is not None:
            logger.info(f"Unloading model: {self.name}")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def extract(self, image: Image.Image, prompt: str) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
        )

        response = self._processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> Optional[str]:
        return self._device


class Qwen2VL2BHandler(ModelHandler):
    """Handler for Qwen2-VL-2B-Instruct model (lighter version)."""

    name = "Qwen/Qwen2-VL-2B-Instruct"
    requires_vram_gb = 6.0

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    def load(self, device: str = "auto", max_retries: int = 3, retry_delay: int = 30) -> None:
        if self._model is not None:
            logger.info(f"Model already loaded: {self.name}")
            return

        logger.info(f"Loading model: {self.name}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        dtype = get_optimal_dtype()

        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"
                logger.warning("CUDA not available, using CPU (will be slow)")
        else:
            self._device = device

        # Retry on OOM - other services may be using GPU
        for attempt in range(max_retries):
            try:
                self._processor = AutoProcessor.from_pretrained(self.name)
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.name,
                    torch_dtype=dtype,
                    device_map=self._device,
                )
                logger.info(f"Model loaded on {self._device}")
                return
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"OOM error on attempt {attempt + 1}/{max_retries}")
                        logger.info(f"GPU busy - waiting {retry_delay}s for other services to release memory...")
                        # Clean up any partial allocation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import time
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to load model after {max_retries} attempts")
                        raise
                else:
                    # Not an OOM error - raise immediately
                    raise

    def unload(self) -> None:
        if self._model is not None:
            logger.info(f"Unloading model: {self.name}")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def extract(self, image: Image.Image, prompt: str) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
        )

        response = self._processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

        return response

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> Optional[str]:
        return self._device
