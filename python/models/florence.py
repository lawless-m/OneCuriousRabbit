"""
Florence-2 model handler.
"""

import gc
import logging
from typing import Optional

import torch
from PIL import Image

from .base import ModelHandler

logger = logging.getLogger(__name__)


class Florence2Handler(ModelHandler):
    """Handler for Microsoft Florence-2-large model."""

    name = "microsoft/Florence-2-large"
    requires_vram_gb = 3.0

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    def load(self, device: str = "auto") -> None:
        if self._model is not None:
            logger.info(f"Model already loaded: {self.name}")
            return

        logger.info(f"Loading model: {self.name}")
        from transformers import AutoProcessor, AutoModelForCausalLM

        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda:0"
                dtype = torch.float16
            else:
                self._device = "cpu"
                dtype = torch.float32
                logger.warning("CUDA not available, using CPU")
        else:
            self._device = device
            dtype = torch.float16 if "cuda" in device else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)

        logger.info(f"Model loaded on {self._device}")

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
        """
        Florence-2 uses task-specific prompts. For general VQA/captioning,
        we use the MORE_DETAILED_CAPTION or OCR tasks and include the user prompt.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Florence-2 works best with specific task prompts
        # For invoice extraction, we use OCR + detailed caption
        task_prompt = "<MORE_DETAILED_CAPTION>"

        inputs = self._processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse Florence output format
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        # Florence returns dict with task key, extract the caption
        if task_prompt in parsed:
            caption = parsed[task_prompt]
        else:
            caption = str(parsed)

        # Now do OCR to get text content
        ocr_prompt = "<OCR>"
        ocr_inputs = self._processor(
            text=ocr_prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        ocr_ids = self._model.generate(
            input_ids=ocr_inputs["input_ids"],
            pixel_values=ocr_inputs["pixel_values"],
            max_new_tokens=2048,
            num_beams=3,
            do_sample=False,
        )

        ocr_text = self._processor.batch_decode(
            ocr_ids, skip_special_tokens=False
        )[0]

        ocr_parsed = self._processor.post_process_generation(
            ocr_text,
            task=ocr_prompt,
            image_size=(image.width, image.height),
        )

        if ocr_prompt in ocr_parsed:
            ocr_content = ocr_parsed[ocr_prompt]
        else:
            ocr_content = str(ocr_parsed)

        # Combine caption and OCR for context
        # Return as a structured response that the server can parse
        combined = f"""Image Description: {caption}

OCR Text Content:
{ocr_content}

Please extract the invoice data based on the above information.
Format as JSON with header (supplier_name, invoice_number, invoice_date, po_reference, currency, net_total, vat_amount, gross_total, payment_due_date) and line_items array.
"""
        return combined

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_info(self) -> Optional[str]:
        return self._device
