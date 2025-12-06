#!/usr/bin/env python3
"""
Invoice OCR Inference Server

FastAPI server that handles model loading and inference requests from the Rust CLI.
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Invoice OCR Inference Server")

# Global model state
model = None
processor = None
model_name: Optional[str] = None
device: Optional[str] = None


class InferenceRequest(BaseModel):
    image_path: str
    prompt: str


class InferenceResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


def load_model(name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Load the multimodal model."""
    global model, processor, model_name, device

    if model is not None:
        logger.info(f"Model already loaded: {model_name}")
        return

    logger.info(f"Loading model: {name}")

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Determine device
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        logger.warning("CUDA not available, using CPU (will be slow)")

    processor = AutoProcessor.from_pretrained(name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=device,
    )

    model_name = name
    logger.info(f"Model loaded on {device}")


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        # Find the end of the code block
        lines = text.split("\n")
        # Skip first line (```json or ```)
        start_idx = 1
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    text = text.strip()

    # Try to parse
    return json.loads(text)


def extract_from_image(image_path: str, prompt: str) -> dict:
    """Run extraction on an image."""
    global model, processor

    if model is None:
        load_model()

    from PIL import Image

    # Load image
    image = Image.open(image_path)

    # Construct messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1,
        do_sample=False,
    )

    # Decode response
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]

    logger.debug(f"Raw model response: {response[:500]}...")

    # Parse JSON
    return parse_json_response(response)


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Check server status and model state."""
    return StatusResponse(
        status="ok",
        model_loaded=model is not None,
        model_name=model_name,
        device=device,
    )


@app.post("/extract", response_model=InferenceResponse)
async def extract(request: InferenceRequest):
    """Extract invoice data from an image."""
    try:
        # Validate image path
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

        # Run extraction
        data = extract_from_image(str(image_path), request.prompt)

        return InferenceResponse(success=True, data=data)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return InferenceResponse(
            success=False,
            error=f"Failed to parse model output as JSON: {e}",
        )
    except Exception as e:
        logger.exception("Extraction failed")
        return InferenceResponse(success=False, error=str(e))


@app.post("/load")
async def load(model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Explicitly load a model."""
    try:
        load_model(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Invoice OCR Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model to use",
    )

    args = parser.parse_args()

    if args.preload:
        load_model(args.model)

    uvicorn.run(app, host=args.host, port=args.port)
