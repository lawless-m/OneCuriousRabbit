#!/usr/bin/env python3
"""
Invoice OCR Inference Server

FastAPI server that handles model loading and inference requests from the Rust CLI.
Supports multiple model backends with hot-swapping capability.
"""

import base64
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from models import get_handler, list_available_models, DEFAULT_MODEL, ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Invoice OCR Inference Server")

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
current_handler: Optional[ModelHandler] = None


class InferenceRequest(BaseModel):
    image_path: str
    prompt: str


class Base64Request(BaseModel):
    filename: str
    data: str  # base64 encoded image or PDF
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


class ModelInfo(BaseModel):
    name: str
    requires_vram_gb: float


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    current: Optional[str] = None


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 1
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    text = text.strip()
    return json.loads(text)


def load_model(name: str = DEFAULT_MODEL) -> None:
    """Load a model, unloading any currently loaded model first."""
    global current_handler

    # Check if same model already loaded
    if current_handler is not None and current_handler.name == name:
        logger.info(f"Model already loaded: {name}")
        return

    # Unload current model if different
    if current_handler is not None:
        logger.info(f"Switching models: {current_handler.name} -> {name}")
        current_handler.unload()
        current_handler = None

    # Load new model
    handler = get_handler(name)
    handler.load()
    current_handler = handler


def extract_from_image(image_path: str, prompt: str) -> dict:
    """Run extraction on an image file path."""
    from PIL import Image
    image = Image.open(image_path)
    return extract_from_pil_image(image, prompt)


def extract_from_pil_image(image, prompt: str) -> dict:
    """Run extraction on a PIL Image."""
    global current_handler

    if current_handler is None:
        load_model()

    # Get raw response from model
    raw_response = current_handler.extract(image, prompt)
    logger.debug(f"Raw model response: {raw_response[:500]}...")

    # Parse JSON
    return parse_json_response(raw_response)


def decode_base64_to_image(data: str, filename: str):
    """Decode base64 data to PIL Image, handling PDFs if needed."""
    from PIL import Image
    import subprocess

    # Decode base64
    image_bytes = base64.b64decode(data)

    # Check if it's a PDF
    if filename.lower().endswith('.pdf') or image_bytes[:4] == b'%PDF':
        # Convert PDF to image using pdftoppm
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(image_bytes)
            pdf_path = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Convert first page to PNG
                result = subprocess.run(
                    ['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', '150', pdf_path, f'{tmpdir}/page'],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise ValueError(f"PDF conversion failed: {result.stderr}")

                # Find the output file
                import glob
                png_files = glob.glob(f'{tmpdir}/page*.png')
                if not png_files:
                    raise ValueError("PDF conversion produced no output")

                return Image.open(png_files[0]).copy()
        finally:
            Path(pdf_path).unlink(missing_ok=True)
    else:
        # Regular image
        return Image.open(io.BytesIO(image_bytes))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Check server status and model state."""
    return StatusResponse(
        status="ok",
        model_loaded=current_handler is not None and current_handler.is_loaded,
        model_name=current_handler.name if current_handler else None,
        device=current_handler.device_info if current_handler else None,
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """List available models."""
    models = [
        ModelInfo(name=m["name"], requires_vram_gb=m["requires_vram_gb"])
        for m in list_available_models()
    ]
    return ModelsResponse(
        models=models,
        current=current_handler.name if current_handler else None,
    )


@app.post("/extract", response_model=InferenceResponse)
async def extract(request: InferenceRequest):
    """Extract invoice data from an image."""
    try:
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

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


@app.post("/extract-base64", response_model=InferenceResponse)
async def extract_base64(request: Base64Request):
    """Extract invoice data from base64-encoded image or PDF (for web UI)."""
    try:
        # Decode and convert to image
        image = decode_base64_to_image(request.data, request.filename)

        # Run extraction
        data = extract_from_pil_image(image, request.prompt)
        return InferenceResponse(success=True, data=data)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return InferenceResponse(
            success=False,
            error=f"Failed to parse model output as JSON: {e}",
        )
    except Exception as e:
        logger.exception("Base64 extraction failed")
        return InferenceResponse(success=False, error=str(e))


@app.post("/load")
async def load(model_name: str = DEFAULT_MODEL):
    """Load a specific model (unloads current model first)."""
    try:
        load_model(model_name)
        return {
            "status": "ok",
            "model": current_handler.name,
            "device": current_handler.device_info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload():
    """Unload the current model to free memory."""
    global current_handler

    if current_handler is None:
        return {"status": "ok", "message": "No model loaded"}

    model_name = current_handler.name
    current_handler.unload()
    current_handler = None

    return {"status": "ok", "message": f"Unloaded {model_name}"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Invoice OCR Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model to use (default: Qwen/Qwen2-VL-7B-Instruct)",
    )

    args = parser.parse_args()

    if args.preload:
        load_model(args.model)

    uvicorn.run(app, host=args.host, port=args.port)
