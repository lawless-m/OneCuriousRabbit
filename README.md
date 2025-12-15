# Invoice OCR

Extract structured data from PDF invoices using multimodal LLMs (Qwen2-VL) running locally on GPU.

## Features

- PDF to image conversion with configurable DPI
- Multimodal LLM inference via Python server
- Single and multi-page invoice support
- JSON, CSV, and Excel output formats
- Schema validation with confidence scoring
- Batch processing with summary reports
- Automatic archiving of processed files
- Retry logic for failed extractions

## System Requirements

### Hardware
- NVIDIA GPU with 16GB+ VRAM (RTX 3090 recommended)
- 64GB+ system RAM
- Linux (tested on Debian)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install poppler-utils
```

**Fedora:**
```bash
sudo dnf install poppler-utils
```

**Arch Linux:**
```bash
sudo pacman -S poppler
```

### Python Dependencies

```bash
cd python
pip install -r requirements.txt
```

Requires PyTorch with CUDA support and Hugging Face Transformers.

## Installation

### Build from Source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/lawless-m/OneCuriousRabbit.git
cd OneCuriousRabbit
cargo build --release

# Binary will be at target/release/invoice-ocr
```

## Configuration

Initialize configuration and directories:

```bash
./target/release/invoice-ocr init
```

This creates `config/default.toml`:

```toml
[model]
name = "Qwen/Qwen2-VL-7B-Instruct"
backend = "transformers"
device = "cuda:0"

[processing]
input_dir = "./input"
output_dir = "./output"
archive_dir = "./archive"
image_dpi = 150

[output]
format = "json"
confidence_threshold = 0.7

[inference]
server_url = "http://10.99.0.3:8765"
```

## Usage

### 1. Start the Inference Server

```bash
# Start server (model loads on first request)
python python/inference_server.py

# Or preload the model
python python/inference_server.py --preload
```

### 2. Process Invoices

```bash
# Process single invoice
invoice-ocr process invoice.pdf

# Process all PDFs in input directory
invoice-ocr process --all

# Output as CSV
invoice-ocr process --format csv invoice.pdf

# Output both JSON and Excel
invoice-ocr process --format both --all

# Skip archiving
invoice-ocr process --no-archive invoice.pdf

# Custom retry count
invoice-ocr process --retries 5 invoice.pdf
```

### 3. Check Status

```bash
invoice-ocr status
```

## Output Format

JSON output includes:

```json
{
  "extraction_version": "1.0",
  "source_file": "invoice.pdf",
  "confidence": {
    "overall": 0.92,
    "header": 0.95,
    "line_items": 0.88
  },
  "header": {
    "supplier_name": "Acme Ltd",
    "invoice_number": "INV-001",
    "invoice_date": "2024-01-15",
    "currency": "GBP",
    "net_total": 1000.00,
    "vat_amount": 200.00,
    "gross_total": 1200.00
  },
  "line_items": [
    {
      "line_number": 1,
      "description": "Widget",
      "quantity": 10,
      "unit_price": 100.00,
      "line_total": 1000.00
    }
  ],
  "warnings": []
}
```

## Model Options

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen/Qwen2-VL-7B-Instruct | ~16GB | Recommended |
| Qwen/Qwen2-VL-2B-Instruct | ~6GB | Faster, less accurate |
| microsoft/Florence-2-large | ~3GB | Lightweight |

Configure in `config/default.toml` or Python server args.

## Troubleshooting

### "Cannot connect to inference server"
Start the Python server: `python python/inference_server.py`

### "pdftoppm not found"
Install poppler-utils (see System Dependencies above)

### Low confidence scores
- Check image quality (try higher DPI in config)
- Review prompt in `prompts/invoice_extract.txt`
- Some invoice formats may need prompt adjustments

### CUDA out of memory
- Use smaller model (Qwen2-VL-2B)
- Enable int8 quantization in Python server

## Web Interface

A drag-and-drop web interface is available at `web/index.html` with features:
- PDF and image upload
- Real-time extraction results
- Excel download (Header + Line Items sheets)
- Automatic status monitoring

### Deployment via Nginx

The inference server runs on port 8765. Configure nginx to serve the frontend and proxy API requests:

```bash
# Inference server runs on its own port
python python/inference_server.py  # Runs on 10.99.0.3:8765
```

See `nginx.conf.example` for reverse proxy configuration.

## GPU Memory Management

This service implements two-layer GPU coordination:

### 1. OOM Retry (Always Active)
- Automatically retries 3 times with 30s delays on CUDA OOM errors
- Works with any service that auto-unloads (like Ollama with OLLAMA_KEEP_ALIVE=30s)
- No coordination required

### 2. Service Signaling (Optional)
The inference server implements endpoints for active coordination:

- **Auto-unload on idle**: `--auto-unload-minutes 5` unloads model after 5 minutes of inactivity
- **Request-unload endpoint**: `POST http://10.99.0.3:8765/request-unload` - other services can request unload if idle
- **Enhanced status**: `/status` shows idle time and auto-unload configuration

**Before loading a large model in another service:**
```bash
# Request Invoice OCR to unload if idle
python request_gpu_unload.py

# Or directly via curl
curl -X POST http://10.99.0.3:8765/request-unload
```

This gives faster, more predictable GPU coordination with OOM retry as fallback.

See `/home/matt/Git/claude-skills/.claude/skills/Vram-GPU-OOM-memory-management/SKILL.md` for implementing this pattern in other GPU services.

## License

MIT
