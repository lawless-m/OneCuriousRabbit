# Invoice OCR Extraction System

## Project Overview

A local document processing system that extracts structured data from PDF invoices using multimodal LLMs, specifically Qwen2-VL, running on a 3090 GPU (24GB VRAM). The system replaces manual copy-paste workflows for PO reconciliation.

## Problem Statement

Current PDF-to-text solutions fail to reliably extract invoice data due to complex layouts, tables, and mixed born-digital/scanned documents. Manual copy-paste is tedious and error-prone. A multimodal LLM approach treats each page as an image, sidestepping text extraction ordering issues entirely.

## Hardware Target

- NVIDIA RTX 3090 (24GB VRAM)
- Dual CPU Xeon
- 64GB System RAM
- Debian Linux

## Core Requirements

### Input
- PDF invoices (born-digital and scanned)
- Single and multi-page documents
- Regular suppliers with consistent formats

### Output
- Structured JSON (Power Query compatible)
- Optional CSV/Excel export
- Confidence scores for review flagging

### Data to Extract

**Header Fields:**
- Supplier name
- Invoice number
- Invoice date
- PO reference
- Currency
- Net total
- VAT/tax amount
- Gross total
- Payment terms/due date (if present)

**Line Items (per item):**
- Line number
- Product/item code (if present)
- Description
- Quantity
- Unit of measure
- Unit price
- VAT rate (if per-line)
- Line total

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF In    │────▶│  PDF→Image  │────▶│  Qwen2-VL   │────▶│ JSON Output │
│  (folder)   │     │ (per page)  │     │  Inference  │     │   + CSV     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Config    │
                                        │ (model swap)│
                                        └─────────────┘
```

### Components

1. **PDF Processor** - Converts PDF pages to images (PNG/JPEG)
2. **Model Runner** - Loads and runs the multimodal LLM
3. **Prompt Engine** - Constructs extraction prompts with schema
4. **Output Formatter** - Converts model output to JSON/CSV/Excel
5. **CLI Interface** - User interaction and batch processing
6. **Config System** - Model selection and parameters

## Technology Choices

### Primary Stack
- **Language:** Rust for CLI, orchestration, PDF handling, output formatting
- **Model Inference:** Python wrapper (unavoidable for Transformers/vLLM ecosystem)
- **PDF to Image:** `pdf2image` via Poppler, or `mupdf` bindings
- **Model:** Qwen2-VL-7B-Instruct (primary), with alternatives configurable

### Model Options (configurable)
| Model | Size | VRAM Usage | Notes |
|-------|------|------------|-------|
| Qwen2-VL-7B-Instruct | 7B | ~16GB FP16 | Primary choice, excellent document understanding |
| Qwen2-VL-2B-Instruct | 2B | ~6GB FP16 | Faster, lighter, may sacrifice accuracy |
| Florence-2-large | 0.7B | ~3GB | Specialised vision, less flexible output |
| Moondream2 | 1.8B | ~4GB | Lightweight alternative |
| LLaVA-1.6-34B | 34B | ~20GB Q4 | If 7B proves insufficient |

### Inference Backend Options
- **Transformers** - Simplest, good for getting started
- **vLLM** - Better throughput if batching needed
- **Ollama** - Easiest deployment, slightly less control
- **llama.cpp** - If moving away from Python entirely (limited multimodal support currently)

## Output Schema

```json
{
  "extraction_version": "1.0",
  "source_file": "invoice_001.pdf",
  "extracted_at": "2024-01-15T10:30:00Z",
  "model_used": "Qwen2-VL-7B-Instruct",
  "confidence": {
    "overall": 0.95,
    "header": 0.98,
    "line_items": 0.92
  },
  "header": {
    "supplier_name": "Acme Supplies Ltd",
    "invoice_number": "INV-2024-001234",
    "invoice_date": "2024-01-10",
    "po_reference": "PO-5678",
    "currency": "GBP",
    "net_total": 1500.00,
    "vat_amount": 300.00,
    "gross_total": 1800.00,
    "payment_due_date": "2024-02-10"
  },
  "line_items": [
    {
      "line_number": 1,
      "product_code": "WDG-001",
      "description": "Widget Type A",
      "quantity": 100,
      "unit": "each",
      "unit_price": 10.00,
      "vat_rate": 20.0,
      "line_total": 1000.00
    },
    {
      "line_number": 2,
      "product_code": "WDG-002",
      "description": "Widget Type B",
      "quantity": 50,
      "unit": "each",
      "unit_price": 10.00,
      "vat_rate": 20.0,
      "line_total": 500.00
    }
  ],
  "warnings": [],
  "raw_text": null
}
```

## Directory Structure

```
invoice-ocr/
├── Cargo.toml
├── config/
│   └── default.toml          # Model settings, paths, thresholds
├── src/
│   ├── main.rs               # CLI entry point
│   ├── pdf.rs                # PDF to image conversion
│   ├── config.rs             # Configuration loading
│   ├── output.rs             # JSON/CSV/Excel formatting
│   └── inference/
│       └── bridge.rs         # Python inference bridge
├── python/
│   ├── requirements.txt
│   ├── inference_server.py   # Model loading and inference
│   └── models/
│       └── qwen.py           # Qwen-specific handling
├── prompts/
│   └── invoice_extract.txt   # Extraction prompt template
├── input/                    # Drop PDFs here
├── output/                   # Processed results
└── archive/                  # Processed PDFs moved here
```

## Configuration

```toml
# config/default.toml

[model]
name = "Qwen2-VL-7B-Instruct"
backend = "transformers"  # transformers, vllm, ollama
quantization = "none"     # none, int8, int4
device = "cuda:0"

[processing]
input_dir = "./input"
output_dir = "./output"
archive_dir = "./archive"
archive_processed = true
image_dpi = 150           # Higher = better quality, slower
image_format = "png"

[output]
format = "json"           # json, csv, both
include_raw_text = false
confidence_threshold = 0.7  # Flag for review below this

[inference]
max_tokens = 4096
temperature = 0.1         # Low for structured extraction
```

## CLI Interface

```bash
# Process single file
invoice-ocr process invoice.pdf

# Process all PDFs in input directory
invoice-ocr process --all

# Process with specific model
invoice-ocr process --model qwen2-vl-2b invoice.pdf

# Output to CSV instead of JSON
invoice-ocr process --format csv invoice.pdf

# List available models
invoice-ocr models

# Check system/GPU status
invoice-ocr status
```

## Phases and Milestones

### Phase 1: Foundation (Week 1-2)
**Goal:** Basic end-to-end pipeline working

- [ ] Project scaffolding (Rust + Python structure)
- [ ] PDF to image conversion working
- [ ] Qwen2-VL-7B loading and basic inference
- [ ] Hardcoded prompt, JSON output
- [ ] Process single PDF, output JSON file
- [ ] Manual verification of output quality

**Deliverable:** Can drop a PDF in, get JSON out. May need prompt tweaking.

### Phase 2: Robustness (Week 3-4)
**Goal:** Handle real-world invoice variety

- [ ] Multi-page PDF support
- [ ] Prompt engineering for reliable extraction
- [ ] Schema validation on output
- [ ] Confidence scoring implementation
- [ ] Error handling (corrupt PDFs, failed extractions)
- [ ] Test with actual invoice samples from regular suppliers

**Deliverable:** Reliable extraction across different supplier formats.

### Phase 3: Usability (Week 5-6)
**Goal:** Ready for regular use

- [ ] Configuration file support
- [ ] CLI polish (help text, progress indicators)
- [ ] CSV/Excel export option
- [ ] Batch processing with summary report
- [ ] Archive processed files option
- [ ] Low-confidence flagging for review

**Deliverable:** Usable tool for daily workflow.

### Phase 4: Flexibility (Week 7-8)
**Goal:** Model swapping and optimisation

- [ ] Model switching via config
- [ ] Add Florence-2 support
- [ ] Add Qwen2-VL-2B support
- [ ] Performance benchmarking (speed vs accuracy)
- [ ] Optional: vLLM backend for throughput
- [ ] Documentation and usage guide

**Deliverable:** Production-ready with model options.

### Phase 5: Polish (Optional, Week 9+)
**Goal:** Nice-to-haves if time permits

- [ ] Simple web UI for drag-and-drop
- [ ] Watched folder mode (auto-process new files)
- [ ] Supplier-specific prompt templates
- [ ] Extraction accuracy reporting
- [ ] Integration with accounting system (if needed)

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model accuracy insufficient | Medium | High | Test early with real invoices; have model alternatives ready |
| Complex table layouts fail | Medium | Medium | Prompt engineering; consider line-item-only mode |
| Python/Rust bridge complexity | Low | Medium | Start with subprocess/JSON IPC, upgrade if needed |
| Scanned invoice quality issues | Low | Medium | Test with actual scanned samples early |
| Multi-page invoices mishandled | Low | Medium | Design for multi-page from start |

## Success Criteria

1. **Accuracy:** >95% correct extraction on header fields for regular suppliers
2. **Line Items:** >90% correct line item extraction (description, quantity, price, total)
3. **Speed:** <30 seconds per invoice page
4. **Reliability:** Graceful handling of edge cases, no crashes
5. **Usability:** Non-technical user can process invoices with minimal training

## Notes

- Start with Transformers backend for simplicity; vLLM only if throughput becomes an issue
- The Python inference component is unavoidable given the current multimodal ecosystem; keep it isolated
- Confidence scoring may be tricky - start with simple heuristics (parsed OK, all required fields present)
- Regular suppliers mean we can potentially fine-tune prompts per supplier if needed
- Power Query handles JSON natively, so that's the primary output format

## Next Steps

1. Set up development environment on 3090 machine
2. Test Qwen2-VL-7B loading and basic inference
3. Gather sample invoices from actual suppliers
4. Begin Phase 1 implementation
