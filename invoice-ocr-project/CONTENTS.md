# Invoice OCR Project - Documentation Contents

## Overview

This documentation package covers the planning and design of a local invoice extraction system using multimodal LLMs (primarily Qwen2-VL) running on a 3090 GPU.

## Files

### PLANNING.md (Start Here)
The main planning document. Covers:
- Project overview and problem statement
- Hardware requirements
- Architecture and component design
- Technology choices and model options
- Output schema (JSON structure)
- Project directory structure
- Configuration format
- CLI interface design
- **Phases and milestones** (5 phases, ~8 weeks)
- Risk mitigation
- Success criteria

### PROMPTS.md
Prompt templates for the LLM. Covers:
- Primary extraction prompt (header + line items)
- Multi-page invoice handling
- Validation/verification prompt
- Supplier-specific prompt additions
- Confidence assessment prompt
- Prompt engineering notes and known issues

### TECHNICAL.md
Implementation details. Covers:
- PDF to image conversion (Poppler, DPI settings)
- Model inference code examples (Transformers, Ollama, vLLM)
- Rust-Python bridge options
- Output formatting (JSON, CSV, Excel)
- Error handling strategies
- Performance considerations
- Testing strategy

## Recommended Reading Order

1. **PLANNING.md** - Understand scope, architecture, and timeline
2. **PROMPTS.md** - Review extraction prompts before implementation
3. **TECHNICAL.md** - Reference during implementation

## Quick Start Checklist

After reviewing the docs, the first steps are:

1. Set up Python environment with Transformers and Qwen2-VL dependencies
2. Test model loading and basic inference on the 3090
3. Test PDF → image conversion with sample invoices
4. Run first extraction and evaluate output quality
5. Begin Phase 1 implementation

## Notes

- The Rust/Python split is intentional: Rust for CLI, file handling, and output; Python for model inference (unavoidable given the ecosystem)
- Start with Transformers backend, only move to vLLM if throughput becomes an issue
- JSON is the primary output format (Power Query compatible); CSV/Excel are optional exports
- Model swapping is designed in from the start - easy to try alternatives if Qwen2-VL-7B doesn't meet accuracy targets
