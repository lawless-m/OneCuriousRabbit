# Technical Implementation Notes

## PDF to Image Conversion

### Recommended Approach: Poppler (pdftoppm)

```bash
# Install on Debian
sudo apt install poppler-utils

# Convert PDF to PNG images
pdftoppm -png -r 150 invoice.pdf invoice_page

# Output: invoice_page-1.png, invoice_page-2.png, etc.
```

**DPI Considerations:**
- 150 DPI: Good balance of quality/speed for most born-digital PDFs
- 200 DPI: Better for scanned documents or small text
- 300 DPI: Maximum quality, slower processing, larger files

### Rust Integration Options

1. **Subprocess call to pdftoppm** (simplest)
   ```rust
   use std::process::Command;
   
   fn pdf_to_images(pdf_path: &Path, output_dir: &Path, dpi: u32) -> Result<Vec<PathBuf>> {
       let output = Command::new("pdftoppm")
           .args(["-png", "-r", &dpi.to_string()])
           .arg(pdf_path)
           .arg(output_dir.join("page"))
           .output()?;
       // Collect generated files...
   }
   ```

2. **mupdf-rs crate** (native, more control)
   - Rust bindings to MuPDF
   - No subprocess overhead
   - More complex setup

3. **pdfium-render** (another native option)
   - Bindings to PDFium (Chrome's PDF engine)
   - Good quality rendering

### Image Format Choice

**PNG:** Lossless, larger files, better for text
**JPEG:** Smaller files, slight quality loss, fine for most cases

Recommendation: PNG for maximum accuracy, JPEG if storage/transfer is a concern.

---

## Model Inference

### Option 1: Transformers (Python)

```python
# python/inference_server.py

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import json

class InvoiceExtractor:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
    
    def extract(self, image_path: str, prompt: str) -> dict:
        image = Image.open(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to("cuda:0")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False
        )
        
        response = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Parse JSON from response
        return self._parse_json(response)
    
    def _parse_json(self, text: str) -> dict:
        # Strip markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        return json.loads(text)
```

### Option 2: Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model (if available)
ollama pull qwen2-vl:7b

# Run inference via API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2-vl:7b",
  "prompt": "...",
  "images": ["base64_encoded_image"],
  "stream": false
}'
```

Note: Check Ollama model availability - multimodal support varies.

### Option 3: vLLM (for throughput)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    trust_remote_code=True,
    dtype="float16"
)

# vLLM handles batching automatically
```

---

## Rust-Python Bridge

### Option 1: JSON over stdin/stdout (simplest)

```rust
// src/inference/bridge.rs

use std::process::{Command, Stdio};
use std::io::Write;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct InferenceRequest {
    image_path: String,
    prompt: String,
}

#[derive(Deserialize)]
struct InferenceResponse {
    success: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
}

pub fn run_inference(image_path: &str, prompt: &str) -> Result<serde_json::Value> {
    let request = InferenceRequest {
        image_path: image_path.to_string(),
        prompt: prompt.to_string(),
    };
    
    let mut child = Command::new("python")
        .args(["python/inference_cli.py"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    
    let stdin = child.stdin.as_mut().unwrap();
    serde_json::to_writer(stdin, &request)?;
    
    let output = child.wait_with_output()?;
    let response: InferenceResponse = serde_json::from_slice(&output.stdout)?;
    
    match response.data {
        Some(data) => Ok(data),
        None => Err(anyhow::anyhow!(response.error.unwrap_or_default()))
    }
}
```

### Option 2: HTTP Server (better for persistent model)

Python side runs a FastAPI/Flask server, Rust calls via HTTP.

```python
# python/inference_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
extractor = None  # Lazy load

@app.post("/extract")
async def extract(image_path: str, prompt: str):
    global extractor
    if extractor is None:
        extractor = InvoiceExtractor()
    return extractor.extract(image_path, prompt)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

```rust
// Rust side
let client = reqwest::blocking::Client::new();
let response = client.post("http://127.0.0.1:8765/extract")
    .json(&request)
    .send()?;
```

### Option 3: PyO3 (native embedding)

More complex but eliminates subprocess overhead. Probably overkill for this use case.

---

## Output Formatting

### JSON (Primary)

```rust
// src/output.rs

use serde::Serialize;
use std::fs::File;

pub fn write_json<T: Serialize>(data: &T, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, data)?;
    Ok(())
}
```

### CSV Export

```rust
use csv::Writer;

pub fn write_csv(invoices: &[Invoice], path: &Path) -> Result<()> {
    let mut wtr = Writer::from_path(path)?;
    
    // Header row
    wtr.write_record(&[
        "source_file", "supplier_name", "invoice_number", 
        "invoice_date", "po_reference", "net_total", 
        "vat_amount", "gross_total"
    ])?;
    
    for inv in invoices {
        wtr.write_record(&[
            &inv.source_file,
            &inv.header.supplier_name,
            &inv.header.invoice_number,
            // ... etc
        ])?;
    }
    
    wtr.flush()?;
    Ok(())
}

// Separate CSV for line items with invoice_number as foreign key
pub fn write_line_items_csv(invoices: &[Invoice], path: &Path) -> Result<()> {
    // Similar structure...
}
```

### Excel Export

Use `rust_xlsxwriter` crate:

```rust
use rust_xlsxwriter::{Workbook, Format};

pub fn write_excel(invoices: &[Invoice], path: &Path) -> Result<()> {
    let mut workbook = Workbook::new();
    
    // Headers sheet
    let headers = workbook.add_worksheet();
    headers.set_name("Invoices")?;
    // Write header data...
    
    // Line items sheet
    let items = workbook.add_worksheet();
    items.set_name("Line Items")?;
    // Write line item data...
    
    workbook.save(path)?;
    Ok(())
}
```

---

## Error Handling

### PDF Conversion Failures
- Corrupt PDF: Log error, skip file, continue batch
- Password protected: Detect and report, skip file
- Zero pages: Report as invalid

### Model Inference Failures
- CUDA out of memory: Reduce batch size, retry
- Model timeout: Set reasonable timeout, retry once
- Invalid JSON response: Retry with stricter prompt, flag for review

### Validation Failures
- Missing required fields: Flag for review
- Totals don't match: Flag with warning, still output
- Implausible values: Flag for review

---

## Performance Considerations

### GPU Memory
- Qwen2-VL-7B FP16: ~16GB VRAM
- Leaves headroom on 24GB for image processing
- If memory issues: try int8 quantization via bitsandbytes

### Processing Speed
- Model loading: 30-60 seconds (once per session)
- Per-page inference: 10-20 seconds typical
- PDF conversion: <1 second per page

### Batch Processing
- Keep model loaded between files
- Process all pages of one PDF before moving to next
- Consider parallel PDF conversion while inference runs

---

## Testing Strategy

### Unit Tests
- JSON schema validation
- CSV/Excel output formatting
- Configuration loading

### Integration Tests
- End-to-end PDF → JSON pipeline
- Multi-page document handling
- Error recovery

### Accuracy Tests
- Sample invoices with known correct values
- Compare extracted vs expected
- Track accuracy metrics over time

### Test Invoice Sources
- Create synthetic invoices with known values
- Use redacted real invoices from regular suppliers
- Include edge cases: handwritten notes, stamps, multiple VAT rates
