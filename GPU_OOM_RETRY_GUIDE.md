# GPU OOM Retry Pattern

Simple pattern for sharing GPU memory across multiple services without coordination.

## Strategy
1. All services try to load models normally
2. Catch OOM errors
3. Wait 30-60 seconds (for other services to auto-unload)
4. Retry up to 3 times
5. Configure all services to unload quickly when idle

## Python (PyTorch / Transformers)

```python
import torch
import time

def load_model_with_retry(max_retries=3, retry_delay=30):
    for attempt in range(max_retries):
        try:
            # Your model loading code
            model = MyModel.from_pretrained("model-name")
            model.to("cuda")
            return model

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"OOM on attempt {attempt+1}, waiting {retry_delay}s...")
                    torch.cuda.empty_cache()  # Clean up
                    time.sleep(retry_delay)
                else:
                    raise  # Give up after max retries
            else:
                raise  # Not OOM, raise immediately
```

## ComfyUI / Flux (Python-based)

Add to your workflow/node:

```python
# In your model loading function
import torch
import time

def load_flux_model(path, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Your Flux/ComfyUI loading code
            model = comfy.utils.load_torch_file(path)
            return model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"GPU busy, retrying in 30s...")
                    torch.cuda.empty_cache()
                    time.sleep(30)
                else:
                    raise
            else:
                raise
```

## Ollama

Ollama already handles this! Just configure quick unloading:

```bash
# In /etc/systemd/system/ollama.service.d/override.conf
Environment="OLLAMA_KEEP_ALIVE=30s"
```

## Shell Scripts

For any GPU command:

```bash
#!/bin/bash
MAX_RETRIES=3
RETRY_DELAY=30

for i in $(seq 1 $MAX_RETRIES); do
    if your-gpu-command; then
        exit 0
    fi

    if [ $i -lt $MAX_RETRIES ]; then
        echo "GPU busy, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    fi
done

echo "Failed after $MAX_RETRIES attempts"
exit 1
```

## Key Settings

### Invoice OCR
✅ Already configured - retries OOM 3x with 30s delays

### Ollama
✅ Already configured - `OLLAMA_KEEP_ALIVE=30s`

### Your Other Services
Add the retry pattern above to each service's model loading code.

## How It Works

**12:00** - Scheduled Qwen task starts, loads 4GB
**12:01** - User uploads invoice, tries to load 18GB → OOM
**12:01** - Invoice OCR waits 30s
**12:01:30** - Qwen task finishes, auto-unloads after 30s
**12:02** - Invoice OCR retry succeeds, loads 18GB
**12:03** - Invoice processing completes, unloads
**12:03:30** - GPU is free again

No coordination needed - just resilient retry logic!
