#!/bin/bash
while true; do
    clear
    echo "=== Qwen2-VL-7B Download Progress ==="
    echo
    du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/ 2>/dev/null
    echo
    echo "Downloading files:"
    ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/blobs/*.incomplete 2>/dev/null | awk '{print "  " $5}'

    COUNT=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/blobs/*.incomplete 2>/dev/null | wc -l)
    echo
    echo "Files remaining: $COUNT / 4"

    if [ "$COUNT" -eq 0 ]; then
        echo
        echo "Download complete!"
        break
    fi

    sleep 30
done
