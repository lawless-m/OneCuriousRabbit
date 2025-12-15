#!/usr/bin/env python3
"""
GPU VRAM Manager - Acts as a gatekeeper for GPU memory allocation
Coordinates multiple services using the same GPU to prevent OOM errors
"""

import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
VRAM_STATE_FILE = "/tmp/vram_manager_state.json"
TOTAL_VRAM_GB = 24  # RTX 3090
RESERVED_VRAM_GB = 2  # Keep some free for allocations
IDLE_TIMEOUT = 300  # 5 minutes idle = unload

# State tracking
allocations = {}
lock = threading.Lock()


def get_gpu_memory_used():
    """Get current GPU memory usage in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip()) / 1024  # Convert MB to GB
    except:
        return 0


def save_state():
    """Save allocation state to disk"""
    with open(VRAM_STATE_FILE, 'w') as f:
        json.dump({
            'allocations': {
                k: {
                    'vram_gb': v['vram_gb'],
                    'allocated_at': v['allocated_at'].isoformat(),
                    'last_used': v['last_used'].isoformat(),
                    'service': v['service'],
                    'model': v['model']
                }
                for k, v in allocations.items()
            }
        }, f, indent=2)


def load_state():
    """Load allocation state from disk"""
    global allocations
    try:
        with open(VRAM_STATE_FILE, 'r') as f:
            data = json.load(f)
            allocations = {
                k: {
                    'vram_gb': v['vram_gb'],
                    'allocated_at': datetime.fromisoformat(v['allocated_at']),
                    'last_used': datetime.fromisoformat(v['last_used']),
                    'service': v['service'],
                    'model': v['model']
                }
                for k, v in data['allocations'].items()
            }
    except:
        allocations = {}


def get_available_vram():
    """Calculate available VRAM"""
    allocated = sum(a['vram_gb'] for a in allocations.values())
    return TOTAL_VRAM_GB - RESERVED_VRAM_GB - allocated


def cleanup_idle_allocations():
    """Remove allocations that have been idle too long"""
    now = datetime.now()
    to_remove = []

    with lock:
        for alloc_id, info in allocations.items():
            idle_time = (now - info['last_used']).total_seconds()
            if idle_time > IDLE_TIMEOUT:
                to_remove.append(alloc_id)

        for alloc_id in to_remove:
            print(f"Auto-releasing idle allocation: {alloc_id} ({allocations[alloc_id]['service']})")
            del allocations[alloc_id]

        if to_remove:
            save_state()


@app.route('/status', methods=['GET'])
def status():
    """Get current VRAM status"""
    cleanup_idle_allocations()

    return jsonify({
        'total_vram_gb': TOTAL_VRAM_GB,
        'reserved_gb': RESERVED_VRAM_GB,
        'available_gb': get_available_vram(),
        'allocated_gb': sum(a['vram_gb'] for a in allocations.values()),
        'actual_gpu_used_gb': get_gpu_memory_used(),
        'allocations': {
            k: {
                'service': v['service'],
                'model': v['model'],
                'vram_gb': v['vram_gb'],
                'allocated_at': v['allocated_at'].isoformat(),
                'last_used': v['last_used'].isoformat(),
                'idle_seconds': (datetime.now() - v['last_used']).total_seconds()
            }
            for k, v in allocations.items()
        }
    })


@app.route('/request', methods=['POST'])
def request_vram():
    """
    Request VRAM allocation - BLOCKS until space is available
    Body: {
        "service": "invoice-ocr",
        "model": "Qwen2-VL-7B",
        "vram_gb": 18,
        "priority": 5,  # 1-10, higher = more important
        "wait": true,   # If true, block until space available
        "timeout": 300  # Max seconds to wait (default 300 = 5min)
    }
    """
    data = request.json
    service = data.get('service', 'unknown')
    model = data.get('model', 'unknown')
    requested_gb = data.get('vram_gb', 0)
    priority = data.get('priority', 5)
    should_wait = data.get('wait', True)
    timeout = data.get('timeout', 300)

    start_time = time.time()
    wait_count = 0

    while True:
        cleanup_idle_allocations()

        with lock:
            available = get_available_vram()

            if requested_gb <= available:
                # Space available - allocate now
                alloc_id = f"{service}_{int(time.time())}"
                allocations[alloc_id] = {
                    'service': service,
                    'model': model,
                    'vram_gb': requested_gb,
                    'allocated_at': datetime.now(),
                    'last_used': datetime.now(),
                    'priority': priority
                }

                save_state()

                return jsonify({
                    'success': True,
                    'allocation_id': alloc_id,
                    'allocated_gb': requested_gb,
                    'available_gb': get_available_vram(),
                    'waited_seconds': int(time.time() - start_time)
                })

        # Not enough space
        if not should_wait:
            return jsonify({
                'success': False,
                'message': f'Insufficient VRAM: requested {requested_gb}GB, available {available:.1f}GB',
                'available_gb': available,
                'wait_recommended': True
            }), 503

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return jsonify({
                'success': False,
                'message': f'Timeout after {int(elapsed)}s waiting for {requested_gb}GB VRAM',
                'available_gb': available
            }), 408

        # Wait and retry
        wait_count += 1
        if wait_count == 1:
            print(f"[{service}] Waiting for {requested_gb}GB VRAM (currently {available:.1f}GB available)...")
        time.sleep(5)  # Check every 5 seconds


@app.route('/heartbeat/<alloc_id>', methods=['POST'])
def heartbeat(alloc_id):
    """Update last-used timestamp to prevent auto-release"""
    with lock:
        if alloc_id in allocations:
            allocations[alloc_id]['last_used'] = datetime.now()
            save_state()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Allocation not found'}), 404


@app.route('/release/<alloc_id>', methods=['POST'])
def release(alloc_id):
    """Release VRAM allocation"""
    with lock:
        if alloc_id in allocations:
            del allocations[alloc_id]
            save_state()
            return jsonify({'success': True, 'available_gb': get_available_vram()})
        return jsonify({'success': False, 'message': 'Allocation not found'}), 404


if __name__ == '__main__':
    print("Starting VRAM Manager...")
    print(f"Total VRAM: {TOTAL_VRAM_GB}GB")
    print(f"Reserved: {RESERVED_VRAM_GB}GB")
    print(f"Allocatable: {TOTAL_VRAM_GB - RESERVED_VRAM_GB}GB")

    load_state()

    # Cleanup thread
    def cleanup_loop():
        while True:
            time.sleep(60)
            cleanup_idle_allocations()

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    app.run(host='0.0.0.0', port=5555)
