#!/usr/bin/env python3
"""
Request GPU services to unload their models.

Usage:
    python request_gpu_unload.py

This script polls known GPU services and requests them to unload if idle.
Use this before loading a large model to ensure GPU memory is available.
"""

import requests
import sys
import time

# Known GPU services that support /request-unload
SERVICES = [
    "http://10.99.0.3:8765",  # Invoice OCR (Qwen2-VL)
    # Add other services here as they implement the protocol
]


def request_unload(service_url: str, timeout: int = 5) -> dict:
    """Request a service to unload its model."""
    try:
        response = requests.post(
            f"{service_url}/request-unload",
            timeout=timeout,
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    print("Requesting GPU services to unload models...")
    unloaded_count = 0
    busy_count = 0

    for service in SERVICES:
        print(f"\nChecking {service}...")
        result = request_unload(service)

        if "error" in result:
            print(f"  ⚠️  Could not reach service: {result['error']}")
            continue

        if result.get("unloaded"):
            print(f"  ✓ Unloaded {result.get('message', 'model')}")
            unloaded_count += 1
        elif result.get("status") == "busy":
            print(f"  ⏱  {result.get('message', 'Service busy')}")
            busy_count += 1
        else:
            print(f"  ℹ️  {result.get('message', 'No action needed')}")

    print(f"\n{'='*50}")
    print(f"Unloaded: {unloaded_count} services")
    print(f"Busy: {busy_count} services")

    if busy_count > 0:
        print("\n⏱  Some services are busy. Wait ~30s and retry, or use OOM retry pattern.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
