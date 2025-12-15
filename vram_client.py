"""
VRAM Manager Client - Simple wrapper for services to request GPU memory
"""

import requests
import time
import atexit

class VRAMClient:
    def __init__(self, service_name, manager_url="http://localhost:5555"):
        self.service_name = service_name
        self.manager_url = manager_url
        self.allocation_id = None

    def request(self, model_name, vram_gb, wait=True, timeout=300, priority=5):
        """Request VRAM allocation - blocks until available"""
        response = requests.post(
            f"{self.manager_url}/request",
            json={
                "service": self.service_name,
                "model": model_name,
                "vram_gb": vram_gb,
                "wait": wait,
                "timeout": timeout,
                "priority": priority
            },
            timeout=timeout + 10
        )

        if response.status_code == 200:
            data = response.json()
            self.allocation_id = data['allocation_id']
            waited = data.get('waited_seconds', 0)
            if waited > 0:
                print(f"[{self.service_name}] Got VRAM after waiting {waited}s")
            return self.allocation_id
        else:
            raise Exception(f"VRAM request failed: {response.json().get('message', 'Unknown error')}")

    def heartbeat(self):
        """Send heartbeat to keep allocation alive"""
        if not self.allocation_id:
            return
        try:
            requests.post(
                f"{self.manager_url}/heartbeat/{self.allocation_id}",
                timeout=5
            )
        except:
            pass

    def release(self):
        """Release VRAM allocation"""
        if not self.allocation_id:
            return
        try:
            requests.post(
                f"{self.manager_url}/release/{self.allocation_id}",
                timeout=5
            )
            print(f"[{self.service_name}] Released VRAM")
            self.allocation_id = None
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Example usage:
if __name__ == "__main__":
    # Request VRAM (will wait if not available)
    with VRAMClient("test-service") as client:
        alloc_id = client.request("test-model", vram_gb=5, wait=True)
        print(f"Got allocation: {alloc_id}")

        # Do work...
        time.sleep(10)

        # Send heartbeat periodically if work takes long
        client.heartbeat()

        # Automatically released on exit
