from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


REQUIRED_MODULES = ["torch", "fastapi", "uvicorn", "numpy"]


def check_dependencies() -> bool:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Run: python3 -m pip install -r backend/requirements.txt")
        return False
    return True




def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_port(preferred: int, fallbacks: list[int]) -> int:
    for port in [preferred] + fallbacks:
        if is_port_free(port):
            return port
    return preferred
def start_process(cmd, name):
    return subprocess.Popen(cmd, cwd=BASE_DIR)


def main() -> None:
    if not check_dependencies():
        return
    procs = []
    try:
        backend = start_process([sys.executable, "-m", "backend.server"], "backend")
        procs.append(backend)
        frontend_port = pick_port(8000, [8002, 8003, 8080])
        frontend = start_process([sys.executable, "-m", "http.server", str(frontend_port)], "frontend")
        procs.append(frontend)

        print("Backend running at http://127.0.0.1:8001")
        print(f"Frontend running at http://127.0.0.1:{frontend_port}")
        print("Press Ctrl+C to stop both.")

        while True:
            time.sleep(0.5)
            for proc in procs:
                if proc.poll() is not None:
                    return
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
