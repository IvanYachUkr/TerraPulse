"""Kill dashboard processes (FastAPI on :8000, Vite on :5173)."""

import subprocess
import sys


def kill_port(port):
    """Find and kill the process listening on the given port."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        killed = False
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                try:
                    subprocess.run(
                        ["taskkill", "/f", "/pid", pid],
                        capture_output=True, text=True
                    )
                    print(f"  Killed PID {pid} on port {port}")
                    killed = True
                except Exception:
                    pass
        if not killed:
            print(f"  No process found on port {port}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("Stopping dashboard...")
    print(f"[1/2] Port 8000 (FastAPI):")
    kill_port(8000)
    print(f"[2/2] Port 5173 (Vite):")
    kill_port(5173)
    print("Done.")


if __name__ == "__main__":
    main()
