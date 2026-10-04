"""
LLM Systems Lab - Application Runner
Starts the inspection server and web dashboard on http://localhost:8000.
"""

import sys
from pathlib import Path
import uvicorn

# Ensure both repository root and src/ are on sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
for p in [str(BASE_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    print(f"[*] Starting LLM Architecture & Telemetry Dashboard on http://localhost:{port} ...")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=True)

