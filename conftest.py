"""
Pytest configuration for LLM Architecture & Telemetry Dashboard.
Ensures src/ and src/backend are included in sys.path during test discovery.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
