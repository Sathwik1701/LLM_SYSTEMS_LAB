"""
LLM Architecture & Telemetry Dashboard - Core Package
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path so 'backend' or 'src.backend' both resolve cleanly
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
