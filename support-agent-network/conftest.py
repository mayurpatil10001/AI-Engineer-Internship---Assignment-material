"""
pytest configuration: adds project root to sys.path and sets working directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is importable as a package root
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
