"""Shared test config: repo-root imports + safety against accidental API spend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
