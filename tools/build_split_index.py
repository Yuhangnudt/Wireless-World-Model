#!/usr/bin/env python3
"""CLI wrapper for :mod:`wwm.splits` (kept for copy-paste GitHub usage)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wwm.splits import _cli


if __name__ == "__main__":
    _cli()
