#!/usr/bin/env python3
"""Run tokenizer representation analysis without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialect_representation.analysis import main


if __name__ == "__main__":
    main()

