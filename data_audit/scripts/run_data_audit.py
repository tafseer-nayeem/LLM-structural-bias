#!/usr/bin/env python3
"""Run the data audit package without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dialect_data_audit.audit import main


if __name__ == "__main__":
    main()

