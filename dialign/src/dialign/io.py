"""Input helpers for DiAlign."""

from __future__ import annotations

import json
from pathlib import Path


def load_markers(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as stream:
        pairs = json.load(stream)
    markers = set()
    for pair in pairs:
        ame = pair.get("ame", pair.get("us"))
        bre = pair.get("bre", pair.get("uk"))
        if ame is not None:
            markers.add(str(ame).lower())
        if bre is not None:
            markers.add(str(bre).lower())
    return markers
