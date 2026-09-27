"""General text preprocessing and schema extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for the normalization used in the corpus audits."""

    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    ascii_only: bool = True
    split_dash_slash: bool = True
    alphabetic_only: bool = True
    normalize_whitespace: bool = True


DEFAULT_NORMALIZATION = NormalizationConfig()


def normalize_text(text: Any, config: NormalizationConfig = DEFAULT_NORMALIZATION) -> str:
    """Normalize text using the same defaults as the original pretraining notebooks."""

    if text is None:
        return ""
    text = str(text)
    if config.lowercase:
        text = text.lower()
    if config.remove_html:
        text = re.sub(r"<[^>]+>", " ", text)
    if config.remove_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
    if config.remove_emails:
        text = re.sub(r"\S+@\S+", " ", text)
    if config.ascii_only:
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
    if config.split_dash_slash:
        text = text.replace("-", " ").replace("/", " ")
    if config.alphabetic_only:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    if config.normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def recursive_strings(value: Any) -> Iterator[str]:
    """Yield every string contained in nested dict/list structures."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from recursive_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from recursive_strings(child)


def values_for_path(row: Dict[str, Any], path: str) -> List[Any]:
    """Resolve dotted field paths with optional '*' wildcards against a dataset row."""

    items: List[Any] = [row]
    for part in path.split("."):
        next_items: List[Any] = []
        for item in items:
            if part == "*":
                if isinstance(item, dict):
                    next_items.extend(item.values())
                elif isinstance(item, list):
                    next_items.extend(item)
                continue
            if isinstance(item, dict) and part in item:
                next_items.append(item[part])
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict) and part in child:
                        next_items.append(child[part])
        items = next_items
        if not items:
            break
    return items


def extract_text(row: Dict[str, Any], text_fields: List[str]) -> str:
    """Extract all configured text fields from a dataset row."""

    pieces: List[str] = []
    for field in text_fields:
        for value in values_for_path(row, field):
            pieces.extend(recursive_strings(value))
    return "\n".join(piece for piece in pieces if piece)


def passes_filters(row: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Return True if a row satisfies the configured exact-match filters."""

    if not filters:
        return True
    for field, expected in filters.items():
        values = values_for_path(row, field)
        if not values or all(value != expected for value in values):
            return False
    return True
