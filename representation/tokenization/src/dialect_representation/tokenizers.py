"""Tokenizer adapters used by the representation audit."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TokenizerAdapter:
    name: str
    backend: str
    tokenizer: Any
    vocab_size: Optional[int] = None
    total_vocab_size: Optional[int] = None

    def tokens(self, text: str) -> List[Any]:
        if self.backend == "tiktoken":
            return self.tokenizer.encode(text)
        if self.backend == "hf":
            return self.tokenizer.tokenize(text)
        if self.backend == "character":
            return [ch for ch in text if not ch.isspace()]
        if self.backend == "regex":
            return re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def token_count(self, text: str) -> int:
        return len(self.tokens(text))


def load_tokenizer(entry: Dict[str, Any], hf_token: Optional[str] = None) -> TokenizerAdapter:
    """Load one tokenizer from its configuration entry."""

    backend = entry["backend"]
    name = entry["name"]

    if backend == "tiktoken":
        import tiktoken

        if entry.get("encoding_name"):
            enc = tiktoken.get_encoding(entry["encoding_name"])
        else:
            enc = tiktoken.encoding_for_model(entry["model"])
        return TokenizerAdapter(
            name=name,
            backend="tiktoken",
            tokenizer=enc,
            vocab_size=getattr(enc, "n_vocab", None),
            total_vocab_size=getattr(enc, "n_vocab", None),
        )

    if backend == "hf":
        from transformers import AutoTokenizer

        kwargs: Dict[str, Any] = {}
        token = hf_token or (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HF_ACCESS_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if token:
            kwargs["token"] = token
        if entry.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        if entry.get("fix_mistral_regex"):
            kwargs["fix_mistral_regex"] = True
        tokenizer = AutoTokenizer.from_pretrained(entry["repo_id"], **kwargs)
        return TokenizerAdapter(
            name=name,
            backend="hf",
            tokenizer=tokenizer,
            vocab_size=getattr(tokenizer, "vocab_size", None),
            total_vocab_size=len(tokenizer),
        )

    if backend in {"character", "regex"}:
        return TokenizerAdapter(name=name, backend=backend, tokenizer=None, vocab_size=None, total_vocab_size=None)

    raise ValueError(f"Unknown tokenizer backend: {backend}")
