"""Google Books Ngram access with a local SQLite frequency store."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import requests


class NgramRequestError(RuntimeError):
    pass


class GoogleNgramFrequencies:
    def __init__(
        self,
        store_path: Path,
        timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        self.store_path = store_path
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = requests.Session()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(store_path)
        self.database.execute(
            """
            CREATE TABLE IF NOT EXISTS frequencies (
                ngram TEXT NOT NULL,
                corpus_id INTEGER NOT NULL,
                start_year INTEGER NOT NULL,
                end_year INTEGER NOT NULL,
                smoothing INTEGER NOT NULL,
                frequency REAL NOT NULL,
                PRIMARY KEY (ngram, corpus_id, start_year, end_year, smoothing)
            )
            """
        )

    def close(self) -> None:
        self.database.close()
        self.session.close()

    def __enter__(self) -> "GoogleNgramFrequencies":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read(
        self, ngram: str, corpus_id: int, start_year: int, end_year: int, smoothing: int
    ) -> float | None:
        row = self.database.execute(
            """
            SELECT frequency FROM frequencies
            WHERE ngram = ? AND corpus_id = ? AND start_year = ? AND end_year = ? AND smoothing = ?
            """,
            (ngram, corpus_id, start_year, end_year, smoothing),
        ).fetchone()
        return None if row is None else float(row[0])

    def _write(
        self,
        ngram: str,
        corpus_id: int,
        start_year: int,
        end_year: int,
        smoothing: int,
        frequency: float,
    ) -> None:
        self.database.execute(
            "INSERT OR REPLACE INTO frequencies VALUES (?, ?, ?, ?, ?, ?)",
            (ngram, corpus_id, start_year, end_year, smoothing, frequency),
        )
        self.database.commit()

    @staticmethod
    def _combined_series(payload: list[dict]) -> list[float] | None:
        for item in payload:
            if str(item.get("ngram", "")).endswith(" (All)"):
                return item.get("timeseries") or None
        for item in payload:
            if item.get("type") == "CASE_INSENSITIVE":
                return item.get("timeseries") or None
        return payload[0].get("timeseries") if payload else None

    def frequency(
        self,
        ngram: str,
        corpus_id: int,
        start_year: int,
        end_year: int,
        smoothing: int,
    ) -> float:
        stored = self._read(ngram, corpus_id, start_year, end_year, smoothing)
        if stored is not None:
            return stored

        params = {
            "content": ngram,
            "year_start": start_year,
            "year_end": end_year,
            "corpus": corpus_id,
            "smoothing": smoothing,
            "case_insensitive": "true",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    "https://books.google.com/ngrams/json",
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                series = self._combined_series(response.json())
                value = sum(series) / len(series) if series else 0.0
                self._write(ngram, corpus_id, start_year, end_year, smoothing, value)
                return value
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        raise NgramRequestError(f"Could not retrieve frequency for {ngram!r}") from last_error
