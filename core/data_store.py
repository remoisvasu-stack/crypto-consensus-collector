"""core/data_store.py — Thread-safe rolling buffer for live ticks (24/7).

The recorder collects one option-chain feature row per minute. We keep the most
recent N rows in memory so the dashboard can show a recent-rows view without
re-reading the CSV.

BTC trades 24/7, so there are no trading-day boundaries: rows are deduplicated by
their minute bucket (ts // 60_000) rather than by (Date, Time Slot), and the
buffer holds a continuous stream.
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Optional

import pandas as pd

# One day of 1-minute rows kept in memory for the dashboard's recent-rows view.
DEFAULT_BUFFER = 1440


class TickBuffer:
    """Append-only rolling buffer keyed by minute bucket of `ts` (epoch ms)."""

    def __init__(self, maxlen: int = DEFAULT_BUFFER):
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._last_key: Optional[int] = None

    @staticmethod
    def _key(row: dict) -> Optional[int]:
        ts = row.get('ts')
        try:
            return int(ts) // 60_000
        except (TypeError, ValueError):
            return None

    def append(self, row: dict) -> bool:
        """Add a row; returns False if it falls in the same minute as the last."""
        key = self._key(row)
        with self._lock:
            if key is not None and key == self._last_key:
                return False
            self._buf.append(row)
            self._last_key = key
            return True

    def to_frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(list(self._buf))

    def latest(self) -> Optional[dict]:
        with self._lock:
            return dict(self._buf[-1]) if self._buf else None

    def size(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._last_key = None
