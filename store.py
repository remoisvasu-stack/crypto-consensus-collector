"""store.py — Append-only HF dataset writer for the collector.

The HF dataset is the single source of truth for training history. This module
NEVER overwrites the file wholesale: each flush pulls the current CSV, appends
only rows whose `ts` is not already present, and pushes the result back.

Why append-only matters: the collector emits a new `ts` every minute, while the
user only ever edits OLD `ts` rows through the HF web UI. Those two row-sets are
disjoint, so the collector can never clobber a manual correction. Backups are
free — every push is a git commit in the dataset repo.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from typing import Optional

import pandas as pd

log = logging.getLogger("collector.store")

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "hugfacmaster/btc-consensus-data")
HISTORY_FILENAME = os.getenv("HISTORY_FILENAME", "training_history.csv")

_lock = threading.Lock()


def hub_enabled() -> bool:
    return bool(HF_TOKEN and HF_DATASET_REPO)


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def ensure_repo() -> bool:
    if not hub_enabled():
        return False
    try:
        _api().create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset",
                           exist_ok=True, private=True)
        return True
    except Exception:
        log.exception("Failed to ensure dataset repo")
        return False


def _download_current() -> Optional[pd.DataFrame]:
    """Pull the current history CSV from the hub, or None if it doesn't exist yet."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HISTORY_FILENAME,
                               repo_type="dataset", token=HF_TOKEN)
        return pd.read_csv(path)
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            log.info("%s not on hub yet (first run)", HISTORY_FILENAME)
        else:
            log.warning("Download history failed: %s", e)
        return None


def append_rows(rows: list[dict]) -> dict:
    """Merge `rows` into the hub CSV, keeping only new `ts` values, then push.

    Re-pulls immediately before pushing so a concurrent manual edit on the hub is
    preserved (its rows survive the merge; we only add brand-new timestamps).
    """
    if not rows:
        return {"ok": True, "added": 0, "reason": "no_rows"}
    if not hub_enabled():
        log.warning("HF hub not configured — %d rows not persisted", len(rows))
        return {"ok": False, "added": 0, "reason": "hub_disabled"}

    with _lock:
        current = _download_current()
        incoming = pd.DataFrame(rows)
        if current is not None and not current.empty:
            known = set(current["ts"].astype("int64")) if "ts" in current else set()
            fresh = incoming[~incoming["ts"].astype("int64").isin(known)]
            combined = pd.concat([current, fresh], ignore_index=True)
        else:
            fresh = incoming
            combined = incoming
        # Defensive: collapse any duplicate ts (keep last), keep chronological order.
        combined = (combined.drop_duplicates(subset=["ts"], keep="last")
                            .sort_values("ts").reset_index(drop=True))

        added = int(len(fresh))
        if added == 0:
            return {"ok": True, "added": 0, "reason": "all_known"}

        buf = io.BytesIO(combined.to_csv(index=False).encode())
        try:
            _api().upload_file(
                path_or_fileobj=buf, path_in_repo=HISTORY_FILENAME,
                repo_id=HF_DATASET_REPO, repo_type="dataset",
                commit_message=f"collector: +{added} rows ({len(combined)} total)",
            )
        except Exception:
            log.exception("Upload history failed")
            return {"ok": False, "added": 0, "reason": "upload_failed"}

        log.info("Appended %d new rows (%d total) to %s",
                 added, len(combined), HISTORY_FILENAME)
        return {"ok": True, "added": added, "total": int(len(combined))}
