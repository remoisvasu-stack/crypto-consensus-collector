"""core/hf_store.py — Persist the recorded data log to an HF Hub dataset repo.

On startup:  download training_history.csv from hub → local /data
After writes: schedule a background upload back to hub

Env vars:
    HF_TOKEN         – write-access HF token
    HF_DATASET_REPO  – e.g. "youruser/btc-consensus-data"
"""
from __future__ import annotations
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "ndsideload/btc-consensus-data")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

HISTORY_FILENAME = "training_history.csv"

_upload_lock = threading.Lock()
_pending = False


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def hub_enabled() -> bool:
    return bool(HF_TOKEN and HF_DATASET_REPO)


def ensure_repo_exists() -> bool:
    if not hub_enabled():
        return False
    try:
        _api().create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset",
                           exist_ok=True, private=True)
        return True
    except Exception:
        log.exception("Failed to verify HF dataset repo")
        return False


def _download_file(filename: str, local_path: Path) -> bool:
    if not hub_enabled():
        return False
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename,
                        repo_type="dataset", token=HF_TOKEN,
                        local_dir=str(local_path.parent),
                        local_dir_use_symlinks=False)
        log.info("Downloaded %s", filename)
        return True
    except Exception as e:
        if "404" in str(e) or "EntryNotFound" in type(e).__name__:
            log.info("%s not on hub (first run?)", filename)
        else:
            log.warning("Download %s failed: %s", filename, e)
        return False


def _upload_file(local_path: Path, path_in_repo: str) -> bool:
    if not hub_enabled() or not local_path.exists():
        return False
    try:
        _api().upload_file(path_or_fileobj=str(local_path),
                           path_in_repo=path_in_repo,
                           repo_id=HF_DATASET_REPO, repo_type="dataset")
        log.info("Uploaded %s", path_in_repo)
        return True
    except Exception:
        log.exception("Upload %s failed", path_in_repo)
        return False


def restore_all_from_hub() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / HISTORY_FILENAME
    r = {HISTORY_FILENAME: "exists" if p.exists() else _download_file(HISTORY_FILENAME, p)}
    log.info("Restore: %s", r)
    return r


def _persist_all():
    p = DATA_DIR / HISTORY_FILENAME
    if p.exists():
        _upload_file(p, HISTORY_FILENAME)


def schedule_persist():
    global _pending
    if not hub_enabled():
        return
    with _upload_lock:
        if _pending:
            return
        _pending = True

    def _do():
        global _pending
        time.sleep(30)
        try:
            _persist_all()
        finally:
            with _upload_lock:
                _pending = False

    threading.Thread(target=_do, daemon=True).start()
