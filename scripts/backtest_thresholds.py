"""scripts/backtest_thresholds.py — Calibrate the BTC move-threshold ladder.

The NIFTY system used a {0.10, 0.20, 0.30, 0.45, 0.65}% ladder over a 120-min
horizon. BTC is far more volatile, so those thresholds would be hit almost every
bar and carry no signal. This script measures, over historical 1-minute spot
bars, how often price moves by each candidate threshold within each horizon —
the *base rate* (label density) the consensus models would learn against.

Goal: pick a 5-rung ladder where the smallest rung fires often-but-not-always
and the largest rung is rare-but-not-never, so the nested 3of5/4of5/5of5
consensus has real discriminative range.

Labels use CLOSE-to-CLOSE max move (matching add_targets in consensus.py, which
sees one price per minute), not intrabar OHLC extremes.

Usage:
    python scripts/backtest_thresholds.py --days 30
    python scripts/backtest_thresholds.py --days 60 --horizons 60,120,180
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.binance_feed import fetch_klines  # noqa: E402

# Candidate thresholds (% move) to measure. Wider than NIFTY's on purpose.
GRID = [0.10, 0.20, 0.30, 0.45, 0.65, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def load_klines(symbol: str, days: int) -> pd.DataFrame:
    """Page backwards through 1m klines to assemble ~days of history."""
    want = days * 1440
    end_ms = None
    chunks: list[list[dict]] = []
    got = 0
    while got < want:
        batch = fetch_klines(symbol, "1m", limit=1000, end_ms=end_ms)
        if not batch:
            break
        chunks.append(batch)
        got += len(batch)
        end_ms = batch[0]["open_time"] - 1   # page strictly older
        if len(batch) < 1000:
            break
        time.sleep(0.15)                       # be polite to the API
    rows = [r for c in reversed(chunks) for r in c]
    df = pd.DataFrame(rows).drop_duplicates("open_time").sort_values("open_time")
    return df.reset_index(drop=True)


def forward_extremes(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Max up / max down close-to-close move over the next `horizon` bars."""
    fwd_max = close.rolling(horizon).max().shift(-horizon)
    fwd_min = close.rolling(horizon).min().shift(-horizon)
    up_move = fwd_max / close - 1.0           # >= 0
    dn_move = fwd_min / close - 1.0           # <= 0
    return up_move, dn_move


def report(df: pd.DataFrame, horizons: list[int]) -> None:
    close = df["close"]
    rv = close.pct_change().std() * np.sqrt(1440)   # ~daily realized vol
    print(f"\nLoaded {len(df):,} 1m bars "
          f"({(df['open_time'].iloc[-1] - df['open_time'].iloc[0]) / 86400000:.1f} days)")
    print(f"Approx daily realized vol: {rv * 100:.2f}%\n")

    for H in horizons:
        up, dn = forward_extremes(close, H)
        valid = up.notna()
        n = int(valid.sum())
        print(f"-- Horizon {H} min  (n={n:,}) " + "-" * 30)
        print(f"  {'thr%':>6} {'UP hit%':>9} {'DN hit%':>9} {'either%':>9}")
        for t in GRID:
            uh = float((up[valid] >= t / 100).mean()) * 100
            dh = float((dn[valid] <= -t / 100).mean()) * 100
            eh = float(((up[valid] >= t / 100) | (dn[valid] <= -t / 100)).mean()) * 100
            print(f"  {t:>6.2f} {uh:>9.1f} {dh:>9.1f} {eh:>9.1f}")
        print()

    _suggest(close, horizons)


def _suggest(close: pd.Series, horizons: list[int]) -> None:
    """Pick rungs by target UP-hit base rates, on the middle horizon."""
    H = horizons[len(horizons) // 2]
    up, _ = forward_extremes(close, H)
    valid = up.notna()
    targets = [0.80, 0.55, 0.35, 0.18, 0.07]   # desired UP-hit rate per rung
    ladder = []
    for tgt in targets:
        best = min(GRID, key=lambda t: abs((up[valid] >= t / 100).mean() - tgt))
        ladder.append(best)
    # enforce strictly increasing, unique
    seen, mono = set(), []
    cur = 0.0
    for t in ladder:
        t = max(t, cur + 0.05)
        while t in seen:
            t = round(t + 0.05, 2)
        mono.append(round(t, 2))
        seen.add(t)
        cur = t
    print(f"Suggested ladder @ horizon {H}min (UP-hit ~{[int(x*100) for x in targets]}%):")
    print(f"  THRESHOLDS = {mono}")
    print("  (sanity-check against the table above; tune by judgement)\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--horizons", default="60,120,180")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    df = load_klines(args.symbol, args.days)
    report(df, horizons)


if __name__ == "__main__":
    main()
