"""models/consensus.py — multi-strategy BTC consensus predictor (24/7).

Inference-only: loads 10 pre-trained XGBoost classifiers (UP/DOWN × a 5-rung
%-move ladder) on a forward horizon. Each classifier predicts whether price will
move by its target percentage within the next HORIZON_MIN minutes. Models are
trained offline and shipped in / restored to the model dir; this module never
fits them.

Three nested consensus strategies fire when *every* model in their threshold
subset (one direction) predicts probability above prob_threshold:

    3of5  →  thresholds[:3]
    4of5  →  thresholds[:4]
    5of5  →  thresholds[:5]

Subsets are nested, so a 5of5 fire implies 4of5 and 3of5 fire the same way.

This is the BTC/Binance port of the original NIFTY engine. Key differences:
  * 24/7 — no trading-day boundaries. All rolling windows are continuous and
    forward labels are bounded by *timestamp*, not row position, so collector
    gaps don't corrupt returns or targets.
  * Volatility gate uses ATM implied vol (atm_iv), not India VIX.
  * Features come from core.features.FEATURE_COLUMNS (option-chain analytics)
    plus engineered price/time/change features added here over the buffer.

Alert-only by design: it computes signals; it does not place orders.
"""
from __future__ import annotations
import os
import json
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from core.features import FEATURE_COLUMNS

log = logging.getLogger(__name__)

# 5-rung %-move ladder + forward horizon. Calibrated for BTC (~1.87% daily vol);
# see scripts/backtest_thresholds.py. Overridable via env for re-tuning.
THRESHOLDS = [float(x) for x in
              os.getenv('CONSENSUS_THRESHOLDS', '0.15,0.30,0.50,0.75,1.10').split(',')]
HORIZON_MIN = int(os.getenv('CONSENSUS_HORIZON_MIN', '120'))

# Nested consensus subsets, derived from the ladder.
STRATEGIES = {
    '3of5': THRESHOLDS[:3],
    '4of5': THRESHOLDS[:4],
    '5of5': THRESHOLDS[:5],
}
DEFAULT_MODEL_DIR = Path(os.getenv('CONSENSUS_MODEL_DIR', '/data/consensus_models'))

# ---------------------------------------------------------------------------
# Feature engineering — continuous (no day grouping)
# ---------------------------------------------------------------------------
# Engineered price features over a continuous 1-min spot series.
ENG_PRICE = ['ret_1m', 'ret_5m', 'ret_15m', 'ret_30m',
             'vol_15m', 'vol_30m', 'hi_15m', 'lo_15m']
# Cyclical UTC time-of-day + day-of-week (BTC has session/weekend seasonality).
ENG_TIME = ['tod_sin', 'tod_cos', 'dow_sin', 'dow_cos']
# Momentum of the key option-chain analytics (mirrors NIFTY WorkBook's "Change"
# columns: OI PCR Change, VIX Change, …). 5-bar diffs.
CHANGE_BASE = ['atm_iv', 'oi_pcr', 'oi_pcr_2pct', 'oi_pcr_4pct', 'oi_pcr_10pct',
               'vol_pcr', 'net_dir_score', 'perp_oi']
ENG_CHANGE = [f'{c}_chg5' for c in CHANGE_BASE]
ENG_FEATURES = ENG_PRICE + ENG_TIME + ENG_CHANGE

# Default training feature set: option-chain levels + everything engineered.
ALL_FEATURES = FEATURE_COLUMNS + ENG_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered price/time/change features over a continuous series.

    Requires columns: 'ts' (epoch ms), 'spot', plus the option-chain features in
    CHANGE_BASE. Caller need not pre-sort; we sort by ts.
    """
    df = df.copy().sort_values('ts').reset_index(drop=True)
    dt = pd.to_datetime(df['ts'], unit='ms', utc=True)

    # Cyclical time encodings (no "market open" exists 24/7).
    mod = dt.dt.hour * 60 + dt.dt.minute          # minute of UTC day, 0..1439
    df['tod_sin'] = np.sin(2 * np.pi * mod / 1440)
    df['tod_cos'] = np.cos(2 * np.pi * mod / 1440)
    dow = dt.dt.dayofweek                          # 0..6
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Continuous price features (no groupby — series is one unbroken stream).
    s = df['spot']
    df['ret_1m'] = s.pct_change(1)
    df['ret_5m'] = s.pct_change(5)
    df['ret_15m'] = s.pct_change(15)
    df['ret_30m'] = s.pct_change(30)
    df['vol_15m'] = s.pct_change().rolling(15).std()
    df['vol_30m'] = s.pct_change().rolling(30).std()
    df['hi_15m'] = s.rolling(15).max() / s - 1
    df['lo_15m'] = s.rolling(15).min() / s - 1

    # Option-analytics momentum (5-bar change).
    for c in CHANGE_BASE:
        df[f'{c}_chg5'] = df[c].diff(5) if c in df.columns else np.nan
    return df


# ---------------------------------------------------------------------------
# Model bundle & predictor
# ---------------------------------------------------------------------------
@dataclass
class ConsensusModel:
    """Bundle of 10 XGBoost classifiers and the metadata needed to use them."""
    feature_names: list[str]
    models: dict[str, xgb.XGBClassifier]   # key: 'up_0.3', 'down_1.1', ...
    trained_at: str
    train_rows: int
    train_dates: list[str]
    thresholds: list[float] = field(default_factory=lambda: list(THRESHOLDS))
    horizon_min: int = HORIZON_MIN

    @classmethod
    def load(cls, model_dir: Path) -> "ConsensusModel":
        meta = json.loads((model_dir / "meta.json").read_text())
        thr = meta.get('thresholds', list(THRESHOLDS))
        models = {}
        for d in ('up', 'down'):
            for t in thr:
                k = f"{d}_{t}"
                m = xgb.XGBClassifier()
                m.load_model(model_dir / f"{k}.json")
                models[k] = m
        return cls(feature_names=meta['feature_names'], models=models,
                   trained_at=meta['trained_at'], train_rows=meta['train_rows'],
                   train_dates=meta['train_dates'], thresholds=thr,
                   horizon_min=meta.get('horizon_min', HORIZON_MIN))


# ---------------------------------------------------------------------------
# Live signal computation
# ---------------------------------------------------------------------------
@dataclass
class SignalResult:
    """Result of evaluating one live tick."""
    timestamp: str
    spot: float
    atm_iv: float
    fired: bool
    direction: Optional[str]                # 'UP' or 'DOWN' or None
    raw_probabilities: dict[str, float]
    rank_pct: dict[str, float]
    up_votes: int = 0
    dn_votes: int = 0
    up_confidence: float = 0.0
    down_confidence: float = 0.0
    conf_spread: float = 0.0
    strategies: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ConsensusPredictor:
    """Holds the 10 models + a rolling buffer of recent probabilities.

    The trailing buffer stores each model's recent probability outputs so we can
    report where the current probability sits as a percentile rank (`rank_pct`)
    and derive an UP/DOWN confidence score. Firing itself uses raw probabilities.
    """

    def __init__(self, model: ConsensusModel, trail_size: int = 1875):
        self.model = model
        self._proba_buffers: dict[str, deque] = {
            k: deque(maxlen=trail_size) for k in model.models.keys()
        }
        self.trail_size = trail_size

    def predict_one(self, x: np.ndarray) -> dict[str, float]:
        x_2d = x.reshape(1, -1)
        return {k: float(m.predict_proba(x_2d)[0, 1])
                for k, m in self.model.models.items()}

    def evaluate(self,
                 features: dict[str, float],
                 spot: float,
                 atm_iv: float,
                 timestamp: Optional[str] = None,
                 *,
                 prob_threshold: float = 0.50,
                 iv_floor: float = 0.0) -> SignalResult:
        """Evaluate one live observation against the 3of5 / 4of5 / 5of5 strategies.

        A strategy fires for a direction when every model in its threshold subset
        predicts probability > prob_threshold and ATM IV clears the floor.
        """
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        reasons: list[str] = []
        thr = self.model.thresholds
        subsets = {'3of5': thr[:3], '4of5': thr[:4], '5of5': thr[:5]}

        try:
            x = np.array([features[f] for f in self.model.feature_names],
                         dtype=np.float32)
        except KeyError as e:
            return SignalResult(timestamp=timestamp, spot=spot, atm_iv=atm_iv,
                                fired=False, direction=None,
                                raw_probabilities={}, rank_pct={},
                                reasons=[f"missing_feature:{e.args[0]}"])

        if not np.isfinite(x).all():
            bad = [f for f, v in zip(self.model.feature_names, x)
                   if not np.isfinite(v)]
            return SignalResult(timestamp=timestamp, spot=spot, atm_iv=atm_iv,
                                fired=False, direction=None,
                                raw_probabilities={}, rank_pct={},
                                reasons=[f"non_finite_features:{','.join(bad[:3])}"])

        probs = self.predict_one(x)

        ranks: dict[str, float] = {}
        for k, p in probs.items():
            buf = self._proba_buffers[k]
            buf.append(p)
            ranks[k] = sum(1 for v in buf if v <= p) / len(buf)

        def pp(key: str) -> float:
            """Prob for a model, or 0.0 if it wasn't trained (degenerate labels) —
            a missing model never votes YES and never lets its subset fire."""
            return probs.get(key, 0.0)

        def mean_rank(keys: list[str]) -> float:
            vals = [ranks[k] for k in keys if k in ranks]
            return round(sum(vals) / len(vals) * 100, 1) if vals else 0.0

        up_confidence = mean_rank([f"up_{t}" for t in thr])
        down_confidence = mean_rank([f"down_{t}" for t in thr])
        conf_spread = round(abs(up_confidence - down_confidence), 1)

        up_votes = sum(1 for t in thr if pp(f"up_{t}") > prob_threshold)
        dn_votes = sum(1 for t in thr if pp(f"down_{t}") > prob_threshold)

        # Volatility gate — blocks every strategy when ATM IV is below the floor.
        iv_ok = atm_iv >= iv_floor
        if not iv_ok:
            reasons.append(f"iv_gate:{atm_iv:.3f}<{iv_floor:.3f}")

        strategies: dict[str, dict] = {}
        for name, subset in subsets.items():
            up_fire = iv_ok and all(pp(f"up_{t}") > prob_threshold for t in subset)
            dn_fire = iv_ok and all(pp(f"down_{t}") > prob_threshold for t in subset)
            sdir = 'UP' if up_fire else ('DOWN' if dn_fire else None)
            sub_up_conf = mean_rank([f"up_{t}" for t in subset])
            sub_dn_conf = mean_rank([f"down_{t}" for t in subset])
            sub_conf = (sub_up_conf if sdir == 'UP'
                        else sub_dn_conf if sdir == 'DOWN' else 0.0)
            sub_spread = round(abs(sub_up_conf - sub_dn_conf), 1)
            strategies[name] = {'up': up_fire, 'down': dn_fire,
                                'fired': up_fire or dn_fire, 'direction': sdir,
                                'up_confidence': sub_up_conf,
                                'down_confidence': sub_dn_conf,
                                'confidence': sub_conf,
                                'conf_spread': sub_spread}

        fired = any(s['fired'] for s in strategies.values())
        if any(s['up'] for s in strategies.values()):
            direction = 'UP'
        elif any(s['down'] for s in strategies.values()):
            direction = 'DOWN'
        else:
            direction = None

        if not fired and iv_ok:
            reasons.append(f"no_consensus:up={up_votes}/5,dn={dn_votes}/5")

        return SignalResult(
            timestamp=timestamp, spot=spot, atm_iv=atm_iv,
            fired=fired, direction=direction,
            raw_probabilities=probs, rank_pct=ranks,
            up_votes=up_votes, dn_votes=dn_votes,
            up_confidence=up_confidence, down_confidence=down_confidence,
            conf_spread=conf_spread, strategies=strategies, reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def latest_model_dir(base: Path = DEFAULT_MODEL_DIR) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    return candidates[0] if candidates else None
