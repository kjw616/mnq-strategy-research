# Pure directional check for the 9/21 EMA-cross-at-open theory - decoupled
# from ema_cross_backtest.py's stop/target trade mechanics entirely. Same
# question validate_dashboard.py asks of the 713 Dashboard's bias/trend
# read, asked here of this signal instead: after a qualifying cross, does
# price actually keep moving in that direction more often than a coin flip
# would predict, at a range of horizons?
#
# Built directly on ema_cross_strategy.find_cross_candidates() - the exact
# same event definition (and confluence filters) ema_cross_backtest.py
# trades, so this measures the real signal, not a similar-but-different
# stand-in for it. Same three configs, same walk-forward discipline
# (in-sample 2023-2024 and out-of-sample 2025-present scored independently,
# no tuning against the OOS file).
#
# Baselines, matching validate_dashboard.py: 50% (does the signal carry any
# directional information at all) and "always long" (the fraction of
# candidate events that happened to drift up over the horizon regardless of
# which way the cross pointed - guards against a horizon during which the
# whole sample just trended up).
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.validate_ema_cross

import math

import pandas as pd

from .data import load_1m_bars
from .ema_cross_strategy import find_cross_candidates

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

HORIZON_HOURS = [0.5, 1.0, 2.0, 4.0, 6.5]  # 6.5h ~= cross time -> 16:00 ET close

CONFIGS = {
    "raw": dict(require_vwap_confluence=False, require_htf_bias_confluence=False),
    "vwap": dict(require_vwap_confluence=True, require_htf_bias_confluence=False),
    "vwap_htf": dict(require_vwap_confluence=True, require_htf_bias_confluence=True),
}


def _forward_return(df_1m: pd.DataFrame, t: pd.Timestamp, entry_price: float, horizon_hours: float) -> float | None:
    target = t + pd.Timedelta(hours=horizon_hours)
    pos = df_1m.index.searchsorted(target)
    if pos >= len(df_1m):
        return None
    return float(df_1m["close"].iloc[pos] - entry_price)


def _binom_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    """Normal-approximation two-sided p-value for hits/n vs p0 - avoids a
    scipy dependency for a straightforward large-n proportion test."""
    if n == 0:
        return float("nan")
    phat = hits / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan")
    z = (phat - p0) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def evaluate(df_1m: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate event, one forward-return column per horizon."""
    if candidates.empty:
        return candidates
    out = candidates[["time", "direction", "entry_price"]].copy()
    for h in HORIZON_HOURS:
        out[f"fwd_ret_{h}h"] = [
            _forward_return(df_1m, t, p, h) for t, p in zip(candidates["time"], candidates["entry_price"])
        ]
    return out


def summarize(evaluated: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} (n={len(evaluated)} candidate events) ===")
    if evaluated.empty:
        print("  No candidate events.")
        return

    for h in HORIZON_HOURS:
        col = f"fwd_ret_{h}h"
        sub = evaluated.dropna(subset=[col])
        n = len(sub)
        if n == 0:
            print(f"  {h:>4.1f}h  no samples (all past end of data)")
            continue
        sub_is_long = sub["direction"] == "long"
        up_frac = (sub[col] > 0).mean()
        correct = ((sub_is_long) & (sub[col] > 0)) | ((~sub_is_long) & (sub[col] < 0))
        hits = int(correct.sum())
        hit_rate = hits / n
        p = _binom_two_sided_p(hits, n, 0.5)
        avg_ret_long = sub.loc[sub_is_long, col].mean() if sub_is_long.any() else float("nan")
        avg_ret_short = sub.loc[~sub_is_long, col].mean() if (~sub_is_long).any() else float("nan")
        print(f"  {h:>4.1f}h  n={n:4d}  hit_rate={hit_rate*100:5.1f}%  vs 50%: p={p:.4f}  "
              f"'always long' baseline={up_frac*100:5.1f}%  "
              f"avg_fwd_ret(long)={avg_ret_long:+.2f}  avg_fwd_ret(short)={avg_ret_short:+.2f}")


def run(path: str, file_label: str) -> None:
    df = load_1m_bars(path)
    print(f"\n########## {file_label}: {path} ({df.index[0]} -> {df.index[-1]}, {len(df)} bars) ##########")

    for name, kwargs in CONFIGS.items():
        candidates = find_cross_candidates(df, **kwargs)
        evaluated = evaluate(df, candidates)
        summarize(evaluated, f"{file_label} - {name}")


if __name__ == "__main__":
    run(IN_SAMPLE_PATH, "IN-SAMPLE 2023-2024")
    run(OOS_PATH, "OUT-OF-SAMPLE 2025-present")
