# Tests whether pre-cross market context predicts whether a 9/21 EMA cross
# actually follows through as a real trend - the standard trend-following
# read (cross up = long, cross down = short), NOT the fade theory in
# validate_ema_cross_fade.py. Five candidate filters, each capturing a
# different idea of "is this cross a real trend start, or just chop":
#   - adx                    trend strength already present (Wilder ADX 14)
#   - ema_slope_aligned      was the 21 EMA already sloping the same way as
#                            the new cross (a continuation), or is this a
#                            sharp reversal against its own recent slope
#   - atr_ratio              volatility expanding (>1) or contracting (<1)
#                            into the cross vs its own 100-bar average
#   - volume_ratio           recent (5-bar) volume vs its 60-bar average -
#                            participation behind the move
#   - bars_since_last_cross  how long since the last 9/21 cross of EITHER
#                            direction - small means the EMAs are actively
#                            whipsawing (chop), large means this cross
#                            follows a settled stretch
#
# Uses every cross across the full dataset (no session-open window, no
# one-per-day cap) to maximize sample size for this exploratory study -
# broader in scope than validate_ema_cross_fade.py's 9:30-11:00 AM theory.
#
# Buckets (Low/Mid/High tertiles) are cut on IN-SAMPLE quantile thresholds
# only, then the exact same thresholds are reapplied to OOS - so OOS never
# leaks into how "Low"/"High" get defined, same walk-forward discipline as
# the rest of this codebase.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_trend_predictors

import math

import numpy as np
import pandas as pd

from .crt_strategy import wilder_atr
from .data import load_1m_bars
from .ema_cross_strategy import ema_cross_signal
from .indicators import wilder_adx

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

HORIZON_HOURS = [1.0, 2.0, 4.0]

FAST_LEN = 9
SLOW_LEN = 21
SLOPE_LOOKBACK = 10
ADX_LEN = 14
ATR_LEN = 14
ATR_AVG_WINDOW = 100
VOL_FAST_WINDOW = 5
VOL_SLOW_WINDOW = 60

FEATURES = ["adx", "ema_slope_aligned", "atr_ratio", "volume_ratio", "bars_since_last_cross"]


def find_all_crosses_with_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Every 9/21 EMA cross plus five pre-cross context features (see module
    docstring), all computed from data available at or before the cross bar -
    no lookahead. `direction` follows the standard trend-following read."""
    sig = ema_cross_signal(df_1m["close"], FAST_LEN, SLOW_LEN)
    is_cross = (sig["cross_up"] | sig["cross_down"]).to_numpy()
    cross_up = sig["cross_up"].to_numpy()[is_cross]
    direction_sign = np.where(cross_up, 1.0, -1.0)

    atr = wilder_atr(df_1m, ATR_LEN)
    atr_avg = atr.rolling(ATR_AVG_WINDOW, min_periods=ATR_AVG_WINDOW).mean()
    atr_ratio = (atr / atr_avg).to_numpy()[is_cross]

    adx = wilder_adx(df_1m, ADX_LEN).to_numpy()[is_cross]

    ema_slope_raw = (sig["ema_slow"] - sig["ema_slow"].shift(SLOPE_LOOKBACK)) / atr
    ema_slope_aligned = ema_slope_raw.to_numpy()[is_cross] * direction_sign

    vol_fast = df_1m["volume"].rolling(VOL_FAST_WINDOW, min_periods=VOL_FAST_WINDOW).mean()
    vol_slow = df_1m["volume"].rolling(VOL_SLOW_WINDOW, min_periods=VOL_SLOW_WINDOW).mean()
    volume_ratio = (vol_fast / vol_slow).to_numpy()[is_cross]

    cross_bar_positions = np.flatnonzero(is_cross)
    bars_since_last_cross = np.diff(cross_bar_positions, prepend=-1).astype(float)
    if len(bars_since_last_cross):
        bars_since_last_cross[0] = np.nan  # no prior cross to measure from

    return pd.DataFrame({
        "time": df_1m.index[is_cross],
        "bar_index": cross_bar_positions,
        "direction": np.where(cross_up, "long", "short"),
        "entry_price": df_1m["close"].to_numpy()[is_cross],
        "atr_at_entry": atr.to_numpy()[is_cross],
        "adx": adx,
        "ema_slope_aligned": ema_slope_aligned,
        "atr_ratio": atr_ratio,
        "volume_ratio": volume_ratio,
        "bars_since_last_cross": bars_since_last_cross,
    })


def _forward_return(df_1m: pd.DataFrame, t: pd.Timestamp, entry_price: float, horizon_hours: float) -> float | None:
    target = t + pd.Timedelta(hours=horizon_hours)
    pos = df_1m.index.searchsorted(target)
    if pos >= len(df_1m):
        return None
    return float(df_1m["close"].iloc[pos] - entry_price)


def _binom_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    if n == 0:
        return float("nan")
    phat = hits / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan")
    z = (phat - p0) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def evaluate(df_1m: pd.DataFrame, crosses: pd.DataFrame) -> pd.DataFrame:
    if crosses.empty:
        return crosses
    out = crosses.copy()
    for h in HORIZON_HOURS:
        out[f"fwd_ret_{h}h"] = [
            _forward_return(df_1m, t, p, h) for t, p in zip(crosses["time"], crosses["entry_price"])
        ]
    return out


def _quantile_edges(values: pd.Series) -> np.ndarray:
    """Tertile cut points from a Series (NaN dropped) - defines Low/Mid/High."""
    return np.quantile(values.dropna(), [1 / 3, 2 / 3])


def _bucket(values: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = np.array(["Low", "Mid", "High"], dtype=object)
    idx = np.clip(np.digitize(values.to_numpy(), edges), 0, 2)
    out = pd.Series(labels[idx], index=values.index, dtype=object)
    out[values.isna()] = None
    return out


def _hit_stats(sub: pd.DataFrame, col: str) -> tuple[int, float, float]:
    is_long = sub["direction"] == "long"
    correct = (is_long & (sub[col] > 0)) | (~is_long & (sub[col] < 0))
    n = len(sub)
    hits = int(correct.sum())
    hit_rate = hits / n if n else float("nan")
    return n, hit_rate, _binom_two_sided_p(hits, n)


def report_feature(evaluated_in: pd.DataFrame, evaluated_oos: pd.DataFrame, feature: str) -> None:
    edges = _quantile_edges(evaluated_in[feature])
    print(f"\n--- {feature}  (in-sample tertile cuts: {edges[0]:.3f} / {edges[1]:.3f}) ---")

    for label, evaluated in [("IN-SAMPLE", evaluated_in), ("OOS", evaluated_oos)]:
        bucket = _bucket(evaluated[feature], edges)
        print(f"  {label}")
        for h in HORIZON_HOURS:
            col = f"fwd_ret_{h}h"
            parts = []
            for b in ["Low", "Mid", "High"]:
                sub = evaluated[(bucket == b) & evaluated[col].notna()]
                n, hr, p = _hit_stats(sub, col)
                flag = "*" if (not math.isnan(p) and p < 0.05) else " "
                parts.append(f"{b:4s} n={n:5d} hit={hr*100:5.1f}%{flag}(p={p:.3f})")
            print(f"    {h:>4.1f}h  " + "  ".join(parts))


def run(path: str, file_label: str) -> pd.DataFrame:
    df = load_1m_bars(path)
    print(f"\n########## {file_label}: {path} ({df.index[0]} -> {df.index[-1]}, {len(df)} bars) ##########")
    crosses = find_all_crosses_with_features(df)
    evaluated = evaluate(df, crosses)
    print(f"n={len(evaluated)} total crosses")
    return evaluated


if __name__ == "__main__":
    evaluated_in = run(IN_SAMPLE_PATH, "IN-SAMPLE 2023-2024")
    evaluated_oos = run(OOS_PATH, "OUT-OF-SAMPLE 2025-present")

    print("\n" + "=" * 90)
    print("FEATURE BREAKDOWN - does Low/Mid/High beat the raw ~50% hit rate, and hold up OOS?")
    print("(* marks p < 0.05 vs 50%; thresholds are fit on in-sample only, reused as-is on OOS)")
    print("=" * 90)
    for feature in FEATURES:
        report_feature(evaluated_in, evaluated_oos, feature)
