# Tests a fade/reversal read of the 9/21 EMA cross, the opposite of the
# usual trend-following interpretation: a cross to the downside (9 EMA
# crosses below 21 EMA) is scored as a BULLISH signal, and a cross to the
# upside as BEARISH. Detects every qualifying cross that falls inside the
# 9:30-11:00 AM ET window each day - no one-trade-per-session cap, unlike
# ema_cross_strategy.find_cross_candidates (which stops at the first cross
# in a narrower open-only window). This measures the full population of
# 9/21 crosses during the NY morning session instead.
#
# VWAP is checked as a confluence filter using this codebase's standard
# convention (price already on the signal's expected side of VWAP at the
# moment of the cross), same as ema_cross_strategy's require_vwap_confluence
# - just applied to the faded (expected) direction rather than the raw
# cross direction. Reported as a breakdown against the unfiltered set.
#
# Same directional-accuracy design as validate_ema_cross.py: no stop/target
# mechanics, just "did price actually move the called direction" at a range
# of forward horizons, with a 50% baseline and an "always long" baseline
# (guards against a horizon where the whole sample just trended one way).
# Same walk-forward discipline - in-sample 2023-2024 and out-of-sample
# 2025-present scored independently, never tuned against the OOS file.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.validate_ema_cross_fade

import math

import pandas as pd

from .data import load_1m_bars
from .ema_cross_strategy import ema_cross_signal
from .indicators import session_vwap

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

HORIZON_HOURS = [0.5, 1.0, 2.0, 4.0, 6.5]

FAST_LEN = 9
SLOW_LEN = 21

WINDOW_START_HOUR = 9
WINDOW_START_MINUTE = 30
WINDOW_END_HOUR = 11
WINDOW_END_MINUTE = 0
TZ = "America/New_York"


def find_all_crosses(
    df_1m: pd.DataFrame,
    fast_len: int = FAST_LEN,
    slow_len: int = SLOW_LEN,
    tz: str = TZ,
) -> pd.DataFrame:
    """Every 9/21 EMA cross that falls inside the 9:30-11:00 AM ET window
    each day, direction reinterpreted per the fade rule under test:
    cross_down -> expected bullish ('long'), cross_up -> expected bearish
    ('short'). `vwap_confluence` flags whether price was already on the
    expected direction's side of VWAP at the cross."""
    sig = ema_cross_signal(df_1m["close"], fast_len, slow_len)
    vwap = session_vwap(df_1m)

    local_time = df_1m.index.tz_convert(tz)
    minutes_of_day = local_time.hour * 60 + local_time.minute
    window_start = WINDOW_START_HOUR * 60 + WINDOW_START_MINUTE
    window_end = WINDOW_END_HOUR * 60 + WINDOW_END_MINUTE
    in_window = (minutes_of_day >= window_start) & (minutes_of_day < window_end)

    is_cross = ((sig["cross_up"] | sig["cross_down"]).to_numpy()) & in_window
    cross_up = sig["cross_up"].to_numpy()[is_cross]

    out = pd.DataFrame({
        "time": df_1m.index[is_cross],
        "raw_cross": ["cross_up" if up else "cross_down" for up in cross_up],
        "expected_direction": ["short" if up else "long" for up in cross_up],
        "entry_price": df_1m["close"].to_numpy()[is_cross],
        "vwap": vwap.to_numpy()[is_cross],
    })
    is_long = out["expected_direction"] == "long"
    above_vwap = out["entry_price"] > out["vwap"]
    out["vwap_confluence"] = (is_long & above_vwap) | (~is_long & ~above_vwap)
    return out


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


def evaluate(df_1m: pd.DataFrame, crosses: pd.DataFrame) -> pd.DataFrame:
    """One row per cross event, one forward-return column per horizon."""
    if crosses.empty:
        return crosses
    out = crosses.copy()
    for h in HORIZON_HOURS:
        out[f"fwd_ret_{h}h"] = [
            _forward_return(df_1m, t, p, h) for t, p in zip(crosses["time"], crosses["entry_price"])
        ]
    return out


def summarize(evaluated: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} (n={len(evaluated)} cross events) ===")
    if evaluated.empty:
        print("  No cross events.")
        return

    for h in HORIZON_HOURS:
        col = f"fwd_ret_{h}h"
        sub = evaluated.dropna(subset=[col])
        n = len(sub)
        if n == 0:
            print(f"  {h:>4.1f}h  no samples (all past end of data)")
            continue
        sub_is_long = sub["expected_direction"] == "long"
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

    crosses = find_all_crosses(df)
    evaluated = evaluate(df, crosses)

    summarize(evaluated, f"{file_label} - all crosses (no VWAP filter)")
    summarize(evaluated[evaluated["vwap_confluence"]], f"{file_label} - VWAP-confluent crosses only")


if __name__ == "__main__":
    run(IN_SAMPLE_PATH, "IN-SAMPLE 2023-2024")
    run(OOS_PATH, "OUT-OF-SAMPLE 2025-present")
