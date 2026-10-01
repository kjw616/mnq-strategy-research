# Feature-mining pass over fib_golden_zone_strategy's LIMIT-entry trades -
# the one config in this whole exploration that's actually cleared
# breakeven with real margin and held up out-of-sample (sweep -> ATR-
# relative displacement leg -> retrace into the 61.8%-78.6% fib zone ->
# resting limit fill at 0.705, NO Rejection Block requirement - every way
# RB got added on top made results worse, see rb_5m_backtest.py /
# rb_5m_gated_strategy.py).
#
# Two things this reports:
#   1. How often it actually fires during the 9:30-11:00am ET session
#      specifically - trades/session-day, win rate, avg R, in 15-minute
#      slices across the window (not just the window as one lump).
#   2. Whether any feature already carried on each trade (direction,
#      liquidity_kind, clean_zone, stop distance, leg magnitude, day of
#      week) predicts a win - reported for in-sample AND OOS side by side,
#      same discipline as rb_5m_backtest.py's RB study: a bucket only
#      counts as a real pattern if it holds up in BOTH, not just one.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.fib_golden_zone_pattern_analysis

import numpy as np
import pandas as pd

from .data import load_1m_bars
from .fib_golden_zone_strategy import run_fib_golden_zone_strategy

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"
TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "11:00"
MIN_BUCKET_N = 15  # a bucket smaller than this isn't worth reading anything into


def add_features(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    local = pd.to_datetime(out["time"]).dt.tz_convert(TZ)
    out["local_time"] = local
    out["hour_et"] = local.dt.hour
    out["dow"] = local.dt.day_name()
    out["date"] = local.dt.date

    leg_mag = (out["leg_extreme"] - out["leg_start"]).abs()
    out["leg_magnitude"] = leg_mag
    out["risk_to_leg_ratio"] = out["stop_distance_points"] / leg_mag
    out["win"] = out["outcome"] == "win"
    return out


def in_session(trades: pd.DataFrame) -> pd.DataFrame:
    start, end = pd.Timestamp(SESSION_START).time(), pd.Timestamp(SESSION_END).time()
    t = trades["local_time"].dt.time
    return trades[(t >= start) & (t < end)]


def session_frequency(label: str, trades: pd.DataFrame) -> None:
    sess = in_session(trades)
    decided = sess[sess["outcome"].isin(["win", "loss"])]
    n_days_total = trades["date"].nunique()
    n_days_with_trade = decided["date"].nunique()
    print(f"--- {label}: how often it fires in the {SESSION_START}-{SESSION_END} ET session ---")
    print(f"  Trading days in data: {n_days_total}   |   days with >=1 session trade: {n_days_with_trade} "
          f"({n_days_with_trade/n_days_total*100:.0f}%)")
    print(f"  Session trades: {len(decided)}   |   avg per day-with-a-trade: {len(decided)/max(n_days_with_trade,1):.2f}")
    if decided.empty:
        return
    wr = decided["win"].mean()
    avg_r = decided["r_multiple"].mean()
    print(f"  Win rate: {wr*100:.1f}%   Avg R: {avg_r:+.3f}\n")

    print("  by 15-minute slice within the session:")
    bins = pd.date_range("09:30", "11:00", freq="15min").time
    labels = [f"{bins[i].strftime('%H:%M')}-{bins[i+1].strftime('%H:%M')}" for i in range(len(bins) - 1)]
    # pd.cut needs numeric bin edges, not datetime.time objects - use minutes since midnight instead.
    minutes = decided["local_time"].dt.hour * 60 + decided["local_time"].dt.minute
    bin_edges = [b.hour * 60 + b.minute for b in bins]
    slot = pd.cut(minutes, bins=bin_edges, labels=labels, right=False)
    g = decided.groupby(slot, observed=True).agg(n=("win", "size"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
    for k, r in g.iterrows():
        flag = "" if r["n"] >= MIN_BUCKET_N else "  (n<15, low confidence)"
        print(f"    {k:12s} n={int(r['n']):4d}  win_rate={r['win_rate']*100:5.1f}%  avg_R={r['avg_r']:+.3f}{flag}")
    print()


def feature_breakdown(label: str, col: str, in_trades: pd.DataFrame, oos_trades: pd.DataFrame, bins=None, bin_labels=None) -> None:
    print(f"  {label}:")
    for src_label, trades in [("in", in_trades), ("oos", oos_trades)]:
        decided = trades[trades["outcome"].isin(["win", "loss"])].copy()
        d = decided
        if bins is not None:
            d = d.copy()
            d[col] = pd.cut(d[col], bins=bins, labels=bin_labels)
        g = d.groupby(col, observed=True).agg(n=("win", "size"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
        row_strs = []
        for k, r in g.iterrows():
            if r["n"] >= MIN_BUCKET_N:
                row_strs.append(f"{str(k)}: n={int(r['n'])} wr={r['win_rate']*100:.1f}% avgR={r['avg_r']:+.2f}")
        print(f"    {src_label:4s} " + "  |  ".join(row_strs))
    print()


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    trades_in = add_features(run_fib_golden_zone_strategy(df_in, entry_mode="limit"))
    trades_oos = add_features(run_fib_golden_zone_strategy(df_oos, entry_mode="limit"))

    print("=" * 78)
    print("SESSION FREQUENCY")
    print("=" * 78)
    session_frequency("IN-SAMPLE 2023-2024", trades_in)
    session_frequency("OOS 2025-present", trades_oos)

    print("=" * 78)
    print(f"FEATURE BREAKDOWN, {SESSION_START}-{SESSION_END} ET SESSION ONLY (in-sample vs OOS)")
    print("=" * 78)
    sess_in, sess_oos = in_session(trades_in), in_session(trades_oos)
    feature_breakdown("direction", "direction", sess_in, sess_oos)
    feature_breakdown("liquidity_kind (session level vs 1m swing pivot)", "liquidity_kind", sess_in, sess_oos)
    feature_breakdown("clean_zone", "clean_zone", sess_in, sess_oos)
    feature_breakdown("day of week", "dow", sess_in, sess_oos)
    feature_breakdown("stop distance (points)", "stop_distance_points", sess_in, sess_oos,
                       bins=[0, 8, 15, 25, np.inf], bin_labels=["<8", "8-15", "15-25", "25+"])
    feature_breakdown("leg magnitude (points)", "leg_magnitude", sess_in, sess_oos,
                       bins=[0, 20, 40, 70, np.inf], bin_labels=["<20", "20-40", "40-70", "70+"])
    feature_breakdown("risk / leg magnitude ratio", "risk_to_leg_ratio", sess_in, sess_oos,
                       bins=[0, 0.28, 0.30, 0.32, np.inf], bin_labels=["<.28", ".28-.30", ".30-.32", ".32+"])

    print("=" * 78)
    print("FEATURE BREAKDOWN, FULL DAY (in-sample vs OOS) - for comparison")
    print("=" * 78)
    feature_breakdown("hour of day (ET)", "hour_et", trades_in, trades_oos)
