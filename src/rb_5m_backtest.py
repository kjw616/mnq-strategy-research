# Studies 5-minute Rejection Blocks (poi.check_rb - a candle that wicks
# through a confirmed 5m swing level and closes back on the other side,
# wick >= RB_WICK_RATIO x body, same definition and default ratio this
# codebase already uses for RBs on 4H/1H in dashboard_engine.py/strategy.py)
# on their own merits, before folding them into any larger sweep/fib-zone
# system.
#
# For every 5m RB found: walks forward on 1-MINUTE bars (finer than the RB
# itself formed on) to find the first retest of its own zone - a resting
# limit at rb_near (the candle's own body edge, "the point where the wick
# meets the body") fills the moment price returns to it; stop sits
# RB_STOP_BUFFER_POINTS (fixed, not ATR-scaled - always 5, per spec) beyond
# rb_far (that same candle's own wick extreme). Tags every RB - retested or
# not, win or loss - with descriptive features (wick/body ratio, RB range,
# direction, hour, day of week, bars until retest) so the report at the
# bottom can break results down by those features and show which
# characteristics actually predict a working retest, rather than assuming
# any RB that fits the shape is equally good.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.rb_5m_backtest

import numpy as np
import pandas as pd

from .data import load_1m_bars, resample_bars
from .poi import check_rb
from .structure import structure_bias

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

PIVOT_LENGTH = 5             # matches this repo's existing RB convention (dashboard_engine.PIVOT_LENGTH)
RB_WICK_RATIO = 2.0          # matches this repo's existing RB convention (dashboard_engine.RB_WICK_RATIO)
RB_STOP_BUFFER_POINTS = 5.0  # fixed points beyond the RB's own wick extreme, not ATR-scaled - the standard, per spec
MAX_RETEST_HOURS = 4.0       # an RB nobody has come back to within this long is treated as stale/no longer live context
MAX_RESOLUTION_HOURS = 24.0  # cap on how long a filled trade is tracked before it's called a timeout
TARGET_R_MULTIPLES = [2.0, 4.0, 8.0]
TZ = "America/New_York"


def find_5m_rbs(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Every 5-minute RB (bull or bear), one row each, with its own zone
    [rb_near, rb_far] and descriptive features. Non-repainting: swing_high/
    swing_low come from structure_bias's confirmed-pivot timing, the same
    source every other RB use in this codebase (dashboard_engine.py,
    strategy.py) already relies on."""
    df_5m = resample_bars(df_1m, "5min")
    struct = structure_bias(df_5m, PIVOT_LENGTH)
    rb = check_rb(df_5m, struct["swing_high"], struct["swing_low"], RB_WICK_RATIO)

    o, c, h, l = df_5m["open"], df_5m["close"], df_5m["high"], df_5m["low"]
    body = (c - o).abs()
    rng = h - l

    rows = []
    for direction, flag_col, near_col, far_col in [
        ("bull", "bull_rb", "bull_top", "bull_bottom"),
        ("bear", "bear_rb", "bear_bottom", "bear_top"),
    ]:
        flags = rb[flag_col]
        for t in flags.index[flags.fillna(False)]:
            near, far = rb.loc[t, near_col], rb.loc[t, far_col]
            b = body.loc[t]
            wick = abs(near - far)
            local = t.tz_convert(TZ)
            rows.append({
                "time": t, "direction": direction, "rb_near": near, "rb_far": far,
                "wick_points": wick, "body_points": b,
                "wick_body_ratio": (wick / b) if b > 0 else np.inf,
                "range_points": rng.loc[t],
                "hour_et": local.hour, "dow": local.day_name(),
            })
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def simulate_retests(rbs: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """For every RB, finds its first 1-minute retest (if any, within
    MAX_RETEST_HOURS) and resolves a trade at each of TARGET_R_MULTIPLES
    (entry=rb_near, stop=rb_far -/+ RB_STOP_BUFFER_POINTS) against 1-minute
    bars, capped at MAX_RESOLUTION_HOURS. Same worst-case ordering
    convention as every other backtest in this repo: if a single bar
    touches both stop and target, stop wins."""
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)
    retest_span = int(MAX_RETEST_HOURS * 60)
    resolve_span = int(MAX_RESOLUTION_HOURS * 60)

    out_rows = []
    for row in rbs.itertuples():
        formed_at = row.time + pd.Timedelta(minutes=5)  # this 5m bar's close - when the RB becomes knowable
        start_i = idx.searchsorted(formed_at, side="left")
        end_i = min(n, start_i + retest_span)
        result = row._asdict()
        del result["Index"]

        if start_i >= end_i:
            result.update(retested=False, bars_to_retest=np.nan)
            out_rows.append(result)
            continue

        is_long = row.direction == "bull"
        near, far = row.rb_near, row.rb_far
        touch_mask = (lows[start_i:end_i] <= near) if is_long else (highs[start_i:end_i] >= near)
        if not touch_mask.any():
            result.update(retested=False, bars_to_retest=np.nan)
            out_rows.append(result)
            continue

        rel = int(np.argmax(touch_mask))
        entry_i = start_i + rel
        entry_price = near
        stop_price = (far - RB_STOP_BUFFER_POINTS) if is_long else (far + RB_STOP_BUFFER_POINTS)
        risk = abs(entry_price - stop_price)
        result.update(
            retested=True, bars_to_retest=rel, entry_time=idx[entry_i],
            entry_price=entry_price, stop_price=stop_price, risk_points=risk,
        )
        if risk <= 0:
            out_rows.append(result)
            continue

        res_end = min(n, entry_i + resolve_span)
        seg_low = lows[entry_i:res_end]
        seg_high = highs[entry_i:res_end]
        hit_stop_mask = (seg_low <= stop_price) if is_long else (seg_high >= stop_price)
        stop_rel = int(np.argmax(hit_stop_mask)) if hit_stop_mask.any() else None

        for rmult in TARGET_R_MULTIPLES:
            target_price = entry_price + risk * rmult if is_long else entry_price - risk * rmult
            hit_target_mask = (seg_high >= target_price) if is_long else (seg_low <= target_price)
            target_rel = int(np.argmax(hit_target_mask)) if hit_target_mask.any() else None

            if stop_rel is not None and (target_rel is None or stop_rel <= target_rel):
                outcome, r_achieved = "loss", -1.0
            elif target_rel is not None:
                outcome, r_achieved = "win", rmult
            else:
                outcome, r_achieved = "timeout", np.nan
            result[f"outcome_r{rmult:g}"] = outcome
            result[f"r_multiple_r{rmult:g}"] = r_achieved

        out_rows.append(result)

    return pd.DataFrame(out_rows)


def report(label: str, trades: pd.DataFrame) -> None:
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"Total 5m RBs found: {len(trades)}")
    retested = trades[trades["retested"]]
    print(f"Retested within {MAX_RETEST_HOURS:.0f}h: {len(retested)} ({len(retested)/len(trades)*100:.1f}%)")
    if retested.empty:
        print("No retests to analyze.\n")
        return
    print(f"Avg risk (stop distance): {retested['risk_points'].mean():.1f} pts  "
          f"(90th pct {retested['risk_points'].quantile(0.90):.1f} pts)")
    print(f"Avg bars-to-retest (1m bars): {retested['bars_to_retest'].mean():.1f}\n")

    for rmult in TARGET_R_MULTIPLES:
        col = f"outcome_r{rmult:g}"
        decided = retested[retested[col].isin(["win", "loss"])]
        n = len(decided)
        if n == 0:
            continue
        wr = (decided[col] == "win").mean()
        breakeven = 1 / (1 + rmult)
        rcol = f"r_multiple_r{rmult:g}"
        pf_num = decided.loc[decided[col] == "win", rcol].sum()
        pf_den = abs(decided.loc[decided[col] == "loss", rcol].sum())
        pf = pf_num / pf_den if pf_den else float("inf")
        avg_r = decided[rcol].mean()
        print(f"  target={rmult:g}R   n={n:5d}  win_rate={wr*100:5.1f}%  breakeven={breakeven*100:4.1f}%  "
              f"avg_R={avg_r:+.3f}  PF={pf:.2f}")

    print("\n--- breakdown at target=8R ---")
    decided = retested[retested["outcome_r8"].isin(["win", "loss"])].copy()
    decided["win"] = decided["outcome_r8"] == "win"

    def bucket_report(col: str, bins=None, labels=None):
        d = decided.copy()
        if bins is not None:
            d[col] = pd.cut(d[col], bins=bins, labels=labels)
        g = d.groupby(col, observed=True)["win"].agg(["mean", "count"])
        for k, r in g.iterrows():
            if r["count"] >= 15:
                print(f"    {str(k):18s} n={int(r['count']):5d}  win_rate={r['mean']*100:5.1f}%")

    print("  by direction:")
    bucket_report("direction")
    print("  by wick/body ratio:")
    bucket_report("wick_body_ratio", bins=[2, 2.5, 3, 4, 6, np.inf], labels=["2-2.5x", "2.5-3x", "3-4x", "4-6x", "6x+"])
    print("  by RB range (points):")
    bucket_report("range_points", bins=[0, 10, 20, 40, np.inf], labels=["<10", "10-20", "20-40", "40+"])
    print("  by hour (ET):")
    bucket_report("hour_et")
    print("  by day of week:")
    bucket_report("dow")
    print()


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    rbs_in = find_5m_rbs(df_in)
    rbs_oos = find_5m_rbs(df_oos)

    trades_in = simulate_retests(rbs_in, df_in)
    trades_oos = simulate_retests(rbs_oos, df_oos)

    trades_in.to_csv("data/rb_5m_in_sample.csv", index=False)
    trades_oos.to_csv("data/rb_5m_oos.csv", index=False)

    report("IN-SAMPLE: 2023-2024 - 5m RBs, 1m retest entry", trades_in)
    report("OUT-OF-SAMPLE: 2025-present - 5m RBs, 1m retest entry", trades_oos)
