# Two things: (1) how the fib golden zone strategy performs specifically in
# the 10:00-10:30am ET window, at the same depth as the earlier 9:30-11:00
# report, and (2) whether that window's underlying PRICE ACTION - not just
# this strategy's results in it - actually looks different from the rest of
# the day. (2) is independent of the strategy entirely: it's computed
# straight off the 1-minute OHLCV data, so it can explain (or fail to
# explain) *why* a window performs differently, rather than just reporting
# that it does.
#
# Price-action metrics per HALF_HOUR bucket, averaged across every trading
# day in each sample:
#   net_move     - |close at bucket end - open at bucket start|, i.e. how
#                  far price actually ended up traveling in that half hour.
#   path_length  - sum of |1-minute close-to-close change| across the same
#                  half hour, i.e. the total distance price covered
#                  wiggling back and forth to get there.
#   efficiency   - net_move / path_length (Kaufman-style efficiency ratio).
#                  1.0 = a dead straight line in one direction, near 0 =
#                  pure chop that round-trips back to where it started.
#                  This is the number that actually explains whether a
#                  sweep-and-continue setup like ours has clean displacement
#                  to work with in that window, or just noise.
#   avg_range/avg_volume - per-1-minute-bar averages, the usual volatility/
#                  participation proxies.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.fib_golden_zone_10am_window_analysis

import numpy as np
import pandas as pd

from .combined_analysis import longest_streak
from .data import load_1m_bars
from .fib_golden_zone_strategy import run_fib_golden_zone_strategy
from .report import summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"
TZ = "America/New_York"
WINDOW_START = "10:00"
WINDOW_END = "10:30"
BUCKET_MINUTES = 30


def window_only(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    local = pd.to_datetime(trades["time"]).dt.tz_convert(TZ)
    t = local.dt.time
    s, e = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    return trades[(t >= s) & (t < e)]


def strategy_report(label: str, trades: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    decided = trades[trades["outcome"].isin(["win", "loss"])].sort_values("exit_time")
    if decided.empty:
        print("No trades.")
        return
    stats = summarize_outcomes(decided)
    wins = decided[decided["outcome"] == "win"]
    losses = decided[decided["outcome"] == "loss"]
    n_days = (decided["time"].max() - decided["time"].min()).days or 1
    print(f"Trades: {stats['n_decided']}  over {n_days} days (~{stats['n_decided']/(n_days/7):.2f}/week)")
    print(f"Win rate: {stats['win_rate']*100:.1f}%   Avg R: {stats['avg_r']:+.3f}   PF: {stats['profit_factor']:.2f}")
    print(f"Avg win: {wins['r_multiple'].mean():+.2f}R   Avg loss: {losses['r_multiple'].mean():+.2f}R")
    print(f"Longest win streak: {longest_streak(decided['outcome'], 'win')}   "
          f"Longest loss streak: {longest_streak(decided['outcome'], 'loss')}")
    print(f"Avg stop distance: {decided['stop_distance_points'].mean():.1f} pts")


def price_action_by_bucket(df_1m: pd.DataFrame, bucket_minutes: int = BUCKET_MINUTES) -> pd.DataFrame:
    local = df_1m.index.tz_convert(TZ)
    minute_of_day = local.hour * 60 + local.minute
    bucket_start_min = (minute_of_day // bucket_minutes) * bucket_minutes

    g = pd.DataFrame({
        "date": local.normalize(), "bucket": bucket_start_min,
        "open": df_1m["open"].to_numpy(), "high": df_1m["high"].to_numpy(),
        "low": df_1m["low"].to_numpy(), "close": df_1m["close"].to_numpy(),
        "volume": df_1m["volume"].to_numpy(),
        "bar_range": (df_1m["high"] - df_1m["low"]).to_numpy(),
    })
    grouped = g.groupby(["date", "bucket"])
    per_window = grouped.agg(
        open=("open", "first"), close=("close", "last"),
        avg_bar_range=("bar_range", "mean"), avg_volume=("volume", "mean"),
    )
    per_window["path_length"] = grouped["close"].apply(lambda s: s.diff().abs().sum())
    per_window["net_move"] = (per_window["close"] - per_window["open"]).abs()
    per_window["efficiency"] = per_window["net_move"] / per_window["path_length"].replace(0, np.nan)

    summary = per_window.groupby("bucket").agg(
        n_days=("open", "size"),
        avg_net_move=("net_move", "mean"),
        avg_path_length=("path_length", "mean"),
        avg_efficiency=("efficiency", "mean"),
        avg_bar_range=("avg_bar_range", "mean"),
        avg_volume=("avg_volume", "mean"),
    )
    return summary


def print_price_action(label: str, summary: pd.DataFrame) -> None:
    print(f"\n{'='*88}\n{label} - price action by {BUCKET_MINUTES}-min bucket (ET)\n{'='*88}")
    print(f"{'bucket':>12}{'n_days':>8}{'net_move':>11}{'path_len':>11}{'efficiency':>12}{'avg_range':>11}{'avg_vol':>10}")
    ranked_efficiency = summary["avg_efficiency"].rank(ascending=False)
    for bucket_start, r in summary.sort_index().iterrows():
        h, m = divmod(int(bucket_start), 60)
        label_str = f"{h:02d}:{m:02d}"
        flag = "  <==" if h == 10 and m == 0 else ""
        rank = int(ranked_efficiency.loc[bucket_start])
        print(f"{label_str:>12}{int(r['n_days']):>8}{r['avg_net_move']:>10.1f}p{r['avg_path_length']:>10.1f}p"
              f"{r['avg_efficiency']:>11.3f}{r['avg_bar_range']:>10.2f}p{r['avg_volume']:>10.0f}"
              f"  (eff.rank {rank}/{len(summary)}){flag}")


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    trades_in = run_fib_golden_zone_strategy(df_in, entry_mode="limit")
    trades_oos = run_fib_golden_zone_strategy(df_oos, entry_mode="limit")

    strategy_report(f"IN-SAMPLE, {WINDOW_START}-{WINDOW_END} ET only", window_only(trades_in, WINDOW_START, WINDOW_END))
    strategy_report(f"OOS, {WINDOW_START}-{WINDOW_END} ET only", window_only(trades_oos, WINDOW_START, WINDOW_END))

    print_price_action("IN-SAMPLE 2023-2024", price_action_by_bucket(df_in))
    print_price_action("OOS 2025-present", price_action_by_bucket(df_oos))
