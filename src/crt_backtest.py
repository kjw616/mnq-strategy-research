# Backtests CRT v21 (see crt_strategy.py) across 2 years of MNQ 1-minute
# data on several chart timeframes, so we can see which timeframe the
# strategy actually performs best on instead of assuming it from feel.
#
# Run it yourself:
#   source venv/bin/activate
#   python src/crt_backtest.py

import pandas as pd

from .crt_strategy import run_crt_strategy
from .data import load_1m_bars
from .report import add_hour_tags, summarize_crt_outcomes

DATA_PATH = "data/mnq_1m_2023_2025.parquet"

# (resample rule, minutes-per-bar) - 5m/4h included alongside the 15m-1H
# range the live indicator is restricted to, so the data can confirm or
# challenge that assumption rather than just take it on faith.
TIMEFRAMES = [
    ("5min", 5),
    ("15min", 15),
    ("30min", 30),
    ("1h", 60),
    ("4h", 240),
]


def run_all_timeframes(df_1m: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for rule, minutes in TIMEFRAMES:
        trades = run_crt_strategy(df_1m, chart_tf_rule=rule, chart_tf_minutes=minutes)
        if not trades.empty:
            trades["chart_tf"] = rule
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_tf_comparison(all_trades: pd.DataFrame) -> None:
    print("=" * 72)
    print("RESULTS BY CHART TIMEFRAME")
    print("=" * 72)
    for rule, _ in TIMEFRAMES:
        sub = all_trades[all_trades["chart_tf"] == rule]
        s = summarize_crt_outcomes(sub) if not sub.empty else {"n_decided": 0}
        if s["n_decided"] == 0:
            print(f"{rule:6s}  no trades")
            continue
        print(
            f"{rule:6s}  n={s['n_decided']:4d}  wins={s['n_win']:4d}  losses={s['n_loss']:4d}  "
            f"timeouts={s['n_timeout']:3d}  win_rate={s['win_rate']*100:5.1f}%  "
            f"avg_R={s['avg_r']:6.3f}  total_R={s['total_r']:8.2f}  PF={s['profit_factor']:.2f}"
        )


def print_hour_breakdown(all_trades: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("RESULTS BY ENTRY HOUR (America/New_York, all timeframes combined)")
    print("=" * 72)
    tagged = add_hour_tags(all_trades)
    for hour in range(24):
        sub = tagged[tagged["entry_hour"] == hour]
        s = summarize_crt_outcomes(sub) if not sub.empty else {"n_decided": 0}
        if s["n_decided"] == 0:
            continue
        print(
            f"{hour:02d}:00  n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  "
            f"avg_R={s['avg_r']:6.3f}  total_R={s['total_r']:8.2f}"
        )


if __name__ == "__main__":
    df_1m = load_1m_bars(DATA_PATH)
    print(f"Loaded {len(df_1m):,} 1-minute bars: {df_1m.index.min()} -> {df_1m.index.max()}")

    all_trades = run_all_timeframes(df_1m)
    all_trades.to_csv("data/crt_backtest_trades.csv", index=False)
    print(f"Total signals across all timeframes: {len(all_trades):,}\n")

    print_tf_comparison(all_trades)
    print_hour_breakdown(all_trades)
