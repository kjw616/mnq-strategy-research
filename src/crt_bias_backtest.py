# Backtests the CRT-as-bias / 1-minute-entry architecture (see
# crt_bias_ltf_strategy.py) across several higher timeframes for the bias
# layer, entries always on 1-minute closes.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.crt_bias_backtest

import pandas as pd

from .crt_bias_ltf_strategy import run_crt_bias_ltf_strategy
from .data import load_1m_bars
from .report import add_hour_tags, summarize_crt_outcomes

DATA_PATH = "data/mnq_1m_2023_2025.parquet"
# 5min and 4H dropped - too little (4H) or too noisy/low-value (5min) a
# sample to keep tracking; focusing tuning effort on the three that matter.
BIAS_TIMEFRAMES = [("15min", 15), ("30min", 30), ("1h", 60)]


def run_all_bias_timeframes(df_1m: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for rule, minutes in BIAS_TIMEFRAMES:
        trades = run_crt_bias_ltf_strategy(df_1m, bias_tf_rule=rule, bias_tf_minutes=minutes)
        if not trades.empty:
            trades["bias_tf"] = rule
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_bias_tf_comparison(all_trades: pd.DataFrame) -> None:
    print("=" * 88)
    print("RESULTS BY BIAS TIMEFRAME (entries always on 1-minute)")
    print("=" * 88)
    for rule, _ in BIAS_TIMEFRAMES:
        sub = all_trades[all_trades["bias_tf"] == rule]
        s = summarize_crt_outcomes(sub) if not sub.empty else {"n_decided": 0}
        if s["n_decided"] == 0:
            print(f"{rule:6s}  no trades")
            continue
        gte4r = (sub[sub["outcome"].isin(["win", "loss"])]["r_multiple"] >= 4).sum()
        max_r = sub["r_multiple"].max()
        print(
            f"{rule:6s}  n={s['n_decided']:4d}  wins={s['n_win']:4d}  losses={s['n_loss']:4d}  "
            f"timeouts={s['n_timeout']:3d}  win_rate={s['win_rate']*100:5.1f}%  "
            f"avg_R={s['avg_r']:6.3f}  total_R={s['total_r']:8.2f}  PF={s['profit_factor']:.2f}  "
            f"trades>=4R={gte4r:3d}  max_R={max_r:6.2f}"
        )


def print_hour_breakdown(all_trades: pd.DataFrame) -> None:
    print()
    print("=" * 88)
    print("RESULTS BY ENTRY HOUR (America/New_York, all bias timeframes combined)")
    print("=" * 88)
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

    all_trades = run_all_bias_timeframes(df_1m)
    all_trades.to_csv("data/crt_bias_backtest_trades.csv", index=False)
    print(f"Total signals across all bias timeframes: {len(all_trades):,}\n")

    print_bias_tf_comparison(all_trades)
    print_hour_breakdown(all_trades)
