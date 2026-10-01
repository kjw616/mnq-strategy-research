# Tests whether EMA fast/slow alignment adds value as a CONFLUENCE FILTER
# on top of fib_golden_zone_strategy's limit-entry trades - the one system
# in this whole exploration that's actually cleared breakeven and held up
# OOS. Different question from ema_cross_pair_grid_search.py (which asked
# "does the cross itself predict forward price, standalone, at the open") -
# this asks "if you only take a fib-zone trade when the EMA state already
# agrees with its direction, does that raise the win rate," which is closer
# to how confluence actually gets used in practice: not as its own signal,
# but as a filter on a setup that already has other things going for it.
#
# "Aligned" here means the STATE at entry (fast EMA above/below slow EMA on
# the 1-minute chart, at the trade's own entry bar), not a fresh cross event
# - a fib-zone trade can fire well after the last actual crossover, so
# testing for a recent cross specifically would throw away most of the
# sample for no clear reason. Same six pairs as the earlier grid search
# (the 5 in-sample "top" pairs + the current 9/21 default), same walk-
# forward discipline (in-sample and OOS reported independently, nothing
# chosen using OOS).
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.fib_golden_zone_ema_confluence

import numpy as np
import pandas as pd

from .data import load_1m_bars
from .fib_golden_zone_strategy import run_fib_golden_zone_strategy
from .report import summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

PAIRS = [(9, 21), (12, 21), (9, 26), (12, 34), (8, 26), (12, 26)]
MIN_N = 20


def tag_ema_alignment(trades: pd.DataFrame, df_1m: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """True where the EMA state at each trade's own entry bar already
    agrees with that trade's direction (fast > slow for a long, fast < slow
    for a short) - the state as of that bar's close, not a fresh cross."""
    ema_fast = df_1m["close"].ewm(span=fast, adjust=False).mean().to_numpy()
    ema_slow = df_1m["close"].ewm(span=slow, adjust=False).mean().to_numpy()
    bar_idx = trades["bar_index"].to_numpy()
    is_long = (trades["direction"] == "long").to_numpy()
    fast_above = ema_fast[bar_idx] > ema_slow[bar_idx]
    return pd.Series(np.where(is_long, fast_above, ~fast_above), index=trades.index)


def report(label: str, trades: pd.DataFrame, df_1m: pd.DataFrame) -> None:
    print(f"\n{'='*88}\n{label}\n{'='*88}")
    decided = trades[trades["outcome"].isin(["win", "loss"])].copy()
    overall = summarize_outcomes(decided)
    print(f"Overall (no EMA filter): n={overall['n_decided']}  win_rate={overall['win_rate']*100:.1f}%  "
          f"avg_R={overall['avg_r']:+.3f}  PF={overall['profit_factor']:.2f}\n")

    print(f"{'pair':>8}{'aligned n':>12}{'aligned WR':>13}{'aligned avgR':>15}"
          f"{'not-aligned n':>16}{'not WR':>10}{'not avgR':>11}")
    for fast, slow in PAIRS:
        aligned = tag_ema_alignment(decided, df_1m, fast, slow)
        yes = decided[aligned]
        no = decided[~aligned]
        s_yes = summarize_outcomes(yes)
        s_no = summarize_outcomes(no)
        n_yes, n_no = s_yes.get("n_decided", 0), s_no.get("n_decided", 0)
        yes_str = f"{s_yes['win_rate']*100:5.1f}%" if n_yes >= MIN_N else "  n/a"
        yes_r = f"{s_yes['avg_r']:+.3f}" if n_yes >= MIN_N else "  n/a"
        no_str = f"{s_no['win_rate']*100:5.1f}%" if n_no >= MIN_N else "  n/a"
        no_r = f"{s_no['avg_r']:+.3f}" if n_no >= MIN_N else "  n/a"
        print(f"{fast:>3}/{slow:<4}{n_yes:>12}{yes_str:>13}{yes_r:>15}{n_no:>16}{no_str:>10}{no_r:>11}")


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    trades_in = run_fib_golden_zone_strategy(df_in, entry_mode="limit")
    trades_oos = run_fib_golden_zone_strategy(df_oos, entry_mode="limit")

    report("IN-SAMPLE 2023-2024", trades_in, df_in)
    report("OOS 2025-present", trades_oos, df_oos)
