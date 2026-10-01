# Same question as fib_golden_zone_ema_confluence.py (does EMA alignment
# raise the win rate of fib_golden_zone_strategy's already-working trades)
# but holding the pair fixed at your actual 9/21 and varying the TIMEFRAME
# the EMA is read on instead - directly testing what
# ema_cross_multi_tf_backtest.py already found for a standalone EMA-cross
# system (15min is the one timeframe that's robustly profitable both
# periods; 1min loses to cost, 5min is marginal, 30min flips sign between
# in-sample and OOS) against your actual use case: EMA state as a
# confluence filter on a different, already-validated setup.
#
# "Aligned" = the EMA state (fast above/below slow) on the given timeframe,
# non-repainting (align_htf_to_ltf - only the most recently CLOSED HTF bar,
# never the one still forming), evaluated at each trade's own entry bar.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.fib_golden_zone_ema_timeframe_confluence

import numpy as np
import pandas as pd

from .data import align_htf_to_ltf, load_1m_bars, resample_bars
from .fib_golden_zone_strategy import run_fib_golden_zone_strategy
from .report import summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

FAST, SLOW = 9, 21
TIMEFRAMES = ["1min", "5min", "15min", "30min"]
MIN_N = 20


def htf_fast_above(df_1m: pd.DataFrame, timeframe: str) -> np.ndarray:
    if timeframe == "1min":
        close = df_1m["close"]
        ema_fast = close.ewm(span=FAST, adjust=False).mean()
        ema_slow = close.ewm(span=SLOW, adjust=False).mean()
        return (ema_fast > ema_slow).to_numpy()

    htf = resample_bars(df_1m, timeframe)
    ema_fast = htf["close"].ewm(span=FAST, adjust=False).mean()
    ema_slow = htf["close"].ewm(span=SLOW, adjust=False).mean()
    state = (ema_fast > ema_slow).to_frame("fast_above")
    aligned = align_htf_to_ltf(df_1m.index, state, timeframe)
    # merge_asof leaves NaN (and an object-dtype column) for 1m bars before
    # the first HTF close - fillna alone doesn't restore bool dtype, and a
    # bitwise ~ on an object array of Python bools does int inversion, not
    # logical negation, so the explicit astype(bool) here isn't optional.
    return aligned["fast_above"].fillna(False).astype(bool).to_numpy()


def report(label: str, trades: pd.DataFrame, df_1m: pd.DataFrame) -> None:
    print(f"\n{'='*88}\n{label}\n{'='*88}")
    decided = trades[trades["outcome"].isin(["win", "loss"])].copy()
    overall = summarize_outcomes(decided)
    print(f"Overall (no EMA filter): n={overall['n_decided']}  win_rate={overall['win_rate']*100:.1f}%  "
          f"avg_R={overall['avg_r']:+.3f}  PF={overall['profit_factor']:.2f}\n")

    print(f"{'tf':>7}{'aligned n':>12}{'aligned WR':>13}{'aligned avgR':>15}"
          f"{'not-aligned n':>16}{'not WR':>10}{'not avgR':>11}")
    for tf in TIMEFRAMES:
        fast_above = htf_fast_above(df_1m, tf)
        is_long = (decided["direction"] == "long").to_numpy()
        bar_idx = decided["bar_index"].to_numpy()
        aligned = np.where(is_long, fast_above[bar_idx], ~fast_above[bar_idx])
        yes = decided[aligned]
        no = decided[~aligned]
        s_yes = summarize_outcomes(yes)
        s_no = summarize_outcomes(no)
        n_yes, n_no = s_yes.get("n_decided", 0), s_no.get("n_decided", 0)
        yes_str = f"{s_yes['win_rate']*100:5.1f}%" if n_yes >= MIN_N else "  n/a"
        yes_r = f"{s_yes['avg_r']:+.3f}" if n_yes >= MIN_N else "  n/a"
        no_str = f"{s_no['win_rate']*100:5.1f}%" if n_no >= MIN_N else "  n/a"
        no_r = f"{s_no['avg_r']:+.3f}" if n_no >= MIN_N else "  n/a"
        print(f"{tf:>7}{n_yes:>12}{yes_str:>13}{yes_r:>15}{n_no:>16}{no_str:>10}{no_r:>11}")


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    trades_in = run_fib_golden_zone_strategy(df_in, entry_mode="limit")
    trades_oos = run_fib_golden_zone_strategy(df_oos, entry_mode="limit")

    report("IN-SAMPLE 2023-2024", trades_in, df_in)
    report("OOS 2025-present", trades_oos, df_oos)
