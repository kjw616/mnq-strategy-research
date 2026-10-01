# HTF-bias + LTF-entry version of the 9/21 EMA cross: rather than waiting
# for a 15-minute bar to close before acting on its trend
# (ema_cross_multi_tf_backtest.py's approach - entries only fire on the
# 15-min bar's own close), this reads the 15-minute EMA-cross relationship
# as a standing STATE - bullish while the 9 EMA sits above the 21 on
# 15-min bars, bearish while below - and pulls the actual entry trigger
# down to the 1-minute chart's own faster 9/21 cross, taken only when it
# agrees with the current 15-min bias. The goal: keep the quality context
# that made 15-min bars work (ema_cross_multi_tf_backtest.py), but get in
# faster/closer to the actual turn instead of up to 15 minutes late.
#
# Non-repainting: the 15-min bias only becomes visible on the 1-minute
# timeline from the instant that 15-min bar actually closes
# (data.align_htf_to_ltf) - same discipline as htf_sweep_strategy's
# htf_bias_agreement, just built on an EMA-cross bias instead of
# structure_bias().
#
# Targets 1.5-2R, not the 0.5R config from ema_cross_grid_search_winrate.py
# - that one's edge sat right on top of its own breakeven win rate (too
# thin to trust), so this starts from a payoff ratio with actual room.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_htf_bias_ltf_entry

import numpy as np
import pandas as pd

from .backtest import resolve_trade_outcomes
from .crt_strategy import wilder_atr
from .data import align_htf_to_ltf, load_1m_bars, resample_bars
from .ema_cross_predictor_backtest import BREAKEVEN_R, attach_stop_target
from .ema_cross_multi_tf_backtest import add_dollar_pnl
from .ema_cross_strategy import ema_cross_signal
from .ema_cross_trend_predictors import IN_SAMPLE_PATH, OOS_PATH
from .report import max_drawdown_r, summarize_outcomes

HTF_RULE = "15min"
FAST_LEN = 9
SLOW_LEN = 21
ATR_LEN = 14
STOP_ATR_MULT = 1.5
TARGET_R_MULTIPLES = [1.5, 2.0]


def htf_ema_bias(df_1m: pd.DataFrame, rule: str = HTF_RULE, fast_len: int = FAST_LEN, slow_len: int = SLOW_LEN) -> pd.DataFrame:
    """1 = 9 EMA above 21 EMA on `rule`-resampled bars, -1 = below, 0 = tied
    (only possible right at the very first bar). A standing state, not a
    punctual cross event - same sign as ema_cross_signal's own fast-minus-
    slow diff, just exposed as a bias to align onto a faster timeline."""
    htf = resample_bars(df_1m, rule)
    sig = ema_cross_signal(htf["close"], fast_len, slow_len)
    bias = np.sign(sig["ema_fast"] - sig["ema_slow"])
    return bias.rename("bias").to_frame()


def find_ltf_entries(
    df_1m: pd.DataFrame,
    rule: str = HTF_RULE,
    fast_len: int = FAST_LEN,
    slow_len: int = SLOW_LEN,
    atr_len: int = ATR_LEN,
    one_per_regime: bool = True,
    stop_basis: str = "ltf",  # "ltf" = 1-min ATR, "htf" = rule-timeframe ATR aligned down
) -> pd.DataFrame:
    """1-minute 9/21 EMA crosses, kept only when they agree with the current
    (non-repainting) `rule`-timeframe bias. Entry price/time/bar_index are
    all native 1-minute values - no HTF->LTF remapping needed since the
    trigger itself already fires on the 1-minute chart.

    one_per_regime=True keeps only the FIRST agreeing 1-minute cross after
    each time the HTF bias flips - without this, "agrees with the bias"
    barely filters anything (the bias only flips direction, it doesn't stay
    silent), so trade frequency stays close to the raw 1-minute cross count
    and the whole point of moving to a slower timeframe (fewer, higher
    quality entries) is lost. With it, this behaves like the 15-minute
    signal's frequency, but entering as soon as price actually turns on the
    1-minute chart instead of waiting out the full 15-minute bar."""
    bias_htf = htf_ema_bias(df_1m, rule, fast_len, slow_len)
    bias_ltf = align_htf_to_ltf(df_1m.index, bias_htf, rule)["bias"].fillna(0).to_numpy()

    sig_ltf = ema_cross_signal(df_1m["close"], fast_len, slow_len)
    if stop_basis == "htf":
        htf = resample_bars(df_1m, rule)
        atr_htf = wilder_atr(htf, atr_len).rename("atr").to_frame()
        atr = align_htf_to_ltf(df_1m.index, atr_htf, rule)["atr"]
    else:
        atr = wilder_atr(df_1m, atr_len)

    cross_up = sig_ltf["cross_up"].to_numpy()
    cross_down = sig_ltf["cross_down"].to_numpy()
    long_entry = cross_up & (bias_ltf == 1)
    short_entry = cross_down & (bias_ltf == -1)
    is_entry = long_entry | short_entry

    if one_per_regime:
        regime_change = np.zeros(len(bias_ltf), dtype=bool)
        regime_change[1:] = bias_ltf[1:] != bias_ltf[:-1]
        regime_id = np.cumsum(regime_change)
        entry_regimes = regime_id[is_entry]
        first_in_regime = np.zeros(is_entry.sum(), dtype=bool)
        _, first_positions = np.unique(entry_regimes, return_index=True)
        first_in_regime[first_positions] = True
        entry_idx = np.flatnonzero(is_entry)
        keep = np.zeros(len(bias_ltf), dtype=bool)
        keep[entry_idx[first_in_regime]] = True
        is_entry = keep

    idx = np.flatnonzero(is_entry)
    return pd.DataFrame({
        "time": df_1m.index[idx],
        "bar_index": idx,
        "direction": np.where(long_entry[idx], "long", "short"),
        "entry_price": df_1m["close"].to_numpy()[idx],
        "atr_at_entry": atr.to_numpy()[idx],
    })


def report(label: str, resolved: pd.DataFrame, n_days: float) -> None:
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven"])]
    if decided.empty:
        print(f"  {label}: no trades")
        return
    s = summarize_outcomes(decided)
    dd = max_drawdown_r(decided)
    d = add_dollar_pnl(resolved)
    print(
        f"  {label}: n={s['n_decided']:5d} (~{s['n_decided']/(n_days/365):5.0f}/yr)  "
        f"win_rate={s['win_rate']*100:5.1f}%  pf={s['profit_factor']:.3f}  avg_r={s['avg_r']:+.4f}  "
        f"total_r={s['total_r']:+.1f}  max_dd={dd:.0f}R  total_$={d['dollar_pnl'].sum():+,.2f}  "
        f"avg_contracts={d['contracts'].mean():.2f}"
    )


def run() -> None:
    df_1m_in = load_1m_bars(IN_SAMPLE_PATH)
    df_1m_oos = load_1m_bars(OOS_PATH)
    n_days_in = (df_1m_in.index.max() - df_1m_in.index.min()).days
    n_days_oos = (df_1m_oos.index.max() - df_1m_oos.index.min()).days

    for stop_basis in ["ltf", "htf"]:
        entries_in = find_ltf_entries(df_1m_in, stop_basis=stop_basis)
        entries_oos = find_ltf_entries(df_1m_oos, stop_basis=stop_basis)
        basis_label = "1-min ATR" if stop_basis == "ltf" else f"{HTF_RULE} ATR (aligned down)"
        print(f"HTF bias: {HTF_RULE} 9/21 EMA state, LTF entry: 1-min 9/21 EMA cross agreeing with it, stop basis: {basis_label}")
        print(f"Entries: in-sample={len(entries_in)}  oos={len(entries_oos)}\n")

        for target_r in TARGET_R_MULTIPLES:
            print(f"--- target {target_r}R (stop {STOP_ATR_MULT}x ATR({ATR_LEN}) on {basis_label}) ---")
            trades_in = attach_stop_target(entries_in, STOP_ATR_MULT, target_r)
            trades_oos = attach_stop_target(entries_oos, STOP_ATR_MULT, target_r)
            resolved_in = resolve_trade_outcomes(trades_in, df_1m_in, BREAKEVEN_R)
            resolved_oos = resolve_trade_outcomes(trades_oos, df_1m_oos, BREAKEVEN_R)
            report("IN-SAMPLE", resolved_in, n_days_in)
            report("OOS       ", resolved_oos, n_days_oos)
            print()
        print()


if __name__ == "__main__":
    run()
