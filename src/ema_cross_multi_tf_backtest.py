# ema_cross_predictor_backtest.py found a real, walk-forward-replicated
# R-multiple edge (avoiding crosses that fire late - already-elevated ADX,
# long since the last cross) but it died once realistic position sizing and
# commission/slippage were applied: on 1-minute bars this system trades
# ~8,000+ times/year with tight ATR stops, so per-trade costs swamp a ~+0.01R
# edge. This module re-runs the SAME pipeline - same feature functions, same
# filter, same cost model - on coarser bars (5min/15min/30min) instead of
# 1-minute, to see whether cutting trade frequency by an order of magnitude
# is enough for the edge to clear its own costs.
#
# The 9/21 EMA cross and all five features (ADX, EMA slope, ATR ratio,
# volume ratio, bars-since-last-cross) are recomputed directly on the
# resampled OHLCV bars - these are standard bar-count-based indicators
# (ADX(14), ATR(14), ...), timeframe-agnostic by construction, exactly like
# using them on a 5-minute chart in TradingView. Trade outcomes are still
# resolved against 1-MINUTE bars for intrabar stop/target precision (not the
# coarser bars the signal fires on) - each HTF cross's decision time (the
# instant its bar closes, htf_open + duration) is mapped onto the exact
# 1-minute bar that produced that same closing price, confirmed to match
# exactly (see git history / test in chat), then resolve_trade_outcomes
# walks forward from there at 1-minute resolution.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_multi_tf_backtest

import numpy as np
import pandas as pd

from .backtest import resolve_trade_outcomes
from .bot.sizing import compute_contracts
from .data import load_1m_bars, resample_bars
from .ema_cross_predictor_backtest import BREAKEVEN_R, STOP_ATR_MULT, attach_stop_target
from .ema_cross_trend_predictors import IN_SAMPLE_PATH, OOS_PATH, _quantile_edges, find_all_crosses_with_features
from .report import summarize_outcomes

TIMEFRAMES = ["1min", "5min", "15min", "30min"]
TARGET_R_MULTIPLES = [1.5, 2.0]

TICK_SIZE = 0.25
TICK_VALUE = 0.50
RISK_USD = 125.0
MAX_CONTRACTS = 5
MAX_RISK_PER_TRADE_USD = 250.0
COMMISSION_PER_CONTRACT_RT = 0.74
SLIPPAGE_TICKS_PER_SIDE = 1.0


def crosses_on_timeframe(df_1m: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross+feature events detected on `rule`-resampled bars, with bar_index/
    entry_price/time remapped onto the 1-minute frame so resolve_trade_outcomes
    can walk forward at full intrabar precision. Returns (df_htf, crosses)."""
    df_htf = resample_bars(df_1m, rule) if rule != "1min" else df_1m
    crosses = find_all_crosses_with_features(df_htf)
    if rule == "1min":
        return df_htf, crosses

    duration = pd.Timedelta(rule)
    decision_time = df_htf.index[crosses["bar_index"].to_numpy()] + duration
    ltf_pos = df_1m.index.searchsorted(decision_time, side="left") - 1
    valid = (ltf_pos >= 0) & (ltf_pos < len(df_1m))
    crosses = crosses[valid].copy()
    ltf_pos = ltf_pos[valid]

    crosses["bar_index"] = ltf_pos
    crosses["entry_price"] = df_1m["close"].to_numpy()[ltf_pos]
    crosses["time"] = df_1m.index[ltf_pos]
    return df_htf, crosses


def add_dollar_pnl(resolved: pd.DataFrame) -> pd.DataFrame:
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven"])].copy()
    contracts, dollar_pnl = [], []
    for row in decided.itertuples(index=False):
        n = compute_contracts(
            row.entry_price, row.stop_price, TICK_SIZE, TICK_VALUE,
            RISK_USD, MAX_CONTRACTS, MAX_RISK_PER_TRADE_USD,
        )
        risk_per_contract = (abs(row.entry_price - row.stop_price) / TICK_SIZE) * TICK_VALUE
        cost = n * (COMMISSION_PER_CONTRACT_RT + SLIPPAGE_TICKS_PER_SIDE * 2 * TICK_VALUE)
        gross = row.r_multiple * risk_per_contract * n
        contracts.append(n)
        dollar_pnl.append(gross - cost)
    decided["contracts"] = contracts
    decided["dollar_pnl"] = dollar_pnl
    return decided


def build_filter(crosses_in: pd.DataFrame):
    """Same best filter found in ema_cross_predictor_backtest.py: exclude
    crosses that fire with already-elevated ADX or a long stretch since the
    last cross - both were consistent walk-forward losers in R-multiple
    terms. Thresholds fit on this timeframe's own in-sample data."""
    adx_edges = _quantile_edges(crosses_in["adx"])
    bsc_edges = _quantile_edges(crosses_in["bars_since_last_cross"])
    return lambda df: (df["adx"] <= adx_edges[1]) & (df["bars_since_last_cross"] <= bsc_edges[1])


def report_period(label: str, decided: pd.DataFrame, n_days: float) -> dict:
    if decided.empty:
        print(f"    {label:5s} n=0")
        return {"n": 0}
    stats = summarize_outcomes(decided)
    total = decided["dollar_pnl"].sum()
    avg = decided["dollar_pnl"].mean()
    print(
        f"    {label:5s} n={len(decided):5d} (~{len(decided)/(n_days/365):5.0f}/yr)  "
        f"pf={stats['profit_factor']:.3f}  avg_r={stats['avg_r']:+.4f}  "
        f"avg_$={avg:+7.2f}  total_$={total:+11,.2f}  avg_contracts={decided['contracts'].mean():.2f}"
    )
    return {"n": len(decided), "total": total, "avg": avg, "pf": stats["profit_factor"]}


def run() -> None:
    df_1m_in = load_1m_bars(IN_SAMPLE_PATH)
    df_1m_oos = load_1m_bars(OOS_PATH)

    for tf in TIMEFRAMES:
        print("\n" + "=" * 100)
        print(f"TIMEFRAME: {tf}")
        print("=" * 100)

        df_htf_in, crosses_in = crosses_on_timeframe(df_1m_in, tf)
        df_htf_oos, crosses_oos = crosses_on_timeframe(df_1m_oos, tf)
        print(f"  crosses: in-sample={len(crosses_in)}  oos={len(crosses_oos)}")
        if len(crosses_in) < 30 or len(crosses_oos) < 30:
            print("  too few crosses on this timeframe, skipping")
            continue

        filter_fn = build_filter(crosses_in)
        n_days_in = (df_1m_in.index.max() - df_1m_in.index.min()).days
        n_days_oos = (df_1m_oos.index.max() - df_1m_oos.index.min()).days

        for target_r in TARGET_R_MULTIPLES:
            print(f"\n  --- target {target_r}R (stop {STOP_ATR_MULT}x ATR on {tf} bars) ---")
            trades_in = attach_stop_target(crosses_in, STOP_ATR_MULT, target_r)
            trades_oos = attach_stop_target(crosses_oos, STOP_ATR_MULT, target_r)
            resolved_in = resolve_trade_outcomes(trades_in, df_1m_in, BREAKEVEN_R)
            resolved_oos = resolve_trade_outcomes(trades_oos, df_1m_oos, BREAKEVEN_R)

            print("  baseline (no filter):")
            report_period("IN", add_dollar_pnl(resolved_in), n_days_in)
            report_period("OOS", add_dollar_pnl(resolved_oos), n_days_oos)

            print("  filtered (exclude late/extended crosses):")
            report_period("IN", add_dollar_pnl(resolved_in[filter_fn(resolved_in)]), n_days_in)
            report_period("OOS", add_dollar_pnl(resolved_oos[filter_fn(resolved_oos)]), n_days_oos)


if __name__ == "__main__":
    run()
