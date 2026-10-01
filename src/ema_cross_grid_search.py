# Broader grid search around the 15-minute finding from
# ema_cross_multi_tf_backtest.py: sweeps stop distance, target R-multiple,
# filter strictness, and neighboring timeframes (10/15/20min) to see how
# much better this can get, with win rate reported alongside PF/avg_r/$.
#
# Selection discipline matters here more than ever, because a wide grid is
# exactly how you accidentally p-hack a backtest: every config is ranked
# and picked using IN-SAMPLE results ONLY (2023-2024). OOS (2025-present)
# numbers for the resulting picks are reported afterward as a genuine
# holdout check, never used to choose among configs - if OOS were used to
# pick the "winner" too, we'd just be overfitting to a bigger haystack.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_grid_search

import numpy as np
import pandas as pd

from .backtest import resolve_trade_outcomes
from .data import load_1m_bars
from .ema_cross_predictor_backtest import BREAKEVEN_R, attach_stop_target
from .ema_cross_trend_predictors import IN_SAMPLE_PATH, OOS_PATH, _quantile_edges
from .ema_cross_multi_tf_backtest import add_dollar_pnl, crosses_on_timeframe
from .report import max_drawdown_r, summarize_outcomes

TIMEFRAMES = ["10min", "15min", "20min"]
STOP_ATR_MULTS = [1.0, 1.5, 2.0, 2.5]
TARGET_R_MULTIPLES = [1.5, 2.0, 2.5, 3.0, 4.0]

MIN_N_IN = 100
MIN_N_OOS = 60


def build_filter_variants(crosses_in: pd.DataFrame) -> dict:
    """Filter thresholds fit on in-sample only. 'none' = every cross,
    'loose' = exclude the worst tertile of ADX/bars-since-cross (the filter
    from ema_cross_predictor_backtest.py), 'strict' = exclude the worst HALF
    (median split) for a more aggressive cut."""
    adx_terc = _quantile_edges(crosses_in["adx"])
    bsc_terc = _quantile_edges(crosses_in["bars_since_last_cross"])
    adx_med = crosses_in["adx"].median()
    bsc_med = crosses_in["bars_since_last_cross"].median()

    return {
        "none": lambda df: pd.Series(True, index=df.index),
        "loose (exclude worst tertile)": lambda df: (df["adx"] <= adx_terc[1]) & (df["bars_since_last_cross"] <= bsc_terc[1]),
        "strict (exclude worst half)": lambda df: (df["adx"] <= adx_med) & (df["bars_since_last_cross"] <= bsc_med),
    }


def stats_for(resolved: pd.DataFrame, n_days: float) -> dict | None:
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven"])]
    if len(decided) == 0:
        return None
    s = summarize_outcomes(decided)
    d = add_dollar_pnl(resolved)
    return {
        "n": s["n_decided"], "win_rate": s["win_rate"], "pf": s["profit_factor"],
        "avg_r": s["avg_r"], "total_r": s["total_r"], "max_dd_r": max_drawdown_r(decided),
        "total_dollar": d["dollar_pnl"].sum(), "avg_dollar": d["dollar_pnl"].mean() if len(d) else 0.0,
        "per_year": s["n_decided"] / (n_days / 365),
    }


def run() -> None:
    df_1m_in = load_1m_bars(IN_SAMPLE_PATH)
    df_1m_oos = load_1m_bars(OOS_PATH)
    n_days_in = (df_1m_in.index.max() - df_1m_in.index.min()).days
    n_days_oos = (df_1m_oos.index.max() - df_1m_oos.index.min()).days

    results = []

    for tf in TIMEFRAMES:
        _, crosses_in = crosses_on_timeframe(df_1m_in, tf)
        _, crosses_oos = crosses_on_timeframe(df_1m_oos, tf)
        filters = build_filter_variants(crosses_in)

        for stop_mult in STOP_ATR_MULTS:
            for target_r in TARGET_R_MULTIPLES:
                trades_in = attach_stop_target(crosses_in, stop_mult, target_r)
                trades_oos = attach_stop_target(crosses_oos, stop_mult, target_r)
                resolved_in = resolve_trade_outcomes(trades_in, df_1m_in, BREAKEVEN_R)
                resolved_oos = resolve_trade_outcomes(trades_oos, df_1m_oos, BREAKEVEN_R)

                for filt_name, filt_fn in filters.items():
                    s_in = stats_for(resolved_in[filt_fn(resolved_in)], n_days_in)
                    if s_in is None or s_in["n"] < MIN_N_IN:
                        continue
                    s_oos = stats_for(resolved_oos[filt_fn(resolved_oos)], n_days_oos)
                    if s_oos is None or s_oos["n"] < MIN_N_OOS:
                        continue
                    results.append({
                        "tf": tf, "stop_mult": stop_mult, "target_r": target_r, "filter": filt_name,
                        "in": s_in, "oos": s_oos,
                    })

    # Selection uses IN-SAMPLE only - rank by in-sample avg R (per-trade
    # expectancy), the same metric the earlier, narrower search used.
    results.sort(key=lambda r: r["in"]["avg_r"], reverse=True)

    print(f"Evaluated {len(results)} configs with enough trades in both periods.\n")
    print("Top 15 by IN-SAMPLE avg R (selection uses ONLY this column - OOS shown after, untouched by ranking):\n")
    header = f"{'tf':6s} {'stop':5s} {'tgt':5s} {'filter':30s} | {'IN n':>6s} {'IN wr':>6s} {'IN pf':>6s} {'IN avgR':>8s} {'IN $':>10s} {'IN dd':>6s} | {'OOS n':>6s} {'OOS wr':>7s} {'OOS pf':>7s} {'OOS avgR':>9s} {'OOS $':>10s} {'OOS dd':>6s}"
    print(header)
    print("-" * len(header))
    for r in results[:15]:
        i, o = r["in"], r["oos"]
        print(
            f"{r['tf']:6s} {r['stop_mult']:<5.1f} {r['target_r']:<5.1f} {r['filter']:30s} | "
            f"{i['n']:6d} {i['win_rate']*100:5.1f}% {i['pf']:6.3f} {i['avg_r']:+8.4f} {i['total_dollar']:+10,.0f} {i['max_dd_r']:6.0f} | "
            f"{o['n']:6d} {o['win_rate']*100:6.1f}% {o['pf']:7.3f} {o['avg_r']:+9.4f} {o['total_dollar']:+10,.0f} {o['max_dd_r']:6.0f}"
        )

    best = results[0]
    print("\n" + "=" * 100)
    print("BEST BY IN-SAMPLE SELECTION (this is the pick - not cherry-picked from OOS results):")
    i, o = best["in"], best["oos"]
    print(f"  {best['tf']} bars, {best['stop_mult']}x ATR stop, {best['target_r']}R target, filter: {best['filter']}")
    print(f"  IN-SAMPLE:  n={i['n']} (~{i['per_year']:.0f}/yr)  win_rate={i['win_rate']*100:.1f}%  PF={i['pf']:.3f}  "
          f"avg_r={i['avg_r']:+.4f}  total_r={i['total_r']:+.1f}  max_dd={i['max_dd_r']:.0f}R  total_$={i['total_dollar']:+,.2f}")
    print(f"  OOS:        n={o['n']} (~{o['per_year']:.0f}/yr)  win_rate={o['win_rate']*100:.1f}%  PF={o['pf']:.3f}  "
          f"avg_r={o['avg_r']:+.4f}  total_r={o['total_r']:+.1f}  max_dd={o['max_dd_r']:.0f}R  total_$={o['total_dollar']:+,.2f}")
    holds_up = o["pf"] > 1.0 and o["avg_r"] > 0
    print(f"  Holds up OOS: {'YES' if holds_up else 'NO'}")
    print("=" * 100)


if __name__ == "__main__":
    run()
