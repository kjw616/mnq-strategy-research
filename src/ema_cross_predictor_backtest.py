# Grid search over ema_cross_trend_predictors.py's five pre-cross features,
# now scored the way a trend-following system actually makes money: real
# ATR stop/target resolution (win/loss/breakeven, R-multiple), not a fixed-
# horizon close-to-close direction check. That prior study found no feature
# that beat 50% on raw direction - but a trend system doesn't need to beat
# 50% on direction alone if winners run further than losers (R-asymmetry),
# so this re-tests the same features/thresholds against actual trade outcomes.
#
# All crosses across the full dataset, standard trend-following read (cross
# up = long, cross down = short) - same event set as ema_cross_trend_predictors.
# Trades can overlap (no one-open-position-at-a-time cap); this measures
# per-trade expectancy of a filtered signal, not a realistic single-account
# equity curve.
#
# Same walk-forward discipline as the rest of this codebase: every filter's
# bucket thresholds are fit on IN-SAMPLE quantiles only (see
# ema_cross_trend_predictors._quantile_edges) and reused as-is on OOS. A
# config only gets reported as a "hit" if it clears the bar on BOTH periods
# independently - in-sample-only results are never enough on their own.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_predictor_backtest

import numpy as np
import pandas as pd

from .backtest import resolve_trade_outcomes
from .data import load_1m_bars
from .ema_cross_trend_predictors import FEATURES, IN_SAMPLE_PATH, OOS_PATH, _quantile_edges, find_all_crosses_with_features
from .report import max_drawdown_r, summarize_outcomes

STOP_ATR_MULT = 1.5
TARGET_R_MULTIPLES = [1.5, 2.0, 3.0]
BREAKEVEN_R = 1e6  # disabled - isolate the filter's edge from breakeven-timing effects

# Minimum trades required in BOTH periods for a config to be worth reporting
MIN_N = 30
# "Good" bar: profitable (profit_factor > 1, positive avg_r) in both periods
GOOD_PF = 1.0


def attach_stop_target(crosses: pd.DataFrame, stop_atr_mult: float, target_r_multiple: float) -> pd.DataFrame:
    out = crosses.copy()
    is_long = (out["direction"] == "long").to_numpy()
    stop_dist = out["atr_at_entry"].to_numpy() * stop_atr_mult
    entry = out["entry_price"].to_numpy()
    out["stop_price"] = np.where(is_long, entry - stop_dist, entry + stop_dist)
    out["target_price"] = np.where(is_long, entry + stop_dist * target_r_multiple, entry - stop_dist * target_r_multiple)
    return out


def build_filters(evaluated_in: pd.DataFrame) -> dict:
    """Filter configs (name -> boolean-mask function), thresholds fit on
    in-sample only. Includes the unfiltered baseline plus each feature's
    Low and High tertile, extremes only (Mid dropped - we care whether an
    edge sits at the tails, not the ambiguous middle)."""
    filters = {"baseline (no filter)": lambda df: pd.Series(True, index=df.index)}
    for feature in FEATURES:
        edges = _quantile_edges(evaluated_in[feature])

        def make(feature=feature, lo=edges[0]):
            return lambda df: df[feature] <= lo

        def make_hi(feature=feature, hi=edges[1]):
            return lambda df: df[feature] > hi

        filters[f"{feature} Low (<= {edges[0]:.3f})"] = make()
        filters[f"{feature} High (> {edges[1]:.3f})"] = make_hi()
    return filters


def stats_row(resolved: pd.DataFrame) -> dict:
    if resolved.empty:
        return {"n_decided": 0}
    s = summarize_outcomes(resolved)
    if s["n_decided"] == 0:
        return s
    s["max_dd_r"] = max_drawdown_r(resolved)
    return s


def run() -> None:
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)
    crosses_in = find_all_crosses_with_features(df_in)
    crosses_oos = find_all_crosses_with_features(df_oos)
    print(f"IN-SAMPLE crosses: {len(crosses_in)}   OOS crosses: {len(crosses_oos)}")

    filters = build_filters(crosses_in)
    hits = []

    for target_r in TARGET_R_MULTIPLES:
        trades_in = attach_stop_target(crosses_in, STOP_ATR_MULT, target_r)
        trades_oos = attach_stop_target(crosses_oos, STOP_ATR_MULT, target_r)
        resolved_in = resolve_trade_outcomes(trades_in, df_in, BREAKEVEN_R)
        resolved_oos = resolve_trade_outcomes(trades_oos, df_oos, BREAKEVEN_R)

        for name, mask_fn in filters.items():
            sub_in = resolved_in[mask_fn(resolved_in)]
            sub_oos = resolved_oos[mask_fn(resolved_oos)]
            s_in = stats_row(sub_in)
            s_oos = stats_row(sub_oos)

            if s_in.get("n_decided", 0) < MIN_N or s_oos.get("n_decided", 0) < MIN_N:
                continue

            good = (
                s_in["profit_factor"] > GOOD_PF and s_in["avg_r"] > 0
                and s_oos["profit_factor"] > GOOD_PF and s_oos["avg_r"] > 0
            )
            row = {
                "filter": name, "stop_atr_mult": STOP_ATR_MULT, "target_r": target_r,
                "n_in": s_in["n_decided"], "pf_in": s_in["profit_factor"], "avg_r_in": s_in["avg_r"],
                "win_rate_in": s_in["win_rate"], "dd_in": s_in["max_dd_r"],
                "n_oos": s_oos["n_decided"], "pf_oos": s_oos["profit_factor"], "avg_r_oos": s_oos["avg_r"],
                "win_rate_oos": s_oos["win_rate"], "dd_oos": s_oos["max_dd_r"],
                "good": good,
            }
            if good:
                hits.append(row)

            marker = " <-- HOLDS UP BOTH PERIODS" if good else ""
            print(
                f"[{target_r:.1f}R target] {name:38s} "
                f"IN: n={s_in['n_decided']:5d} pf={s_in['profit_factor']:.2f} avg_r={s_in['avg_r']:+.3f} wr={s_in['win_rate']*100:4.1f}% dd={s_in['max_dd_r']:.1f}R  |  "
                f"OOS: n={s_oos['n_decided']:5d} pf={s_oos['profit_factor']:.2f} avg_r={s_oos['avg_r']:+.3f} wr={s_oos['win_rate']*100:4.1f}% dd={s_oos['max_dd_r']:.1f}R"
                f"{marker}"
            )

    print("\n" + "=" * 100)
    if hits:
        print(f"{len(hits)} config(s) profitable (PF>1, avg_r>0) in BOTH in-sample and OOS:")
        for row in sorted(hits, key=lambda r: r["pf_oos"], reverse=True):
            print(f"  {row['filter']}  @ {row['target_r']}R target  ->  "
                  f"IN pf={row['pf_in']:.2f}/avg_r={row['avg_r_in']:+.3f}  "
                  f"OOS pf={row['pf_oos']:.2f}/avg_r={row['avg_r_oos']:+.3f}")
    else:
        print("No config was profitable in both periods at once.")
    print("=" * 100)


if __name__ == "__main__":
    run()
