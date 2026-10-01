# Same grid-search machinery as ema_cross_grid_search.py, but optimizing for
# WIN RATE instead of avg R. The lever for that is tighter targets relative
# to the stop (and/or wider stops) - a 2-4R target trades win rate for
# payoff size, so this sweeps down into 0.5-2.0R targets and wider stops to
# find where win rate peaks while still keeping positive expectancy (PF>1,
# avg_r>0) - a high win rate with a negative edge (frequent small wins wiped
# out by rare large losses) isn't actually useful, so that screen stays in.
#
# Same selection discipline as before: every config is ranked by IN-SAMPLE
# win rate ONLY; OOS numbers for the resulting picks are reported afterward,
# untouched by the ranking, as a genuine holdout check.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_grid_search_winrate

from .backtest import resolve_trade_outcomes
from .data import load_1m_bars
from .ema_cross_predictor_backtest import BREAKEVEN_R, attach_stop_target
from .ema_cross_grid_search import build_filter_variants, stats_for
from .ema_cross_multi_tf_backtest import crosses_on_timeframe
from .ema_cross_trend_predictors import IN_SAMPLE_PATH, OOS_PATH

TIMEFRAMES = ["10min", "15min", "20min"]
STOP_ATR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
TARGET_R_MULTIPLES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

MIN_N_IN = 150
MIN_N_OOS = 80


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
                    if not (s_in["pf"] > 1.0 and s_in["avg_r"] > 0):
                        continue  # screen out high-win-rate-but-negative-expectancy configs
                    s_oos = stats_for(resolved_oos[filt_fn(resolved_oos)], n_days_oos)
                    if s_oos is None or s_oos["n"] < MIN_N_OOS:
                        continue
                    results.append({
                        "tf": tf, "stop_mult": stop_mult, "target_r": target_r, "filter": filt_name,
                        "in": s_in, "oos": s_oos,
                    })

    results.sort(key=lambda r: r["in"]["win_rate"], reverse=True)

    print(f"Evaluated {len(results)} configs (PF>1 and avg_r>0 in-sample, enough trades both periods).\n")
    print("Top 15 by IN-SAMPLE win rate (selection uses ONLY this column - OOS shown after, untouched by ranking):\n")
    header = f"{'tf':6s} {'stop':5s} {'tgt':5s} {'filter':30s} | {'IN n':>6s} {'IN wr':>6s} {'IN pf':>6s} {'IN avgR':>8s} {'IN $':>10s} {'IN dd':>6s} | {'OOS n':>6s} {'OOS wr':>7s} {'OOS pf':>7s} {'OOS avgR':>9s} {'OOS $':>10s} {'OOS dd':>6s}"
    print(header)
    print("-" * len(header))
    for r in results[:15]:
        i, o = r["in"], r["oos"]
        print(
            f"{r['tf']:6s} {r['stop_mult']:<5.1f} {r['target_r']:<5.2f} {r['filter']:30s} | "
            f"{i['n']:6d} {i['win_rate']*100:5.1f}% {i['pf']:6.3f} {i['avg_r']:+8.4f} {i['total_dollar']:+10,.0f} {i['max_dd_r']:6.0f} | "
            f"{o['n']:6d} {o['win_rate']*100:6.1f}% {o['pf']:7.3f} {o['avg_r']:+9.4f} {o['total_dollar']:+10,.0f} {o['max_dd_r']:6.0f}"
        )

    print("\nSame rows, but only ones where OOS ALSO stayed profitable (PF>1, avg_r>0) - i.e. actually held up:")
    holds_up = [r for r in results if r["oos"]["pf"] > 1.0 and r["oos"]["avg_r"] > 0]
    for r in holds_up[:15]:
        i, o = r["in"], r["oos"]
        print(
            f"{r['tf']:6s} {r['stop_mult']:<5.1f} {r['target_r']:<5.2f} {r['filter']:30s} | "
            f"{i['n']:6d} {i['win_rate']*100:5.1f}% {i['pf']:6.3f} {i['avg_r']:+8.4f} {i['total_dollar']:+10,.0f} {i['max_dd_r']:6.0f} | "
            f"{o['n']:6d} {o['win_rate']*100:6.1f}% {o['pf']:7.3f} {o['avg_r']:+9.4f} {o['total_dollar']:+10,.0f} {o['max_dd_r']:6.0f}"
        )
    if not holds_up:
        print("  none")


if __name__ == "__main__":
    run()
