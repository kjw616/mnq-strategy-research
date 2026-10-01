# Walk-forward test of the 1H RB / London / NWOG / premium-discount
# configuration (see rb_1h_london_nwog_strategy.py) - the pattern described
# in Powell Trades' "How to structure a Rejection block entry" video.
# Same in-sample/OOS discipline as every other backtest in this repo.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.rb_1h_london_nwog_backtest

import pandas as pd

from .data import load_1m_bars
from .rb_1h_london_nwog_strategy import run_rb_1h_london_nwog_strategy
from .report import max_drawdown_r, print_report, summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

TARGET_R_MULTIPLES = [4.0, 6.0, 8.0]


def report_period(label: str, trades: pd.DataFrame) -> dict:
    print("=" * 78)
    print(label)
    print("=" * 78)
    if trades.empty:
        print("No trades.\n")
        return {"n_decided": 0}

    print_report(trades)
    stats = summarize_outcomes(trades)
    dd = max_drawdown_r(trades)
    n_days = (trades["time"].max() - trades["time"].min()).days or 1
    print(f"\nMax drawdown: {dd:.2f}R")
    print(f"Trade frequency: {stats['n_decided']} trades over {n_days} days "
          f"(~{stats['n_decided']/(n_days/7):.2f}/week)")
    print(f"Avg stop distance: {trades['stop_distance_points'].mean():.1f} pts\n")
    stats["max_drawdown_r"] = dd
    return stats


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    print("=" * 78)
    print("CONFIG SWEEP @ 8R (in-sample + OOS, no selection - just scanning)")
    print("=" * 78)
    print(f"{'london':>8}{'prem/disc':>11}{'in_n':>7}{'in_wr':>8}{'in_PF':>7}{'oos_n':>7}{'oos_wr':>8}{'oos_PF':>7}")
    for london in [False, True]:
        for pd_filter in [False, True]:
            t_in = run_rb_1h_london_nwog_strategy(df_in, target_r_multiple=8.0,
                                                   require_london_session=london, require_premium_discount=pd_filter)
            t_oos = run_rb_1h_london_nwog_strategy(df_oos, target_r_multiple=8.0,
                                                    require_london_session=london, require_premium_discount=pd_filter)
            s_in, s_oos = summarize_outcomes(t_in), summarize_outcomes(t_oos)
            n_in, n_oos = s_in.get("n_decided", 0), s_oos.get("n_decided", 0)
            in_str = f"{n_in:>7}{s_in['win_rate']*100:>7.1f}%{s_in['profit_factor']:>7.2f}" if n_in else f"{n_in:>7}{'':>8}{'':>7}"
            oos_str = f"{n_oos:>7}{s_oos['win_rate']*100:>7.1f}%{s_oos['profit_factor']:>7.2f}" if n_oos else f"{n_oos:>7}{'':>8}{'':>7}"
            print(f"{str(london):>8}{str(pd_filter):>11}{in_str}{oos_str}")
    print()

    for rr in TARGET_R_MULTIPLES:
        print("\n" + "#" * 78)
        print(f"TARGET = {rr}R")
        print("#" * 78)
        trades_in = run_rb_1h_london_nwog_strategy(df_in, target_r_multiple=rr)
        trades_oos = run_rb_1h_london_nwog_strategy(df_oos, target_r_multiple=rr)

        if rr == TARGET_R_MULTIPLES[-1]:
            if not trades_in.empty:
                trades_in.to_csv("data/rb_1h_london_nwog_in_sample_trades.csv", index=False)
            if not trades_oos.empty:
                trades_oos.to_csv("data/rb_1h_london_nwog_oos_trades.csv", index=False)

        stats_in = report_period(f"IN-SAMPLE 2023-2024 @ {rr}R", trades_in)
        stats_oos = report_period(f"OOS 2025-present @ {rr}R", trades_oos)

        if stats_in.get("n_decided", 0) and stats_oos.get("n_decided", 0):
            be = 1 / (1 + rr)
            print(f"  Breakeven WR needed: {be*100:.1f}%")
            print(f"  In-sample: WR={stats_in['win_rate']*100:.1f}%  PF={stats_in['profit_factor']:.2f}  "
                  f"avg_R={stats_in['avg_r']:+.3f}")
            print(f"  OOS:       WR={stats_oos['win_rate']*100:.1f}%  PF={stats_oos['profit_factor']:.2f}  "
                  f"avg_R={stats_oos['avg_r']:+.3f}")
