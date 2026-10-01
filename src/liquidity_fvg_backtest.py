# Backtests Liquidity_FVG_v1.pine (see liquidity_fvg_strategy.py) across 2
# years of MNQ 1-minute data - the indicator is built for the 1min/5min
# chart, and since entries are event-driven rather than chart-timeframe-
# driven, running on 1-minute data directly reflects both.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.liquidity_fvg_backtest

from .data import load_1m_bars
from .liquidity_fvg_strategy import run_liquidity_fvg_strategy
from .report import add_hour_tags, print_report, summarize_outcomes

DATA_PATH = "data/mnq_1m_2023_2025.parquet"

if __name__ == "__main__":
    df_1m = load_1m_bars(DATA_PATH)
    print(f"Loaded {len(df_1m):,} 1-minute bars: {df_1m.index.min()} -> {df_1m.index.max()}")

    # Default config chosen after comparing against the original liquidity-
    # pool-target design and a sweep of fixed R multiples (1.0-4.0), both
    # all-day and NY-AM-only. Avg R is nearly flat across that whole range
    # (0.45-0.53) - total R peaks at 2.0R (566.0) but 1.5R (538.5) gives up
    # only ~5% of that for a meaningfully smoother ride: win rate 60.1% vs
    # 51.1%, longest losing streak 7 vs 8, max drawdown 7R vs 9R. That small
    # a total-R gap is plausibly in-sample noise rather than a real
    # difference, so 1.5R is the default. Consistent in both years (avg R
    # 0.47 in 2023, 0.54 in 2024).
    # Note: NY hours 1,2,5,7,9 walk-forward validated as consistently above-
    # average (avg R ~0.69, win rate ~56% in BOTH years, vs ~0.53/51% for
    # all-day) - deliberately left OFF by default, keeping the higher total
    # R from trading all day rather than narrowing to fewer, higher-quality
    # setups. Revisit with use_time_filter=True and a custom hour set if
    # that tradeoff ever looks worth it.
    trades, diagnostics = run_liquidity_fvg_strategy(
        df_1m, use_time_filter=False, use_rr_filter=False, fixed_r_target=1.5,
    )

    n_open = (trades["outcome"] == "open_at_data_end").sum() if not trades.empty else 0
    print(f"\nTotal trades taken: {len(trades):,}")
    print(f"  Still open at end of data (excluded from win rate): {n_open:,}")
    print(f"Setups skipped - reversal confirmed but no untapped opposing target existed: {diagnostics['skipped_no_target']:,}")
    print(f"Setups skipped - reversal confirmed but failed the time/RR filter: {diagnostics['skipped_by_filter']:,}")

    if trades.empty:
        print("\nNo trades to report on.")
    else:
        trades.to_csv("data/liquidity_fvg_backtest_trades.csv", index=False)
        print()
        print_report(trades)

        print()
        print("=" * 72)
        print("RESULTS BY ENTRY HOUR (America/New_York)")
        print("=" * 72)
        tagged = add_hour_tags(trades)
        for hour in range(24):
            sub = tagged[tagged["entry_hour"] == hour]
            s = summarize_outcomes(sub) if not sub.empty else {"n_decided": 0}
            if s["n_decided"] == 0:
                continue
            print(
                f"{hour:02d}:00  n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  "
                f"avg_R={s['avg_r']:6.3f}  total_R={s['total_r']:8.2f}"
            )
