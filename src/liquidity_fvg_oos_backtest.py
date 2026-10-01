# Out-of-sample test of Keen's Indicator v2.1 (sl_shrink_factor=0.3,
# fixed_r_target=1.5) on data/mnq_1m_2025_present.parquet - the 2025-01-01
# through present period that was NOT part of the original 2023-2024
# backtest/walk-forward validation. Same methodology as
# src/liquidity_fvg_backtest.py, just pointed at the new data and with the
# v2.1 shrink factor applied.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.liquidity_fvg_oos_backtest

from .data import load_1m_bars
from .liquidity_fvg_strategy import run_liquidity_fvg_strategy
from .report import add_hour_tags, print_report, summarize_outcomes

DATA_PATH = "data/mnq_1m_2025_present.parquet"

if __name__ == "__main__":
    df_1m = load_1m_bars(DATA_PATH)
    print(f"Loaded {len(df_1m):,} 1-minute bars: {df_1m.index.min()} -> {df_1m.index.max()}")

    trades, diagnostics = run_liquidity_fvg_strategy(
        df_1m, use_time_filter=False, use_rr_filter=False, fixed_r_target=1.5, sl_shrink_factor=0.3,
    )

    n_open = (trades["outcome"] == "open_at_data_end").sum() if not trades.empty else 0
    print(f"\nTotal trades taken: {len(trades):,}")
    print(f"  Still open at end of data (excluded from win rate): {n_open:,}")
    print(f"Setups skipped - reversal confirmed but no untapped opposing target existed: {diagnostics['skipped_no_target']:,}")
    print(f"Setups skipped - reversal confirmed but failed the time/RR filter: {diagnostics['skipped_by_filter']:,}")

    if trades.empty:
        print("\nNo trades to report on.")
    else:
        trades.to_csv("data/liquidity_fvg_oos_2025_trades.csv", index=False)
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
            print(f"  {hour:2d}:00  n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:.2f}")

        print()
        print("=" * 72)
        print("YEAR-BY-YEAR SPLIT (2025 vs 2026 YTD)")
        print("=" * 72)
        import pandas as pd
        t = trades.copy()
        t["year"] = pd.to_datetime(t["time"]).dt.year
        for yr in sorted(t["year"].unique()):
            sub = t[t["year"] == yr]
            s = summarize_outcomes(sub)
            if s["n_decided"] == 0:
                continue
            print(f"  {yr}: n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:.2f}  PF={s['profit_factor']:.2f}")
