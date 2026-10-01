# Combines the full in-sample (2023-2024) and out-of-sample (2025-present)
# Keen's Indicator v2.1 trade logs into one continuous record and reports
# the full set of summary stats across the whole thing: 2023-01-03 through
# whatever the most recent out-of-sample run covered.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.combined_analysis

import pandas as pd

from .report import summarize_outcomes

IN_SAMPLE_PATH = "data/liquidity_fvg_shrink03_trades.csv"
OOS_PATH = "data/liquidity_fvg_oos_2025_trades.csv"
TZ = "America/New_York"


def load_combined() -> pd.DataFrame:
    a = pd.read_csv(IN_SAMPLE_PATH, parse_dates=["time", "exit_time"])
    b = pd.read_csv(OOS_PATH, parse_dates=["time", "exit_time"])
    combined = pd.concat([a, b], ignore_index=True).sort_values("time").reset_index(drop=True)
    combined["time_local"] = combined["time"].dt.tz_convert(TZ)
    combined["date"] = combined["time_local"].dt.date
    combined["month"] = combined["time_local"].dt.tz_localize(None).dt.to_period("M")
    combined["year"] = combined["time_local"].dt.year
    combined["dow"] = combined["time_local"].dt.day_name()
    return combined


def longest_streak(outcomes: pd.Series, target: str) -> int:
    best = cur = 0
    for o in outcomes:
        if o == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def max_drawdown_r(r_multiples: pd.Series) -> float:
    equity = r_multiples.cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    return drawdown.max()


if __name__ == "__main__":
    combined = load_combined()
    combined.to_csv("data/liquidity_fvg_combined_trades.csv", index=False)

    decided = combined[combined["outcome"].isin(["win", "loss", "breakeven"])].copy()
    stats = summarize_outcomes(decided)

    span_days = (decided["time"].max() - decided["time"].min()).days

    print("=" * 72)
    print("COMBINED KEEN'S INDICATOR v2.1 RESULTS (2023-01-03 -> present)")
    print("sl_shrink_factor=0.3, fixed_r_target=1.5, no time/RR filter")
    print("=" * 72)
    print(f"Full range: {decided['time'].min()} -> {decided['time'].max()}  ({span_days:,} calendar days)")
    print(f"Total decided trades: {stats['n_decided']:,}")
    print(f"  Wins:      {stats['n_win']:5,d}  ({stats['win_rate']*100:.1f}%)")
    print(f"  Losses:    {stats['n_loss']:5,d}  ({stats['loss_rate']*100:.1f}%)")
    print(f"  Breakeven: {stats['n_breakeven']:5,d}  ({stats['breakeven_rate']*100:.1f}%)")
    print(f"  Average R: {stats['avg_r']:.3f}")
    print(f"  Total R:   {stats['total_r']:.2f}")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Std dev of R (per-trade): {decided['r_multiple'].std():.3f}")
    print(f"  Best single trade:  {decided['r_multiple'].max():.2f}R")
    print(f"  Worst single trade: {decided['r_multiple'].min():.2f}R")

    print(f"\n  Longest winning streak: {longest_streak(decided['outcome'], 'win')} trades")
    print(f"  Longest losing streak:  {longest_streak(decided['outcome'], 'loss')} trades")
    print(f"  Max drawdown: {max_drawdown_r(decided['r_multiple']):.2f}R")

    # --- Per-day aggregation ---
    by_day = decided.groupby("date").agg(
        n_trades=("outcome", "size"),
        n_wins=("outcome", lambda s: (s == "win").sum()),
        total_r=("r_multiple", "sum"),
    )
    by_day["win_rate"] = by_day["n_wins"] / by_day["n_trades"]

    trading_days = len(by_day)
    print(f"\nTrading days with >=1 trade: {trading_days:,}  (avg {stats['n_decided']/trading_days:.2f} trades/day)")

    most_active_day = by_day["n_trades"].idxmax()
    print(f"Most trades in a single day: {by_day.loc[most_active_day, 'n_trades']:.0f} on {most_active_day}")

    best_day = by_day["total_r"].idxmax()
    print(f"Best day (total R):  {best_day}  {by_day.loc[best_day, 'total_r']:.2f}R  ({by_day.loc[best_day, 'n_trades']:.0f} trades)")
    worst_day = by_day["total_r"].idxmin()
    print(f"Worst day (total R): {worst_day}  {by_day.loc[worst_day, 'total_r']:.2f}R  ({by_day.loc[worst_day, 'n_trades']:.0f} trades)")

    # --- Per-month aggregation ---
    by_month = decided.groupby("month").apply(summarize_outcomes, include_groups=False)
    by_month_df = pd.DataFrame(list(by_month), index=by_month.index)

    best_month_r = by_month_df["total_r"].idxmax()
    worst_month_r = by_month_df["total_r"].idxmin()
    best_month_wr = by_month_df["win_rate"].idxmax()
    worst_month_wr = by_month_df["win_rate"].idxmin()
    print(f"\nBest month (total R):  {best_month_r}  {by_month_df.loc[best_month_r, 'total_r']:.2f}R  (win rate {by_month_df.loc[best_month_r, 'win_rate']*100:.1f}%, n={by_month_df.loc[best_month_r, 'n_decided']:.0f})")
    print(f"Worst month (total R): {worst_month_r}  {by_month_df.loc[worst_month_r, 'total_r']:.2f}R  (win rate {by_month_df.loc[worst_month_r, 'win_rate']*100:.1f}%, n={by_month_df.loc[worst_month_r, 'n_decided']:.0f})")
    print(f"Best month (win rate):  {best_month_wr}  {by_month_df.loc[best_month_wr, 'win_rate']*100:.1f}%  (n={by_month_df.loc[best_month_wr, 'n_decided']:.0f})")
    print(f"Worst month (win rate): {worst_month_wr}  {by_month_df.loc[worst_month_wr, 'win_rate']*100:.1f}%  (n={by_month_df.loc[worst_month_wr, 'n_decided']:.0f})")

    n_losing_months = (by_month_df["total_r"] < 0).sum()
    n_total_months = len(by_month_df)
    print(f"Losing months: {n_losing_months} of {n_total_months} ({n_losing_months/n_total_months*100:.1f}%)")

    # --- Year-by-year ---
    print("\nBy year:")
    for yr in sorted(decided["year"].unique()):
        sub = decided[decided["year"] == yr]
        s = summarize_outcomes(sub)
        print(f"  {yr}: n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:7.2f}  PF={s['profit_factor']:.2f}")

    # --- Day of week ---
    print("\nBy day of week:")
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        sub = decided[decided["dow"] == d]
        if sub.empty:
            continue
        s = summarize_outcomes(sub)
        print(f"  {d:9s} n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:7.2f}")

    # --- Direction (long vs short) ---
    print("\nBy direction:")
    for direction in decided["direction"].unique():
        sub = decided[decided["direction"] == direction]
        s = summarize_outcomes(sub)
        print(f"  {direction:6s} n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:7.2f}")
