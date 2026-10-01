# Walk-forward backtest for the fib-golden-zone RB strategy (see
# fib_golden_zone_strategy.py) - same in-sample/OOS split discipline as
# htf_sweep_backtest.py: 2023-2024 is the design period, 2025-present is
# genuine out-of-sample, and parameters are never tuned against the OOS file.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.fib_golden_zone_backtest

import pandas as pd

from .bot.sizing import compute_contracts
from .combined_analysis import longest_streak
from .data import load_1m_bars
from .fib_golden_zone_strategy import run_fib_golden_zone_strategy
from .report import max_drawdown_r, print_report, summarize_outcomes

NY_OPEN_START = "09:30"
NY_OPEN_END = "11:00"
NY_TZ = "America/New_York"

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

# MNQ micro contract specs. RISK_USD sits in the middle of the $100-150
# target range; MAX_CONTRACTS bounds the tail where a very tight stop would
# otherwise imply an unreasonable contract count.
TICK_SIZE = 0.25
TICK_VALUE = 0.50
RISK_USD = 125.0
MAX_CONTRACTS = 8
MAX_RISK_PER_TRADE_USD = 175.0
TARGET_DOLLAR_GOAL = 800.0

# Same conservative cost assumptions as htf_sweep_backtest.py - no live fill
# data exists yet for this variant either.
COMMISSION_PER_CONTRACT_RT = 0.74
SLIPPAGE_TICKS_PER_SIDE = 1.0


def add_dollar_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    contracts, dollar_pnl = [], []
    for _, row in out.iterrows():
        n = compute_contracts(
            row["entry_price"], row["stop_price"], TICK_SIZE, TICK_VALUE,
            RISK_USD, MAX_CONTRACTS, MAX_RISK_PER_TRADE_USD,
        )
        risk_per_contract = (abs(row["entry_price"] - row["stop_price"]) / TICK_SIZE) * TICK_VALUE
        still_open = row["outcome"] == "open_at_data_end"
        contracts.append(n)
        dollar_pnl.append(0.0 if still_open else row["r_multiple"] * risk_per_contract * n)
    out["contracts"] = contracts
    out["dollar_pnl"] = dollar_pnl
    return out


def apply_trading_costs(
    trades: pd.DataFrame,
    commission_per_contract_rt: float = COMMISSION_PER_CONTRACT_RT,
    slippage_ticks_per_side: float = SLIPPAGE_TICKS_PER_SIDE,
) -> pd.DataFrame:
    out = trades.copy()
    slippage_dollars_per_contract = slippage_ticks_per_side * 2 * TICK_VALUE
    cost_per_contract = commission_per_contract_rt + slippage_dollars_per_contract
    out["cost"] = out["contracts"] * cost_per_contract
    out["dollar_pnl_net"] = out["dollar_pnl"] - out["cost"]
    return out


def dollar_profit_factor(trades: pd.DataFrame, pnl_col: str) -> float:
    wins = trades.loc[trades[pnl_col] > 0, pnl_col].sum()
    losses = trades.loc[trades[pnl_col] < 0, pnl_col].sum()
    return wins / abs(losses) if losses != 0 else float("inf")


def ny_open_only(trades: pd.DataFrame) -> pd.DataFrame:
    """Slices to entries whose entry time falls in [NY_OPEN_START, NY_OPEN_END)
    ET - the NYSE cash-open window, minute-precise (not just the hour-bucket
    9-11am tagging report.add_session_tags does)."""
    if trades.empty:
        return trades
    local_time = pd.to_datetime(trades["time"]).dt.tz_convert(NY_TZ).dt.time
    start, end = pd.Timestamp(NY_OPEN_START).time(), pd.Timestamp(NY_OPEN_END).time()
    return trades[(local_time >= start) & (local_time < end)]


def report_period(label: str, trades: pd.DataFrame) -> dict:
    print("=" * 72)
    print(label)
    print("=" * 72)
    if trades.empty:
        print("No trades.\n")
        return {"n_decided": 0}

    trades = add_dollar_pnl(trades)
    trades = apply_trading_costs(trades)
    print_report(trades)

    stats = summarize_outcomes(trades)
    dd = max_drawdown_r(trades)
    total_dollar_net = trades["dollar_pnl_net"].sum()
    pf_dollar_net = dollar_profit_factor(trades, "dollar_pnl_net")
    n_days = (trades["time"].max() - trades["time"].min()).days or 1
    trades_per_week = stats["n_decided"] / (n_days / 7)

    decided = trades[trades["outcome"].isin(["win", "loss", "breakeven"])].sort_values("exit_time")
    wins = decided[decided["outcome"] == "win"]
    losses = decided[decided["outcome"] == "loss"]
    avg_stop_pts = decided["stop_distance_points"].mean()
    p90_stop_pts = decided["stop_distance_points"].quantile(0.90)
    avg_win_dollar = wins["dollar_pnl_net"].mean() if len(wins) else float("nan")
    avg_loss_dollar = losses["dollar_pnl_net"].mean() if len(losses) else float("nan")
    avg_contracts = decided["contracts"].mean()

    win_streak = longest_streak(decided["outcome"], "win")
    loss_streak = longest_streak(decided["outcome"], "loss")
    avg_win_r = wins["r_multiple"].mean() if len(wins) else float("nan")
    avg_loss_r = losses["r_multiple"].mean() if len(losses) else float("nan")
    best_r = decided["r_multiple"].max()
    worst_r = decided["r_multiple"].min()
    std_r = decided["r_multiple"].std()

    print(f"\nMax drawdown: {dd:.2f}R")
    print(f"Trade frequency: {stats['n_decided']} trades over {n_days} days (~{trades_per_week:.1f}/week)")
    print(f"Avg contracts/trade (${RISK_USD:.0f} risk target): {avg_contracts:.2f}")
    print(f"Stop distance: avg {avg_stop_pts:.1f} pts, 90th pct {p90_stop_pts:.1f} pts")
    print(f"Longest winning streak: {win_streak} trades   |   Longest losing streak: {loss_streak} trades")
    print(f"Avg win:  {avg_win_r:+.2f}R  (${avg_win_dollar:,.2f} net)   |   goal: ${TARGET_DOLLAR_GOAL:,.0f}+/winner")
    print(f"Avg loss: {avg_loss_r:+.2f}R  (${avg_loss_dollar:,.2f} net)")
    print(f"Best trade: {best_r:+.2f}R   |   Worst trade: {worst_r:+.2f}R   |   Std dev (R): {std_r:.2f}")
    print(f"Total net $ P&L (after commission+slippage): ${total_dollar_net:,.2f}")
    print(f"Net $ profit factor (after costs): {pf_dollar_net:.2f}\n")

    stats["max_drawdown_r"] = dd
    stats["total_dollar_pnl_net"] = total_dollar_net
    stats["dollar_profit_factor_net"] = pf_dollar_net
    stats["trades_per_week"] = trades_per_week
    stats["avg_stop_pts"] = avg_stop_pts
    stats["win_streak"] = win_streak
    stats["loss_streak"] = loss_streak
    stats["avg_win_dollar"] = avg_win_dollar
    stats["avg_loss_dollar"] = avg_loss_dollar
    return stats


if __name__ == "__main__":
    df_in_sample = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)
    print(f"In-sample:     {len(df_in_sample):,} bars, {df_in_sample.index.min()} -> {df_in_sample.index.max()}")
    print(f"Out-of-sample: {len(df_oos):,} bars, {df_oos.index.min()} -> {df_oos.index.max()}\n")

    # entry_mode="limit" is the function default - a resting limit order at
    # the 0.705 fib level, which is how you said you'd actually trade this,
    # not a market entry that waits for a closed rejection candle.
    trades_in_sample_full = run_fib_golden_zone_strategy(df_in_sample)
    trades_oos_full = run_fib_golden_zone_strategy(df_oos)

    # Primary: restricted to the 9:30-11:00am ET NYSE-open window only, per
    # your request - this is what all the headline stats below are.
    trades_in_sample = ny_open_only(trades_in_sample_full)
    trades_oos = ny_open_only(trades_oos_full)

    if not trades_in_sample.empty:
        trades_in_sample.to_csv("data/fib_golden_zone_in_sample_930_1100_trades.csv", index=False)
    if not trades_oos.empty:
        trades_oos.to_csv("data/fib_golden_zone_oos_930_1100_trades.csv", index=False)

    stats_in = report_period("IN-SAMPLE: 2023-2024, 9:30-11:00am ET only (limit entry @ 0.705)", trades_in_sample)
    stats_oos = report_period("OUT-OF-SAMPLE: 2025-present, 9:30-11:00am ET only (limit entry @ 0.705)", trades_oos)

    print("=" * 72)
    print("WALK-FORWARD CHECK (9:30-11:00am ET only)")
    print("=" * 72)
    if stats_in.get("n_decided", 0) and stats_oos.get("n_decided", 0):
        print(f"  Win rate:      in-sample {stats_in['win_rate']*100:.1f}%  vs  OOS {stats_oos['win_rate']*100:.1f}%")
        print(f"  Avg R:         in-sample {stats_in['avg_r']:.3f}  vs  OOS {stats_oos['avg_r']:.3f}")
        print(f"  Profit factor (R): in-sample {stats_in['profit_factor']:.2f}  vs  OOS {stats_oos['profit_factor']:.2f}")
        print(f"  Net $ PF:      in-sample {stats_in['dollar_profit_factor_net']:.2f}  vs  OOS {stats_oos['dollar_profit_factor_net']:.2f}")
        print(f"  Max drawdown:  in-sample {stats_in['max_drawdown_r']:.2f}R  vs  OOS {stats_oos['max_drawdown_r']:.2f}R")
        if stats_oos["avg_r"] < stats_in["avg_r"] * 0.5 or (stats_oos["profit_factor"] < 1.0 <= stats_in["profit_factor"]):
            print("\n  WARNING: OOS materially worse than in-sample - possible overfitting, don't trust this config yet.")
    else:
        print("  Not enough trades in one or both periods to compare.")

    # For context only: the same two periods with no time-of-day restriction.
    report_period("(for comparison) IN-SAMPLE: 2023-2024, full day", trades_in_sample_full)
    report_period("(for comparison) OUT-OF-SAMPLE: 2025-present, full day", trades_oos_full)

    # Compact comparison matrix: entry_mode x require_clean_zone, in-sample
    # only (never tune against OOS). Reported as summary stats, not a full
    # report_period each, since this is meant to be scanned, not read in
    # depth - whichever config looks best here should get its own full,
    # OOS-checked report_period run separately before being trusted.
    print("=" * 72)
    print("CONFIG SWEEP (in-sample only - exploratory, not OOS-validated)")
    print("=" * 72)
    header = f"{'entry_mode':<12}{'clean_only':<12}{'n':>6}{'win_rate':>10}{'breakeven':>11}{'avg_R':>9}{'PF(R)':>8}"
    print(header)
    for mode in ["limit", "rb_confirm"]:
        for clean_only in [False, True]:
            t = run_fib_golden_zone_strategy(df_in_sample, entry_mode=mode, require_clean_zone=clean_only)
            s = summarize_outcomes(t)
            n = s.get("n_decided", 0)
            if n:
                breakeven = 1 / (1 + 8.0)  # target_r_multiple default
                print(f"{mode:<12}{str(clean_only):<12}{n:>6}{s['win_rate']*100:>9.1f}%{breakeven*100:>10.1f}%{s['avg_r']:>9.3f}{s['profit_factor']:>8.2f}")
            else:
                print(f"{mode:<12}{str(clean_only):<12}{n:>6}   (no trades)")
