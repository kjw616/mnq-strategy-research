# Does layering the 9/21/50 EMA-stack regime filter (trend.
# ema_stack_alignment, computed on 5-minute bars) on top of htf_sweep_
# strategy's existing structure-bias signal actually improve it, or just
# cut trade count for no gain? Same walk-forward discipline as htf_sweep_
# backtest.py - in-sample (2023-2024) and out-of-sample (2025-present)
# scored independently, never tuned against the OOS file.
#
# Baseline config here is htf_sweep_strategy's own defaults (use_htf_bias=
# True, use_session_levels=True, use_swing_pool=True) - this isolates the
# EMA-stack filter's marginal effect on top of the strategy's already-
# existing directional gate, not a comparison against no filter at all.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.htf_sweep_ema_filter_backtest

import pandas as pd

from .bot.sizing import compute_contracts
from .data import load_1m_bars
from .htf_sweep_strategy import run_htf_sweep_strategy
from .report import max_drawdown_r, print_report, summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

# Matches htf_sweep_backtest.py's own MNQ sizing assumptions.
TICK_SIZE = 0.25
TICK_VALUE = 0.50
RISK_USD = 125.0
MAX_CONTRACTS = 5
MAX_RISK_PER_TRADE_USD = 250.0

CONFIGS = {
    "baseline": dict(use_ema_stack_filter=False),
    "ema_stack_filter": dict(use_ema_stack_filter=True),
}


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


def report_period(label: str, trades: pd.DataFrame) -> dict:
    print("=" * 72)
    print(label)
    print("=" * 72)
    if trades.empty:
        print("No trades.\n")
        return {"n_decided": 0}

    trades = add_dollar_pnl(trades)
    print_report(trades)

    stats = summarize_outcomes(trades)
    dd = max_drawdown_r(trades)
    total_dollar = trades["dollar_pnl"].sum()
    n_days = (trades["time"].max() - trades["time"].min()).days or 1
    trades_per_week = stats["n_decided"] / (n_days / 7)

    print(f"\nMax drawdown: {dd:.2f}R")
    print(f"Trade frequency: {stats['n_decided']} trades over {n_days} days (~{trades_per_week:.1f}/week)")
    print(f"Total $ P&L: ${total_dollar:,.2f}\n")

    stats["max_drawdown_r"] = dd
    stats["total_dollar_pnl"] = total_dollar
    stats["trades_per_week"] = trades_per_week
    return stats


def compare_walk_forward(label: str, stats_in: dict, stats_oos: dict) -> None:
    print("=" * 72)
    print(f"{label} - WALK-FORWARD CHECK")
    print("=" * 72)
    if stats_in.get("n_decided", 0) and stats_oos.get("n_decided", 0):
        print(f"  Win rate:      in-sample {stats_in['win_rate']*100:.1f}%  vs  OOS {stats_oos['win_rate']*100:.1f}%")
        print(f"  Avg R:         in-sample {stats_in['avg_r']:.3f}  vs  OOS {stats_oos['avg_r']:.3f}")
        print(f"  Profit factor: in-sample {stats_in['profit_factor']:.2f}  vs  OOS {stats_oos['profit_factor']:.2f}")
        print(f"  Max drawdown:  in-sample {stats_in['max_drawdown_r']:.2f}R  vs  OOS {stats_oos['max_drawdown_r']:.2f}R")
    else:
        print("  Not enough trades in one or both periods to compare.")
    print()


def compare_filter_effect(period_label: str, stats_baseline: dict, stats_filtered: dict) -> None:
    print("=" * 72)
    print(f"{period_label} - DOES THE EMA-STACK FILTER HELP?")
    print("=" * 72)
    if stats_baseline.get("n_decided", 0) and stats_filtered.get("n_decided", 0):
        print(f"  Trade count:   baseline {stats_baseline['n_decided']}  ->  filtered {stats_filtered['n_decided']} "
              f"({stats_filtered['n_decided'] / stats_baseline['n_decided'] * 100:.0f}% kept)")
        print(f"  Win rate:      baseline {stats_baseline['win_rate']*100:.1f}%  ->  filtered {stats_filtered['win_rate']*100:.1f}%")
        print(f"  Avg R:         baseline {stats_baseline['avg_r']:.3f}  ->  filtered {stats_filtered['avg_r']:.3f}")
        print(f"  Profit factor: baseline {stats_baseline['profit_factor']:.2f}  ->  filtered {stats_filtered['profit_factor']:.2f}")
    else:
        print("  Not enough trades to compare.")
    print()


if __name__ == "__main__":
    df_in_sample = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)
    print(f"In-sample:     {len(df_in_sample):,} bars, {df_in_sample.index.min()} -> {df_in_sample.index.max()}")
    print(f"Out-of-sample: {len(df_oos):,} bars, {df_oos.index.min()} -> {df_oos.index.max()}\n")

    stats = {}
    for name, kwargs in CONFIGS.items():
        trades_in = run_htf_sweep_strategy(df_in_sample, **kwargs)
        trades_oos = run_htf_sweep_strategy(df_oos, **kwargs)

        stats_in = report_period(f"IN-SAMPLE: 2023-2024 - {name}", trades_in)
        stats_oos = report_period(f"OUT-OF-SAMPLE: 2025-present - {name}", trades_oos)
        stats[name] = {"in": stats_in, "oos": stats_oos}

        compare_walk_forward(name, stats_in, stats_oos)

    compare_filter_effect("IN-SAMPLE: 2023-2024", stats["baseline"]["in"], stats["ema_stack_filter"]["in"])
    compare_filter_effect("OUT-OF-SAMPLE: 2025-present", stats["baseline"]["oos"], stats["ema_stack_filter"]["oos"])
