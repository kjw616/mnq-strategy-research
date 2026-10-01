# Walk-forward backtest for the new HTF-bias sweep+displacement strategy
# (see htf_sweep_strategy.py) - runs 2023-2024 (in-sample/design) and
# 2025-present (genuine out-of-sample) completely independently, never
# tuning parameters against the OOS file. A real effect holds up in both; if
# one period keeps improving while the other plateaus or reverses, that's
# overfitting to noise in whichever period looks better, not a real edge.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.htf_sweep_backtest

import pandas as pd

from .bot.sizing import compute_contracts
from .data import load_1m_bars
from .htf_sweep_strategy import run_htf_sweep_strategy
from .report import max_drawdown_r, print_report, summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

# MNQ micro contract specs + example fixed-dollar-risk sizing, matching
# bot_config.example.json - real $ P&L needs actual position sizing, not
# just R-multiples (tight-stop trades force heavier size, and commissions/
# slippage scale with contract count).
TICK_SIZE = 0.25
TICK_VALUE = 0.50
RISK_USD = 125.0
MAX_CONTRACTS = 5
MAX_RISK_PER_TRADE_USD = 250.0

# Commission + slippage assumptions - no live fill data exists yet, so these
# are deliberately conservative-but-plausible ballpark figures for a
# low-cost futures broker (e.g. AMP/Tradovate-tier commission around
# $0.35-0.85/side; using the cheaper end at $0.37/side = $0.74 round-trip).
# Slippage assumes 1 tick of adverse fill on EACH side (entry and exit) -
# a real assumption for a market/stop order in a liquid micro contract like
# MNQ, not a guess in either direction. Both are trivially overridable.
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
    """Deducts commission + slippage from each trade's dollar_pnl. Requires
    add_dollar_pnl() to have already run (needs 'contracts' and
    'dollar_pnl'). Slippage is modeled as adverse fill on BOTH entry and
    exit, in ticks - the round-trip cost scales with position size the same
    way commission does, since both are genuinely per-contract costs."""
    out = trades.copy()
    slippage_dollars_per_contract = slippage_ticks_per_side * 2 * TICK_VALUE  # entry + exit
    cost_per_contract = commission_per_contract_rt + slippage_dollars_per_contract
    out["cost"] = out["contracts"] * cost_per_contract
    out["dollar_pnl_net"] = out["dollar_pnl"] - out["cost"]
    return out


def dollar_profit_factor(trades: pd.DataFrame, pnl_col: str) -> float:
    """Profit factor computed in real dollars (post-cost) rather than
    R-multiples - the number that actually determines whether a strategy
    survives commissions/slippage, not just whether its signal is good."""
    wins = trades.loc[trades[pnl_col] > 0, pnl_col].sum()
    losses = trades.loc[trades[pnl_col] < 0, pnl_col].sum()
    return wins / abs(losses) if losses != 0 else float("inf")


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
    print(f"Total $ P&L (${RISK_USD:.0f} risk/trade, max {MAX_CONTRACTS} contracts): ${total_dollar:,.2f}\n")

    stats["max_drawdown_r"] = dd
    stats["total_dollar_pnl"] = total_dollar
    stats["trades_per_week"] = trades_per_week
    return stats


if __name__ == "__main__":
    df_in_sample = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)
    print(f"In-sample:     {len(df_in_sample):,} bars, {df_in_sample.index.min()} -> {df_in_sample.index.max()}")
    print(f"Out-of-sample: {len(df_oos):,} bars, {df_oos.index.min()} -> {df_oos.index.max()}\n")

    trades_in_sample = run_htf_sweep_strategy(df_in_sample)
    trades_oos = run_htf_sweep_strategy(df_oos)

    if not trades_in_sample.empty:
        trades_in_sample.to_csv("data/htf_sweep_in_sample_trades.csv", index=False)
    if not trades_oos.empty:
        trades_oos.to_csv("data/htf_sweep_oos_trades.csv", index=False)

    stats_in = report_period("IN-SAMPLE: 2023-2024", trades_in_sample)
    stats_oos = report_period("OUT-OF-SAMPLE: 2025-present", trades_oos)

    print("=" * 72)
    print("WALK-FORWARD CHECK")
    print("=" * 72)
    if stats_in.get("n_decided", 0) and stats_oos.get("n_decided", 0):
        print(f"  Win rate:      in-sample {stats_in['win_rate']*100:.1f}%  vs  OOS {stats_oos['win_rate']*100:.1f}%")
        print(f"  Avg R:         in-sample {stats_in['avg_r']:.3f}  vs  OOS {stats_oos['avg_r']:.3f}")
        print(f"  Profit factor: in-sample {stats_in['profit_factor']:.2f}  vs  OOS {stats_oos['profit_factor']:.2f}")
        print(f"  Max drawdown:  in-sample {stats_in['max_drawdown_r']:.2f}R  vs  OOS {stats_oos['max_drawdown_r']:.2f}R")
        if stats_oos["avg_r"] < stats_in["avg_r"] * 0.5 or (stats_oos["profit_factor"] < 1.0 <= stats_in["profit_factor"]):
            print("\n  WARNING: OOS materially worse than in-sample - possible overfitting, don't trust this config yet.")
    else:
        print("  Not enough trades in one or both periods to compare.")
