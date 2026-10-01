# Backtest for the 9/21 EMA-cross-at-open theory (see ema_cross_strategy.py).
# Three configs, each run independently in-sample (2023-2024) and
# out-of-sample (2025-present), same walk-forward discipline as
# htf_sweep_backtest.py - never tune against the OOS file:
#   1. raw       - the cross-at-open signal alone, no confluence filter.
#   2. vwap      - + require price already on the cross's side of VWAP.
#   3. vwap_htf  - + require 5m/30m structure bias to also agree (the
#                  "direction bias" layer this codebase's more validated
#                  htf_sweep strategy already leans on).
#
# Target is 1:1 R (matching the stop distance exactly). Since a 1:1 target
# and this codebase's default breakeven_r=1.0 would sit at the *same* price
# (the breakeven-move and the target become the same trigger, and
# resolve_trade_outcomes conservatively resolves any same-bar ambiguity as
# "stop side happened first"), breakeven_r here is scaled to half the
# target instead - moves the stop to entry once halfway there, not exactly
# at the target itself.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_backtest

import pandas as pd

from .backtest import resolve_trade_outcomes
from .bot.sizing import compute_contracts
from .data import load_1m_bars
from .ema_cross_strategy import run_ema_cross_strategy
from .report import max_drawdown_r, print_report, summarize_outcomes

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

# MNQ micro contract specs + example fixed-dollar-risk sizing, matching
# htf_sweep_backtest.py's assumptions for comparability across strategies.
TICK_SIZE = 0.25
TICK_VALUE = 0.50
RISK_USD = 125.0
MAX_CONTRACTS = 5
MAX_RISK_PER_TRADE_USD = 250.0
COMMISSION_PER_CONTRACT_RT = 0.74
SLIPPAGE_TICKS_PER_SIDE = 1.0

TARGET_R_MULTIPLE = 1.0
# Breakeven-stop management is disabled (not scaled to the target) for a 1:1
# target: with breakeven_r < 1.0, a trade that pulls back to entry after
# crossing the breakeven trigger exits as "breakeven" immediately - even if
# it would have gone on to reach the (nearby) target later - which silently
# converts a chunk of would-be wins into breakevens and made results *worse*
# at 1:1 than at 2:1 the first time this ran. A 1:1 target is already tight
# enough that stop/target alone is the fair, uncomplicated test.
BREAKEVEN_R = 1e6

CONFIGS = {
    "raw": dict(require_vwap_confluence=False, require_htf_bias_confluence=False),
    "vwap": dict(require_vwap_confluence=True, require_htf_bias_confluence=False),
    "vwap_htf": dict(require_vwap_confluence=True, require_htf_bias_confluence=True),
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


def run_and_report(label: str, df: pd.DataFrame, strategy_kwargs: dict) -> tuple[dict, pd.DataFrame]:
    print("=" * 72)
    print(label)
    print("=" * 72)

    trades = run_ema_cross_strategy(df, target_r_multiple=TARGET_R_MULTIPLE, **strategy_kwargs)
    if trades.empty:
        print("No trades.\n")
        return {"n_decided": 0}, trades

    resolved = resolve_trade_outcomes(trades, df, breakeven_r=BREAKEVEN_R)
    resolved = add_dollar_pnl(resolved)
    print_report(resolved)

    stats = summarize_outcomes(resolved)
    dd = max_drawdown_r(resolved)
    total_dollar = resolved["dollar_pnl"].sum()
    n_days = (resolved["time"].max() - resolved["time"].min()).days or 1
    print(f"\nMax drawdown: {dd:.2f}R")
    print(f"Trade frequency: {stats['n_decided']} trades over {n_days} days (~{stats['n_decided'] / (n_days / 365):.0f}/year)")
    print(f"Total $ P&L (${RISK_USD:.0f} risk/trade, max {MAX_CONTRACTS} contracts): ${total_dollar:,.2f}\n")

    stats["max_drawdown_r"] = dd
    stats["total_dollar_pnl"] = total_dollar
    return stats, resolved


def compare(label_prefix: str, stats_in: dict, stats_oos: dict) -> None:
    print("=" * 72)
    print(f"{label_prefix} - WALK-FORWARD CHECK")
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
    print()


if __name__ == "__main__":
    df_in_sample = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)
    print(f"In-sample:     {len(df_in_sample):,} bars, {df_in_sample.index.min()} -> {df_in_sample.index.max()}")
    print(f"Out-of-sample: {len(df_oos):,} bars, {df_oos.index.min()} -> {df_oos.index.max()}\n")

    for name, kwargs in CONFIGS.items():
        stats_in, resolved_in = run_and_report(f"IN-SAMPLE: 2023-2024 - {name}", df_in_sample, kwargs)
        stats_oos, resolved_oos = run_and_report(f"OUT-OF-SAMPLE: 2025-present - {name}", df_oos, kwargs)

        if not resolved_in.empty:
            resolved_in.to_csv(f"data/ema_cross_{name}_in_sample_trades.csv", index=False)
        if not resolved_oos.empty:
            resolved_oos.to_csv(f"data/ema_cross_{name}_oos_trades.csv", index=False)

        compare(name, stats_in, stats_oos)
