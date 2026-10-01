# Reusable summary report for any resolved-trades DataFrame (output of
# backtest.resolve_trade_outcomes). Point this at a new entry idea's trades
# and get the same stats/breakdown every time, without rebuilding this by hand.

import pandas as pd


def add_session_tags(trades: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Tags each trade as 'NY' (9-11am ET), 'Asia' (8-10pm ET), or 'Other',
    matching the two session windows this whole project is built around."""
    out = trades.copy()
    et = pd.to_datetime(out[time_col]).dt.tz_convert("America/New_York")
    hour = et.dt.hour

    def tag(h):
        if 9 <= h < 11:
            return "NY"
        if 20 <= h < 22:
            return "Asia"
        return "Other"

    out["session"] = hour.map(tag)
    return out


def summarize_outcomes(resolved: pd.DataFrame) -> dict:
    """Core win/loss/breakeven stats + R performance for a resolved-trades frame."""
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven"])]
    n = len(decided)
    if n == 0:
        return {"n_decided": 0}

    wins = (decided["outcome"] == "win").sum()
    losses = (decided["outcome"] == "loss").sum()
    be = (decided["outcome"] == "breakeven").sum()
    gross_win_r = decided.loc[decided["outcome"] == "win", "r_multiple"].sum()
    gross_loss_r = decided.loc[decided["outcome"] == "loss", "r_multiple"].sum()  # negative

    return {
        "n_decided": n,
        "n_win": int(wins),
        "n_loss": int(losses),
        "n_breakeven": int(be),
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "breakeven_rate": be / n,
        "avg_r": decided["r_multiple"].mean(),
        "total_r": decided["r_multiple"].sum(),
        "profit_factor": (gross_win_r / abs(gross_loss_r)) if gross_loss_r != 0 else float("inf"),
    }


def print_report(resolved: pd.DataFrame) -> None:
    stats = summarize_outcomes(resolved)
    if stats["n_decided"] == 0:
        print("No decided trades to report on.")
        return

    print(f"Decided trades: {stats['n_decided']}")
    print(f"  Wins:      {stats['n_win']:4d}  ({stats['win_rate']*100:.1f}%)")
    print(f"  Losses:    {stats['n_loss']:4d}  ({stats['loss_rate']*100:.1f}%)")
    print(f"  Breakeven: {stats['n_breakeven']:4d}  ({stats['breakeven_rate']*100:.1f}%)")
    print(f"  Average R: {stats['avg_r']:.3f}")
    print(f"  Total R:   {stats['total_r']:.2f}")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")

    if "session" not in resolved.columns:
        resolved = add_session_tags(resolved)
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven"])]
    print("\nBy session:")
    for session in ["NY", "Asia", "Other"]:
        sub = decided[decided["session"] == session]
        if len(sub) == 0:
            continue
        s = summarize_outcomes(sub)
        print(f"  {session:6s} n={s['n_decided']:4d}  win_rate={s['win_rate']*100:5.1f}%  avg_R={s['avg_r']:.3f}  total_R={s['total_r']:.2f}")


def summarize_crt_outcomes(resolved: pd.DataFrame) -> dict:
    """Same shape as summarize_outcomes, but for CRT-strategy results: a
    "timeout" outcome (neither stop nor target hit in time) sits alongside
    win/loss instead of "breakeven", and - matching the Pine dashboard -
    win_rate is wins / (wins + losses), with timeouts excluded from that
    denominator rather than counted against the strategy."""
    decided = resolved[resolved["outcome"].isin(["win", "loss", "timeout"])]
    resolved_only = resolved[resolved["outcome"].isin(["win", "loss"])]
    n_resolved = len(resolved_only)
    if len(decided) == 0:
        return {"n_decided": 0}

    wins = (decided["outcome"] == "win").sum()
    losses = (decided["outcome"] == "loss").sum()
    timeouts = (decided["outcome"] == "timeout").sum()
    gross_win_r = decided.loc[decided["outcome"] == "win", "r_multiple"].sum()
    gross_loss_r = decided.loc[decided["outcome"] == "loss", "r_multiple"].sum()  # negative

    return {
        "n_decided": len(decided),
        "n_win": int(wins),
        "n_loss": int(losses),
        "n_timeout": int(timeouts),
        "win_rate": (wins / n_resolved) if n_resolved > 0 else float("nan"),
        "avg_r": decided["r_multiple"].mean(),
        "total_r": decided["r_multiple"].sum(),
        "profit_factor": (gross_win_r / abs(gross_loss_r)) if gross_loss_r != 0 else float("inf"),
    }


def max_drawdown_r(resolved: pd.DataFrame) -> float:
    """Largest peak-to-trough drawdown in cumulative R, walking trades in the
    order they closed. PF/win rate/total R can all look great while hiding a
    brutal losing stretch along the way - this is the number that catches
    that, and should always be judged alongside those, never skipped."""
    decided = resolved[resolved["outcome"].isin(["win", "loss", "breakeven", "timeout"])].sort_values("exit_time")
    if decided.empty:
        return 0.0
    cum = decided["r_multiple"].cumsum()
    drawdown = cum.cummax() - cum
    return float(drawdown.max())


def add_hour_tags(trades: pd.DataFrame, time_col: str = "time", tz: str = "America/New_York") -> pd.DataFrame:
    """Tags each trade with the hour (0-23, in `tz`) its entry fired in -
    finer-grained than add_session_tags' NY/Asia/Other buckets, for finding
    which specific hour of day actually performs best."""
    out = trades.copy()
    out["entry_hour"] = pd.to_datetime(out[time_col]).dt.tz_convert(tz).dt.hour
    return out
