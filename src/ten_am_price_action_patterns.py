# Pure price-action study of the 10:00-10:30am ET window, no strategy or
# indicator involved - just what price actually tends to do there, day by
# day, compared to the half hour right before it (9:30-10:00) and the rest
# of the morning. Four questions, each with a real yes/no or measurable
# answer per trading day:
#
#   1. Continuation vs reversal - does the 10:00-10:30 move tend to run the
#      SAME direction as the 9:30-10:00 move (momentum), or the OPPOSITE
#      (mean reversion)? Tested against a 50% coin-flip baseline, same
#      significance test validate_ema_cross.py already uses.
#   2. Liquidity sweep behavior - does 10:00-10:30 trade beyond the prior
#      half hour's high or low (sweep it), and if so, does it reverse back
#      inside that range by the close (a real ICT-style sweep-and-reject)
#      or hold beyond it (a genuine breakout continuation)?
#   3. Session-extreme frequency - how often does the 10:00-10:30 window
#      contain the actual high or low of the whole AM session (9:30-12:00)?
#      If this window disproportionately produces the day's turning point,
#      that's worth knowing regardless of any specific entry strategy.
#   4. Directional bias - does 10:00-10:30 skew up or down more often, and
#      by how much on each side.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ten_am_price_action_patterns

import math

import numpy as np
import pandas as pd

from .data import load_1m_bars

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"
TZ = "America/New_York"

PRIOR_START, PRIOR_END = "09:30", "10:00"
WINDOW_START, WINDOW_END = "10:00", "10:30"
SESSION_START, SESSION_END = "09:30", "12:00"


def _binom_two_sided_p(hits: int, n: int, p0: float = 0.5) -> float:
    if n == 0:
        return float("nan")
    phat = hits / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan")
    z = (phat - p0) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _window_agg(df_1m: pd.DataFrame, local: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    t = local.time
    s, e = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    mask = (t >= s) & (t < e)
    sub = pd.DataFrame({
        "date": local.normalize()[mask],
        "open": df_1m["open"].to_numpy()[mask], "close": df_1m["close"].to_numpy()[mask],
        "high": df_1m["high"].to_numpy()[mask], "low": df_1m["low"].to_numpy()[mask],
    })
    return sub.groupby("date").agg(open=("open", "first"), close=("close", "last"),
                                    high=("high", "max"), low=("low", "min"))


def build_daily_table(df_1m: pd.DataFrame) -> pd.DataFrame:
    local = df_1m.index.tz_convert(TZ)
    prior = _window_agg(df_1m, local, PRIOR_START, PRIOR_END).add_prefix("prior_")
    window = _window_agg(df_1m, local, WINDOW_START, WINDOW_END).add_prefix("win_")
    session = _window_agg(df_1m, local, SESSION_START, SESSION_END).add_prefix("sess_")

    daily = prior.join(window, how="inner").join(session, how="inner")
    daily["prior_move"] = daily["prior_close"] - daily["prior_open"]
    daily["win_move"] = daily["win_close"] - daily["win_open"]
    daily["same_direction"] = np.sign(daily["prior_move"]) == np.sign(daily["win_move"])
    daily["swept_prior_high"] = daily["win_high"] > daily["prior_high"]
    daily["swept_prior_low"] = daily["win_low"] < daily["prior_low"]
    daily["rejected_after_high_sweep"] = daily["swept_prior_high"] & (daily["win_close"] < daily["prior_high"])
    daily["held_after_high_sweep"] = daily["swept_prior_high"] & (daily["win_close"] >= daily["prior_high"])
    daily["rejected_after_low_sweep"] = daily["swept_prior_low"] & (daily["win_close"] > daily["prior_low"])
    daily["held_after_low_sweep"] = daily["swept_prior_low"] & (daily["win_close"] <= daily["prior_low"])
    daily["win_has_session_high"] = daily["win_high"] >= daily["sess_high"]
    daily["win_has_session_low"] = daily["win_low"] <= daily["sess_low"]
    return daily.dropna(subset=["prior_move", "win_move"])


def report(label: str, daily: pd.DataFrame) -> None:
    n = len(daily)
    print(f"\n{'='*78}\n{label}  (n={n} trading days)\n{'='*78}")

    # 1. continuation vs reversal
    same = int(daily["same_direction"].sum())
    p = _binom_two_sided_p(same, n, 0.5)
    avg_move_same = daily.loc[daily["same_direction"], "win_move"].abs().mean()
    avg_move_opp = daily.loc[~daily["same_direction"], "win_move"].abs().mean()
    print(f"\n1. Continuation vs reversal (10:00-10:30 same direction as 9:30-10:00):")
    print(f"   Same direction: {same}/{n} ({same/n*100:.1f}%)  vs 50%: p={p:.4f}")
    print(f"   Avg |move| when continuing: {avg_move_same:.1f}pts   when reversing: {avg_move_opp:.1f}pts")

    # 2. sweep behavior
    for side in ["high", "low"]:
        swept = daily[f"swept_prior_{side}"]
        n_swept = int(swept.sum())
        rejected = int(daily[f"rejected_after_{side}_sweep"].sum())
        held = int(daily[f"held_after_{side}_sweep"].sum())
        print(f"\n2. {side.upper()} of 9:30-10:00 swept during 10:00-10:30: {n_swept}/{n} ({n_swept/n*100:.1f}%)")
        if n_swept:
            print(f"   -> rejected back inside (sweep+reverse): {rejected}/{n_swept} ({rejected/n_swept*100:.1f}%)")
            print(f"   -> held beyond it (real breakout):        {held}/{n_swept} ({held/n_swept*100:.1f}%)")

    # 3. session-extreme frequency
    has_high = int(daily["win_has_session_high"].sum())
    has_low = int(daily["win_has_session_low"].sum())
    either = int((daily["win_has_session_high"] | daily["win_has_session_low"]).sum())
    print(f"\n3. 10:00-10:30 sets the AM session (9:30-12:00) extreme:")
    print(f"   Session high: {has_high}/{n} ({has_high/n*100:.1f}%)   "
          f"Session low: {has_low}/{n} ({has_low/n*100:.1f}%)   "
          f"Either: {either}/{n} ({either/n*100:.1f}%)")
    print(f"   (baseline if extremes were uniformly likely across a ~2.5h AM session: ~{30/150*100:.1f}% per side)")

    # 4. directional bias
    up_days = int((daily["win_move"] > 0).sum())
    down_days = int((daily["win_move"] < 0).sum())
    p_dir = _binom_two_sided_p(up_days, up_days + down_days, 0.5)
    print(f"\n4. Directional bias of the 10:00-10:30 move itself:")
    print(f"   Up days: {up_days}/{n} ({up_days/n*100:.1f}%)   Down days: {down_days}/{n} ({down_days/n*100:.1f}%)  "
          f"vs 50%: p={p_dir:.4f}")
    print(f"   Avg up move: {daily.loc[daily['win_move']>0,'win_move'].mean():+.1f}pts   "
          f"Avg down move: {daily.loc[daily['win_move']<0,'win_move'].mean():+.1f}pts")
    print(f"   Std dev of move: {daily['win_move'].std():.1f}pts   Median |move|: {daily['win_move'].abs().median():.1f}pts")


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    daily_in = build_daily_table(df_in)
    daily_oos = build_daily_table(df_oos)

    report("IN-SAMPLE 2023-2024", daily_in)
    report("OOS 2025-present", daily_oos)
