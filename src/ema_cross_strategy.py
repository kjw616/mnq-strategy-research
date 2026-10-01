# Tests a specific opening-range theory: when the 9 EMA crosses the 21 EMA
# within a few minutes of the NY session open, does that cross mark the
# start of a real trend for the rest of the session? Deliberately isolated
# from this codebase's other confluences (PD arrays, structure_bias) - the
# point is to find out whether the EMA-cross-at-open signal itself has any
# edge before stacking anything else on top of it. Two optional confluence
# layers can be stacked on top, each independently toggleable:
#   - VWAP position: price already on the cross's side of session VWAP.
#   - HTF structure bias: 5m AND 30m structure_bias() agree with the cross
#     direction, reusing htf_sweep_strategy's htf_bias_agreement (the same
#     "direction bias" this codebase's more validated strategy already
#     relies on) rather than re-deriving a second version of it here.
#
# One trade per session, max: the first qualifying cross inside the open
# window each day. A candidate that fails a confluence filter does NOT count
# against that limit, so a later, agreeing cross in the same window can
# still trigger - only a cross that's actually taken closes the session out.

import numpy as np
import pandas as pd

from .crt_strategy import wilder_atr
from .htf_sweep_strategy import htf_bias_agreement
from .indicators import session_vwap


def ema_cross_signal(close: pd.Series, fast_len: int = 9, slow_len: int = 21) -> pd.DataFrame:
    """9/21 EMA values plus cross_up/cross_down booleans - fires only when the
    fast/slow relationship actually flips sign, not just whenever fast sits
    above slow (which would be true on every bar of an existing trend)."""
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()
    diff = ema_fast - ema_slow
    prev_diff = diff.shift(1)
    cross_up = (diff > 0) & (prev_diff <= 0)
    cross_down = (diff < 0) & (prev_diff >= 0)
    return pd.DataFrame({
        "ema_fast": ema_fast, "ema_slow": ema_slow,
        "cross_up": cross_up, "cross_down": cross_down,
    }, index=close.index)


def find_cross_candidates(
    df_1m: pd.DataFrame,
    fast_len: int = 9,
    slow_len: int = 21,
    market_open_hour: int = 9,
    market_open_minute: int = 30,
    window_minutes: int = 30,
    require_vwap_confluence: bool = True,
    require_htf_bias_confluence: bool = False,
    bias_5m_length: int = 3,
    bias_30m_length: int = 3,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """The signal itself, with no stop/target/risk mechanics attached: the
    first 9/21 EMA cross each session that falls inside [market_open,
    market_open + window_minutes) in `tz`, optionally filtered by VWAP
    position and/or 5m+30m structure bias agreement (see module docstring).
    One row per qualifying session at most - a candidate that fails a
    confluence filter is skipped without closing the session out, so a
    later, agreeing cross in the same window can still qualify.

    This is the single source of truth for "what counts as a qualifying
    cross" - run_ema_cross_strategy (adds ATR stop/target for a tradeable
    backtest) and validate_ema_cross (adds forward-return lookups for a
    pure directional-accuracy check) both build on this exact same set of
    events, so neither can silently drift from the other's definition.
    """
    sig = ema_cross_signal(df_1m["close"], fast_len, slow_len)
    vwap = session_vwap(df_1m)
    htf_bias = htf_bias_agreement(df_1m, bias_5m_length, bias_30m_length) if require_htf_bias_confluence else None

    local_time = df_1m.index.tz_convert(tz)
    open_minutes_of_day = market_open_hour * 60 + market_open_minute
    minutes_of_day = local_time.hour * 60 + local_time.minute
    in_window = (minutes_of_day >= open_minutes_of_day) & (minutes_of_day < open_minutes_of_day + window_minutes)
    session_date = local_time.normalize()

    closes = df_1m["close"].to_numpy()
    cross_up = sig["cross_up"].to_numpy()
    cross_down = sig["cross_down"].to_numpy()
    ema_fast = sig["ema_fast"].to_numpy()
    ema_slow = sig["ema_slow"].to_numpy()
    vwap_arr = vwap.to_numpy()
    in_window_arr = np.asarray(in_window)
    idx = df_1m.index
    n = len(df_1m)

    used_sessions: set = set()
    candidates: list[dict] = []

    for i in range(n):
        if not in_window_arr[i]:
            continue
        session_key = session_date[i]
        if session_key in used_sessions:
            continue
        if not (cross_up[i] or cross_down[i]):
            continue

        is_long = bool(cross_up[i])
        close = closes[i]

        if require_vwap_confluence:
            v = vwap_arr[i]
            if np.isnan(v) or (is_long and close <= v) or (not is_long and close >= v):
                continue

        if require_htf_bias_confluence:
            wanted = 1 if is_long else -1
            if htf_bias[i] != wanted:
                continue

        used_sessions.add(session_key)
        candidates.append({
            "time": idx[i], "direction": "long" if is_long else "short",
            "entry_price": close, "bar_index": i,
            "ema_fast": ema_fast[i], "ema_slow": ema_slow[i], "vwap": vwap_arr[i],
        })

    return pd.DataFrame(candidates)


def run_ema_cross_strategy(
    df_1m: pd.DataFrame,
    fast_len: int = 9,
    slow_len: int = 21,
    market_open_hour: int = 9,
    market_open_minute: int = 30,
    window_minutes: int = 30,
    require_vwap_confluence: bool = True,
    require_htf_bias_confluence: bool = False,
    bias_5m_length: int = 3,
    bias_30m_length: int = 3,
    atr_len: int = 14,
    stop_atr_mult: float = 1.5,
    target_r_multiple: float = 1.0,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Attaches an ATR-based stop/target to each find_cross_candidates()
    event - plain ATR risk, not a swing-level lookup, since the goal here is
    isolating the EMA-cross signal itself. Returns a trades frame shaped for
    backtest.resolve_trade_outcomes (entry_price, stop_price, target_price,
    bar_index, direction) - call that next to get win/loss/breakeven +
    r_multiple.
    """
    candidates = find_cross_candidates(
        df_1m, fast_len, slow_len, market_open_hour, market_open_minute, window_minutes,
        require_vwap_confluence, require_htf_bias_confluence, bias_5m_length, bias_30m_length, tz,
    )
    if candidates.empty:
        return candidates

    atr = wilder_atr(df_1m, atr_len)
    atr_at_entry = atr.to_numpy()[candidates["bar_index"].to_numpy()]
    valid = ~np.isnan(atr_at_entry) & (atr_at_entry > 0)
    candidates = candidates[valid].copy()
    atr_at_entry = atr_at_entry[valid]

    is_long = (candidates["direction"] == "long").to_numpy()
    stop_distance = atr_at_entry * stop_atr_mult
    entry_price = candidates["entry_price"].to_numpy()
    stop_price = np.where(is_long, entry_price - stop_distance, entry_price + stop_distance)
    target_price = np.where(
        is_long, entry_price + stop_distance * target_r_multiple, entry_price - stop_distance * target_r_multiple,
    )

    candidates["stop_price"] = stop_price
    candidates["target_price"] = target_price
    return candidates.reset_index(drop=True)
