# Tracks confirmed swing highs/lows over time (a running "liquidity pool")
# and whether each has since been swept (price traded beyond it). Used to
# tag every detected FVG/RB as "clean" (no unswept liquidity resting inside
# it) or not - per the idea that a PD array with old, untouched liquidity
# still sitting inside it often doesn't react until that liquidity gets
# taken first, especially on higher timeframes.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import session_levels
from .structure import find_pivot_highs, find_pivot_lows


@dataclass
class SwingPoint:
    price: float
    direction: str  # "high" or "low"
    confirm_bar: int


def tag_zone_purity(
    htf_df: pd.DataFrame,
    pivot_length: int,
    bull_flag: pd.Series, bull_top: pd.Series, bull_bottom: pd.Series,
    bear_flag: pd.Series, bear_top: pd.Series, bear_bottom: pd.Series,
    exclude_bull_price: pd.Series | None = None,
    exclude_bear_price: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (bull_clean, bear_clean): booleans aligned to htf_df's index,
    True on any bar where a detected zone (bull or bear) has no unswept
    prior swing point resting strictly inside its [bottom, top] range.
    None on bars with no zone detected there.

    `exclude_bull_price` / `exclude_bear_price` (used for Rejection Blocks):
    the swing level the RB itself just reacted off of - we don't want the
    zone's own trigger counting against it as "leftover" liquidity, only
    *other* levels still resting inside it.

    Timing: the purity check runs using `active` exactly as it stood coming
    into this bar - i.e. before this bar's own high/low can sweep anything.
    That's what lets an RB's own trigger level (which this same bar's wick
    is in the middle of breaking) still be visible to the exclude check
    above, and lets a genuinely nearby old level register as "inside" at
    all - checking after the sweep update would make that structurally
    impossible, since anything still active *after* this bar's sweep must,
    by definition, sit outside this bar's own high/low range entirely.
    """
    is_ph = find_pivot_highs(htf_df["high"], pivot_length)
    is_pl = find_pivot_lows(htf_df["low"], pivot_length)
    highs = htf_df["high"].to_numpy()
    lows = htf_df["low"].to_numpy()
    n = len(htf_df)

    active: list[SwingPoint] = []
    bull_clean = [None] * n
    bear_clean = [None] * n

    for i in range(n):
        confirm_i = i - pivot_length
        if confirm_i >= 0:
            if is_ph.iloc[confirm_i]:
                active.append(SwingPoint(price=highs[confirm_i], direction="high", confirm_bar=confirm_i))
            if is_pl.iloc[confirm_i]:
                active.append(SwingPoint(price=lows[confirm_i], direction="low", confirm_bar=confirm_i))

        if bool(bull_flag.iloc[i]):
            lo, hi = bull_bottom.iloc[i], bull_top.iloc[i]
            excl = exclude_bull_price.iloc[i] if exclude_bull_price is not None else None
            bull_clean[i] = not any(lo < p.price < hi and p.price != excl for p in active)
        if bool(bear_flag.iloc[i]):
            lo, hi = bear_bottom.iloc[i], bear_top.iloc[i]
            excl = exclude_bear_price.iloc[i] if exclude_bear_price is not None else None
            bear_clean[i] = not any(lo < p.price < hi and p.price != excl for p in active)

        active = [
            p for p in active
            if not ((p.direction == "high" and highs[i] > p.price) or (p.direction == "low" and lows[i] < p.price))
        ]

    return (
        pd.Series(bull_clean, index=htf_df.index),
        pd.Series(bear_clean, index=htf_df.index),
    )


def build_swing_liquidity_pool(htf_df: pd.DataFrame, pivot_length: int, tf_label: str) -> pd.DataFrame:
    """
    Every confirmed swing high/low over htf_df, as a full event-log pool:
    one row per swing point, with the bar it confirmed on (start_time) and -
    if price has since traded through it - the bar that swept it (end_time).
    None/NaT end_time means still resting (unswept) as of the last bar in
    htf_df. tf_label (e.g. "1H", "4H", "15m") is stamped on every row as
    "tf" so this pool can be concatenated with others (and with
    build_fvg_pool's output, same column) into one combined liquidity view.

    Same sweep definition tag_zone_purity uses internally: a swing high is
    swept the first bar whose high trades above it; a swing low, the first
    bar whose low trades below it - wick-based, not close-based. This
    returns the pool itself (not just a purity flag derived from it) so a
    caller can reconstruct "what was resting liquidity as of time t" for
    any t, live or historical - see pool_as_of() in .pool.
    """
    is_ph = find_pivot_highs(htf_df["high"], pivot_length)
    is_pl = find_pivot_lows(htf_df["low"], pivot_length)
    highs = htf_df["high"].to_numpy()
    lows = htf_df["low"].to_numpy()
    idx = htf_df.index
    n = len(htf_df)

    active: list[SwingPoint] = []
    records = []

    for i in range(n):
        confirm_i = i - pivot_length
        if confirm_i >= 0:
            if is_ph.iloc[confirm_i]:
                active.append(SwingPoint(price=highs[confirm_i], direction="high", confirm_bar=confirm_i))
            if is_pl.iloc[confirm_i]:
                active.append(SwingPoint(price=lows[confirm_i], direction="low", confirm_bar=confirm_i))

        still_active = []
        for p in active:
            swept = (p.direction == "high" and highs[i] > p.price) or (p.direction == "low" and lows[i] < p.price)
            if swept:
                records.append({
                    "price": p.price, "direction": p.direction, "tf": tf_label,
                    "start_time": idx[p.confirm_bar], "end_time": idx[i],
                })
            else:
                still_active.append(p)
        active = still_active

    for p in active:
        records.append({
            "price": p.price, "direction": p.direction, "tf": tf_label,
            "start_time": idx[p.confirm_bar], "end_time": None,
        })

    out = pd.DataFrame(records, columns=["price", "direction", "tf", "start_time", "end_time"])
    return out.sort_values("start_time").reset_index(drop=True)


def build_session_liquidity_pool(
    df_1m: pd.DataFrame, start_hour: int, end_offset_hours: int, tz: str, tf_label: str,
) -> pd.DataFrame:
    """
    Each day's session high/low (see data.session_levels) as an event-log
    pool in the same shape build_swing_liquidity_pool produces. A level
    starts resting the moment its session closes (start_time - same timing
    session_levels already aligns on) and stops resting at whichever comes
    first: price sweeping it (wick-based, same rule as the swing pool), or
    the *next* occurrence of that same session closing and superseding it
    with a fresh level (~24h later) - at that point the old level is
    considered stale and dropped regardless of whether it was ever swept.
    """
    high_aligned, low_aligned = session_levels(df_1m, start_hour, end_offset_hours, tz)
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    records = []
    for direction, level_series, price_arr, breaks in [
        ("high", high_aligned, highs, lambda seg, lvl: seg > lvl),
        ("low", low_aligned, lows, lambda seg, lvl: seg < lvl),
    ]:
        vals = level_series.to_numpy()
        i = 0
        while i < n:
            if np.isnan(vals[i]):
                i += 1
                continue
            level = vals[i]
            start_i = i
            j = i
            while j < n and not np.isnan(vals[j]) and vals[j] == level:
                j += 1
            swept_mask = breaks(price_arr[start_i:j], level)
            rel = int(np.argmax(swept_mask)) if swept_mask.any() else None
            end_time = idx[start_i + rel] if rel is not None else (idx[j] if j < n else None)
            records.append({
                "price": level, "direction": direction, "tf": tf_label,
                "start_time": idx[start_i], "end_time": end_time,
            })
            i = j

    out = pd.DataFrame(records, columns=["price", "direction", "tf", "start_time", "end_time"])
    return out.sort_values("start_time").reset_index(drop=True)
