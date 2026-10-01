# Fair Value Gap, Rejection Block, and Order Block detection - the zone
# types (PD arrays) the 713 Dashboard treats as POIs. FVG/RB ported from
# f_checkFvg() / f_checkRb(); Order Block is new here (see check_ob).

import numpy as np
import pandas as pd

from .liquidity import tag_zone_purity


def check_fvg(df: pd.DataFrame, min_size: float) -> pd.DataFrame:
    """
    3-candle Fair Value Gap: a bull FVG is a gap between the current bar's
    low and the high from 2 bars ago - nothing traded in between, so it's
    an "imbalance" price often revisits later. Bear FVG is the mirror image.
    """
    h2 = df["high"].shift(2)
    l2 = df["low"].shift(2)

    gap_bull = df["low"] - h2
    gap_bear = l2 - df["high"]

    out = pd.DataFrame(index=df.index)
    out["bull_fvg"] = (gap_bull > 0) & (gap_bull >= min_size)
    out["bear_fvg"] = (gap_bear > 0) & (gap_bear >= min_size)
    # Box coordinates - only meaningful on rows where the matching flag is True.
    out["bull_top"] = df["low"]
    out["bull_bottom"] = h2
    out["bear_top"] = l2
    out["bear_bottom"] = df["high"]
    return out


def check_rb(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series, wick_ratio: float) -> pd.DataFrame:
    """
    Rejection Block: a candle that wicks through a recent swing level but
    closes back on the other side of it, with a wick at least `wick_ratio`
    times the size of its own body - a decisive rejection, not a random poke.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bottom = pd.concat([o, c], axis=1).min(axis=1)
    upper_wick = h - body_top
    lower_wick = body_bottom - l

    wick_ok_upper = (body == 0) & (upper_wick > 0) | (body != 0) & (upper_wick >= wick_ratio * body)
    wick_ok_lower = (body == 0) & (lower_wick > 0) | (body != 0) & (lower_wick >= wick_ratio * body)

    out = pd.DataFrame(index=df.index)
    out["bear_rb"] = swing_high.notna() & (h > swing_high) & (c < swing_high) & wick_ok_upper
    out["bull_rb"] = swing_low.notna() & (l < swing_low) & (c > swing_low) & wick_ok_lower
    out["bull_top"] = body_bottom
    out["bull_bottom"] = l
    out["bear_top"] = h
    out["bear_bottom"] = body_top
    return out


def check_ob(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series, max_lookback: int = 20) -> pd.DataFrame:
    """
    Order Block: the last opposite-colored candle before price *closes*
    beyond the swing level in effect at that time (a break of structure,
    not just a wick through it like check_rb) - the presumed origin candle
    for the move that broke structure. Bullish OB: the last down-close
    (close < open) candle before a close above the current swing_high;
    the OB zone is that origin candle's own [low, high]. Bearish OB is the
    mirror image against swing_low.

    Only the *first* close that breaks a given swing_high/swing_low level
    counts (once that level's been broken, staying above it isn't a new
    break) - `max_lookback` bounds how far back the origin-candle search
    goes, so a long run of same-colored candles before the break doesn't
    return an origin candle from an unrelated earlier move.
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    sh = swing_high.to_numpy(); sl = swing_low.to_numpy()
    n = len(df)

    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    bull_top = np.full(n, np.nan)
    bull_bottom = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bottom = np.full(n, np.nan)

    last_sh_broken = None
    last_sl_broken = None

    for i in range(n):
        if not np.isnan(sh[i]) and c[i] > sh[i] and sh[i] != last_sh_broken:
            j = i - 1
            while j >= 0 and i - j <= max_lookback and c[j] >= o[j]:
                j -= 1
            if j >= 0 and i - j <= max_lookback:
                bull_ob[i] = True
                bull_top[i], bull_bottom[i] = h[j], l[j]
            last_sh_broken = sh[i]

        if not np.isnan(sl[i]) and c[i] < sl[i] and sl[i] != last_sl_broken:
            j = i - 1
            while j >= 0 and i - j <= max_lookback and c[j] <= o[j]:
                j -= 1
            if j >= 0 and i - j <= max_lookback:
                bear_ob[i] = True
                bear_top[i], bear_bottom[i] = h[j], l[j]
            last_sl_broken = sl[i]

    out = pd.DataFrame(index=df.index)
    out["bull_ob"] = bull_ob
    out["bear_ob"] = bear_ob
    out["bull_top"] = bull_top
    out["bull_bottom"] = bull_bottom
    out["bear_top"] = bear_top
    out["bear_bottom"] = bear_bottom
    return out


def _build_zone_pool(
    zone_df: pd.DataFrame, df_tf: pd.DataFrame, kind: str, tf_label: str,
    purity_pivot_length: int, max_age_bars: int | None = None,
    exclude_bull_price: pd.Series | None = None, exclude_bear_price: pd.Series | None = None,
) -> pd.DataFrame:
    """Shared event-log pool bookkeeping behind build_fvg_pool/build_rb_pool/
    build_ob_pool: check_fvg/check_rb/check_ob all return the same shape
    (bull_<kind>/bear_<kind> flags + top/bottom columns), so mitigation-on-
    first-touch, aging, and purity-tagging are identical regardless of
    which zone type produced them. See build_fvg_pool's docstring for the
    exact mitigation/expiry/clean rules - they apply here unchanged."""
    bull_flag, bear_flag = zone_df[f"bull_{kind}"], zone_df[f"bear_{kind}"]
    bull_top, bull_bottom = zone_df["bull_top"], zone_df["bull_bottom"]
    bear_top, bear_bottom = zone_df["bear_top"], zone_df["bear_bottom"]

    bull_clean, bear_clean = tag_zone_purity(
        df_tf, purity_pivot_length, bull_flag, bull_top, bull_bottom, bear_flag, bear_top, bear_bottom,
        exclude_bull_price=exclude_bull_price, exclude_bear_price=exclude_bear_price,
    )

    highs = df_tf["high"].to_numpy()
    lows = df_tf["low"].to_numpy()
    idx = df_tf.index
    n = len(df_tf)

    records = []
    for direction, flag_ser, top_ser, bottom_ser, clean_ser in [
        ("bull", bull_flag, bull_top, bull_bottom, bull_clean),
        ("bear", bear_flag, bear_top, bear_bottom, bear_clean),
    ]:
        flags = flag_ser.to_numpy()
        tops = top_ser.to_numpy()
        bottoms = bottom_ser.to_numpy()
        for i in np.flatnonzero(flags.astype(bool)):
            top, bottom = tops[i], bottoms[i]
            start = i + 1
            cutoff = n if max_age_bars is None else min(n, start + max_age_bars)
            end_time, reason = None, None
            if start < cutoff:
                touch_mask = (lows[start:cutoff] <= top) & (highs[start:cutoff] >= bottom)
                rel = int(np.argmax(touch_mask)) if touch_mask.any() else None
                if rel is not None:
                    end_time, reason = idx[start + rel], "mitigated"
            if end_time is None and max_age_bars is not None and start + max_age_bars < n:
                end_time, reason = idx[start + max_age_bars], "expired"
            clean_val = clean_ser.iloc[i]
            records.append({
                "top": top, "bottom": bottom, "direction": direction, "kind": kind.upper(), "tf": tf_label,
                "start_time": idx[i], "end_time": end_time, "removal_reason": reason,
                "clean": bool(clean_val) if pd.notna(clean_val) else None,
            })

    out = pd.DataFrame(
        records,
        columns=["top", "bottom", "direction", "kind", "tf", "start_time", "end_time", "removal_reason", "clean"],
    )
    return out.sort_values("start_time").reset_index(drop=True)


def build_fvg_pool(
    df_tf: pd.DataFrame, min_size: float, purity_pivot_length: int, tf_label: str,
    max_age_bars: int | None = None,
) -> pd.DataFrame:
    """
    Every FVG on df_tf (an already-resampled single-timeframe OHLC frame,
    e.g. 1H/4H/15min/5min bars) as a full event-log pool: one row per gap,
    with the bar it formed on (start_time) and, once removed, the bar and
    reason (removal_reason) it left the pool on. None/NaT end_time means
    still untapped as of the last bar in df_tf.

    Mitigation ("mitigated"): the same rule strategy.py's
    _update_htf_mitigation uses - the first later bar whose [low, high]
    range overlaps the gap's [bottom, top] at all counts as tapped, not a
    full close-through. Expiry ("expired"): only applies if max_age_bars is
    given - a gap untouched that many bars ages out of the pool even though
    price never actually traded into it.

    clean (via liquidity.tag_zone_purity, evaluated at the gap's own
    formation bar): True if no other unswept swing point (purity_pivot_
    length) was resting inside the gap the moment it formed - a
    discretionary "is this gap likely to actually react" tag, not a filter;
    non-clean gaps stay in the pool too.
    """
    fvg = check_fvg(df_tf, min_size)
    return _build_zone_pool(fvg, df_tf, "fvg", tf_label, purity_pivot_length, max_age_bars)


def build_rb_pool(
    df_tf: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series, wick_ratio: float,
    purity_pivot_length: int, tf_label: str, max_age_bars: int | None = None,
) -> pd.DataFrame:
    """Every Rejection Block (check_rb) on df_tf as an event-log pool, same
    shape and mitigation/expiry/clean rules as build_fvg_pool - see its
    docstring. exclude_bull_price/exclude_bear_price are passed through to
    tag_zone_purity so an RB's own trigger swing level doesn't count
    against its own cleanliness (see tag_zone_purity's docstring)."""
    rb = check_rb(df_tf, swing_high, swing_low, wick_ratio)
    return _build_zone_pool(
        rb, df_tf, "rb", tf_label, purity_pivot_length, max_age_bars,
        exclude_bull_price=swing_low, exclude_bear_price=swing_high,
    )


def build_ob_pool(
    df_tf: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series,
    purity_pivot_length: int, tf_label: str, max_age_bars: int | None = None, max_lookback: int = 20,
) -> pd.DataFrame:
    """Every Order Block (check_ob) on df_tf as an event-log pool, same
    shape and mitigation/expiry/clean rules as build_fvg_pool - see its
    docstring."""
    ob = check_ob(df_tf, swing_high, swing_low, max_lookback=max_lookback)
    return _build_zone_pool(ob, df_tf, "ob", tf_label, purity_pivot_length, max_age_bars)
