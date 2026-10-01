# Threshold-relative zigzag swing detection. Unlike structure.py's pivot
# logic - which confirms a pivot after a fixed bar count on each side - a
# "leg" here is confirmed the moment price reverses from its running extreme
# by >= the threshold reading that was in effect when that extreme was set.
# Duration is whatever it takes to hit that threshold, so legs naturally
# stretch or compress with the prevailing volatility regime instead of us
# having to pick a fixed window a priori.
#
# The threshold itself is pluggable (see atr_threshold / pct_threshold
# below): ATR-relative adapts to the local intraday vol regime, percent-of-
# price adapts to secular price-level drift (e.g. between the 2023-2024
# backtest data and today's much higher price) - use whichever the current
# swing definition calls for.

import numpy as np
import pandas as pd

from .data import align_htf_to_ltf, resample_bars
from .strategy import wilder_atr


def atr_15m_aligned(df_1m: pd.DataFrame, length: int = 14) -> pd.Series:
    """15-minute Wilder ATR(length), brought onto the 1-minute timeline
    non-repainting: every 1m bar only ever sees the most recently *closed*
    15m bar's ATR, never the one still forming."""
    htf = resample_bars(df_1m, "15min")
    htf_atr = wilder_atr(htf, length).to_frame("atr_15m")
    aligned = align_htf_to_ltf(df_1m.index, htf_atr, "15min")
    return aligned["atr_15m"]


def atr_threshold(atr: pd.Series, atr_mult: float) -> pd.Series:
    return atr * atr_mult


def pct_threshold(df_1m: pd.DataFrame, pct: float) -> pd.Series:
    """Required move size as a flat percentage of price - e.g. pct=0.008
    means "must move >= 0.8% from the extreme," regardless of what ATR is
    doing at that moment."""
    return df_1m["close"] * pct


def zigzag_legs(df_1m: pd.DataFrame, threshold: pd.Series, atr: pd.Series | None = None) -> pd.DataFrame:
    """
    Confirmed swing legs via threshold zigzag. Tracks a running high and a
    running low simultaneously until one side reverses by >= the threshold
    reading at that extreme's bar - at that point the extreme is confirmed
    as a pivot, a leg is recorded from the previous pivot to this one, and
    tracking flips to the opposite side.

    `threshold` is the absolute point move required to confirm a leg, per
    bar (see atr_threshold/pct_threshold). `atr` is optional and only used
    to additionally report each leg's magnitude in ATR-multiples, even when
    `threshold` itself isn't ATR-based.

    The in-progress leg at the end of the data (not yet confirmed) is
    dropped - we only want completed legs for pattern mining.
    """
    high = df_1m["high"].to_numpy()
    low = df_1m["low"].to_numpy()
    thresh_arr = threshold.to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    start_i = 0
    while start_i < n and np.isnan(thresh_arr[start_i]):
        start_i += 1
    if start_i >= n:
        return _empty_legs()

    direction = 0  # 0 = undecided, 1 = tracking a high (up-swing), -1 = tracking a low (down-swing)
    anchor_i = start_i
    anchor_price = None
    hi_i, hi_price = start_i, high[start_i]
    lo_i, lo_price = start_i, low[start_i]

    legs = []

    for i in range(start_i, n):
        if direction <= 0:
            if low[i] < lo_price:
                lo_i, lo_price = i, low[i]
            thresh = thresh_arr[lo_i]
            if not np.isnan(thresh) and (high[i] - lo_price) >= thresh:
                if direction == -1:
                    legs.append(_leg(idx, anchor_i, lo_i, anchor_price, lo_price, "down"))
                anchor_i, anchor_price = lo_i, lo_price
                direction = 1
                hi_i, hi_price = i, high[i]
                continue

        if direction >= 0:
            if high[i] > hi_price:
                hi_i, hi_price = i, high[i]
            thresh = thresh_arr[hi_i]
            if not np.isnan(thresh) and (hi_price - low[i]) >= thresh:
                if direction == 1:
                    legs.append(_leg(idx, anchor_i, hi_i, anchor_price, hi_price, "up"))
                anchor_i, anchor_price = hi_i, hi_price
                direction = -1
                lo_i, lo_price = i, low[i]

    out = pd.DataFrame(legs)
    if out.empty:
        return _empty_legs()
    out["magnitude"] = (out["end_price"] - out["start_price"]).abs()
    out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60
    out["pct_move"] = out["magnitude"] / out["start_price"]
    if atr is not None:
        out["atr_at_start"] = atr.reindex(out["start_time"]).to_numpy()
        out["magnitude_atr_mult"] = out["magnitude"] / out["atr_at_start"]
    return out


def _leg(idx, start_i, end_i, start_price, end_price, direction):
    return {
        "start_time": idx[start_i],
        "end_time": idx[end_i],
        "start_price": start_price,
        "end_price": end_price,
        "direction": direction,
    }


def _empty_legs() -> pd.DataFrame:
    cols = [
        "start_time", "end_time", "start_price", "end_price", "direction",
        "magnitude", "duration_min", "pct_move", "atr_at_start", "magnitude_atr_mult",
    ]
    return pd.DataFrame(columns=cols)


def add_precursor_features(
    legs: pd.DataFrame, df_1m: pd.DataFrame, atr: pd.Series,
    lookback_min: int = 90, tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    For each leg, characterizes the `lookback_min` minutes immediately
    before it started:
      - start_hour: hour of day (0-23) in `tz` the leg began in.
      - pre_range: high-low range over the lookback window.
      - pre_range_atr_ratio: that range divided by the ATR in effect at leg
        start - the compression/expansion yardstick. Low = tight range
        (compression) right before the leg; high = already choppy/expanded.
    """
    out = legs.copy()
    start_hours = []
    pre_ranges = []
    ratios = []

    for start_time, atr_at_start in zip(out["start_time"], out["atr_at_start"]):
        window = df_1m.loc[start_time - pd.Timedelta(minutes=lookback_min): start_time]
        if len(window) < 2 or pd.isna(atr_at_start) or atr_at_start == 0:
            pre_ranges.append(np.nan)
            ratios.append(np.nan)
        else:
            rng = window["high"].max() - window["low"].min()
            pre_ranges.append(rng)
            ratios.append(rng / atr_at_start)
        start_hours.append(start_time.tz_convert(tz).hour)

    out["start_hour"] = start_hours
    out["pre_range"] = pre_ranges
    out["pre_range_atr_ratio"] = ratios
    return out
