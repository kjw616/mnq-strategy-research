# Session-anchored VWAP: resets at 6:00 PM ET, matching the futures session
# open (when CME Globex opens the new trading day) rather than a plain
# midnight or calendar-day reset.

import numpy as np
import pandas as pd


def session_vwap(df: pd.DataFrame, anchor_hour: int = 18, tz: str = "America/New_York") -> pd.Series:
    """
    Volume-weighted average price, reset each time a new session begins at
    `anchor_hour` in `tz`. Uses the standard "typical price" (H+L+C)/3.

    df.index must be a tz-aware DatetimeIndex (e.g. UTC, straight from
    load_1m_bars). We convert to `tz` only to *find* each session's boundary -
    pandas handles the DST shift for us - then group and reset there.
    """
    local_time = df.index.tz_convert(tz)
    # Shifting every timestamp back by `anchor_hour` hours and taking the date
    # groups everything from anchor_hour on day D through anchor_hour-1 on day
    # D+1 under the single label "D" - i.e. one group per session.
    session_id = (local_time - pd.Timedelta(hours=anchor_hour)).floor("D")

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = df["volume"].groupby(session_id).cumsum()

    return cum_tp_vol / cum_vol


def wilder_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average Directional Index, Wilder's original smoothing (same
    convention as Pine's ta.dmi()/[adx] plot) - trend STRENGTH regardless of
    direction, high when price is trending, low when it's chopping."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
