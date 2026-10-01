# Confirmed-swing market structure bias, ported from the Pine indicator's
# f_structureBias(). See the module docstring in structure_bias() for the
# non-repainting timing rule this depends on getting right.

import pandas as pd


def find_pivot_highs(high: pd.Series, length: int) -> pd.Series:
    """True at bar i if high[i] is the extreme of the window [i-length, i+length] -
    same definition as Pine's ta.pivothigh(length, length)."""
    window = 2 * length + 1
    rolling_max = high.rolling(window, center=True).max()
    return high == rolling_max


def find_pivot_lows(low: pd.Series, length: int) -> pd.Series:
    window = 2 * length + 1
    rolling_min = low.rolling(window, center=True).min()
    return low == rolling_min


def structure_bias(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    Adds a 'bias' column: 1 = bullish, -1 = bearish, 0 = neutral (no swing
    pair confirmed yet).

    Non-repainting timing: is_pivot_high/is_pivot_low are computed with a
    centered window, which technically "peeks" at future bars to find where
    the pivot is. That's fine - we never let the bias at bar i use a pivot
    until `length` bars after it actually happened (see `confirm_i` below),
    which is exactly when Pine's ta.pivothigh/pivotlow would have revealed
    it too. So the bias at any bar i only ever reflects information that was
    genuinely available by bar i in real time.
    """
    is_ph = find_pivot_highs(df["high"], length)
    is_pl = find_pivot_lows(df["low"], length)

    n = len(df)
    bias = [0] * n
    swing_high_col = [None] * n
    swing_low_col = [None] * n
    last_high = prev_high = None
    last_low = prev_low = None
    cur_bias = 0

    for i in range(n):
        confirm_i = i - length  # bar whose pivot status becomes knowable *now*
        if confirm_i >= 0:
            if is_ph.iloc[confirm_i]:
                prev_high = last_high
                last_high = df["high"].iloc[confirm_i]
            if is_pl.iloc[confirm_i]:
                prev_low = last_low
                last_low = df["low"].iloc[confirm_i]

            if last_high is not None and prev_high is not None and last_low is not None and prev_low is not None:
                if last_high > prev_high and last_low > prev_low:
                    cur_bias = 1
                elif last_high < prev_high and last_low < prev_low:
                    cur_bias = -1
        bias[i] = cur_bias
        swing_high_col[i] = last_high
        swing_low_col[i] = last_low

    out = df.copy()
    out["bias"] = bias
    # dtype=float turns the placeholder None values into proper NaN, so later
    # numeric comparisons (e.g. in check_rb) don't choke on a Python None.
    out["swing_high"] = pd.Series(swing_high_col, index=df.index, dtype=float)
    out["swing_low"] = pd.Series(swing_low_col, index=df.index, dtype=float)
    return out
