# Trend signal, deliberately separate from structure.py's structure_bias():
# bias requires a confirmed swing pair (lags by the pivot-confirmation
# window), trend reacts bar-by-bar to where price sits relative to a moving
# reference. Faster, but prone to whipsaw right around the reference line -
# that's the tradeoff the 713 Dashboard surfaces rather than hides.

import pandas as pd

from .data import resample_bars


def ema_trend(df_1m: pd.DataFrame, rule: str, ema_len: int = 50) -> pd.DataFrame:
    """EMA-cross trend on a resampled HTF (1H/4H/15min/5min): 1 = close above
    the EMA, -1 = below, 0 = exactly on it. Same definition as
    crt_strategy.py's _htf_bias - kept here as the shared, public version so
    other modules (dashboard engine, validation) don't reach into a private
    helper. Index is the HTF bar's own (open-time) index - align onto the
    1-minute timeline with data.align_htf_to_ltf before using it live."""
    htf = resample_bars(df_1m, rule)
    ema = htf["close"].ewm(span=ema_len, adjust=False).mean()
    trend = pd.Series(0, index=htf.index)
    trend[htf["close"] > ema] = 1
    trend[htf["close"] < ema] = -1
    return pd.DataFrame({"trend": trend, "ema": ema}, index=htf.index)


def vwap_trend(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """1 = close above session VWAP, -1 = below, 0 = exactly on it (or VWAP
    not yet defined for the bar, e.g. the very first tick of a session).
    Inherently causal on the 1-minute timeline already - no HTF alignment
    lag needed, unlike ema_trend."""
    trend = pd.Series(0, index=close.index)
    trend[close > vwap] = 1
    trend[close < vwap] = -1
    trend[vwap.isna()] = 0
    return trend


def ema_stack_alignment(df_1m: pd.DataFrame, rule: str, fast_len: int = 9, mid_len: int = 21, slow_len: int = 50) -> pd.DataFrame:
    """A regime *filter*, deliberately not a trigger like ema_trend's single
    cross: 1 = bullish stack (close > fast EMA > mid EMA > slow EMA, all in
    that order), -1 = bearish stack (the mirrored order), 0 = not stacked
    (a mixed/transitional regime - the common case, and intentionally the
    default read rather than forcing every bar into a side).

    Computed on a resampled HTF (e.g. "5min"), not the raw 1-minute chart -
    validate_ema_cross.py found the 9/21 EMA cross has no edge as a
    1-minute-bar trigger, so this exists to test the opposite role: a
    calmer, HTF-computed alignment state used to filter an existing entry
    signal (e.g. htf_sweep_strategy's structure bias), not to fire trades on
    its own. fast_len/mid_len default to 9/21 to match the theory that
    prompted this; slow_len defaults to 50 to match this codebase's existing
    HTF trend length (ema_trend's default, crt_strategy.py's ema_len).

    Index is the HTF bar's own (open-time) index - align onto the 1-minute
    timeline with data.align_htf_to_ltf before using it live, same as
    ema_trend."""
    htf = resample_bars(df_1m, rule)
    close = htf["close"]
    ema_fast = close.ewm(span=fast_len, adjust=False).mean()
    ema_mid = close.ewm(span=mid_len, adjust=False).mean()
    ema_slow = close.ewm(span=slow_len, adjust=False).mean()

    bullish = (close > ema_fast) & (ema_fast > ema_mid) & (ema_mid > ema_slow)
    bearish = (close < ema_fast) & (ema_fast < ema_mid) & (ema_mid < ema_slow)
    alignment = pd.Series(0, index=htf.index)
    alignment[bullish] = 1
    alignment[bearish] = -1
    return pd.DataFrame(
        {"alignment": alignment, "ema_fast": ema_fast, "ema_mid": ema_mid, "ema_slow": ema_slow}, index=htf.index,
    )
