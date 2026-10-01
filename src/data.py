# Loading 1-minute MNQ bars and building higher-timeframe (1H/4H) views from
# them, the same role Pine's request.security("60"/"240", ...) calls play in
# the original indicator.

import numpy as np
import pandas as pd


def load_1m_bars(path: str) -> pd.DataFrame:
    """Load 1-minute OHLCV bars, indexed by UTC timestamp, sorted oldest-first."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col="ts_event", parse_dates=True)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build higher-timeframe bars out of 1-minute bars, e.g. rule='1h' or '4h'."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg)
    out = out.dropna(subset=["open"])  # drop empty buckets (e.g. weekend gaps)
    return out


def align_htf_to_ltf(ltf_index: pd.DatetimeIndex, htf_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Brings higher-timeframe columns (bias, swing_high, swing_low, ...) down onto
    the 1-minute timeline, the way Pine's request.security(..., lookahead=
    barmerge.lookahead_off) does: every 1-minute bar only ever sees the value
    from the most recently *closed* HTF bar, never the one still forming.

    htf_df's index holds each HTF bar's *open* time. We shift that forward by
    one bar's duration to get its *close* time - the moment its values become
    knowable - then, for every 1-minute timestamp, look backward to the most
    recent HTF close at or before it.
    """
    duration = pd.Timedelta(rule)
    available = htf_df.copy()
    available["_htf_open"] = available.index  # survives the merge as a normal column,
    available.index = available.index + duration  # unlike the index used to join on

    ltf_frame = pd.DataFrame(index=ltf_index).reset_index(names="ts")
    htf_frame = available.reset_index(names="ts")

    aligned = pd.merge_asof(ltf_frame, htf_frame, on="ts", direction="backward")
    return aligned.set_index("ts")


def session_levels(df_1m: pd.DataFrame, start_hour: int, end_offset_hours: int, tz: str) -> tuple[pd.Series, pd.Series]:
    """Per-day session high/low (e.g. Asia, London), exposed on the 1-minute
    timeline starting the instant the session actually closes
    (`end_offset_hours` past that day's local midnight - 24 for a session
    that closes at next midnight, e.g. Asia; 5 for one that closes same-day
    at 5am, e.g. London). Mirrors align_htf_to_ltf's backward as-of join: a
    1-minute bar only ever sees a session's high/low once that session has
    genuinely closed, never one still forming."""
    local = df_1m.index.tz_convert(tz)
    duration = end_offset_hours - start_hour
    in_session = (local.hour >= start_hour) & (local.hour < start_hour + duration)

    session_date = local.normalize()
    sub = pd.DataFrame({"high": df_1m["high"].to_numpy(), "low": df_1m["low"].to_numpy()}, index=session_date)
    sub = sub[in_session]
    if sub.empty:
        nan_series = pd.Series(np.nan, index=df_1m.index)
        return nan_series, nan_series.copy()

    grouped = sub.groupby(sub.index).agg(high=("high", "max"), low=("low", "min"))
    grouped.index = (grouped.index + pd.Timedelta(hours=end_offset_hours)).tz_convert(df_1m.index.tz)
    grouped.index.name = "ts"

    ltf = pd.DataFrame(index=df_1m.index).reset_index(names="ts")
    htf = grouped.reset_index()
    aligned = pd.merge_asof(ltf, htf, on="ts", direction="backward").set_index("ts")
    return aligned["high"], aligned["low"]
