# Tests the pattern described in Powell Trades' "How to structure a
# Rejection block entry" video - now grounded in actual quotes pulled by
# scrubbing through the video's burned-in captions frame by frame (no real
# transcript track exists for this channel), not just a search-result
# summary sentence. Two corrections from the first pass at this, both
# straight from his own words:
#
#   1. "Technically, it's not a rejection block until the candle closes
#      bearish [for a bear RB]... [not] the last up-close candle, I don't
#      really care" - he's explicitly rejecting the Order Block definition
#      (poi.check_ob: last opposite-colored candle before a break) and
#      requiring the WICKING CANDLE ITSELF to close in the rejecting
#      direction (close < open for a bear RB, close > open for a bull RB).
#      poi.check_rb alone doesn't check this - it only checks close vs. the
#      swept level, not the candle's own color - so require_bearish_
#      bullish_close below adds it as an extra condition on top.
#   2. He points at a "little five-minute rejection block" mid-video, on
#      the same chart where he'd been reading 1H structure - not fixed to
#      one timeframe. RBS_TIMEFRAMES below defaults to both 1h and 5min,
#      pooled together, instead of 1H-only.
#
# "Um that's pretty much like unfilled new week opening gaps are [the]
# confluences I don't really need much more if we have that" - his own
# words suggest gap + RB alone is close to sufficient; London-session and
# premium/discount are kept as togglable (default OFF) rather than hard
# gates, since he didn't describe them as always-required the way the gap
# itself is ("completely unfilled" - require_unfilled_nwog defaults to
# True now, confirmed by his actual wording, not loosened as a workaround).
#
# Definitions:
#   RB             - poi.check_rb on each timeframe in RBS_TIMEFRAMES
#                     (default 1h + 5min, pooled), same wick_ratio=2.0/
#                     pivot_length=5 convention this codebase already uses
#                     for RBs (dashboard_engine.py), PLUS the candle's own
#                     close must be on the rejecting side of its own open
#                     (see above) - the correction from his actual words.
#                     rb_near = the candle's own body edge, rb_far = its
#                     wick extreme - same zone shape as every other RB in
#                     this codebase.
#   London session - (optional, require_london_session) the RB's own bar
#                     must CLOSE within london_start_hour..+london_window_
#                     hours ET, the same London window htf_sweep_strategy.py
#                     already uses for session-level liquidity.
#   NWOG           - the gap between the last 1-minute print before a
#                     weekend/holiday gap (>= min_gap_hours between
#                     consecutive bars) and the first print after it.
#                     Bullish if the new week opened above the old week's
#                     close, bearish if below. Active (the bias read) from
#                     the moment the new week starts until price first
#                     trades back through the old week's close level (the
#                     gap "fills") or the next week's gap supersedes it,
#                     whichever comes first - same active/expire
#                     bookkeeping shape as liquidity.py's swept-level pools.
#   Premium/discount - (optional, require_premium_discount) the current
#                     week's running high/low (from that week's own start
#                     bar to the current bar) midpoint. Below the midpoint =
#                     discount, above = premium.
#   Entry filter   - a bull RB only counts if NWOG bias is bullish (+ in
#                     discount, if require_premium_discount). A bear RB
#                     only counts if NWOG bias is bearish (+ in premium).
#
# Entry/stop mechanics are otherwise identical to rb_5m_gated_strategy.py:
# a 1-minute retest of the RB's own zone, entry at rb_near, stop
# rb_stop_buffer_points (fixed, always 5, per spec) beyond rb_far, one
# trade at a time, oldest live candidate wins a same-bar tie.

import numpy as np
import pandas as pd

from .poi import check_rb
from .structure import structure_bias

PIVOT_LENGTH = 5      # matches this repo's existing RB convention (dashboard_engine.PIVOT_LENGTH)
RB_WICK_RATIO = 2.0   # matches this repo's existing RB convention (dashboard_engine.RB_WICK_RATIO)
RBS_TIMEFRAMES = ("1h", "5min")  # pooled together - he reads RBs across timeframes, not one fixed TF
LONDON_START_HOUR = 2
LONDON_WINDOW_HOURS = 5
MIN_WEEKEND_GAP_HOURS = 6  # a gap this large between consecutive 1m bars marks a week boundary
TZ = "America/New_York"


def _resample(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return df_1m.resample(timeframe).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def _find_rbs_on_tf(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """RBs on a single timeframe, WITH the candle-closes-in-the-rejecting-
    direction condition his own words added on top of poi.check_rb's bare
    wick/close-vs-level shape (see module docstring point 1)."""
    df_tf = _resample(df_1m, timeframe)
    struct = structure_bias(df_tf, PIVOT_LENGTH)
    rb = check_rb(df_tf, struct["swing_high"], struct["swing_low"], RB_WICK_RATIO)

    o, c = df_tf["open"], df_tf["close"]
    body = (c - o).abs()
    bearish_close = c < o
    bullish_close = c > o

    rows = []
    for direction, flag_col, near_col, far_col, close_ok in [
        (1, "bull_rb", "bull_top", "bull_bottom", bullish_close),
        (-1, "bear_rb", "bear_bottom", "bear_top", bearish_close),
    ]:
        flags = rb[flag_col].fillna(False) & close_ok
        for t in flags.index[flags]:
            near, far = rb.loc[t, near_col], rb.loc[t, far_col]
            b = body.loc[t]
            wick = abs(near - far)
            rows.append({
                "formed_at": t, "dir": direction, "rb_near": near, "rb_far": far,
                "wick_body_ratio": (wick / b) if b > 0 else np.inf, "tf": timeframe,
            })
    return pd.DataFrame(rows)


def _find_rbs(df_1m: pd.DataFrame, timeframes=RBS_TIMEFRAMES) -> pd.DataFrame:
    """RBs pooled across every timeframe in `timeframes`, sorted by
    formation time - a candidate from any of them can be taken, whichever
    gets retested first (see the FSM's live-pool loop below)."""
    frames = [_find_rbs_on_tf(df_1m, tf) for tf in timeframes]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["formed_at", "dir", "rb_near", "rb_far", "wick_body_ratio", "tf"]
    )
    return out.sort_values("formed_at").reset_index(drop=True)


def _week_boundaries(df_1m: pd.DataFrame) -> np.ndarray:
    """Bar indices where a new trading week starts - the first bar after a
    gap of >= MIN_WEEKEND_GAP_HOURS from the previous bar. Always includes
    index 0 (the start of the data is itself a "week start" for range-
    tracking purposes, even though there's no real NWOG to measure there)."""
    idx = df_1m.index
    gaps = idx.to_series().diff()
    starts = np.flatnonzero(gaps.to_numpy() >= np.timedelta64(MIN_WEEKEND_GAP_HOURS, "h"))
    return np.concatenate([[0], starts])


def _nwog_bias(df_1m: pd.DataFrame, week_starts: np.ndarray, require_unfilled: bool = False) -> np.ndarray:
    """1 = bullish gap, -1 = bearish, 0 = no read (either no gap that week,
    or - only when require_unfilled=True - it's already been filled, or
    this is the very first week in the data with no prior close to gap
    from).

    require_unfilled=True is the literal reading of the YouTube summary's
    "unfilled new week opening gap" - checked first, and on this data it's
    almost never true at the same time as an RB actually forms (see
    conversation: only 11-26 of ~300 1H RBs over 2 years even have a
    still-open matching gap - too strict to backtest meaningfully).
    require_unfilled=False (the default here) treats the gap's direction as
    that WEEK's standing bias regardless of whether price has since traded
    back through it - the more usable, and arguably more common, way
    "gap bias" gets applied in practice: a context read for the week, not a
    live invalidation condition tracked tick by tick."""
    n = len(df_1m)
    closes = df_1m["close"].to_numpy()
    opens = df_1m["open"].to_numpy()
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()

    bias = np.zeros(n, dtype=int)
    for k in range(1, len(week_starts)):  # week_starts[0] is the data's own start, not a real gap
        start = week_starts[k]
        prior_close = closes[start - 1]
        week_open = opens[start]
        direction = 1 if week_open > prior_close else (-1 if week_open < prior_close else 0)
        if direction == 0:
            continue
        end = week_starts[k + 1] if k + 1 < len(week_starts) else n
        if require_unfilled:
            seg_low, seg_high = lows[start:end], highs[start:end]
            filled_mask = (seg_low <= prior_close) if direction == 1 else (seg_high >= prior_close)
            active_len = int(np.argmax(filled_mask)) if filled_mask.any() else (end - start)
        else:
            active_len = end - start
        bias[start:start + active_len] = direction
    return bias


def _weekly_midpoint(df_1m: pd.DataFrame, week_starts: np.ndarray) -> np.ndarray:
    """Running (week_high + week_low) / 2 from each week's own start bar
    through the current bar - the dealing range premium/discount is judged
    against, recomputed fresh each week."""
    n = len(df_1m)
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    mid = np.full(n, np.nan)
    for k in range(len(week_starts)):
        start = week_starts[k]
        end = week_starts[k + 1] if k + 1 < len(week_starts) else n
        run_high = np.maximum.accumulate(highs[start:end])
        run_low = np.minimum.accumulate(lows[start:end])
        mid[start:end] = (run_high + run_low) / 2
    return mid


def run_rb_1h_london_nwog_strategy(
    df_1m: pd.DataFrame,
    rb_stop_buffer_points: float = 5.0,
    target_r_multiple: float = 8.0,
    max_retest_hours: float = 72.0,
    max_trade_bars: int | None = None,
    require_unfilled_nwog: bool = True,
    require_london_session: bool = False,
    require_premium_discount: bool = False,
    timeframes=RBS_TIMEFRAMES,
) -> pd.DataFrame:
    """Same FSM shape as rb_5m_gated_strategy.py: one trade at a time, RBs
    (pooled across `timeframes`, gated by NWOG bias and optionally London
    session / premium-discount) queue up as live candidates, first one
    whose zone gets retested on a 1-minute bar wins. Returns the same
    trade-record schema as every other strategy in this codebase."""
    rbs = _find_rbs(df_1m, timeframes)

    week_starts = _week_boundaries(df_1m)
    nwog = _nwog_bias(df_1m, week_starts, require_unfilled=require_unfilled_nwog)
    weekly_mid = _weekly_midpoint(df_1m, week_starts)
    nwog_series = pd.Series(nwog, index=df_1m.index)
    mid_series = pd.Series(weekly_mid, index=df_1m.index)

    formed_local = rbs["formed_at"].dt.tz_convert(TZ)
    in_london = (formed_local.dt.hour >= LONDON_START_HOUR) & (formed_local.dt.hour < LONDON_START_HOUR + LONDON_WINDOW_HOURS)

    rb_bias = nwog_series.reindex(rbs["formed_at"], method="ffill").to_numpy()
    rb_mid = mid_series.reindex(rbs["formed_at"], method="ffill").to_numpy()
    # rb_near is a bull RB's body edge / a bear RB's body edge - already
    # very close to that candle's actual close, good enough as the
    # reference price for premium/discount without a second reindex.
    ref_price = rbs["rb_near"].to_numpy()
    in_discount = ref_price < rb_mid
    in_premium = ref_price > rb_mid

    bias_ok = ((rbs["dir"] == 1) & (rb_bias == 1)) | ((rbs["dir"] == -1) & (rb_bias == -1))
    pd_ok = ((rbs["dir"] == 1) & in_discount) | ((rbs["dir"] == -1) & in_premium) if require_premium_discount \
        else pd.Series(True, index=rbs.index)
    london_ok = in_london if require_london_session else pd.Series(True, index=rbs.index)

    keep = bias_ok.to_numpy() & pd_ok.to_numpy() & london_ok.to_numpy()
    rbs = rbs[keep].reset_index(drop=True)

    idx = df_1m.index
    n = len(df_1m)
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()

    formed_bar = idx.searchsorted(rbs["formed_at"].to_numpy(), side="left")
    expire_at = rbs["formed_at"] + pd.Timedelta(hours=max_retest_hours)
    expire_bar = idx.searchsorted(expire_at.to_numpy(), side="left")

    live: list[dict] = []
    next_candidate = 0
    n_candidates = len(rbs)

    active_trade = False
    entry_bar = stop_price = target_price = None
    is_long = None

    trades: list[dict] = []

    for i in range(n):
        high, low, close = highs[i], lows[i], closes[i]

        while next_candidate < n_candidates and formed_bar[next_candidate] <= i:
            live.append({
                "dir": rbs["dir"].iloc[next_candidate],
                "near": rbs["rb_near"].iloc[next_candidate],
                "far": rbs["rb_far"].iloc[next_candidate],
                "wick_body_ratio": rbs["wick_body_ratio"].iloc[next_candidate],
                "expire_bar": expire_bar[next_candidate],
            })
            next_candidate += 1

        if live:
            live = [c for c in live if c["expire_bar"] >= i]

        if not active_trade and live:
            taken = None
            for c in live:
                touched = (low <= c["near"]) if c["dir"] == 1 else (high >= c["near"])
                if touched:
                    taken = c
                    break
            if taken is not None:
                entry_price = taken["near"]
                stop = (taken["far"] - rb_stop_buffer_points) if taken["dir"] == 1 \
                    else (taken["far"] + rb_stop_buffer_points)
                risk = abs(entry_price - stop)
                if risk > 0:
                    is_long = taken["dir"] == 1
                    target_price = entry_price + risk * target_r_multiple if is_long \
                        else entry_price - risk * target_r_multiple
                    stop_price = stop
                    entry_bar, active_trade = i, True
                    trades.append({
                        "time": idx[i], "direction": "long" if is_long else "short",
                        "entry_price": entry_price, "stop_price": stop_price, "target_price": target_price,
                        "rb_near": taken["near"], "rb_far": taken["far"],
                        "wick_body_ratio": taken["wick_body_ratio"],
                        "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
                    })
                live.remove(taken)

        if active_trade:
            in_window = max_trade_bars is None or (i - entry_bar) <= max_trade_bars
            hit_stop = (low <= stop_price) if is_long else (high >= stop_price)
            hit_target = (high >= target_price) if is_long else (low <= target_price)
            timed_out = (not in_window) and not hit_stop and not hit_target

            if hit_stop:
                trades[-1].update(outcome="loss", exit_time=idx[i], exit_price=stop_price)
                active_trade = False
            elif hit_target:
                trades[-1].update(outcome="win", exit_time=idx[i], exit_price=target_price)
                active_trade = False
            elif timed_out:
                trades[-1].update(outcome="timeout", exit_time=idx[i], exit_price=close)
                active_trade = False

    result = pd.DataFrame(trades)
    if result.empty:
        return result

    still_open = result["outcome"].isna()
    result.loc[still_open, "outcome"] = "open_at_data_end"
    result.loc[still_open, "exit_time"] = idx[-1]
    result.loc[still_open, "exit_price"] = closes[-1]

    risk = (result["entry_price"] - result["stop_price"]).abs()
    direction_sign = np.where(result["direction"] == "long", 1, -1)
    result["r_multiple"] = (result["exit_price"] - result["entry_price"]) * direction_sign / risk
    result["stop_distance_points"] = risk
    return result
