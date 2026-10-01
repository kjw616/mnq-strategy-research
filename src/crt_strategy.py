# Port of CRT_v21.pine: Candle 1 (large-bodied range) + Candle 2 (small-body
# sweep that closes back inside C1's range) arms a setup; Candle 3 sweeping
# C2's OPPOSITE wick and closing hard through it confirms; the Fair Value Gap
# that displacement leaves behind is the entry zone. Every stage is a hard
# yes/no gate - no scoring anywhere, same as the Pine version. Cross-reference
# against CRT_v21.pine's FSM section, which this mirrors closely on purpose.
#
# Unlike strategy.py (the ICT port, which always resolves trades against
# 1-minute bars regardless of chart TF), this resolves trades using the same
# chart-timeframe bars the setup itself was detected on - because that's
# what CRT_v21.pine actually does when it's running on a 15m/30m/1H chart.
# Using 1-minute precision here would make the backtest more accurate than
# the live indicator actually is, which isn't what we want to measure.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import align_htf_to_ltf, resample_bars
from .trend import ema_trend

ST_SEARCH, ST_ARMED, ST_CHECK_FVG, ST_AWAIT_ENTRY = range(4)


@dataclass
class CrtSetup:
    direction: str  # "bull" or "bear"
    c1_high: float
    c1_low: float
    c2_high: float
    c2_low: float
    c2_open: float
    c2_close: float
    armed_bar: int
    reaction_bar: int | None = None
    fvg_top: float | None = None
    fvg_bottom: float | None = None


def wilder_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Same smoothing Pine's ta.atr() uses (Wilder's, not a plain average)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _htf_bias(df_1m: pd.DataFrame, rule: str, ema_len: int) -> pd.DataFrame:
    """EMA-cross bias on a resampled HTF (1H/4H): 1 bullish, -1 bearish, 0
    neither - matches CRT_v21.pine's bias1H/bias4H exactly. Thin wrapper
    around trend.ema_trend (the shared, public version), kept here so
    existing callers' "bias" column name doesn't change."""
    return ema_trend(df_1m, rule, ema_len).rename(columns={"trend": "bias"})[["bias"]]


def detect_crt_patterns(
    chart_bars: pd.DataFrame,
    atr: pd.Series,
    c1_min_atr_mult: float,
    c1_min_body_ratio: float,
    c2_max_body_ratio: float,
    liquidity_lookback: int = 1,
) -> pd.DataFrame:
    """Vectorized Candle 1 + Candle 2 detection - matches bullCRT/bearCRT in
    the Pine script. Candle 3 confirmation happens later, bar by bar, since
    it depends on FSM state (a later candle, not fixed at a 2-bar lookback).

    `liquidity_lookback` (default 1, a no-op - see below) requires C1's side
    that actually gets swept to be a genuine local extreme: its high must be
    the highest high of the trailing `liquidity_lookback` bars (itself
    included) for a bearish setup, its low the lowest low for a bullish one.
    This is what separates "C1 happened to be a big candle" from "C1's
    extreme is real resting liquidity" - backward-looking only (no centered-
    pivot lookahead), so it stays valid at the exact bar C1 already sits at.
    """
    c1_high, c1_low = chart_bars["high"].shift(2), chart_bars["low"].shift(2)
    c1_open, c1_close = chart_bars["open"].shift(2), chart_bars["close"].shift(2)
    c2_high, c2_low = chart_bars["high"].shift(1), chart_bars["low"].shift(1)
    c2_open, c2_close = chart_bars["open"].shift(1), chart_bars["close"].shift(1)

    double_sweep = (c2_high > c1_high) & (c2_low < c1_low)

    c1_range = c1_high - c1_low
    c1_ratio = (c1_close - c1_open).abs() / c1_range.replace(0, np.nan)
    c1_valid = (c1_range > atr * c1_min_atr_mult) & (c1_ratio > c1_min_body_ratio)

    trailing_high = chart_bars["high"].rolling(liquidity_lookback).max().shift(2)
    trailing_low = chart_bars["low"].rolling(liquidity_lookback).min().shift(2)
    c1_is_swing_high = c1_high >= trailing_high
    c1_is_swing_low = c1_low <= trailing_low

    bear_sweep = (c2_high > c1_high) & (c2_close < c1_high)
    bull_sweep = (c2_low < c1_low) & (c2_close > c1_low)

    c2_range = c2_high - c2_low
    c2_ratio = (c2_close - c2_open).abs() / c2_range.replace(0, np.nan)
    c2_small_body = c2_ratio <= c2_max_body_ratio

    out = pd.DataFrame(index=chart_bars.index)
    out["c1_high"], out["c1_low"] = c1_high, c1_low
    out["c2_high"], out["c2_low"] = c2_high, c2_low
    out["c2_open"], out["c2_close"] = c2_open, c2_close
    out["bull_crt"] = (c1_valid & bull_sweep & c2_small_body & c1_is_swing_low & ~double_sweep).fillna(False)
    out["bear_crt"] = (c1_valid & bear_sweep & c2_small_body & c1_is_swing_high & ~double_sweep).fillna(False)
    return out


def run_crt_strategy(
    df_1m: pd.DataFrame,
    chart_tf_rule: str,
    chart_tf_minutes: int,
    ema_len: int = 50,
    atr_len: int = 14,
    c1_min_atr_mult: float = 0.8,
    c1_min_body_ratio: float = 0.6,
    c2_max_body_ratio: float = 0.5,
    min_fvg_atr: float = 0.15,
    reaction_window_min: int = 90,
    entry_window_min: int = 60,
    stop_buffer_atr: float = 0.15,
    target_r_multiple: float = 2.0,
    max_trade_bars: int = 20,
) -> pd.DataFrame:
    """Runs the full CRT pipeline on `chart_tf_rule` bars (e.g. "15min",
    "1h") built from 1-minute data, and returns every entry with its
    resolved outcome (win/loss/timeout/open_at_data_end) and r_multiple."""
    chart_bars = resample_bars(df_1m, chart_tf_rule)
    atr = wilder_atr(chart_bars, atr_len)
    patterns = detect_crt_patterns(chart_bars, atr, c1_min_atr_mult, c1_min_body_ratio, c2_max_body_ratio)

    ctx_1h = _htf_bias(df_1m, "1h", ema_len)
    ctx_4h = _htf_bias(df_1m, "4h", ema_len)
    bias_1h = align_htf_to_ltf(chart_bars.index, ctx_1h, "1h")["bias"].fillna(0).to_numpy()
    bias_4h = align_htf_to_ltf(chart_bars.index, ctx_4h, "4h")["bias"].fillna(0).to_numpy()
    final_bias = np.where(bias_4h != 0, bias_4h, bias_1h)

    reaction_window_bars = max(1, round(reaction_window_min / chart_tf_minutes))
    entry_window_bars = max(1, round(entry_window_min / chart_tf_minutes))

    highs, lows, closes = chart_bars["high"].to_numpy(), chart_bars["low"].to_numpy(), chart_bars["close"].to_numpy()
    atr_arr = atr.to_numpy()
    idx = chart_bars.index
    n = len(chart_bars)

    c1h_arr, c1l_arr = patterns["c1_high"].to_numpy(), patterns["c1_low"].to_numpy()
    c2h_arr, c2l_arr = patterns["c2_high"].to_numpy(), patterns["c2_low"].to_numpy()
    c2o_arr, c2c_arr = patterns["c2_open"].to_numpy(), patterns["c2_close"].to_numpy()
    bull_crt_arr, bear_crt_arr = patterns["bull_crt"].to_numpy(), patterns["bear_crt"].to_numpy()

    fsm_state = ST_SEARCH
    active_crt: CrtSetup | None = None
    active_trade = False
    entry_bar = stop_price = target_price = None
    is_long = None

    trades: list[dict] = []

    for i in range(n):
        high, low, close = highs[i], lows[i], closes[i]
        bull_bias_ok = final_bias[i] == 1
        bear_bias_ok = final_bias[i] == -1

        # --- Stage 1: SEARCH -> ARMED ---
        if fsm_state == ST_SEARCH:
            if not active_trade and bull_crt_arr[i] and bull_bias_ok:
                active_crt = CrtSetup("bull", c1h_arr[i], c1l_arr[i], c2h_arr[i], c2l_arr[i], c2o_arr[i], c2c_arr[i], i)
                fsm_state = ST_ARMED
            elif not active_trade and bear_crt_arr[i] and bear_bias_ok:
                active_crt = CrtSetup("bear", c1h_arr[i], c1l_arr[i], c2h_arr[i], c2l_arr[i], c2o_arr[i], c2c_arr[i], i)
                fsm_state = ST_ARMED

        # --- Bias flip safety reset ---
        if fsm_state != ST_SEARCH and active_crt is not None:
            bias_ok = bull_bias_ok if active_crt.direction == "bull" else bear_bias_ok
            if not bias_ok:
                fsm_state, active_crt = ST_SEARCH, None

        # --- Stage 2: ARMED -> Candle 3 found ---
        if fsm_state == ST_ARMED and active_crt is not None:
            is_bull = active_crt.direction == "bull"
            reacted = (close > active_crt.c2_high) if is_bull else (close < active_crt.c2_low)
            if reacted:
                active_crt.reaction_bar = i
                fsm_state = ST_CHECK_FVG
            elif i - active_crt.armed_bar > reaction_window_bars:
                fsm_state, active_crt = ST_SEARCH, None

        # --- Stage 2b: exactly one bar after Candle 3 - did it leave an FVG? ---
        if fsm_state == ST_CHECK_FVG and active_crt is not None and i == active_crt.reaction_bar + 1:
            is_bull = active_crt.direction == "bull"
            gap_size = (low - active_crt.c2_high) if is_bull else (active_crt.c2_low - high)
            atr_now = atr_arr[i]
            if not np.isnan(atr_now) and gap_size >= atr_now * min_fvg_atr:
                if is_bull:
                    active_crt.fvg_top, active_crt.fvg_bottom = low, active_crt.c2_high
                else:
                    active_crt.fvg_top, active_crt.fvg_bottom = active_crt.c2_low, high
                fsm_state = ST_AWAIT_ENTRY
            else:
                fsm_state, active_crt = ST_SEARCH, None

        # --- Stage 3: AWAIT_ENTRY -> ENTRY (first touch fires, no score gate) ---
        # Guarded to bars strictly after the FVG-confirming bar - otherwise
        # this runs in the same iteration Stage 2b just set fvg_top/bottom
        # FROM this same bar's own high/low, so "retest" is trivially true
        # every time (no actual retracement ever required). See CRT_v21.pine
        # for the matching fix.
        if fsm_state == ST_AWAIT_ENTRY and active_crt is not None and i > active_crt.reaction_bar + 1:
            is_bull = active_crt.direction == "bull"
            retest = low <= active_crt.fvg_top and high >= active_crt.fvg_bottom
            if retest:
                entry_price = active_crt.fvg_top if is_bull else active_crt.fvg_bottom
                atr_now = atr_arr[i]
                stop_price = (active_crt.fvg_bottom - atr_now * stop_buffer_atr) if is_bull \
                    else (active_crt.fvg_top + atr_now * stop_buffer_atr)
                risk = abs(entry_price - stop_price)
                target_price = entry_price + risk * target_r_multiple if is_bull \
                    else entry_price - risk * target_r_multiple
                entry_bar, is_long, active_trade = i, is_bull, True
                trades.append({
                    "time": idx[i], "direction": "long" if is_bull else "short",
                    "entry_price": entry_price, "stop_price": stop_price, "target_price": target_price,
                    "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
                })
                fsm_state, active_crt = ST_SEARCH, None
            elif i - active_crt.reaction_bar > entry_window_bars:
                fsm_state, active_crt = ST_SEARCH, None

        # --- Trade resolution (independent of FSM once a trade is live) ---
        if active_trade:
            in_window = entry_bar <= i <= entry_bar + max_trade_bars
            hit_stop = in_window and ((low <= stop_price) if is_long else (high >= stop_price))
            hit_target = in_window and ((high >= target_price) if is_long else (low <= target_price))
            timed_out = (i == entry_bar + max_trade_bars) and not hit_stop and not hit_target
            if hit_target and not hit_stop:
                trades[-1].update(outcome="win", exit_time=idx[i], exit_price=target_price)
                active_trade = False
            elif hit_stop:
                trades[-1].update(outcome="loss", exit_time=idx[i], exit_price=stop_price)
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
    return result
