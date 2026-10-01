# Second architecture for CRT: instead of CRT being the entry vehicle itself
# (crt_strategy.py), CRT now plays the role your ICT script's market
# structure bias plays - a directional gate, not a win/loss unit. While bias
# is active, EVERY time 1-minute price crosses beyond C2's OPPOSITE wick (the
# same "sweep the untouched liquidity pocket" idea as Candle 3 in the
# single-TF version, just happening on 1-minute closes instead of a single
# HTF candle) is a fresh entry - several trades per bias window, not
# one-and-done.
#
# Raw C1+C2 alone is NOT selective enough to trust as bias on its own - on
# 5min bars it fires on ~6.7% of all candles (9,475 times over 2 years).
# In the single-TF version that never mattered because the later stages
# (Candle 3 reaction, then requiring a real FVG) did the actual filtering.
# Here, a raw pattern is only a PENDING candidate: it becomes live, tradeable
# bias only once price closes beyond its own C2 opposite wick within the
# reaction window - the same confirmation Candle 3 used to provide, just
# without also requiring an FVG afterward (that job now belongs to the LTF
# crossover entries).
#
# Stop is tame and 1-minute-native: just beyond the entry bar's own high/low.
# No fixed target - a chandelier ATR trailing stop rides the trend and only
# ever ratchets in the trade's favor, so a trade that runs can actually
# realize a large R instead of being capped at a fixed multiple.

import numpy as np
import pandas as pd

from .crt_strategy import _htf_bias, detect_crt_patterns, wilder_atr
from .data import align_htf_to_ltf, resample_bars


def _bias_timeline(bias_bars: pd.DataFrame, patterns: pd.DataFrame, reaction_window_bars: int,
                    htf_bias: np.ndarray) -> pd.DataFrame:
    """Walks the bias timeframe bar by bar. A raw CRT only starts as a
    PENDING candidate if it agrees with `htf_bias` (the 1H/4H EMA-cross
    bias, confluence from a timeframe higher still than the CRT's own) - a
    bearish CRT against a bullish htf_bias is ignored outright, never even
    considered. A candidate that does qualify becomes live bias only once
    price closes beyond its own C2 opposite wick within
    `reaction_window_bars` - proving real follow-through, not just a raw
    sweep-and-reclaim. An already-live bias is left alone by a fresh
    same-direction pattern's *pending* cycle (it doesn't interrupt what's
    already confirmed) and only actually changes when its own invalidation
    trips or a new candidate independently reacts."""
    closes = bias_bars["close"].to_numpy()
    bull_crt = patterns["bull_crt"].to_numpy()
    bear_crt = patterns["bear_crt"].to_numpy()
    c2_high = patterns["c2_high"].to_numpy()
    c2_low = patterns["c2_low"].to_numpy()
    n = len(bias_bars)

    bias_dir = np.zeros(n, dtype=int)
    ref_high = np.full(n, np.nan)
    ref_low = np.full(n, np.nan)

    cur_dir = 0
    cur_high = cur_low = np.nan
    pend_dir = 0
    pend_high = pend_low = np.nan
    pend_bar = None

    for j in range(n):
        if bull_crt[j] and htf_bias[j] == 1:
            pend_dir, pend_high, pend_low, pend_bar = 1, c2_high[j], c2_low[j], j
        elif bear_crt[j] and htf_bias[j] == -1:
            pend_dir, pend_high, pend_low, pend_bar = -1, c2_high[j], c2_low[j], j
        elif pend_dir != 0:
            reacted = (closes[j] > pend_high) if pend_dir == 1 else (closes[j] < pend_low)
            if reacted:
                cur_dir, cur_high, cur_low = pend_dir, pend_high, pend_low
                pend_dir = 0
            elif j - pend_bar > reaction_window_bars:
                pend_dir = 0  # never confirmed - candidate is dead

        if cur_dir == 1 and not np.isnan(cur_low) and closes[j] < cur_low:
            cur_dir = 0  # reclaimed the swept low -> bullish thesis is dead
        elif cur_dir == -1 and not np.isnan(cur_high) and closes[j] > cur_high:
            cur_dir = 0  # reclaimed the swept high -> bearish thesis is dead

        bias_dir[j], ref_high[j], ref_low[j] = cur_dir, cur_high, cur_low

    return pd.DataFrame({"bias_dir": bias_dir, "ref_high": ref_high, "ref_low": ref_low}, index=bias_bars.index)


def run_crt_bias_ltf_strategy(
    df_1m: pd.DataFrame,
    bias_tf_rule: str,
    bias_tf_minutes: int,
    bias_atr_len: int = 14,
    c1_min_atr_mult: float = 0.8,
    c1_min_body_ratio: float = 0.6,
    c2_max_body_ratio: float = 0.5,
    liquidity_lookback: int = 10,
    ema_len: int = 50,
    reaction_window_min: int = 90,
    ltf_atr_len: int = 14,
    ltf_momentum_atr_mult: float = 0.15,
    stop_buffer_atr: float = 1.5,
    trail_atr_mult: float = 8.0,
    max_trade_minutes: int | None = None,
) -> pd.DataFrame:
    """CRT-as-bias on `bias_tf_rule` (e.g. "15min", "1h") is the CONFLUENCE
    layer only - it never wins or loses on its own. Entries fire on
    1-minute closes crossing the bias's C2 opposite-wick level BY AT LEAST
    `ltf_momentum_atr_mult` x 1-minute ATR (not a bare tick-cross - that was
    letting the tame stop get chopped by ordinary 1-minute noise before any
    real move developed), then ride a tame 1-minute stop with an ATR
    chandelier trailing exit. A CRT only counts as confluence if C1's swept
    side is a genuine local extreme (`liquidity_lookback`) AND the 1H/4H
    EMA-cross bias agrees with its direction - raw C1+C2 alone fires far too
    often to trust by itself. Returns every entry with its resolved outcome
    and r_multiple (computed against the INITIAL stop's risk, not the
    trailing level, so it reflects what was actually risked)."""
    bias_bars = resample_bars(df_1m, bias_tf_rule)
    bias_atr = wilder_atr(bias_bars, bias_atr_len)
    patterns = detect_crt_patterns(bias_bars, bias_atr, c1_min_atr_mult, c1_min_body_ratio, c2_max_body_ratio,
                                    liquidity_lookback)

    ctx_1h = _htf_bias(df_1m, "1h", ema_len)
    ctx_4h = _htf_bias(df_1m, "4h", ema_len)
    b1 = align_htf_to_ltf(bias_bars.index, ctx_1h, "1h")["bias"].fillna(0).to_numpy()
    b4 = align_htf_to_ltf(bias_bars.index, ctx_4h, "4h")["bias"].fillna(0).to_numpy()
    htf_bias = np.where(b4 != 0, b4, b1)

    reaction_window_bars = max(1, round(reaction_window_min / bias_tf_minutes))
    bias_tl = _bias_timeline(bias_bars, patterns, reaction_window_bars, htf_bias)

    aligned = align_htf_to_ltf(df_1m.index, bias_tl, bias_tf_rule)
    bias_dir = aligned["bias_dir"].fillna(0).to_numpy()
    ref_high = aligned["ref_high"].to_numpy()
    ref_low = aligned["ref_low"].to_numpy()

    atr_1m = wilder_atr(df_1m, ltf_atr_len).to_numpy()
    highs, lows, closes = df_1m["high"].to_numpy(), df_1m["low"].to_numpy(), df_1m["close"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    active_trade = False
    entry_i = stop_price = running_extreme = None
    is_long = None

    trades: list[dict] = []

    for i in range(1, n):
        close, high, low = closes[i], highs[i], lows[i]
        prev_close = closes[i - 1]
        bdir = bias_dir[i]

        if not active_trade:
            momentum = atr_1m[i] * ltf_momentum_atr_mult if not np.isnan(atr_1m[i]) else 0.0
            # Threshold sits `momentum` beyond the raw level - a bare tick-
            # cross of ref_high/ref_low no longer counts, only a close that
            # actually displaces past it with some real force. Comparing
            # prev_close against the SAME threshold (not the raw level)
            # keeps this a genuine crossing, so it can still re-trigger
            # several times per bias as price legs through it.
            is_long = None
            if bdir == 1 and not np.isnan(ref_high[i]):
                threshold = ref_high[i] + momentum
                if prev_close <= threshold and close > threshold:
                    is_long = True
            elif bdir == -1 and not np.isnan(ref_low[i]):
                threshold = ref_low[i] - momentum
                if prev_close >= threshold and close < threshold:
                    is_long = False

            if is_long is not None:
                buf = atr_1m[i] * stop_buffer_atr if not np.isnan(atr_1m[i]) else 0.0
                entry_price = close
                stop_price = (low - buf) if is_long else (high + buf)
                if stop_price != entry_price:
                    entry_i, active_trade = i, True
                    running_extreme = high if is_long else low
                    trades.append({
                        "time": idx[i], "direction": "long" if is_long else "short",
                        "entry_price": entry_price, "initial_stop": stop_price,
                        "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
                        "max_favorable_r": 0.0,
                    })

        elif active_trade:
            timed_out = max_trade_minutes is not None and (i - entry_i) >= max_trade_minutes
            risk = abs(trades[-1]["entry_price"] - trades[-1]["initial_stop"])

            if is_long:
                running_extreme = max(running_extreme, high)
                trail_candidate = running_extreme - trail_atr_mult * atr_1m[i] if not np.isnan(atr_1m[i]) else stop_price
                stop_price = max(stop_price, trail_candidate)
                hit_stop = low <= stop_price
                cur_favorable_r = (running_extreme - trades[-1]["entry_price"]) / risk if risk > 0 else 0.0
            else:
                running_extreme = min(running_extreme, low)
                trail_candidate = running_extreme + trail_atr_mult * atr_1m[i] if not np.isnan(atr_1m[i]) else stop_price
                stop_price = min(stop_price, trail_candidate)
                hit_stop = high >= stop_price
                cur_favorable_r = (trades[-1]["entry_price"] - running_extreme) / risk if risk > 0 else 0.0

            trades[-1]["max_favorable_r"] = max(trades[-1]["max_favorable_r"], cur_favorable_r)

            if hit_stop:
                # "win"/"loss" gets decided below from the r_multiple sign
                # once exit_price is known, not here.
                trades[-1].update(outcome="stopped", exit_time=idx[i], exit_price=stop_price)
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

    risk = (result["entry_price"] - result["initial_stop"]).abs()
    direction_sign = np.where(result["direction"] == "long", 1, -1)
    result["r_multiple"] = (result["exit_price"] - result["entry_price"]) * direction_sign / risk

    stopped = result["outcome"] == "stopped"
    result.loc[stopped, "outcome"] = np.where(result.loc[stopped, "r_multiple"] > 0, "win", "loss")
    return result
