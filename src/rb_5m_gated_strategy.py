# 5-minute Rejection Blocks (poi.check_rb - a candle that wicks through a
# confirmed 5m swing level and closes back on the other side, wick >=
# rb_wick_ratio x body, same shape/default ratio this codebase already uses
# for RBs on 4H/1H in dashboard_engine.py/strategy.py) gated by HTF bias
# agreement (htf_sweep_strategy.htf_bias_agreement: 5m AND 30m
# structure_bias() must agree and match the RB's own direction).
#
# The sweep and the rejection are the SAME candle here, not two separate
# stages - check_rb already requires wicking beyond a confirmed swing level
# (that's the sweep) and closing back (that's the rejection), so no
# separate 1-minute sweep-detection stage is needed. This is a deliberate
# change from fib_golden_zone_strategy.py's design, driven by what
# rb_5m_backtest.py's raw-RB study found: an RB's own candle shape (wick/
# body ratio, range) carried no signal on its own, so the bet here is that
# the missing ingredient is DIRECTIONAL CONTEXT (does the broader structure
# agree with this particular rejection), not a stricter candle-shape filter.
#
# Entry is a 1-minute retest of the RB's own zone [rb_near, rb_far] -
# rb_near = the candle's own body edge ("where the wick meets the body"),
# fills the moment price returns to it via a resting limit; stop sits
# rb_stop_buffer_points beyond rb_far (the candle's own wick extreme) -
# fixed points, not ATR-scaled, per spec (this is meant to always be a
# small, tight, easy-to-reason-about number, not something that grows with
# volatility).
#
# One trade tracked at a time, same discipline as every other strategy in
# this repo (htf_sweep_strategy.py, fib_golden_zone_strategy.py) - real
# capital can't be in two setups simultaneously. If multiple live,
# bias-aligned RBs are still waiting for a retest and more than one gets
# touched on the same bar, the OLDEST (first formed) wins - simplest
# defensible tie-break, not a scored choice between them.

import numpy as np
import pandas as pd

from .data import resample_bars
from .htf_sweep_strategy import htf_bias_agreement
from .poi import check_rb
from .structure import structure_bias


def _find_5m_rbs(df_1m: pd.DataFrame, pivot_length: int, rb_wick_ratio: float) -> pd.DataFrame:
    """Every 5-minute RB (bull or bear), one row each - same detection as
    rb_5m_backtest.find_5m_rbs, duplicated here (not imported) so this file
    has no dependency on that exploratory script and can evolve separately."""
    df_5m = resample_bars(df_1m, "5min")
    struct = structure_bias(df_5m, pivot_length)
    rb = check_rb(df_5m, struct["swing_high"], struct["swing_low"], rb_wick_ratio)

    o, c = df_5m["open"], df_5m["close"]
    body = (c - o).abs()

    rows = []
    for direction, flag_col, near_col, far_col in [
        (1, "bull_rb", "bull_top", "bull_bottom"),
        (-1, "bear_rb", "bear_bottom", "bear_top"),
    ]:
        flags = rb[flag_col]
        for t in flags.index[flags.fillna(False)]:
            near, far = rb.loc[t, near_col], rb.loc[t, far_col]
            b = body.loc[t]
            wick = abs(near - far)
            rows.append({
                "formed_at": t, "dir": direction, "rb_near": near, "rb_far": far,
                "wick_body_ratio": (wick / b) if b > 0 else np.inf,
            })
    out = pd.DataFrame(rows)
    return out.sort_values("formed_at").reset_index(drop=True)


def run_rb_5m_gated_strategy(
    df_1m: pd.DataFrame,
    pivot_length: int = 5,
    rb_wick_ratio: float = 2.0,
    use_htf_bias: bool = True,
    bias_5m_length: int = 3,
    bias_30m_length: int = 3,
    rb_stop_buffer_points: float = 5.0,
    target_r_multiple: float = 8.0,
    max_retest_hours: float = 4.0,
    max_trade_bars: int | None = None,
) -> pd.DataFrame:
    """Runs the full FSM bar by bar on 1-minute data. Returns every entry
    with its resolved outcome, r_multiple, and the RB geometry it was
    measured against (rb_near, rb_far, wick_body_ratio), same output shape
    (direction "long"/"short", outcome in {win,loss,timeout,
    open_at_data_end}) as htf_sweep_strategy.py / fib_golden_zone_strategy.py
    so it plugs into the same report.py/backtest helpers unchanged."""
    rbs = _find_5m_rbs(df_1m, pivot_length, rb_wick_ratio)

    if use_htf_bias:
        bias = htf_bias_agreement(df_1m, bias_5m_length, bias_30m_length)
        bias_series = pd.Series(bias, index=df_1m.index)
        # Bias reading in effect at the moment each RB's own 5m candle
        # closes (the same instant its zone becomes knowable) - backward
        # as-of, same non-repainting convention align_htf_to_ltf uses
        # everywhere else in this codebase.
        rb_bias = bias_series.reindex(rbs["formed_at"], method="ffill").to_numpy()
        rbs = rbs[rbs["dir"] == rb_bias].reset_index(drop=True)

    idx = df_1m.index
    n = len(df_1m)
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()

    formed_bar = idx.searchsorted(rbs["formed_at"].to_numpy(), side="left")
    expire_at = rbs["formed_at"] + pd.Timedelta(hours=max_retest_hours)
    expire_bar = idx.searchsorted(expire_at.to_numpy(), side="left")

    live: list[dict] = []  # candidate RBs formed but not yet retested/expired
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
