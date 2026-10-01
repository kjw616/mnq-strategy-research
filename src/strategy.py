# Full port of the MNQ ICT A+ Setup Finder Pine Script: bias -> POI -> MSS ->
# LTF retest entry. Walks 1-minute bars forward one at a time (like Pine
# itself does) and returns every entry signal found. Cross-reference against
# MNQ_ICT_A_Plus_Setup_Finder.pine - this mirrors its FSM section closely on
# purpose, so the two can be checked against each other.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import align_htf_to_ltf, resample_bars
from .liquidity import tag_zone_purity
from .poi import check_fvg, check_rb
from .structure import structure_bias

ST_SEARCH, ST_POI_ACTIVE, ST_ARMED_MSS, ST_AWAIT_ENTRY = range(4)


@dataclass
class Poi:
    top: float
    bottom: float
    direction: str  # "bull" or "bear"
    kind: str  # "FVG", "RB", "IFVG"
    tf: str  # "1H", "4H", "LTF"
    bar_created: int
    mitigated: bool = False
    clean: bool | None = None  # True = no unswept liquidity resting inside; None = not evaluated (LTF zones)


def wilder_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Same smoothing Pine's ta.atr() uses (Wilder's, not a plain average)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _build_htf_context(df_1m, rule, pivot_len, min_fvg_pts, rb_wick_ratio):
    """Everything we need to know about each 1H or 4H bar, in one wide frame."""
    htf = resample_bars(df_1m, rule)
    struct = structure_bias(htf, pivot_len)
    fvg = check_fvg(htf, min_fvg_pts)
    rb = check_rb(htf, struct["swing_high"], struct["swing_low"], rb_wick_ratio)

    ctx = pd.DataFrame(index=htf.index)
    ctx["bias"] = struct["bias"]
    ctx["swing_high"] = struct["swing_high"]
    ctx["swing_low"] = struct["swing_low"]
    ctx["bull_fvg"] = fvg["bull_fvg"]
    ctx["bear_fvg"] = fvg["bear_fvg"]
    ctx["fvg_bull_top"] = fvg["bull_top"]
    ctx["fvg_bull_bottom"] = fvg["bull_bottom"]
    ctx["fvg_bear_top"] = fvg["bear_top"]
    ctx["fvg_bear_bottom"] = fvg["bear_bottom"]
    ctx["bull_rb"] = rb["bull_rb"]
    ctx["bear_rb"] = rb["bear_rb"]
    ctx["rb_bull_top"] = rb["bull_top"]
    ctx["rb_bull_bottom"] = rb["bull_bottom"]
    ctx["rb_bear_top"] = rb["bear_top"]
    ctx["rb_bear_bottom"] = rb["bear_bottom"]

    fvg_bull_clean, fvg_bear_clean = tag_zone_purity(
        htf, pivot_len,
        fvg["bull_fvg"], fvg["bull_top"], fvg["bull_bottom"],
        fvg["bear_fvg"], fvg["bear_top"], fvg["bear_bottom"],
    )
    rb_bull_clean, rb_bear_clean = tag_zone_purity(
        htf, pivot_len,
        rb["bull_rb"], rb["bull_top"], rb["bull_bottom"],
        rb["bear_rb"], rb["bear_top"], rb["bear_bottom"],
        exclude_bull_price=struct["swing_low"],
        exclude_bear_price=struct["swing_high"],
    )
    ctx["fvg_bull_clean"] = fvg_bull_clean
    ctx["fvg_bear_clean"] = fvg_bear_clean
    ctx["rb_bull_clean"] = rb_bull_clean
    ctx["rb_bear_clean"] = rb_bear_clean
    return ctx


def _add_poi(pool, top, bottom, direction, kind, tf, bar_idx, clean=None):
    pool.append(Poi(top=top, bottom=bottom, direction=direction, kind=kind, tf=tf, bar_created=bar_idx, clean=clean))


def _trim_pool(pool, max_keep):
    while len(pool) > max_keep:
        idx_to_remove = next((i for i, p in enumerate(pool) if p.mitigated), None)
        if idx_to_remove is None:
            idx_to_remove = 0
        pool.pop(idx_to_remove)


def _update_htf_mitigation(pool, bar_high, bar_low, bar_idx, max_age_bars):
    for p in pool:
        if not p.mitigated:
            if bar_low <= p.top and bar_high >= p.bottom:
                p.mitigated = True
            elif bar_idx - p.bar_created > max_age_bars:
                p.mitigated = True


def _select_poi(pool, want_dir, ref_price):
    best, best_dist = None, None
    for p in pool:
        if not p.mitigated and p.direction == want_dir:
            valid = p.top < ref_price if want_dir == "bull" else p.bottom > ref_price
            if valid:
                dist = (ref_price - p.top) if want_dir == "bull" else (p.bottom - ref_price)
                if best_dist is None or dist < best_dist:
                    best_dist, best = dist, p
    return best


def _update_ltf_zones(pool, close, bar_idx, max_age_bars):
    for p in list(pool):  # snapshot - we may append new IFVGs while iterating
        if not p.mitigated:
            if p.direction == "bull" and close < p.bottom:
                p.mitigated = True
                _add_poi(pool, p.top, p.bottom, "bear", "IFVG", "LTF", bar_idx)
            elif p.direction == "bear" and close > p.top:
                p.mitigated = True
                _add_poi(pool, p.top, p.bottom, "bull", "IFVG", "LTF", bar_idx)
            elif bar_idx - p.bar_created > max_age_bars:
                p.mitigated = True


def _select_entry_zone(pool, want_dir, mss_bar, search_window):
    for p in reversed(pool):  # newest first
        if not p.mitigated and p.direction == want_dir and p.bar_created >= mss_bar - search_window:
            return p
    return None


def _pick_target(swing_a, swing_b, entry_price, direction, risk, max_r):
    """Nearest untapped opposing swing level (1H or 4H, whichever is closer)
    that's actually on the correct side of entry to serve as a target - capped
    at `max_r` x the trade's own risk, so a distant swing level (common when
    stops are tight) doesn't produce an unrealistic target."""
    candidates = []
    for s in (swing_a, swing_b):
        if not np.isnan(s):
            if direction == "long" and s > entry_price:
                candidates.append(s)
            elif direction == "short" and s < entry_price:
                candidates.append(s)
    if not candidates:
        return np.nan
    nearest = min(candidates) if direction == "long" else max(candidates)
    if risk is None or np.isnan(risk) or risk <= 0:
        return nearest
    cap_price = entry_price + max_r * risk if direction == "long" else entry_price - max_r * risk
    return min(nearest, cap_price) if direction == "long" else max(nearest, cap_price)


def run_strategy(
    df_1m: pd.DataFrame,
    htf_pivot_len: int = 5,
    min_swing_atr_mult: float = 0.5,
    min_htf_fvg_pts: float = 5.0,
    min_ltf_fvg_pts: float = 1.0,
    rb_wick_ratio: float = 2.0,
    poi_max_age_min: int = 1000,
    max_pois_tracked: int = 3,
    reaction_window_min: int = 100,
    fvg_search_min: int = 25,
    max_target_r: float = 5.0,
) -> pd.DataFrame:
    """
    Runs the full pipeline over 1-minute OHLCV data and returns every entry
    signal found: timestamp, direction, entry price. Assumes df_1m is 1-minute
    bars (so "minutes" timing inputs equal bar counts directly - Pine has to
    convert because its chart timeframe can vary; ours doesn't).
    """
    ctx_1h = _build_htf_context(df_1m, "1h", htf_pivot_len, min_htf_fvg_pts, rb_wick_ratio)
    ctx_4h = _build_htf_context(df_1m, "4h", htf_pivot_len, min_htf_fvg_pts, rb_wick_ratio)

    a1 = align_htf_to_ltf(df_1m.index, ctx_1h, "1h")
    a4 = align_htf_to_ltf(df_1m.index, ctx_4h, "4h")

    atr_arr = wilder_atr(df_1m, 14).to_numpy()
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    bias1h_arr, bias4h_arr = a1["bias"].to_numpy(), a4["bias"].to_numpy()
    open_1h_arr, open_4h_arr = a1["_htf_open"].to_numpy(), a4["_htf_open"].to_numpy()
    swh_1h_arr, swl_1h_arr = a1["swing_high"].to_numpy(), a1["swing_low"].to_numpy()
    swh_4h_arr, swl_4h_arr = a4["swing_high"].to_numpy(), a4["swing_low"].to_numpy()

    def cols(a):
        return (
            a["bull_fvg"].to_numpy(), a["bear_fvg"].to_numpy(),
            a["fvg_bull_top"].to_numpy(), a["fvg_bull_bottom"].to_numpy(),
            a["fvg_bear_top"].to_numpy(), a["fvg_bear_bottom"].to_numpy(),
            a["bull_rb"].to_numpy(), a["bear_rb"].to_numpy(),
            a["rb_bull_top"].to_numpy(), a["rb_bull_bottom"].to_numpy(),
            a["rb_bear_top"].to_numpy(), a["rb_bear_bottom"].to_numpy(),
            a["fvg_bull_clean"].to_numpy(), a["fvg_bear_clean"].to_numpy(),
            a["rb_bull_clean"].to_numpy(), a["rb_bear_clean"].to_numpy(),
        )

    (bull_fvg_1h, bear_fvg_1h, fvgbt_1h, fvgbb_1h, fvgrt_1h, fvgrb_1h,
     bull_rb_1h, bear_rb_1h, rbbt_1h, rbbb_1h, rbrt_1h, rbrb_1h,
     fvg_bull_clean_1h, fvg_bear_clean_1h, rb_bull_clean_1h, rb_bear_clean_1h) = cols(a1)
    (bull_fvg_4h, bear_fvg_4h, fvgbt_4h, fvgbb_4h, fvgrt_4h, fvgrb_4h,
     bull_rb_4h, bear_rb_4h, rbbt_4h, rbbb_4h, rbrt_4h, rbrb_4h,
     fvg_bull_clean_4h, fvg_bear_clean_4h, rb_bull_clean_4h, rb_bear_clean_4h) = cols(a4)

    poi_max_age_bars = poi_max_age_min
    reaction_window_bars = reaction_window_min
    fvg_search_bars = fvg_search_min

    htf_pool: list[Poi] = []
    ltf_zones: list[Poi] = []

    fsm_state = ST_SEARCH
    active_poi = None
    entry_zone_obj = None
    poi_reached_bar = None
    mss_bar_idx = None
    last_bias_at_selection = None
    armed_extreme = None
    armed_swing_ref = None
    stop_price_for_trade = None

    last_open_1h = None
    last_open_4h = None

    trades = []

    for i in range(n):
        high, low, close = highs[i], lows[i], closes[i]

        # --- Step 2: new HTF bar closes -> check for new POIs -----------
        cur_open_1h = open_1h_arr[i]
        if pd.notna(cur_open_1h) and cur_open_1h != last_open_1h:
            if bull_fvg_1h[i]:
                _add_poi(htf_pool, fvgbt_1h[i], fvgbb_1h[i], "bull", "FVG", "1H", i, clean=bool(fvg_bull_clean_1h[i]))
            if bear_fvg_1h[i]:
                _add_poi(htf_pool, fvgrt_1h[i], fvgrb_1h[i], "bear", "FVG", "1H", i, clean=bool(fvg_bear_clean_1h[i]))
            if bull_rb_1h[i]:
                _add_poi(htf_pool, rbbt_1h[i], rbbb_1h[i], "bull", "RB", "1H", i, clean=bool(rb_bull_clean_1h[i]))
            if bear_rb_1h[i]:
                _add_poi(htf_pool, rbrt_1h[i], rbrb_1h[i], "bear", "RB", "1H", i, clean=bool(rb_bear_clean_1h[i]))
            last_open_1h = cur_open_1h

        cur_open_4h = open_4h_arr[i]
        if pd.notna(cur_open_4h) and cur_open_4h != last_open_4h:
            if bull_fvg_4h[i]:
                _add_poi(htf_pool, fvgbt_4h[i], fvgbb_4h[i], "bull", "FVG", "4H", i, clean=bool(fvg_bull_clean_4h[i]))
            if bear_fvg_4h[i]:
                _add_poi(htf_pool, fvgrt_4h[i], fvgrb_4h[i], "bear", "FVG", "4H", i, clean=bool(fvg_bear_clean_4h[i]))
            if bull_rb_4h[i]:
                _add_poi(htf_pool, rbbt_4h[i], rbbb_4h[i], "bull", "RB", "4H", i, clean=bool(rb_bull_clean_4h[i]))
            if bear_rb_4h[i]:
                _add_poi(htf_pool, rbrt_4h[i], rbrb_4h[i], "bear", "RB", "4H", i, clean=bool(rb_bear_clean_4h[i]))
            last_open_4h = cur_open_4h

        _update_htf_mitigation(htf_pool, high, low, i, poi_max_age_bars)
        _trim_pool(htf_pool, max_pois_tracked * 4)

        # --- Step 3 prep: LTF FVG / IFVG pool ----------------------------
        if i >= 2:
            h2, l2 = highs[i - 2], lows[i - 2]
            gap_bull, gap_bear = low - h2, l2 - high
            if gap_bull > 0 and gap_bull >= min_ltf_fvg_pts:
                _add_poi(ltf_zones, low, h2, "bull", "FVG", "LTF", i)
            if gap_bear > 0 and gap_bear >= min_ltf_fvg_pts:
                _add_poi(ltf_zones, l2, high, "bear", "FVG", "LTF", i)
        _update_ltf_zones(ltf_zones, close, i, poi_max_age_bars)
        _trim_pool(ltf_zones, 30)

        # --- Bias -----------------------------------------------------------
        b4 = bias4h_arr[i]
        b1 = bias1h_arr[i]
        b4 = 0 if np.isnan(b4) else int(b4)
        b1 = 0 if np.isnan(b1) else int(b1)
        final_bias = b4 if b4 != 0 else b1

        # --- Bias flip safety reset ------------------------------------
        if fsm_state != ST_SEARCH and last_bias_at_selection is not None and final_bias != last_bias_at_selection:
            fsm_state = ST_SEARCH
            active_poi = None
            entry_zone_obj = None

        # --- FSM ---------------------------------------------------------
        if fsm_state == ST_SEARCH and final_bias != 0:
            candidate = _select_poi(htf_pool, "bull" if final_bias == 1 else "bear", close)
            if candidate is not None:
                active_poi = candidate
                last_bias_at_selection = final_bias
                fsm_state = ST_POI_ACTIVE

        if fsm_state == ST_POI_ACTIVE and active_poi is not None:
            reached = low <= active_poi.top and high >= active_poi.bottom
            if reached:
                active_poi.mitigated = True
                poi_reached_bar = i
                armed_extreme = low if active_poi.direction == "bull" else high
                armed_swing_ref = None
                fsm_state = ST_ARMED_MSS
            elif active_poi.mitigated or (i - active_poi.bar_created > poi_max_age_bars):
                fsm_state = ST_SEARCH
                active_poi = None

        if fsm_state == ST_ARMED_MSS and active_poi is not None:
            is_long = active_poi.direction == "bull"
            if armed_swing_ref is None:
                swing_big_enough = False
            else:
                swing_depth = (armed_swing_ref - armed_extreme) if is_long else (armed_extreme - armed_swing_ref)
                atr_now = atr_arr[i]
                swing_big_enough = not np.isnan(atr_now) and swing_depth >= min_swing_atr_mult * atr_now

            prev_close = closes[i - 1] if i > 0 else np.nan
            bull_mss = is_long and swing_big_enough and prev_close <= armed_swing_ref and close > armed_swing_ref
            bear_mss = (not is_long) and swing_big_enough and prev_close >= armed_swing_ref and close < armed_swing_ref

            if bull_mss or bear_mss:
                mss_bar_idx = i
                stop_price_for_trade = armed_swing_ref  # the level that broke to confirm the shift
                fsm_state = ST_AWAIT_ENTRY
            else:
                if is_long:
                    if low < armed_extreme:
                        armed_extreme, armed_swing_ref = low, None
                    elif armed_swing_ref is None or high > armed_swing_ref:
                        armed_swing_ref = high
                else:
                    if high > armed_extreme:
                        armed_extreme, armed_swing_ref = high, None
                    elif armed_swing_ref is None or low < armed_swing_ref:
                        armed_swing_ref = low

                if i - poi_reached_bar > reaction_window_bars:
                    fsm_state = ST_SEARCH
                    active_poi = None

        if fsm_state == ST_AWAIT_ENTRY and active_poi is not None:
            if entry_zone_obj is None:
                entry_zone_obj = _select_entry_zone(ltf_zones, active_poi.direction, mss_bar_idx, fvg_search_bars)
            if entry_zone_obj is not None:
                retest = low <= entry_zone_obj.top and high >= entry_zone_obj.bottom
                if retest:
                    is_long = active_poi.direction == "bull"
                    direction = "long" if is_long else "short"
                    risk_for_trade = abs(close - stop_price_for_trade) if stop_price_for_trade is not None else np.nan
                    target_price = _pick_target(swh_1h_arr[i], swh_4h_arr[i], close, direction, risk_for_trade, max_target_r) if is_long \
                        else _pick_target(swl_1h_arr[i], swl_4h_arr[i], close, direction, risk_for_trade, max_target_r)
                    trades.append({
                        "time": idx[i],
                        "direction": direction,
                        "entry_price": close,
                        "stop_price": stop_price_for_trade,
                        "target_price": target_price,
                        "bar_index": i,
                        "poi_kind": active_poi.kind,
                        "poi_tf": active_poi.tf,
                        "poi_clean": active_poi.clean,
                    })
                    fsm_state = ST_SEARCH
                    active_poi = None
                    entry_zone_obj = None
            if mss_bar_idx is not None and i - mss_bar_idx > reaction_window_bars:
                fsm_state = ST_SEARCH
                active_poi = None
                entry_zone_obj = None

    return pd.DataFrame(
        trades,
        columns=["time", "direction", "entry_price", "stop_price", "target_price", "bar_index", "poi_kind", "poi_tf", "poi_clean"],
    )
