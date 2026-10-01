# Directly tests whether price reacts when it first touches an untapped
# 1H/4H FVG or RB, and whether that depends on VWAP sitting inside the zone
# or the zone being "clean" (no old liquidity resting inside it). No
# entry/exit strategy here - this measures raw PD array behavior on its own,
# independent of the earlier bias -> POI -> MSS -> retest pipeline.

import numpy as np
import pandas as pd

from .data import align_htf_to_ltf
from .strategy import _build_htf_context


def collect_htf_zones(df_1m, htf_pivot_len=5, min_htf_fvg_pts=5.0, rb_wick_ratio=2.0):
    """Every 1H/4H FVG/RB detected over the dataset, as a flat list of dicts.
    bar_created is expressed in the 1-minute (LTF) index, matching df_1m."""
    zones = []
    n = len(df_1m)

    for rule, tf in [("1h", "1H"), ("4h", "4H")]:
        ctx = _build_htf_context(df_1m, rule, htf_pivot_len, min_htf_fvg_pts, rb_wick_ratio)
        aligned = align_htf_to_ltf(df_1m.index, ctx, rule)

        open_arr = aligned["_htf_open"].to_numpy()
        bull_fvg, bear_fvg = aligned["bull_fvg"].to_numpy(), aligned["bear_fvg"].to_numpy()
        fvg_bt, fvg_bb = aligned["fvg_bull_top"].to_numpy(), aligned["fvg_bull_bottom"].to_numpy()
        fvg_rt, fvg_rb = aligned["fvg_bear_top"].to_numpy(), aligned["fvg_bear_bottom"].to_numpy()
        fvg_bull_clean, fvg_bear_clean = aligned["fvg_bull_clean"].to_numpy(), aligned["fvg_bear_clean"].to_numpy()
        bull_rb, bear_rb = aligned["bull_rb"].to_numpy(), aligned["bear_rb"].to_numpy()
        rb_bt, rb_bb = aligned["rb_bull_top"].to_numpy(), aligned["rb_bull_bottom"].to_numpy()
        rb_rt, rb_rb = aligned["rb_bear_top"].to_numpy(), aligned["rb_bear_bottom"].to_numpy()
        rb_bull_clean, rb_bear_clean = aligned["rb_bull_clean"].to_numpy(), aligned["rb_bear_clean"].to_numpy()

        last_open = None
        for i in range(n):
            cur_open = open_arr[i]
            if pd.notna(cur_open) and cur_open != last_open:
                if bull_fvg[i]:
                    zones.append(dict(top=fvg_bt[i], bottom=fvg_bb[i], direction="bull", kind="FVG", tf=tf,
                                       bar_created=i, clean=bool(fvg_bull_clean[i])))
                if bear_fvg[i]:
                    zones.append(dict(top=fvg_rt[i], bottom=fvg_rb[i], direction="bear", kind="FVG", tf=tf,
                                       bar_created=i, clean=bool(fvg_bear_clean[i])))
                if bull_rb[i]:
                    zones.append(dict(top=rb_bt[i], bottom=rb_bb[i], direction="bull", kind="RB", tf=tf,
                                       bar_created=i, clean=bool(rb_bull_clean[i])))
                if bear_rb[i]:
                    zones.append(dict(top=rb_rt[i], bottom=rb_rb[i], direction="bear", kind="RB", tf=tf,
                                       bar_created=i, clean=bool(rb_bear_clean[i])))
                last_open = cur_open

    return zones


def _first_true_index(mask: np.ndarray):
    return int(np.argmax(mask)) if mask.any() else None


def build_zone_entries(zones: list, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Turns every zone's first touch into an actual trade candidate, compatible
    with backtest.resolve_trade_outcomes(): entry at the touch bar's close,
    stop at the zone's far edge (the level that invalidates the reaction
    thesis), target at 1x the zone's own height beyond the near edge - same
    definitions as find_zone_touches_and_reactions(), just packaged as
    tradeable entries instead of a reacted/failed outcome.
    """
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    records = []
    for z in zones:
        top, bottom, direction = z["top"], z["bottom"], z["direction"]
        height = top - bottom
        if height <= 0:
            continue
        start = z["bar_created"] + 1
        if start >= n:
            continue

        touch_mask = (lows[start:] <= top) & (highs[start:] >= bottom)
        rel_touch = _first_true_index(touch_mask)
        if rel_touch is None:
            continue
        touch_i = start + rel_touch

        if direction == "bull":
            stop_price, target_price = bottom, top + height
        else:
            stop_price, target_price = top, bottom - height

        records.append({
            "time": idx[touch_i],
            "direction": "long" if direction == "bull" else "short",
            "entry_price": closes[touch_i],
            "stop_price": stop_price,
            "target_price": target_price,
            "bar_index": touch_i,
            "kind": z["kind"],
            "tf": z["tf"],
            "clean": z["clean"],
        })

    return pd.DataFrame(records)


def find_zone_touches_and_reactions(zones: list, df_1m: pd.DataFrame, vwap: pd.Series) -> pd.DataFrame:
    """
    For each zone, finds its first touch after creation, then walks forward
    to see which happens first: price reacts away by 1x the zone's own
    height (outcome="reacted"), or price fully closes through the zone's far
    edge, invalidating it (outcome="failed"). "undetermined" means neither
    happened before the data ran out. vwap_confluence is True when VWAP sat
    inside the zone at the moment of that first touch.
    """
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    vwap_arr = vwap.to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    records = []
    for z in zones:
        top, bottom, direction = z["top"], z["bottom"], z["direction"]
        height = top - bottom
        if height <= 0:
            continue
        start = z["bar_created"] + 1
        if start >= n:
            continue

        touch_mask = (lows[start:] <= top) & (highs[start:] >= bottom)
        rel_touch = _first_true_index(touch_mask)
        if rel_touch is None:
            continue
        touch_i = start + rel_touch

        vwap_at_touch = vwap_arr[touch_i]
        close_at_touch = closes[touch_i]
        if np.isnan(vwap_at_touch):
            vwap_trend, trend_aligned = None, None
        else:
            vwap_trend = "up" if close_at_touch > vwap_at_touch else "down"
            trend_aligned = (direction == "bull" and vwap_trend == "up") or (direction == "bear" and vwap_trend == "down")

        if direction == "bull":
            invalidation, target = bottom, top + height
            invalidated_mask = lows[touch_i:] < invalidation
            reacted_mask = highs[touch_i:] >= target
        else:
            invalidation, target = top, bottom - height
            invalidated_mask = highs[touch_i:] > invalidation
            reacted_mask = lows[touch_i:] <= target

        inv_idx = _first_true_index(invalidated_mask)
        react_idx = _first_true_index(reacted_mask)

        if inv_idx is None and react_idx is None:
            outcome = "undetermined"
        elif inv_idx is None:
            outcome = "reacted"
        elif react_idx is None:
            outcome = "failed"
        else:
            outcome = "failed" if inv_idx <= react_idx else "reacted"

        records.append({
            "touch_time": idx[touch_i],
            "direction": direction,
            "kind": z["kind"],
            "tf": z["tf"],
            "clean": z["clean"],
            "vwap_trend": vwap_trend,
            "trend_aligned": trend_aligned,
            "outcome": outcome,
        })

    return pd.DataFrame(records)
