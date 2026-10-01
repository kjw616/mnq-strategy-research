# Python port of Liquidity_FVG_v1.pine: sweep of an Asia/London session
# high or low -> reversal confirmed by a close through the FVG formed on
# the approach into that sweep -> that FVG inverts (flips role) -> entry on
# retracement into it -> SL at the swept level -> TP at the nearest
# untapped opposing session level. One trade tracked at a time, same as the
# live indicator (a new setup can't anchor while a trade is still open).
#
# Fix vs. the first live version of the indicator: if no opposing session
# level is still untapped at confirmation time, the setup is dropped
# instead of firing an entry with no target. Those were sitting open
# forever on the chart since a "hit" check against a missing target never
# fires - never losing, so they inflated the apparent win rate without
# ever having a real chance to lose. `skipped_no_target` in the returned
# diagnostics counts how often that happened.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import session_levels
from .structure import find_pivot_highs, find_pivot_lows


@dataclass
class _FVG:
    top: float
    bottom: float
    bar_index: int = -1


def _passes_filters(entry_price, sl_price, tp_price, ny_hour,
                     use_time_filter, trading_start_hour, trading_end_hour,
                     use_rr_filter, avoid_rr_min, avoid_rr_max) -> bool:
    time_ok = (not use_time_filter) or (trading_start_hour <= ny_hour <= trading_end_hour)
    if not use_rr_filter:
        return time_ok
    risk = abs(entry_price - sl_price)
    planned_rr = abs(tp_price - entry_price) / risk if risk > 0 else np.nan
    rr_ok = not (avoid_rr_min <= planned_rr <= avoid_rr_max)
    return time_ok and rr_ok


def run_liquidity_fvg_strategy(
    df_1m: pd.DataFrame,
    tz: str = "America/New_York",
    asia_start_hour: int = 18,
    asia_end_offset_hours: int = 24,
    london_start_hour: int = 2,
    london_end_offset_hours: int = 5,
    use_time_filter: bool = True,
    trading_start_hour: int = 7,
    trading_end_hour: int = 10,
    use_rr_filter: bool = True,
    avoid_rr_min: float = 10.0,
    avoid_rr_max: float = 20.0,
    fixed_r_target: float | None = None,
    use_mss_confirmation: bool = False,
    mss_pivot_length: int = 3,
    use_normal_fvg_retest: bool = False,
    sl_shrink_factor: float = 1.0,
    hold_tp_fixed: bool = False,
    tick_size: float | None = None,
    require_close_in_zone: bool = False,
    require_pristine_fvg: bool = False,
    max_bars_awaiting_confirm: int | None = None,
    max_bars_awaiting_retest: int | None = None,
    fvg_selection_window: int | None = None,
    market_order_entry: bool = False,
    require_price_reaches_entry: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Filter defaults came out of a 2023/2024 walk-forward backtest and held
    up in BOTH years independently, not just on the combined sample: NY AM
    (7-10am NY) held steady at avg R +0.80/+0.76 while every other hour of
    the day decayed from +0.58 (2023) to +0.10 (2024); trades planning a
    10-20x reward:risk at entry were flat-to-negative in both years while
    every other planned-R band was positive in both.

    `fixed_r_target`: if set, TP is entry +/- that many R (using the same SL)
    instead of the nearest untapped opposing session level - tests the idea
    that most trades are directionally right but don't travel far enough to
    reach a far liquidity-pool target before reversing.

    `use_mss_confirmation`: if set, reversal confirmation additionally
    requires the close to break the most recent confirmed LTF swing point
    (the last pullback high/low that formed on the approach into the sweep,
    on the 1-minute chart itself) - not just the FVG boundary. A real market
    structure shift, not only a gap fill. Snapshotted once at anchor time so
    it can't drift to reflect the recovery move itself. Uses the same
    non-repainting confirm-lag as structure.structure_bias(): a pivot at bar
    i only becomes usable `mss_pivot_length` bars later, same moment Pine's
    ta.pivothigh/pivotlow would reveal it.

    `use_normal_fvg_retest`: if set, the approach-leg FVG is discarded once
    it's confirmed the reversal, and entry instead waits for a FRESH FVG to
    form in the new direction and retests THAT one - a conventional FVG
    retest rather than treating the approach-leg FVG itself as inverted.

    `tick_size`: if set, entry/SL/TP are rounded to this grid before use,
    matching the live Pine indicator's math.round_to_mintick calls - left
    off by default since the original backtest numbers never applied it.

    `require_close_in_zone`: if set, entry requires the bar's CLOSE (not
    just its high/low range) to land inside the retest zone - a real held
    retest, not a single volatile candle wicking through the whole zone (and
    often past SL/TP too) without ever settling there.

    `require_pristine_fvg`: if set, an FVG becomes ineligible for use the
    moment price touches into its range AT ALL (even a brief wick) - not
    just once it's fully closed through, the normal "filled" rule. A gap
    that's already been dipped into once is a weaker, less believable
    unmitigated zone to invert and retest later.

    `max_bars_awaiting_confirm` / `max_bars_awaiting_retest`: if set, a
    pending setup expires (same as an outright invalidation) if it sits
    waiting longer than this many bars for confirmation, or for the retest
    after confirming. Without a cap, a setup can sit pending for hours
    before finally triggering - a valid but much lower-conviction version of
    the same pattern as a sharp, fast V-shaped reversal that plays out in
    minutes. These caps let you require the fast version specifically.

    `fvg_selection_window`: if set, anchoring doesn't just grab whichever
    pending FVG formed most recently - it looks back this many bars and
    picks the LARGEST (tallest) opposing FVG among only those recent
    candidates, instead of always the last one in. Bounded to a recent
    window deliberately - the full pending pool can contain gaps from
    months earlier that are still technically "unfilled" but practically
    irrelevant to the current setup.

    `market_order_entry`: default False assumes a resting LIMIT order sits at
    the zone's exact midpoint and fills the instant the bar's range overlaps
    it - the best price obtainable within the zone, but a price the bar's
    range may only have brushed past on a wick rather than genuinely traded
    through to the middle. Set True to instead assume a MARKET order firing
    the instant price first touches the zone at all: for a long, that's the
    zone's top edge (price is retracing DOWN into it from above); for a
    short, the bottom edge (retracing UP into it from below). Always a worse
    fill than the midpoint - the entry price only changes, not whether a
    trade triggers at all, since the retest condition itself (bar range
    overlaps the zone) is unchanged either way. Stop and target are still
    derived from whichever entry_candidate results, same as always.

    `require_price_reaches_entry`: the entry trigger and the assumed fill
    price were checking two DIFFERENT things by default - "did the bar's
    range overlap the zone at all" vs "fill at the midpoint" - so a bar that
    only grazed the far edge of the zone from setup_entry could trigger a
    trade priced as if it had reached the middle, when real price never got
    anywhere close (confirmed against live tick data: a bar can overlap a
    zone by a single tick at one edge while entry sits at the midpoint,
    which a resting limit order there would never have filled at all). Set
    True to require price to actually trade AT OR THROUGH setup_entry
    itself before the trade triggers - the genuinely correct condition for
    what a resting limit order needs to fill. Independent of
    market_order_entry: combine both to get the strict, actually-correct
    market/limit distinction, or use market_order_entry alone to keep the
    original (loose, optimistic) zone-overlap trigger with a worse assumed
    price - not recommended once you know this, but left available for
    comparison."""
    is_ph = find_pivot_highs(df_1m["high"], mss_pivot_length).to_numpy()
    is_pl = find_pivot_lows(df_1m["low"], mss_pivot_length).to_numpy()

    asia_high, asia_low = session_levels(df_1m, asia_start_hour, asia_end_offset_hours, tz)
    london_high, london_low = session_levels(df_1m, london_start_hour, london_end_offset_hours, tz)

    ah, al = asia_high.to_numpy(), asia_low.to_numpy()
    lh, ll = london_high.to_numpy(), london_low.to_numpy()
    highs, lows, closes = df_1m["high"].to_numpy(), df_1m["low"].to_numpy(), df_1m["close"].to_numpy()
    ny_hours = df_1m.index.tz_convert(tz).hour.to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    asia_h_swept = asia_l_swept = london_h_swept = london_l_swept = False
    prev_ah = prev_al = prev_lh = prev_ll = np.nan

    pending_bull: list[_FVG] = []
    pending_bear: list[_FVG] = []

    setup_dir = None  # "long" / "short" / None
    setup_fvg: _FVG | None = None
    setup_sl = setup_entry = setup_tp = None
    setup_swept_session = None
    setup_liquidity_tp = None
    setup_mss_level = None
    entry_fvg: _FVG | None = None
    # staged, not entry_fvg itself: a fresh FVG's own top/bottom are derived
    # from the bar it forms on, so checking "is price back in the zone" on
    # that SAME bar is trivially always true (top literally equals that
    # bar's own low/high) - not a real retracement. Staging delays it by
    # exactly one bar so the retest check only ever sees past-bar levels.
    staged_entry_fvg = None
    staged_setup_entry = staged_setup_tp = staged_setup_liquidity_tp = None
    awaiting_confirm = awaiting_fresh_fvg = awaiting_retest = False
    setup_anchor_bar = setup_confirm_bar = None

    trade_active = False
    trade_is_long = False
    trade_sl = trade_tp = None

    last_swing_high = last_swing_low = np.nan

    trades: list[dict] = []
    skipped_no_target = 0
    skipped_by_filter = 0
    skipped_bad_geometry = 0
    skipped_no_mss_level = 0
    skipped_no_fresh_fvg = 0
    skipped_confirm_timeout = 0
    skipped_retest_timeout = 0
    # funnel diagnostics: how many sweeps happen at all, how many of those
    # occur while flat (a new setup could anchor), how many actually anchor
    # (a pending opposing FVG existed), and how many confirm the reversal -
    # `len(trades)` is the final "entered" count, the last funnel stage
    funnel_total_sweeps = 0
    funnel_sweeps_while_flat = 0
    funnel_anchored = 0
    funnel_confirmed = 0

    for i in range(n):
        high, low, close = highs[i], lows[i], closes[i]

        if staged_entry_fvg is not None:
            entry_fvg = staged_entry_fvg
            setup_entry, setup_tp, setup_liquidity_tp = staged_setup_entry, staged_setup_tp, staged_setup_liquidity_tp
            awaiting_retest = True
            setup_confirm_bar = i  # use_normal_fvg_retest path - the other path sets this at its own awaiting_retest transition (line ~410/433); this one never did, so max_bars_awaiting_retest crashed (None - int) whenever this path was used.
            staged_entry_fvg = None

        # LTF swing points, confirmed `mss_pivot_length` bars after they
        # form - mirrors structure.structure_bias()'s non-repainting timing
        confirm_i = i - mss_pivot_length
        if confirm_i >= 0:
            if is_ph[confirm_i]:
                last_swing_high = highs[confirm_i]
            if is_pl[confirm_i]:
                last_swing_low = lows[confirm_i]

        # session levels reset to "unswept" the instant a new instance closes
        if ah[i] != prev_ah:
            asia_h_swept = False
        if al[i] != prev_al:
            asia_l_swept = False
        if lh[i] != prev_lh:
            london_h_swept = False
        if ll[i] != prev_ll:
            london_l_swept = False
        prev_ah, prev_al, prev_lh, prev_ll = ah[i], al[i], lh[i], ll[i]

        asia_h_just_swept = (not asia_h_swept) and not np.isnan(ah[i]) and high > ah[i]
        asia_l_just_swept = (not asia_l_swept) and not np.isnan(al[i]) and low < al[i]
        london_h_just_swept = (not london_h_swept) and not np.isnan(lh[i]) and high > lh[i]
        london_l_just_swept = (not london_l_swept) and not np.isnan(ll[i]) and low < ll[i]
        if asia_h_just_swept:
            asia_h_swept = True
        if asia_l_just_swept:
            asia_l_swept = True
        if london_h_just_swept:
            london_h_swept = True
        if london_l_just_swept:
            london_l_swept = True

        low_swept_now = asia_l_just_swept or london_l_just_swept
        high_swept_now = asia_h_just_swept or london_h_just_swept
        swept_low_level = al[i] if asia_l_just_swept else ll[i]
        swept_high_level = ah[i] if asia_h_just_swept else lh[i]
        funnel_total_sweeps += int(low_swept_now) + int(high_swept_now)
        if not (awaiting_confirm or awaiting_fresh_fvg or awaiting_retest or trade_active):
            funnel_sweeps_while_flat += int(low_swept_now) + int(high_swept_now)

        # prune against the EXISTING pool first, using this bar's price
        # action - before this bar's own fresh FVG (if any) gets appended
        # below. Otherwise a fresh gap's top/bottom (derived from this same
        # bar) would trivially fail its own pristine/filled check on
        # creation (e.g. low > f.top is low > low, always false).
        if require_pristine_fvg:
            # stricter than the normal "filled" rule below - ineligible the
            # instant price touches into the zone at all, not just once
            # fully closed through it
            pending_bull = [f for f in pending_bull if low > f.top]
            pending_bear = [f for f in pending_bear if high < f.bottom]
        else:
            pending_bull = [f for f in pending_bull if close >= f.bottom]
            pending_bear = [f for f in pending_bear if close <= f.top]

        # standard 3-candle FVG check - appended AFTER pruning above, so a
        # fresh gap never faces a pruning check against its own formation bar
        if i >= 2:
            if highs[i - 2] < low:
                pending_bull.append(_FVG(top=low, bottom=highs[i - 2], bar_index=i))
            if lows[i - 2] > high:
                pending_bear.append(_FVG(top=lows[i - 2], bottom=high, bar_index=i))

        def nearest_untapped_high():
            candidates = [v for v, swept in ((ah[i], asia_h_swept), (lh[i], london_h_swept))
                          if not swept and not np.isnan(v)]
            return min(candidates) if candidates else np.nan

        def nearest_untapped_low():
            candidates = [v for v, swept in ((al[i], asia_l_swept), (ll[i], london_l_swept))
                          if not swept and not np.isnan(v)]
            return max(candidates) if candidates else np.nan

        def _round_tick(price):
            if tick_size is None:
                return price
            return round(round(price / tick_size) * tick_size, 10)

        def resolve_target(direction, fvg):
            if market_order_entry:
                # first price actually touched entering the zone - top for a
                # long (retracing down into it), bottom for a short
                # (retracing up into it) - see market_order_entry's docstring.
                raw_entry = fvg.top if direction == "long" else fvg.bottom
            else:
                raw_entry = (fvg.top + fvg.bottom) / 2
            entry_candidate = _round_tick(raw_entry)
            # sl_shrink_factor < 1.0 moves the stop closer to entry, proportional
            # to the original entry-to-swept-level distance (1.0 = unchanged,
            # exactly at the swept level; 0.5 = halfway between entry and it).
            original_risk = abs(entry_candidate - setup_sl)
            if direction == "long":
                effective_sl = entry_candidate - sl_shrink_factor * (entry_candidate - setup_sl)
            else:
                effective_sl = entry_candidate + sl_shrink_factor * (setup_sl - entry_candidate)
            liquidity_tp = nearest_untapped_high() if direction == "long" else nearest_untapped_low()
            if fixed_r_target is not None:
                effective_sl = _round_tick(effective_sl)
                # hold_tp_fixed=True: target stays at the ORIGINAL (unshrunk) risk
                # distance - isolates the effect of a closer stop alone, without
                # also pulling the target closer. R-multiples are still reported
                # against the actual (shrunk) risk taken, so a win here is worth
                # MORE than fixed_r_target R, not exactly fixed_r_target R.
                tp_risk = original_risk if hold_tp_fixed else abs(entry_candidate - effective_sl)
                tp = entry_candidate + fixed_r_target * tp_risk if direction == "long" else entry_candidate - fixed_r_target * tp_risk
                tp = _round_tick(tp)
            else:
                tp = liquidity_tp
            return entry_candidate, effective_sl, tp, liquidity_tp

        def geometry_ok(direction, entry, sl, tp):
            # Bug guard: setup_sl (the level resolve_target's shrink/target
            # math is anchored to) can go stale relative to entry_candidate -
            # most notably in the use_normal_fvg_retest path, where a fresh
            # FVG can form well after the original anchor and price may have
            # drifted back through setup_sl in the meantime without
            # triggering the sweep-based invalidation elsewhere (that only
            # fires on a NEW sweep event, not on price re-crossing a level
            # it already swept once). When that happens, sl/tp end up on the
            # wrong side of entry (even coinciding exactly in the worst
            # case) instead of straddling it properly. Reject anything that
            # doesn't have the sane long/short shape rather than trade it.
            if np.isnan(tp) or np.isnan(sl):
                return False
            return sl < entry < tp if direction == "long" else tp < entry < sl

        def select_fvg(pending_list):
            """Default: grab whichever FVG formed most recently (LIFO). With
            fvg_selection_window set: look back that many bars and pick the
            LARGEST (tallest) FVG among only those recent candidates,
            instead of always the last one in."""
            if fvg_selection_window is None:
                return pending_list.pop()
            cutoff = i - fvg_selection_window
            candidates = [f for f in pending_list if f.bar_index >= cutoff]
            if not candidates:
                return pending_list.pop()
            best = max(candidates, key=lambda f: f.top - f.bottom)
            pending_list.remove(best)
            return best

        # anchor a new setup (only when flat). With MSS confirmation on, a
        # setup needs a confirmed swing point to break - if none exists yet,
        # skip rather than anchor a setup that can never be validated (the
        # same "hangs forever waiting" trap the missing-target fix avoided).
        if low_swept_now and not awaiting_confirm and not awaiting_fresh_fvg and not awaiting_retest and not trade_active and pending_bear:
            if use_mss_confirmation and np.isnan(last_swing_high):
                skipped_no_mss_level += 1
            else:
                setup_fvg = select_fvg(pending_bear)
                setup_dir = "long"
                setup_sl = swept_low_level
                setup_swept_session = "asia" if asia_l_just_swept else "london"
                setup_mss_level = last_swing_high
                awaiting_confirm = True
                setup_anchor_bar = i
                funnel_anchored += 1

        if high_swept_now and not awaiting_confirm and not awaiting_fresh_fvg and not awaiting_retest and not trade_active and pending_bull:
            if use_mss_confirmation and np.isnan(last_swing_low):
                skipped_no_mss_level += 1
            else:
                setup_fvg = select_fvg(pending_bull)
                setup_dir = "short"
                setup_sl = swept_high_level
                setup_swept_session = "asia" if asia_h_just_swept else "london"
                setup_mss_level = last_swing_low
                awaiting_confirm = True
                setup_anchor_bar = i
                funnel_anchored += 1

        # a sweep on the opposite side before we confirm invalidates the thesis
        if awaiting_confirm or awaiting_fresh_fvg or awaiting_retest:
            if setup_dir == "long" and high_swept_now:
                if awaiting_fresh_fvg:
                    skipped_no_fresh_fvg += 1
                awaiting_confirm = awaiting_fresh_fvg = awaiting_retest = False
                setup_fvg = None
                entry_fvg = None
            elif setup_dir == "short" and low_swept_now:
                if awaiting_fresh_fvg:
                    skipped_no_fresh_fvg += 1
                awaiting_confirm = awaiting_fresh_fvg = awaiting_retest = False
                setup_fvg = None
                entry_fvg = None

        # a setup that's taken too long to confirm or retest expires outright
        # - same reset as an opposite-side invalidation, just timed out
        # instead of thesis-broken. Requires the fast, sharp version of the
        # pattern (a real V-shaped reversal) rather than accepting a setup
        # that's been drifting for hours as equally valid.
        if awaiting_confirm and max_bars_awaiting_confirm is not None and (i - setup_anchor_bar) > max_bars_awaiting_confirm:
            skipped_confirm_timeout += 1
            awaiting_confirm = awaiting_fresh_fvg = awaiting_retest = False
            setup_fvg = None
            entry_fvg = None
        elif awaiting_retest and max_bars_awaiting_retest is not None and (i - setup_confirm_bar) > max_bars_awaiting_retest:
            skipped_retest_timeout += 1
            awaiting_confirm = awaiting_fresh_fvg = awaiting_retest = False
            setup_fvg = None
            entry_fvg = None

        # reversal confirmation: close through the approach-leg FVG, and
        # (when enabled) also through the swing point snapshotted at anchor
        # time - both need to be true on the same bar to confirm. What
        # happens next depends on use_normal_fvg_retest:
        #   off (default, "inverse FVG"): the SAME approach-leg FVG becomes
        #     the retest zone - it was resistance/support in the sweep
        #     direction, now flipped.
        #   on ("normal FVG retest"): that FVG is discarded once it's done
        #     its job of confirming the reversal - we instead wait for a
        #     FRESH FVG to form in the new (post-reversal) direction, and
        #     retest THAT one, the more conventional "FVG left behind by
        #     the confirming displacement" entry.
        mss_long_ok = (not use_mss_confirmation) or (setup_mss_level is not None and close > setup_mss_level)
        mss_short_ok = (not use_mss_confirmation) or (setup_mss_level is not None and close < setup_mss_level)
        if awaiting_confirm and setup_fvg is not None:
            if setup_dir == "long" and close > setup_fvg.top and mss_long_ok:
                awaiting_confirm = False
                funnel_confirmed += 1
                if use_normal_fvg_retest:
                    awaiting_fresh_fvg = True
                    setup_fvg = None
                else:
                    entry_candidate, effective_sl, tp, liquidity_tp = resolve_target("long", setup_fvg)
                    if np.isnan(tp):
                        skipped_no_target += 1
                        setup_fvg = None
                    elif not geometry_ok("long", entry_candidate, effective_sl, tp):
                        skipped_bad_geometry += 1
                        setup_fvg = None
                    elif not _passes_filters(entry_candidate, effective_sl, tp, ny_hours[i],
                                              use_time_filter, trading_start_hour, trading_end_hour,
                                              use_rr_filter, avoid_rr_min, avoid_rr_max):
                        skipped_by_filter += 1
                        setup_fvg = None
                    else:
                        entry_fvg = setup_fvg
                        setup_sl = effective_sl
                        setup_entry, setup_tp, setup_liquidity_tp = entry_candidate, tp, liquidity_tp
                        awaiting_retest = True
                        setup_confirm_bar = i
                        setup_fvg = None
            elif setup_dir == "short" and close < setup_fvg.bottom and mss_short_ok:
                awaiting_confirm = False
                funnel_confirmed += 1
                if use_normal_fvg_retest:
                    awaiting_fresh_fvg = True
                    setup_fvg = None
                else:
                    entry_candidate, effective_sl, tp, liquidity_tp = resolve_target("short", setup_fvg)
                    if np.isnan(tp):
                        skipped_no_target += 1
                        setup_fvg = None
                    elif not geometry_ok("short", entry_candidate, effective_sl, tp):
                        skipped_bad_geometry += 1
                        setup_fvg = None
                    elif not _passes_filters(entry_candidate, effective_sl, tp, ny_hours[i],
                                              use_time_filter, trading_start_hour, trading_end_hour,
                                              use_rr_filter, avoid_rr_min, avoid_rr_max):
                        skipped_by_filter += 1
                        setup_fvg = None
                    else:
                        entry_fvg = setup_fvg
                        setup_sl = effective_sl
                        setup_entry, setup_tp, setup_liquidity_tp = entry_candidate, tp, liquidity_tp
                        awaiting_retest = True
                        setup_confirm_bar = i
                        setup_fvg = None

        # normal-FVG-retest mode only: search for the next fresh FVG in the
        # post-reversal direction (reusing this bar's own FVG check above).
        #
        # Bug fix: waiting for a fresh FVG has no time limit by default, and
        # during that wait price can drift back through the ORIGINAL swept
        # level (setup_sl) without tripping the session-sweep-based
        # invalidation above (that only fires on a NEW sweep event, not on
        # price generically re-crossing a level it already swept once this
        # session). resolve_target() computes the fresh FVG's stop/target
        # relative to setup_sl assuming it's still on the correct side of
        # entry - once price has drifted back past it, that assumption
        # breaks and can produce a degenerate stop==target (or worse,
        # inverted risk) trade. Re-validating the reclaim here, the same way
        # the original confirmation required it, closes that gap.
        if awaiting_fresh_fvg:
            reclaim_broken = (setup_dir == "long" and close < setup_sl) or (setup_dir == "short" and close > setup_sl)
            if reclaim_broken:
                skipped_no_fresh_fvg += 1
                awaiting_fresh_fvg = False
                setup_fvg = None

        if awaiting_fresh_fvg:
            fresh_fvg = None
            # the FVG-detection step above already appended this bar's fresh
            # FVG (if any) as the last element - and it always survives its
            # own creation bar's pruning (close sits inside [low, high],
            # which by construction of the gap is already past the bottom).
            if setup_dir == "long" and i >= 2 and highs[i - 2] < low:
                fresh_fvg = pending_bull.pop()
            elif setup_dir == "short" and i >= 2 and lows[i - 2] > high:
                fresh_fvg = pending_bear.pop()

            if fresh_fvg is not None:
                entry_candidate, effective_sl, tp, liquidity_tp = resolve_target(setup_dir, fresh_fvg)
                awaiting_fresh_fvg = False
                if np.isnan(tp):
                    skipped_no_target += 1
                elif not geometry_ok(setup_dir, entry_candidate, effective_sl, tp):
                    skipped_bad_geometry += 1
                elif not _passes_filters(entry_candidate, effective_sl, tp, ny_hours[i],
                                          use_time_filter, trading_start_hour, trading_end_hour,
                                          use_rr_filter, avoid_rr_min, avoid_rr_max):
                    skipped_by_filter += 1
                else:
                    # staged rather than set directly - see the comment by
                    # staged_entry_fvg's declaration.
                    staged_entry_fvg = fresh_fvg
                    setup_sl = effective_sl
                    staged_setup_entry, staged_setup_tp, staged_setup_liquidity_tp = entry_candidate, tp, liquidity_tp

        # entry: retracement back into the retest zone. Not elif'd against the
        # resolution check below - a single volatile bar can both trigger the
        # retracement AND reach SL/TP, exactly as the live Pine indicator
        # re-runs its whole script (entry check, then resolution check) once
        # per bar and would resolve a same-bar entry+exit on the spot.
        # Missing this let a same-bar SL touch get silently skipped whenever
        # that bar's range also reached TP, since resolution didn't start
        # checking until the NEXT bar - falsely turning some same-bar
        # whipsaws into wins.
        if not trade_active and awaiting_retest and entry_fvg is not None:
            if require_price_reaches_entry:
                # the actually-correct condition for a resting limit order:
                # it only fills if price trades AT OR THROUGH the specific
                # entry price, not merely "somewhere in the zone." The
                # zone-overlap check below can trigger on a bar that only
                # grazed the FAR side of the zone from setup_entry, without
                # price ever reaching the level the fill is assumed to
                # happen at - the entry trigger and the assumed fill price
                # were checking two different things.
                in_zone = (low <= setup_entry) if setup_dir == "long" else (high >= setup_entry)
            elif require_close_in_zone:
                in_zone = entry_fvg.bottom <= close <= entry_fvg.top
            else:
                in_zone = low <= entry_fvg.top and high >= entry_fvg.bottom
            if in_zone and setup_entry != setup_sl:
                trade_active = True
                trade_is_long = setup_dir == "long"
                trade_sl = setup_sl
                trade_tp = setup_tp
                trades.append({
                    "time": idx[i], "direction": setup_dir,
                    "entry_price": setup_entry, "stop_price": setup_sl, "target_price": setup_tp,
                    "liquidity_target_price": setup_liquidity_tp, "swept_session": setup_swept_session,
                    "fvg_top": entry_fvg.top, "fvg_bottom": entry_fvg.bottom,
                    "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
                })
                awaiting_retest = False
                entry_fvg = None
                setup_dir = None

        if trade_active:
            hit_sl = (low <= trade_sl) if trade_is_long else (high >= trade_sl)
            hit_tp = (high >= trade_tp) if trade_is_long else (low <= trade_tp)
            # SL takes priority if both are touched in the same bar - can't
            # know intrabar order from OHLC alone, so assume the worst case.
            if hit_sl:
                trades[-1].update(outcome="loss", exit_time=idx[i], exit_price=trade_sl)
                trade_active = False
            elif hit_tp:
                trades[-1].update(outcome="win", exit_time=idx[i], exit_price=trade_tp)
                trade_active = False

    result = pd.DataFrame(trades)
    diagnostics = {
        "skipped_no_target": skipped_no_target,
        "skipped_by_filter": skipped_by_filter,
        "skipped_no_mss_level": skipped_no_mss_level,
        "skipped_no_fresh_fvg": skipped_no_fresh_fvg,
        "skipped_confirm_timeout": skipped_confirm_timeout,
        "skipped_retest_timeout": skipped_retest_timeout,
        "funnel_total_sweeps": funnel_total_sweeps,
        "funnel_sweeps_while_flat": funnel_sweeps_while_flat,
        "funnel_anchored": funnel_anchored,
        "funnel_confirmed": funnel_confirmed,
        "funnel_entered": len(trades),
    }
    if result.empty:
        return result, diagnostics

    still_open = result["outcome"].isna()
    result.loc[still_open, "outcome"] = "open_at_data_end"
    result.loc[still_open, "exit_time"] = idx[-1]
    result.loc[still_open, "exit_price"] = closes[-1]

    risk = (result["entry_price"] - result["stop_price"]).abs()
    direction_sign = np.where(result["direction"] == "long", 1, -1)
    result["r_multiple"] = (result["exit_price"] - result["entry_price"]) * direction_sign / risk

    return result, diagnostics
