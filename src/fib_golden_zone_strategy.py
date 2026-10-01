# Sweep -> displacement -> fib "golden zone" retracement -> entry. Sibling
# to htf_sweep_strategy.py (same HTF-bias gate, same liquidity-sweep
# detection, same ATR-relative displacement-leg confirmation) but does NOT
# enter on the displacement leg itself. Instead, once a leg is confirmed, it
# waits for price to retrace back INTO the fib golden zone (61.8%-78.6% by
# default) of that leg - the entry sits close to the original swept
# extreme, so the stop (placed just beyond that extreme) stays tight
# relative to a target set as a large R-multiple, which is the whole point
# of this variant: tight stop, oversized target.
#
# Three entry_mode options for how the zone actually gets traded:
#   "rb_retest"  - (the default - RB is the actual entry trigger) waits for
#                  a CLOSED rejection candle (RB) off the zone's near edge,
#                  same shape as poi.check_rb - then, instead of entering on
#                  that candle, waits for a LATER bar to trade back into the
#                  RB's own zone and fills at rb_near (the block's edge
#                  closest to where price already reversed to - "the
#                  beginning of the RB" from price's perspective coming back
#                  down/up into it) with a stop rb_stop_buffer_points beyond
#                  rb_far (the RB candle's own wick extreme) - a fixed-point
#                  stop, not ATR-scaled, per spec. If price never comes back
#                  (rb_retest_window_bars elapses) or trades back past
#                  rb_far before ever touching rb_near (the rejection
#                  itself failing), the candidate is dropped with no trade.
#   "rb_confirm" - waits for the same closed rejection candle but enters
#                  immediately at that candle's own close instead of waiting
#                  for a retest - what you'd get from a market-order entry
#                  the moment the RB closes, no retest patience required.
#                  Kept for comparison against "rb_retest".
#   "limit"      - ignores RB rejection behavior entirely: fills the instant
#                  price first trades through fib_entry_level (a single
#                  price inside the zone, default 0.705, the conventional
#                  ICT "OTE" level), no closed-candle confirmation at all.
#                  Kept for comparison; was the default in an earlier
#                  version of this file before RB was made the actual
#                  trigger.
#
# Definitions:
#   HTF bias        - same as htf_sweep_strategy.htf_bias_agreement: 5m AND
#                      30m structure_bias() must agree and be non-zero.
#   Liquidity sweep - same two pools as htf_sweep_strategy: session
#                      (Asia/London high/low) and swing (unswept confirmed
#                      1-minute pivots, capped at MAX_LIQUIDITY_POOL/side).
#   Displacement leg - identical ATR-relative leg confirmation to
#                      htf_sweep_strategy: the move away from the worst
#                      excursion reached while armed must reach
#                      >= leg_atr_mult x 1-minute ATR(14), with the close
#                      having reclaimed the swept level.
#   Golden zone     - once the leg first confirms, the running extreme IN
#                      THE LEG'S DIRECTION (leg_peak for a long, leg_trough
#                      for a short) keeps updating every bar the leg keeps
#                      extending. The zone is the [fib_zone_start,
#                      fib_zone_end] retracement band of [leg_start
#                      (=armed_extreme, the swept extreme), leg_peak/trough]
#                      - recomputed every bar off the current leg_peak/
#                      trough, so a leg that keeps extending keeps deferring
#                      the retracement watch rather than locking in a stale
#                      zone.
#   RB confirmation - (entry_mode="rb_retest"/"rb_confirm") a bar whose wick
#                      trades into the zone (through its near edge,
#                      fib_zone_start) but whose CLOSE rejects back out
#                      beyond that same edge, with the rejecting wick >=
#                      rb_wick_ratio x that bar's own body - the same
#                      decisive-rejection shape as poi.check_rb, just keyed
#                      to the fib zone's edge instead of a swing level. Its
#                      own zone [rb_near, rb_far] = [that candle's own body
#                      edge, that candle's own wick extreme] - identical
#                      shape to poi.check_rb's bull_top/bull_bottom.
#   Clean zone      - at the moment of fill, checks the SAME active swing
#                      pool this file already tracks for sweep detection:
#                      True if no other still-unswept 1-minute swing high or
#                      low is resting strictly inside [zone_bottom,
#                      zone_top]. Always recorded on the trade; only
#                      enforced as a gate if require_clean_zone=True.
#   Invalidation    - if price trades back beyond leg_start (i.e. re-sweeps
#                      past the ORIGINAL swept extreme) before a fill, or
#                      the bias flips, or zone_window_bars elapses since the
#                      leg confirmed, the candidate is dropped with no trade.
#   Stop/target     - stop sits just beyond leg_start (the original swept
#                      extreme) plus a small ATR buffer - NOT beyond the
#                      zone's far edge, so depth of retracement doesn't
#                      widen the stop. Target is target_r_multiple x that
#                      risk - deliberately large (the reason this variant
#                      exists at all): a deep retracement entry means the
#                      stop-to-entry distance is only roughly
#                      (1 - fib_zone_start) to (1 - fib_zone_end) of the
#                      leg's magnitude, so a real displacement leg still
#                      produces a comparatively tight stop.
#
# Same-bar sweep+leg-confirm+zone-entry is possible in principle but
# vanishingly rare in practice (a single bar would have to sweep, displace
# a full ATR multiple, retrace into the golden zone, AND reject - four
# distinct excursions); no special-casing needed, the FSM handles it
# correctly either way since each check runs unconditionally in sequence.

import numpy as np
import pandas as pd

from .crt_strategy import wilder_atr
from .data import session_levels
from .htf_sweep_strategy import MAX_LIQUIDITY_POOL, htf_bias_agreement
from .structure import find_pivot_highs, find_pivot_lows


def run_fib_golden_zone_strategy(
    df_1m: pd.DataFrame,
    bias_5m_length: int = 3,
    bias_30m_length: int = 3,
    sweep_pivot_length: int = 5,
    reaction_window_bars: int = 15,
    leg_atr_mult: float = 1.5,
    ltf_atr_len: int = 14,
    stop_buffer_atr_mult: float = 0.1,
    fib_zone_start: float = 0.618,
    fib_zone_end: float = 0.786,
    entry_mode: str = "rb_retest",  # "rb_retest" (RB forms, then wait for a retest fill), "rb_confirm" (enter at the RB's own close), or "limit" (bare fib-level touch, no RB required)
    fib_entry_level: float = 0.705,  # used when entry_mode="limit" - must sit between fib_zone_start and fib_zone_end
    rb_wick_ratio: float = 1.0,  # used when entry_mode="rb_retest" or "rb_confirm"
    rb_retest_window_bars: int = 30,  # used when entry_mode="rb_retest" - bars allowed for price to come back and retest the RB zone
    rb_stop_buffer_points: float = 5.0,  # used when entry_mode="rb_retest" - fixed points beyond the RB's own wick extreme, not ATR-scaled
    require_clean_zone: bool = False,
    zone_window_bars: int = 60,
    target_r_multiple: float = 8.0,
    max_trade_bars: int | None = None,
    use_htf_bias: bool = True,
    use_session_levels: bool = True,
    use_swing_pool: bool = True,
    asia_start_hour: int = 18,
    asia_end_offset_hours: int = 24,
    london_start_hour: int = 2,
    london_end_offset_hours: int = 5,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Runs the full FSM (FLAT -> ARMED -> ZONE_WATCH -> IN_TRADE) bar by bar
    on 1-minute data. Returns every entry with its resolved outcome,
    r_multiple, liquidity_kind, and the leg/zone geometry it was measured
    against (leg_start, leg_extreme, zone_top, zone_bottom, clean_zone) so a
    caller can sanity-check the stop distance and target distance directly."""
    htf_bias = htf_bias_agreement(df_1m, bias_5m_length, bias_30m_length)

    is_ph = find_pivot_highs(df_1m["high"], sweep_pivot_length).to_numpy()
    is_pl = find_pivot_lows(df_1m["low"], sweep_pivot_length).to_numpy()
    atr_1m = wilder_atr(df_1m, ltf_atr_len).to_numpy()

    asia_high, asia_low = session_levels(df_1m, asia_start_hour, asia_end_offset_hours, tz)
    london_high, london_low = session_levels(df_1m, london_start_hour, london_end_offset_hours, tz)
    ah, al = asia_high.to_numpy(), asia_low.to_numpy()
    lh, ll = london_high.to_numpy(), london_low.to_numpy()

    opens = df_1m["open"].to_numpy()
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    active_highs: list[float] = []
    active_lows: list[float] = []

    asia_h_swept = asia_l_swept = london_h_swept = london_l_swept = False
    prev_ah = prev_al = prev_lh = prev_ll = np.nan

    armed_dir = 0  # 0 none, 1 long candidate, -1 short candidate
    armed_swept_level = np.nan
    armed_extreme = np.nan  # = leg_start once frozen: worst excursion while armed
    armed_bar = None
    armed_kind = None
    leg_confirmed = False
    leg_extreme = np.nan   # running peak (long) / trough (short) since armed_bar
    zone_confirm_bar = None
    rb_pending = False     # RB has formed, now watching for a retest fill (entry_mode="rb_retest" only)
    rb_near = rb_far = np.nan
    rb_confirm_bar = None
    rb_clean = None

    active_trade = False
    entry_bar = stop_price = target_price = None
    is_long = None

    trades: list[dict] = []

    def _open(dir_, entry_price, stop, swept_level, kind, ztop, zbottom, clean_, rb_n=np.nan, rb_f=np.nan):
        """Opens a trade at the current bar i - a closure so it can set the
        loop-scoped trade-in-progress state (active_trade etc.) that the
        resolution check further down this same iteration depends on,
        letting entry and resolution still collide on the same bar exactly
        like htf_sweep_strategy.py's equivalent same-bar case."""
        nonlocal is_long, stop_price, target_price, entry_bar, active_trade
        risk = abs(entry_price - stop)
        if risk <= 0:
            return
        is_long = dir_ == 1
        target_price = entry_price + risk * target_r_multiple if is_long else entry_price - risk * target_r_multiple
        stop_price = stop
        entry_bar, active_trade = i, True
        trades.append({
            "time": idx[i], "direction": "long" if is_long else "short",
            "entry_price": entry_price, "stop_price": stop_price, "target_price": target_price,
            "swept_level": swept_level, "leg_start": leg_start, "leg_extreme": leg_extreme,
            "zone_top": ztop, "zone_bottom": zbottom, "liquidity_kind": kind,
            "clean_zone": clean_, "rb_near": rb_n, "rb_far": rb_f,
            "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
        })

    for i in range(n):
        high, low, close, o = highs[i], lows[i], closes[i], opens[i]
        bdir = htf_bias[i]
        long_ok = (not use_htf_bias) or bdir == 1
        short_ok = (not use_htf_bias) or bdir == -1

        confirm_i = i - sweep_pivot_length
        if confirm_i >= 0:
            if is_ph[confirm_i]:
                active_highs.append(highs[confirm_i])
                if len(active_highs) > MAX_LIQUIDITY_POOL:
                    active_highs.pop(0)
            if is_pl[confirm_i]:
                active_lows.append(lows[confirm_i])
                if len(active_lows) > MAX_LIQUIDITY_POOL:
                    active_lows.pop(0)

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

        swept_lows_now = [p for p in active_lows if low < p]
        if swept_lows_now:
            active_lows = [p for p in active_lows if low >= p]
        swept_highs_now = [p for p in active_highs if high > p]
        if swept_highs_now:
            active_highs = [p for p in active_highs if high <= p]

        if not active_trade:
            if armed_dir != 0:
                # Worst excursion (armed_extreme) only tracks BEFORE the leg
                # confirms - once confirmed it freezes as leg_start, the
                # fixed anchor the stop and the resweep-invalidation check
                # are both measured against. leg_extreme (the leg-direction
                # running peak/trough) keeps tracking the whole time, pre-
                # and post-confirmation, so it's already warmed up the
                # moment the leg first confirms and keeps deferring the
                # retracement watch for as long as the leg keeps extending.
                if armed_dir == 1:
                    if not leg_confirmed:
                        armed_extreme = min(armed_extreme, low)
                    leg_extreme = high if np.isnan(leg_extreme) else max(leg_extreme, high)
                else:
                    if not leg_confirmed:
                        armed_extreme = max(armed_extreme, high)
                    leg_extreme = low if np.isnan(leg_extreme) else min(leg_extreme, low)

                still_ok = long_ok if armed_dir == 1 else short_ok
                stale = (not leg_confirmed and (i - armed_bar) > reaction_window_bars) or \
                        (leg_confirmed and not rb_pending and (i - zone_confirm_bar) > zone_window_bars) or \
                        (rb_pending and (i - rb_confirm_bar) > rb_retest_window_bars)
                # Re-sweep past the ORIGINAL swept extreme invalidates a
                # zone-phase (or retest-phase) candidate outright - this is
                # a full break of the level the whole setup (and its stop,
                # for "limit"/"rb_confirm" modes) is anchored to, not just a
                # deep-but-valid retracement into the zone.
                resweep = leg_confirmed and (
                    (armed_dir == 1 and low < armed_extreme) or (armed_dir == -1 and high > armed_extreme)
                )
                if not still_ok or stale or resweep:
                    armed_dir, leg_confirmed, rb_pending = 0, False, False

            if armed_dir == 0:
                low_candidates, high_candidates = [], []
                if use_swing_pool:
                    if swept_lows_now:
                        low_candidates.append(("swing", max(swept_lows_now)))
                    if swept_highs_now:
                        high_candidates.append(("swing", min(swept_highs_now)))
                if use_session_levels:
                    if asia_l_just_swept:
                        low_candidates.append(("session", al[i]))
                    if london_l_just_swept:
                        low_candidates.append(("session", ll[i]))
                    if asia_h_just_swept:
                        high_candidates.append(("session", ah[i]))
                    if london_h_just_swept:
                        high_candidates.append(("session", lh[i]))

                if long_ok and low_candidates:
                    armed_kind, armed_swept_level = max(low_candidates, key=lambda kv: kv[1])
                    armed_dir, armed_extreme, armed_bar = 1, low, i
                    leg_extreme = high
                elif short_ok and high_candidates:
                    armed_kind, armed_swept_level = min(high_candidates, key=lambda kv: kv[1])
                    armed_dir, armed_extreme, armed_bar = -1, high, i
                    leg_extreme = low

            if armed_dir != 0 and not leg_confirmed:
                atr_now = atr_1m[i]
                if not np.isnan(atr_now) and atr_now > 0:
                    reclaimed = (close > armed_swept_level) if armed_dir == 1 else (close < armed_swept_level)
                    leg_size = (close - armed_extreme) if armed_dir == 1 else (armed_extreme - close)
                    if reclaimed and leg_size >= atr_now * leg_atr_mult:
                        leg_confirmed, zone_confirm_bar = True, i

            if armed_dir != 0 and leg_confirmed and not rb_pending:
                leg_start = armed_extreme
                span = abs(leg_extreme - leg_start)
                if span > 0:
                    if armed_dir == 1:
                        zone_top = leg_extreme - fib_zone_start * span
                        zone_bottom = leg_extreme - fib_zone_end * span
                    else:
                        zone_bottom = leg_extreme + fib_zone_start * span
                        zone_top = leg_extreme + fib_zone_end * span

                    # Clean-zone check: any OTHER still-unswept swing high/low
                    # (from the same active pool sweep-detection already
                    # maintains, with the trigger level itself already
                    # excluded - it left the pool the bar it swept) resting
                    # strictly inside this zone. Recorded regardless of mode;
                    # only enforced as a gate if require_clean_zone=True.
                    clean = not any(zone_bottom < p < zone_top for p in active_highs) and \
                            not any(zone_bottom < p < zone_top for p in active_lows)

                    if entry_mode == "limit":
                        limit_price = leg_extreme - fib_entry_level * span if armed_dir == 1 \
                            else leg_extreme + fib_entry_level * span
                        filled = (low <= limit_price) if armed_dir == 1 else (high >= limit_price)
                        if filled and (clean or not require_clean_zone):
                            atr_now = atr_1m[i]
                            buf = 0.0 if np.isnan(atr_now) else atr_now * stop_buffer_atr_mult
                            stop = (leg_start - buf) if armed_dir == 1 else (leg_start + buf)
                            _open(armed_dir, limit_price, stop, armed_swept_level, armed_kind, zone_top, zone_bottom, clean)
                        if filled:
                            armed_dir, leg_confirmed = 0, False
                    else:
                        # "rb_retest" or "rb_confirm" - both need an actual
                        # closed rejection candle off the zone's near edge
                        # first; rb_near/rb_far are that candle's own body
                        # edge / wick extreme, same shape as poi.check_rb.
                        body_bottom, body_top = min(o, close), max(o, close)
                        body = abs(close - o)
                        if armed_dir == 1:
                            lower_wick = body_bottom - low
                            wick_ok = (body == 0 and lower_wick > 0) or (body != 0 and lower_wick >= rb_wick_ratio * body)
                            rb_now = low <= zone_top and close > zone_top and wick_ok
                            rb_near_cand, rb_far_cand = body_bottom, low
                        else:
                            upper_wick = high - body_top
                            wick_ok = (body == 0 and upper_wick > 0) or (body != 0 and upper_wick >= rb_wick_ratio * body)
                            rb_now = high >= zone_bottom and close < zone_bottom and wick_ok
                            rb_near_cand, rb_far_cand = body_top, high

                        if rb_now and entry_mode == "rb_confirm":
                            if clean or not require_clean_zone:
                                atr_now = atr_1m[i]
                                buf = 0.0 if np.isnan(atr_now) else atr_now * stop_buffer_atr_mult
                                stop = (leg_start - buf) if armed_dir == 1 else (leg_start + buf)
                                _open(armed_dir, close, stop, armed_swept_level, armed_kind, zone_top, zone_bottom, clean)
                            armed_dir, leg_confirmed = 0, False
                        elif rb_now:  # entry_mode == "rb_retest": RB confirmed, now watch for a retest
                            if clean or not require_clean_zone:
                                rb_pending = True
                                rb_near, rb_far, rb_confirm_bar, rb_clean = rb_near_cand, rb_far_cand, i, clean
                            else:
                                armed_dir, leg_confirmed = 0, False

            # entry_mode="rb_retest" only, and never on the RB's own confirm
            # bar (rb_near/rb_far are that bar's own wick, so a same-bar
            # check would trivially "touch" - a retest needs a LATER bar).
            if armed_dir != 0 and rb_pending and i > rb_confirm_bar:
                rb_failed = (armed_dir == 1 and low < rb_far) or (armed_dir == -1 and high > rb_far)
                touched = (low <= rb_near) if armed_dir == 1 else (high >= rb_near)
                if touched:
                    stop = (rb_far - rb_stop_buffer_points) if armed_dir == 1 else (rb_far + rb_stop_buffer_points)
                    _open(armed_dir, rb_near, stop, armed_swept_level, armed_kind, zone_top, zone_bottom, rb_clean,
                          rb_n=rb_near, rb_f=rb_far)
                    armed_dir, leg_confirmed, rb_pending = 0, False, False
                elif rb_failed:
                    armed_dir, leg_confirmed, rb_pending = 0, False, False

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
