# New MNQ indicator - separate from Keen's Indicator (liquidity-sweep + FVG,
# see liquidity_fvg_strategy.py). Entries fire on the 1-minute chart, but the
# edge is meant to come from higher-timeframe directional bias: only take a
# setup when BOTH the 5-minute and 30-minute structure_bias() agree, then
# look for a liquidity sweep against that bias followed by a real
# displacement LEG reclaiming it. No PD arrays (FVG/RB) - tried, tested,
# made every version of this strategy worse (see conversation), so this
# version drops them entirely and focuses purely on liquidity + displacement.
#
# Definitions:
#   HTF bias      - structure_bias() on 5m AND 30m must both read the same
#                    non-zero direction. Any disagreement = no trade.
#   Liquidity     - two kinds of resting liquidity, either can be swept:
#                      "session" - Asia/London session high/low, same
#                        definition as Keen's Indicator (data.session_levels).
#                      "swing"   - ANY still-unswept CONFIRMED 1-minute swing
#                        point (structure.find_pivot_highs/lows), not just
#                        the most recent one - a pool of old highs/lows that
#                        stays live as a draw until price actually trades
#                        through it, mirroring liquidity.py's active-pool
#                        pattern. Capped at MAX_LIQUIDITY_POOL per side
#                        (oldest dropped first) so it can't grow unbounded
#                        over 2 years of 1-minute pivots.
#                    A sweep = price trades beyond ANY currently-active
#                    level (session or swing pool) against the current HTF
#                    bias. When more than one level gets swept on the same
#                    bar, the NEAREST one (closest to price beforehand) is
#                    used as the reference - the most recently "resting",
#                    most relevant liquidity of the group.
#   Displacement leg - not a single-candle heuristic: the reversal away from
#                    the sweep must build into a genuine LEG, measured the
#                    same way swing.py's zigzag_legs() confirms a leg - the
#                    move FROM the worst excursion reached while armed must
#                    reach >= leg_atr_mult x 1-minute ATR(14) before it
#                    counts, however many bars that takes. Close must also
#                    have reclaimed the swept level itself.
#   Stop/target   - stop sits just beyond the worst excursion reached while
#                    armed, plus a small ATR buffer. Target is a plain
#                    R-multiple off that structure-derived risk.
#
# Entry is immediate, at the bar the leg threshold is first crossed - no
# retest/retrace into a zone required (tried and dropped earlier: cut trade
# count ~90% for no improvement in win rate, avg R, or profit factor).
#
# Same-bar sweep+confirm: a candidate can arm AND its displacement leg can
# confirm on the very same bar (a single sharp reversal candle can both
# sweep and travel leg_atr_mult x ATR before its own close) - the sweep-
# detection and leg-confirmation checks are sequenced, not gated to
# "only after the sweep bar", specifically to allow this.
#
# Same-bar entry+resolution: the entry check and the stop/target resolution
# check are NOT mutually exclusive (no elif between them) - a single
# confirming bar can already reach the target within the same bar it
# triggers entry on. Stop can't fire on the entry bar itself by construction
# here (armed_extreme already includes that bar's own low/high, so the stop
# always sits beyond it) - but target can, so this still matters. If a later
# bar ever touches both stop and target, stop wins (worst-case assumption,
# since OHLC alone can't reveal true intrabar order).

import numpy as np
import pandas as pd

from .crt_strategy import wilder_atr
from .data import align_htf_to_ltf, resample_bars, session_levels
from .structure import find_pivot_highs, find_pivot_lows, structure_bias
from .trend import ema_stack_alignment

MAX_LIQUIDITY_POOL = 50


def htf_bias_agreement(df_1m: pd.DataFrame, bias_5m_length: int, bias_30m_length: int) -> np.ndarray:
    """1 = bullish, -1 = bearish, 0 = no trade, aligned onto the 1-minute
    timeline non-repainting via align_htf_to_ltf. Requires 5m and 30m
    structure_bias() to be non-zero AND agree - not just "30m sets direction,
    5m can't oppose" - the stricter, more clearly-stated of the two options."""
    bias_5m = structure_bias(resample_bars(df_1m, "5min"), bias_5m_length)[["bias"]]
    bias_30m = structure_bias(resample_bars(df_1m, "30min"), bias_30m_length)[["bias"]]
    b5 = align_htf_to_ltf(df_1m.index, bias_5m, "5min")["bias"].fillna(0).to_numpy()
    b30 = align_htf_to_ltf(df_1m.index, bias_30m, "30min")["bias"].fillna(0).to_numpy()
    return np.where((b5 == b30) & (b5 != 0), b5, 0).astype(int)


def run_htf_sweep_strategy(
    df_1m: pd.DataFrame,
    bias_5m_length: int = 3,
    bias_30m_length: int = 3,
    sweep_pivot_length: int = 5,
    reaction_window_bars: int = 15,
    confirmation_mode: str = "leg",  # "leg" (multi-bar ATR-relative move) or "candle" (single-candle heuristic)
    leg_atr_mult: float = 1.5,
    candle_atr_mult: float = 1.5,
    candle_min_body_ratio: float = 0.6,
    ltf_atr_len: int = 14,
    stop_buffer_atr_mult: float = 0.1,
    stop_shrink_factor: float = 1.0,
    target_r_multiple: float = 1.0,
    max_trade_bars: int | None = None,
    use_htf_bias: bool = True,
    use_session_levels: bool = True,
    use_swing_pool: bool = True,
    use_ema_stack_filter: bool = False,
    ema_stack_rule: str = "5min",
    ema_stack_fast: int = 9,
    ema_stack_mid: int = 21,
    ema_stack_slow: int = 50,
    asia_start_hour: int = 18,
    asia_end_offset_hours: int = 24,
    london_start_hour: int = 2,
    london_end_offset_hours: int = 5,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Runs the full FSM (FLAT -> ARMED -> IN_TRADE) bar by bar on 1-minute
    data and returns every entry with its resolved outcome, r_multiple
    (computed against the initial stop's risk), and `liquidity_kind`
    ("session" or "swing") recording which type of level was swept. One
    trade tracked at a time - no new candidate arms while a trade is active.

    use_ema_stack_filter (default off, so existing callers are unaffected):
    an additional gate on top of use_htf_bias, requiring trend.
    ema_stack_alignment on `ema_stack_rule` (default 5min) to also agree
    with the candidate's direction - see that function's docstring for why
    it's a state-based regime filter rather than a cross-event trigger.
    Applied to both arming a new candidate and staying armed, the same way
    use_htf_bias already is."""
    htf_bias = htf_bias_agreement(df_1m, bias_5m_length, bias_30m_length)

    if use_ema_stack_filter:
        stack_htf = ema_stack_alignment(df_1m, ema_stack_rule, ema_stack_fast, ema_stack_mid, ema_stack_slow)[["alignment"]]
        ema_stack = align_htf_to_ltf(df_1m.index, stack_htf, ema_stack_rule)["alignment"].fillna(0).to_numpy()
    else:
        ema_stack = None

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

    active_highs: list[float] = []  # unswept confirmed 1m swing highs - the "old highs" pool
    active_lows: list[float] = []   # unswept confirmed 1m swing lows - the "old lows" pool

    asia_h_swept = asia_l_swept = london_h_swept = london_l_swept = False
    prev_ah = prev_al = prev_lh = prev_ll = np.nan

    armed_dir = 0  # 0 none, 1 long candidate, -1 short candidate
    armed_swept_level = np.nan
    armed_extreme = np.nan  # worst excursion while armed (running min low / max high)
    armed_bar = None
    armed_kind = None

    active_trade = False
    entry_bar = stop_price = target_price = None
    is_long = None

    trades: list[dict] = []

    for i in range(n):
        high, low, close = highs[i], lows[i], closes[i]
        bdir = htf_bias[i]
        # use_htf_bias=False: both directions always permitted (pure
        # liquidity-sweep mean reversion, no directional gate at all).
        long_ok = (not use_htf_bias) or bdir == 1
        short_ok = (not use_htf_bias) or bdir == -1
        if use_ema_stack_filter:
            sdir = ema_stack[i]
            long_ok = long_ok and sdir == 1
            short_ok = short_ok and sdir == -1

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

        # Swing-pool sweeps are pruned every bar regardless of trade/armed
        # state - real liquidity gets consumed by price action whether or
        # not we happen to be watching for a new candidate at that moment.
        # Session-level sweeps (above) are already tracked unconditionally
        # the same way, via the swept flags.
        swept_lows_now = [p for p in active_lows if low < p]
        if swept_lows_now:
            active_lows = [p for p in active_lows if low >= p]
        swept_highs_now = [p for p in active_highs if high > p]
        if swept_highs_now:
            active_highs = [p for p in active_highs if high <= p]

        if not active_trade:
            if armed_dir != 0:
                armed_extreme = min(armed_extreme, low) if armed_dir == 1 else max(armed_extreme, high)
                still_ok = long_ok if armed_dir == 1 else short_ok
                if not still_ok or (i - armed_bar) > reaction_window_bars:
                    armed_dir = 0  # bias flipped or the candidate went stale

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
                elif short_ok and high_candidates:
                    armed_kind, armed_swept_level = min(high_candidates, key=lambda kv: kv[1])
                    armed_dir, armed_extreme, armed_bar = -1, high, i

            # Displacement-leg/entry check - deliberately not gated to "only
            # on bars after the sweep bar", so a single candle that both
            # sweeps AND travels a full leg's worth in one move counts too.
            if armed_dir != 0:
                atr_now = atr_1m[i]
                if not np.isnan(atr_now) and atr_now > 0:
                    reclaimed = (close > armed_swept_level) if armed_dir == 1 else (close < armed_swept_level)
                    if confirmation_mode == "candle":
                        rng = high - low
                        body = abs(close - opens[i])
                        confirmed = (
                            reclaimed and rng > 0
                            and rng >= atr_now * candle_atr_mult
                            and (body / rng) >= candle_min_body_ratio
                        )
                    else:
                        leg_size = (close - armed_extreme) if armed_dir == 1 else (armed_extreme - close)
                        confirmed = reclaimed and leg_size >= atr_now * leg_atr_mult

                    if confirmed:
                        buf = atr_now * stop_buffer_atr_mult
                        entry_price = close
                        raw_stop = (armed_extreme - buf) if armed_dir == 1 else (armed_extreme + buf)
                        # stop_shrink_factor < 1.0 pulls the stop closer to
                        # entry, proportional to the entry-to-raw-stop
                        # distance (1.0 = unchanged; 0.5 = halfway) - the
                        # same asymmetric-risk mechanic that makes Keen's
                        # Indicator's shrink=0.5/rr=2.0 config work well.
                        raw_risk = abs(entry_price - raw_stop)
                        if armed_dir == 1:
                            stop_price = entry_price - stop_shrink_factor * (entry_price - raw_stop)
                        else:
                            stop_price = entry_price + stop_shrink_factor * (raw_stop - entry_price)
                        risk = abs(entry_price - stop_price)  # actual risk taken (shrunk) - used for r_multiple
                        if risk > 0:
                            is_long = armed_dir == 1
                            # target distance is fixed to the ORIGINAL (unshrunk)
                            # risk, not the shrunk one - shrinking only tightens
                            # the stop, it doesn't pull the target in too. A win
                            # is then worth MORE than target_r_multiple R against
                            # the actual (shrunk) risk taken - the real asymmetric
                            # mechanic behind Keen's Indicator's shrink config,
                            # which my first pass here got wrong (scaled both
                            # stop and target together, so the R:R ratio never
                            # actually changed and shrinking only ever hurt).
                            target_price = entry_price + raw_risk * target_r_multiple if is_long \
                                else entry_price - raw_risk * target_r_multiple
                            entry_bar, active_trade = i, True
                            trades.append({
                                "time": idx[i], "direction": "long" if is_long else "short",
                                "entry_price": entry_price, "stop_price": stop_price, "target_price": target_price,
                                "swept_level": armed_swept_level, "armed_extreme": armed_extreme,
                                "liquidity_kind": armed_kind,
                                "bar_index": i, "outcome": None, "exit_time": pd.NaT, "exit_price": np.nan,
                            })
                        armed_dir = 0

        # Resolution - runs unconditionally, not elif'd against the entry
        # check above, so it can fire on the very bar entry just happened on.
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
    return result
