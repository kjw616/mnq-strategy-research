"""Contract sizing for the execution bot.

Fixed-dollar-risk sizing has a known failure mode on this strategy: the
tightest-stop trades (the tail of the slShrinkFactor design - see
Keens_Indicator_v2.1.pine) can imply hundreds of contracts for a modest
dollar risk target. `max_contracts` clamps that tail down to a sane size
instead of skipping the trade - MAE analysis on those trades showed little
real drawdown before winning, so they're worth taking small rather than not
taking at all.

The opposite tail (a stop wide enough that even 1 contract's risk exceeds
`risk_usd`) is handled with its own, separate ceiling: `max_risk_per_trade_usd`.
If 1 contract's risk fits under that ceiling, the trade is still taken at 1
contract (accepting more risk than `risk_usd` on that trade, but a bounded
amount) rather than skipped outright - otherwise a small `risk_usd` value
(e.g. sized for a small account) would silently skip every trade whose stop
happens to be a bit wider than usual, and this strategy's edge comes from
taking every signal, not just the tightest-stop ones. Only stops wide enough
to blow through `max_risk_per_trade_usd` too are actually skipped.
"""

import math


def compute_contracts(
    entry: float,
    stop: float,
    tick_size: float,
    tick_value: float,
    risk_usd: float,
    max_contracts: int,
    max_risk_per_trade_usd: float | None = None,
) -> int:
    """Returns the number of contracts to trade, clamped to `max_contracts`.

    Returns 0 if the stop distance is degenerate (entry == stop, or either
    tick parameter is non-positive), or if even a single contract's risk
    exceeds both `risk_usd` and `max_risk_per_trade_usd` (when the latter is
    given - if it's None, any stop wider than `risk_usd` allows is skipped,
    the original conservative default).
    """
    if tick_size <= 0 or tick_value <= 0 or max_contracts <= 0:
        return 0

    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 0

    ticks = stop_distance / tick_size
    risk_per_contract = ticks * tick_value
    if risk_per_contract <= 0:
        return 0

    raw_contracts = math.floor(risk_usd / risk_per_contract)
    if raw_contracts < 1:
        if max_risk_per_trade_usd is not None and risk_per_contract <= max_risk_per_trade_usd:
            return 1
        return 0

    return min(raw_contracts, max_contracts)
