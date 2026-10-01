"""One-trade-at-a-time state machine, mirroring Keens_Indicator_v2.1.pine's
own model: it never anchors a new setup while a trade is active, so the bot
should never place a new order while one is pending or open either - both
to stay faithful to what was backtested, and as a hard guard against acting
on a duplicate or late-arriving webhook.

FLAT -> PENDING_ENTRY: an "armed" signal was placed as a resting limit order.
PENDING_ENTRY -> FLAT: cancelled (an "invalidated" signal arrived before fill).
PENDING_ENTRY -> POSITION_OPEN: the stream confirmed a real fill.
POSITION_OPEN -> FLAT: the stream confirmed the bracket (SL or TP) closed it.
"""

import asyncio
import enum
from dataclasses import dataclass


class Phase(enum.Enum):
    FLAT = "flat"
    PENDING_ENTRY = "pending_entry"
    POSITION_OPEN = "position_open"


@dataclass
class TradeState:
    phase: Phase = Phase.FLAT
    direction: str | None = None  # "long" / "short"
    entry_order_id: int | None = None
    stop_order_id: int | None = None
    target_order_id: int | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None


class StateMachine:
    def __init__(self):
        self._state = TradeState()
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """Callers should hold this for the whole check-then-act sequence
        (e.g. "is it flat? if so, place order and mark pending") so a
        webhook and a stream event can't race each other."""
        return self._lock

    def snapshot(self) -> TradeState:
        return self._state

    def can_accept_new_signal(self) -> bool:
        return self._state.phase == Phase.FLAT

    def mark_pending(self, direction: str, entry_order_id: int,
                      stop_order_id: int | None, target_order_id: int | None,
                      entry_price: float, stop_price: float, target_price: float):
        self._state = TradeState(
            phase=Phase.PENDING_ENTRY,
            direction=direction,
            entry_order_id=entry_order_id,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
        )

    def mark_filled(self):
        if self._state.phase != Phase.PENDING_ENTRY:
            return
        self._state.phase = Phase.POSITION_OPEN

    def mark_flat(self):
        self._state = TradeState()

    def matches_pending_entry(self, order_id: int) -> bool:
        return self._state.phase == Phase.PENDING_ENTRY and self._state.entry_order_id == order_id

    def matches_open_position(self, order_id: int) -> bool:
        if self._state.phase != Phase.POSITION_OPEN:
            return False
        return order_id in (self._state.stop_order_id, self._state.target_order_id)
