"""Receives Keens_Indicator_v2.1.pine's "armed"/"invalidated" alerts and
turns them into real Tradovate orders.

TradingView's alert webhooks can't send custom headers - the shared secret
has to live in the URL itself, so the alert's webhook URL is configured as
`https://<host>/webhook/<WEBHOOK_SHARED_SECRET>` (the same pattern used by
TradersPost/PickMyTrade: a unique, unguessable per-user URL).
"""

import json
import logging

from fastapi import FastAPI, HTTPException, Request

from . import config as config_module
from .sizing import compute_contracts
from .state import StateMachine
from .tradovate_client import TradovateClient

logger = logging.getLogger(__name__)

app = FastAPI()


def _get_client(request: Request) -> TradovateClient:
    return request.app.state.tradovate_client


def _get_state(request: Request) -> StateMachine:
    return request.app.state.trade_state


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    expected_secret = request.app.state.secrets.webhook_shared_secret
    if secret != expected_secret:
        # Deliberately generic - don't hint at what a valid secret looks like.
        raise HTTPException(status_code=404)

    raw_body = (await request.body()).decode("utf-8", errors="replace")
    # Pine sends a human-readable line first (for phone push notifications)
    # then a trailing JSON line for us - pull out the JSON tail rather than
    # assuming the whole body is JSON. See Keens_Indicator_v2.1.pine's
    # Alerts section for the exact message format.
    json_start = raw_body.find("{")
    if json_start == -1:
        logger.warning("Webhook received a body with no JSON payload: %r", raw_body[:200])
        raise HTTPException(status_code=400, detail="no JSON payload found")
    try:
        payload = json.loads(raw_body[json_start:])
    except ValueError:
        logger.warning("Webhook received unparseable JSON tail: %r", raw_body[:200])
        raise HTTPException(status_code=400, detail="invalid JSON")

    event = payload.get("event")
    client = _get_client(request)
    state = _get_state(request)

    if event == "armed":
        await _handle_armed(payload, client, state)
    elif event == "invalidated":
        await _handle_invalidated(payload, client, state)
    else:
        logger.warning("Webhook received unknown event: %r", payload)

    return {"ok": True}


async def _handle_armed(payload: dict, client: TradovateClient, state: StateMachine):
    async with state.lock:
        if not state.can_accept_new_signal():
            logger.info("Ignoring 'armed' signal - a trade is already pending/open. %s", payload)
            return

        trading_config = config_module.load_trading_config()  # fresh read - risk/cap can change on a whim

        if not trading_config.enabled:
            logger.info("Ignoring 'armed' signal - bot_config.json 'enabled' is false. %s", payload)
            return

        direction = payload["direction"]
        entry = float(payload["entry"])
        stop = float(payload["stop"])
        target = float(payload["target"])

        contracts = compute_contracts(
            entry=entry, stop=stop,
            tick_size=trading_config.tick_size, tick_value=trading_config.tick_value,
            risk_usd=trading_config.risk_per_trade_usd, max_contracts=trading_config.max_contracts,
            max_risk_per_trade_usd=trading_config.max_risk_per_trade_usd,
        )
        if contracts == 0:
            logger.warning(
                "Skipping signal - sizing returned 0 contracts (stop too wide for risk_per_trade_usd=%s, "
                "or degenerate entry/stop). %s", trading_config.risk_per_trade_usd, payload,
            )
            return

        action = "Buy" if direction == "long" else "Sell"
        response = await client.place_bracket_entry(
            action=action, order_qty=contracts,
            entry_price=entry, stop_price=stop, target_price=target,
        )

        # placeOSO's response is expected to carry the parent entry order's
        # id; bracket order ids aren't guaranteed to come back in this same
        # response (per Tradovate's own docs/forum, resolved during the
        # manual demo test - see plan). If they're absent here, the stream's
        # order-sync events fill them in via StateMachine once observed.
        entry_order_id = response.get("orderId")
        if entry_order_id is None:
            logger.error("place_bracket_entry response missing orderId: %s", response)
            return

        state.mark_pending(
            direction=direction, entry_order_id=entry_order_id,
            stop_order_id=response.get("bracket1", {}).get("orderId"),
            target_order_id=response.get("bracket2", {}).get("orderId"),
            entry_price=entry, stop_price=stop, target_price=target,
        )
        logger.info("Placed %s entry: %s contracts @ %s (SL %s / TP %s), orderId=%s",
                    direction, contracts, entry, stop, target, entry_order_id)


async def _handle_invalidated(payload: dict, client: TradovateClient, state: StateMachine):
    async with state.lock:
        snapshot = state.snapshot()
        if snapshot.phase.value != "pending_entry":
            logger.info("Ignoring 'invalidated' signal - no resting order to cancel. %s", payload)
            return

        # Double-check against Tradovate's own order status rather than
        # trusting local state alone - guards against a race where the
        # entry filled a moment before this invalidation arrived.
        order = await client.get_order(snapshot.entry_order_id)
        order_status = order.get("ordStatus") or order.get("status")
        if order_status in ("Filled", "Completed"):
            logger.warning(
                "'invalidated' arrived but order %s already filled - leaving position alone. %s",
                snapshot.entry_order_id, payload,
            )
            return

        await client.cancel_order(snapshot.entry_order_id)
        state.mark_flat()
        logger.info("Cancelled resting order %s on invalidation.", snapshot.entry_order_id)
