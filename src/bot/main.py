"""Entrypoint. Normal run:

    python -m src.bot.main

Prints your Tradovate accounts (to fill in bot_config.json's account_id/
account_spec) without starting the bot:

    python -m src.bot.main --list-accounts
"""

import asyncio
import logging
import sys

import uvicorn

from . import config as config_module
from .state import StateMachine
from .tradovate_client import TradovateClient
from .tradovate_stream import TradovateStream
from .webhook_server import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def list_accounts():
    secrets = config_module.Secrets.load()
    # Only needs tradovate_env/base_url from bot_config.json - account_id/
    # account_spec aren't required yet (that's what we're here to find).
    raw_config = config_module.read_raw_config()
    env = raw_config.get("tradovate_env", "demo")
    stub_config = config_module.TradingConfig(
        tradovate_env=env, risk_per_trade_usd=0, max_contracts=0,
        symbol="", tick_size=0, tick_value=0, account_id=0, account_spec="",
    )
    client = TradovateClient(secrets, stub_config)
    try:
        accounts = await client.list_accounts()
        print(f"\nAccounts on your {env} environment:\n")
        for acct in accounts:
            print(f"  id={acct.get('id')}  name={acct.get('name')}  active={acct.get('active')}")
        print(f"\nFill the matching one into bot_config.json under accounts.{env}.")
    finally:
        await client.aclose()


def _on_stream_event(state: StateMachine, event: dict):
    """Best-effort mapping of Tradovate's user-sync push events onto the
    trade state machine. The exact incremental-event shape should be
    confirmed against a real demo connection (plan build-order step 3) -
    this logs anything it doesn't recognize rather than silently dropping
    it, so gaps are visible instead of hidden."""
    orders = event.get("orders", [])
    fills = event.get("fills", [])

    if fills:
        snapshot = state.snapshot()
        for fill in fills:
            order_id = fill.get("orderId")
            if snapshot.phase.value == "pending_entry" and order_id == snapshot.entry_order_id:
                state.mark_filled()
                logger.info("Entry order %s filled - position now open.", order_id)
            elif state.matches_open_position(order_id):
                state.mark_flat()
                logger.info("Exit order %s filled - position closed, back to flat.", order_id)

    for order in orders:
        order_id = order.get("id")
        status = order.get("ordStatus") or order.get("status")
        if status in ("Canceled", "Rejected") and state.matches_pending_entry(order_id):
            logger.warning("Entry order %s was %s outside of our own cancel flow.", order_id, status)
            state.mark_flat()

    if not orders and not fills:
        logger.debug("Unrecognized stream event shape: %s", event)


async def run_bot():
    secrets = config_module.Secrets.load()
    trading_config = config_module.load_trading_config()

    client = TradovateClient(secrets, trading_config)
    state = StateMachine()

    stream = TradovateStream(
        trading_config,
        get_access_token=lambda: client.access_token,
        on_event=lambda event: _on_stream_event(state, event),
    )

    app.state.tradovate_client = client
    app.state.trade_state = state
    app.state.secrets = secrets

    uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(uvicorn_config)

    try:
        await asyncio.gather(server.serve(), stream.run_forever())
    finally:
        await client.aclose()


if __name__ == "__main__":
    if "--list-accounts" in sys.argv:
        asyncio.run(list_accounts())
    else:
        asyncio.run(run_bot())
