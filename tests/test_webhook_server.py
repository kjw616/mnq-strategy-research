"""Exercises the webhook against a stub Tradovate client - no real network
calls. Focused on: secret checking, the hybrid human-text+JSON message
parsing (matches Keens_Indicator_v2.1.pine's actual alert format), the
one-trade-at-a-time guard, and the cancel-on-invalidate flow."""

import pytest
from fastapi.testclient import TestClient

from src.bot import webhook_server
from src.bot.config import TradingConfig
from src.bot.state import StateMachine

SECRET = "test-secret"

LONG_ARMED_BODY = (
    "🟢 MNQ1! LONG SETUP\n📍 Limit Buy: 21050.25\n🛑 Stop: 21040.00\n🎯 Target: 21065.625\n"
    '{"event":"armed","symbol":"MNQ1!","direction":"long","entry":21050.25,"stop":21040.00,"target":21065.625}'
)
LONG_INVALIDATED_BODY = (
    "⚪ MNQ1! LONG invalidated\n❌ Cancel your resting BUY order\n"
    '{"event":"invalidated","symbol":"MNQ1!","direction":"long"}'
)


class StubSecrets:
    webhook_shared_secret = SECRET


class StubTradovateClient:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.order_status = "Working"

    async def place_bracket_entry(self, action, order_qty, entry_price, stop_price, target_price):
        self.placed.append((action, order_qty, entry_price, stop_price, target_price))
        return {"orderId": 555}

    async def get_order(self, order_id):
        return {"ordStatus": self.order_status}

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"orderId": order_id}


@pytest.fixture
def client(monkeypatch):
    stub_config = TradingConfig(
        tradovate_env="demo", risk_per_trade_usd=125, max_contracts=5,
        symbol="MNQZ5", tick_size=0.25, tick_value=0.50,
        account_id=1, account_spec="demo-account",
    )
    monkeypatch.setattr(webhook_server.config_module, "load_trading_config", lambda: stub_config)

    webhook_server.app.state.secrets = StubSecrets()
    webhook_server.app.state.trade_state = StateMachine()
    stub_tradovate = StubTradovateClient()
    webhook_server.app.state.tradovate_client = stub_tradovate

    test_client = TestClient(webhook_server.app)
    test_client.stub_tradovate = stub_tradovate
    return test_client


def test_wrong_secret_rejected(client):
    resp = client.post("/webhook/wrong-secret", content=LONG_ARMED_BODY)
    assert resp.status_code == 404
    assert client.stub_tradovate.placed == []


def test_armed_places_bracket_order_with_correct_size(client):
    resp = client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    assert resp.status_code == 200
    action, qty, entry, stop, target = client.stub_tradovate.placed[0]
    assert action == "Buy"
    assert qty == 5  # clamped: 40-tick stop -> 6 raw, capped at max_contracts=5
    assert entry == 21050.25
    assert stop == 21040.00
    assert target == 21065.625

    state = webhook_server.app.state.trade_state.snapshot()
    assert state.phase.value == "pending_entry"
    assert state.entry_order_id == 555


def test_second_armed_signal_ignored_while_pending(client):
    client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    assert len(client.stub_tradovate.placed) == 1


def test_invalidated_cancels_resting_order(client):
    client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    resp = client.post(f"/webhook/{SECRET}", content=LONG_INVALIDATED_BODY)
    assert resp.status_code == 200
    assert client.stub_tradovate.cancelled == [555]
    assert webhook_server.app.state.trade_state.snapshot().phase.value == "flat"


def test_invalidated_does_not_cancel_an_already_filled_order(client):
    client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    client.stub_tradovate.order_status = "Filled"
    client.post(f"/webhook/{SECRET}", content=LONG_INVALIDATED_BODY)
    assert client.stub_tradovate.cancelled == []


def test_body_without_json_tail_is_rejected(client):
    resp = client.post(f"/webhook/{SECRET}", content="just some plain text, no JSON here")
    assert resp.status_code == 400


def test_disabled_config_ignores_armed_signal(client, monkeypatch):
    disabled_config = TradingConfig(
        tradovate_env="demo", risk_per_trade_usd=125, max_contracts=5,
        symbol="MNQZ5", tick_size=0.25, tick_value=0.50,
        account_id=1, account_spec="demo-account", enabled=False,
    )
    monkeypatch.setattr(webhook_server.config_module, "load_trading_config", lambda: disabled_config)

    resp = client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)
    assert resp.status_code == 200
    assert client.stub_tradovate.placed == []
    assert webhook_server.app.state.trade_state.snapshot().phase.value == "flat"


def test_disabled_config_still_allows_cancel(client, monkeypatch):
    # Pausing new entries shouldn't block cleaning up an already-resting order.
    client.post(f"/webhook/{SECRET}", content=LONG_ARMED_BODY)

    disabled_config = TradingConfig(
        tradovate_env="demo", risk_per_trade_usd=125, max_contracts=5,
        symbol="MNQZ5", tick_size=0.25, tick_value=0.50,
        account_id=1, account_spec="demo-account", enabled=False,
    )
    monkeypatch.setattr(webhook_server.config_module, "load_trading_config", lambda: disabled_config)

    resp = client.post(f"/webhook/{SECRET}", content=LONG_INVALIDATED_BODY)
    assert resp.status_code == 200
    assert client.stub_tradovate.cancelled == [555]
