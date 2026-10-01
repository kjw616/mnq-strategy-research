"""Async REST client for Tradovate's order-entry API.

Auth flow, placeOSO field names, and the demo/live base URLs were confirmed
against Tradovate's own developer docs and community forum during the
design phase (tradovate/example-api-js, community.tradovate.com) - not
guessed. Still: the community forum has several threads of developers
hitting trouble specifically with the bracket ("OSO") part of order
placement, so the manual demo-account test (see the plan's build order)
should confirm bracket1/bracket2 actually go live as resting orders after
the parent fills before this is trusted with real signals. If that proves
flaky, `place_limit_entry` + a follow-up `place_oco_exit` call (issued once
the fill event arrives over the WebSocket stream) is the fallback - kept as
a separate method here for exactly that reason, not as dead code.
"""

import time

import httpx

from .config import Secrets, TradingConfig


class TradovateAuthError(RuntimeError):
    pass


class TradovateClient:
    def __init__(self, secrets: Secrets, trading_config: TradingConfig):
        self._secrets = secrets
        self._config = trading_config
        self._http = httpx.AsyncClient(base_url=trading_config.base_url, timeout=10.0)
        self._access_token: str | None = None
        self._expiration_epoch: float = 0.0

    async def aclose(self):
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    async def _ensure_token(self):
        # 60s safety buffer before actual expiry.
        if self._access_token and time.time() < self._expiration_epoch - 60:
            return
        await self._authenticate()

    async def _authenticate(self):
        body = {
            "name": self._secrets.username,
            "password": self._secrets.password,
            "appId": self._secrets.app_id,
            "appVersion": self._secrets.app_version,
            "cid": self._secrets.cid,
            "sec": self._secrets.sec,
            "deviceId": self._secrets.device_id,
        }
        resp = await self._http.post("/auth/accesstokenrequest", json=body)
        resp.raise_for_status()
        data = resp.json()
        if "accessToken" not in data:
            raise TradovateAuthError(f"Auth failed: {data}")
        self._access_token = data["accessToken"]
        # expirationTime is an ISO string from Tradovate; fall back to a
        # conservative 1-hour assumption if the field is ever missing.
        expiration_iso = data.get("expirationTime")
        if expiration_iso:
            import datetime

            self._expiration_epoch = datetime.datetime.fromisoformat(
                expiration_iso.replace("Z", "+00:00")
            ).timestamp()
        else:
            self._expiration_epoch = time.time() + 3600

    async def _headers(self) -> dict:
        await self._ensure_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    @property
    def access_token(self) -> str | None:
        """For handing to tradovate_stream.py's WebSocket authorize frame."""
        return self._access_token

    # ------------------------------------------------------------------
    # Account discovery (used by `main.py --list-accounts`)
    # ------------------------------------------------------------------
    async def list_accounts(self) -> list[dict]:
        headers = await self._headers()
        resp = await self._http.get("/account/list", headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    async def place_bracket_entry(
        self, action: str, order_qty: int, entry_price: float,
        stop_price: float, target_price: float,
    ) -> dict:
        """One-shot entry + bracket (placeOSO). `action` is "Buy" or "Sell"
        for the entry; the bracket legs close the position so they take the
        opposite action automatically below."""
        exit_action = "Sell" if action == "Buy" else "Buy"
        headers = await self._headers()
        body = {
            "accountId": self._config.account_id,
            "accountSpec": self._config.account_spec,
            "symbol": self._config.symbol,
            "action": action,
            "orderType": "Limit",
            "price": entry_price,
            "orderQty": order_qty,
            "isAutomated": True,
            "bracket1": {
                "action": exit_action,
                "orderType": "Stop",
                "stopPrice": stop_price,
            },
            "bracket2": {
                "action": exit_action,
                "orderType": "Limit",
                "price": target_price,
            },
        }
        resp = await self._http.post("/order/placeOSO", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def place_oco_exit(
        self, action: str, order_qty: int, stop_price: float, target_price: float,
    ) -> dict:
        """Fallback: place the stop+target OCO pair separately, once a fill
        event confirms the entry is actually a live position. Use if
        place_bracket_entry's brackets prove unreliable in the demo test."""
        headers = await self._headers()
        body = {
            "accountId": self._config.account_id,
            "accountSpec": self._config.account_spec,
            "symbol": self._config.symbol,
            "action": action,
            "orderQty": order_qty,
            "isAutomated": True,
            "order1": {"orderType": "Stop", "stopPrice": stop_price},
            "order2": {"orderType": "Limit", "price": target_price},
        }
        resp = await self._http.post("/order/placeOCO", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def place_limit_entry(self, action: str, order_qty: int, entry_price: float) -> dict:
        """Entry only, no bracket - paired with place_oco_exit as the fallback path."""
        headers = await self._headers()
        body = {
            "accountId": self._config.account_id,
            "accountSpec": self._config.account_spec,
            "symbol": self._config.symbol,
            "action": action,
            "orderType": "Limit",
            "price": entry_price,
            "orderQty": order_qty,
            "isAutomated": True,
        }
        resp = await self._http.post("/order/placeOrder", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def cancel_order(self, order_id: int) -> dict:
        headers = await self._headers()
        resp = await self._http.post("/order/cancelOrder", json={"orderId": order_id}, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def get_order(self, order_id: int) -> dict:
        headers = await self._headers()
        resp = await self._http.get(f"/order/item", params={"id": order_id}, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def list_orders(self) -> list[dict]:
        headers = await self._headers()
        resp = await self._http.get("/order/list", headers=headers)
        resp.raise_for_status()
        return resp.json()
