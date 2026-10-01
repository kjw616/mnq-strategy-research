"""Tradovate real-time WebSocket stream: the authoritative source of truth
for whether an order has actually filled (not the Pine script's own chart-
side "entry filled" alertcondition, which only reflects the chart).

Wire protocol (newline-delimited, SockJS-style) confirmed against
Tradovate's own developer docs and community forum during design:
  - server sends a bare "o" on connect
  - client authorizes with "authorize\\n0\\n\\n{accessToken}"
  - client sends "[]" every ~2.5s as a heartbeat once authorized
  - requests are "{endpoint}\\n{requestId}\\n\\n{jsonBody}"
  - responses arrive as "a" + a JSON array of {"i", "s", "d"} objects

What's confirmed less rigorously: the exact shape of *incremental* push
events after the initial user/syncrequest snapshot (vs. the snapshot's own
{"orders": [...], "fills": [...], "positions": [...]} shape). This client
forwards every parsed event dict to `on_event` unchanged rather than
committing to a rigid schema - state.py inspects the dict defensively.
Confirm the real incremental shape against a demo connection (build-order
step 3 in the plan) before trusting this for live signals.
"""

import asyncio
import json
import logging

import websockets

from .config import TradingConfig

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 2.5


class TradovateStream:
    def __init__(self, config: TradingConfig, get_access_token, on_event):
        """`get_access_token` is a zero-arg callable returning the current
        bearer token (so a mid-session token refresh in TradovateClient is
        picked up automatically). `on_event` is called with each parsed
        event dict (the "d" payload of every array-frame entry)."""
        self._config = config
        self._get_access_token = get_access_token
        self._on_event = on_event
        self._ws = None
        self._request_id = 1  # 0 is reserved for the auth frame
        self._stop = False

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def run_forever(self):
        """Reconnects with backoff on any disconnect - a live signal bot
        can't afford to just silently stop listening."""
        backoff = 1
        while not self._stop:
            try:
                await self._run_once()
                backoff = 1
            except (websockets.exceptions.WebSocketException, OSError, asyncio.TimeoutError) as e:
                logger.warning("Tradovate stream disconnected (%s), reconnecting in %ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def stop(self):
        self._stop = True

    async def _run_once(self):
        async with websockets.connect(self._config.ws_url, ping_interval=None) as ws:
            self._ws = ws

            first = await ws.recv()
            if first != "o":
                raise RuntimeError(f"Expected open frame 'o', got {first!r}")

            await ws.send(f"authorize\n0\n\n{self._get_access_token()}")
            auth_resp = await ws.recv()
            if not self._frame_is_success(auth_resp):
                raise RuntimeError(f"Tradovate WS authorization failed: {auth_resp!r}")

            sync_id = self._next_request_id()
            await ws.send(f"user/syncrequest\n{sync_id}\n\n{{}}")

            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                async for message in ws:
                    self._handle_message(message)
            finally:
                heartbeat_task.cancel()

    async def _heartbeat_loop(self, ws):
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            await ws.send("[]")

    @staticmethod
    def _frame_is_success(frame: str) -> bool:
        if not frame.startswith("a"):
            return False
        try:
            payload = json.loads(frame[1:])
        except json.JSONDecodeError:
            return False
        return any(entry.get("s") == 200 for entry in payload if isinstance(entry, dict))

    def _handle_message(self, message: str):
        if message in ("o", "h"):
            return  # heartbeat / keepalive, nothing to do
        if message == "c":
            raise RuntimeError("Tradovate closed the WebSocket")
        if not message.startswith("a"):
            logger.debug("Unrecognized Tradovate WS frame: %r", message)
            return
        try:
            entries = json.loads(message[1:])
        except json.JSONDecodeError:
            logger.warning("Failed to parse Tradovate WS frame: %r", message)
            return
        for entry in entries:
            if isinstance(entry, dict) and "d" in entry:
                self._on_event(entry["d"])
