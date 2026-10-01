"""Bot configuration: secrets from .env (rarely change), trading knobs from
bot_config.json (meant to be hand-edited on a whim, no restart required).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_CONFIG_PATH = REPO_ROOT / "bot_config.json"

TRADOVATE_BASE_URLS = {
    "demo": "https://demo.tradovateapi.com/v1",
    "live": "https://live.tradovateapi.com/v1",
}
TRADOVATE_WS_URLS = {
    "demo": "wss://demo.tradovateapi.com/v1/websocket",
    "live": "wss://live.tradovateapi.com/v1/websocket",
}


@dataclass(frozen=True)
class Secrets:
    """Tradovate API credentials + webhook auth. Changes rarely, lives in
    .env (not committed) rather than bot_config.json."""

    username: str
    password: str
    app_id: str
    app_version: str
    cid: str
    sec: str
    device_id: str
    webhook_shared_secret: str

    @classmethod
    def load(cls) -> "Secrets":
        required = {
            "username": "TRADOVATE_USERNAME",
            "password": "TRADOVATE_PASSWORD",
            "app_id": "TRADOVATE_APP_ID",
            "app_version": "TRADOVATE_APP_VERSION",
            "cid": "TRADOVATE_CID",
            "sec": "TRADOVATE_SEC",
            "device_id": "TRADOVATE_DEVICE_ID",
            "webhook_shared_secret": "WEBHOOK_SHARED_SECRET",
        }
        values = {}
        missing = []
        for field, env_key in required.items():
            val = os.getenv(env_key)
            if not val:
                missing.append(env_key)
            values[field] = val
        if missing:
            raise SystemExit(
                f"Missing required .env values: {', '.join(missing)} "
                f"(check .env at the project root)."
            )
        return cls(**values)


@dataclass(frozen=True)
class TradingConfig:
    """Trading knobs - re-read fresh from bot_config.json on every signal so
    risk_per_trade_usd / max_contracts can be changed without restarting the
    bot. `symbol` is the actual Tradovate contract (e.g. "MNQZ5"), NOT
    TradingView's continuous ticker ("MNQ1!") - update it yourself each
    quarterly roll."""

    tradovate_env: str
    risk_per_trade_usd: float
    max_contracts: int
    symbol: str
    tick_size: float
    tick_value: float
    account_id: int
    account_spec: str
    max_risk_per_trade_usd: float | None = None
    enabled: bool = True

    @property
    def base_url(self) -> str:
        return TRADOVATE_BASE_URLS[self.tradovate_env]

    @property
    def ws_url(self) -> str:
        return TRADOVATE_WS_URLS[self.tradovate_env]


def read_raw_config(path: Path = BOT_CONFIG_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} not found - copy bot_config.example.json to bot_config.json "
            f"and fill in your account_id/account_spec (see `--list-accounts`)."
        )
    return json.loads(path.read_text())


def load_trading_config(path: Path = BOT_CONFIG_PATH) -> TradingConfig:
    raw = read_raw_config(path)
    env = raw["tradovate_env"]
    if env not in TRADOVATE_BASE_URLS:
        raise SystemExit(f"tradovate_env must be 'demo' or 'live', got {env!r}")
    account = raw["accounts"][env]
    if not account.get("account_id") or not account.get("account_spec"):
        raise SystemExit(
            f"bot_config.json: accounts.{env}.account_id/account_spec not filled in yet - "
            f"run with --list-accounts to find them."
        )
    return TradingConfig(
        tradovate_env=env,
        risk_per_trade_usd=raw["risk_per_trade_usd"],
        max_contracts=raw["max_contracts"],
        symbol=raw["symbol"],
        tick_size=raw["tick_size"],
        tick_value=raw["tick_value"],
        account_id=account["account_id"],
        account_spec=account["account_spec"],
        max_risk_per_trade_usd=raw.get("max_risk_per_trade_usd"),
        enabled=raw.get("enabled", True),
    )
