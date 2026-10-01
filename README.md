# MNQ Strategy Research & Execution Bot

An end-to-end toolkit for researching, validating, and automating intraday
strategies on Micro Nasdaq futures (MNQ): TradingView indicators written in
Pine Script, faithful Python ports of them for backtesting on two-plus years
of 1-minute data, and a webhook-driven bot that executes signals on
Tradovate.

Built with [Claude Code](https://claude.com/claude-code) as an agentic
development partner: I defined requirements, directed the work, and
verified every result against tests and real market data.

**Stack:** Python (pandas, NumPy), FastAPI + Uvicorn,
httpx / WebSockets (Tradovate REST + streaming APIs), Databento (historical
CME data), Pine Script v6, pytest.

## Highlights

- **Pine → Python ports that match bar-for-bar.** Each indicator in
  [`pine/`](pine/) has a Python counterpart in `src/` that walks 1-minute
  bars forward the way Pine does, with non-repainting higher-timeframe
  alignment, so backtests reflect what the live indicator would have shown.
- **Walk-forward discipline.** Strategies are designed on 2023–2024 and then
  run untouched on 2025–present as a genuine out-of-sample check.
- **Tick-level verification.** Trades where the entry bar also touched the
  stop (ambiguous on 1-minute data) are settled using trade-by-trade prints
  from Databento instead of assumptions (`src/resolve_ambiguous_trades.py`,
  `src/verify_config.py`).
- **Honest negative results.** Ideas that don't survive validation are
  reported as such — e.g. an EMA-cross edge that replicated walk-forward
  disappeared once realistic commissions and slippage were applied
  (`src/ema_cross_multi_tf_backtest.py` documents the follow-up).
- **Production-minded execution bot.** One-trade-at-a-time state machine,
  risk-based position sizing with hard caps, shared-secret webhook auth,
  hot-reloaded config, and a demo/live environment switch
  ([`src/bot/`](src/bot/), tested in [`tests/`](tests/)).

## Project layout

| Path | What's there |
|---|---|
| `pine/` | TradingView indicators (Keen's Indicator v2.1, CRT, ICT A+ Setup Finder, 713, Swing Watch) |
| `src/bot/` | Tradovate execution bot: webhook server, REST/WebSocket clients, sizing, state |
| `src/data.py`, `structure.py`, `trend.py`, `poi.py`, `liquidity.py`, `swing.py`, `indicators.py` | Shared building blocks: HTF resampling, market structure, FVG/RB/OB detection, liquidity pools, VWAP |
| `src/*_strategy.py` / `src/*_backtest.py` | Strategy ports and their walk-forward backtests (liquidity sweep + FVG, CRT, HTF sweep, fib golden zone, EMA cross, rejection blocks) |
| `src/validate_*.py`, `src/report.py` | Directional-edge checks and a reusable win/loss/R summary report |
| `src/download_mnq_*.py` | Pull MNQ 1-minute bars from Databento (data files are not committed) |
| `docs/SETUP_GUIDE.md` | Beginner-friendly walkthrough for running your own copy of the bot |

## Getting the data

Market data isn't included in the repo (it's licensed). With a
`DATABENTO_API_KEY` in `.env`, run:

```bash
python src/download_mnq_data.py       # 2023-2024 -> data/mnq_1m_2023_2025.parquet
python src/download_mnq_oos_data.py   # 2025-present -> data/mnq_1m_2025_present.parquet
```

Both are small billed downloads (a few dollars each); each script prints the
estimated cost and asks for confirmation before downloading.

---

# Keen's Indicator v2.1 — Automated Execution Bot

Listens for [Keens_Indicator_v2.1.pine](pine/Keens_Indicator_v2.1.pine)'s
TradingView alerts and places the exact limit entry + bracket (stop/target)
on a Tradovate account automatically. See `src/bot/` for the implementation,
and [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for a step-by-step setup
walkthrough.

## One-time setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your Tradovate API credentials
   (Tradovate: Application Settings → Add-Ons → API Access) and a
   `WEBHOOK_SHARED_SECRET` (generate one with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
3. Copy `bot_config.example.json` to `bot_config.json`.
4. Run `python -m src.bot.main --list-accounts` to find your account ID/name,
   fill them into `bot_config.json` under `accounts.demo` (or `accounts.live`).
5. In TradingView, add an alert on the indicator: Condition → "Keen's
   Indicator v2.1" → "alert() function calls". Set the webhook URL to
   `https://<your-host>/webhook/<WEBHOOK_SHARED_SECRET>`.

## Running it day-to-day

Start it with:

```bash
python -m src.bot.main
```

This blocks in the foreground and prints logs directly to that terminal -
every signal received, every order placed/skipped/cancelled, and every fill
gets a log line (see `main.py`'s and `webhook_server.py`'s `logger.info`/
`logger.warning` calls). There's no separate dashboard - **the log stream in
that terminal is how you watch what the bot is doing**, and Tradovate's own
web/desktop UI is the source of truth for what's actually happened to your
account (positions, fills, order status) - if the two ever seem to
disagree, trust Tradovate's UI.

A few log lines worth recognizing:
- `Placed <direction> entry: N contracts @ ...` - a real order went out.
- `Skipping signal - sizing returned 0 contracts ...` - a signal arrived but
  was deliberately not traded (stop too wide for both `risk_per_trade_usd`
  and `max_risk_per_trade_usd` - see the sizing table below).
- `Ignoring 'armed' signal - a trade is already pending/open` - working as
  designed, one trade at a time.
- `Ignoring 'armed' signal - bot_config.json 'enabled' is false` - the pause
  flag is on.

If you close the terminal (or it crashes), the bot stops entirely -
including its Tradovate WebSocket connection. See "If something goes wrong"
below before just restarting it.

## Changing risk / contract sizing

All of this lives in `bot_config.json` and takes effect on the **very next
signal** — no restart needed, since the bot re-reads this file fresh every
time an alert comes in.

| Field | What it controls |
|---|---|
| `enabled` | `true`/`false` - the pause switch, see "Turning the bot on/off" below. |
| `risk_per_trade_usd` | Target dollar risk per trade. Contracts = `floor(risk_per_trade_usd / (stop_distance_in_ticks * tick_value))`. |
| `max_contracts` | Hard ceiling on contracts for any single trade, no matter how tight the stop (protects against the "tight stop implies hundreds of contracts" tail). |
| `max_risk_per_trade_usd` | If the stop is too *wide* for `risk_per_trade_usd` to afford even 1 contract, the trade is still taken at 1 contract as long as that 1 contract's risk is under this ceiling — otherwise it's skipped. Prevents "just take 1 no matter what" from exposing you to an unbounded loss on an unusually wide-stop trade. |
| `tradovate_env` | `"demo"` or `"live"` - which account/environment orders go to. |
| `symbol` | The actual Tradovate contract (e.g. `"MNQZ5"`), not TradingView's continuous ticker - update every quarterly roll. |
| `tick_size` / `tick_value` | Only change these if you switch instruments - they're specific to MNQ (0.25 pts / $0.50 per tick). |
| `accounts.demo` / `accounts.live` | Account ID/name for each environment - found via `--list-accounts`. |

Reference from the actual 1,086-trade backtest (v2.1 defaults): median stop
distance is **2.34 points** (~$4.69 risk on 1 MNQ contract), 90th percentile
is **7.05 points** (~$14.10). Most trades are cheap to size; only the
occasional wide-stop trade needs `max_risk_per_trade_usd` to avoid being
skipped.

Just edit the numbers and save - e.g. to scale up once comfortable:

```json
"risk_per_trade_usd": 250,
"max_risk_per_trade_usd": 500,
"max_contracts": 10,
```

## Turning the bot on/off

**Quick pause (bot process keeps running, stays connected to Tradovate):**
set `"enabled": false` in `bot_config.json` and save. New "armed" signals are
ignored until you flip it back to `true` — no restart needed. Cancelling an
already-resting order on an "invalidated" signal still works while paused
(pausing only blocks *new* entries, not cleanup of one already placed).

**Fully stopping the bot (process exits, WebSocket disconnects):**

```bash
python -m src.bot.main
```

Ctrl+C in that terminal to stop it. If you're running it as a background
service (e.g. via `systemd` or `screen`/`tmux` on a VPS), stop/start it the
same way you would any other long-running process.

Note: stopping the process does **not** cancel any order already resting on
Tradovate - check Tradovate's own UI if you need to flatten/cancel something
while the bot is off. `enabled: false` is the safer way to pause if you want
the bot to keep managing an in-flight trade's bracket while blocking new ones.

**There's no "panic button" beyond this.** The bot can't flatten a position
or cancel an order on demand — if you need to get out of something *right
now*, do it directly in Tradovate's own UI, not through the bot. `enabled:
false` only stops *new* entries; it doesn't touch a position that's already
open.

## If something goes wrong

**Restarting after a crash or an intentional stop:** the bot's memory of
"is a trade pending/open" lives only in the running process (`state.py`) -
it resets to flat the instant the process restarts, even if Tradovate
actually still has a resting order or open position from before the crash.
That means restarting blind could let the bot accept a brand-new signal
while an old trade is still live on Tradovate, breaking the one-trade-at-a-
time guarantee. **Before restarting, check Tradovate's UI for any open
position or resting order first** - if one exists, either let it resolve
naturally or handle it manually before starting the bot back up.

**Common startup errors and what they mean:**
- `Missing required .env values: ...` - one of the `.env` fields is still
  blank; the message lists exactly which ones.
- `bot_config.json not found` - you haven't copied `bot_config.example.json`
  to `bot_config.json` yet.
- `accounts.<env>.account_id/account_spec not filled in yet` - run
  `python -m src.bot.main --list-accounts` and fill those in.
- `tradovate_env must be 'demo' or 'live'` - typo in that field.

**Webhook returning an error when TradingView fires an alert:**
- `404` - the secret in the alert's webhook URL doesn't match
  `WEBHOOK_SHARED_SECRET` in `.env`.
- `400` - the alert body didn't contain a parseable JSON tail; check the
  Pine script's alert message wasn't edited/truncated.

## Going live

1. Confirm `tradovate_env` is `"demo"` and test end-to-end there first
   (see the plan's build-order steps for the manual/curl tests).
2. Only after a clean demo soak test during real market hours: set
   `tradovate_env` to `"live"` and confirm `accounts.live` is filled in.
3. Update `symbol` each quarterly futures roll (e.g. `"MNQZ5"` → `"MNQH6"`) -
   this is not automatic.
