# Setting up Keen's Indicator + the execution bot

A step-by-step walkthrough for running your own copy of this setup from
scratch: a TradingView indicator that spots a specific setup and an
automated bot that places the trade for you on Tradovate.

**Read this whole thing before you touch real money.** This is a real bot
that places real orders with real dollars. You're setting up your own
independent copy — your own Tradovate account, your own credentials, your
own risk settings. Test everything on Tradovate's demo/simulation
account first, and don't skip that step just because it feels slow.

## What this actually costs

Before you go any further, here's what you're signing up to pay for. These
are the costs of the *system itself* — what's required to make this work
at all.

**Required, ongoing:**
- **TradingView subscription** that supports webhook alerts — the free
  plan can't send them, you need at least the Essential/Pro tier
  (~$13-15/month).
- **Real-time market data add-on** for whatever exchange you're charting
  (Nasdaq, for MNQ) — usually a few dollars a month, check TradingView's
  own data store for the exact current price.
- **Tradovate API Access add-on** — currently ~$25/month. This is what
  unlocks the bot being able to place orders at all.
- **Tradovate commissions** — a small per-contract fee charged on every
  trade the bot places (charged automatically, not a subscription — scales
  with how often you trade, not a flat monthly cost).

**Required, one-time/capital (not a fee — this is your own trading money):**
- **At least $1,000 funded into a live Tradovate account** — Tradovate's
  own requirement just to unlock API access, even though you'll be
  testing on their separate demo/simulation environment. This money stays
  yours; it's not spent, just required to be sitting in the account.

**Only needed later, not for initial setup/testing:**
- **A cloud VPS** (~$5-6/month) — only once you want the bot reachable by
  TradingView from anywhere, so it can run without your own computer being
  on. Not needed while you're testing locally.
- **Tradovate's "TradingView configuration" add-on** (~$10/month) — this
  is for *manually* viewing/trading your Tradovate account from inside
  TradingView's chart. The bot itself doesn't use this at all (it talks to
  Tradovate directly), so skip it unless you specifically want that manual
  view too.

Prices change — double-check current numbers directly on each site rather
than trusting this list blindly months from now.

## What you're getting

- `pine/Keens_Indicator_v2.1.pine` — the TradingView indicator
- `src/bot/` — the bot's code
- `requirements.txt` — the Python packages it needs
- `.env.example` and `bot_config.example.json` — templates for your own
  secrets/settings (you'll copy these and fill in your own values)
- `README.md` — a technical reference for day-to-day use once it's running
- this file

## Step 1: Get Python installed

You need Python 3.10 or newer. If you don't already have it, download it
from [python.org](https://python.org) and install it normally.

## Step 2: Set up the project folder

Clone this repo, open a terminal in its folder, and run:

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Add the indicator to TradingView

1. Open TradingView, go to the Pine Editor.
2. Paste in the contents of `pine/Keens_Indicator_v2.1.pine`.
3. Save it and add it to a chart (works on the 1-minute or 5-minute
   timeframe — it's built for those).

## Step 4: Set up your own Tradovate account

1. Sign up for Tradovate if you don't already have an account.
2. Fund a **live** account with at least $1,000 — this is Tradovate's own
   requirement to unlock API access, even though we'll point everything at
   the **demo/simulation** environment for testing first.
3. Subscribe to Tradovate's **API Access** add-on (currently ~$25/month).
4. Generate your API key: Settings → Add-Ons → API Access → Generate. This
   shows you a **Client ID (cid)** and **Client Secret (sec)** — copy both
   immediately, they're only shown once.

## Step 5: Configure your secrets

Copy `.env.example` to a new file called `.env` in the same folder, and
fill in:

| Field | Where it comes from |
|---|---|
| `TRADOVATE_USERNAME` / `TRADOVATE_PASSWORD` | Your own Tradovate login |
| `TRADOVATE_CID` / `TRADOVATE_SEC` | From step 4 |
| `TRADOVATE_APP_ID` / `TRADOVATE_APP_VERSION` | Made up by you — e.g. `MyBot` / `1.0` |
| `TRADOVATE_DEVICE_ID` | A random unique string you generate once |
| `WEBHOOK_SHARED_SECRET` | A random secret string you generate once |

Generate the last two with:

```bash
python -c "import uuid, secrets; print('DEVICE_ID:', uuid.uuid4()); print('WEBHOOK_SECRET:', secrets.token_urlsafe(32))"
```

**Never share this `.env` file with anyone. Never commit it to git or
upload it anywhere.** (It's already in `.gitignore`.)

## Step 6: Configure your own risk settings

Copy `bot_config.example.json` to `bot_config.json`. This is where you set
**your own** risk per trade and contract cap, sized for your account.
Two fields matter most:

- `risk_per_trade_usd` — how many dollars you're willing to risk on a
  single trade.
- `max_contracts` — the hard ceiling on contract size no matter what the
  math says.

Start conservative (a small dollar amount relative to your account size)
and scale up only once you're comfortable — see `README.md`'s "Changing
risk / contract sizing" section for the full explanation of every field.

## Step 7: Find your account ID

Run:

```bash
python -m src.bot.main --list-accounts
```

This logs in with your credentials and prints your account(s). Copy your
**demo** account's ID and name into `bot_config.json` under
`accounts.demo`.

## Step 8: Point TradingView's alert at your bot

In TradingView, right-click the indicator → Add Alert → set Condition to
"Keen's Indicator v2.1" → "alert() function calls". Set the webhook URL to:

```
https://<wherever-your-bot-is-reachable>/webhook/<your-WEBHOOK_SHARED_SECRET>
```

If you're just testing locally, your bot won't be reachable from
TradingView's servers yet (a URL like `localhost` only works on your own
machine) — use a tunnel like `ngrok` for testing, or host the bot on a
VPS.

## Step 9: Test before you trust it

With `tradovate_env` set to `"demo"` in `bot_config.json`, run the bot:

```bash
python -m src.bot.main
```

Watch the terminal for log output as real signals come in on demo. Let it
run through several real signals during market hours before you even think
about touching your live account. `README.md` has a full section on what
the log lines mean and how to tell if something's wrong.

## Step 10: Turning it on/off

- **Pause without stopping:** set `"enabled": false` in `bot_config.json`
  and save — takes effect instantly, no restart.
- **Fully stop:** Ctrl+C in the terminal it's running in.
- There's no emergency "flatten everything" button in the bot itself — if
  you ever need to get out of a position immediately, do it directly in
  Tradovate's own app, not through the bot.

Full details, troubleshooting, and the "going live" checklist are in
`README.md` — keep it next to the code for reference.
