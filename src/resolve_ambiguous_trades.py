# Settles the ~708 trades where the buggy and fixed backtest logic disagreed
# (same-bar entry+resolution) by pulling actual trade-by-trade prints from
# Databento for each disputed entry-bar minute and checking the REAL
# chronological order: did price actually trade through the FVG retest zone
# (true entry) before or after it traded through the stop level?
#
# Costs ~$0.004 per trade queried (a fixed per-request floor, not
# duration-based) - roughly $2.90 for all 708. Progress is saved
# incrementally so this can resume if interrupted.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.resolve_ambiguous_trades

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import databento as db
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

IN_PATH = "data/disagreement_trades_to_resolve.csv"
OUT_PATH = "data/disagreement_trades_resolved.csv"
MAX_WORKERS = 6  # modest concurrency - each Historical() client call is a separate HTTP request


def resolve_one(client, row):
    start = row["time"]
    end = start + pd.Timedelta(minutes=1)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=["MNQ.c.0"], stype_in="continuous",
        schema="trades", start=start, end=end,
    )
    df = data.to_df()
    if df.empty:
        return "no_trade_data"

    direction = row["direction"]
    fvg_top, fvg_bottom = row["fvg_top"], row["fvg_bottom"]
    sl, tp = row["stop_price"], row["target_price"]

    entry_ts = None
    for ts, price in zip(df.index, df["price"]):
        if fvg_bottom <= price <= fvg_top:
            entry_ts = ts
            break
    if entry_ts is None:
        return "never_entered_this_bar"

    after = df[df.index >= entry_ts]
    for ts, price in zip(after.index, after["price"]):
        hit_sl = (price <= sl) if direction == "long" else (price >= sl)
        hit_tp = (price >= tp) if direction == "long" else (price <= tp)
        if hit_sl:
            return "loss"
        if hit_tp:
            return "win"
    return "defer"  # neither hit within this bar after true entry - next bar decides, same either way


def _worker(api_key, row):
    # separate client per call, not shared across threads - avoids any
    # question of whether the Historical client is safe for concurrent use
    client = db.Historical(api_key)
    try:
        return row["time"], resolve_one(client, row)
    except Exception as e:
        return row["time"], f"error: {e}"


if __name__ == "__main__":
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not found - check your .env file at the project root.")

    trades = pd.read_csv(IN_PATH)
    trades["time"] = pd.to_datetime(trades["time"], utc=True)

    if os.path.exists(OUT_PATH):
        done = pd.read_csv(OUT_PATH)
        done["time"] = pd.to_datetime(done["time"], utc=True)
        done_times = set(done["time"])
        results = done.to_dict("records")
    else:
        done_times = set()
        results = []

    remaining = trades[~trades["time"].isin(done_times)]
    print(f"{len(done_times)} already resolved, {len(remaining)} remaining.")

    rows_by_time = {row["time"]: row for _, row in remaining.iterrows()}
    completed = 0
    # all file writes happen here in the main thread only - workers just
    # return a result, never touch the CSV, so there's no concurrent-write risk
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, api_key, row): ts for ts, row in rows_by_time.items()}
        for future in as_completed(futures):
            ts, true_outcome = future.result()
            row = rows_by_time[ts]
            results.append({**row.to_dict(), "true_outcome": true_outcome})
            completed += 1
            if completed % 25 == 0 or completed == len(remaining):
                pd.DataFrame(results).to_csv(OUT_PATH, index=False)
                print(f"  {completed}/{len(remaining)} resolved and saved.")

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"Done. {len(results)} total trades resolved -> {OUT_PATH}")
