# One-off sanity check: confirms our Databento API key works, shows what a
# small MNQ data pull costs, then downloads that small sample so we have real
# candles to look at. Not part of the final platform - just a wiring test.

import os

import databento as db
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DATABENTO_API_KEY")

if not api_key:
    raise SystemExit("DATABENTO_API_KEY not found - check your .env file at the project root.")

client = db.Historical(api_key)

# Free call - just proves the key authenticates, no cost.
datasets = client.metadata.list_datasets()
print(f"Connected. Key is valid. {len(datasets)} datasets available on your plan.")

# MNQ.c.0 = the continuously-stitched front-month MNQ contract, so we don't
# have to manually track contract roll dates (MNQZ5 -> MNQH6 etc.) ourselves.
dataset = "GLBX.MDP3"
symbol = "MNQ.c.0"
schema = "ohlcv-1m"
start = "2024-01-02"
end = "2024-01-03"

cost = client.metadata.get_cost(
    dataset=dataset,
    symbols=[symbol],
    stype_in="continuous",
    schema=schema,
    start=start,
    end=end,
)
print(f"Estimated cost for 1 day of MNQ 1-minute bars: ${cost:.4f}")

data = client.timeseries.get_range(
    dataset=dataset,
    symbols=[symbol],
    stype_in="continuous",
    schema=schema,
    start=start,
    end=end,
)

df = data.to_df()
print(f"\nDownloaded {len(df)} 1-minute bars for {symbol}.")
print(df.head())

out_path = "data/mnq_test_sample.csv"
df.to_csv(out_path)
print(f"\nSaved to {out_path}")
