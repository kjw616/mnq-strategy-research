# Pulls MNQ 1-minute bars for the out-of-sample period NOT covered by
# data/mnq_1m_2023_2025.parquet (which despite its name only runs through
# 2024-12-31). This is a real, billed download (~$2 as of this writing) -
# run it once, then reuse the local file.
#
# Run it yourself:
#   source venv/bin/activate
#   python src/download_mnq_oos_data.py

import os
import sys

import databento as db
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DATABENTO_API_KEY")

if not api_key:
    raise SystemExit("DATABENTO_API_KEY not found - check your .env file at the project root.")

client = db.Historical(api_key)

dataset = "GLBX.MDP3"
symbol = "MNQ.c.0"  # continuously-stitched front-month contract
schema = "ohlcv-1m"
start = "2025-01-01"
end = "2026-07-28"  # today - Databento serves up through the current day

cost = client.metadata.get_cost(
    dataset=dataset,
    symbols=[symbol],
    stype_in="continuous",
    schema=schema,
    start=start,
    end=end,
)
print(f"About to download {start} to {end} of MNQ 1-minute bars. Estimated cost: ${cost:.2f}")

if "--yes" not in sys.argv:
    input("Press Enter to confirm and proceed, or Ctrl+C to cancel... ")

data = client.timeseries.get_range(
    dataset=dataset,
    symbols=[symbol],
    stype_in="continuous",
    schema=schema,
    start=start,
    end=end,
)

df = data.to_df()
print(f"Downloaded {len(df)} 1-minute bars.")

out_path = "data/mnq_1m_2025_present.parquet"
df.to_parquet(out_path)
print(f"Saved to {out_path}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
