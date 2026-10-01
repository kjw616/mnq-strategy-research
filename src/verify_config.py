# Fully tick-verifies a given (shrink_factor, fixed_r_target) config: finds
# every trade where the entry bar's own range also touches the stop (the
# same-bar-whipsaw ambiguity), checks whether that actually matters (does
# continuing from the next bar onward already agree with "loss"?), and only
# spends on tick data for the trades where it doesn't.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.verify_config <shrink_factor> <fixed_r_target>

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import databento as db
import pandas as pd
from dotenv import load_dotenv

from .data import load_1m_bars
from .liquidity_fvg_strategy import run_liquidity_fvg_strategy
from .report import summarize_outcomes
from .resolve_ambiguous_trades import resolve_one

load_dotenv()
MAX_WORKERS = 6


def touches_sl_entry_bar(row, df):
    if row["time"] not in df.index:
        return False
    bar = df.loc[row["time"]]
    return bar["low"] <= row["stop_price"] if row["direction"] == "long" else bar["high"] >= row["stop_price"]


def continuation_outcome(idx, lows, highs, sl, tp, direction):
    """What happens checking strictly AFTER the entry bar - i.e. the
    resolution the old (pre-fix) code would have used, computed fresh here
    since that code path no longer exists standalone. Operates on raw numpy
    arrays (not DataFrame.iloc, which is far too slow in a per-bar loop)."""
    for i in range(idx + 1, len(lows)):
        if direction == "long":
            if lows[i] <= sl:
                return "loss"
            if highs[i] >= tp:
                return "win"
        else:
            if highs[i] >= sl:
                return "loss"
            if lows[i] <= tp:
                return "win"
    return "open_at_data_end"


def _worker(api_key, row):
    client = db.Historical(api_key)
    try:
        return row["time"], resolve_one(client, row)
    except Exception as e:
        return row["time"], f"error: {e}"


if __name__ == "__main__":
    shrink = float(sys.argv[1])
    rr = float(sys.argv[2])
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not found.")

    df_in = load_1m_bars("data/mnq_1m_2023_2025.parquet")
    df_oos = load_1m_bars("data/mnq_1m_2025_present.parquet")

    t_in, _ = run_liquidity_fvg_strategy(df_in, use_time_filter=False, use_rr_filter=False, fixed_r_target=rr, sl_shrink_factor=shrink)
    t_oos, _ = run_liquidity_fvg_strategy(df_oos, use_time_filter=False, use_rr_filter=False, fixed_r_target=rr, sl_shrink_factor=shrink)
    t = pd.concat([t_in, t_oos], ignore_index=True)
    t = t[t["outcome"].isin(["win", "loss"])].copy()
    t["time"] = pd.to_datetime(t["time"], utc=True)

    def get_df(ts):
        return df_in if ts < pd.Timestamp("2025-01-01", tz="UTC") else df_oos

    # precompute numpy arrays + a fast timestamp->position lookup per dataset once,
    # instead of repeated DataFrame.loc/.get_loc calls per trade
    arrays = {}
    for name, df in [("in", df_in), ("oos", df_oos)]:
        arrays[name] = {
            "lows": df["low"].to_numpy(),
            "highs": df["high"].to_numpy(),
            "pos": {ts: i for i, ts in enumerate(df.index)},
        }

    def which_dataset(ts):
        return "in" if ts < pd.Timestamp("2025-01-01", tz="UTC") else "oos"

    flagged_idx = [i for i, row in t.iterrows() if touches_sl_entry_bar(row, get_df(row["time"]))]
    print(f"{len(flagged_idx)} of {len(t)} trades flagged (entry bar also touches SL).")

    cont = {}
    for i in flagged_idx:
        row = t.loc[i]
        ds = arrays[which_dataset(row["time"])]
        pos = ds["pos"][row["time"]]
        cont[i] = continuation_outcome(pos, ds["lows"], ds["highs"], row["stop_price"], row["target_price"], row["direction"])
    disagree_idx = [i for i in flagged_idx if cont[i] != "loss"]
    print(f"{len(disagree_idx)} of those actually disagree with the assumed loss (need tick verification).")
    print(f"Estimated cost: ~${len(disagree_idx) * 0.0041:.2f}")

    # resumable cache, keyed by exact entry timestamp - so a kill mid-run
    # (or a crash) doesn't throw away tick data already paid for and fetched
    cache_path = f"data/tick_cache_shrink{shrink}_rr{rr}.csv"
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        cache_df["time"] = pd.to_datetime(cache_df["time"], utc=True)
        resolved = dict(zip(cache_df["time"], cache_df["true_outcome"]))
    else:
        resolved = {}
    print(f"{len(resolved)} already cached from a prior run.")

    todo_idx = [i for i in disagree_idx if t.loc[i, "time"] not in resolved]
    print(f"{len(todo_idx)} left to fetch.")

    def save_cache():
        pd.DataFrame({"time": list(resolved.keys()), "true_outcome": list(resolved.values())}).to_csv(cache_path, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, api_key, t.loc[i]): i for i in todo_idx}
        done_n = 0
        for future in as_completed(futures):
            i = futures[future]
            ts, outcome = future.result()
            resolved[ts] = outcome
            done_n += 1
            if done_n % 25 == 0 or done_n == len(todo_idx):
                save_cache()
                print(f"  {done_n}/{len(todo_idx)} tick-resolved and saved.")
    save_cache()
    resolved = {i: resolved[t.loc[i, "time"]] for i in disagree_idx}

    final_outcomes = []
    for i, row in t.iterrows():
        if i in resolved:
            r = resolved[i]
            final_outcomes.append(r if r in ("win", "loss") else cont[i])
        elif i in cont:
            final_outcomes.append(cont[i])
        else:
            final_outcomes.append(row["outcome"])
    t["final_outcome"] = final_outcomes
    t["r_final"] = t["final_outcome"].map({"win": rr, "loss": -1.0})

    decided = t[t["final_outcome"].isin(["win", "loss"])]
    n = len(decided)
    wins = (decided["final_outcome"] == "win").sum()
    total_r = decided["r_final"].sum()
    gross_win = decided.loc[decided["final_outcome"] == "win", "r_final"].sum()
    gross_loss = decided.loc[decided["final_outcome"] == "loss", "r_final"].sum()
    eq = decided["r_final"].cumsum()
    dd = (eq.cummax() - eq).max()

    print()
    print(f"=== TICK-VERIFIED: shrink={shrink} R:R={rr} ===")
    print(f"n={n}  win_rate={wins/n*100:.1f}%  avg_R={total_r/n:.3f}  total_R={total_r:.2f}  PF={gross_win/abs(gross_loss):.2f}  max_DD={dd:.2f}R")

    out_path = f"data/verified_shrink{shrink}_rr{rr}_trades.csv"
    t.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
