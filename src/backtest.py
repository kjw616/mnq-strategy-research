# Resolves each strategy signal into a win/loss/breakeven outcome using the
# rules from the mentoring conversation:
#   - stop-loss  = the MSS swing point (already recorded per-trade)
#   - target     = nearest untapped opposing swing level (already recorded)
#   - breakeven  = once price moves `breakeven_r` x the initial risk in your
#                  favor, the stop moves to entry price
#
# Simplification worth knowing: 1-minute bars can't tell us which of two
# price levels was touched first if both fall inside the same bar's range.
# When that happens we conservatively assume the stop-side event happened
# first - this never gives the strategy credit it might not have earned.

import numpy as np
import pandas as pd


def resolve_trade_outcomes(trades: pd.DataFrame, df_1m: pd.DataFrame, breakeven_r: float = 1.0) -> pd.DataFrame:
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    idx = df_1m.index
    n = len(df_1m)

    outcomes, exit_times, exit_prices, r_multiples = [], [], [], []

    for row in trades.itertuples(index=False):
        entry_price = row.entry_price
        stop_price = row.stop_price
        target_price = row.target_price
        bar_index = row.bar_index
        is_long = row.direction == "long"

        if pd.isna(stop_price) or pd.isna(target_price) or stop_price == entry_price:
            outcomes.append("no_target")
            exit_times.append(pd.NaT)
            exit_prices.append(np.nan)
            r_multiples.append(np.nan)
            continue

        risk = abs(entry_price - stop_price)
        breakeven_trigger = entry_price + breakeven_r * risk if is_long else entry_price - breakeven_r * risk
        effective_stop = stop_price
        breakeven_hit = False

        outcome = exit_time = exit_price = None
        j = bar_index + 1
        while j < n:
            h, l = highs[j], lows[j]

            if not breakeven_hit:
                if (is_long and h >= breakeven_trigger) or (not is_long and l <= breakeven_trigger):
                    breakeven_hit = True
                    effective_stop = entry_price

            stop_touched = (l <= effective_stop) if is_long else (h >= effective_stop)
            if stop_touched:
                outcome = "breakeven" if breakeven_hit else "loss"
                exit_time, exit_price = idx[j], effective_stop
                break

            target_touched = (h >= target_price) if is_long else (l <= target_price)
            if target_touched:
                outcome = "win"
                exit_time, exit_price = idx[j], target_price
                break

            j += 1

        if outcome is None:
            outcome = "open_at_data_end"
            exit_time, exit_price = idx[-1], closes[-1]

        r = (exit_price - entry_price) / risk if is_long else (entry_price - exit_price) / risk

        outcomes.append(outcome)
        exit_times.append(exit_time)
        exit_prices.append(exit_price)
        r_multiples.append(r)

    result = trades.copy()
    result["outcome"] = outcomes
    result["exit_time"] = exit_times
    result["exit_price"] = exit_prices
    result["r_multiple"] = r_multiples
    return result
