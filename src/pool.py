# Shared query helper for the event-log pools liquidity.build_swing_
# liquidity_pool() and poi.build_fvg_pool() both produce: one row per
# resting item, with a start_time (confirmed/formed) and an end_time (swept/
# mitigated/expired, or None if still live as of the pool's last bar).

import pandas as pd


def pool_as_of(pool: pd.DataFrame, t: pd.Timestamp, max_size: int | None = None) -> pd.DataFrame:
    """Rows still live at time t: started at or before t, and not yet ended
    (or ended strictly after t) - i.e. what was actually "resting" at that
    moment, live or historical. max_size, if given, keeps only the
    most-recently-started rows (a display cap, not a filter on which items
    exist)."""
    live = pool[(pool["start_time"] <= t) & (pool["end_time"].isna() | (pool["end_time"] > t))]
    live = live.sort_values("start_time")
    if max_size is not None and len(live) > max_size:
        live = live.tail(max_size)
    return live
