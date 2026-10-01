# Grid search over EMA fast/slow period PAIRS - every prior EMA-cross file
# in this repo (ema_cross_strategy.py, ema_cross_grid_search.py,
# ema_cross_multi_tf_backtest.py, validate_ema_cross.py) fixed the pair at
# 9/21 and swept everything else (timeframe, stop/target, filters) around
# it. This is the first thing in the repo that actually varies the pair
# itself.
#
# Reuses find_cross_candidates() unchanged (same event definition every
# other EMA-cross file already trades/validates - a cross within
# window_minutes of the NY open) and validate_ema_cross.py's own
# methodology - pure directional accuracy (does price keep moving that way
# more than a coin flip, at a range of horizons), decoupled from any
# specific stop/target mechanics, since the ask here is "which pair works
# best as a confluence read," not "which pair trades best standalone."
#
# Selection discipline matches ema_cross_grid_search.py exactly: pairs are
# RANKED using IN-SAMPLE (2023-2024) hit rate only, at a single primary
# horizon. OOS (2025-present) numbers for the resulting top picks are
# reported afterward as a genuine holdout check, never used to choose among
# pairs - a wide grid is exactly how you'd otherwise accidentally p-hack a
# 20-pair search into an overfit ranking.
#
# Run it yourself:
#   source venv/bin/activate
#   python -m src.ema_cross_pair_grid_search

import math

import pandas as pd

from .data import load_1m_bars
from .ema_cross_strategy import find_cross_candidates
from .validate_ema_cross import _binom_two_sided_p, evaluate

IN_SAMPLE_PATH = "data/mnq_1m_2023_2025.parquet"
OOS_PATH = "data/mnq_1m_2025_present.parquet"

FAST_CANDIDATES = [5, 8, 9, 12, 20]
SLOW_CANDIDATES = [13, 21, 26, 34, 50]
MIN_SLOW_FAST_RATIO = 1.6  # skip near-degenerate pairs like 20/21

PRIMARY_HORIZON_HOURS = 2.0   # what pairs get ranked by
DETAIL_HORIZONS = [0.5, 1.0, 2.0, 4.0, 6.5]  # shown in full for the top picks
MIN_N = 100  # a pair with fewer candidate events than this isn't worth ranking


def build_pairs() -> list[tuple[int, int]]:
    pairs = []
    for f in FAST_CANDIDATES:
        for s in SLOW_CANDIDATES:
            if s / f >= MIN_SLOW_FAST_RATIO:
                pairs.append((f, s))
    return pairs


def score_pair(df_1m: pd.DataFrame, fast: int, slow: int, horizon: float) -> dict:
    candidates = find_cross_candidates(
        df_1m, fast_len=fast, slow_len=slow,
        require_vwap_confluence=False, require_htf_bias_confluence=False,
    )
    if candidates.empty:
        return {"n": 0, "hit_rate": float("nan"), "p": float("nan"), "avg_ret_long": float("nan"), "avg_ret_short": float("nan")}

    evaluated = evaluate(df_1m, candidates)
    col = f"fwd_ret_{horizon}h"
    sub = evaluated.dropna(subset=[col])
    n = len(sub)
    if n == 0:
        return {"n": 0, "hit_rate": float("nan"), "p": float("nan"), "avg_ret_long": float("nan"), "avg_ret_short": float("nan")}

    is_long = sub["direction"] == "long"
    correct = ((is_long) & (sub[col] > 0)) | ((~is_long) & (sub[col] < 0))
    hits = int(correct.sum())
    hit_rate = hits / n
    p = _binom_two_sided_p(hits, n, 0.5)
    avg_ret_long = sub.loc[is_long, col].mean() if is_long.any() else float("nan")
    avg_ret_short = sub.loc[~is_long, col].mean() if (~is_long).any() else float("nan")
    return {"n": n, "hit_rate": hit_rate, "p": p, "avg_ret_long": avg_ret_long, "avg_ret_short": avg_ret_short}


def full_detail(df_1m: pd.DataFrame, fast: int, slow: int, label: str) -> None:
    candidates = find_cross_candidates(
        df_1m, fast_len=fast, slow_len=slow,
        require_vwap_confluence=False, require_htf_bias_confluence=False,
    )
    evaluated = evaluate(df_1m, candidates)
    print(f"  {label} ({fast}/{slow}, n={len(evaluated)} candidate events):")
    if evaluated.empty:
        print("    No candidate events.")
        return
    for h in DETAIL_HORIZONS:
        col = f"fwd_ret_{h}h"
        sub = evaluated.dropna(subset=[col])
        n = len(sub)
        if n == 0:
            print(f"    {h:>4.1f}h  no samples")
            continue
        is_long = sub["direction"] == "long"
        correct = ((is_long) & (sub[col] > 0)) | ((~is_long) & (sub[col] < 0))
        hit_rate = correct.mean()
        up_frac = (sub[col] > 0).mean()
        p = _binom_two_sided_p(int(correct.sum()), n, 0.5)
        print(f"    {h:>4.1f}h  n={n:4d}  hit_rate={hit_rate*100:5.1f}%  vs 50%: p={p:.4f}  "
              f"'always long' baseline={up_frac*100:5.1f}%")


if __name__ == "__main__":
    df_in = load_1m_bars(IN_SAMPLE_PATH)
    df_oos = load_1m_bars(OOS_PATH)

    pairs = build_pairs()
    print(f"Testing {len(pairs)} fast/slow pairs, ranked on in-sample @ {PRIMARY_HORIZON_HOURS}h horizon\n")

    rows = []
    for fast, slow in pairs:
        s_in = score_pair(df_in, fast, slow, PRIMARY_HORIZON_HOURS)
        if s_in["n"] < MIN_N:
            continue
        rows.append({"fast": fast, "slow": slow, **{f"in_{k}": v for k, v in s_in.items()}})

    ranked = pd.DataFrame(rows).sort_values("in_hit_rate", ascending=False).reset_index(drop=True)

    print("=" * 88)
    print(f"IN-SAMPLE RANKING @ {PRIMARY_HORIZON_HOURS}h (selection made here, OOS untouched until after)")
    print("=" * 88)
    print(f"{'pair':>8}{'n':>7}{'hit_rate':>11}{'p-value':>10}{'avg_ret_long':>15}{'avg_ret_short':>16}")
    for _, r in ranked.iterrows():
        print(f"{r['fast']:>3.0f}/{r['slow']:<4.0f}{r['in_n']:>7.0f}{r['in_hit_rate']*100:>10.1f}%"
              f"{r['in_p']:>10.4f}{r['in_avg_ret_long']:>+15.2f}{r['in_avg_ret_short']:>+16.2f}")

    print(f"\n{'9/21 (current default)':^88}")
    cur = ranked[(ranked["fast"] == 9) & (ranked["slow"] == 21)]
    if not cur.empty:
        rank_pos = ranked.index[(ranked["fast"] == 9) & (ranked["slow"] == 21)][0] + 1
        print(f"  rank {rank_pos} of {len(ranked)}, hit_rate={cur.iloc[0]['in_hit_rate']*100:.1f}%")

    top_n = 5
    top_pairs = list(zip(ranked.head(top_n)["fast"].astype(int), ranked.head(top_n)["slow"].astype(int)))
    if (9, 21) not in top_pairs:
        top_pairs.append((9, 21))  # always show the current default for comparison, even if it didn't rank top

    print("\n" + "=" * 88)
    print(f"OOS HOLDOUT CHECK - top {top_n} in-sample pairs + current 9/21, full horizon detail")
    print("=" * 88)
    for fast, slow in top_pairs:
        print()
        full_detail(df_in, fast, slow, "IN-SAMPLE 2023-2024")
        full_detail(df_oos, fast, slow, "OOS 2025-present    ")
