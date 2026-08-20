"""
collect/batters.py
===========================
Build per-batter rolling offensive stats from Statcast PA events.

Produces ONE file: data/batter_rolling_stats.csv — one row per
(batter, game) with rolling and season-to-date rates, all lagged via
shift(1) so a row never contains information from its own game.

    obp_l15 / obp_l30 / obp_season                   on-base rate
    k_rate_l15 / k_rate_l30 / k_rate_season          strikeout rate
    bb_rate_l15 / bb_rate_l30 / bb_rate_season       walk rate
    hard_hit_rate_l15 / _l30 / _season               95+ mph EV per ball in play
    barrel_rate_l15 / _l30 / _season                 barrels per ball in play
    ops_proxy_l15 / ops_proxy_l30                    OBP x 1.5

features.py consumes this in build_lineup_quality_features()
to give the hits and walks models batter-side signal that isn't just K rate
— how hard the opposing lineup actually hits the ball, and what on-base
rate it's running.

INPUT
-----
    data/statcast_pa_events_all.csv   (from collect/statcast.py)

USAGE
-----
    python run.py collect batters            # build if not cached
    python run.py collect batters --force    # rebuild from scratch

The result is cached. Without --force an existing
data/batter_rolling_stats.csv is left alone, so re-running after a data
refresh is a no-op unless you ask for a rebuild — run with --force after
refresh.py or these stats silently freeze at their build date.

Takes ~1-2 minutes for six seasons.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_model.paths import DATA_DIR as OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")



def build_batter_rolling_stats(force=False):
    """
    From statcast_pa_events_all.csv, compute per-batter rolling stats
    that can be joined to their actual game lineup appearances.

    For each batter-game, computes:
      - Rolling 15/30-game OBP proxy (H + BB + HBP) / PA
      - Rolling K% and BB%
      - Rolling hard-hit and barrel rate (from batted ball events)
      - Season-to-date aggregates

    Every window is shifted by one game before aggregating, so a batter's
    row for a given game reflects only prior games — no leakage.

    Returns a DataFrame with one row per batter per game.
    """
    print("\n═══ Building Per-Batter Rolling Stats from Statcast ═══")

    cache_file = OUTPUT_DIR / "batter_rolling_stats.csv"
    if cache_file.exists() and not force:
        print("  Loading from cache...")
        df = pd.read_csv(cache_file)
        print(f"  ✓ {len(df):,} batter-game records loaded "
              f"({df['batter'].nunique():,} batters)")
        if "game_date" in df.columns:
            through = pd.to_datetime(df["game_date"]).max().date()
            print(f"    Cached through {through}. "
                  f"Re-run with --force to rebuild from current data.")
        return df

    pa_path = OUTPUT_DIR / "statcast_pa_events_all.csv"
    if not pa_path.exists():
        print(f"  ✗ {pa_path} not found — run collect/statcast.py first.")
        return pd.DataFrame()

    print("  Loading Statcast PA events...")
    pa = pd.read_csv(pa_path)
    pa["game_date"] = pd.to_datetime(pa["game_date"])

    required_cols = ["batter", "game_pk", "game_date",
                     "was_K", "was_BB", "was_HBP", "was_hit"]
    missing = [c for c in required_cols if c not in pa.columns]
    if missing:
        print(f"  ⚠ Missing columns: {missing}. Rebuilding PA event flags...")
        if "events" in pa.columns:
            pa["was_K"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
            pa["was_BB"] = pa["events"].isin(["walk", "intent_walk"]).astype(int)
            pa["was_HBP"] = (pa["events"] == "hit_by_pitch").astype(int)
            pa["was_hit"] = pa["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
            pa["was_in_play"] = (~pa["events"].isin([
                "strikeout", "strikeout_double_play", "walk", "intent_walk", "hit_by_pitch"
            ])).astype(int)
        else:
            print("  ✗ Cannot rebuild flags — 'events' column missing.")
            return pd.DataFrame()

    # Hard-hit / barrel flags
    if "launch_speed" in pa.columns:
        pa["was_hard_hit"] = (pa["launch_speed"] >= 95).fillna(False).astype(int)
        pa["was_barrel"] = (
            (pa["launch_speed"] >= 98) & (pa["launch_angle"].between(26, 30))
        ).fillna(False).astype(int)
    else:
        pa["was_hard_hit"] = 0
        pa["was_barrel"] = 0

    pa["was_on_base"] = (pa["was_hit"] | pa["was_BB"] | pa["was_HBP"]).astype(int)
    pa["was_pa"] = 1  # Every row is already a PA-terminating event

    print(f"  {len(pa):,} plate appearances across {pa['batter'].nunique():,} batters")

    # ── Aggregate to batter-game level ──────────────────────────────────
    print("  Aggregating to batter-game level...")
    grp_cols = ["batter", "game_pk", "game_date"]
    if "stand" in pa.columns:
        grp_cols.append("stand")
    if "home_team" in pa.columns:
        grp_cols += ["home_team", "away_team"]

    batter_game = pa.groupby(grp_cols).agg(
        pa_count=("was_pa", "sum"),
        k_count=("was_K", "sum"),
        bb_count=("was_BB", "sum"),
        hbp_count=("was_HBP", "sum"),
        hit_count=("was_hit", "sum"),
        on_base_count=("was_on_base", "sum"),
        hard_hit_count=("was_hard_hit", "sum"),
        barrel_count=("was_barrel", "sum"),
        in_play_count=("was_in_play", "sum") if "was_in_play" in pa.columns else ("was_pa", "sum"),
    ).reset_index()

    batter_game["obp_game"] = batter_game["on_base_count"] / batter_game["pa_count"].replace(0, np.nan)
    batter_game["k_rate_game"] = batter_game["k_count"] / batter_game["pa_count"].replace(0, np.nan)
    batter_game["bb_rate_game"] = batter_game["bb_count"] / batter_game["pa_count"].replace(0, np.nan)

    in_play = batter_game["in_play_count"].replace(0, np.nan)
    batter_game["hard_hit_rate_game"] = batter_game["hard_hit_count"] / in_play
    batter_game["barrel_rate_game"] = batter_game["barrel_count"] / in_play

    batter_game = batter_game.sort_values(["batter", "game_date"])

    # ── Rolling + season-to-date windows (all shift(1) lagged) ──────────
    print("  Computing rolling averages...")

    def rolling_avg(series, n, min_p=3):
        return series.shift(1).rolling(n, min_periods=min_p).mean()

    metrics = ["obp_game", "k_rate_game", "bb_rate_game",
               "hard_hit_rate_game", "barrel_rate_game"]

    for metric in metrics:
        base_name = metric.replace("_game", "")
        grp = batter_game.groupby("batter")[metric]
        batter_game[f"{base_name}_l15"] = grp.transform(lambda x: rolling_avg(x, 15, 5))
        batter_game[f"{base_name}_l30"] = grp.transform(lambda x: rolling_avg(x, 30, 10))
        batter_game[f"{base_name}_season"] = batter_game.groupby(
            ["batter", batter_game["game_date"].dt.year]
        )[metric].transform(lambda x: x.shift(1).expanding(min_periods=5).mean())

    # OPS proxy (SLG is hard without pitch-level data, use OBP*1.5 as proxy)
    batter_game["ops_proxy_l15"] = batter_game["obp_l15"] * 1.5
    batter_game["ops_proxy_l30"] = batter_game["obp_l30"] * 1.5

    batter_game.to_csv(cache_file, index=False)
    print(f"  ✓ Batter rolling stats: {len(batter_game):,} batter-game records")
    print(f"    {batter_game['batter'].nunique():,} unique batters")
    print(f"    Through {batter_game['game_date'].max().date()}")
    print(f"    Saved to {cache_file}")
    return batter_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-batter rolling stats from Statcast PA events.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if data/batter_rolling_stats.csv "
                             "already exists (use after a data refresh)")
    args = parser.parse_args()

    df = build_batter_rolling_stats(force=args.force)

    if len(df) > 0:
        print("\nNext: python run.py features")
