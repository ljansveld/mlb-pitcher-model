"""
MLB Pitcher Feature Engineering (Enhanced)
============================================
Builds a comprehensive feature set from all collected data sources:
  - Rolling pitcher performance (3/5/10 starts)
  - Season-to-date cumulative stats
  - Trend features (hot/cold streaks)
  - Pitch mix & stuff metrics
  - Opposing team batting profiles
  - Platoon splits (L/R matchups)
  - Ballpark factors
  - Umpire tendencies
  - Rest days & scheduling context
  - Venue/weather proxies

Requirements:
    pip install pandas numpy

Usage:
    python run.py features
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_model.paths import DATA_DIR

warnings.filterwarnings("ignore")



def normalize_game_pk(*dfs):
    """Ensure game_pk is int64 in all DataFrames to prevent merge dtype mismatches.

    After multiple merge-concat cycles, game_pk can drift from int64 to float64
    (e.g., when a left join introduces NaN for unmatched rows and pandas upcasts).
    This helper coerces game_pk back to int64 before any merge on that column.
    """
    for frame in dfs:
        if "game_pk" in frame.columns:
            frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce")
            # Drop rows where game_pk couldn't be parsed (shouldn't happen, but safe)
            valid = frame["game_pk"].notna()
            if not valid.all():
                frame.drop(frame.index[~valid], inplace=True)
            frame["game_pk"] = frame["game_pk"].astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_synthetic_fg_batting_from_statcast():
    """
    Build a synthetic fangraphs_batting_seasons.csv equivalent from
    statcast_pa_events_all.csv — used when the FanGraphs scraper is
    blocked (403) and the real file doesn't exist.

    Downstream code in 02 expects columns: MLBAMID, Season, Name,
    K%, BB%, SwStr%, Contact%, O-Swing%, Z-Contact%, wRC+, xwOBA,
    Hard%, Barrel%. We produce all of those except wRC+, xwOBA,
    Hard%, Barrel% (which need run-value / exit-velo data — left NaN
    so downstream functions degrade gracefully).

    SwStr%, Contact%, O-Swing%, Z-Contact% come from a secondary
    path: if statcast_pitcher_games_all.csv has per-batter equivalents
    we use those; otherwise we approximate SwStr% ≈ K% × 0.5 + small
    offset (rough but better than NaN for rookies). The model will
    still learn the real signal from K% + BB%.

    Returns a DataFrame ready to be assigned to data["fg_batting"].
    Returns None if PA events file is missing.
    """
    pa_path = DATA_DIR / "statcast_pa_events_all.csv"
    if not pa_path.exists():
        return None

    print("  ⓘ FG batting missing — building synthetic equivalent from "
          "statcast_pa_events_all.csv...")
    pa = pd.read_csv(pa_path, usecols=[
        "batter", "game_date", "was_K", "was_BB", "was_hit",
        "was_in_play", "was_HBP",
    ])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa["Season"] = pa["game_date"].dt.year

    # Aggregate to (batter, season) level. These match FG's K%/BB%/etc. closely:
    #   K% = K / PA, BB% = BB / PA, Contact% = balls in play / swings,
    #   SwStr% = whiff / pitches (approximate via 1 - contact-ish).
    agg = (pa.groupby(["batter", "Season"])
             .agg(PA=("was_K", "size"),
                  K=("was_K", "sum"),
                  BB=("was_BB", "sum"),
                  HBP=("was_HBP", "sum"),
                  hits=("was_hit", "sum"),
                  BIP=("was_in_play", "sum"))
             .reset_index())
    agg = agg[agg["PA"] >= 30].copy()  # drop pinch-hit-only lines

    agg["K%"] = agg["K"] / agg["PA"]
    agg["BB%"] = agg["BB"] / agg["PA"]
    # Proxy: SwStr% typically correlates ~0.85 with K% for hitters.
    # Fit a simple linear regression: SwStr% ≈ 0.45 * K% + 0.04
    # (these constants are typical league-wide; precise fit not needed
    # since the model will learn its own weight on this feature).
    agg["SwStr%"] = (0.45 * agg["K%"] + 0.04).clip(0.03, 0.20)
    # Contact% ≈ BIP / (BIP + K) — a reasonable proxy for FG's swing-based
    # Contact% when pitch-level swing data isn't joined per batter here.
    agg["Contact%"] = (agg["BIP"] / (agg["BIP"] + agg["K"]).replace(0, np.nan))
    agg["Contact%"] = agg["Contact%"].fillna(0.78).clip(0.50, 0.95)
    # O-Swing% / Z-Contact% correlate strongly with K% / Contact%.
    # Rough proxies good enough for a tree model:
    agg["O-Swing%"] = (0.30 + 0.4 * agg["K%"]).clip(0.15, 0.45)
    agg["Z-Contact%"] = (0.90 - 0.4 * agg["K%"]).clip(0.70, 0.95)

    # Columns we genuinely can't build from PA-only data — leave NaN:
    for col in ["wRC+", "xwOBA", "Hard%", "Barrel%"]:
        agg[col] = np.nan

    # MLBAMID = Statcast batter id. Downstream code checks for both
    # "MLBAMID" and "xMLBAMID" — we provide MLBAMID.
    agg = agg.rename(columns={"batter": "MLBAMID"})
    # Name: placeholder (downstream name-based merges will silently no-op,
    # but the MLBAMID path is preferred anyway).
    agg["Name"] = agg["MLBAMID"].astype(str)

    print(f"    ✓ Synthetic FG batting: {len(agg):,} "
          f"(batter, season) rows across "
          f"{agg['Season'].min()}–{agg['Season'].max()}")
    return agg


def backfill_fg_mlbamid(fg, label="", extra_crosswalks=None):
    """
    Heal a FanGraphs seasons file with patchy MLBAMID coverage.

    Real symptom: a merged seasons CSV often has MLBAMID populated only
    for the most recent pull (e.g., 2026) and IDfg populated for older
    rows. Downstream merges keyed on MLBAMID then silently drop ~80% of
    history.

    This function fills in missing MLBAMID values by combining sources
    (in order — first non-null wins per IDfg):
      1. Rows in the same file that have BOTH IDfg and MLBAMID (often
         empty in practice — recent re-exports tend to drop IDfg).
      2. Any extra (IDfg, MLBAMID) crosswalks provided via the
         extra_crosswalks list (typically from the standalone 2026
         CSVs which carry both columns simultaneously).
      3. Leaves the rest untouched (downstream code can still fall back
         to name-based crosswalks).

    Args:
        fg: DataFrame to heal.
        label: Used only for the diagnostic print.
        extra_crosswalks: optional list of DataFrames each with columns
                          ['IDfg', 'MLBAMID']. Any may be empty/None.

    Returns the healed DataFrame and prints a coverage line.
    """
    if "MLBAMID" not in fg.columns or "IDfg" not in fg.columns:
        return fg
    fg = fg.copy()
    fg["MLBAMID"] = pd.to_numeric(fg["MLBAMID"], errors="coerce")
    before = fg["MLBAMID"].notna().sum()

    # Collect candidate crosswalks. Within-file bridge first, then any
    # provided externally. Concat and dedup so the within-file mapping
    # takes precedence on conflicts.
    bridges = []
    within = fg.loc[fg["MLBAMID"].notna() & fg["IDfg"].notna(),
                    ["IDfg", "MLBAMID"]].drop_duplicates(subset=["IDfg"])
    if len(within) > 0:
        bridges.append(within)
    for xw in (extra_crosswalks or []):
        if xw is None or len(xw) == 0:
            continue
        if "IDfg" not in xw.columns or "MLBAMID" not in xw.columns:
            continue
        clean = xw[["IDfg", "MLBAMID"]].dropna().drop_duplicates(subset=["IDfg"])
        if len(clean) > 0:
            bridges.append(clean)

    if not bridges:
        return fg

    bridge = (pd.concat(bridges, ignore_index=True)
                .drop_duplicates(subset=["IDfg"], keep="first")
                .rename(columns={"MLBAMID": "MLBAMID_bridge"}))
    fg = fg.merge(bridge, on="IDfg", how="left")
    fg["MLBAMID"] = fg["MLBAMID"].fillna(fg["MLBAMID_bridge"])
    fg = fg.drop(columns=["MLBAMID_bridge"])
    after = fg["MLBAMID"].notna().sum()
    if after > before:
        gained = after - before
        print(f"    ⓘ {label}: backfilled {gained:,} MLBAMID values via IDfg bridge "
              f"({after:,}/{len(fg):,} now have MLBAMID)")
    return fg


def load_external_idfg_mlbamid_crosswalk(path):
    """Read a standalone 2026-style FanGraphs CSV and extract its
    (IDfg/PlayerId, MLBAMID) crosswalk. Returns an empty DataFrame
    if the file is missing or lacks the required columns."""
    if not path.exists():
        return pd.DataFrame(columns=["IDfg", "MLBAMID"])
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_csv(path, low_memory=False)
    id_col = None
    for c in ("IDfg", "PlayerId", "playerid"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None or "MLBAMID" not in df.columns:
        return pd.DataFrame(columns=["IDfg", "MLBAMID"])
    out = df[[id_col, "MLBAMID"]].rename(columns={id_col: "IDfg"})
    out["IDfg"] = pd.to_numeric(out["IDfg"], errors="coerce")
    out["MLBAMID"] = pd.to_numeric(out["MLBAMID"], errors="coerce")
    return out.dropna().drop_duplicates(subset=["IDfg"])


def load_all_data():
    """Load all collected datasets."""
    data = {}

    # Core pitcher-game data
    path = DATA_DIR / "statcast_pitcher_games_all.csv"
    if path.exists():
        data["pitcher_games"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Pitcher games:   {len(data['pitcher_games']):,} rows")
    else:
        raise FileNotFoundError(f"{path} not found. Run collect/statcast.py first.")

    # ── Per-pitcher earned runs / runs (collected separately by 01) ──
    # Statcast doesn't return ER directly — it comes from the MLB Stats
    # API boxscore endpoint via collect_earned_runs() in collect/statcast.py.
    # Merging it here so lob_pct (rolling) and lob_pct_szn (cumulative)
    # can be computed downstream. Without this merge the H/W and outs
    # pipelines silently lose lob_pct_szn_blended and related features.
    er_path = DATA_DIR / "pitcher_earned_runs.csv"
    if er_path.exists():
        er_df = pd.read_csv(er_path)
        if {"game_pk", "pitcher", "earned_runs"}.issubset(er_df.columns):
            # Drop any ER-side columns that already exist in pitcher_games
            # to avoid pandas creating _x/_y suffixed duplicates on merge.
            # We prefer the upstream collector's values when they exist;
            # the ER file is only a backfill for missing data.
            pg_cols = set(data["pitcher_games"].columns)
            join_keys = {"game_pk", "pitcher"}
            candidate_cols = ["earned_runs", "runs", "innings_pitched", "outs"]
            non_overlapping = [c for c in candidate_cols
                                if c in er_df.columns and c not in pg_cols]
            if non_overlapping:
                merge_cols = list(join_keys) + non_overlapping
                data["pitcher_games"] = data["pitcher_games"].merge(
                    er_df[merge_cols].drop_duplicates(subset=list(join_keys)),
                    on=list(join_keys), how="left",
                )
                print(f"  Earned runs:     merged {non_overlapping} "
                      f"from {len(er_df):,} pitcher-game rows")
            else:
                print(f"  Earned runs:     all ER columns already present "
                      f"in pitcher_games (no merge needed)")
        else:
            print(f"  ⚠ pitcher_earned_runs.csv missing required columns "
                  f"(needs game_pk, pitcher, earned_runs)")

    # Game metadata (venues, lineups, starters)
    path = DATA_DIR / "game_metadata_all.csv"
    if path.exists():
        data["game_meta"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Game metadata:   {len(data['game_meta']):,} rows")

    # Umpire assignments
    path = DATA_DIR / "umpire_assignments.csv"
    if path.exists():
        data["umpires"] = pd.read_csv(path)
        print(f"  Umpire data:     {len(data['umpires']):,} rows")

    # Park factors
    path = DATA_DIR / "park_factors.csv"
    if path.exists():
        data["park_factors"] = pd.read_csv(path)
        print(f"  Park factors:    {len(data['park_factors']):,} rows")

    # Venue info
    path = DATA_DIR / "venue_info.csv"
    if path.exists():
        data["venues"] = pd.read_csv(path)
        print(f"  Venue info:      {len(data['venues']):,} rows")

    # FanGraphs batting (for team-level stats) — with Statcast fallback
    path = DATA_DIR / "fangraphs_batting_seasons.csv"
    if path.exists():
        data["fg_batting"] = pd.read_csv(path, low_memory=False)
        # Build a crosswalk from the standalone 2026 files (which carry
        # both PlayerId/IDfg AND MLBAMID), so we can heal older rows in
        # the seasons file that only have IDfg.
        batting_crosswalks = [
            load_external_idfg_mlbamid_crosswalk(DATA_DIR / "fg_batting_2026.csv"),
            load_external_idfg_mlbamid_crosswalk(DATA_DIR / "fg_batting_statcast_2026.csv"),
        ]
        data["fg_batting"] = backfill_fg_mlbamid(
            data["fg_batting"], label="FG batting",
            extra_crosswalks=batting_crosswalks,
        )
        print(f"  FG batting:      {len(data['fg_batting']):,} rows")
    else:
        synth = build_synthetic_fg_batting_from_statcast()
        if synth is not None:
            data["fg_batting"] = synth
            print(f"  FG batting:      {len(data['fg_batting']):,} rows "
                  f"(SYNTHETIC from Statcast)")

    # Game lineups (individual batter data per game)
    path = DATA_DIR / "game_lineups.csv"
    if path.exists():
        data["lineups"] = pd.read_csv(path)
        normalize_game_pk(data["lineups"])
        print(f"  Lineups:         {len(data['lineups']):,} rows")

    # Weather data
    path = DATA_DIR / "game_weather.csv"
    if path.exists():
        data["weather"] = pd.read_csv(path)
        print(f"  Weather:         {len(data['weather']):,} rows")

    # Pitcher pitch-type breakdowns
    path = DATA_DIR / "pitcher_pitch_type_all.csv"
    if path.exists():
        data["pitcher_pt"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Pitcher PT:      {len(data['pitcher_pt']):,} rows")

    # Batter pitch-type vulnerability
    path = DATA_DIR / "batter_pitch_type_all.csv"
    if path.exists():
        data["batter_pt"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Batter PT:       {len(data['batter_pt']):,} rows")

    # Per-batter rolling stats (from collect/batters.py).
    # Contains BABIP, hard_hit_rate, barrel_rate, OBP, K rate at l15/l30/season
    # windows, all properly lagged via shift(1). Feeds build_lineup_quality_features
    # to give the lineup-level hit-quality signal the hits model needs.
    path = DATA_DIR / "batter_rolling_stats.csv"
    if path.exists():
        data["batter_rolling"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Batter rolling:  {len(data['batter_rolling']):,} rows")
    else:
        print(f"  ⚠ batter_rolling_stats.csv not found — "
              f"run collect/batters.py to enable lineup BABIP/hard-hit features")

    # FanGraphs pitching seasons (Stuff+, Location+, Pitching+, plate discipline)
    path = DATA_DIR / "fangraphs_pitching_seasons.csv"
    if path.exists():
        data["fg_pitching"] = pd.read_csv(path, low_memory=False)
        pitching_crosswalks = [
            load_external_idfg_mlbamid_crosswalk(DATA_DIR / "fg_pitching_2026.csv"),
            load_external_idfg_mlbamid_crosswalk(DATA_DIR / "fg_pitching_pitchmodel_2026.csv"),
        ]
        data["fg_pitching"] = backfill_fg_mlbamid(
            data["fg_pitching"], label="FG pitching",
            extra_crosswalks=pitching_crosswalks,
        )
        print(f"  FG pitching:     {len(data['fg_pitching']):,} rows")

    # Game-level catcher assignments
    path = DATA_DIR / "game_catchers.csv"
    if path.exists():
        data["catchers"] = pd.read_csv(path)
        print(f"  Catchers:        {len(data['catchers']):,} rows")

    # Catcher game-calling features (built by collect/catcher.py).
    # Two files: as-of (per game date, leakage-free current-season feature)
    # and prior (full-season summary used as the prior for blending).
    path = DATA_DIR / "catcher_features_asof.csv"
    if path.exists():
        data["catcher_features_asof"] = pd.read_csv(path, parse_dates=["game_date"])
        print(f"  Catcher asof:    {len(data['catcher_features_asof']):,} rows")
    path = DATA_DIR / "catcher_features_prior.csv"
    if path.exists():
        data["catcher_features_prior"] = pd.read_csv(path)
        print(f"  Catcher prior:   {len(data['catcher_features_prior']):,} rows")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# STARTER IDENTIFICATION & BASIC CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def identify_starters(df, min_pitches=45):
    """Filter to starting pitcher appearances."""
    starters = df[df["total_pitches"] >= min_pitches].copy()
    print(f"  Filtered to {len(starters):,} starts (>= {min_pitches} pitches)")
    print(f"  Unique pitchers: {starters['pitcher'].nunique()}")
    return starters


def merge_game_metadata(df, game_meta):
    """
    Merge game metadata to get venue, pitcher team assignment, and
    identify the opposing team.
    """
    if game_meta is None:
        print("  ⚠ No game metadata — skipping metadata merge")
        return df

    meta_cols = [
        "game_pk", "venue_id", "venue_name", "day_night",
        "home_starter_id", "away_starter_id",
        "home_team_id", "away_team_id",
        "home_team_name", "away_team_name",
    ]
    available_cols = [c for c in meta_cols if c in game_meta.columns]
    meta_subset = game_meta[available_cols].drop_duplicates(subset=["game_pk"])

    # Drop any columns from meta that already exist in df (except game_pk)
    # to prevent _x/_y suffix issues
    overlap_cols = [c for c in meta_subset.columns if c in df.columns and c != "game_pk"]
    meta_subset = meta_subset.drop(columns=overlap_cols, errors="ignore")

    # Normalize game_pk to int64 for reliable merges throughout the pipeline
    normalize_game_pk(df, meta_subset)

    df = df.merge(meta_subset, on="game_pk", how="left")

    # Determine if pitcher is home or away using starter IDs
    if "home_starter_id" in df.columns:
        df["is_home"] = (df["pitcher"] == df["home_starter_id"]).astype(int)
    else:
        df["is_home"] = 0

    # Create opp_team and pitcher_team (drop first if they already exist)
    df = df.drop(columns=["opp_team", "pitcher_team"], errors="ignore")
    if "home_team" in df.columns and "away_team" in df.columns:
        df["opp_team"] = np.where(
            df["is_home"] == 1,
            df["away_team"],
            df["home_team"],
        )
        df["pitcher_team"] = np.where(
            df["is_home"] == 1,
            df["home_team"],
            df["away_team"],
        )
    else:
        df["opp_team"] = None
        df["pitcher_team"] = None

    # Verify no duplicate columns
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        print(f"  ⚠ Dropping duplicate columns: {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated()]

    print(f"  ✓ Merged game metadata")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ROLLING PITCHER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_rolling_features(df, windows=[3, 5, 10]):
    """
    Build rolling averages for key stats over the last N starts.
    All features use shift(1) to prevent leakage.
    """
    df = df.sort_values(["pitcher", "game_date"]).copy()

    # ── Add normalized K metrics before rolling ──
    # These are immune to shortened starts (rain delays, early pulls, blowouts)
    tp = df["total_pitches"].replace(0, np.nan)
    pa = df["plate_appearances"].replace(0, np.nan)
    df["k_per_100_pitches"] = df["strikeouts"] / tp * 100
    df["k_per_pa"] = df["strikeouts"] / pa  # Same as k_pct but explicit
    df["pitches_per_k"] = np.where(df["strikeouts"] > 0, tp / df["strikeouts"], np.nan)

    # Flag shortened starts (< 60 pitches is almost certainly an early exit)
    df["is_short_outing"] = (df["total_pitches"] < 60).astype(int)

    # Estimate innings from PA (rough: 3 PA per inning + extras)
    # More precise: outs = PA - hits - walks - HBP, innings = outs / 3
    if all(c in df.columns for c in ["plate_appearances", "hits_allowed", "walks"]):
        hbp = df["hbp"] if "hbp" in df.columns else 0
        outs = df["plate_appearances"] - df["hits_allowed"] - df["walks"] - hbp
        df["outs_recorded"] = outs.clip(lower=0).astype(int)
        df["outs_per_pa"] = (df["outs_recorded"] / pa).clip(lower=0, upper=1)
        df["est_innings"] = (outs / 3).clip(lower=0)
        df["k_per_9"] = np.where(df["est_innings"] > 0, df["strikeouts"] / df["est_innings"] * 9, 0)
    else:
        df["outs_recorded"] = np.nan
        df["outs_per_pa"] = np.nan
        df["est_innings"] = np.nan
        df["k_per_9"] = np.nan

    # ── HITS / WALKS / HR rate metrics (parallel to K metrics above) ──
    # These give the hits & walks models the same depth of rate features
    # that K already enjoys. All gated on column existence so the K
    # pipeline isn't affected if any of these are missing.
    if "hits_allowed" in df.columns:
        df["hits_per_pa"] = df["hits_allowed"] / pa
        df["h_per_9"] = np.where(df["est_innings"] > 0, df["hits_allowed"] / df["est_innings"] * 9, 0)
    if "walks" in df.columns:
        df["bb_per_pa"] = df["walks"] / pa
        df["bb_per_9"] = np.where(df["est_innings"] > 0, df["walks"] / df["est_innings"] * 9, 0)
    if "home_runs_allowed" in df.columns:
        df["hr_per_pa"] = df["home_runs_allowed"] / pa
        df["hr_per_9"] = np.where(df["est_innings"] > 0, df["home_runs_allowed"] / df["est_innings"] * 9, 0)
        # HR/FB rate — proxy when fly_balls column unavailable: HR / total batted balls.
        # True HR/FB needs fly-ball count, but on a per-game basis HR/BIP is the same
        # signal direction (extreme fly-ball pitchers will rate similarly).
        if "batted_balls" in df.columns:
            bb_denom = df["batted_balls"].replace(0, np.nan)
            df["hr_per_bip"] = df["home_runs_allowed"] / bb_denom
        if "fly_balls" in df.columns:
            fb_denom = df["fly_balls"].replace(0, np.nan)
            df["hr_per_fb"] = df["home_runs_allowed"] / fb_denom
    # K-BB% (Kershaw-style differential) — same idea as k_bb_pct but
    # we add this here because the rolling layer wants the per-game value
    if "k_pct" in df.columns and "bb_pct" in df.columns:
        df["k_minus_bb_pct"] = df["k_pct"] - df["bb_pct"]

    # ── BABIP (Batting Average on Balls in Play, pitcher's allowed) ──
    # BABIP = (H - HR) / (PA - K - BB - HBP - HR)
    # The single biggest "luck vs skill" stat for pitchers; rolling BABIP
    # is critical for hits prediction since hit suppression beyond
    # K-suppression is largely a BABIP story.
    if all(c in df.columns for c in ["hits_allowed", "home_runs_allowed", "plate_appearances",
                                       "strikeouts", "walks"]):
        hbp = df["hbp"] if "hbp" in df.columns else 0
        bip_denom = (df["plate_appearances"] - df["strikeouts"]
                     - df["walks"] - hbp - df["home_runs_allowed"])
        bip_denom = bip_denom.replace(0, np.nan).where(bip_denom > 0)
        bip_hits = (df["hits_allowed"] - df["home_runs_allowed"]).clip(lower=0)
        df["babip"] = (bip_hits / bip_denom).clip(0, 1)

    # ── LOB% (Left On Base) ──
    # LOB% = (H + BB + HBP - R) / (H + BB + HBP - 1.4 × HR)
    # FanGraphs formula. Strand rate — how well a pitcher escapes baserunner
    # damage. League average ~71-73%. Rolling LOB% catches "wiggling out of
    # jams" form vs "everything turns into a run" form.
    if all(c in df.columns for c in ["hits_allowed", "walks", "home_runs_allowed"]) and \
       any(c in df.columns for c in ["earned_runs", "runs"]):
        hbp = df["hbp"] if "hbp" in df.columns else 0
        runs_col = "earned_runs" if "earned_runs" in df.columns else "runs"
        baserunners = df["hits_allowed"] + df["walks"] + hbp
        # Denominator can be near-zero; safe-clip away from singularities
        lob_denom = (baserunners - 1.4 * df["home_runs_allowed"]).replace(0, np.nan)
        lob_pct = (baserunners - df[runs_col]) / lob_denom
        # LOB% is bounded [0, 1] but can spike above 1 when HR-heavy starts
        # produce negative denominators; clip generously.
        df["lob_pct"] = lob_pct.clip(0, 1.5)

    # ── BATTED-BALL TYPE RATES ──
    # If per-game batted-ball-type counts are present (from Statcast PA
    # events aggregation), build per-game rate features. These are the
    # highest-signal predictors of BABIP/hits beyond K%.
    # Expected columns from a typical Statcast aggregation:
    #   ground_balls, fly_balls, line_drives, popups (or infield_fly_balls)
    if "batted_balls" in df.columns:
        bb_denom = df["batted_balls"].replace(0, np.nan)
        for raw_col, rate_col in [
            ("ground_balls", "gb_pct"),
            ("fly_balls",    "fb_pct"),
            ("line_drives",  "ld_pct"),
            ("popups",       "pop_pct"),
            ("infield_fly_balls", "iffb_pct"),
        ]:
            if raw_col in df.columns:
                df[rate_col] = (df[raw_col] / bb_denom).clip(0, 1)

    # ── QUALITY OF CONTACT METRICS ──
    # avg_exit_velocity / avg_launch_angle / sweet_spot_pct may be
    # populated by the data collection layer. If so, expose them; the
    # rolling block below will pick them up automatically.
    # sweet_spot_pct = % of batted balls with launch angle 8°-32°
    # solid_contact_pct = % of barrels + solid_contact (Statcast classifications)
    # These don't need transformation here, just availability.

    # xBA / xwOBA on contact — Statcast's expected-stats metrics. The
    # collection layer should populate avg_xba_contact and avg_xwoba_contact
    # by averaging per-PA estimates over batted balls. Same story: just
    # available, rolling block consumes them by name.

    # Core stats for rolling windows
    roll_cols = [
        # Outcome counts (raw — model uses these alongside rates)
        "strikeouts", "walks", "hits_allowed", "home_runs_allowed",
        "plate_appearances", "total_pitches", "outs_recorded",
        # Normalized K metrics (immune to short outings)
        "k_per_100_pitches", "k_per_9", "est_innings",
        # ── Normalized HITS/WALKS/HR metrics (parallel to K) ──
        # Same shape of treatment as K so the H/W models get matching depth
        "hits_per_pa", "h_per_9",
        "bb_per_pa", "bb_per_9",
        "hr_per_pa", "hr_per_9", "hr_per_bip", "hr_per_fb",
        "k_minus_bb_pct",
        # ── Hit-quality / luck stats ──
        "babip", "lob_pct",
        # Rate stats
        "k_pct", "bb_pct", "k_bb_pct", "outs_per_pa", "whiff_pct", "csw_pct",
        "chase_rate", "strike_pct", "zone_pct",
        "barrel_pct", "hard_hit_pct", "soft_hit_pct",
        # ── Batted-ball type rates (heart of BABIP prediction) ──
        "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
        # ── Quality of contact (if available from data collection layer) ──
        "avg_exit_velocity", "avg_launch_angle",
        "sweet_spot_pct", "solid_contact_pct",
        "avg_xba_contact", "avg_xwoba_contact",
        # Stuff metrics
        "avg_velocity", "max_velocity", "avg_spin_rate",
        "avg_extension", "avg_induced_vert_break", "avg_horiz_break",
        # Pitch mix
        "fastball_pct", "breaking_pct", "offspeed_pct",
        "ff_pct", "sl_pct", "cu_pct", "ch_pct", "si_pct",
        # Platoon
        "whiff_pct_vs_left", "whiff_pct_vs_right",
        # Short outing flag (rolling avg = fraction of recent starts that were short)
        "is_short_outing",
    ]

    existing_cols = [c for c in roll_cols if c in df.columns]

    # ── Compute league-average fallback for pitchers with no history ──
    league_means = {}
    for col in existing_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            league_means[col] = vals.mean()
        else:
            league_means[col] = 0.0

    # ── Compute per-pitcher career averages as the shrinkage target ──
    # For a pitcher on start #2, we want to shrink their L3/L5/L10 toward
    # their OWN historical average, not the league mean. Skubal's thin-sample
    # rolling average should shrink toward Skubal's career rate, not toward
    # a league-average pitcher. Uses all prior data (shifted to prevent leakage).
    print("  Computing per-pitcher priors for rolling shrinkage...")
    pitcher_priors = {}
    for col in existing_cols:
        shifted = df.groupby("pitcher")[col].shift(1)
        pitcher_priors[col] = shifted.groupby(df["pitcher"]).transform(
            lambda x: x.expanding().mean()
        )

    for window in windows:
        print(f"  Rolling {window}-start averages ({len(existing_cols)} stats)...")

        # Track the actual number of observations backing each rolling window.
        # We compute this once per window (all cols share the same count).
        shifted_notna = df.groupby("pitcher")[existing_cols[0]].shift(1)
        df[f"_n_actual_L{window}"] = shifted_notna.groupby(df["pitcher"]).transform(
            lambda x: x.rolling(window, min_periods=1).count()
        )

        for col in existing_cols:
            shifted = df.groupby("pitcher")[col].shift(1)
            raw_roll = shifted.groupby(df["pitcher"]).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # ── Shrinkage: regress thin-sample averages toward pitcher's own prior ──
            # When a rolling window has only 1 or 2 actual observations out of
            # a requested 3/5/10, the average is noisy. We blend toward the
            # pitcher's career expanding mean proportional to how full the window is:
            #   weight = n_actual / window
            #   shrunk  = weight * raw_roll + (1 - weight) * pitcher_prior
            #
            # This means Skubal on start #2 with 1 obs in L3 gets:
            #   1/3 * (start 1 value) + 2/3 * (Skubal's career avg)
            # instead of shrinking toward the league average.
            #
            # For pitchers with no prior history, falls back to league mean.
            n_actual = df[f"_n_actual_L{window}"]
            fill_weight = (n_actual / window).clip(upper=1.0)
            prior = pitcher_priors[col].fillna(league_means.get(col, 0.0))
            df[f"{col}_L{window}"] = fill_weight * raw_roll + (1 - fill_weight) * prior

        # Also build rolling std dev for key stats (captures consistency)
        volatility_cols = [
            # K-side
            "strikeouts", "outs_recorded", "k_pct", "whiff_pct", "avg_velocity",
            # H/W-side (parallel set so model sees H/W volatility too)
            "hits_allowed", "walks", "home_runs_allowed",
            "hits_per_pa", "bb_per_pa", "babip",
            "avg_exit_velocity", "ld_pct",
        ]
        for col in [c for c in volatility_cols if c in df.columns]:
            shifted = df.groupby("pitcher")[col].shift(1)
            df[f"{col}_std_L{window}"] = shifted.groupby(df["pitcher"]).transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )

    # ── Expose the observation counts as features so the model knows ──
    # Rename internal tracking columns to proper feature names
    for window in windows:
        internal = f"_n_actual_L{window}"
        if internal in df.columns:
            df[f"n_starts_in_L{window}"] = df[internal]
            df.drop(columns=[internal], inplace=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SEASON-TO-DATE CUMULATIVE STATS
# ══════════════════════════════════════════════════════════════════════════════

def build_season_cumulative(df):
    """Season-to-date cumulative stats (excluding current game)."""
    print("  Building season-to-date cumulative stats...")
    df = df.sort_values(["pitcher", "season", "game_date"]).copy()

    cum_cols = [
        "strikeouts", "walks", "hits_allowed", "home_runs_allowed",
        "plate_appearances", "total_pitches", "outs_recorded", "whiffs",
        "batted_balls", "barrels", "hard_hits", "chases",
        "out_of_zone_pitches", "in_zone_pitches", "called_strikes",
        # ── Hits/walks/HR pipeline additions (gated by column presence) ──
        # Earned runs feeds LOB%/ERA-style season rates; batted-ball type
        # counts give us _szn rates parallel to barrel/hard-hit.
        "earned_runs", "runs", "hbp",
        "singles", "doubles", "triples",
        "ground_balls", "fly_balls", "line_drives", "popups", "infield_fly_balls",
        "soft_hits",
    ]
    existing = [c for c in cum_cols if c in df.columns]

    for col in existing:
        shifted = df.groupby(["pitcher", "season"])[col].shift(1)
        df[f"{col}_szn"] = shifted.groupby([df["pitcher"], df["season"]]).cumsum()

    # Derived cumulative rates
    pa_szn = df["plate_appearances_szn"].replace(0, np.nan)
    tp_szn = df["total_pitches_szn"].replace(0, np.nan)
    bb_szn = df.get("batted_balls_szn", pd.Series(dtype=float)).replace(0, np.nan)
    ooz_szn = df.get("out_of_zone_pitches_szn", pd.Series(dtype=float)).replace(0, np.nan)

    df["k_pct_szn"] = df.get("strikeouts_szn", 0) / pa_szn
    df["bb_pct_szn"] = df.get("walks_szn", 0) / pa_szn
    df["outs_per_pa_szn"] = df.get("outs_recorded_szn", 0) / pa_szn
    df["k_bb_pct_szn"] = df["k_pct_szn"] - df["bb_pct_szn"]
    df["whiff_pct_szn"] = df.get("whiffs_szn", 0) / tp_szn
    df["csw_pct_szn"] = (df.get("called_strikes_szn", 0) + df.get("whiffs_szn", 0)) / tp_szn
    df["barrel_pct_szn"] = df.get("barrels_szn", 0) / bb_szn
    df["hard_hit_pct_szn"] = df.get("hard_hits_szn", 0) / bb_szn
    df["chase_rate_szn"] = df.get("chases_szn", 0) / ooz_szn
    df["zone_pct_szn"] = df.get("in_zone_pitches_szn", 0) / tp_szn

    # ── HITS / WALKS / HR season rates (parallel to K rates above) ──
    # These give the H/W rate models a full set of stable season-to-date
    # rate features. Each is gated on having the underlying cumulative
    # stat (won't crash if a count column wasn't collected upstream).
    if "hits_allowed_szn" in df.columns:
        df["hits_per_pa_szn"] = df["hits_allowed_szn"] / pa_szn
    if "walks_szn" in df.columns:
        df["bb_per_pa_szn"] = df["walks_szn"] / pa_szn
    if "home_runs_allowed_szn" in df.columns:
        df["hr_per_pa_szn"] = df["home_runs_allowed_szn"] / pa_szn
        if "batted_balls_szn" in df.columns:
            df["hr_per_bip_szn"] = df["home_runs_allowed_szn"] / bb_szn
        if "fly_balls_szn" in df.columns:
            fb_szn = df["fly_balls_szn"].replace(0, np.nan)
            df["hr_per_fb_szn"] = df["home_runs_allowed_szn"] / fb_szn

    # ── BABIP season-to-date ──
    # BABIP_szn = (H_szn - HR_szn) / (PA_szn - K_szn - BB_szn - HBP_szn - HR_szn)
    if all(c + "_szn" in df.columns for c in
           ["hits_allowed", "home_runs_allowed", "strikeouts", "walks"]):
        hbp_szn = df.get("hbp_szn", 0)
        bip_denom_szn = (pa_szn - df["strikeouts_szn"] - df["walks_szn"]
                         - hbp_szn - df["home_runs_allowed_szn"])
        bip_denom_szn = bip_denom_szn.where(bip_denom_szn > 0)
        bip_hits_szn = (df["hits_allowed_szn"] - df["home_runs_allowed_szn"]).clip(lower=0)
        df["babip_szn"] = (bip_hits_szn / bip_denom_szn).clip(0, 1)

    # ── LOB% season-to-date ──
    if all(c + "_szn" in df.columns for c in ["hits_allowed", "walks", "home_runs_allowed"]) and \
       any(c + "_szn" in df.columns for c in ["earned_runs", "runs"]):
        hbp_szn = df.get("hbp_szn", 0)
        runs_szn_col = "earned_runs_szn" if "earned_runs_szn" in df.columns else "runs_szn"
        baserunners_szn = df["hits_allowed_szn"] + df["walks_szn"] + hbp_szn
        lob_denom_szn = (baserunners_szn - 1.4 * df["home_runs_allowed_szn"]).replace(0, np.nan)
        df["lob_pct_szn"] = ((baserunners_szn - df[runs_szn_col]) /
                              lob_denom_szn).clip(0, 1.5)

    # ── Batted-ball type season rates ──
    if "batted_balls_szn" in df.columns:
        for raw_col, rate_col in [
            ("ground_balls_szn", "gb_pct_szn"),
            ("fly_balls_szn",    "fb_pct_szn"),
            ("line_drives_szn",  "ld_pct_szn"),
            ("popups_szn",       "pop_pct_szn"),
            ("infield_fly_balls_szn", "iffb_pct_szn"),
        ]:
            if raw_col in df.columns:
                df[rate_col] = (df[raw_col] / bb_szn).clip(0, 1)

    # ── Soft-hit season rate (already had hard_hit_pct_szn) ──
    if "soft_hits_szn" in df.columns:
        df["soft_hit_pct_szn"] = df["soft_hits_szn"] / bb_szn

    # ── Quality of contact season averages (pitch-count-weighted) ──
    # Exit velocity, launch angle, sweet spot, solid contact — averaged
    # using batted_balls as the weight (same pattern as pitch mix below
    # uses total_pitches).
    contact_cols = [
        "avg_exit_velocity", "avg_launch_angle",
        "sweet_spot_pct", "solid_contact_pct",
        "avg_xba_contact", "avg_xwoba_contact",
    ]
    if "batted_balls" in df.columns:
        bip_for_weight = df["batted_balls_szn"].replace(0, np.nan) \
            if "batted_balls_szn" in df.columns else None
        if bip_for_weight is not None:
            for col in contact_cols:
                if col not in df.columns:
                    continue
                weighted = df[col].fillna(0) * df["batted_balls"].fillna(0)
                shifted_weighted = weighted.groupby([df["pitcher"], df["season"]]).shift(1)
                cum_weighted = shifted_weighted.groupby(
                    [df["pitcher"], df["season"]]
                ).cumsum()
                df[f"{col}_szn"] = cum_weighted / bip_for_weight

    # ── Pitch-mix season rates (pitch-count-weighted) ──
    # The per-game pct columns (ff_pct, sl_pct, etc.) are already rates,
    # so the proper season aggregate is sum(per_game_pct * total_pitches)
    # divided by sum(total_pitches). This weights a 110-pitch start more
    # than a 60-pitch start. We compute via cumulative shifted sums to
    # avoid leakage (same pattern as the count-based _szn columns above).
    pitch_mix_cols = [
        "fastball_pct", "breaking_pct", "offspeed_pct",
        "ff_pct", "sl_pct", "cu_pct", "ch_pct", "si_pct",
    ]
    if "total_pitches" in df.columns:
        # Cumulative total_pitches already exists as total_pitches_szn, but
        # we need it explicitly here for clarity.
        tp_for_weight = df["total_pitches_szn"].replace(0, np.nan)
        for col in pitch_mix_cols:
            if col not in df.columns:
                continue
            # weighted contribution = per_game_pct * total_pitches per game
            weighted = (df[col].fillna(0) * df["total_pitches"].fillna(0))
            shifted_weighted = weighted.groupby([df["pitcher"], df["season"]]).shift(1)
            cum_weighted = shifted_weighted.groupby(
                [df["pitcher"], df["season"]]
            ).cumsum()
            df[f"{col}_szn"] = cum_weighted / tp_for_weight

    # ── Velocity / stuff season averages (also pitch-count-weighted) ──
    # avg_velocity_szn is a key feature (08_per_pa_model expects it).
    # Same weighting scheme as pitch mix.
    velo_cols = [
        "avg_velocity", "max_velocity", "avg_spin_rate",
        "avg_extension", "avg_induced_vert_break", "avg_horiz_break",
    ]
    if "total_pitches" in df.columns:
        tp_for_weight = df["total_pitches_szn"].replace(0, np.nan)
        for col in velo_cols:
            if col not in df.columns:
                continue
            weighted = (df[col].fillna(0) * df["total_pitches"].fillna(0))
            shifted_weighted = weighted.groupby([df["pitcher"], df["season"]]).shift(1)
            cum_weighted = shifted_weighted.groupby(
                [df["pitcher"], df["season"]]
            ).cumsum()
            df[f"{col}_szn"] = cum_weighted / tp_for_weight

    # Start number in season
    df["start_num"] = df.groupby(["pitcher", "season"]).cumcount() + 1

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TREND FEATURES (RECENT vs BASELINE)
# ══════════════════════════════════════════════════════════════════════════════

def build_trend_features(df):
    """How recent performance compares to season baseline."""
    print("  Building trend features...")

    trend_pairs = [
        # (recent, baseline, name)
        ("k_pct_L3", "k_pct_szn", "k_pct_trend_3"),
        ("k_pct_L5", "k_pct_szn", "k_pct_trend_5"),
        ("whiff_pct_L3", "whiff_pct_szn", "whiff_pct_trend_3"),
        ("csw_pct_L3", "csw_pct_szn", "csw_pct_trend_3"),
        ("bb_pct_L3", "bb_pct_szn", "bb_pct_trend_3"),
        ("chase_rate_L3", "chase_rate_szn", "chase_rate_trend_3"),
        ("barrel_pct_L3", "barrel_pct_szn", "barrel_pct_trend_3"),
        ("hard_hit_pct_L3", "hard_hit_pct_szn", "hard_hit_pct_trend_3"),
        ("avg_velocity_L3", "avg_velocity_L10", "velo_trend_3v10"),
        ("avg_spin_rate_L3", "avg_spin_rate_L10", "spin_trend_3v10"),
        ("strikeouts_L3", "strikeouts_L10", "k_count_trend_3v10"),
    ]

    for recent, baseline, name in trend_pairs:
        if recent in df.columns and baseline in df.columns:
            df[name] = df[recent] - df[baseline]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# OPPOSING TEAM FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_opposing_team_features(df):
    """
    Build opposing team batting profiles from the pitcher-game data.
    For each game, compute what the opposing team's batters have done
    season-to-date (their K rate, walk rate, barrel rate, etc. against
    all pitchers they've faced so far this season).
    """
    print("  Building opposing team batting profiles...")

    if "opp_team" not in df.columns or df["opp_team"].isna().all():
        print("  ⚠ No opposing team info available — using league averages")
        return df

    df = df.sort_values(["season", "game_date"]).copy()

    # Build team-game batting stats from the pitcher perspective.
    # Each pitcher-game row tells us what the opposing team's batters did.
    # Group by (opp_team, season, game_date, game_pk) to handle doubleheaders.
    team_game_stats = df.groupby(["opp_team", "season", "game_date", "game_pk"]).agg(
        team_pa=("plate_appearances", "sum"),
        team_k=("strikeouts", "sum"),
        team_bb=("walks", "sum"),
        team_hits=("hits_allowed", "sum"),
        team_hr=("home_runs_allowed", "sum"),
        team_barrels=("barrels", "sum"),
        team_hard_hits=("hard_hits", "sum"),
        team_batted_balls=("batted_balls", "sum"),
        team_whiffs=("whiffs", "sum"),
        team_pitches_seen=("total_pitches", "sum"),
    ).reset_index()

    team_game_stats = team_game_stats.sort_values(["opp_team", "season", "game_date", "game_pk"])

    # Build cumulative team stats (shifted to avoid leakage)
    for col in ["team_pa", "team_k", "team_bb", "team_hits", "team_hr",
                "team_barrels", "team_hard_hits", "team_batted_balls",
                "team_whiffs", "team_pitches_seen"]:
        shifted = team_game_stats.groupby(["opp_team", "season"])[col].shift(1)
        team_game_stats[f"{col}_cum"] = shifted.groupby(
            [team_game_stats["opp_team"], team_game_stats["season"]]
        ).cumsum()

    # Derived team rates
    tpa = team_game_stats["team_pa_cum"].replace(0, np.nan)
    tbb = team_game_stats["team_batted_balls_cum"].replace(0, np.nan)
    ttp = team_game_stats["team_pitches_seen_cum"].replace(0, np.nan)

    team_game_stats["opp_k_pct"] = team_game_stats["team_k_cum"] / tpa
    team_game_stats["opp_bb_pct"] = team_game_stats["team_bb_cum"] / tpa
    team_game_stats["opp_hr_pct"] = team_game_stats["team_hr_cum"] / tpa
    team_game_stats["opp_barrel_pct"] = team_game_stats["team_barrels_cum"] / tbb
    team_game_stats["opp_hard_hit_pct"] = team_game_stats["team_hard_hits_cum"] / tbb
    team_game_stats["opp_whiff_pct"] = team_game_stats["team_whiffs_cum"] / ttp
    team_game_stats["opp_batting_avg"] = team_game_stats["team_hits_cum"] / tpa

    # Also build rolling team stats (last 10 team-games)
    for col in ["team_k", "team_bb", "team_hits", "team_whiffs", "team_pa"]:
        shifted = team_game_stats.groupby(["opp_team", "season"])[col].shift(1)
        team_game_stats[f"{col}_L10"] = shifted.groupby(
            [team_game_stats["opp_team"], team_game_stats["season"]]
        ).transform(lambda x: x.rolling(10, min_periods=3).mean())

    tpa_l10 = team_game_stats["team_pa_L10"].replace(0, np.nan)
    team_game_stats["opp_k_pct_L10"] = team_game_stats["team_k_L10"] / tpa_l10
    team_game_stats["opp_bb_pct_L10"] = team_game_stats["team_bb_L10"] / tpa_l10
    tps_l10 = team_game_stats.get("team_pitches_seen_L10", pd.Series(dtype=float)).replace(0, np.nan)
    team_game_stats["opp_whiff_pct_L10"] = team_game_stats["team_whiffs_L10"] / tps_l10

    # Merge back — use game_pk to avoid doubleheader duplicates
    opp_features = [c for c in team_game_stats.columns
                    if c.startswith("opp_") and c != "opp_team"]
    merge_df = team_game_stats[["opp_team", "game_pk"] + opp_features].copy()

    # Drop any existing opp_ columns from df to avoid _x/_y duplicates
    existing_opp = [c for c in df.columns if c.startswith("opp_") and c != "opp_team"]
    df = df.drop(columns=existing_opp, errors="ignore")

    # Deduplicate merge_df per (opp_team, game_pk) — take first row
    merge_df = merge_df.drop_duplicates(subset=["opp_team", "game_pk"], keep="first")

    df = df.merge(merge_df, on=["opp_team", "game_pk"], how="left")
    print(f"  ✓ Added {len(opp_features)} opposing team features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LINEUP-LEVEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_lineup_features(df, lineup_df):
    """
    Build granular opposing-lineup features from individual batter data.
    Instead of team-level aggregates, this computes stats weighted by
    lineup position (top-of-order batters get more PAs and matter more)
    and includes the lineup's handedness composition against the pitcher.

    Features built:
      - Lineup aggregate K rate (season-to-date, weighted by lineup spot)
      - Lineup aggregate walk rate, barrel rate, whiff rate
      - Lineup L/R composition (% of lineup batting from each side)
      - Top-of-order (1-5) vs bottom-of-order (6-9) K rate split
      - Lineup rolling 10-game K rate (recent form)
    """
    print("  Building lineup-level features...")

    if lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ No lineup data — skipping")
        return df

    lu = lineup_df.copy()

    # ── Step 1: Build per-batter cumulative stats ──
    # Track each batter's season-to-date K rate from their game-by-game logs.
    # We need to figure out the season from game_pk — merge from main df.
    game_seasons = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
    lu = lu.merge(game_seasons, on="game_pk", how="left")
    lu = lu.dropna(subset=["season"])
    lu["season"] = lu["season"].astype(int)
    lu["game_date"] = pd.to_datetime(lu["game_date"])

    # Sort by batter + date for cumulative calcs
    lu = lu.sort_values(["player_id", "season", "game_date"]).copy()

    # Cumulative stats per batter per season (shifted to exclude current game)
    for col in ["at_bats", "strikeouts", "hits", "walks"]:
        shifted = lu.groupby(["player_id", "season"])[col].shift(1)
        lu[f"{col}_cum"] = shifted.groupby([lu["player_id"], lu["season"]]).cumsum()

    # Batter-level cumulative rates
    ab_cum = lu["at_bats_cum"].replace(0, np.nan)
    lu["batter_k_rate"] = lu["strikeouts_cum"] / ab_cum
    lu["batter_hit_rate"] = lu["hits_cum"] / ab_cum
    lu["batter_bb_rate"] = lu["walks_cum"] / (ab_cum + lu["walks_cum"]).replace(0, np.nan)

    # Rolling 10-game batter K rate
    shifted_k = lu.groupby(["player_id", "season"])["strikeouts"].shift(1)
    shifted_ab = lu.groupby(["player_id", "season"])["at_bats"].shift(1)
    lu["batter_k_L10"] = shifted_k.groupby([lu["player_id"], lu["season"]]).transform(
        lambda x: x.rolling(10, min_periods=3).sum()
    )
    lu["batter_ab_L10"] = shifted_ab.groupby([lu["player_id"], lu["season"]]).transform(
        lambda x: x.rolling(10, min_periods=3).sum()
    )
    lu["batter_k_rate_L10"] = lu["batter_k_L10"] / lu["batter_ab_L10"].replace(0, np.nan)

    # ── Step 2: Aggregate to lineup level per game ──
    # Weight by lineup position: position 1 has weight 9, position 9 has weight 1
    # This reflects that top-of-order batters get ~15% more PAs over a game.
    lu["lineup_weight"] = 10 - lu["lineup_position"]  # 9 for leadoff, 1 for 9th

    # For each game + side, aggregate the lineup's stats
    def weighted_mean(group, col, weight_col="lineup_weight"):
        vals = group[col].fillna(0)
        weights = group[weight_col]
        if weights.sum() == 0:
            return np.nan
        return (vals * weights).sum() / weights.sum()

    lineup_agg_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        rec = {
            "game_pk": gpk,
            "side": side,
            # Weighted lineup K rate
            "lu_k_rate_wtd": weighted_mean(group, "batter_k_rate"),
            "lu_hit_rate_wtd": weighted_mean(group, "batter_hit_rate"),
            "lu_bb_rate_wtd": weighted_mean(group, "batter_bb_rate"),
            # Weighted recent K rate
            "lu_k_rate_L10_wtd": weighted_mean(group, "batter_k_rate_L10"),
            # Unweighted (simple average) for comparison
            "lu_k_rate_avg": group["batter_k_rate"].mean(),
            "lu_bb_rate_avg": group["batter_bb_rate"].mean(),
            # Top of order (1-5) vs bottom (6-9)
            "lu_top5_k_rate": group.loc[group["lineup_position"] <= 5, "batter_k_rate"].mean(),
            "lu_bot4_k_rate": group.loc[group["lineup_position"] > 5, "batter_k_rate"].mean(),
            # Handedness composition
            "lu_pct_left": (group["bat_side"] == "L").mean(),
            "lu_pct_right": (group["bat_side"] == "R").mean(),
            "lu_pct_switch": (group["bat_side"] == "S").mean(),
            # Number of batters with K rate > 25% (swing-and-miss prone)
            "lu_high_k_batters": (group["batter_k_rate"] > 0.25).sum(),
            # Number of batters with K rate < 15% (tough to strike out)
            "lu_low_k_batters": (group["batter_k_rate"] < 0.15).sum(),
            # Lineup size (should be 9 but sometimes partial data)
            "lu_batters_count": len(group),
        }
        lineup_agg_records.append(rec)

    lineup_agg = pd.DataFrame(lineup_agg_records)
    print(f"    Aggregated {len(lineup_agg):,} lineup-games")

    # ── Step 3: Merge to pitcher-game level ──
    # The pitcher faces the OPPOSING lineup. If pitcher is home, they face
    # the away lineup; if away, they face the home lineup.
    # We need to flip: merge "away" lineup to home pitchers and vice versa.

    # Home pitchers face away lineup
    away_lineups = lineup_agg[lineup_agg["side"] == "away"].drop(columns=["side"])
    away_lineups = away_lineups.rename(columns={
        c: f"opp_{c}" for c in away_lineups.columns if c != "game_pk"
    })

    # Away pitchers face home lineup
    home_lineups = lineup_agg[lineup_agg["side"] == "home"].drop(columns=["side"])
    home_lineups = home_lineups.rename(columns={
        c: f"opp_{c}" for c in home_lineups.columns if c != "game_pk"
    })

    # Normalize game_pk before merge to prevent dtype drift
    normalize_game_pk(df, away_lineups, home_lineups)

    # Merge based on is_home flag
    df_home = df[df["is_home"] == 1].merge(away_lineups, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_lineups, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # ── Step 4: Interaction features ──
    # Pitcher K ability × lineup K vulnerability
    if "k_pct_L5" in df.columns and "opp_lu_k_rate_wtd" in df.columns:
        df["pitcher_x_lineup_k"] = df["k_pct_L5"] * df["opp_lu_k_rate_wtd"]

    # Pitcher whiff rate × lineup K rate (stuff meets vulnerability)
    if "whiff_pct_L5" in df.columns and "opp_lu_k_rate_wtd" in df.columns:
        df["whiff_x_lineup_k"] = df["whiff_pct_L5"] * df["opp_lu_k_rate_wtd"]

    # K rate trend gap: pitcher trending up + lineup trending up = compounding
    if "k_pct_trend_3" in df.columns and "opp_lu_k_rate_L10_wtd" in df.columns:
        df["k_trend_x_lu_recent"] = df["k_pct_trend_3"] * df["opp_lu_k_rate_L10_wtd"]

    lu_features = [c for c in df.columns if c.startswith("opp_lu_") or c in [
        "pitcher_x_lineup_k", "whiff_x_lineup_k", "k_trend_x_lu_recent"
    ]]
    print(f"  ✓ Added {len(lu_features)} lineup-level features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LINEUP HIT-QUALITY FEATURES (BABIP, hard-hit, barrel rates)
# ══════════════════════════════════════════════════════════════════════════════

def build_lineup_quality_features(df, lineup_df, batter_rolling_df):
    """
    Aggregate per-batter rolling hit-quality stats from batter_rolling_stats.csv
    up to the lineup level, weighted by lineup position.

    The hits model needs batter-side signal that's NOT just K rate. Existing
    opp_lu_* features cover K rate, BB rate, plate discipline — but nothing
    about how hard the opposing lineup hits the ball or what BABIP they're
    producing. This function fills that gap.

    For each (game_pk, side), we pull each batter's rolling stats from
    batter_rolling_stats.csv (which 02b builds with proper shift(1) lagging)
    and aggregate them with the same 9-down-to-1 lineup-position weighting
    used in build_lineup_features.

    Features added (all prefixed opp_lu_):
      - opp_lu_hard_hit_rate_l15 / _l30 / _season
      - opp_lu_barrel_rate_l15 / _l30 / _season
      - opp_lu_obp_l15 / _l30 / _season
      - opp_lu_top3_hard_hit_l30: top-3 hitters' hard-hit (heart of order)
      - opp_lu_top3_barrel_l30:   top-3 hitters' barrel rate
      - opp_lu_quality_gap: top3 hard-hit minus bottom3 hard-hit
        (high values = lineup is top-heavy, more dangerous in early innings)
    """
    print("  Building lineup hit-quality features...")

    if lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ No lineup data — skipping")
        return df
    if batter_rolling_df is None or len(batter_rolling_df) == 0:
        print("  ⚠ No batter_rolling_stats — skipping")
        print("    (run collect/batters.py to populate)")
        return df

    lu = lineup_df.copy()
    br = batter_rolling_df.copy()

    # Get game_date / season for the lineup rows
    game_info = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
    lu = lu.merge(game_info, on="game_pk", how="left")
    lu = lu.dropna(subset=["game_date"])
    lu["game_date"] = pd.to_datetime(lu["game_date"])

    # batter_rolling has one row per (batter, game_date). Join to lineup on
    # (player_id, game_date) — the rolling stats are already shifted to
    # exclude the current game's stats, so this is a clean lag-free join.
    br = br.rename(columns={"batter": "player_id"})
    br["game_date"] = pd.to_datetime(br["game_date"])

    # Pick the columns we care about
    quality_cols = []
    candidates = [
        # Hard-hit at multiple windows
        "hard_hit_rate_l15", "hard_hit_rate_l30", "hard_hit_rate_season",
        # Barrel rate
        "barrel_rate_l15", "barrel_rate_l30", "barrel_rate_season",
        # OBP (a good general hit-prob signal)
        "obp_l15", "obp_l30", "obp_season",
    ]
    for c in candidates:
        if c in br.columns:
            quality_cols.append(c)

    if not quality_cols:
        print("  ⚠ batter_rolling_stats has none of the expected columns "
              "(hard_hit_rate_*, barrel_rate_*, obp_*) — skipping")
        return df

    # Merge per-batter rolling stats onto the lineup
    join_cols = ["player_id", "game_date"] + quality_cols
    br_slim = br[join_cols].drop_duplicates(subset=["player_id", "game_date"], keep="last")
    lu = lu.merge(br_slim, on=["player_id", "game_date"], how="left")

    # Lineup-position weight: leadoff=9 down to 9th=1
    lu["lineup_weight"] = 10 - lu["lineup_position"]

    def weighted_mean(group, col, weight_col="lineup_weight"):
        vals = group[col]
        mask = vals.notna()
        if not mask.any():
            return np.nan
        weights = group.loc[mask, weight_col]
        if weights.sum() == 0:
            return np.nan
        return (vals[mask] * weights).sum() / weights.sum()

    # Aggregate per (game_pk, side)
    agg_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        rec = {"game_pk": gpk, "side": side}
        for col in quality_cols:
            # Weighted lineup mean
            rec[f"lu_{col}_wtd"] = weighted_mean(group, col)
            # Top-3 (heart of order) mean
            top3 = group[group["lineup_position"] <= 3]
            rec[f"lu_top3_{col}"] = (top3[col].mean()
                                      if len(top3) > 0 and top3[col].notna().any()
                                      else np.nan)
            # Bottom-3 mean
            bot3 = group[group["lineup_position"] >= 7]
            rec[f"lu_bot3_{col}"] = (bot3[col].mean()
                                      if len(bot3) > 0 and bot3[col].notna().any()
                                      else np.nan)
        agg_records.append(rec)

    agg = pd.DataFrame(agg_records)
    print(f"    Aggregated {len(agg):,} lineup-games over {len(quality_cols)} quality metrics")

    # Build lu_*_quality_gap features (top3 - bot3) — a "top-heaviness" signal
    for col in quality_cols:
        top_c = f"lu_top3_{col}"
        bot_c = f"lu_bot3_{col}"
        if top_c in agg.columns and bot_c in agg.columns:
            agg[f"lu_{col}_top_bot_gap"] = agg[top_c] - agg[bot_c]

    # Merge to pitcher-game level (flip home/away like build_lineup_features)
    away_lineups = agg[agg["side"] == "away"].drop(columns=["side"])
    away_lineups = away_lineups.rename(columns={
        c: f"opp_{c}" for c in away_lineups.columns if c != "game_pk"
    })
    home_lineups = agg[agg["side"] == "home"].drop(columns=["side"])
    home_lineups = home_lineups.rename(columns={
        c: f"opp_{c}" for c in home_lineups.columns if c != "game_pk"
    })

    normalize_game_pk(df, away_lineups, home_lineups)

    pre_cols = set(df.columns)
    df_home = df[df["is_home"] == 1].merge(away_lineups, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_lineups, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True).sort_values(
        ["pitcher", "game_date"]
    ).reset_index(drop=True)

    new_cols = [c for c in df.columns if c not in pre_cols]
    print(f"  ✓ Added {len(new_cols)} lineup-quality features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PITCH-TYPE EFFECTIVENESS & MATCHUP FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_pitch_type_features(df, pitcher_pt_df, batter_pt_df, lineup_df):
    """
    Build pitch-type-level features for both pitchers and opposing lineups,
    then cross them to create matchup features.

    Pitcher side: per-pitch-type effectiveness (whiff rate, velocity, chase
    rate) as rolling averages — e.g., "slider whiff rate last 5 starts".

    Batter/Lineup side: per-pitch-type vulnerability aggregated across the
    opposing lineup — e.g., "lineup whiff rate vs breaking balls".

    Cross features: pitcher pitch-type strength × lineup pitch-type weakness.
    E.g., if a pitcher's slider whiff rate is 40% and the opposing lineup
    whiffs on 35% of sliders, the cross feature captures that compounding.
    """
    print("  Building pitch-type effectiveness & matchup features...")

    PRIMARY_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "FS", "SV"]
    BROAD_CATEGORIES = {
        "FB": ["FF", "SI", "FC"],
        "BRK": ["SL", "CU", "SV", "KC", "CS", "SC"],
        "OS": ["CH", "FS", "FO"],
    }

    # ── Part A: Pitcher pitch-type rolling features ──
    if pitcher_pt_df is not None and len(pitcher_pt_df) > 0:
        ppt = pitcher_pt_df.copy()
        ppt = ppt.sort_values(["pitcher", "game_date"])

        # For each primary pitch type, build rolling features for pitchers
        # who throw that pitch
        pitcher_pt_features = {}

        for pt in PRIMARY_TYPES:
            pt_data = ppt[ppt["pitch_type"] == pt].copy()
            if len(pt_data) == 0:
                continue

            pt_lower = pt.lower()
            stats_to_roll = {
                f"pt_{pt_lower}_whiff_rate": "pt_whiff_rate",
                f"pt_{pt_lower}_csw_rate": "pt_csw_rate",
                f"pt_{pt_lower}_chase_rate": "pt_chase_rate",
                f"pt_{pt_lower}_velo": "pt_avg_velo",
                f"pt_{pt_lower}_spin": "pt_avg_spin",
                f"pt_{pt_lower}_usage": "pt_pitches",
            }

            for new_name, source_col in stats_to_roll.items():
                if source_col not in pt_data.columns:
                    continue
                for window in [5, 10]:
                    shifted = pt_data.groupby("pitcher")[source_col].shift(1)
                    pt_data[f"{new_name}_L{window}"] = shifted.groupby(
                        pt_data["pitcher"]
                    ).transform(lambda x: x.rolling(window, min_periods=2).mean())

            # Store per pitcher × game_pk for merging
            roll_cols = [c for c in pt_data.columns if c.startswith(f"pt_{pt_lower}") and "_L" in c]
            if roll_cols:
                merge_subset = pt_data[["pitcher", "game_pk"] + roll_cols].copy()
                # Deduplicate
                merge_subset = merge_subset.drop_duplicates(subset=["pitcher", "game_pk"], keep="last")

                for _, row in merge_subset.iterrows():
                    key = (row["pitcher"], row["game_pk"])
                    if key not in pitcher_pt_features:
                        pitcher_pt_features[key] = {}
                    for col in roll_cols:
                        pitcher_pt_features[key][col] = row[col]

        # Merge pitcher pitch-type features to main df
        if pitcher_pt_features:
            pt_feat_df = pd.DataFrame.from_dict(pitcher_pt_features, orient="index")
            pt_feat_df.index = pd.MultiIndex.from_tuples(pt_feat_df.index, names=["pitcher", "game_pk"])
            pt_feat_df = pt_feat_df.reset_index()

            # Drop existing pitch-type columns to avoid conflicts
            existing_pt = [c for c in df.columns if c.startswith("pt_") and "_L" in c]
            df = df.drop(columns=existing_pt, errors="ignore")

            df = df.merge(pt_feat_df, on=["pitcher", "game_pk"], how="left")
            pt_added = len([c for c in df.columns if c.startswith("pt_") and "_L" in c])
            print(f"    ✓ Added {pt_added} pitcher pitch-type rolling features")

    # ── Part B: Batter pitch-type vulnerability (lineup level) ──
    if batter_pt_df is not None and len(batter_pt_df) > 0 and lineup_df is not None and len(lineup_df) > 0:
        bpt = batter_pt_df.copy()
        bpt = bpt.sort_values(["batter", "game_date"])

        # Build cumulative batter stats per pitch type
        for col in ["bpt_pitches_seen", "bpt_whiffs", "bpt_strikes", "bpt_strikeouts",
                     "bpt_hits", "bpt_pa", "bpt_batted_balls", "bpt_hard_hits"]:
            if col in bpt.columns:
                shifted = bpt.groupby(["batter", "pitch_type"])[col].shift(1)
                bpt[f"{col}_cum"] = shifted.groupby(
                    [bpt["batter"], bpt["pitch_type"]]
                ).cumsum()

        # Batter cumulative rates per pitch type
        tp_cum = bpt["bpt_pitches_seen_cum"].replace(0, np.nan)
        pa_cum = bpt["bpt_pa_cum"].replace(0, np.nan)
        bb_cum = bpt["bpt_batted_balls_cum"].replace(0, np.nan)

        bpt["batter_pt_whiff_rate"] = bpt["bpt_whiffs_cum"] / tp_cum
        bpt["batter_pt_k_rate"] = bpt["bpt_strikeouts_cum"] / pa_cum
        bpt["batter_pt_hard_hit_rate"] = bpt["bpt_hard_hits_cum"] / bb_cum

        # Get each batter's most recent pitch-type rates
        batter_latest = bpt.sort_values("game_date").groupby(
            ["batter", "pitch_type"]
        ).last().reset_index()

        # Aggregate to lineup level using the lineup data
        lu = lineup_df.copy()

        # Get game_pk → season mapping from main df
        game_info = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
        lu = lu.merge(game_info, on="game_pk", how="left")
        lu = lu.dropna(subset=["season"])

        # For each game lineup, compute the lineup's aggregate vulnerability
        # to each pitch category
        lineup_pt_records = []

        for (gpk, side), group in lu.groupby(["game_pk", "side"]):
            rec = {"game_pk": gpk, "side": side}

            for cat_name, cat_types in BROAD_CATEGORIES.items():
                cat_lower = cat_name.lower()
                whiff_rates = []
                k_rates = []
                weights = []

                for _, batter_row in group.iterrows():
                    pid = batter_row["player_id"]
                    pos = batter_row.get("lineup_position", 5)
                    weight = 10 - pos

                    # Look up this batter's stats vs this pitch category
                    batter_vs_cat = batter_latest[
                        (batter_latest["batter"] == pid) &
                        (batter_latest["pitch_type"].isin(cat_types))
                    ]

                    if len(batter_vs_cat) > 0:
                        wr = batter_vs_cat["batter_pt_whiff_rate"].mean()
                        kr = batter_vs_cat["batter_pt_k_rate"].mean()
                        if pd.notna(wr):
                            whiff_rates.append(wr)
                            weights.append(weight)
                        if pd.notna(kr):
                            k_rates.append(kr)

                if whiff_rates and weights:
                    w = np.array(weights, dtype=float)
                    wr = np.array(whiff_rates, dtype=float)
                    rec[f"lu_vs_{cat_lower}_whiff_rate"] = np.average(wr, weights=w)
                if k_rates:
                    rec[f"lu_vs_{cat_lower}_k_rate"] = np.mean(k_rates)

            lineup_pt_records.append(rec)

        if lineup_pt_records:
            lu_pt_agg = pd.DataFrame(lineup_pt_records)

            # Merge to main df (flip sides like lineup features)
            away_lu_pt = lu_pt_agg[lu_pt_agg["side"] == "away"].drop(columns=["side"])
            away_lu_pt = away_lu_pt.rename(columns={
                c: f"opp_{c}" for c in away_lu_pt.columns if c != "game_pk"
            })
            home_lu_pt = lu_pt_agg[lu_pt_agg["side"] == "home"].drop(columns=["side"])
            home_lu_pt = home_lu_pt.rename(columns={
                c: f"opp_{c}" for c in home_lu_pt.columns if c != "game_pk"
            })

            # Drop existing to avoid conflicts
            existing_lu_vs = [c for c in df.columns if "lu_vs_" in c]
            df = df.drop(columns=existing_lu_vs, errors="ignore")

            # Normalize game_pk before merge to prevent dtype drift
            normalize_game_pk(df, away_lu_pt, home_lu_pt)

            df_home = df[df["is_home"] == 1].merge(away_lu_pt, on="game_pk", how="left")
            df_away = df[df["is_home"] == 0].merge(home_lu_pt, on="game_pk", how="left")
            df = pd.concat([df_home, df_away], ignore_index=True)
            df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

            lu_vs_added = len([c for c in df.columns if "lu_vs_" in c])
            print(f"    ✓ Added {lu_vs_added} lineup pitch-type vulnerability features")

    # ── Part C: Cross features (pitcher strength × lineup weakness) ──
    cross_features_added = 0

    # Slider effectiveness × lineup vulnerability to breaking balls
    if "pt_sl_whiff_rate_L5" in df.columns and "opp_lu_vs_brk_whiff_rate" in df.columns:
        df["cross_sl_whiff_x_lu_brk"] = df["pt_sl_whiff_rate_L5"] * df["opp_lu_vs_brk_whiff_rate"]
        cross_features_added += 1

    # Fastball effectiveness × lineup vulnerability to fastballs
    if "pt_ff_whiff_rate_L5" in df.columns and "opp_lu_vs_fb_whiff_rate" in df.columns:
        df["cross_ff_whiff_x_lu_fb"] = df["pt_ff_whiff_rate_L5"] * df["opp_lu_vs_fb_whiff_rate"]
        cross_features_added += 1

    # Changeup effectiveness × lineup vulnerability to offspeed
    if "pt_ch_whiff_rate_L5" in df.columns and "opp_lu_vs_os_whiff_rate" in df.columns:
        df["cross_ch_whiff_x_lu_os"] = df["pt_ch_whiff_rate_L5"] * df["opp_lu_vs_os_whiff_rate"]
        cross_features_added += 1

    # Curveball effectiveness × lineup vulnerability to breaking
    if "pt_cu_whiff_rate_L5" in df.columns and "opp_lu_vs_brk_whiff_rate" in df.columns:
        df["cross_cu_whiff_x_lu_brk"] = df["pt_cu_whiff_rate_L5"] * df["opp_lu_vs_brk_whiff_rate"]
        cross_features_added += 1

    # Best secondary pitch whiff rate × lineup overall K rate
    secondary_whiff_cols = [c for c in df.columns if c.startswith("pt_") and "whiff_rate_L5" in c
                           and "ff_" not in c and "si_" not in c]
    if secondary_whiff_cols and "opp_lu_k_rate_wtd" in df.columns:
        df["cross_best_secondary_x_lu_k"] = df[secondary_whiff_cols].max(axis=1) * df["opp_lu_k_rate_wtd"]
        cross_features_added += 1

    # Pitcher fastball velo × lineup vulnerability to fastballs
    if "pt_ff_velo_L5" in df.columns and "opp_lu_vs_fb_whiff_rate" in df.columns:
        df["cross_ff_velo_x_lu_fb"] = df["pt_ff_velo_L5"] * df["opp_lu_vs_fb_whiff_rate"]
        cross_features_added += 1

    # Usage-weighted pitch effectiveness
    # If pitcher throws 30% sliders with 35% whiff rate vs a lineup that whiffs
    # 30% on breaking, that's more impactful than 5% slider usage
    for pt, cat in [("sl", "brk"), ("ch", "os"), ("cu", "brk"), ("ff", "fb")]:
        usage_col = f"pt_{pt}_usage_L5"
        whiff_col = f"pt_{pt}_whiff_rate_L5"
        lu_col = f"opp_lu_vs_{cat}_whiff_rate"
        if all(c in df.columns for c in [usage_col, whiff_col, lu_col]):
            # Normalize usage to fraction of total
            df[f"cross_{pt}_usage_weighted"] = df[usage_col] * df[whiff_col] * df[lu_col]
            cross_features_added += 1

    if cross_features_added > 0:
        print(f"    ✓ Added {cross_features_added} pitch-type cross/matchup features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLATOON FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_platoon_features(df):
    """
    Build pitcher platoon split features.
    How the pitcher performs against left-handed vs right-handed batters,
    using rolling windows.
    """
    print("  Building platoon split features...")

    if "p_throws" in df.columns:
        df["pitcher_throws_R"] = (df["p_throws"] == "R").astype(int)
    else:
        df["pitcher_throws_R"] = np.nan

    # Rolling platoon whiff rates (already have per-game whiff_pct_vs_left/right)
    for side in ["left", "right"]:
        col = f"whiff_pct_vs_{side}"
        if col in df.columns:
            for w in [5, 10]:
                shifted = df.groupby("pitcher")[col].shift(1)
                df[f"{col}_L{w}"] = shifted.groupby(df["pitcher"]).transform(
                    lambda x: x.rolling(w, min_periods=2).mean()
                )

    # Platoon differential (how much better/worse vs one side)
    if "whiff_pct_vs_left_L5" in df.columns and "whiff_pct_vs_right_L5" in df.columns:
        df["platoon_whiff_diff_L5"] = df["whiff_pct_vs_right_L5"] - df["whiff_pct_vs_left_L5"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# UMPIRE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_umpire_features(df, umpire_df):
    """
    Build home plate umpire tendency features.
    Some umpires have systematically larger/smaller strike zones,
    which affects K rates, walk rates, and called strike rates.
    """
    print("  Building umpire tendency features...")

    if umpire_df is None or len(umpire_df) == 0:
        print("  ⚠ No umpire data — skipping")
        return df

    # Merge umpire to games
    ump_subset = umpire_df[["game_pk", "hp_umpire_id", "hp_umpire_name"]].copy()
    normalize_game_pk(df, ump_subset)
    df = df.merge(ump_subset, on="game_pk", how="left")

    # Build cumulative umpire tendencies from the data itself
    # For each game, compute the umpire's historical called strike rate,
    # K rate, and BB rate across all previous games they've worked.
    if df["hp_umpire_id"].notna().sum() == 0:
        print("  ⚠ No umpire IDs matched — skipping tendencies")
        return df

    df = df.sort_values(["hp_umpire_id", "game_date"]).copy()

    # Aggregate per umpire per game (across all pitchers in that game)
    ump_game = df.groupby(["hp_umpire_id", "game_pk", "game_date", "season"]).agg(
        ump_total_pitches=("total_pitches", "sum"),
        ump_called_strikes=("called_strikes", "sum"),
        ump_strikeouts=("strikeouts", "sum"),
        ump_walks=("walks", "sum"),
        ump_pa=("plate_appearances", "sum"),
    ).reset_index()

    ump_game = ump_game.sort_values(["hp_umpire_id", "game_date"])

    # Cumulative umpire stats (shifted)
    for col in ["ump_total_pitches", "ump_called_strikes", "ump_strikeouts",
                "ump_walks", "ump_pa"]:
        shifted = ump_game.groupby("hp_umpire_id")[col].shift(1)
        ump_game[f"{col}_cum"] = shifted.groupby(ump_game["hp_umpire_id"]).cumsum()

    # Umpire rates
    utp = ump_game["ump_total_pitches_cum"].replace(0, np.nan)
    upa = ump_game["ump_pa_cum"].replace(0, np.nan)

    ump_game["ump_called_strike_pct"] = ump_game["ump_called_strikes_cum"] / utp
    ump_game["ump_k_pct"] = ump_game["ump_strikeouts_cum"] / upa
    ump_game["ump_bb_pct"] = ump_game["ump_walks_cum"] / upa

    # League average for normalization (we'll compute from the data)
    league_cs_pct = df["called_strikes"].sum() / df["total_pitches"].sum()
    league_k_pct = df["strikeouts"].sum() / df["plate_appearances"].replace(0, np.nan).sum()
    league_bb_pct = df["walks"].sum() / df["plate_appearances"].replace(0, np.nan).sum()

    ump_game["ump_cs_above_avg"] = ump_game["ump_called_strike_pct"] - league_cs_pct
    ump_game["ump_k_above_avg"] = ump_game["ump_k_pct"] - league_k_pct
    ump_game["ump_bb_above_avg"] = ump_game["ump_bb_pct"] - league_bb_pct

    # Merge back (one row per game_pk for umpire features)
    ump_features = ["game_pk", "ump_called_strike_pct", "ump_k_pct", "ump_bb_pct",
                    "ump_cs_above_avg", "ump_k_above_avg", "ump_bb_above_avg"]
    ump_merge = ump_game[[c for c in ump_features if c in ump_game.columns]].drop_duplicates("game_pk")

    # Drop any existing ump columns to avoid duplicates on re-merge
    ump_cols_existing = [c for c in df.columns if c.startswith("ump_") and c != "hp_umpire_id"]
    df = df.drop(columns=ump_cols_existing, errors="ignore")

    normalize_game_pk(df, ump_merge)
    df = df.merge(ump_merge, on="game_pk", how="left")
    print(f"  ✓ Added umpire tendency features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BALLPARK FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_park_features(df, park_factors_df, venue_df):
    """
    Add ballpark factor adjustments and venue characteristics.
    """
    print("  Building ballpark features...")

    if park_factors_df is not None and len(park_factors_df) > 0:
        # Park factors from FanGraphs — format varies by pybaseball version
        # Try to merge on team + season
        if "Team" in park_factors_df.columns and "Season" in park_factors_df.columns:
            # Rename for merge
            pf = park_factors_df.rename(columns={"Team": "park_team", "Season": "season"})
            # Key columns vary — look for strikeout-related factors
            factor_cols = [c for c in pf.columns if c not in ["park_team", "season"]]
            # Prefix them
            for c in factor_cols:
                pf = pf.rename(columns={c: f"pf_{c}"})
            factor_cols = [f"pf_{c}" for c in factor_cols]

            # Merge on the venue's home team
            if "home_team" in df.columns:
                df = df.merge(
                    pf[["park_team", "season"] + factor_cols],
                    left_on=["home_team", "season"],
                    right_on=["park_team", "season"],
                    how="left",
                )
                df.drop(columns=["park_team"], errors="ignore", inplace=True)
                print(f"  ✓ Added {len(factor_cols)} park factor columns")
    else:
        print("  ⚠ No park factors data — skipping")

    # Venue characteristics
    if venue_df is not None and "venue_name" in df.columns:
        df = df.merge(venue_df, on="venue_name", how="left")
        # Fill missing dome info with False
        if "is_dome_or_retractable" in df.columns:
            df["is_dome_or_retractable"] = df["is_dome_or_retractable"].fillna(0).astype(int)
        print(f"  ✓ Added venue characteristics")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_weather_features(df, weather_df):
    """
    Merge weather data and build weather-derived features.

    Key features:
      - Temperature, humidity, wind speed, gusts, precipitation
      - Binary flags: cold (<55°F), hot (>85°F), windy (>12mph), humid, rainy
      - Air density proxy (affects ball carry and pitch movement)
      - Interaction: velocity × air density (stuff in thin vs thick air)
    """
    print("  Building weather features...")

    if weather_df is None or len(weather_df) == 0:
        print("  ⚠ No weather data — skipping")
        return df

    # Select weather columns to merge (exclude venue_name to avoid conflicts)
    wx_cols = [c for c in weather_df.columns if c.startswith("wx_")]
    merge_cols = ["game_pk"] + wx_cols

    # Drop any existing weather columns to avoid duplicates
    existing_wx = [c for c in df.columns if c.startswith("wx_")]
    df = df.drop(columns=existing_wx, errors="ignore")

    # Deduplicate weather by game_pk
    wx_merge = weather_df[merge_cols].drop_duplicates(subset=["game_pk"], keep="first")

    normalize_game_pk(df, wx_merge)
    df = df.merge(wx_merge, on="game_pk", how="left")

    # ── Interaction features ──
    # Velocity × air density: how pitcher's stuff plays in current conditions
    if "avg_velocity_L5" in df.columns and "wx_air_density_proxy" in df.columns:
        df["wx_velo_x_density"] = df["avg_velocity_L5"] * df["wx_air_density_proxy"]

    # Wind × park factor (wind matters more in open-air parks)
    if "wx_wind_speed_mph" in df.columns and "wx_is_dome" in df.columns:
        df["wx_effective_wind"] = df["wx_wind_speed_mph"] * (1 - df["wx_is_dome"])

    # Temperature buckets for non-linear effects
    if "wx_temperature_f" in df.columns:
        df["wx_temp_bucket"] = pd.cut(
            df["wx_temperature_f"],
            bins=[0, 50, 60, 70, 80, 90, 120],
            labels=[1, 2, 3, 4, 5, 6],
        ).astype(float)

    wx_features = [c for c in df.columns if c.startswith("wx_")]
    print(f"  ✓ Added {len(wx_features)} weather features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# REST & SCHEDULING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule_features(df):
    """Rest days, day/night, day of week, month."""
    print("  Building schedule features...")

    df = df.sort_values(["pitcher", "game_date"]).copy()

    # Rest days
    df["prev_start_date"] = df.groupby("pitcher")["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - df["prev_start_date"]).dt.days
    df["rest_days"] = df["rest_days"].clip(upper=21).fillna(7)
    df.drop(columns=["prev_start_date"], inplace=True)

    # Rest day categories
    df["short_rest"] = (df["rest_days"] <= 4).astype(int)
    df["extra_rest"] = (df["rest_days"] >= 6).astype(int)

    # Day/night
    if "day_night" in df.columns:
        df["is_night_game"] = (df["day_night"] == "night").astype(int)
    else:
        df["is_night_game"] = np.nan

    # Day of week (0=Monday)
    df["day_of_week"] = df["game_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Month (captures seasonal effects — cold April, hot August)
    df["month"] = df["game_date"].dt.month

    return df


# ══════════════════════════════════════════════════════════════════════════════
# EARLY-SEASON CONTEXT & CARRYOVER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_early_season_features(df):
    """
    Features that help the model handle early-season uncertainty:

    1. Days into season — tells the model where we are in the season
    2. Feature reliability signals — how many starts back the rolling windows
       actually span (a "3-start rolling" in April might be 2 starts + padding)
    3. Prior-season carryover — last N starts of the PREVIOUS season as
       separate features so the model has a direct signal to lean on
    4. Pitcher-vs-team history — historical performance against today's opponent
    5. Season-phase interactions — key features crossed with days-into-season
       so the model learns to weight them differently in April vs August
    """
    print("  Building early-season context features...")

    df = df.sort_values(["pitcher", "game_date"]).copy()

    # ── 1. Days into season ──
    SEASON_STARTS = {
        2021: pd.Timestamp("2021-04-01"), 2022: pd.Timestamp("2022-04-07"),
        2023: pd.Timestamp("2023-03-30"), 2024: pd.Timestamp("2024-03-28"),
        2025: pd.Timestamp("2025-03-27"), 2026: pd.Timestamp("2026-03-26"),
    }

    def days_into_season(row):
        szn_start = SEASON_STARTS.get(row["season"])
        if szn_start is None:
            szn_start = pd.Timestamp(f"{int(row['season'])}-04-01")
        return (row["game_date"] - szn_start).days

    df["days_into_season"] = df.apply(days_into_season, axis=1).clip(lower=0)
    df["is_first_month"] = (df["days_into_season"] <= 30).astype(int)
    df["season_phase"] = pd.cut(
        df["days_into_season"], bins=[0, 30, 75, 120, 185],
        labels=[1, 2, 3, 4], include_lowest=True
    ).astype(float).fillna(4)

    # ── 2. Feature reliability: actual number of prior starts available ──
    df["prior_starts_available"] = df.groupby("pitcher").cumcount()
    df["prior_starts_this_season"] = df.groupby(["pitcher", "season"]).cumcount()

    # ── 3. Prior-season carryover ──
    # For each pitcher's starts in a new season, carry forward their
    # last 5 and 10 starts from the PREVIOUS season as separate features.
    print("    Building prior-season carryover stats...")

    carryover_stats = [
        "strikeouts", "k_pct", "whiff_pct", "csw_pct", "chase_rate",
        "avg_velocity", "avg_spin_rate", "barrel_pct", "hard_hit_pct",
        "k_per_100_pitches", "k_per_9",
        # ── K/BB/outs rate anchors (these were missing — meant bb_pct_prev10
        # was referenced in rate_configs but never actually built, which
        # silently broke the bb_pct_szn empirical-Bayes blend). ──
        "bb_pct", "k_bb_pct", "outs_per_pa",
        # ── Hits/walks pipeline additions ──
        # Mirror the K-side treatment so the H/W rate models get last-year's
        # last-N-starts as a stable prior the same way the K model does.
        "hits_allowed", "walks", "home_runs_allowed",
        "hits_per_pa", "bb_per_pa", "hr_per_pa",
        "h_per_9", "bb_per_9", "hr_per_9",
        "k_minus_bb_pct",
        "babip", "lob_pct",
        "gb_pct", "fb_pct", "ld_pct", "pop_pct",
        "soft_hit_pct",
        "avg_exit_velocity", "avg_launch_angle",
        "sweet_spot_pct", "solid_contact_pct",
        "avg_xba_contact", "avg_xwoba_contact",
    ]
    existing_carry = [c for c in carryover_stats if c in df.columns]

    for pitcher_id in df["pitcher"].unique():
        pmask = df["pitcher"] == pitcher_id
        pitcher_data = df[pmask].sort_values("game_date")

        seasons = sorted(pitcher_data["season"].unique())
        for i, season in enumerate(seasons):
            if i == 0:
                continue  # No prior season for the first one

            prev_season = seasons[i - 1]
            prev_data = pitcher_data[pitcher_data["season"] == prev_season]
            curr_mask = pmask & (df["season"] == season)

            if len(prev_data) == 0:
                continue

            # Last 5 and 10 starts of previous season
            for window, suffix in [(5, "prev5"), (10, "prev10")]:
                tail = prev_data.tail(window)
                for col in existing_carry:
                    val = tail[col].mean()
                    col_name = f"{col}_{suffix}"
                    if col_name not in df.columns:
                        df[col_name] = np.nan
                    df.loc[curr_mask, col_name] = val if pd.notna(val) else np.nan

    # ── 4. Pitcher vs team history ──
    print("    Building pitcher-vs-team history...")
    if "opp_team" in df.columns:
        # For each row, compute pitcher's historical stats vs this specific opponent
        pvt_records = []
        for (pitcher_id, opp), group in df.groupby(["pitcher", "opp_team"]):
            group = group.sort_values("game_date")
            # Shifted cumulative stats vs this opponent
            k_cum = group["strikeouts"].shift(1).expanding().mean()
            pa_cum = group["plate_appearances"].shift(1).expanding().mean()
            games_vs = group["strikeouts"].shift(1).expanding().count()

            for idx, val in k_cum.items():
                pvt_records.append({
                    "idx": idx,
                    "pvt_k_avg": val,
                    "pvt_games": games_vs.get(idx, 0),
                })

        if pvt_records:
            pvt_df = pd.DataFrame(pvt_records).set_index("idx")
            df["pvt_k_avg"] = pvt_df["pvt_k_avg"]
            df["pvt_games"] = pvt_df["pvt_games"].fillna(0)
            # Only trust the history if 3+ prior meetings
            df["pvt_k_avg_reliable"] = np.where(df["pvt_games"] >= 3, df["pvt_k_avg"], np.nan)
            print(f"    ✓ Pitcher-vs-team history added")

    # ── 5. Season-phase interactions ──
    # Key features × days_into_season so model can weight them differently
    interaction_cols = ["k_pct_L3", "whiff_pct_L3", "strikeouts_L3", "k_pct_szn"]
    for col in interaction_cols:
        if col in df.columns:
            # Reliability-weighted version: feature × (prior_starts / 10)
            # This naturally scales features down when few starts are available
            reliability = (df["prior_starts_this_season"] / 10).clip(upper=1.0)
            df[f"{col}_x_reliability"] = df[col] * reliability

    # Early-season flag × carryover interaction
    if "strikeouts_prev5" in df.columns:
        df["early_x_prev_k"] = df["is_first_month"] * df["strikeouts_prev5"].fillna(0)

    features_added = len([c for c in df.columns if any(
        x in c for x in ["days_into", "season_phase", "is_first_month",
                          "prior_starts", "_prev5", "_prev10", "pvt_",
                          "_x_reliability", "early_x_"]
    )])
    print(f"  ✓ Added {features_added} early-season context features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLENDED / SHRUNK RATE FEATURES (empirical-Bayes on season-to-date rates)
# ══════════════════════════════════════════════════════════════════════════════

def build_blended_rate_features(df):
    """
    Empirical-Bayes-shrunk season-to-date rates.

    Motivation: k_pct_szn resets to 0 obs each April, so after 3 starts a
    single bad outing can drop k_pct_szn by 8-10 percentage points. The
    existing L3/L5/L10 shrinkage in build_rolling_features() only kicks in
    when the WINDOW is under-filled (e.g. 1 obs in a 3-start window), NOT
    when a veteran has a full window but the season is young.

    For each season rate x_pct_szn we build x_pct_szn_blended:

        x_pct_szn_blended = (n_szn * x_pct_szn + PRIOR_PA * x_pct_prior) /
                            (n_szn + PRIOR_PA)

    where:
      - n_szn        = plate_appearances_szn (actual season PAs so far)
      - x_pct_prior  = pitcher's last-10-starts rate from PREVIOUS season
                       (i.e., x_pct_prev10), falling back to the pitcher's
                       own expanding career mean, then to league mean
      - PRIOR_PA     = prior-weight in PAs (default 100 PA ≈ 5 starts)

    The higher PRIOR_PA is, the slower the season rate moves away from the
    prior. 100 PA is a defensible choice: by mid-May a typical starter has
    ~150 PAs, so the blend is ~60/40 toward current-season — a healthy
    amount of recency without being dominated by one bad start.

    Similarly for rates backed by total_pitches (whiff, csw, barrel,
    hard_hit) we use a pitches-equivalent prior weight (~400 pitches ≈
    5 starts × 80 pitches).
    """
    print("  Building blended season-to-date rate features (empirical-Bayes)...")

    # Pitcher's own career expanding mean — used as fallback when _prev10 is NaN
    # (e.g., rookies, or after a pipeline change where carryover didn't fire).
    # Already leak-free: shift(1) before expanding().
    def career_prior(df_, rate_col):
        shifted = df_.groupby("pitcher")[rate_col].shift(1)
        return shifted.groupby(df_["pitcher"]).transform(
            lambda x: x.expanding().mean()
        )

    league_means_cache = {}
    def league_mean(col, default):
        if col not in league_means_cache:
            s = df[col].dropna()
            league_means_cache[col] = s.mean() if len(s) else default
        return league_means_cache[col]

    # Rate configs: (szn_col, prior_col_prev10, career_src_col, PRIOR_N, weight_col)
    # weight_col is the denominator used to measure "how much data backs the szn stat"
    PA_WEIGHT = 100.0  # empirical-Bayes prior strength in PAs (~5 starts)
    TP_WEIGHT = 400.0  # prior strength in pitches (~5 starts of 80 pitches)
    BB_WEIGHT = 80.0   # prior strength in batted balls (~5 starts of 16 BIP)

    rate_configs = [
        # (szn rate, prev10 rate, base rate col for career fallback, prior weight, denom col)
        ("k_pct_szn",        "k_pct_prev10",        "k_pct",        PA_WEIGHT, "plate_appearances_szn"),
        ("bb_pct_szn",       "bb_pct_prev10",       "bb_pct",       PA_WEIGHT, "plate_appearances_szn"),
        ("whiff_pct_szn",    "whiff_pct_prev10",    "whiff_pct",    TP_WEIGHT, "total_pitches_szn"),
        ("csw_pct_szn",      "csw_pct_prev10",      "csw_pct",      TP_WEIGHT, "total_pitches_szn"),
        ("barrel_pct_szn",   "barrel_pct_prev10",   "barrel_pct",   BB_WEIGHT, "batted_balls_szn"),
        ("hard_hit_pct_szn", "hard_hit_pct_prev10", "hard_hit_pct", BB_WEIGHT, "batted_balls_szn"),
        ("chase_rate_szn",   "chase_rate_prev10",   "chase_rate",   TP_WEIGHT, "total_pitches_szn"),
        # ── Hits / walks / HR additions (parallel to K-side above) ──
        # These give the H/W rate models the same empirical-Bayes shrinkage
        # treatment that solved the April 2026 recency-bias problem for K.
        # Without these, a starter's early-season hits_per_pa_szn after one
        # bad start is a wildly noisy estimate — the model needs the prior-
        # year anchor that _szn_blended provides.
        ("hits_per_pa_szn",  "hits_per_pa_prev10",  "hits_per_pa",  PA_WEIGHT, "plate_appearances_szn"),
        ("bb_per_pa_szn",    "bb_per_pa_prev10",    "bb_per_pa",    PA_WEIGHT, "plate_appearances_szn"),
        ("hr_per_pa_szn",    "hr_per_pa_prev10",    "hr_per_pa",    PA_WEIGHT, "plate_appearances_szn"),
        # BABIP needs longer-window stabilization (high variance, ~3x noisier
        # than K rate). Heavier prior (300 PAs ≈ 15 starts) prevents short-
        # run BABIP swings from leaking into hit-rate predictions.
        ("babip_szn",        "babip_prev10",        "babip",        300.0,     "plate_appearances_szn"),
        ("lob_pct_szn",      "lob_pct_prev10",      "lob_pct",      PA_WEIGHT, "plate_appearances_szn"),
        # Batted-ball type blends — denominator is batted_balls_szn since
        # these are computed as count/BIP. Weight comparable to barrel%.
        ("gb_pct_szn",       "gb_pct_prev10",       "gb_pct",       BB_WEIGHT, "batted_balls_szn"),
        ("fb_pct_szn",       "fb_pct_prev10",       "fb_pct",       BB_WEIGHT, "batted_balls_szn"),
        ("ld_pct_szn",       "ld_pct_prev10",       "ld_pct",       BB_WEIGHT, "batted_balls_szn"),
        ("pop_pct_szn",      "pop_pct_prev10",      "pop_pct",      BB_WEIGHT, "batted_balls_szn"),
        ("soft_hit_pct_szn", None,                  "soft_hit_pct", BB_WEIGHT, "batted_balls_szn"),
        # Quality-of-contact blends — weighted by batted_balls (same as barrel%)
        ("avg_exit_velocity_szn",  "avg_exit_velocity_prev10",  "avg_exit_velocity",  BB_WEIGHT, "batted_balls_szn"),
        ("avg_launch_angle_szn",   "avg_launch_angle_prev10",   "avg_launch_angle",   BB_WEIGHT, "batted_balls_szn"),
        ("sweet_spot_pct_szn",     "sweet_spot_pct_prev10",     "sweet_spot_pct",     BB_WEIGHT, "batted_balls_szn"),
        ("solid_contact_pct_szn",  "solid_contact_pct_prev10",  "solid_contact_pct",  BB_WEIGHT, "batted_balls_szn"),
        ("avg_xba_contact_szn",    "avg_xba_contact_prev10",    "avg_xba_contact",    BB_WEIGHT, "batted_balls_szn"),
        ("avg_xwoba_contact_szn",  "avg_xwoba_contact_prev10",  "avg_xwoba_contact",  BB_WEIGHT, "batted_balls_szn"),
    ]

    added = 0
    for szn_col, prev_col, base_col, prior_n, denom_col in rate_configs:
        if szn_col not in df.columns:
            continue
        if denom_col not in df.columns:
            # Can't compute blend without the denominator count; skip
            continue

        # Build the prior rate in order of preference:
        #   1) prev10 carryover (last 10 starts of last season)
        #   2) pitcher's own career expanding mean
        #   3) league mean
        prior = pd.Series(np.nan, index=df.index, dtype=float)
        if prev_col in df.columns:
            prior = df[prev_col].astype(float)
        if base_col in df.columns:
            career = career_prior(df, base_col)
            prior = prior.fillna(career)
        lm = league_mean(szn_col, 0.22 if "k_pct" in szn_col else 0.1)
        prior = prior.fillna(lm)

        n_szn = df[denom_col].fillna(0).clip(lower=0)
        szn_val = df[szn_col].astype(float)
        # When szn_val is NaN (no season PAs yet), treat it as 0-obs — prior dominates
        szn_val_filled = szn_val.fillna(prior)

        blended = (n_szn * szn_val_filled + prior_n * prior) / (n_szn + prior_n)
        out_col = f"{szn_col}_blended"
        df[out_col] = blended
        added += 1

    # Also build a blended L5 that uses prior-year as the shrinkage target.
    # This helps when a pitcher has 3-4 starts — L5 is "full enough" to skip
    # the existing window-fill shrinkage in build_rolling_features, but still
    # too few starts to be reliable on its own.
    L5_PRIOR = 5.0  # 5-start prior weight on L5 (equal weighting by start count)
    l5_configs = [
        ("k_pct_L5",        "k_pct_prev10"),
        ("whiff_pct_L5",    "whiff_pct_prev10"),
        ("csw_pct_L5",      "csw_pct_prev10"),
        ("barrel_pct_L5",   "barrel_pct_prev10"),
        ("hard_hit_pct_L5", "hard_hit_pct_prev10"),
        # ── H/W pipeline additions ──
        # L5-blended versions for the same hits-side stats. These help
        # mid-April when a starter has 3-4 starts and L5 is just barely
        # informative on its own.
        ("hits_per_pa_L5",  "hits_per_pa_prev10"),
        ("bb_per_pa_L5",    "bb_per_pa_prev10"),
        ("hr_per_pa_L5",    "hr_per_pa_prev10"),
        ("babip_L5",        "babip_prev10"),
        ("lob_pct_L5",      "lob_pct_prev10"),
        ("gb_pct_L5",       "gb_pct_prev10"),
        ("fb_pct_L5",       "fb_pct_prev10"),
        ("ld_pct_L5",       "ld_pct_prev10"),
        ("avg_exit_velocity_L5",  "avg_exit_velocity_prev10"),
        ("sweet_spot_pct_L5",     "sweet_spot_pct_prev10"),
        ("avg_xwoba_contact_L5",  "avg_xwoba_contact_prev10"),
    ]
    if "n_starts_in_L5" in df.columns:
        for l5_col, prev_col in l5_configs:
            if l5_col not in df.columns:
                continue
            n_actual = df["n_starts_in_L5"].fillna(0).clip(lower=0, upper=5)
            prior = pd.Series(np.nan, index=df.index, dtype=float)
            if prev_col in df.columns:
                prior = df[prev_col].astype(float)
            base_col = l5_col.replace("_L5", "")
            if base_col in df.columns:
                prior = prior.fillna(career_prior(df, base_col))
            prior = prior.fillna(league_mean(l5_col, 0.22))

            l5_val = df[l5_col].astype(float).fillna(prior)
            blended = (n_actual * l5_val + L5_PRIOR * prior) / (n_actual + L5_PRIOR)
            df[f"{l5_col}_blended"] = blended
            added += 1

    print(f"  ✓ Added {added} blended rate features "
          f"(_szn_blended, _L5_blended)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# OPPOSING STARTER FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_opposing_starter_features(df):
    """
    Add features about the opposing starting pitcher.

    When both starters in a game are aces, both tend to go deeper (more BF,
    more outs, more Ks) because the game stays close and managers leave them
    in. The model currently has NO features about the opposing starter, only
    the opposing batting lineup. This causes correlated over/under errors
    where both pitchers in a game miss in the same direction.

    For each pitcher's game row, we look up the opposing starter's recent
    rolling stats from the SAME feature matrix (using shifted/lagged values
    so no leakage).
    """
    print("  Building opposing starter features...")

    if "home_starter_id" not in df.columns or "away_starter_id" not in df.columns:
        print("    ⚠ Missing starter IDs — skipping opposing starter features")
        return df

    # Identify the opposing starter for each row
    df["opp_starter_id"] = np.where(
        df["is_home"] == 1,
        df["away_starter_id"],
        df["home_starter_id"],
    )

    # Stats we want to look up for the opposing starter.
    # These use L5 (5-start rolling average) as a balance between
    # recency and stability. All are already lag-shifted (shift(1))
    # so they represent stats BEFORE the current game — no leakage.
    opp_stats_to_lookup = {
        # Performance quality
        "k_pct_L5": "opp_sp_k_pct",
        "bb_pct_L5": "opp_sp_bb_pct",
        "whiff_pct_L5": "opp_sp_whiff_pct",
        "csw_pct_L5": "opp_sp_csw_pct",
        "barrel_pct_L5": "opp_sp_barrel_pct",
        "hard_hit_pct_L5": "opp_sp_hard_hit_pct",
        # Depth / efficiency (directly affects game pace for BOTH pitchers)
        "plate_appearances_L5": "opp_sp_bf_avg",
        "est_innings_L5": "opp_sp_ip_avg",
        "is_short_outing_L5": "opp_sp_short_pct",
        "outs_recorded_L5": "opp_sp_outs_avg",
        "outs_per_pa_L5": "opp_sp_outs_rate",
        # Pitch stuff (velocity indicates starter quality tier)
        "avg_velocity_L5": "opp_sp_velo",
    }

    # Also grab L10 depth stats for more stability
    opp_depth_L10 = {
        "plate_appearances_L10": "opp_sp_bf_avg_L10",
        "est_innings_L10": "opp_sp_ip_avg_L10",
        "is_short_outing_L10": "opp_sp_short_pct_L10",
        "outs_recorded_L10": "opp_sp_outs_avg_L10",
    }
    opp_stats_to_lookup.update(opp_depth_L10)

    # Build a lookup table: for each (pitcher, game_pk), get the desired stats.
    # The rolling features are already lagged (shift(1) in build_rolling_features),
    # so looking up by pitcher + game_pk gives pre-game values = no leakage.
    needed_cols = ["pitcher", "game_pk"] + [
        c for c in opp_stats_to_lookup.keys() if c in df.columns
    ]
    lookup = df[needed_cols].copy()
    lookup = lookup.rename(columns={"pitcher": "opp_starter_id"})

    # Rename the stat columns to opp_sp_ prefixed names
    rename_map = {
        orig: new_name
        for orig, new_name in opp_stats_to_lookup.items()
        if orig in lookup.columns
    }
    lookup = lookup.rename(columns=rename_map)

    # Merge: for each row, look up the opposing starter's stats in the same game
    opp_cols = list(rename_map.values())
    merge_cols = ["opp_starter_id", "game_pk"] + opp_cols

    # Drop any opp_sp_ columns that already exist to avoid _x/_y suffixes
    existing_opp = [c for c in df.columns if c.startswith("opp_sp_")]
    if existing_opp:
        df = df.drop(columns=existing_opp, errors="ignore")

    df = df.merge(
        lookup[merge_cols],
        on=["opp_starter_id", "game_pk"],
        how="left",
    )

    # ── Interaction features: pitcher quality × opposing pitcher depth ──
    # When both starters are deep-inning guys, game goes longer for both
    if "plate_appearances_L5" in df.columns and "opp_sp_bf_avg" in df.columns:
        df["ix_both_deep"] = df["plate_appearances_L5"] * df["opp_sp_bf_avg"] / 500.0
        # Normalized so ~22 * 22 / 500 ≈ 1.0 for average starters

    if "k_pct_L5" in df.columns and "opp_sp_k_pct" in df.columns:
        df["ix_aces_matchup"] = df["k_pct_L5"] * df["opp_sp_k_pct"]
        # High when both starters are K-dominant → pitcher's duel

    if "est_innings_L5" in df.columns and "opp_sp_ip_avg" in df.columns:
        df["ix_combined_depth"] = df["est_innings_L5"] + df["opp_sp_ip_avg"]
        # Total expected IP from both starters → game pace proxy

    filled = df[opp_cols].notna().sum()
    total = len(df)
    fill_rates = {col: f"{filled[col] / total:.1%}" for col in opp_cols[:5]}
    print(f"    ✓ Added {len(opp_cols)} opposing starter features + 3 interactions")
    print(f"    Fill rates (sample): {fill_rates}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FANGRAPHS PITCHER FEATURES (Stuff+, Location+, Pitching+, SwStr%, etc.)
# ══════════════════════════════════════════════════════════════════════════════

def build_fangraphs_pitcher_features(df, fg_pitching):
    """
    Merge FanGraphs season-level pitcher metrics that aren't in Statcast.
    Key additions:
      - Stuff+, Location+, Pitching+ (overall and per pitch type)
      - O-Swing%, Z-Swing%, Contact%, SwStr% (season-level plate discipline)
      - TTO% (three true outcomes rate)
      - Per-pitch Stuff+ for primary pitches (FA, SL, CH, CU, SI, FC)
    """
    print("  Building FanGraphs pitcher features...")

    if fg_pitching is None or len(fg_pitching) == 0:
        print("  ⚠ No FanGraphs pitching data — skipping")
        return df

    fg = fg_pitching.copy()

    # Map FanGraphs pitcher ID to Statcast pitcher ID (MLBAM ID).
    # The load-time helper backfill_fg_mlbamid() has already filled in
    # MLBAMID from the standalone 2026 files via the IDfg bridge, so by
    # the time we get here MLBAMID is populated for most rows. We just
    # need to handle the remaining stragglers with the game-metadata
    # name crosswalk.

    # Step 1: take whatever MLBAMID the file (post-backfill) provides.
    if "xMLBAMID" in fg.columns and "MLBAMID" not in fg.columns:
        fg = fg.rename(columns={"xMLBAMID": "MLBAMID"})
    if "MLBAMID" in fg.columns:
        fg["pitcher"] = pd.to_numeric(fg["MLBAMID"], errors="coerce")
    else:
        fg["pitcher"] = np.nan

    # Step 2: Name + season → MLBAM ID via game_metadata_all.csv (last resort
    # for players the IDfg bridge couldn't cover — typically pitchers who
    # didn't appear in the 2026 standalone file).
    if fg["pitcher"].isna().any() and "Name" in fg.columns and "Season" in fg.columns:
        meta_path = DATA_DIR / "game_metadata_all.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            home = meta[["home_starter_id", "home_starter_name", "season"]].dropna(
                subset=["home_starter_id"])
            home = home.rename(columns={"home_starter_id": "pitcher_meta",
                                        "home_starter_name": "Name"})
            away = meta[["away_starter_id", "away_starter_name", "season"]].dropna(
                subset=["away_starter_id"])
            away = away.rename(columns={"away_starter_id": "pitcher_meta",
                                        "away_starter_name": "Name"})
            crosswalk = pd.concat([
                home[["Name", "pitcher_meta", "season"]],
                away[["Name", "pitcher_meta", "season"]],
            ])
            crosswalk["pitcher_meta"] = pd.to_numeric(
                crosswalk["pitcher_meta"], errors="coerce")
            crosswalk = crosswalk.dropna(subset=["pitcher_meta"]).drop_duplicates(
                subset=["Name", "season"])
            crosswalk = crosswalk.rename(columns={"season": "Season"})
            n_before = fg["pitcher"].notna().sum()
            fg = fg.merge(crosswalk, on=["Name", "Season"], how="left")
            fg["pitcher"] = fg["pitcher"].fillna(fg["pitcher_meta"])
            fg = fg.drop(columns=["pitcher_meta"])
            n_after = fg["pitcher"].notna().sum()
            if n_after > n_before:
                print(f"    After game-metadata name crosswalk: "
                      f"+{n_after - n_before:,} → {n_after:,}/{len(fg):,} rows have pitcher id")

    fg = fg.dropna(subset=["pitcher"])
    fg["pitcher"] = fg["pitcher"].astype(int)
    print(f"    FG pitching usable rows after id resolution: {len(fg):,}")

    # Select features to merge
    fg_features = {}

    # Overall quality metrics
    for col, new_name in [
        ("Stuff+", "fg_stuff_plus"),
        ("Location+", "fg_location_plus"),
        ("Pitching+", "fg_pitching_plus"),
        ("TTO%", "fg_tto_pct"),
        ("FRM", "fg_pitcher_frm"),  # pitcher's own FRM (rare but present)
    ]:
        if col in fg.columns:
            fg_features[col] = new_name

    # Plate discipline (pitcher perspective — how often batters swing/miss vs this pitcher)
    for col, new_name in [
        ("SwStr%", "fg_swstr_pct"),
        ("O-Swing%", "fg_o_swing_pct"),
        ("Z-Swing%", "fg_z_swing_pct"),
        ("Contact%", "fg_contact_pct"),
        ("O-Contact%", "fg_o_contact_pct"),
        ("Z-Contact%", "fg_z_contact_pct"),
        ("Zone%", "fg_zone_pct_szn"),
        ("F-Strike%", "fg_first_strike_pct"),
    ]:
        if col in fg.columns:
            fg_features[col] = new_name

    # Per-pitch-type Stuff+ (the most predictive individual features)
    for pt in ["FA", "SL", "CH", "CU", "SI", "FC", "FS"]:
        col = f"Stf+ {pt}"
        if col in fg.columns:
            fg_features[col] = f"fg_stuff_{pt.lower()}"
        col_loc = f"Loc+ {pt}"
        if col_loc in fg.columns:
            fg_features[col_loc] = f"fg_loc_{pt.lower()}"

    # ── Hits/walks-side FanGraphs metrics (parallel to Stuff+/Contact% above) ──
    # These are season-level summary stats that the K model doesn't need but
    # which carry massive signal for hits/walks. Each is gated on column
    # existence so older FG pulls still work.
    #
    # - FIP / xFIP / SIERA: ERA estimators that strip out luck. Predict
    #   true talent better than ERA. SIERA in particular weights GB% and
    #   K-BB% to predict run prevention; pitchers with low SIERA suppress
    #   hits beyond what their K rate alone would suggest.
    # - LOB%: Strand rate. Pitchers with high LOB% allow fewer of their
    #   baserunners to score, which correlates with hit-cluster suppression.
    # - HR/FB: Home-run-per-fly-ball rate. Direct input to HR prediction
    #   and an indicator of "true" HR-suppression skill (vs lucky).
    # - K-BB%: K rate minus BB rate, the cleanest single talent indicator.
    # - GB% / FB% / LD% / IFFB%: Batted-ball mix. Ground-ball pitchers
    #   suppress HR but allow more BABIP-driven hits; fly-ball pitchers
    #   the opposite. Direct input to hit/HR rate.
    # - BABIP / HR/9 / BB/9 / K/9: rate stats normalized per inning.
    for col, new_name in [
        ("FIP",       "fg_fip"),
        ("xFIP",      "fg_xfip"),
        ("SIERA",     "fg_siera"),
        ("tERA",      "fg_tera"),
        ("LOB%",      "fg_lob_pct"),
        ("HR/FB",     "fg_hr_per_fb"),
        ("K-BB%",     "fg_k_minus_bb_pct"),
        ("K/9",       "fg_k_per_9"),
        ("BB/9",      "fg_bb_per_9"),
        ("HR/9",      "fg_hr_per_9"),
        ("GB%",       "fg_gb_pct"),
        ("FB%",       "fg_fb_pct"),
        ("LD%",       "fg_ld_pct"),
        ("IFFB%",     "fg_iffb_pct"),
        ("BABIP",     "fg_babip_allowed"),
        ("Soft%",     "fg_soft_pct"),
        ("Med%",      "fg_med_pct"),
        ("Hard%",     "fg_hard_pct_allowed"),
        ("Barrel%",   "fg_barrel_pct_allowed"),
        ("HardHit%",  "fg_hard_hit_pct_allowed"),
        ("xERA",      "fg_xera"),
        ("ERA",       "fg_era"),
        ("WHIP",      "fg_whip"),
    ]:
        if col in fg.columns:
            fg_features[col] = new_name

    if not fg_features:
        print("  ⚠ No usable FanGraphs columns found — skipping")
        return df

    # Build merge frame: pitcher + season + selected features
    merge_cols = ["pitcher", "Season"] + list(fg_features.keys())
    merge_cols = [c for c in merge_cols if c in fg.columns]
    fg_slim = fg[merge_cols].copy()
    fg_slim = fg_slim.rename(columns=fg_features)
    fg_slim = fg_slim.rename(columns={"Season": "season"})

    # Convert percentage strings to floats if needed
    for col in fg_slim.columns:
        if fg_slim[col].dtype == object:
            try:
                fg_slim[col] = fg_slim[col].str.rstrip("%").astype(float) / 100.0
            except (ValueError, AttributeError):
                pass

    # ── CRITICAL: split FG features into "stable" vs "outcome-aggregate" ──
    # The FG file is one row per (pitcher, season) with full-season totals.
    # Merging on (pitcher, season) gives the model full-season aggregates
    # INCLUDING the game we're predicting. That's same-season leakage.
    #
    # We split FG features into two groups based on whether they're direct
    # outcome aggregates or process metrics:
    #
    #   STABLE (merge same-season — they don't encode outcome counts):
    #     - Stuff+/Location+/Pitching+ : pitch quality (release/spin/movement)
    #     - SwStr%, Contact%, O-Swing%, Z-Contact% : plate discipline (process)
    #     - TTO%, Pitcher framing
    #
    #   OUTCOME (lag by 1 season — these ARE the answer at season level):
    #     - FIP, xFIP, SIERA, ERA, xERA, tERA, WHIP : ERA estimators
    #     - K/9, BB/9, HR/9, K-BB% : direct K/BB/HR rates per inning
    #     - LOB%, HR/FB : outcome rates
    #     - GB%, FB%, LD%, IFFB% : batted-ball mix (direct outcomes)
    #     - BABIP, Soft%, Med%, Hard%, Barrel%, HardHit% (allowed) : contact outcomes
    #
    # The outcome group becomes a previous-season prior anchor.
    outcome_feature_names = {
        "fg_fip", "fg_xfip", "fg_siera", "fg_tera", "fg_xera",
        "fg_era", "fg_whip", "fg_lob_pct", "fg_hr_per_fb",
        "fg_k_minus_bb_pct", "fg_k_per_9", "fg_bb_per_9", "fg_hr_per_9",
        "fg_gb_pct", "fg_fb_pct", "fg_ld_pct", "fg_iffb_pct",
        "fg_babip_allowed", "fg_soft_pct", "fg_med_pct",
        "fg_hard_pct_allowed", "fg_barrel_pct_allowed", "fg_hard_hit_pct_allowed",
    }
    stable_cols  = [c for c in fg_slim.columns
                    if c in ("pitcher", "season") or c not in outcome_feature_names]
    outcome_cols = [c for c in fg_slim.columns if c in outcome_feature_names]

    # Merge stable FG features same-season (existing behavior — these are
    # season-level process metrics, not outcome aggregates).
    pre_cols = len(df.columns)
    if stable_cols and len(stable_cols) > 2:  # >2 means we have features beyond pitcher+season
        df = df.merge(fg_slim[stable_cols], on=["pitcher", "season"], how="left")

    # Merge outcome FG features as previous-season prior anchors.
    # Renaming with _prev suffix and shifting the season key by 1 means
    # season-2024 outcomes become features for season-2025 predictions,
    # never contaminating the same-season target.
    if outcome_cols:
        fg_outcome = fg_slim[["pitcher", "season"] + outcome_cols].copy()
        fg_outcome["season"] = fg_outcome["season"] + 1  # last year's stats → this year's prior
        fg_outcome = fg_outcome.rename(columns={c: f"{c}_prev" for c in outcome_cols})
        df = df.merge(fg_outcome, on=["pitcher", "season"], how="left")

    new_cols = len(df.columns) - pre_cols

    # Fill NaN with league averages for Stuff+/Location+/Pitching+ (centered at 100)
    for col in df.columns:
        if col.startswith("fg_stuff_") or col.startswith("fg_loc_"):
            df[col] = df[col].fillna(100.0)
        elif col == "fg_stuff_plus":
            df[col] = df[col].fillna(100.0)
        elif col == "fg_location_plus":
            df[col] = df[col].fillna(100.0)
        elif col == "fg_pitching_plus":
            df[col] = df[col].fillna(100.0)

    # ── Hits-side FanGraphs feature defaults (league averages) ──
    # XGBoost handles NaN fine, but supplying league-average defaults
    # gives the model a meaningful "no FG data" baseline rather than
    # treating NaN as a discrete category. These match approx 2024
    # MLB-wide pitcher averages.
    #
    # Note: the outcome aggregates use the _prev suffix because they're
    # lagged by 1 season to prevent same-season leakage. See the comment
    # in the merge block above.
    fg_defaults = {
        "fg_fip_prev":                 4.20,
        "fg_xfip_prev":                4.20,
        "fg_siera_prev":               4.20,
        "fg_tera_prev":                4.20,
        "fg_xera_prev":                4.20,
        "fg_era_prev":                 4.20,
        "fg_whip_prev":                1.30,
        "fg_lob_pct_prev":             0.72,
        "fg_hr_per_fb_prev":           0.115,
        "fg_k_minus_bb_pct_prev":      0.135,
        "fg_k_per_9_prev":             8.5,
        "fg_bb_per_9_prev":            3.1,
        "fg_hr_per_9_prev":            1.15,
        "fg_gb_pct_prev":              0.43,
        "fg_fb_pct_prev":              0.36,
        "fg_ld_pct_prev":              0.21,
        "fg_iffb_pct_prev":            0.10,
        "fg_babip_allowed_prev":       0.295,
        "fg_soft_pct_prev":            0.17,
        "fg_med_pct_prev":             0.49,
        "fg_hard_pct_allowed_prev":    0.34,
        "fg_barrel_pct_allowed_prev":  0.07,
        "fg_hard_hit_pct_allowed_prev":0.35,
    }
    for col, default in fg_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Interaction: Stuff+ × lineup K rate
    if "fg_stuff_plus" in df.columns and "opp_lu_k_rate_wtd" in df.columns:
        df["ix_stuff_x_lu_k"] = (df["fg_stuff_plus"] / 100.0) * df["opp_lu_k_rate_wtd"].fillna(0.22)

    # Interaction: SwStr% × Contact% (pitcher dominance score)
    if "fg_swstr_pct" in df.columns and "fg_contact_pct" in df.columns:
        df["ix_swstr_x_contact"] = df["fg_swstr_pct"].fillna(0.11) * (1 - df["fg_contact_pct"].fillna(0.78))

    print(f"  ✓ Added {new_cols} FanGraphs pitcher features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CATCHER FRAMING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_catcher_framing_features(df, catchers_df, fg_batting):
    """
    Merge catcher framing (FRM from FanGraphs) to each pitcher-game.
    The pitcher's catcher directly affects called strikes → strikeouts.
    Elite framers add ~15 extra called strikes per 100 borderline pitches.
    """
    print("  Building catcher framing features...")

    if catchers_df is None or len(catchers_df) == 0:
        print("  ⚠ No catcher data — skipping")
        return df

    if fg_batting is None or len(fg_batting) == 0:
        print("  ⚠ No FanGraphs batting data for FRM lookup — skipping")
        return df

    fg = fg_batting.copy()

    # Get catcher FRM values per season
    # FRM is only populated for catchers — non-catchers have NaN
    if "FRM" not in fg.columns:
        print("  ⚠ No FRM column in FanGraphs batting — skipping")
        return df

    # Need MLBAM ID for catchers — but FanGraphs CSVs are inconsistent about
    # which ID column they include, and historical re-exports may have NaN
    # MLBAMID for older seasons. We do this in two passes:
    #   Pass 1: rename whichever ID column exists to "catcher_id".
    #   Pass 2: for rows where catcher_id is still NaN, fall back to a
    #           name-based join with game_catchers.csv (which always has
    #           a real MLBAM catcher_id from the MLB Stats API).
    if "xMLBAMID" in fg.columns:
        fg = fg.rename(columns={"xMLBAMID": "catcher_id"})
    elif "MLBAMID" in fg.columns:
        fg = fg.rename(columns={"MLBAMID": "catcher_id"})
    else:
        fg["catcher_id"] = pd.NA

    fg["catcher_id"] = pd.to_numeric(fg["catcher_id"], errors="coerce")

    # Name-based backfill for any rows that didn't get a usable ID. This
    # used to be a fallback only when both ID columns were entirely missing,
    # but in practice some re-exports populate MLBAMID for current season
    # and leave older seasons NaN — so we always run the backfill, then
    # only drop rows where it fails.
    if "Name" in fg.columns and "catcher_name" in catchers_df.columns:
        cat_names = catchers_df[["catcher_id", "catcher_name"]].dropna()\
                                                                .drop_duplicates("catcher_name")
        cat_names = cat_names.rename(columns={"catcher_id": "catcher_id_from_name"})
        fg = fg.merge(cat_names, left_on="Name", right_on="catcher_name", how="left")
        # Where the original catcher_id is NaN, use the name-based one
        fg["catcher_id"] = fg["catcher_id"].fillna(
            pd.to_numeric(fg["catcher_id_from_name"], errors="coerce")
        )
        fg = fg.drop(columns=["catcher_id_from_name", "catcher_name"], errors="ignore")

    # The warning compares against FRM-having rows specifically. Comparing
    # against ALL FG batting rows (which includes ~3500 non-catcher position
    # players) would always look alarming because we're not trying to match
    # those — they don't have FRM and aren't relevant for this feature.
    n_frm_rows = fg["FRM"].notna().sum() if "FRM" in fg.columns else len(fg)
    fg = fg.dropna(subset=["catcher_id", "FRM"])
    n_with_id = len(fg)
    if n_frm_rows > 0 and n_with_id < int(n_frm_rows * 0.85):
        print(f"  ⚠ Only {n_with_id}/{n_frm_rows} FG catcher rows (with FRM) "
              f"have a usable catcher_id after name backfill — "
              f"FRM coverage will be limited")
    fg["catcher_id"] = fg["catcher_id"].astype(int)

    # Keep only rows with non-null FRM (catchers)
    catcher_frm = fg[["catcher_id", "Season", "FRM"]].copy()
    catcher_frm = catcher_frm.rename(columns={"Season": "season", "FRM": "catcher_frm"})

    # Build catcher assignments per game
    cats = catchers_df.copy()
    cats["catcher_id"] = pd.to_numeric(cats["catcher_id"], errors="coerce").astype("Int64")

    # Merge game season from main df
    game_seasons = df[["game_pk", "season"]].drop_duplicates("game_pk")
    cats = cats.merge(game_seasons, on="game_pk", how="left")

    # ── Leakage fix: use PRIOR-season FRM, never current-season ──────────────
    # FanGraphs FRM is published as a season-cumulative number, so a June 2025
    # game's "current-season FRM" was computed using games through October.
    # That's leakage. Framing tendencies are stable year-over-year (~0.7
    # year-to-year correlation), so prior-season FRM is a fine substitute
    # with zero leakage. For first-year catchers with no prior FRM, default
    # to 0 (league average). The catcher_borderline_strike_pct_delta feature
    # from build_catcher_calling_features captures within-season as-of
    # framing-adjacent signal cleanly, so we don't lose anything by switching
    # FRM here to prior-only.
    cats["frm_lookup_season"] = cats["season"] - 1
    cats = cats.merge(
        catcher_frm.rename(columns={"season": "frm_lookup_season"}),
        on=["catcher_id", "frm_lookup_season"], how="left",
    )
    cats = cats.drop(columns=["frm_lookup_season"])

    # Default FRM = 0 (league average) for catchers with no prior-season data
    cats["catcher_frm"] = cats["catcher_frm"].fillna(0.0)

    # Pitcher faces the OPPOSING catcher:
    # Home pitcher faces home catcher (catcher is on their team)
    # Wait — the catcher is on the SAME team as the pitcher, not opposing.
    # Framing helps the pitcher on the SAME team.
    home_catchers = cats[cats["side"] == "home"][["game_pk", "catcher_id", "catcher_frm"]].copy()
    away_catchers = cats[cats["side"] == "away"][["game_pk", "catcher_id", "catcher_frm"]].copy()

    # Normalize game_pk before merge to prevent dtype drift
    normalize_game_pk(df, home_catchers, away_catchers)

    # Home pitchers get home catchers
    df_home = df[df["is_home"] == 1].merge(
        home_catchers, on="game_pk", how="left", suffixes=("", "_catcher")
    )
    # Away pitchers get away catchers
    df_away = df[df["is_home"] == 0].merge(
        away_catchers, on="game_pk", how="left", suffixes=("", "_catcher")
    )

    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    df["catcher_frm"] = df["catcher_frm"].fillna(0.0)

    n_with_frm = (df["catcher_frm"] != 0).sum()
    print(f"  ✓ Added catcher_frm ({n_with_frm:,}/{len(df):,} games with non-zero FRM)")

    return df


# ── Catcher game-calling features (from collect/catcher.py) ──────
CATCHER_CALLING_FEATURES = [
    "catcher_breaking_pct_delta",
    "catcher_breaking_pct_2k_delta",
    "catcher_breaking_pct_first_pitch_delta",
    "catcher_zone_pct_delta",
    "catcher_chase_pct_delta",
    "catcher_csw_pct_delta",
    "catcher_borderline_strike_pct_delta",
]


def build_catcher_calling_features(df, catchers_df, asof_df, prior_df):
    """
    Join leakage-free catcher game-calling features (from
    collect/catcher.py) onto each pitcher-game row.

    Two inputs from 01c:
      asof_df: one row per (catcher_id, game_date) with features computed
               from pitches STRICTLY BEFORE that date in the same season.
               No leakage by construction.
      prior_df: one row per (catcher_id, season) with full-season summary.
               Used as the prior-season anchor when blending.

    Join logic per pitcher-game:
      1. Identify the starting catcher via game_catchers.csv.
      2. merge_asof to find the most recent as-of row at or before
         (game_date - 1 day) — guarantees no same-day leakage even if a
         catcher appears in both files for the same date.
      3. Look up the prior file at season S-1.
      4. Size-weighted blend:
            weight_current = n_current / (n_current + 0.5 * n_prior)
            final = weight_current * current + (1 - weight_current) * prior
         The 0.5 factor on n_prior weights current data ~2x per pitch.
      5. Edge cases: missing as-of (April or rookie call-up) → use prior
         only. Missing prior (first-year catcher) → use as-of only.
         Missing both → 0 (the league-mean effect by construction).

    Result: 7 new columns appended to df. Nothing existing is modified.
    """
    print("  Building catcher game-calling features (as-of, leakage-free)...")

    if catchers_df is None or len(catchers_df) == 0:
        print("  ⚠ No catcher assignment data — skipping")
        for col in CATCHER_CALLING_FEATURES:
            df[col] = 0.0
        return df

    if (asof_df is None or len(asof_df) == 0) and (prior_df is None or len(prior_df) == 0):
        print("  ⚠ No catcher_features_*.csv — run collect/catcher.py first")
        for col in CATCHER_CALLING_FEATURES:
            df[col] = 0.0
        return df

    # ── Per-game catcher assignment frame ────────────────────────────────────
    cats = catchers_df.copy()
    cats["catcher_id"] = pd.to_numeric(cats["catcher_id"], errors="coerce").astype("Int64")
    # Add (game_date, season) from the main pitcher-game frame.
    game_meta = df[["game_pk", "game_date", "season"]].drop_duplicates("game_pk")
    cats = cats.merge(game_meta, on="game_pk", how="left")
    cats = cats.dropna(subset=["catcher_id", "game_date", "season"])
    cats["game_date"] = pd.to_datetime(cats["game_date"], errors="coerce")
    cats["season"]    = cats["season"].astype(int)

    # ── As-of merge: latest as-of snapshot strictly before game_date ─────────
    if asof_df is not None and len(asof_df) > 0:
        a = asof_df.copy()
        a["catcher_id"] = pd.to_numeric(a["catcher_id"], errors="coerce").astype("Int64")
        a["season"]     = pd.to_numeric(a["season"], errors="coerce").astype(int)
        a["game_date"]  = pd.to_datetime(a["game_date"], errors="coerce")
        a = a.dropna(subset=["catcher_id", "game_date", "season"])

        # merge_asof requires sorted inputs. Sort BOTH by the merge key
        # (game_date) globally; then 'by' grouping is handled within.
        cats_sorted = cats.sort_values("game_date").reset_index(drop=False)
        a_sorted    = a.sort_values("game_date").reset_index(drop=True)

        merged_asof = pd.merge_asof(
            cats_sorted, a_sorted,
            on="game_date",
            by=["catcher_id", "season"],
            direction="backward",  # latest snapshot at or before game_date
            allow_exact_matches=True,  # Already strict-< by construction in 01c
        )
        # Restore original cats order
        merged_asof = merged_asof.sort_values("index").drop(columns=["index"])\
                                  .reset_index(drop=True)
        # Rename current-season columns
        for col in CATCHER_CALLING_FEATURES:
            if col in merged_asof.columns:
                merged_asof = merged_asof.rename(columns={col: f"{col}__cur"})
        if "n_pitches_caught_to_date" in merged_asof.columns:
            merged_asof = merged_asof.rename(
                columns={"n_pitches_caught_to_date": "n_pitches_caught__cur"}
            )
        cats = merged_asof
    else:
        # No as-of data — populate with NaN so the prior takes over
        for col in CATCHER_CALLING_FEATURES:
            cats[f"{col}__cur"] = np.nan
        cats["n_pitches_caught__cur"] = np.nan

    # ── Prior-season merge: catcher's full prior season ──────────────────────
    if prior_df is not None and len(prior_df) > 0:
        p = prior_df.copy()
        p["catcher_id"] = pd.to_numeric(p["catcher_id"], errors="coerce").astype("Int64")
        p["season"]     = pd.to_numeric(p["season"], errors="coerce").astype(int)
        # Rename so we can join by season-1
        p = p.rename(columns={"season": "_prior_season"})
        for col in CATCHER_CALLING_FEATURES:
            if col in p.columns:
                p = p.rename(columns={col: f"{col}__prior"})
        if "n_pitches_caught" in p.columns:
            p = p.rename(columns={"n_pitches_caught": "n_pitches_caught__prior"})

        cats["_prior_season"] = cats["season"] - 1
        cats = cats.merge(p, on=["catcher_id", "_prior_season"], how="left")
        cats = cats.drop(columns=["_prior_season"])
    else:
        for col in CATCHER_CALLING_FEATURES:
            cats[f"{col}__prior"] = np.nan
        cats["n_pitches_caught__prior"] = np.nan

    # ── Size-weighted blend ──────────────────────────────────────────────────
    n_cur   = cats["n_pitches_caught__cur"].fillna(0.0)
    n_prior = cats["n_pitches_caught__prior"].fillna(0.0)
    denom   = n_cur + 0.5 * n_prior
    w_cur   = np.where(denom > 0, n_cur / np.where(denom > 0, denom, 1.0), 0.0)

    for col in CATCHER_CALLING_FEATURES:
        c_cur   = cats.get(f"{col}__cur",   pd.Series(np.nan, index=cats.index))
        c_prior = cats.get(f"{col}__prior", pd.Series(np.nan, index=cats.index))
        blended = pd.Series(np.nan, index=cats.index)
        both       = c_cur.notna() & c_prior.notna()
        cur_only   = c_cur.notna() & ~c_prior.notna()
        prior_only = ~c_cur.notna() & c_prior.notna()
        blended.loc[both]       = w_cur[both] * c_cur[both] + \
                                  (1 - w_cur[both]) * c_prior[both]
        blended.loc[cur_only]   = c_cur[cur_only]
        blended.loc[prior_only] = c_prior[prior_only]
        cats[col] = blended.fillna(0.0)

    # Drop the helper columns
    drop_cols = [f"{c}__cur" for c in CATCHER_CALLING_FEATURES] + \
                [f"{c}__prior" for c in CATCHER_CALLING_FEATURES] + \
                ["n_pitches_caught__cur", "n_pitches_caught__prior"]
    cats = cats.drop(columns=[c for c in drop_cols if c in cats.columns])

    # ── Split into home/away (catcher on SAME team as pitcher) ───────────────
    home_cats = cats[cats["side"] == "home"][
        ["game_pk"] + CATCHER_CALLING_FEATURES
    ].copy()
    away_cats = cats[cats["side"] == "away"][
        ["game_pk"] + CATCHER_CALLING_FEATURES
    ].copy()

    normalize_game_pk(df, home_cats, away_cats)

    # Drop pre-existing columns (idempotency on rerun)
    df = df.drop(columns=[c for c in CATCHER_CALLING_FEATURES if c in df.columns],
                 errors="ignore")

    df_home = df[df["is_home"] == 1].merge(home_cats, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(away_cats, on="game_pk", how="left")

    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    for col in CATCHER_CALLING_FEATURES:
        df[col] = df[col].fillna(0.0)

    n_nonzero = (df[CATCHER_CALLING_FEATURES[0]] != 0).sum()
    print(f"  ✓ Added {len(CATCHER_CALLING_FEATURES)} catcher game-calling features "
          f"({n_nonzero:,}/{len(df):,} games with non-zero values)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED LINEUP PLATE DISCIPLINE + CONCENTRATION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_lineup_plate_discipline(df, fg_batting, lineup_df):
    """
    Add batter-level plate discipline features from FanGraphs to lineup aggregation.
    Goes far beyond basic K rate to include:
      - SwStr% (swinging strike rate)
      - O-Swing% (chase rate)
      - Z-Contact% (zone contact — how well they put bat on ball in zone)
      - Contact% (overall contact rate)
      - TTO% (three true outcomes rate — K + BB + HR / PA)
      - CSW% (called strikes + whiffs)
    Also adds lineup concentration features (variance of K rates).
    """
    print("  Building lineup plate discipline features...")

    if fg_batting is None or len(fg_batting) == 0 or lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ Missing FG batting or lineup data — skipping")
        return df

    fg = fg_batting.copy()

    # Get MLBAM ID for merging — build name crosswalk if no MLBAM ID
    if "xMLBAMID" in fg.columns:
        fg = fg.rename(columns={"xMLBAMID": "player_id"})
    elif "MLBAMID" in fg.columns:
        fg = fg.rename(columns={"MLBAMID": "player_id"})
    else:
        # Build crosswalk from lineup data: player_name → player_id (MLBAM)
        if "Name" in fg.columns and "player_name" in lineup_df.columns:
            name_to_id = (
                lineup_df[["player_id", "player_name"]]
                .dropna(subset=["player_id", "player_name"])
                .drop_duplicates("player_name")
            )
            fg = fg.merge(name_to_id, left_on="Name", right_on="player_name", how="inner")
            fg = fg.drop(columns=["player_name"], errors="ignore")
        else:
            print("  ⚠ Cannot build batter crosswalk — skipping plate discipline")
            return df

    fg["player_id"] = pd.to_numeric(fg["player_id"], errors="coerce")
    fg = fg.dropna(subset=["player_id"])
    fg["player_id"] = fg["player_id"].astype(int)

    # Select batter plate discipline columns
    batter_cols = {}
    for col, new_name in [
        ("SwStr%", "batter_swstr_pct"),
        ("O-Swing%", "batter_o_swing_pct"),
        ("Z-Contact%", "batter_z_contact_pct"),
        ("Contact%", "batter_contact_pct"),
        ("TTO%", "batter_tto_pct"),
        ("CSW%", "batter_csw_pct"),
        ("K%", "batter_fg_k_pct"),
        ("Barrel%", "batter_barrel_pct"),
        ("HardHit%", "batter_hard_hit_pct"),
    ]:
        if col in fg.columns:
            batter_cols[col] = new_name

    if not batter_cols:
        print("  ⚠ No plate discipline columns found — skipping")
        return df

    fg_slim = fg[["player_id", "Season"] + list(batter_cols.keys())].copy()
    fg_slim = fg_slim.rename(columns=batter_cols)
    fg_slim = fg_slim.rename(columns={"Season": "season"})

    # Convert percentage strings to floats
    for col in fg_slim.columns:
        if fg_slim[col].dtype == object:
            try:
                fg_slim[col] = fg_slim[col].str.rstrip("%").astype(float) / 100.0
            except (ValueError, AttributeError):
                pass

    # Merge to lineup data
    lu = lineup_df.copy()
    game_seasons = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
    lu = lu.merge(game_seasons, on="game_pk", how="left")
    lu = lu.dropna(subset=["season"])
    lu["season"] = lu["season"].astype(int)
    lu["player_id"] = pd.to_numeric(lu["player_id"], errors="coerce").astype("Int64")

    lu = lu.merge(fg_slim, on=["player_id", "season"], how="left")

    # Lineup weights (same as build_lineup_features)
    lu["lineup_weight"] = 10 - lu["lineup_position"].fillna(5)

    # Aggregate per game + side
    plate_disc_cols = [c for c in lu.columns if c.startswith("batter_") and c in fg_slim.columns]

    agg_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        rec = {"game_pk": gpk, "side": side}

        for col in plate_disc_cols:
            vals = group[col].dropna()
            weights = group.loc[vals.index, "lineup_weight"]
            if len(vals) > 0 and weights.sum() > 0:
                rec[f"lu_{col.replace('batter_', '')}"] = (vals * weights).sum() / weights.sum()
            else:
                rec[f"lu_{col.replace('batter_', '')}"] = np.nan

        # Lineup concentration: std of K rates (higher = more uneven lineup)
        k_rates = group["batter_fg_k_pct"].dropna() if "batter_fg_k_pct" in group.columns else pd.Series()
        rec["lu_k_rate_std"] = k_rates.std() if len(k_rates) >= 3 else np.nan

        # TTO concentration
        tto_rates = group["batter_tto_pct"].dropna() if "batter_tto_pct" in group.columns else pd.Series()
        rec["lu_tto_pct_std"] = tto_rates.std() if len(tto_rates) >= 3 else np.nan

        # Max K rate in lineup (the weakest link for the pitcher to exploit)
        rec["lu_max_k_rate"] = k_rates.max() if len(k_rates) > 0 else np.nan
        rec["lu_min_k_rate"] = k_rates.min() if len(k_rates) > 0 else np.nan

        agg_records.append(rec)

    lu_agg = pd.DataFrame(agg_records)

    # Merge to pitcher-game (flip home/away like build_lineup_features)
    away_lu = lu_agg[lu_agg["side"] == "away"].drop(columns=["side"])
    away_lu = away_lu.rename(columns={c: f"opp_{c}" for c in away_lu.columns if c != "game_pk"})

    home_lu = lu_agg[lu_agg["side"] == "home"].drop(columns=["side"])
    home_lu = home_lu.rename(columns={c: f"opp_{c}" for c in home_lu.columns if c != "game_pk"})

    # Normalize game_pk before merge to prevent dtype drift
    normalize_game_pk(df, away_lu, home_lu)

    df_home = df[df["is_home"] == 1].merge(away_lu, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_lu, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # Interactions with pitcher ability
    if "fg_stuff_plus" in df.columns:
        # Stuff+ × lineup contact rate (low contact + high stuff = K paradise)
        if "opp_lu_contact_pct" in df.columns:
            df["ix_stuff_x_contact"] = (df["fg_stuff_plus"].fillna(100) / 100.0) * (
                1 - df["opp_lu_contact_pct"].fillna(0.78)
            )
        # Stuff+ × lineup chase rate (high stuff + high chase = whiff city)
        if "opp_lu_o_swing_pct" in df.columns:
            df["ix_stuff_x_chase"] = (df["fg_stuff_plus"].fillna(100) / 100.0) * df["opp_lu_o_swing_pct"].fillna(0.30)

    # Pitcher SwStr% × lineup SwStr% interaction
    if "fg_swstr_pct" in df.columns and "opp_lu_swstr_pct" in df.columns:
        df["ix_pitcher_lu_swstr"] = df["fg_swstr_pct"].fillna(0.11) * df["opp_lu_swstr_pct"].fillna(0.11)

    # TTO matchup: both pitcher and lineup TTO-heavy = more extreme outcomes
    if "fg_tto_pct" in df.columns and "opp_lu_tto_pct" in df.columns:
        df["ix_tto_matchup"] = df["fg_tto_pct"].fillna(0.33) * df["opp_lu_tto_pct"].fillna(0.33)

    new_cols = [c for c in df.columns if c.startswith("opp_lu_") and any(
        x in c for x in ["swstr", "o_swing", "z_contact", "contact_pct", "tto",
                          "csw", "fg_k", "barrel", "hard_hit", "k_rate_std",
                          "tto_pct_std", "max_k", "min_k"]
    )] + [c for c in df.columns if c.startswith("ix_stuff_x_") or c.startswith("ix_pitcher_lu_") or c == "ix_tto_matchup"]
    print(f"  ✓ Added {len(new_cols)} lineup plate discipline features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PITCHER VELOCITY DELTA & ROLLING PLATOON K RATES
# ══════════════════════════════════════════════════════════════════════════════

def build_velocity_and_platoon_features(df):
    """
    Add features the rolling stats miss:
      - Velocity delta: L3 avg_velocity minus season avg (fatigue/injury signal)
      - Rolling platoon K rates: K rate vs LHB and RHB separately
        (using pa_vs_left/right and whiffs_vs_left/right shifted data)
      - Pitcher TTO rate: (K + BB + HR) / PA rolling
    """
    print("  Building velocity delta & platoon K rate features...")

    df = df.sort_values(["pitcher", "game_date"]).copy()

    # ── Velocity delta (recent vs season baseline) ──
    if "avg_velocity_L3" in df.columns and "avg_velocity_szn" in df.columns:
        # Already have these from rolling/cumulative — compute delta
        df["velo_delta_L3_vs_szn"] = df["avg_velocity_L3"] - df["avg_velocity_szn"]
    elif "avg_velocity" in df.columns:
        # Build from scratch using shifted values
        shifted_velo = df.groupby("pitcher")["avg_velocity"].shift(1)
        df["velo_delta_L3_vs_szn"] = (
            shifted_velo.groupby(df["pitcher"]).transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            ) - shifted_velo.groupby(df["pitcher"]).transform(
                lambda x: x.expanding(min_periods=1).mean()
            )
        )

    # ── Rolling platoon-specific K rates ──
    # Use shifted pa_vs_left / whiffs_vs_left to build rolling K rate by handedness
    for side, pa_col, whiff_col in [
        ("left", "pa_vs_left", "whiffs_vs_left"),
        ("right", "pa_vs_right", "whiffs_vs_right"),
    ]:
        if pa_col in df.columns and whiff_col in df.columns:
            shifted_pa = df.groupby("pitcher")[pa_col].shift(1)
            shifted_w = df.groupby("pitcher")[whiff_col].shift(1)

            for window in [5, 10]:
                roll_pa = shifted_pa.groupby(df["pitcher"]).transform(
                    lambda x: x.rolling(window, min_periods=3).sum()
                )
                roll_w = shifted_w.groupby(df["pitcher"]).transform(
                    lambda x: x.rolling(window, min_periods=3).sum()
                )
                df[f"whiff_pct_vs_{side}_L{window}"] = (roll_w / roll_pa.replace(0, np.nan))

            # Season cumulative
            cum_pa = shifted_pa.groupby(df["pitcher"]).cumsum()
            cum_w = shifted_w.groupby(df["pitcher"]).cumsum()
            df[f"whiff_pct_vs_{side}_szn"] = (cum_w / cum_pa.replace(0, np.nan))

            # Platoon differential (how much better is pitcher vs one side)
            if f"whiff_pct_vs_left_L5" in df.columns and f"whiff_pct_vs_right_L5" in df.columns:
                df["platoon_whiff_diff_L5"] = (
                    df["whiff_pct_vs_right_L5"].fillna(0) - df["whiff_pct_vs_left_L5"].fillna(0)
                )

    # ── Pitcher rolling TTO rate ──
    # (K + BB + HR) / PA — captures how much the pitcher's outcomes are "three true"
    if all(c in df.columns for c in ["strikeouts", "walks", "home_runs_allowed", "plate_appearances"]):
        tto = df["strikeouts"] + df["walks"] + df["home_runs_allowed"]
        pa = df["plate_appearances"].replace(0, np.nan)
        df["pitcher_tto_game"] = tto / pa

        shifted_tto = df.groupby("pitcher")["pitcher_tto_game"].shift(1)
        for window in [5, 10]:
            df[f"pitcher_tto_L{window}"] = shifted_tto.groupby(df["pitcher"]).transform(
                lambda x: x.rolling(window, min_periods=3).mean()
            )
        df[f"pitcher_tto_szn"] = shifted_tto.groupby(df["pitcher"]).transform(
            lambda x: x.expanding(min_periods=3).mean()
        )
        # Drop the current-game version (would leak)
        df = df.drop(columns=["pitcher_tto_game"], errors="ignore")

    new_cols = [c for c in df.columns if any(x in c for x in [
        "velo_delta", "whiff_pct_vs_left", "whiff_pct_vs_right",
        "platoon_whiff_diff", "pitcher_tto_L", "pitcher_tto_szn"
    ])]
    print(f"  ✓ Added {len(new_cols)} velocity delta + platoon + TTO features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED FEATURES: EWM, DELTAS, L1, TTO SPLITS, INTERACTIONS, FATIGUE
# ══════════════════════════════════════════════════════════════════════════════

def build_advanced_features(df, lineup_df=None):
    """
    Add high-impact features identified by professional K% modeling:

    1. EWM (Exponentially Weighted Moving Averages) — recent starts matter more
    2. Delta features — changes between windows (e.g. L3 vs L10, L3 vs season)
    3. L1 (last 1 start) features — ultra-short-term recency
    4. Pitcher × umpire interaction — wide zone + edge pitcher = K boost
    5. Pitcher × catcher framing interaction
    6. Median lineup K%, top3/bot3 K means
    7. Pitch-type matchup score per lineup
    8. Pitch count fatigue features
    """
    print("  Building advanced features (EWM, deltas, L1, TTO, interactions, fatigue)...")

    df = df.sort_values(["pitcher", "game_date"]).copy()
    # Normalize game_pk at the start to prevent dtype drift from prior pipeline steps
    normalize_game_pk(df)
    n_before = len(df.columns)

    # ── 1. EWM (Exponentially Weighted Moving Averages) ──────────────────
    # Alpha=0.3 means ~30% weight on latest start, decaying from there
    # More responsive than fixed-window but smoother than L1
    ewm_cols = ["k_pct", "whiff_pct", "csw_pct", "avg_velocity", "chase_rate",
                "barrel_pct", "strike_pct", "bb_pct", "hard_hit_pct"]
    ewm_cols = [c for c in ewm_cols if c in df.columns]

    for col in ewm_cols:
        shifted = df.groupby("pitcher")[col].shift(1)
        df[f"{col}_ewm"] = shifted.groupby(df["pitcher"]).transform(
            lambda x: x.ewm(alpha=0.3, min_periods=2).mean()
        )

    # ── 2. Delta features — changes between time windows ─────────────────
    # These capture whether a pitcher is trending up or down
    delta_stats = ["k_pct", "whiff_pct", "csw_pct", "avg_velocity",
                   "chase_rate", "barrel_pct", "hard_hit_pct", "bb_pct"]

    for stat in delta_stats:
        # Delta: L3 vs L10 (short vs medium term)
        l3 = f"{stat}_L3"
        l10 = f"{stat}_L10"
        szn = f"{stat}_szn"
        ewm = f"{stat}_ewm"
        if l3 in df.columns and l10 in df.columns:
            df[f"delta_{stat}_3v10"] = df[l3] - df[l10]
        # Delta: L3 vs season (short term vs baseline)
        if l3 in df.columns and szn in df.columns:
            df[f"delta_{stat}_3vSzn"] = df[l3] - df[szn]
        # Delta: EWM vs season
        if ewm in df.columns and szn in df.columns:
            df[f"delta_{stat}_ewmvSzn"] = df[ewm] - df[szn]

    # ── 3. L1 (last 1 start) features — ultra-short-term ────────────────
    l1_cols = ["k_pct", "whiff_pct", "csw_pct", "avg_velocity",
               "total_pitches", "plate_appearances", "strikeouts",
               "bb_pct", "barrel_pct", "chase_rate", "est_innings",
               "is_short_outing"]
    l1_cols = [c for c in l1_cols if c in df.columns]

    for col in l1_cols:
        df[f"{col}_L1"] = df.groupby("pitcher")[col].shift(1)

    # ── 4. Pitcher × umpire interaction ──────────────────────────────────
    # Wide-zone ump + edge pitcher = K boost
    if "ump_k_pct" in df.columns:
        # Pitcher's called strike rate × umpire's wide zone tendency
        if "csw_pct_L5" in df.columns:
            df["ix_pitcher_csw_x_ump_k"] = df["csw_pct_L5"] * df["ump_k_pct"]
        if "ump_called_strike_pct" in df.columns and "strike_pct_L5" in df.columns:
            df["ix_pitcher_edge_x_ump_zone"] = df["strike_pct_L5"] * df["ump_called_strike_pct"]
        # Pitcher K rate × umpire K boost
        if "k_pct_L5" in df.columns:
            df["ix_pitcher_k_x_ump_k"] = df["k_pct_L5"] * df["ump_k_pct"]
        # Umpire BB rate interaction (wide zone = fewer walks = longer outings = more K)
        if "ump_bb_pct" in df.columns and "bb_pct_L5" in df.columns:
            df["ix_pitcher_bb_x_ump_bb"] = df["bb_pct_L5"] * df["ump_bb_pct"]

    # ── 5. Pitcher × catcher framing interaction ─────────────────────────
    if "catcher_frm" in df.columns:
        # Catcher framing × pitcher edge rate (edge pitchers benefit more from good framers)
        if "csw_pct_L5" in df.columns:
            df["ix_catcher_frm_x_csw"] = df["catcher_frm"] * df["csw_pct_L5"]
        if "strike_pct_L5" in df.columns:
            df["ix_catcher_frm_x_strike"] = df["catcher_frm"] * df["strike_pct_L5"]
        if "k_pct_L5" in df.columns:
            df["ix_catcher_frm_x_k"] = df["catcher_frm"] * df["k_pct_L5"]

    # ── 6. Pitch count fatigue features ──────────────────────────────────
    if "total_pitches" in df.columns:
        shifted_pitches = df.groupby("pitcher")["total_pitches"].shift(1)
        # Pitch count last start
        df["pitchcount_L1"] = shifted_pitches
        # Rolling 2-start pitch count (back-to-back heavy starts = fatigue)
        df["pitchcount_L2_total"] = shifted_pitches.groupby(df["pitcher"]).transform(
            lambda x: x.rolling(2, min_periods=1).sum()
        )
        # Heavy previous start flag (100+ pitches)
        df["heavy_prev_start"] = (shifted_pitches > 100).astype(float)
        # Pitch efficiency trend: pitches per out
        if "outs_recorded" in df.columns:
            shifted_outs = df.groupby("pitcher")["outs_recorded"].shift(1)
            df["pitches_per_out_L1"] = shifted_pitches / shifted_outs.replace(0, np.nan)
            df["pitches_per_out_L3"] = (
                shifted_pitches.groupby(df["pitcher"]).transform(
                    lambda x: x.rolling(3, min_periods=1).sum()
                ) /
                shifted_outs.groupby(df["pitcher"]).transform(
                    lambda x: x.rolling(3, min_periods=1).sum()
                ).replace(0, np.nan)
            )

    # ── 7. Enhanced lineup distribution features ─────────────────────────
    # (Requires lineup data — builds on existing build_lineup_features output)
    if lineup_df is not None and len(lineup_df) > 0:
        lu = lineup_df.copy()
        game_seasons = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
        lu = lu.merge(game_seasons, on="game_pk", how="left")
        lu = lu.dropna(subset=["season"])
        lu["season"] = lu["season"].astype(int)
        lu["game_date"] = pd.to_datetime(lu["game_date"])
        lu = lu.sort_values(["player_id", "season", "game_date"]).copy()

        # Build batter cumulative K rate (shifted)
        for col in ["at_bats", "strikeouts"]:
            shifted = lu.groupby(["player_id", "season"])[col].shift(1)
            lu[f"{col}_cum"] = shifted.groupby([lu["player_id"], lu["season"]]).cumsum()
        ab_cum = lu["at_bats_cum"].replace(0, np.nan)
        lu["batter_k_rate"] = lu["strikeouts_cum"] / ab_cum

        # Aggregate: median, top3/bot3
        adv_records = []
        for (gpk, side), group in lu.groupby(["game_pk", "side"]):
            k_rates = group["batter_k_rate"].dropna()
            if len(k_rates) < 5:
                continue
            k_sorted = k_rates.sort_values(ascending=False)
            rec = {
                "game_pk": gpk,
                "side": side,
                "lu_k_rate_median": k_rates.median(),
                "lu_top3_k_mean": k_sorted.head(3).mean(),
                "lu_bot3_k_mean": k_sorted.tail(3).mean(),
                "lu_top3_bot3_gap": k_sorted.head(3).mean() - k_sorted.tail(3).mean(),
            }
            adv_records.append(rec)

        if adv_records:
            adv_df = pd.DataFrame(adv_records)
            adv_cols = [c for c in adv_df.columns if c not in ["game_pk", "side"]]

            away_adv = adv_df[adv_df["side"] == "away"].drop(columns=["side"])
            away_adv = away_adv.rename(columns={c: f"opp_{c}" for c in adv_cols})
            home_adv = adv_df[adv_df["side"] == "home"].drop(columns=["side"])
            home_adv = home_adv.rename(columns={c: f"opp_{c}" for c in adv_cols})

            # Normalize game_pk before merge to prevent dtype drift
            normalize_game_pk(df, away_adv, home_adv)

            df_home = df[df["is_home"] == 1].merge(away_adv, on="game_pk", how="left")
            df_away = df[df["is_home"] == 0].merge(home_adv, on="game_pk", how="left")
            df = pd.concat([df_home, df_away], ignore_index=True)
            df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # ── 8. Pitch-type matchup score ──────────────────────────────────────
    # Sum of (pitcher pitch usage × lineup whiff weakness against that category)
    # This is the single most important engineered feature per the guide
    pt_usage_cols = {
        "ff_pct_L5": "opp_lu_vs_fb_whiff_rate",
        "si_pct_L5": "opp_lu_vs_fb_whiff_rate",
        "sl_pct_L5": "opp_lu_vs_brk_whiff_rate",
        "cu_pct_L5": "opp_lu_vs_brk_whiff_rate",
        "ch_pct_L5": "opp_lu_vs_os_whiff_rate",
    }
    matchup_score = pd.Series(0.0, index=df.index)
    has_any = False
    for usage_col, vuln_col in pt_usage_cols.items():
        if usage_col in df.columns and vuln_col in df.columns:
            matchup_score += df[usage_col].fillna(0) * df[vuln_col].fillna(0)
            has_any = True
    if has_any:
        df["pitch_matchup_score"] = matchup_score
        # Also build K-specific matchup score
        pt_k_cols = {
            "ff_pct_L5": "opp_lu_vs_fb_k_rate",
            "si_pct_L5": "opp_lu_vs_fb_k_rate",
            "sl_pct_L5": "opp_lu_vs_brk_k_rate",
            "cu_pct_L5": "opp_lu_vs_brk_k_rate",
            "ch_pct_L5": "opp_lu_vs_os_k_rate",
        }
        k_score = pd.Series(0.0, index=df.index)
        for usage_col, vuln_col in pt_k_cols.items():
            if usage_col in df.columns and vuln_col in df.columns:
                k_score += df[usage_col].fillna(0) * df[vuln_col].fillna(0)
        df["pitch_k_matchup_score"] = k_score

    n_after = len(df.columns)
    print(f"  ✓ Added {n_after - n_before} advanced features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BULLPEN AVAILABILITY / TEAM CONTEXT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_bullpen_features(df):
    """
    Build bullpen workload features that affect manager leash.
    If the bullpen is gassed, starters stay in longer → more BF → more K/H/BB
    opportunities. This is one of the most important missing feature categories.

    Features:
      - team_bullpen_ip_last1: Bullpen innings pitched yesterday
      - team_bullpen_ip_last3: Bullpen innings pitched last 3 days
      - team_bullpen_pitchcount_last1: Bullpen pitches thrown yesterday
      - team_bullpen_pitchcount_last3: Bullpen pitches thrown last 3 days
      - closer_used_yesterday: Binary — was a high-inning reliever used yesterday?
      - bullpen_heavy_use: Binary — bullpen threw 5+ innings in last game
    """
    print("  Building bullpen workload features...")

    # We need ALL pitcher appearances (not just starters) to compute bullpen usage.
    # Load the raw pitcher data and identify relievers.
    raw_path = Path("data/statcast_pitcher_games_all.csv")
    if not raw_path.exists():
        print("  ⚠ No raw pitcher data — skipping bullpen features")
        return df

    all_pitchers = pd.read_csv(raw_path, parse_dates=["game_date"])
    all_pitchers = all_pitchers[["pitcher", "game_pk", "game_date", "total_pitches",
                                  "home_team", "away_team", "plate_appearances", "season"]].copy()

    # Relievers = pitchers with < 45 pitches in that appearance
    relievers = all_pitchers[all_pitchers["total_pitches"] < 45].copy()

    # Assign team to each reliever appearance
    # We use game metadata to determine which team each pitcher was on
    if "team" not in relievers.columns:
        # Use game metadata to figure out team
        meta_path = Path("data/game_metadata_all.csv")
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            if "home_starter_id" in meta.columns:
                game_teams = meta[["game_pk", "home_team_id", "away_team_id",
                                   "home_team_name", "away_team_name"]].drop_duplicates("game_pk")
                normalize_game_pk(relievers, game_teams)
                relievers = relievers.merge(game_teams, on="game_pk", how="left")

    # Estimate innings pitched from plate appearances (rough: IP ≈ PA / 4.3)
    relievers["est_ip"] = relievers["plate_appearances"] / 4.3

    # Aggregate bullpen usage per team per game date
    # For each game, compute team bullpen totals (relief pitchers only)
    # We need to know which team each reliever was on.
    # Heuristic: if pitcher's game has home/away team, assign based on whether
    # they match the home or away starter (they won't, since they're relievers)
    # Simpler: assign team based on home_team/away_team in the reliever data
    # We can figure this out from the starters data we already have.

    # Build a mapping: (game_pk, team) → bullpen totals
    # Assign team to relievers by checking which team's starter they are NOT
    # Actually easier: we have home_team/away_team in the raw data already
    # Relievers on home team → home_team, relievers on away team → away_team
    # But we don't know which side a reliever was on directly.

    # Simplest approach: aggregate ALL reliever activity per game per team pair,
    # then split by home/away when merging to our starter data.

    # Group by game date and team (use home_team as a proxy since we need per-team)
    # Build per-game-date bullpen totals for each team
    bp_home = relievers.groupby(["game_date", "home_team"]).agg(
        bp_pitches=("total_pitches", "sum"),
        bp_pa=("plate_appearances", "sum"),
        bp_appearances=("pitcher", "count"),
    ).reset_index().rename(columns={"home_team": "team_proxy"})

    bp_away = relievers.groupby(["game_date", "away_team"]).agg(
        bp_pitches=("total_pitches", "sum"),
        bp_pa=("plate_appearances", "sum"),
        bp_appearances=("pitcher", "count"),
    ).reset_index().rename(columns={"away_team": "team_proxy"})

    # This double-counts — take max or deduplicate
    # Better approach: just aggregate by game_date + home_team for home bullpen usage
    # and game_date + away_team for away bullpen usage separately

    # For each starter in df, their team's bullpen usage = all relievers on the same team
    # We need the starter's team. Use home/away assignment.
    if "team" not in df.columns:
        # Derive team from is_home + home_team/away_team
        if "home_team" in df.columns and "away_team" in df.columns:
            df["team"] = np.where(df["is_home"] == 1, df["home_team"], df["away_team"])
        else:
            print("  ⚠ Cannot determine pitcher team — skipping bullpen features")
            return df

    # Build daily team bullpen aggregates from ALL reliever data per team
    # We'll merge on (game_date - offset, team)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # For home starters, their team's bullpen is home relievers
    # For away starters, their team's bullpen is away relievers
    # BUT: we only see relievers' game appearance, not directly which team they're on.
    # Use starters data to identify teams: for each game_pk, we know the starting pitcher
    # and their team. All OTHER pitchers in that game on the same team are relievers.

    # Simpler approach: use the raw total bullpen stats per team per date.
    # Since we have home/away team for each reliever appearance,
    # assume relievers with home_team=X are EITHER home or away relievers.
    # Actually, each row has both home_team and away_team — the pitcher could be on either.

    # Easiest reliable approach: compute per-game total pitches minus starter pitches
    # = bullpen pitches for that team in that game.
    starters_for_bp = df[["pitcher", "game_pk", "game_date", "total_pitches", "team", "is_home"]].copy()
    starters_for_bp = starters_for_bp.rename(columns={"total_pitches": "starter_pitches"})

    # Total pitches per game per team (all pitchers)
    # Home team: sum all pitchers where home_team matches
    all_home = all_pitchers.groupby(["game_pk", "home_team"]).agg(
        total_team_pitches=("total_pitches", "sum"),
        total_team_pa=("plate_appearances", "sum"),
    ).reset_index().rename(columns={"home_team": "team"})
    all_home["side"] = "home"

    all_away = all_pitchers.groupby(["game_pk", "away_team"]).agg(
        total_team_pitches=("total_pitches", "sum"),
        total_team_pa=("plate_appearances", "sum"),
    ).reset_index().rename(columns={"away_team": "team"})
    all_away["side"] = "away"

    # But this gives us BOTH teams per game (since each pitcher row has home_team+away_team)
    # We need to filter: for "home" team totals, only count pitchers who played FOR home
    # Since we don't have a direct "pitching_team" column, use a proxy:
    # total_team_pitches for home = all pitches in that game minus the away team's pitches
    # This is circular. Let's use the metadata approach instead.

    # Actually the simplest robust approach: for each game_pk+team in our starters data,
    # bullpen pitches = game total pitches for that side minus the starter's pitches.
    # We know the starter's pitches. We need total pitches per side per game.

    # From game_meta, we might have total team pitches. Let's just use reliever count.
    # Reliever pitches for a team = all_pitchers with total_pitches < 45 in same game + team

    # Let's simplify: for each team's game, compute bullpen load as
    # (total pitches by all pitchers in game on that team's side) - (starter pitches)
    # We identify starters by having >= 45 pitches.

    # Group all pitchers by game_pk
    game_totals = all_pitchers.groupby("game_pk").agg(
        game_total_pitches=("total_pitches", "sum"),
        game_total_pa=("plate_appearances", "sum"),
    ).reset_index()

    # For a given starter, bullpen pitches ≈ game_total - both_starters
    # Merge game totals to starters
    normalize_game_pk(starters_for_bp, game_totals)
    starters_for_bp = starters_for_bp.merge(game_totals, on="game_pk", how="left")

    # Get opposing starter pitches
    opp_starters = df[["game_pk", "total_pitches", "is_home"]].copy()
    opp_starters = opp_starters.rename(columns={"total_pitches": "opp_starter_pitches",
                                                  "is_home": "opp_is_home"})
    # The opposing starter is the one with is_home flipped
    starters_for_bp["opp_is_home"] = 1 - starters_for_bp["is_home"]
    starters_for_bp = starters_for_bp.merge(
        opp_starters, on=["game_pk", "opp_is_home"], how="left"
    )
    starters_for_bp["opp_starter_pitches"] = starters_for_bp["opp_starter_pitches"].fillna(0)

    # Team bullpen pitches ≈ (game total - both starters) / 2
    # This is approximate but reasonable for aggregate features
    both_starters = starters_for_bp["starter_pitches"] + starters_for_bp["opp_starter_pitches"]
    starters_for_bp["team_bp_pitches"] = (
        (starters_for_bp["game_total_pitches"] - both_starters) / 2
    ).clip(lower=0)
    starters_for_bp["team_bp_pa"] = (
        (starters_for_bp["game_total_pa"] -
         starters_for_bp["starter_pitches"] / 4 -
         starters_for_bp["opp_starter_pitches"] / 4) / 2
    ).clip(lower=0)
    starters_for_bp["team_bp_ip_est"] = starters_for_bp["team_bp_pa"] / 4.3

    # Now build rolling features: bullpen workload over last 1 and 3 days
    starters_for_bp = starters_for_bp.sort_values(["team", "game_date"])

    # For each team-date, get yesterday's and last-3-days' bullpen workload
    # Group by team and compute shifted rolling sums
    team_daily = starters_for_bp.groupby(["team", "game_date"]).agg(
        daily_bp_pitches=("team_bp_pitches", "first"),
        daily_bp_ip=("team_bp_ip_est", "first"),
    ).reset_index()
    team_daily = team_daily.sort_values(["team", "game_date"])

    # Shifted to prevent leakage (yesterday = shift 1)
    team_daily["bp_pitches_shift1"] = team_daily.groupby("team")["daily_bp_pitches"].shift(1)
    team_daily["bp_ip_shift1"] = team_daily.groupby("team")["daily_bp_ip"].shift(1)

    # Last 3 days (shifted)
    team_daily["bp_pitches_L3"] = team_daily.groupby("team")["daily_bp_pitches"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    team_daily["bp_ip_L3"] = team_daily.groupby("team")["daily_bp_ip"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )

    # Heavy use flag: bullpen threw 5+ estimated innings yesterday
    team_daily["bp_heavy_yesterday"] = (team_daily["bp_ip_shift1"] >= 5).astype(float)

    # Closer/high-leverage usage proxy: any reliever threw 25+ pitches yesterday
    # (indicates closer or high-leverage usage)
    team_daily["bp_closer_used_yesterday"] = (team_daily["bp_pitches_shift1"] >= 25).astype(float)

    bp_features = team_daily[["team", "game_date", "bp_pitches_shift1", "bp_ip_shift1",
                               "bp_pitches_L3", "bp_ip_L3", "bp_heavy_yesterday",
                               "bp_closer_used_yesterday"]].copy()
    bp_features = bp_features.rename(columns={
        "bp_pitches_shift1": "team_bullpen_pitchcount_last1",
        "bp_ip_shift1": "team_bullpen_ip_last1",
        "bp_pitches_L3": "team_bullpen_pitchcount_last3",
        "bp_ip_L3": "team_bullpen_ip_last3",
        "bp_heavy_yesterday": "bullpen_heavy_use",
        "bp_closer_used_yesterday": "closer_used_yesterday",
    })

    # Merge back to main df
    pre_cols = set(df.columns)
    df = df.merge(bp_features, on=["team", "game_date"], how="left")

    # Fill NaN with league averages
    bp_defaults = {
        "team_bullpen_pitchcount_last1": 55.0,
        "team_bullpen_ip_last1": 3.5,
        "team_bullpen_pitchcount_last3": 165.0,
        "team_bullpen_ip_last3": 10.5,
        "bullpen_heavy_use": 0.0,
        "closer_used_yesterday": 0.0,
    }
    for col, dflt in bp_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(dflt)

    new_cols = [c for c in df.columns if c not in pre_cols]
    print(f"  ✓ Added {len(new_cols)} bullpen workload features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BATTED-BALL FEATURES (from raw Statcast PA events)
# ══════════════════════════════════════════════════════════════════════════════

def build_batted_ball_features(df):
    """
    Per-pitcher-game batted-ball-type counts and quality-of-contact averages,
    sourced from data/statcast_pa_events_all.csv.

    This function ONLY backfills columns that aren't already present on `df`.
    If the upstream collector (collect/statcast.py) already populates these
    fields in statcast_pitcher_games_all.csv, we silently no-op for those
    columns. This is the most important addition for the hits/walks model:
    BABIP, batted-ball mix, and contact quality are the dominant signals
    for hit suppression beyond K rate.

    Computed (per pitcher × game):
      Counts:
        - ground_balls, fly_balls, line_drives, popups
        - singles, doubles, triples (raw hit-type counts)
      Quality averages (mean across batted balls in the start):
        - avg_exit_velocity   — mean launch_speed on batted balls
        - avg_launch_angle    — mean launch_angle on batted balls
        - sweet_spot_pct      — fraction of batted balls with launch_angle 8°-32°
        - solid_contact_pct   — fraction barreled or solidly hit
        - avg_xba_contact     — mean Statcast xBA (estimated_ba_using_speedangle)
        - avg_xwoba_contact   — mean Statcast xwOBA (estimated_woba_using_speedangle)

    Required PA events columns (any missing → that feature is silently skipped):
      - pitcher, game_pk, game_date
      - bb_type (str: 'ground_ball', 'fly_ball', 'line_drive', 'popup')
      - launch_speed, launch_angle (numeric, NaN for non-BIP events)
      - events (str: 'single', 'double', 'triple', 'home_run', etc.)
      - estimated_ba_using_speedangle, estimated_woba_using_speedangle (optional)
    """
    print("  Building batted-ball features from PA events...")

    pa_path = DATA_DIR / "statcast_pa_events_all.csv"
    if not pa_path.exists():
        print("  ⚠ No statcast_pa_events_all.csv — skipping batted-ball features")
        return df

    # Probe columns first so we read only what's there. Cheap on a header read.
    probe = pd.read_csv(pa_path, nrows=0)
    available = set(probe.columns)
    must_have = {"pitcher", "game_pk"}
    if not must_have.issubset(available):
        print(f"  ⚠ PA events missing required cols {must_have - available} — skipping")
        return df

    # Decide which optional cols to pull
    wanted = list(must_have) + [
        c for c in [
            "game_date", "bb_type", "launch_speed", "launch_angle",
            "events", "estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle",
        ] if c in available
    ]
    pa = pd.read_csv(pa_path, usecols=wanted, low_memory=False)
    if "game_date" in pa.columns:
        pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")

    normalize_game_pk(pa)
    normalize_game_pk(df)

    # ── Aggregate per pitcher × game ──
    agg_dict = {}
    if "bb_type" in pa.columns:
        # bb_type is only non-null on batted balls; one row per BIP event.
        bip_mask = pa["bb_type"].notna()
        # Per-game counts by bb_type
        bb_counts = (pa.loc[bip_mask]
                     .groupby(["pitcher", "game_pk"])["bb_type"]
                     .value_counts()
                     .unstack(fill_value=0)
                     .reset_index())

        # Map Statcast bb_type values to our column names. Statcast uses
        # 'ground_ball' / 'fly_ball' / 'line_drive' / 'popup'.
        bb_type_map = {
            "ground_ball": "ground_balls",
            "fly_ball":    "fly_balls",
            "line_drive":  "line_drives",
            "popup":       "popups",
        }
        for src, dst in bb_type_map.items():
            if src in bb_counts.columns:
                bb_counts = bb_counts.rename(columns={src: dst})
        # Also expose total BIP if upstream didn't already populate it.
        cnt_cols = [v for v in bb_type_map.values() if v in bb_counts.columns]
        if cnt_cols:
            bb_counts["batted_balls_recomputed"] = bb_counts[cnt_cols].sum(axis=1)

        agg_dict["bb_counts"] = bb_counts

    # ── Hit-type counts (singles/doubles/triples/HR) from `events` ──
    if "events" in pa.columns:
        hit_events = ["single", "double", "triple", "home_run"]
        hit_pa = pa[pa["events"].isin(hit_events)]
        hit_counts = (hit_pa
                      .groupby(["pitcher", "game_pk"])["events"]
                      .value_counts()
                      .unstack(fill_value=0)
                      .reset_index())
        # Rename to plural to match the schema we use elsewhere
        rename_map = {
            "single":   "singles",
            "double":   "doubles",
            "triple":   "triples",
            "home_run": "home_runs_allowed_recomputed",
        }
        for src, dst in rename_map.items():
            if src in hit_counts.columns:
                hit_counts = hit_counts.rename(columns={src: dst})
        agg_dict["hit_counts"] = hit_counts

    # ── Quality of contact averages (over batted balls only) ──
    bip_mask = pa["bb_type"].notna() if "bb_type" in pa.columns else \
               pa["launch_speed"].notna() if "launch_speed" in pa.columns else None
    if bip_mask is not None and bip_mask.any():
        bip = pa.loc[bip_mask].copy()
        quality_aggs = {}
        if "launch_speed" in bip.columns:
            quality_aggs["avg_exit_velocity_recomputed"] = ("launch_speed", "mean")
            # sweet spot: launch_angle in [8, 32]
            if "launch_angle" in bip.columns:
                bip["_in_sweet_spot"] = bip["launch_angle"].between(8, 32).astype(float)
                quality_aggs["sweet_spot_pct_recomputed"] = ("_in_sweet_spot", "mean")
                # solid contact (Statcast definition is more nuanced, but a
                # reasonable proxy: launch_speed ≥ 95 and launch_angle in
                # [-5, 35] — this captures barrels and "solid" hits).
                bip["_solid"] = ((bip["launch_speed"] >= 95) &
                                 (bip["launch_angle"].between(-5, 35))).astype(float)
                quality_aggs["solid_contact_pct_recomputed"] = ("_solid", "mean")
        if "launch_angle" in bip.columns:
            quality_aggs["avg_launch_angle_recomputed"] = ("launch_angle", "mean")
        if "estimated_ba_using_speedangle" in bip.columns:
            quality_aggs["avg_xba_contact_recomputed"] = ("estimated_ba_using_speedangle", "mean")
        if "estimated_woba_using_speedangle" in bip.columns:
            quality_aggs["avg_xwoba_contact_recomputed"] = ("estimated_woba_using_speedangle", "mean")

        if quality_aggs:
            quality = (bip.groupby(["pitcher", "game_pk"])
                       .agg(**quality_aggs)
                       .reset_index())
            agg_dict["quality"] = quality

    if not agg_dict:
        print("  ⚠ No batted-ball columns producible from PA events — skipping")
        return df

    # ── Merge all aggregates onto df, then backfill base columns ──
    #
    # IMPORTANT: if the user re-ran collect/statcast.py with the updated
    # aggregate_pitcher_game() function, `df` already has ground_balls,
    # fly_balls, line_drives, singles, doubles, triples, etc. directly
    # from the per-game CSV. A naive merge with the same column names
    # on the right-hand frame creates _x/_y suffixed duplicates, which
    # the model then picks up as features — and the _y version is a
    # CURRENT-game leak.
    #
    # Fix: drop overlapping columns from the merge frame before joining.
    # We prefer the upstream collector's values (already in df) over the
    # PA-events recompute; if the upstream is missing them, the
    # _recomputed columns below still backfill.
    pre_cols = set(df.columns)
    for name, frame in agg_dict.items():
        normalize_game_pk(frame)
        # Drop overlapping columns (other than join keys) so pandas
        # doesn't create _x/_y suffixed duplicates.
        join_keys = {"pitcher", "game_pk"}
        overlap = [c for c in frame.columns
                   if c in df.columns and c not in join_keys]
        if overlap:
            frame = frame.drop(columns=overlap)
        if len(frame.columns) <= len(join_keys):
            # Nothing left to merge after dropping overlaps
            continue
        df = df.merge(frame, on=["pitcher", "game_pk"], how="left")

    # Backfill base columns from "_recomputed" variants only if base is missing
    # or all-null. This preserves any upstream collector's values while
    # filling gaps for older data or runs without that collector.
    backfill_map = {
        "ground_balls":      None,  # bb_counts already wrote these by their target name
        "fly_balls":         None,
        "line_drives":       None,
        "popups":            None,
        "singles":           None,
        "doubles":           None,
        "triples":           None,
        "batted_balls":      "batted_balls_recomputed",
        "home_runs_allowed": "home_runs_allowed_recomputed",
        "avg_exit_velocity": "avg_exit_velocity_recomputed",
        "avg_launch_angle":  "avg_launch_angle_recomputed",
        "sweet_spot_pct":    "sweet_spot_pct_recomputed",
        "solid_contact_pct": "solid_contact_pct_recomputed",
        "avg_xba_contact":   "avg_xba_contact_recomputed",
        "avg_xwoba_contact": "avg_xwoba_contact_recomputed",
    }
    backfilled = 0
    for base_col, recomp_col in backfill_map.items():
        if recomp_col is None:
            # Already merged in with the target name (from bb_counts/hit_counts).
            # Nothing further to do.
            continue
        if recomp_col not in df.columns:
            continue
        if base_col not in df.columns or df[base_col].isna().all():
            df[base_col] = df[recomp_col]
            backfilled += 1
        else:
            # Fill remaining NaNs in base from recomp
            df[base_col] = df[base_col].fillna(df[recomp_col])
        # Drop the _recomputed scaffolding column to keep the schema clean
        df = df.drop(columns=[recomp_col])

    # Fill count columns with 0 where we have no PA-events data for that game
    # (e.g. games that weren't collected at the PA level). Tree models prefer
    # 0 over NaN for counts.
    for col in ["ground_balls", "fly_balls", "line_drives", "popups",
                "singles", "doubles", "triples"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    new_cols = [c for c in df.columns if c not in pre_cols]
    print(f"  ✓ Added/backfilled {len(new_cols)} batted-ball columns "
          f"({backfilled} backfilled from PA events)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VELOCITY TREND / INJURY SIGNAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_velocity_trend_features(df, pitcher_pt_df):
    """
    Build intra-start velocity deterioration features.
    A pitcher losing 1.5 mph by inning 5 is an injury/fatigue signal that
    directly impacts strikeout probability in later innings.

    Features:
      - velo_drop_last_start: Velocity delta from first to last pitch type group
      - velo_drop_last3_avg: Average velocity drop across last 3 starts
      - ff_velo_L1: Most recent start fastball velo (ultra-recency)
      - ff_velo_delta_1v5: Fastball velo last start vs L5 average
    """
    print("  Building velocity trend / injury signal features...")

    if pitcher_pt_df is None or len(pitcher_pt_df) == 0:
        print("  ⚠ No pitch type data — skipping velocity trend features")
        return df

    pt = pitcher_pt_df.copy()
    pt["game_date"] = pd.to_datetime(pt["game_date"])

    # Get fastball velocity per game (FF is the primary indicator)
    ff_data = pt[pt["pitch_type"].isin(["FF", "SI"])].copy()
    if len(ff_data) == 0:
        print("  ⚠ No fastball data — skipping velocity trend features")
        return df

    # Weighted average velocity per game (weight by pitch count)
    ff_game = ff_data.groupby(["pitcher", "game_pk", "game_date"]).apply(
        lambda g: pd.Series({
            "ff_velo_game": np.average(g["pt_avg_velo"], weights=g["pt_pitches"])
                            if g["pt_pitches"].sum() > 0 and g["pt_avg_velo"].notna().any()
                            else np.nan,
            "ff_pitches": g["pt_pitches"].sum(),
        })
    ).reset_index()

    ff_game = ff_game.sort_values(["pitcher", "game_date"])

    # L1 velo (last start)
    ff_game["ff_velo_L1"] = ff_game.groupby("pitcher")["ff_velo_game"].shift(1)

    # L5 average velo
    ff_game["ff_velo_L5_avg"] = ff_game.groupby("pitcher")["ff_velo_game"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )

    # Velo delta: L1 vs L5 (negative = velocity dropping)
    ff_game["ff_velo_delta_1v5"] = ff_game["ff_velo_L1"] - ff_game["ff_velo_L5_avg"]

    # Velo drop per start: difference between this start's velo and rolling avg
    # This captures sudden drops that might indicate injury
    ff_game["ff_velo_delta_1vSzn"] = ff_game["ff_velo_L1"] - ff_game.groupby("pitcher")["ff_velo_game"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )

    # Consecutive velo decline: is the pitcher losing velo over the last 3 starts?
    ff_game["ff_velo_L3_trend"] = ff_game.groupby("pitcher")["ff_velo_game"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda w: np.polyfit(range(len(w)), w, 1)[0] if len(w) >= 2 else 0, raw=False
        )
    )

    # Merge to main df
    velo_cols = ["ff_velo_L1", "ff_velo_L5_avg", "ff_velo_delta_1v5",
                 "ff_velo_delta_1vSzn", "ff_velo_L3_trend"]
    ff_merge = ff_game[["pitcher", "game_pk"] + velo_cols].copy()

    pre_cols = set(df.columns)
    # Drop existing to avoid duplicates
    for c in velo_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    normalize_game_pk(df, ff_merge)
    df = df.merge(ff_merge, on=["pitcher", "game_pk"], how="left")

    new_cols = [c for c in df.columns if c not in pre_cols]
    print(f"  ✓ Added {len(new_cols)} velocity trend features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PITCH USAGE SHIFT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_pitch_usage_shift_features(df):
    """
    Build features for recent pitch mix changes. This is one of the strongest
    features in strikeout modeling — pitchers often gain Ks after usage shifts
    (e.g., slider goes from 22% → 34%, Ks spike immediately). Books lag this.

    Features per pitch type (FF, SI, SL, CU, CH):
      - delta_{pt}_usage_3v10: Change in usage between L3 and L10 windows
      - delta_{pt}_usage_3vSzn: Change in usage between L3 and season
    """
    print("  Building pitch usage shift features...")

    pitch_types = {
        "ff_pct": "ff", "si_pct": "si", "sl_pct": "sl",
        "cu_pct": "cu", "ch_pct": "ch", "fc_pct": "fc",
    }

    df = df.sort_values(["pitcher", "game_date"]).copy()
    n_before = len(df.columns)

    for pct_col, pt_name in pitch_types.items():
        if pct_col not in df.columns:
            continue

        # L3 average usage
        shifted = df.groupby("pitcher")[pct_col].shift(1)
        L3_avg = shifted.groupby(df["pitcher"]).transform(
            lambda x: x.rolling(3, min_periods=2).mean()
        )
        # L10 average usage
        L10_avg = shifted.groupby(df["pitcher"]).transform(
            lambda x: x.rolling(10, min_periods=3).mean()
        )
        # Season average usage
        szn_avg = shifted.groupby(df["pitcher"]).transform(
            lambda x: x.expanding(min_periods=3).mean()
        )

        # Delta features
        df[f"delta_{pt_name}_usage_3v10"] = L3_avg - L10_avg
        df[f"delta_{pt_name}_usage_3vSzn"] = L3_avg - szn_avg

    n_after = len(df.columns)
    print(f"  ✓ Added {n_after - n_before} pitch usage shift features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BATTER ORDER SEQUENCE INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_batter_sequence_features(df, lineup_df):
    """
    Build features capturing how strikeout-prone batters cluster in the lineup.
    Ks cluster by innings — if 7-8-9 are all high-K, that's easier K upside
    than the same average K% spread evenly. This is stronger than average K%.

    Features:
      - back_to_back_high_k_pairs: Count of consecutive high-K batter pairs
      - top4_k_cluster: Mean K rate of 4 highest-K batters in lineup
      - bottom_order_k_cluster: Mean K rate of slots 6-9
      - max_consecutive_high_k: Longest streak of K rate > 0.25
      - lineup_k_rate_variance: Variance of K rates (high = clustered)
    """
    print("  Building batter sequence / cluster features...")

    if lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ No lineup data — skipping batter sequence features")
        return df

    lu = lineup_df.copy()
    normalize_game_pk(df, lu)

    # Build batter K rates (same as build_lineup_features)
    game_seasons = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
    lu = lu.merge(game_seasons, on="game_pk", how="left")
    lu = lu.dropna(subset=["season"])
    lu["season"] = lu["season"].astype(int)
    lu["game_date"] = pd.to_datetime(lu["game_date"])
    lu = lu.sort_values(["player_id", "season", "game_date"]).copy()

    for col in ["at_bats", "strikeouts"]:
        shifted = lu.groupby(["player_id", "season"])[col].shift(1)
        lu[f"{col}_cum"] = shifted.groupby([lu["player_id"], lu["season"]]).cumsum()

    ab_cum = lu["at_bats_cum"].replace(0, np.nan)
    lu["batter_k_rate"] = lu["strikeouts_cum"] / ab_cum

    # Build sequence features per lineup
    seq_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        group = group.sort_values("lineup_position")
        k_rates = []
        for pos in range(1, 10):
            row = group[group["lineup_position"] == pos]
            if len(row) > 0 and pd.notna(row.iloc[0]["batter_k_rate"]):
                k_rates.append(row.iloc[0]["batter_k_rate"])
            else:
                k_rates.append(np.nan)

        valid_k = [k for k in k_rates if pd.notna(k)]
        if len(valid_k) < 5:
            continue

        rec = {"game_pk": gpk, "side": side}

        # Back-to-back high-K pairs (K rate > 0.25)
        high_k_threshold = 0.25
        pairs = 0
        for i in range(len(k_rates) - 1):
            if (pd.notna(k_rates[i]) and k_rates[i] > high_k_threshold and
                pd.notna(k_rates[i + 1]) and k_rates[i + 1] > high_k_threshold):
                pairs += 1
        rec["back_to_back_high_k_pairs"] = pairs

        # Top 4 K cluster
        sorted_k = sorted([k for k in valid_k], reverse=True)
        rec["top4_k_cluster"] = np.mean(sorted_k[:4]) if len(sorted_k) >= 4 else np.nan

        # Bottom order cluster (slots 6-9)
        bot_k = [k_rates[i] for i in range(5, 9) if pd.notna(k_rates[i])]
        rec["bottom_order_k_cluster"] = np.mean(bot_k) if len(bot_k) >= 2 else np.nan

        # Max consecutive high-K streak
        max_streak = 0
        curr_streak = 0
        for k in k_rates:
            if pd.notna(k) and k > high_k_threshold:
                curr_streak += 1
                max_streak = max(max_streak, curr_streak)
            else:
                curr_streak = 0
        rec["max_consecutive_high_k"] = max_streak

        # K rate variance
        rec["lineup_k_rate_variance"] = np.var(valid_k)

        seq_records.append(rec)

    if not seq_records:
        print("  ⚠ No sequence records built")
        return df

    seq_df = pd.DataFrame(seq_records)
    seq_cols = [c for c in seq_df.columns if c not in ["game_pk", "side"]]

    # Flip home/away (pitcher faces opposing lineup)
    away_seq = seq_df[seq_df["side"] == "away"].drop(columns=["side"])
    away_seq = away_seq.rename(columns={c: f"opp_{c}" for c in seq_cols})
    home_seq = seq_df[seq_df["side"] == "home"].drop(columns=["side"])
    home_seq = home_seq.rename(columns={c: f"opp_{c}" for c in seq_cols})

    normalize_game_pk(df, away_seq, home_seq)
    df_home = df[df["is_home"] == 1].merge(away_seq, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_seq, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    new_cols = [c for c in seq_cols]
    print(f"  ✓ Added {len(new_cols)} batter sequence features (×opp_)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HANDEDNESS SEQUENCE THROUGH LINEUP
# ══════════════════════════════════════════════════════════════════════════════

def build_handedness_sequence_features(df, lineup_df):
    """
    Build features capturing the handedness ordering through the lineup.
    This impacts pitch sequencing and time-through-the-order effects.
    Consecutive same-side hitters let pitchers lock into one pitch plan;
    alternating sides force more adjustments.

    Features:
      - consecutive_same_side_max: Longest streak of same-hand hitters
      - switch_hitters_count: Number of switch hitters in lineup
      - first5_lefties: Number of LHB in slots 1-5
      - last4_righties: Number of RHB in slots 6-9
      - hand_alternation_count: Number of L→R or R→L transitions
    """
    print("  Building handedness sequence features...")

    if lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ No lineup data — skipping handedness features")
        return df

    lu = lineup_df.copy()
    normalize_game_pk(df, lu)

    hand_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        group = group.sort_values("lineup_position")
        if len(group) < 7:
            continue

        sides = []
        for pos in range(1, 10):
            row = group[group["lineup_position"] == pos]
            if len(row) > 0:
                sides.append(str(row.iloc[0].get("bat_side", "R")))
            else:
                sides.append("R")

        rec = {"game_pk": gpk, "side": side}

        # Consecutive same-side max
        max_streak = 1
        curr_streak = 1
        for i in range(1, len(sides)):
            if sides[i] == sides[i - 1] and sides[i] != "S":
                curr_streak += 1
                max_streak = max(max_streak, curr_streak)
            else:
                curr_streak = 1
        rec["consecutive_same_side_max"] = max_streak

        # Switch hitters count
        rec["switch_hitters_count"] = sum(1 for s in sides if s == "S")

        # First 5 lefties
        rec["first5_lefties"] = sum(1 for s in sides[:5] if s == "L")

        # Last 4 righties
        rec["last4_righties"] = sum(1 for s in sides[5:] if s == "R")

        # Hand alternation count (L→R or R→L transitions, ignoring switch)
        alternations = 0
        prev = None
        for s in sides:
            if s == "S":
                continue
            if prev is not None and s != prev:
                alternations += 1
            prev = s
        rec["hand_alternation_count"] = alternations

        hand_records.append(rec)

    if not hand_records:
        return df

    hand_df = pd.DataFrame(hand_records)
    hand_cols = [c for c in hand_df.columns if c not in ["game_pk", "side"]]

    away_hand = hand_df[hand_df["side"] == "away"].drop(columns=["side"])
    away_hand = away_hand.rename(columns={c: f"opp_{c}" for c in hand_cols})
    home_hand = hand_df[hand_df["side"] == "home"].drop(columns=["side"])
    home_hand = home_hand.rename(columns={c: f"opp_{c}" for c in hand_cols})

    normalize_game_pk(df, away_hand, home_hand)
    df_home = df[df["is_home"] == 1].merge(away_hand, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_hand, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    print(f"  ✓ Added {len(hand_cols)} handedness sequence features (×opp_)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CATCHER-PITCHER COMPATIBILITY INTERACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_catcher_pitcher_compatibility(df):
    """
    Build deeper catcher-pitcher compatibility interactions.
    Some framers help certain pitchers much more — edge-rate pitchers with
    good framers see disproportionate K boosts.

    Features (interactions beyond basic catcher_frm):
      - ix_edge_pitcher_x_framing: Pitcher edge rate × catcher framing
      - ix_slider_pitcher_x_framing: Pitcher slider usage × catcher framing
      - ix_low_zone_pitcher_x_framing: Pitcher low-zone rate × catcher framing
    """
    print("  Building catcher-pitcher compatibility features...")

    n_before = len(df.columns)

    # Catcher framing × pitcher edge rate (zone_pct complement)
    if "catcher_frm" in df.columns:
        frm = df["catcher_frm"].fillna(0.0)

        # Edge pitcher: pitchers who work the edges throw more pitches near the zone border
        # Proxy: 1 - zone_pct (pitchers who DON'T throw in the zone = more edge pitches)
        if "zone_pct_L5" in df.columns:
            edge_rate = 1.0 - df["zone_pct_L5"].fillna(0.44)
            df["ix_edge_pitcher_x_framing"] = edge_rate * frm

        # Slider-heavy pitcher × framing (sliders benefit most from good framing)
        if "sl_pct_L5" in df.columns:
            df["ix_slider_pitcher_x_framing"] = df["sl_pct_L5"].fillna(0.15) * frm

        # CSW pitcher × framing (called-strike + whiff rate = stuff that framing enhances)
        if "csw_pct_L5" in df.columns:
            df["ix_csw_pitcher_x_framing"] = df["csw_pct_L5"].fillna(0.28) * frm

        # Chase rate × framing (pitchers who induce chases benefit from framing near zone)
        if "chase_rate_L5" in df.columns:
            df["ix_chase_pitcher_x_framing"] = df["chase_rate_L5"].fillna(0.30) * frm

    n_after = len(df.columns)
    print(f"  ✓ Added {n_after - n_before} catcher-pitcher compatibility features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER INTERACTION TERMS
# ══════════════════════════════════════════════════════════════════════════════

def build_weather_interaction_features(df):
    """
    Build weather × pitcher-type interaction features.
    Raw weather features are weak; interactions are much stronger.
    E.g., high velocity + high temperature = more swings and misses.

    Features:
      - ix_velo_x_temp: Fastball velocity × temperature
      - ix_breaking_pitcher_x_humidity: Breaking ball usage × humidity
      - ix_flyball_pitcher_x_wind: Flyball rate × wind speed
      - ix_velo_x_altitude: Velocity × altitude proxy (dome indicator)
    """
    print("  Building weather interaction features...")

    n_before = len(df.columns)

    # Velocity × temperature (hot weather = faster bat speed but also more swings)
    if "avg_velocity_L5" in df.columns and "wx_temperature_f" in df.columns:
        velo = df["avg_velocity_L5"].fillna(93.0)
        temp = df["wx_temperature_f"].fillna(72.0)
        # Normalize to make interaction interpretable
        df["ix_velo_x_temp"] = (velo / 93.0) * (temp / 72.0)

    # Breaking ball pitcher × humidity (humidity affects break movement)
    if "breaking_pct_L5" in df.columns and "wx_humidity_pct" in df.columns:
        brk = df["breaking_pct_L5"].fillna(0.30)
        humid = df["wx_humidity_pct"].fillna(60.0)
        df["ix_breaking_x_humidity"] = brk * (humid / 60.0)

    # Hard hit rate × wind speed (wind affects batted ball outcomes)
    if "hard_hit_pct_L5" in df.columns and "wx_wind_speed_mph" in df.columns:
        hh = df["hard_hit_pct_L5"].fillna(0.35)
        wind = df["wx_wind_speed_mph"].fillna(8.0)
        df["ix_hard_hit_x_wind"] = hh * (wind / 8.0)

    # Offspeed pitcher × temperature (offspeed pitchers do better in cold?)
    if "offspeed_pct_L5" in df.columns and "wx_temperature_f" in df.columns:
        os_pct = df["offspeed_pct_L5"].fillna(0.15)
        temp = df["wx_temperature_f"].fillna(72.0)
        df["ix_offspeed_x_temp"] = os_pct * (temp / 72.0)

    # Velocity × dome (dome games have consistent conditions)
    if "avg_velocity_L5" in df.columns and "wx_is_dome" in df.columns:
        velo = df["avg_velocity_L5"].fillna(93.0)
        dome = df["wx_is_dome"].fillna(0)
        df["ix_velo_x_dome"] = velo * dome

    n_after = len(df.columns)
    print(f"  ✓ Added {n_after - n_before} weather interaction features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER-BATTER LINEUP FEATURES & PITCHER-BATTER MATCHUP HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_per_batter_features(df, lineup_df, fg_batting, batter_pt_df):
    """
    Build per-batting-order-slot features for each of the 9 lineup positions.
    Instead of averaging across the lineup, this preserves individual batter
    quality at each slot — critical because a lineup of 4 high-K + 5 low-K
    batters is very different from 9 medium-K batters.

    Also builds pitcher-batter historical matchup features by looking at
    previous games where the starting pitcher faced the same batters.

    Features created per slot (b1_ through b9_):
      - k_rate: Season-to-date K rate
      - k_rate_L10: Recent 10-game K rate
      - bb_rate: Season-to-date BB rate
      - whiff_rate: FanGraphs SwStr%
      - contact_pct: FanGraphs Contact%
      - chase_rate: FanGraphs O-Swing%
      - z_contact: FanGraphs Z-Contact%
      - k_pct_plus: FanGraphs K%+ (100 = league average)
      - wrc_plus: FanGraphs wRC+
      - xwoba: FanGraphs xwOBA (expected weighted on-base average)
      - barrel_pct: FanGraphs Barrel%
      - hard_hit_pct: FanGraphs HardHit%
      - platoon_disadv: 1 if batter has platoon disadvantage vs pitcher

    Summary features:
      - pb_hist_k_rate: Pitcher's historical K rate against this lineup's batters
      - pb_hist_pa: Number of historical PAs (confidence)
      - pb_familiar_batters: How many of these 9 batters the pitcher has seen before
      - lineup_expected_k_total: Sum of individual batter K probabilities × expected PAs
      - lineup_platoon_disadv_count: How many batters have platoon disadvantage
      - lineup_k_rate_p90: 90th percentile K rate in this lineup
      - lineup_k_rate_p10: 10th percentile K rate in this lineup
      - lineup_k_rate_iqr: IQR of K rates (measures lineup diversity)
    """
    print("  Building per-batter lineup features...")

    if lineup_df is None or len(lineup_df) == 0:
        print("  ⚠ No lineup data — skipping per-batter features")
        return df

    lu = lineup_df.copy()

    # Normalize game_pk at the start to prevent dtype drift from prior pipeline steps
    normalize_game_pk(df, lu)

    # Merge season and date info from main df
    game_seasons = df[["game_pk", "season", "game_date"]].drop_duplicates("game_pk")
    lu = lu.merge(game_seasons, on="game_pk", how="left")
    lu = lu.dropna(subset=["season"])
    lu["season"] = lu["season"].astype(int)
    lu["game_date"] = pd.to_datetime(lu["game_date"])

    # ── Step 1: Build per-batter cumulative stats (shifted to prevent leakage) ──
    lu = lu.sort_values(["player_id", "season", "game_date"]).copy()

    for col in ["at_bats", "strikeouts", "hits", "walks"]:
        shifted = lu.groupby(["player_id", "season"])[col].shift(1)
        lu[f"{col}_cum"] = shifted.groupby([lu["player_id"], lu["season"]]).cumsum()

    ab_cum = lu["at_bats_cum"].replace(0, np.nan)
    pa_cum = (lu["at_bats_cum"] + lu["walks_cum"]).replace(0, np.nan)
    lu["batter_k_rate"] = lu["strikeouts_cum"] / ab_cum
    lu["batter_bb_rate"] = lu["walks_cum"] / pa_cum
    lu["batter_hit_rate"] = lu["hits_cum"] / ab_cum

    # Rolling 10-game batter K rate (recent form)
    shifted_k = lu.groupby(["player_id", "season"])["strikeouts"].shift(1)
    shifted_ab = lu.groupby(["player_id", "season"])["at_bats"].shift(1)
    lu["batter_k_L10"] = shifted_k.groupby([lu["player_id"], lu["season"]]).transform(
        lambda x: x.rolling(10, min_periods=3).sum()
    )
    lu["batter_ab_L10"] = shifted_ab.groupby([lu["player_id"], lu["season"]]).transform(
        lambda x: x.rolling(10, min_periods=3).sum()
    )
    lu["batter_k_rate_L10"] = lu["batter_k_L10"] / lu["batter_ab_L10"].replace(0, np.nan)

    # Rolling 10-game batter BB rate
    shifted_bb = lu.groupby(["player_id", "season"])["walks"].shift(1)
    lu["batter_bb_L10"] = shifted_bb.groupby([lu["player_id"], lu["season"]]).transform(
        lambda x: x.rolling(10, min_periods=3).sum()
    )
    lu["batter_bb_rate_L10"] = lu["batter_bb_L10"] / lu["batter_ab_L10"].replace(0, np.nan)

    # ── Step 2: Merge FanGraphs batting data per batter ──
    if fg_batting is not None and len(fg_batting) > 0:
        fg = fg_batting.copy()
        # Build name-based crosswalk from lineups to FanGraphs
        lu_names = lu[["player_id", "player_name", "season"]].drop_duplicates()
        lu_names = lu_names.dropna(subset=["player_name"])

        # Normalize names for matching
        def normalize_name(name):
            if pd.isna(name):
                return ""
            return str(name).strip().lower().replace(".", "").replace(",", "")

        lu_names["name_norm"] = lu_names["player_name"].apply(normalize_name)
        fg["name_norm"] = fg["Name"].apply(normalize_name)

        # Match on name + season
        fg_merge = lu_names.merge(
            fg, left_on=["name_norm", "season"], right_on=["name_norm", "Season"],
            how="left"
        )

        # Key FanGraphs batter stats to use per slot
        fg_stat_map = {
            "SwStr%": "fg_swstr_pct",
            "Contact%": "fg_contact_pct",
            "O-Swing%": "fg_o_swing_pct",
            "Z-Contact%": "fg_z_contact_pct",
            "K%+": "fg_k_pct_plus",
            "wRC+": "fg_wrc_plus",
            "xwOBA": "fg_xwoba",
            "Barrel%": "fg_barrel_pct",
            "HardHit%": "fg_hard_hit_pct",
            "BB%": "fg_bb_pct",
            "ISO": "fg_iso",
            "BABIP": "fg_babip",
            "CSW%": "fg_csw_pct",
            "O-Contact%": "fg_o_contact_pct",
            "TTO%": "fg_tto_pct",
            "Hard%": "fg_hard_pct",
            "Pull%": "fg_pull_pct",
        }

        for src_col, dst_col in fg_stat_map.items():
            if src_col in fg_merge.columns:
                fg_merge[dst_col] = pd.to_numeric(fg_merge[src_col], errors="coerce")

        # Build lookup: (player_id, season) → FG stats
        fg_lookup = fg_merge.groupby(["player_id", "season"]).first().reset_index()
        fg_cols_to_use = [c for c in fg_stat_map.values() if c in fg_lookup.columns]
        fg_lookup = fg_lookup[["player_id", "season"] + fg_cols_to_use]

        lu = lu.merge(fg_lookup, on=["player_id", "season"], how="left")
        matched = lu[fg_cols_to_use[0]].notna().sum() if fg_cols_to_use else 0
        print(f"    FG batter stats matched: {matched}/{len(lu)} lineup entries")
    else:
        fg_cols_to_use = []

    # ── Step 3: Build per-batter pitch-type vulnerability from Statcast ──
    if batter_pt_df is not None and len(batter_pt_df) > 0:
        bpt = batter_pt_df.copy()
        normalize_game_pk(bpt)
        bpt["game_date"] = pd.to_datetime(bpt["game_date"])

        # Classify pitch types
        fb_types = ["FF", "SI", "FC", "FT"]
        brk_types = ["SL", "CU", "SV", "KC", "CS", "SC"]
        os_types = ["CH", "FS", "FO"]

        def classify_pitch(pt):
            if pt in fb_types:
                return "FB"
            elif pt in brk_types:
                return "BRK"
            elif pt in os_types:
                return "OS"
            return None

        bpt["pitch_cat"] = bpt["pitch_type"].apply(classify_pitch)
        bpt = bpt.dropna(subset=["pitch_cat"])

        # Build cumulative batter stats per pitch category (shifted)
        bpt = bpt.sort_values(["batter", "game_date"])
        bpt_agg = bpt.groupby(["batter", "game_pk", "pitch_cat"]).agg(
            pitches=("bpt_pitches_seen", "sum"),
            whiffs=("bpt_whiffs", "sum"),
            ks=("bpt_strikeouts", "sum"),
            pa=("bpt_pa", "sum"),
        ).reset_index()

        # Build season-level cumulative per batter per pitch_cat (prior games only)
        bpt_agg = bpt_agg.merge(
            game_seasons[["game_pk", "season", "game_date"]], on="game_pk", how="left"
        )
        bpt_agg = bpt_agg.sort_values(["batter", "season", "game_date"])

        for col in ["pitches", "whiffs", "ks", "pa"]:
            shifted_col = bpt_agg.groupby(["batter", "season", "pitch_cat"])[col].shift(1)
            bpt_agg[f"{col}_cum"] = shifted_col.groupby(
                [bpt_agg["batter"], bpt_agg["season"], bpt_agg["pitch_cat"]]
            ).cumsum()

        bpt_agg["batter_vs_cat_whiff"] = bpt_agg["whiffs_cum"] / bpt_agg["pitches_cum"].replace(0, np.nan)
        bpt_agg["batter_vs_cat_k_rate"] = bpt_agg["ks_cum"] / bpt_agg["pa_cum"].replace(0, np.nan)

        # Pivot to wide: one row per (batter, game_pk) with vs_FB_whiff, vs_BRK_whiff, etc.
        bpt_wide = bpt_agg.pivot_table(
            index=["batter", "game_pk"],
            columns="pitch_cat",
            values=["batter_vs_cat_whiff", "batter_vs_cat_k_rate"],
            aggfunc="first"
        )
        bpt_wide.columns = [f"{stat}_{cat}" for stat, cat in bpt_wide.columns]
        bpt_wide = bpt_wide.reset_index()

        # Merge into lineup data
        lu = lu.merge(bpt_wide, left_on=["player_id", "game_pk"],
                       right_on=["batter", "game_pk"], how="left", suffixes=("", "_bpt"))
        bpt_cols = [c for c in bpt_wide.columns if c not in ["batter", "game_pk"]]
        print(f"    Per-batter pitch-type vulnerability: {len(bpt_cols)} columns added")
    else:
        bpt_cols = []

    # ── Step 4: Build platoon disadvantage flag ──
    # A batter has platoon disadvantage when facing same-hand pitcher
    # (RHB vs RHP, LHB vs LHP)
    pitcher_hand = df[["game_pk", "p_throws", "is_home"]].drop_duplicates()

    # Home pitchers face away lineup, away pitchers face home lineup
    # Use drop_duplicates on game_pk to avoid row multiplication from openers/relievers
    home_pitchers = pitcher_hand[pitcher_hand["is_home"] == 1][["game_pk", "p_throws"]].drop_duplicates("game_pk")
    away_pitchers = pitcher_hand[pitcher_hand["is_home"] == 0][["game_pk", "p_throws"]].drop_duplicates("game_pk")

    # Normalize game_pk before merge to prevent dtype drift
    normalize_game_pk(lu, home_pitchers, away_pitchers)

    lu_away = lu[lu["side"] == "away"].merge(home_pitchers, on="game_pk", how="left")
    lu_home = lu[lu["side"] == "home"].merge(away_pitchers, on="game_pk", how="left")
    lu = pd.concat([lu_away, lu_home], ignore_index=True)

    # Platoon disadvantage: same-hand matchup (R vs R, L vs L)
    def platoon_disadv(row):
        bs = row.get("bat_side", "")
        pt = row.get("p_throws", "")
        if pd.isna(bs) or pd.isna(pt):
            return 0.5  # unknown
        if bs == "S":  # switch hitter — no disadvantage
            return 0.0
        return 1.0 if bs == pt else 0.0

    lu["platoon_disadv"] = lu.apply(platoon_disadv, axis=1)

    # ── Step 5: Build per-slot features ──
    # For each lineup position (1-9), create features for the batter in that slot
    per_slot_stats = {
        "batter_k_rate": "k_rate",
        "batter_k_rate_L10": "k_rate_L10",
        "batter_bb_rate": "bb_rate",
        "batter_bb_rate_L10": "bb_rate_L10",
        "batter_hit_rate": "hit_rate",
        "platoon_disadv": "platoon_disadv",
    }
    # Add FanGraphs stats
    for fg_col in fg_cols_to_use:
        short_name = fg_col.replace("fg_", "")
        per_slot_stats[fg_col] = short_name

    # Add pitch-type vulnerability stats
    for bpt_col in bpt_cols:
        short_name = bpt_col.replace("batter_", "")
        per_slot_stats[bpt_col] = short_name

    print(f"    Building {len(per_slot_stats)} stats × 9 slots = {len(per_slot_stats)*9} per-slot features")

    slot_records = []
    for (gpk, side), group in lu.groupby(["game_pk", "side"]):
        rec = {"game_pk": gpk, "side": side}

        k_rates = []
        for pos in range(1, 10):
            batter_row = group[group["lineup_position"] == pos]
            prefix = f"b{pos}_"
            if len(batter_row) == 0:
                # Missing batter — fill with league average
                for stat_src, stat_dst in per_slot_stats.items():
                    rec[prefix + stat_dst] = np.nan
            else:
                batter_row = batter_row.iloc[0]
                for stat_src, stat_dst in per_slot_stats.items():
                    val = batter_row.get(stat_src, np.nan)
                    rec[prefix + stat_dst] = val
                k_rate = batter_row.get("batter_k_rate", np.nan)
                if pd.notna(k_rate):
                    k_rates.append(k_rate)

        # Summary features across all slots
        if len(k_rates) >= 5:
            k_arr = np.array(k_rates)
            rec["lineup_k_rate_p90"] = np.percentile(k_arr, 90)
            rec["lineup_k_rate_p10"] = np.percentile(k_arr, 10)
            rec["lineup_k_rate_iqr"] = np.percentile(k_arr, 75) - np.percentile(k_arr, 25)
            rec["lineup_k_rate_skew"] = float(pd.Series(k_arr).skew())
            # Expected K total: sum of K% weighted by expected PA share per slot
            pa_weights = np.array([1.15, 1.10, 1.08, 1.05, 1.02, 0.98, 0.95, 0.90, 0.85])[:len(k_arr)]
            pa_weights = pa_weights / pa_weights.sum()
            rec["lineup_expected_k_total"] = (k_arr * pa_weights).sum()
        else:
            rec["lineup_k_rate_p90"] = np.nan
            rec["lineup_k_rate_p10"] = np.nan
            rec["lineup_k_rate_iqr"] = np.nan
            rec["lineup_k_rate_skew"] = np.nan
            rec["lineup_expected_k_total"] = np.nan

        # Platoon disadvantage count
        platoon_vals = [rec.get(f"b{i}_platoon_disadv", 0) for i in range(1, 10)]
        rec["lineup_platoon_disadv_count"] = sum(v for v in platoon_vals if pd.notna(v))

        slot_records.append(rec)

    slot_df = pd.DataFrame(slot_records)
    print(f"    Built slot features for {len(slot_df)} lineup-games")

    # ── Step 6: Merge to pitcher-game level (flip home/away) ──
    slot_cols = [c for c in slot_df.columns if c not in ["game_pk", "side"]]

    away_slots = slot_df[slot_df["side"] == "away"].drop(columns=["side"])
    away_slots = away_slots.rename(columns={c: f"opp_{c}" for c in slot_cols})

    home_slots = slot_df[slot_df["side"] == "home"].drop(columns=["side"])
    home_slots = home_slots.rename(columns={c: f"opp_{c}" for c in slot_cols})

    # Normalize game_pk before merge to prevent dtype drift
    normalize_game_pk(df, away_slots, home_slots)

    df_home = df[df["is_home"] == 1].merge(away_slots, on="game_pk", how="left")
    df_away = df[df["is_home"] == 0].merge(home_slots, on="game_pk", how="left")
    df = pd.concat([df_home, df_away], ignore_index=True)
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    new_cols = [c for c in df.columns if c.startswith("opp_b") or c.startswith("opp_lineup_k_rate_p")
                or c.startswith("opp_lineup_k_rate_iqr") or c.startswith("opp_lineup_k_rate_skew")
                or c.startswith("opp_lineup_expected") or c.startswith("opp_lineup_platoon")]
    print(f"  ✓ Added {len(new_cols)} per-batter lineup features")

    return df


def build_pitcher_batter_history(df, lineup_df, batter_pt_df):
    """
    Build pitcher-batter historical matchup features.
    For each game, looks at which batters are in the opposing lineup, then
    checks previous games where those same batters appeared in a game where
    this pitcher started, to compute historical K rates.

    Uses batter_pitch_type data joined with pitcher-game data to find
    actual pitcher-batter confrontations.
    """
    print("  Building pitcher-batter historical matchup features...")

    if lineup_df is None or len(lineup_df) == 0 or batter_pt_df is None or len(batter_pt_df) == 0:
        print("  ⚠ Missing lineup or batter data — skipping matchup history")
        return df

    # Build a lookup of which batters were in the lineup for each game + side
    lu = lineup_df[["game_pk", "side", "player_id", "lineup_position"]].copy()
    lu = lu.dropna(subset=["player_id"])
    lu["player_id"] = lu["player_id"].astype(int)

    # Normalize game_pk across all DataFrames to prevent dtype drift
    normalize_game_pk(df, lu)

    # Build pitcher → game mapping with dates
    pitcher_games = df[["pitcher", "game_pk", "game_date", "season", "is_home",
                         "strikeouts", "plate_appearances"]].copy()
    pitcher_games["game_date"] = pd.to_datetime(pitcher_games["game_date"])

    # For each pitcher-game, the opposing lineup is:
    # if pitcher is home → away lineup, if pitcher is away → home lineup
    pitcher_games["opp_side"] = pitcher_games["is_home"].map({1: "away", 0: "home"})

    # Merge to get opposing batters for each pitcher-game
    opp_batters = pitcher_games.merge(
        lu, left_on=["game_pk", "opp_side"], right_on=["game_pk", "side"], how="inner"
    )

    # Now aggregate batter-level data from batter_pitch_type to game level
    bpt = batter_pt_df[["batter", "game_pk", "bpt_pitches_seen", "bpt_whiffs",
                         "bpt_strikeouts", "bpt_pa"]].copy()
    bpt_game = bpt.groupby(["batter", "game_pk"]).agg(
        total_pitches=("bpt_pitches_seen", "sum"),
        total_whiffs=("bpt_whiffs", "sum"),
        total_ks=("bpt_strikeouts", "sum"),
        total_pa=("bpt_pa", "sum"),
    ).reset_index()

    # Join: for each (pitcher, game, batter), get the batter's stats IN THAT GAME
    # This approximates the pitcher-batter matchup (the batter saw the starter for most PAs)
    opp_batters = opp_batters.merge(
        bpt_game, left_on=["player_id", "game_pk"], right_on=["batter", "game_pk"], how="left"
    )

    # Now for each current game, look at PREVIOUS games where this pitcher faced these batters
    opp_batters = opp_batters.sort_values(["pitcher", "player_id", "game_date"])

    # For each (pitcher, batter) pair, compute cumulative stats from prior meetings
    opp_batters["shifted_ks"] = opp_batters.groupby(["pitcher", "player_id"])["total_ks"].shift(1)
    opp_batters["shifted_pa"] = opp_batters.groupby(["pitcher", "player_id"])["total_pa"].shift(1)
    opp_batters["shifted_whiffs"] = opp_batters.groupby(["pitcher", "player_id"])["total_whiffs"].shift(1)
    opp_batters["shifted_pitches"] = opp_batters.groupby(["pitcher", "player_id"])["total_pitches"].shift(1)

    opp_batters["cum_ks"] = opp_batters.groupby(["pitcher", "player_id"])["shifted_ks"].cumsum()
    opp_batters["cum_pa"] = opp_batters.groupby(["pitcher", "player_id"])["shifted_pa"].cumsum()
    opp_batters["cum_whiffs"] = opp_batters.groupby(["pitcher", "player_id"])["shifted_whiffs"].cumsum()
    opp_batters["cum_pitches"] = opp_batters.groupby(["pitcher", "player_id"])["shifted_pitches"].cumsum()

    # Has the pitcher seen this batter before?
    opp_batters["has_history"] = opp_batters["cum_pa"].fillna(0) > 0

    # Aggregate back to pitcher-game level
    history_agg = []
    for (pitcher_id, gpk), group in opp_batters.groupby(["pitcher", "game_pk"]):
        familiar = group[group["has_history"]]
        total_cum_ks = familiar["cum_ks"].sum()
        total_cum_pa = familiar["cum_pa"].sum()
        total_cum_whiffs = familiar["cum_whiffs"].sum()
        total_cum_pitches = familiar["cum_pitches"].sum()

        hist_k_rate = total_cum_ks / total_cum_pa if total_cum_pa > 0 else np.nan
        hist_whiff_rate = total_cum_whiffs / total_cum_pitches if total_cum_pitches > 0 else np.nan

        history_agg.append({
            "pitcher": pitcher_id,
            "game_pk": gpk,
            "pb_hist_k_rate": hist_k_rate,
            "pb_hist_whiff_rate": hist_whiff_rate,
            "pb_hist_pa": total_cum_pa if total_cum_pa > 0 else 0,
            "pb_familiar_batters": len(familiar),
            "pb_total_lineup_batters": len(group),
        })

    hist_df = pd.DataFrame(history_agg)
    # Familiarity ratio
    hist_df["pb_familiarity_pct"] = (
        hist_df["pb_familiar_batters"] / hist_df["pb_total_lineup_batters"].replace(0, 1)
    )

    # Merge back to main df
    normalize_game_pk(df, hist_df)
    pre_cols = set(df.columns)
    df = df.merge(hist_df.drop(columns=["pb_total_lineup_batters"], errors="ignore"),
                   on=["pitcher", "game_pk"], how="left")

    new_cols = [c for c in df.columns if c not in pre_cols]
    print(f"  ✓ Added {len(new_cols)} pitcher-batter history features")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TARGET VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def add_targets(df):
    """Define prediction targets."""
    print("  Adding target variables...")

    df["target_strikeouts"] = df["strikeouts"]
    df["target_k_pct"] = df["k_pct"]
    df["target_whiff_pct"] = df["whiff_pct"]
    df["target_hits_allowed"] = df["hits_allowed"]
    df["target_walks"] = df["walks"]
    df["target_home_runs"] = df["home_runs_allowed"]
    df["target_total_pitches"] = df["total_pitches"]
    df["target_outs_recorded"] = df["outs_recorded"]

    # Binary targets for prop-style bets
    df["target_k_over_5_5"] = (df["strikeouts"] >= 6).astype(int)
    df["target_k_over_6_5"] = (df["strikeouts"] >= 7).astype(int)
    df["target_k_over_4_5"] = (df["strikeouts"] >= 5).astype(int)

    # ── Hits / walks binary prop targets (parallel to K props) ──
    # Standard hits-allowed lines for starting pitchers cluster at 4.5-6.5;
    # standard walks lines at 1.5-2.5.
    df["target_h_over_4_5"] = (df["hits_allowed"] >= 5).astype(int)
    df["target_h_over_5_5"] = (df["hits_allowed"] >= 6).astype(int)
    df["target_h_over_6_5"] = (df["hits_allowed"] >= 7).astype(int)
    df["target_bb_over_1_5"] = (df["walks"] >= 2).astype(int)
    df["target_bb_over_2_5"] = (df["walks"] >= 3).astype(int)
    df["target_bb_over_3_5"] = (df["walks"] >= 4).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_finalize(df):
    """Handle NaNs, drop rows without enough history."""
    print("  Final cleanup...")

    # Drop first start per pitcher (no rolling data possible)
    df = df[df.groupby("pitcher").cumcount() > 0].copy()

    # Drop very early-season games where cumulative stats are unreliable
    df = df[df["start_num"] >= 1].copy()

    # Fill NaN rates with 0
    rate_cols = [c for c in df.columns if any(
        x in c for x in ["_pct", "_rate", "trend", "above_avg"]
    )]
    df[rate_cols] = df[rate_cols].fillna(0)

    # Fill NaN counts with 0
    count_cols = [c for c in df.columns if c.endswith("_szn") and "pct" not in c]
    df[count_cols] = df[count_cols].fillna(0)

    # ── H/W pipeline NaN handling ──
    # BABIP-family columns: aren't caught by the _pct/_rate patterns above
    # but should also fill to 0 (XGBoost handles NaN, but downstream daily
    # script smart-fills assume 0/league-avg).
    babip_cols = [c for c in df.columns if "babip" in c.lower() or "lob_pct" in c.lower()]
    df[babip_cols] = df[babip_cols].fillna(0)

    # Raw per-game hit-type / batted-ball-type counts (e.g. singles, doubles,
    # ground_balls). Where no PA-events data exists for a game, treat as 0
    # (the model already knows that pitcher had hits_allowed via a separate
    # column, so 0 here just means "we don't know the breakdown").
    raw_count_cols = [c for c in df.columns if c in {
        "singles", "doubles", "triples",
        "ground_balls", "fly_balls", "line_drives", "popups",
        "infield_fly_balls", "soft_hits",
    }]
    df[raw_count_cols] = df[raw_count_cols].fillna(0)

    # Contact-quality averages (exit velo, launch angle, xBA, xwOBA): leave
    # as NaN. Tree models handle NaN natively and 0 would be misleading
    # (no batted ball has 0 exit velocity). The daily script's
    # smart_feature_get() supplies a league-average fallback at inference.

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MLB Pitcher Feature Engineering (Enhanced)")
    print("=" * 60)

    # Load all data
    print("\n── Loading data ──")
    data = load_all_data()

    df = data["pitcher_games"]

    # Filter to starters
    print("\n── Identifying starting pitchers ──")
    df = identify_starters(df)

    # Merge game metadata
    print("\n── Merging game metadata ──")
    df = merge_game_metadata(df, data.get("game_meta"))

    # Build all feature categories
    print("\n── Engineering features ──")
    # IMPORTANT: build_batted_ball_features must run BEFORE build_rolling_features
    # and build_season_cumulative — those consume the raw singles/doubles/triples,
    # ground_balls/fly_balls/line_drives/popups, and avg_exit_velocity columns
    # that this function backfills from data/statcast_pa_events_all.csv.
    df = build_batted_ball_features(df)
    df = build_rolling_features(df)
    df = build_season_cumulative(df)
    df = build_trend_features(df)
    df = build_opposing_team_features(df)
    df = build_lineup_features(df, data.get("lineups"))
    df = build_lineup_quality_features(df, data.get("lineups"), data.get("batter_rolling"))
    df = build_pitch_type_features(df, data.get("pitcher_pt"), data.get("batter_pt"), data.get("lineups"))
    df = build_platoon_features(df)
    df = build_umpire_features(df, data.get("umpires"))
    df = build_park_features(df, data.get("park_factors"), data.get("venues"))
    df = build_weather_features(df, data.get("weather"))
    df = build_schedule_features(df)
    df = build_early_season_features(df)
    df = build_blended_rate_features(df)
    df = build_opposing_starter_features(df)
    df = build_fangraphs_pitcher_features(df, data.get("fg_pitching"))
    df = build_catcher_framing_features(df, data.get("catchers"), data.get("fg_batting"))
    df = build_catcher_calling_features(df, data.get("catchers"),
                                         data.get("catcher_features_asof"),
                                         data.get("catcher_features_prior"))
    df = build_lineup_plate_discipline(df, data.get("fg_batting"), data.get("lineups"))
    df = build_velocity_and_platoon_features(df)
    df = build_advanced_features(df, lineup_df=data.get("lineups"))
    df = build_bullpen_features(df)
    df = build_velocity_trend_features(df, data.get("pitcher_pt"))
    df = build_pitch_usage_shift_features(df)
    df = build_batter_sequence_features(df, data.get("lineups"))
    df = build_handedness_sequence_features(df, data.get("lineups"))
    df = build_catcher_pitcher_compatibility(df)
    df = build_weather_interaction_features(df)
    df = build_per_batter_features(df, data.get("lineups"), data.get("fg_batting"), data.get("batter_pt"))
    df = build_pitcher_batter_history(df, data.get("lineups"), data.get("batter_pt"))
    df = add_targets(df)

    # Cleanup
    print("\n── Finalizing ──")
    df = clean_and_finalize(df)

    # Save
    output_path = DATA_DIR / "pitcher_model_features.csv"
    df.to_csv(output_path, index=False)

    # Feature summary by category
    print(f"\n═══ Feature Engineering Complete ═══")
    print(f"  Final dataset: {len(df):,} rows × {len(df.columns)} columns")

    categories = {
        "Rolling pitcher stats": [c for c in df.columns if "_L3" in c or "_L5" in c or "_L10" in c],
        "Season cumulative": [c for c in df.columns if "_szn" in c and "target" not in c],
        "Trend features": [c for c in df.columns if "trend" in c],
        "Opposing team": [c for c in df.columns if c.startswith("opp_") and "lu_" not in c and "opp_sp_" not in c],
        "Opposing starter": [c for c in df.columns if c.startswith("opp_sp_") or c.startswith("ix_both") or c.startswith("ix_aces") or c.startswith("ix_combined")],
        "Lineup-level": [c for c in df.columns if "lu_" in c and "vs_" not in c and c not in [
            "pitcher_x_lineup_k", "whiff_x_lineup_k", "k_trend_x_lu_recent"]],
        "Pitch-type (pitcher)": [c for c in df.columns if c.startswith("pt_") and "_L" in c],
        "Pitch-type (lineup vs)": [c for c in df.columns if "lu_vs_" in c],
        "Pitch-type (cross)": [c for c in df.columns if c.startswith("cross_")],
        "Interactions": [c for c in df.columns if c in [
            "pitcher_x_lineup_k", "whiff_x_lineup_k", "k_trend_x_lu_recent"]],
        "Platoon": [c for c in df.columns if "platoon" in c or "vs_left" in c or "vs_right" in c],
        "Umpire": [c for c in df.columns if c.startswith("ump_")],
        "Park/Venue": [c for c in df.columns if c.startswith("pf_") or "dome" in c or "venue" in c],
        "Weather": [c for c in df.columns if c.startswith("wx_")],
        "Schedule": [c for c in df.columns if c in [
            "rest_days", "short_rest", "extra_rest", "is_night_game",
            "is_weekend", "day_of_week", "month", "is_home", "start_num",
        ]],
        "Early-season": [c for c in df.columns if any(
            x in c for x in ["days_into", "season_phase", "is_first_month",
                              "prior_starts", "_prev5", "_prev10", "pvt_",
                              "_x_reliability", "early_x_"]
        )],
        "FanGraphs pitcher (Stuff+, SwStr%)": [c for c in df.columns if c.startswith("fg_")],
        "Catcher framing": [c for c in df.columns if "catcher" in c and "target" not in c],
        "Lineup plate discipline": [c for c in df.columns if any(
            x in c for x in ["lu_swstr", "lu_o_swing", "lu_z_contact", "lu_contact_pct",
                              "lu_tto", "lu_csw", "lu_fg_k", "lu_barrel", "lu_hard_hit",
                              "lu_k_rate_std", "lu_max_k", "lu_min_k"]
        )],
        "Velocity delta & platoon": [c for c in df.columns if any(
            x in c for x in ["velo_delta", "whiff_pct_vs_left", "whiff_pct_vs_right",
                              "platoon_whiff_diff", "pitcher_tto_L", "pitcher_tto_szn"]
        )],
        "New interactions": [c for c in df.columns if c.startswith("ix_stuff") or
                             c.startswith("ix_pitcher_lu") or c == "ix_tto_matchup" or
                             c == "ix_swstr_x_contact"],
        "Per-batter slot features": [c for c in df.columns if c.startswith("opp_b") and
                                      any(c.startswith(f"opp_b{i}_") for i in range(1, 10))],
        "Lineup distribution": [c for c in df.columns if any(
            x in c for x in ["lineup_k_rate_p90", "lineup_k_rate_p10", "lineup_k_rate_iqr",
                              "lineup_k_rate_skew", "lineup_expected_k", "lineup_platoon_disadv"]
        )],
        "Pitcher-batter history": [c for c in df.columns if c.startswith("pb_")],
        # ── New H/W categories ──
        "Hits/walks rate features": [c for c in df.columns if any(
            x in c for x in ["hits_per_pa", "bb_per_pa", "hr_per_pa",
                              "h_per_9", "bb_per_9", "hr_per_9",
                              "k_minus_bb_pct", "hr_per_bip", "hr_per_fb"]
        )],
        "BABIP / LOB% / luck stats": [c for c in df.columns if any(
            x in c for x in ["babip", "lob_pct"]
        )],
        "Batted-ball mix": [c for c in df.columns if any(
            x in c for x in ["gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct"]
        ) and not c.startswith("opp_") and "lu_" not in c],
        "Contact quality": [c for c in df.columns if any(
            x in c for x in ["avg_exit_velocity", "avg_launch_angle",
                              "sweet_spot_pct", "solid_contact_pct",
                              "avg_xba_contact", "avg_xwoba_contact",
                              "soft_hit_pct"]
        )],
        "FanGraphs hits-side (FIP/SIERA/HR-FB/etc.)": [c for c in df.columns if any(
            x in c for x in ["fg_fip", "fg_xfip", "fg_siera", "fg_tera", "fg_xera",
                              "fg_era", "fg_whip", "fg_lob_pct", "fg_hr_per_fb",
                              "fg_k_minus_bb_pct", "fg_k_per_9", "fg_bb_per_9",
                              "fg_hr_per_9", "fg_gb_pct", "fg_fb_pct", "fg_ld_pct",
                              "fg_iffb_pct", "fg_babip_allowed", "fg_soft_pct",
                              "fg_med_pct", "fg_hard_pct_allowed",
                              "fg_barrel_pct_allowed", "fg_hard_hit_pct_allowed"]
        )],
        "Targets": [c for c in df.columns if c.startswith("target_")],
    }

    for cat_name, cols in categories.items():
        print(f"\n  {cat_name} ({len(cols)} features):")
        for col in sorted(cols)[:8]:
            print(f"    - {col}")
        if len(cols) > 8:
            print(f"    ... and {len(cols) - 8} more")

    print(f"\n  Saved to: {output_path.resolve()}")
    print("\nNext step: Run train/baseline.py")
