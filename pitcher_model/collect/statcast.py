"""
MLB Pitcher Stats Collection Pipeline (Enhanced)
==================================================
Collects game-level pitcher data from multiple sources:
  - Statcast pitch-level data (raw pitches + pitcher-game aggregations
    + per-PA terminating events)
  - MLB Stats API: schedules, lineups, umpires, venues, starting catchers
  - Opposing team batting stats
  - Ballpark factors

FanGraphs season stats are handled separately by collect/fangraphs.py
because FanGraphs is now behind Cloudflare and blocks pybaseball's
automated requests.

Requirements:
    pip install pybaseball pandas numpy requests tqdm

Usage:
    python run.py collect statcast

Notes:
    - First run takes 1-2 hours for multi-season Statcast pulls
    - Each season is cached to disk — safe to interrupt and resume
    - MLB Stats API calls are lightweight and fast
    - For the current season, the end date auto-tracks to yesterday so
      re-running pulls newly available games. Update _CURRENT_SEASON
      at the top of the file each year.
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pybaseball import (
    statcast,
)

from pitcher_model.paths import DATA_DIR as OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR)
# Note: pybaseball.pitching_stats / batting_stats fetches from FanGraphs
# directly, which is now blocked by Cloudflare (returns 403 on every call).
# FanGraphs collection has been moved to collect/fangraphs.py, which uses
# manually exported CSVs. Do not re-add those imports here.

# park_factors may not exist in all pybaseball versions
try:
    from pybaseball import park_factors
except ImportError:
    park_factors = None
    print("  ℹ pybaseball.park_factors not available in your version — will use fallback")

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

CHUNK_DAYS = 14

# When a new season starts, update _CURRENT_SEASON. SEASONS is derived from
# it. The current-season end date auto-tracks to yesterday so re-running
# pulls newly available games (Statcast lags ~6-12 hours after games end).
_CURRENT_SEASON = 2026
_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

SEASONS = list(range(2021, _CURRENT_SEASON + 1))

SEASON_DATES = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    _CURRENT_SEASON: (f"{_CURRENT_SEASON}-03-25", _yesterday),
}

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Team abbreviation mapping (Statcast abbrev -> MLB API full names)
# pybaseball uses Statcast-style abbreviations
TEAM_ABBREVS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SF": 137, "SEA": 136,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}


# ── Helper: Statcast Collection ─────────────────────────────────────────────
def pull_statcast_chunked(start_dt, end_dt, chunk_days=CHUNK_DAYS):
    """Pull Statcast data in date chunks to avoid API timeouts."""
    all_data = []
    current = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)

    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        print(f"    Pulling {current.date()} to {chunk_end.date()}...")
        try:
            chunk = statcast(
                start_dt=str(current.date()),
                end_dt=str(chunk_end.date()),
            )
            if chunk is not None and len(chunk) > 0:
                all_data.append(chunk)
            time.sleep(2)
        except Exception as e:
            print(f"    ⚠ Error: {e}. Retrying in 10s...")
            time.sleep(10)
            try:
                chunk = statcast(
                    start_dt=str(current.date()),
                    end_dt=str(chunk_end.date()),
                )
                if chunk is not None and len(chunk) > 0:
                    all_data.append(chunk)
            except Exception as e2:
                print(f"    ✗ Skipping chunk: {e2}")
        current = chunk_end + pd.Timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# ── Helper: MLB Stats API ───────────────────────────────────────────────────
def mlb_api_get(endpoint, params=None):
    """Make a request to the MLB Stats API."""
    url = f"{MLB_API_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    ⚠ MLB API error ({endpoint}): {e}")
        return None


def get_schedule_for_date(date_str):
    """Get all games for a given date from MLB API."""
    data = mlb_api_get("schedule", {
        "date": date_str,
        "sportId": 1,
        "hydrate": "linescore,probablePitcher,venue,officialScorer,decisions",
    })
    if data and data.get("dates"):
        return data["dates"][0].get("games", [])
    return []


def get_game_boxscore(game_pk):
    """Get boxscore for a specific game (includes lineups and umpires)."""
    data = mlb_api_get(f"game/{game_pk}/boxscore")
    return data


def get_game_linescore(game_pk):
    """Get linescore (inning-by-inning) for a specific game."""
    data = mlb_api_get(f"game/{game_pk}/linescore")
    return data


# ── Aggregation: Pitch-Level to Pitcher-Game ─────────────────────────────────
PA_EVENTS_PATH = OUTPUT_DIR / "statcast_pa_events_all.csv"

# Pitch-level cache: every pitch from every game, with the columns needed
# for catcher game-calling, pitch sequencing, framing analysis, and
# location-based features. Written incrementally per season (chunked append)
# so a partial run can resume.
PITCHES_PATH = OUTPUT_DIR / "statcast_pitches_all.csv"

# Columns we keep at pitch level. This is intentionally a small subset
# (~25 of ~90 Statcast fields) to avoid the disk and load-time cost of
# saving everything. Anything not on this list either:
#   - causes target leakage if used as a feature (woba_value, estimated_*)
#   - is derivable from what's here (game_year, type from description)
#   - is too niche for current modeling priorities (release_pos_*, vx0)
# If a future feature needs a field that's not here, add it and re-pull.
PITCH_LEVEL_COLUMNS = [
    # Identity / context
    "game_pk", "game_date", "pitcher", "batter", "fielder_2",
    "stand", "p_throws", "inning", "at_bat_number", "pitch_number",
    "home_team", "away_team",
    # Count state
    "balls", "strikes",
    # Pitch characteristics
    "pitch_type", "release_speed", "pfx_x", "pfx_z", "release_spin_rate",
    # Outcome at the pitch level
    "description", "type", "zone",
    # Location
    "plate_x", "plate_z",
    # PA-terminating event (null on non-terminal pitches)
    "events",
    # ── Contact-quality fields (added for H/W/outs pipeline) ──
    # NaN on non-BIP rows. Needed to compute per-game gb_pct, fb_pct,
    # avg_exit_velocity, sweet_spot_pct, avg_xba_contact,
    # avg_xwoba_contact (used by the hits/walks/outs models).
    # Adding them here means future Statcast re-pulls will cache these
    # automatically. Existing pitch caches predate this — to backfill,
    # delete data/statcast_pitches_all.csv and re-run 01.
    "bb_type", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
]


def extract_pa_events(statcast_df):
    """
    Return one row per plate appearance (PA-terminating pitches) with
    minimum columns needed for the per-PA strikeout model in 08, plus
    contact-quality fields needed for the hits/walks/outs models'
    batted-ball features (gb_pct, fb_pct, avg_exit_velocity, etc.).

    The extra fields (bb_type, launch_speed, launch_angle, xBA, xwOBA
    on contact) come automatically from pybaseball's statcast() pull —
    we just need to keep them when slimming the per-PA rows. They're
    NaN for non-BIP events, which is the correct semantics:
    build_batted_ball_features() in 02 only aggregates over rows where
    bb_type is non-null.
    """
    if statcast_df is None or len(statcast_df) == 0:
        return pd.DataFrame()
    pa = statcast_df[statcast_df["events"].notna()].copy()
    if len(pa) == 0:
        return pd.DataFrame()
    keep = ["game_pk", "game_date", "pitcher", "batter",
            "stand", "p_throws", "inning", "at_bat_number",
            "home_team", "away_team", "events",
            # ── Contact-quality columns (added for H/W/outs pipeline) ──
            # These come straight from Statcast — no extra API calls.
            # NaN on non-BIP PAs (K/BB/HBP), which the downstream
            # aggregation in 02 correctly ignores.
            "bb_type", "launch_speed", "launch_angle",
            "estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle",
            ]
    keep = [c for c in keep if c in pa.columns]
    pa = pa[keep].copy()
    pa["was_K"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa["was_BB"] = pa["events"].isin(["walk", "intent_walk"]).astype(int)
    pa["was_HBP"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["was_hit"] = pa["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    pa["was_in_play"] = (~pa["events"].isin([
        "strikeout", "strikeout_double_play",
        "walk", "intent_walk", "hit_by_pitch",
    ])).astype(int)
    return pa


def save_pa_events_chunk(statcast_df, output_path=None):
    """Append PA-terminating rows to data/statcast_pa_events_all.csv."""
    from pathlib import Path as _P
    if output_path is None:
        output_path = PA_EVENTS_PATH
    pa = extract_pa_events(statcast_df)
    if len(pa) == 0:
        return 0
    header = not _P(output_path).exists()
    pa.to_csv(output_path, mode="a", header=header, index=False)
    return len(pa)


def extract_pitch_fields(statcast_df):
    """
    Return every pitch with the curated PITCH_LEVEL_COLUMNS subset. Drops
    columns that aren't present in the input (older Statcast pulls may
    not have every field, though all of these have existed since 2018).
    """
    if statcast_df is None or len(statcast_df) == 0:
        return pd.DataFrame()
    keep = [c for c in PITCH_LEVEL_COLUMNS if c in statcast_df.columns]
    missing = [c for c in PITCH_LEVEL_COLUMNS if c not in statcast_df.columns]
    if missing:
        # Print once per chunk — useful diagnostic if an expected column
        # vanishes upstream (e.g. Statcast schema change).
        print(f"    ⚠ pitch-level: {len(missing)} expected columns missing "
              f"from this chunk: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return statcast_df[keep].copy()


def save_pitches_chunk(statcast_df, output_path=None):
    """Append pitch-level rows to data/statcast_pitches_all.csv.

    Idempotent at the year level via the same _pitches_years_covered
    tracking the caller does for PA events.
    """
    from pathlib import Path as _P
    if output_path is None:
        output_path = PITCHES_PATH
    pitches = extract_pitch_fields(statcast_df)
    if len(pitches) == 0:
        return 0
    header = not _P(output_path).exists()
    pitches.to_csv(output_path, mode="a", header=header, index=False)
    return len(pitches)


def aggregate_pitcher_game(statcast_df):
    """
    Aggregate pitch-level Statcast data to pitcher-game level with
    comprehensive pitch mix and outcome stats.

    All boolean/flag columns use .fillna(False) before .astype(int)
    to handle NaN values safely in source columns.
    """
    df = statcast_df.copy()
    df = df[df["pitcher"].notna()].copy()

    # ── Pitch-level flags ──
    # .isin() is NaN-safe (NaN never matches), so these are fine as-is
    df["is_strike"] = df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt",
    ]).astype(int)
    df["is_whiff"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ]).astype(int)
    df["is_ball"] = df["description"].isin(["ball", "blocked_ball"]).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["is_batted_ball"] = (df["description"] == "hit_into_play").astype(int)

    # Zone columns can have NaN — must fillna before astype
    if "zone" in df.columns:
        df["is_in_zone"] = df["zone"].between(1, 9).fillna(False).astype(int)
        df["is_out_of_zone"] = (~df["zone"].between(1, 9)).fillna(False).astype(int)
    else:
        df["is_in_zone"] = 0
        df["is_out_of_zone"] = 0
    df["is_chase"] = (df["is_out_of_zone"] & df["is_whiff"]).astype(int)

    # Batted ball quality — launch_speed/launch_angle have many NaNs
    if "launch_speed" in df.columns:
        df["is_barrel"] = (
            (df["launch_speed"] >= 98) &
            (df["launch_angle"].between(26, 30))
        ).fillna(False).astype(int)
        df["is_hard_hit"] = (df["launch_speed"] >= 95).fillna(False).astype(int)
        df["is_soft_hit"] = (df["launch_speed"] < 70).fillna(False).astype(int)
        # ── Sweet-spot / solid-contact flags (added for H/W/outs pipeline) ──
        # Sweet spot: launch angle in [8, 32] degrees (Statcast definition).
        # Solid contact: launch_speed >= 95 AND launch_angle in [-5, 35]
        # (covers barrels + solid hits, looser than barrel definition).
        if "launch_angle" in df.columns:
            df["is_sweet_spot"] = (
                df["launch_angle"].between(8, 32)
            ).fillna(False).astype(int)
            df["is_solid_contact"] = (
                (df["launch_speed"] >= 95) &
                (df["launch_angle"].between(-5, 35))
            ).fillna(False).astype(int)
        else:
            df["is_sweet_spot"] = 0
            df["is_solid_contact"] = 0
    else:
        df["is_barrel"] = 0
        df["is_hard_hit"] = 0
        df["is_soft_hit"] = 0
        df["is_sweet_spot"] = 0
        df["is_solid_contact"] = 0

    # ── Batted-ball-type flags (added for H/W/outs pipeline) ──
    # Statcast's bb_type field categorizes each BIP as one of:
    # 'ground_ball', 'fly_ball', 'line_drive', 'popup'. NaN on non-BIP PAs.
    # We aggregate these into per-game counts so downstream feature
    # engineering can compute gb_pct, fb_pct, ld_pct, pop_pct.
    if "bb_type" in df.columns:
        df["is_ground_ball"] = (df["bb_type"] == "ground_ball").fillna(False).astype(int)
        df["is_fly_ball"]    = (df["bb_type"] == "fly_ball").fillna(False).astype(int)
        df["is_line_drive"]  = (df["bb_type"] == "line_drive").fillna(False).astype(int)
        df["is_popup"]       = (df["bb_type"] == "popup").fillna(False).astype(int)
        # Infield fly ball ≈ popup. IFFB% is a FanGraphs-style stat,
        # equivalent to popup_pct here.
        df["is_infield_fly_ball"] = df["is_popup"]
    else:
        df["is_ground_ball"] = 0
        df["is_fly_ball"] = 0
        df["is_line_drive"] = 0
        df["is_popup"] = 0
        df["is_infield_fly_ball"] = 0

    # Event flags — events is NaN for non-terminal pitches, .isin() handles this
    df["is_strikeout"] = df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    df["is_walk"] = df["events"].isin(["walk"]).astype(int)
    df["is_hbp"] = df["events"].isin(["hit_by_pitch"]).astype(int)
    df["is_single"] = (df["events"] == "single").astype(int)
    df["is_double"] = (df["events"] == "double").astype(int)
    df["is_triple"] = (df["events"] == "triple").astype(int)
    df["is_home_run"] = (df["events"] == "home_run").astype(int)
    df["is_hit"] = df["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    df["is_plate_appearance"] = df["events"].notna().astype(int)

    # ── Pitch type flags ──
    pitch_types = {
        "fastball": ["FF", "SI", "FC"],
        "breaking": ["SL", "CU", "KC", "CS", "SV", "SC"],
        "offspeed": ["CH", "FS", "FO", "KN", "EP"],
    }
    for category, codes in pitch_types.items():
        df[f"is_{category}"] = df["pitch_type"].isin(codes).astype(int)

    for pt in ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "SV", "KC"]:
        df[f"is_pt_{pt}"] = (df["pitch_type"] == pt).astype(int)

    # ── Batter handedness ──
    df["is_vs_left"] = (df["stand"] == "L").astype(int)
    df["is_vs_right"] = (df["stand"] == "R").astype(int)

    # ── Grouping & Aggregation ──
    group_keys = ["pitcher", "game_pk", "game_date", "home_team", "away_team"]
    if "p_throws" in df.columns:
        group_keys.append("p_throws")

    grouped = df.groupby(group_keys)

    agg = grouped.agg(
        # Volume
        total_pitches=("pitch_type", "count"),
        plate_appearances=("is_plate_appearance", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        # Outcomes
        strikeouts=("is_strikeout", "sum"),
        walks=("is_walk", "sum"),
        hbp=("is_hbp", "sum"),
        hits_allowed=("is_hit", "sum"),
        singles=("is_single", "sum"),
        doubles=("is_double", "sum"),
        triples=("is_triple", "sum"),
        home_runs_allowed=("is_home_run", "sum"),
        # Pitch quality
        strikes=("is_strike", "sum"),
        balls=("is_ball", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        in_zone_pitches=("is_in_zone", "sum"),
        out_of_zone_pitches=("is_out_of_zone", "sum"),
        chases=("is_chase", "sum"),
        # Contact quality
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        soft_hits=("is_soft_hit", "sum"),
        # ── Sweet-spot / solid-contact (H/W/outs pipeline additions) ──
        # Counts of batted balls in the sweet-spot launch-angle band and
        # of "solid" contact (95+ EV in a reasonable launch-angle range).
        # Rates (sweet_spot_pct, solid_contact_pct) are derived below.
        sweet_spot_hits=("is_sweet_spot", "sum"),
        solid_contact_hits=("is_solid_contact", "sum"),
        # ── Batted-ball-type counts (H/W/outs pipeline additions) ──
        # These give us gb_pct, fb_pct, ld_pct, pop_pct, iffb_pct per
        # game. The hits/walks/outs models lean heavily on batted-ball
        # mix — GB pitchers convert more BIPs to outs (double plays);
        # FB pitchers allow more HRs but fewer BABIP hits.
        ground_balls=("is_ground_ball", "sum"),
        fly_balls=("is_fly_ball", "sum"),
        line_drives=("is_line_drive", "sum"),
        popups=("is_popup", "sum"),
        infield_fly_balls=("is_infield_fly_ball", "sum"),
        # ── Contact-quality averages (H/W/outs pipeline additions) ──
        # Pandas .mean() with NaN-skip is the default — gives us the
        # average exit velocity, launch angle, xBA-on-contact, and
        # xwOBA-on-contact across the batted balls in this start. NaN
        # if zero BIPs (the downstream rate will also be NaN, which is
        # handled by the daily script's smart_feature_get fallback).
        avg_exit_velocity=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),
        # Stuff metrics
        avg_velocity=("release_speed", "mean"),
        max_velocity=("release_speed", "max"),
        avg_spin_rate=("release_spin_rate", "mean"),
        avg_extension=("release_extension", "mean"),
        avg_induced_vert_break=("pfx_z", "mean"),
        avg_horiz_break=("pfx_x", "mean"),
        # Pitch mix
        fastball_count=("is_fastball", "sum"),
        breaking_count=("is_breaking", "sum"),
        offspeed_count=("is_offspeed", "sum"),
        # Individual pitch types
        ff_count=("is_pt_FF", "sum"),
        si_count=("is_pt_SI", "sum"),
        fc_count=("is_pt_FC", "sum"),
        sl_count=("is_pt_SL", "sum"),
        cu_count=("is_pt_CU", "sum"),
        ch_count=("is_pt_CH", "sum"),
        fs_count=("is_pt_FS", "sum"),
        sv_count=("is_pt_SV", "sum"),
        kc_count=("is_pt_KC", "sum"),
        # Platoon
        pa_vs_left=("is_vs_left", "sum"),
        pa_vs_right=("is_vs_right", "sum"),
    ).reset_index()

    # Platoon whiff counts — compute separately to avoid lambda issues
    whiff_vs_left = df[df["is_vs_left"] == 1].groupby(group_keys)["is_whiff"].sum().reset_index()
    whiff_vs_left = whiff_vs_left.rename(columns={"is_whiff": "whiffs_vs_left"})
    whiff_vs_right = df[df["is_vs_right"] == 1].groupby(group_keys)["is_whiff"].sum().reset_index()
    whiff_vs_right = whiff_vs_right.rename(columns={"is_whiff": "whiffs_vs_right"})

    agg = agg.merge(whiff_vs_left, on=group_keys, how="left")
    agg = agg.merge(whiff_vs_right, on=group_keys, how="left")
    agg["whiffs_vs_left"] = agg["whiffs_vs_left"].fillna(0)
    agg["whiffs_vs_right"] = agg["whiffs_vs_right"].fillna(0)

    # ── xBA / xwOBA on contact (post-merge so we can gate on column presence) ──
    # estimated_ba_using_speedangle and estimated_woba_using_speedangle
    # are Statcast's "expected" stats based on exit velocity + launch
    # angle. NaN on non-BIP. We average over batted balls only.
    # These are the closest things to "what should have happened" given
    # the contact made — direct hit-suppression signal beyond barrel%.
    if "estimated_ba_using_speedangle" in df.columns:
        # Restrict to BIPs (rows where launch_speed is non-null) so the
        # mean isn't diluted by NaN-filled non-BIP rows.
        bip_only = df[df["launch_speed"].notna()] if "launch_speed" in df.columns else df
        if "estimated_ba_using_speedangle" in bip_only.columns:
            xba_avg = (bip_only.groupby(group_keys)
                       ["estimated_ba_using_speedangle"].mean()
                       .reset_index()
                       .rename(columns={"estimated_ba_using_speedangle": "avg_xba_contact"}))
            agg = agg.merge(xba_avg, on=group_keys, how="left")
    if "estimated_woba_using_speedangle" in df.columns:
        bip_only = df[df["launch_speed"].notna()] if "launch_speed" in df.columns else df
        if "estimated_woba_using_speedangle" in bip_only.columns:
            xwoba_avg = (bip_only.groupby(group_keys)
                         ["estimated_woba_using_speedangle"].mean()
                         .reset_index()
                         .rename(columns={"estimated_woba_using_speedangle": "avg_xwoba_contact"}))
            agg = agg.merge(xwoba_avg, on=group_keys, how="left")

    # ── Derived rates ──
    tp = agg["total_pitches"].replace(0, np.nan)
    pa = agg["plate_appearances"].replace(0, np.nan)
    bb = agg["batted_balls"].replace(0, np.nan)
    ooz = agg["out_of_zone_pitches"].replace(0, np.nan)

    agg["strike_pct"] = agg["strikes"] / tp
    agg["whiff_pct"] = agg["whiffs"] / tp
    agg["csw_pct"] = (agg["called_strikes"] + agg["whiffs"]) / tp
    agg["zone_pct"] = agg["in_zone_pitches"] / tp
    agg["chase_rate"] = agg["chases"] / ooz
    agg["k_pct"] = agg["strikeouts"] / pa
    agg["bb_pct"] = agg["walks"] / pa
    agg["k_bb_pct"] = agg["k_pct"] - agg["bb_pct"]
    agg["barrel_pct"] = agg["barrels"] / bb
    agg["hard_hit_pct"] = agg["hard_hits"] / bb
    agg["soft_hit_pct"] = agg["soft_hits"] / bb

    # ── Batted-ball-type and sweet-spot rates (H/W/outs pipeline additions) ──
    # All denominators use batted_balls (BIP count). NaN-safe via `bb`
    # already having 0s replaced with NaN.
    agg["gb_pct"] = agg["ground_balls"] / bb
    agg["fb_pct"] = agg["fly_balls"] / bb
    agg["ld_pct"] = agg["line_drives"] / bb
    agg["pop_pct"] = agg["popups"] / bb
    agg["iffb_pct"] = agg["infield_fly_balls"] / bb
    agg["sweet_spot_pct"] = agg["sweet_spot_hits"] / bb
    agg["solid_contact_pct"] = agg["solid_contact_hits"] / bb

    # ── Hits / walks / HR per-PA rates ──
    # These are also computed in features.py's rolling
    # block, but having them in the per-game data lets downstream
    # consumers (cumulative szn, prev10 carryover) see them as proper
    # input columns instead of needing rolling-loop derivation.
    agg["hits_per_pa"] = agg["hits_allowed"] / pa
    agg["bb_per_pa"] = agg["walks"] / pa
    agg["hr_per_pa"] = agg["home_runs_allowed"] / pa

    # Pitch mix percentages
    agg["fastball_pct"] = agg["fastball_count"] / tp
    agg["breaking_pct"] = agg["breaking_count"] / tp
    agg["offspeed_pct"] = agg["offspeed_count"] / tp

    for pt in ["ff", "si", "fc", "sl", "cu", "ch", "fs", "sv", "kc"]:
        agg[f"{pt}_pct"] = agg[f"{pt}_count"] / tp

    # Platoon rates
    pitches_vs_left = agg["pa_vs_left"].replace(0, np.nan)
    pitches_vs_right = agg["pa_vs_right"].replace(0, np.nan)
    agg["whiff_pct_vs_left"] = agg["whiffs_vs_left"] / pitches_vs_left
    agg["whiff_pct_vs_right"] = agg["whiffs_vs_right"] / pitches_vs_right

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    return agg


def aggregate_pitcher_pitch_type(statcast_df):
    """
    Aggregate pitch-level data to pitcher × game × pitch_type level.
    Gives per-pitch-type stats like slider whiff rate, fastball velocity,
    changeup chase rate per game for each pitcher.
    """
    df = statcast_df.copy()
    df = df[df["pitcher"].notna() & df["pitch_type"].notna()].copy()

    # Map to broad categories + keep individual types
    type_map = {
        "FF": "FB", "SI": "FB", "FC": "FB",
        "SL": "BRK", "CU": "BRK", "KC": "BRK", "CS": "BRK", "SV": "BRK", "SC": "BRK",
        "CH": "OS", "FS": "OS", "FO": "OS", "KN": "OS", "EP": "OS",
    }
    df["pitch_category"] = df["pitch_type"].map(type_map).fillna("OTHER")

    df["is_whiff"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ]).astype(int)
    df["is_strike"] = df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt",
    ]).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["is_batted_ball"] = (df["description"] == "hit_into_play").astype(int)
    df["is_hit"] = df["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    if "zone" in df.columns:
        df["is_out_of_zone"] = (~df["zone"].between(1, 9)).fillna(False).astype(int)
        df["is_chase"] = (df["is_out_of_zone"] & df["is_whiff"]).astype(int)
    else:
        df["is_chase"] = 0
        df["is_out_of_zone"] = 0

    if "launch_speed" in df.columns:
        df["is_hard_hit"] = (df["launch_speed"] >= 95).fillna(False).astype(int)
    else:
        df["is_hard_hit"] = 0

    group_keys = ["pitcher", "game_pk", "game_date", "pitch_type"]
    grouped = df.groupby(group_keys)

    agg = grouped.agg(
        pt_pitches=("pitch_type", "count"),
        pt_whiffs=("is_whiff", "sum"),
        pt_strikes=("is_strike", "sum"),
        pt_called_strikes=("is_called_strike", "sum"),
        pt_chases=("is_chase", "sum"),
        pt_ooz=("is_out_of_zone", "sum"),
        pt_batted_balls=("is_batted_ball", "sum"),
        pt_hard_hits=("is_hard_hit", "sum"),
        pt_hits=("is_hit", "sum"),
        pt_avg_velo=("release_speed", "mean"),
        pt_avg_spin=("release_spin_rate", "mean"),
        pt_avg_vert_break=("pfx_z", "mean"),
        pt_avg_horiz_break=("pfx_x", "mean"),
    ).reset_index()

    # Derived rates
    tp = agg["pt_pitches"].replace(0, np.nan)
    ooz = agg["pt_ooz"].replace(0, np.nan)
    bb = agg["pt_batted_balls"].replace(0, np.nan)

    agg["pt_whiff_rate"] = agg["pt_whiffs"] / tp
    agg["pt_strike_rate"] = agg["pt_strikes"] / tp
    agg["pt_csw_rate"] = (agg["pt_called_strikes"] + agg["pt_whiffs"]) / tp
    agg["pt_chase_rate"] = agg["pt_chases"] / ooz
    agg["pt_hard_hit_rate"] = agg["pt_hard_hits"] / bb
    agg["pt_hit_rate"] = agg["pt_hits"] / bb

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    return agg


def aggregate_batter_pitch_type(statcast_df):
    """
    Aggregate pitch-level data to batter × game × pitch_type level.
    Gives per-pitch-type vulnerability: how each batter performs against
    fastballs, sliders, changeups, etc.
    """
    df = statcast_df.copy()
    df = df[df["batter"].notna() & df["pitch_type"].notna()].copy()

    df["is_whiff"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ]).astype(int)
    df["is_strike"] = df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt",
    ]).astype(int)
    df["is_strikeout"] = df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    df["is_hit"] = df["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    df["is_plate_appearance"] = df["events"].notna().astype(int)
    df["is_batted_ball"] = (df["description"] == "hit_into_play").astype(int)

    if "launch_speed" in df.columns:
        df["is_hard_hit"] = (df["launch_speed"] >= 95).fillna(False).astype(int)
    else:
        df["is_hard_hit"] = 0

    # Aggregate per batter × pitch type (across ALL games, for cumulative lookups)
    # We do per-game so we can build rolling features later
    group_keys = ["batter", "game_pk", "game_date", "pitch_type"]
    grouped = df.groupby(group_keys)

    agg = grouped.agg(
        bpt_pitches_seen=("pitch_type", "count"),
        bpt_whiffs=("is_whiff", "sum"),
        bpt_strikes=("is_strike", "sum"),
        bpt_strikeouts=("is_strikeout", "sum"),
        bpt_hits=("is_hit", "sum"),
        bpt_pa=("is_plate_appearance", "sum"),
        bpt_batted_balls=("is_batted_ball", "sum"),
        bpt_hard_hits=("is_hard_hit", "sum"),
    ).reset_index()

    # Derived rates per game
    tp = agg["bpt_pitches_seen"].replace(0, np.nan)
    bb = agg["bpt_batted_balls"].replace(0, np.nan)
    pa = agg["bpt_pa"].replace(0, np.nan)

    agg["bpt_whiff_rate"] = agg["bpt_whiffs"] / tp
    agg["bpt_k_rate"] = agg["bpt_strikeouts"] / pa
    agg["bpt_hit_rate"] = agg["bpt_hits"] / bb
    agg["bpt_hard_hit_rate"] = agg["bpt_hard_hits"] / bb

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    return agg


# ── Step 1: Statcast Data ───────────────────────────────────────────────────
def collect_statcast_data():
    """Pull Statcast data season by season, aggregate to multiple levels."""
    print("\n═══ Step 1: Collecting Statcast Pitch Data ═══")
    all_games = []
    all_pitcher_pt = []
    all_batter_pt = []

    # Compute which seasons are already represented in the PA events file.
    # The bug before was treating "file exists" as "all years present" —
    # so after 2021 wrote rows, seasons 2022+ got skipped because the
    # cache-hit early exit fired. Now we check per-year coverage.
    _pa_years_covered = set()
    if PA_EVENTS_PATH.exists():
        try:
            _existing_dates = pd.read_csv(
                PA_EVENTS_PATH, usecols=["game_date"]
            )["game_date"]
            _existing_dates = pd.to_datetime(_existing_dates, errors="coerce")
            _pa_years_covered = set(_existing_dates.dt.year.dropna().astype(int).unique())
            print(f"  PA events file already covers years: "
                  f"{sorted(_pa_years_covered) if _pa_years_covered else '(none)'}")
        except Exception as _e:
            print(f"  ⚠ Could not read PA events file: {_e}")

    # Same per-year coverage check for the pitch-level file. This is the
    # raw-pitch cache used by catcher game-calling, pitch sequencing, and
    # any other features that need pitch-level data with count + pitch_type
    # + fielder_2. If a season isn't here, we re-pull even if pitcher-game
    # aggregations are cached — pulling once and saving is far cheaper than
    # discovering later that the data isn't recoverable.
    _pitches_years_covered = set()
    if PITCHES_PATH.exists():
        try:
            _existing_dates = pd.read_csv(
                PITCHES_PATH, usecols=["game_date"]
            )["game_date"]
            _existing_dates = pd.to_datetime(_existing_dates, errors="coerce")
            _pitches_years_covered = set(_existing_dates.dt.year.dropna().astype(int).unique())
            print(f"  Pitches file already covers years: "
                  f"{sorted(_pitches_years_covered) if _pitches_years_covered else '(none)'}")
        except Exception as _e:
            print(f"  ⚠ Could not read pitches file: {_e}")

    for year in SEASONS:
        start, end = SEASON_DATES[year]
        cache_file = OUTPUT_DIR / f"statcast_pitcher_games_{year}.csv"
        pt_pitcher_cache = OUTPUT_DIR / f"pitcher_pitch_type_{year}.csv"
        pt_batter_cache = OUTPUT_DIR / f"batter_pitch_type_{year}.csv"

        _all_caches_ok = (cache_file.exists() and pt_pitcher_cache.exists()
                          and pt_batter_cache.exists())
        _year_in_pa = year in _pa_years_covered
        _year_in_pitches = year in _pitches_years_covered

        # All three of these need to be true to skip the pull. If any are
        # missing, we re-pull the raw Statcast and write the missing ones.
        # The aggregations themselves are cheap; the bottleneck is the
        # Statcast API calls inside pull_statcast_chunked.
        if _all_caches_ok and _year_in_pa and _year_in_pitches:
            print(f"  Season {year}: loading from cache...")
            all_games.append(pd.read_csv(cache_file, parse_dates=["game_date"]))
            all_pitcher_pt.append(pd.read_csv(pt_pitcher_cache, parse_dates=["game_date"]))
            all_batter_pt.append(pd.read_csv(pt_batter_cache, parse_dates=["game_date"]))
            continue

        # Diagnose what's forcing the re-pull so the user can see why
        _missing_for_year = []
        if not _all_caches_ok:
            _missing_for_year.append("aggregation caches")
        if not _year_in_pa:
            _missing_for_year.append("PA events")
        if not _year_in_pitches:
            _missing_for_year.append("pitch-level cache")
        if _missing_for_year:
            print(f"  Season {year}: re-pulling raw to populate "
                  f"{', '.join(_missing_for_year)}...")

        print(f"  Season {year}: pulling pitch data...")
        raw = pull_statcast_chunked(start, end)
        if len(raw) == 0:
            print(f"  ⚠ No data for {year}")
            continue

        # Aggregation 1: pitcher-game level (existing)
        if not cache_file.exists():
            print(f"  Aggregating {len(raw):,} pitches to pitcher-game level...")
            game_level = aggregate_pitcher_game(raw)
            game_level["season"] = year
            game_level.to_csv(cache_file, index=False)
            all_games.append(game_level)
            print(f"  ✓ {len(game_level):,} pitcher-game rows")
        else:
            all_games.append(pd.read_csv(cache_file, parse_dates=["game_date"]))

        # Per-PA event save (for train/strikeouts_per_pa.py). Runs whenever we
        # pulled `raw` above. Idempotence: only skip if this specific
        # year is already in the PA file. To force a clean rebuild,
        # delete statcast_pa_events_all.csv before running.
        if not _year_in_pa:
            n_pa = save_pa_events_chunk(raw)
            if n_pa:
                print(f"  ✓ {n_pa:,} PA-terminating rows appended to "
                      f"statcast_pa_events_all.csv")
                _pa_years_covered.add(year)

        # Pitch-level save (for catcher game-calling, sequencing, framing).
        # Same idempotence: if the year is already in the pitches file, skip.
        # To force a rebuild, delete statcast_pitches_all.csv. This is the
        # largest output file (~250MB per season as CSV), so we don't want
        # to write it twice.
        if not _year_in_pitches:
            n_pitches = save_pitches_chunk(raw)
            if n_pitches:
                print(f"  ✓ {n_pitches:,} pitch-level rows appended to "
                      f"statcast_pitches_all.csv")
                _pitches_years_covered.add(year)

        # Aggregation 2: pitcher × pitch type per game
        if not pt_pitcher_cache.exists():
            print(f"  Aggregating to pitcher × pitch type level...")
            pitcher_pt = aggregate_pitcher_pitch_type(raw)
            pitcher_pt["season"] = year
            pitcher_pt.to_csv(pt_pitcher_cache, index=False)
            all_pitcher_pt.append(pitcher_pt)
            print(f"  ✓ {len(pitcher_pt):,} pitcher-pitch-type-game rows")
        else:
            all_pitcher_pt.append(pd.read_csv(pt_pitcher_cache, parse_dates=["game_date"]))

        # Aggregation 3: batter × pitch type per game
        if not pt_batter_cache.exists():
            print(f"  Aggregating to batter × pitch type level...")
            batter_pt = aggregate_batter_pitch_type(raw)
            batter_pt["season"] = year
            batter_pt.to_csv(pt_batter_cache, index=False)
            all_batter_pt.append(batter_pt)
            print(f"  ✓ {len(batter_pt):,} batter-pitch-type-game rows")
        else:
            all_batter_pt.append(pd.read_csv(pt_batter_cache, parse_dates=["game_date"]))

    # Save combined files
    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "statcast_pitcher_games_all.csv", index=False)
        print(f"\n  ✓ Total pitcher-game: {len(combined):,} rows")

    if all_pitcher_pt:
        combined_pt = pd.concat(all_pitcher_pt, ignore_index=True)
        combined_pt.to_csv(OUTPUT_DIR / "pitcher_pitch_type_all.csv", index=False)
        print(f"  ✓ Total pitcher-pitch-type: {len(combined_pt):,} rows")

    if all_batter_pt:
        combined_bpt = pd.concat(all_batter_pt, ignore_index=True)
        combined_bpt.to_csv(OUTPUT_DIR / "batter_pitch_type_all.csv", index=False)
        print(f"  ✓ Total batter-pitch-type: {len(combined_bpt):,} rows")

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()


# ── Step 2: FanGraphs Stats — handled by collect/fangraphs.py ───────────
# FanGraphs is behind Cloudflare and blocks pybaseball's automated requests
# with HTTP 403. Manual CSV exports go through collect/fangraphs.py.
# The canonical files (data/fangraphs_pitching_seasons.csv,
# data/fangraphs_batting_seasons.csv) are read directly by
# features.py and downstream scripts — they don't need
# anything from collect/statcast.py for FG.


# ── Step 3: Opposing Team Batting Aggregates ─────────────────────────────────
def collect_opposing_team_stats(statcast_games_df):
    """
    Build game-by-game opposing team batting profiles.
    For each game, compute the opposing team's rolling batting tendencies
    using the Statcast data we already have (what their batters have done
    against all pitchers this season so far).
    """
    print("\n═══ Step 3: Building Opposing Team Batting Profiles ═══")

    # We'll re-derive team batting from the pitcher-game data.
    # Each pitcher-game row tells us how Team X's batters performed
    # against that pitcher. We flip the perspective.
    df = statcast_games_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Determine opposing team: the pitcher faces the other team
    # We need pitcher->team mapping. We'll infer this:
    # If pitcher threw at home, they face the away team, and vice versa.
    # Without explicit mapping yet, we'll build team stats from the data
    # as a whole — each row's stats represent what the opposing lineup did.

    # Build a lookup: for each (team, date), what are their cumulative
    # batting stats up to that date? We aggregate across all pitcher
    # appearances that team's batters had.

    # For this, we need to know which team was batting.
    # In our aggregated data, the pitcher faced the "other" team.
    # We'll handle this properly in feature engineering with the
    # schedule data below.

    print("  ℹ Team batting profiles will be built in feature engineering")
    print("    using schedule data to identify opposing lineups")


# ── Step 4: MLB API Game Metadata (Umpires, Venues, Lineups) ────────────────
def collect_game_metadata():
    """
    Pull game-level metadata from the MLB Stats API:
    - Home plate umpire
    - Venue / ballpark
    - Starting lineups (for platoon features)
    - Weather (when available)
    """
    print("\n═══ Step 4: Collecting Game Metadata (MLB Stats API) ═══")

    all_metadata = []

    for year in SEASONS:
        cache_file = OUTPUT_DIR / f"game_metadata_{year}.csv"
        if cache_file.exists():
            print(f"  Season {year}: loading from cache...")
            df = pd.read_csv(cache_file)
            all_metadata.append(df)
            continue

        start_str, end_str = SEASON_DATES[year]
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        current = start
        season_games = []

        print(f"  Season {year}: pulling game metadata...")
        while current <= end:
            date_str = str(current.date())
            games = get_schedule_for_date(date_str)

            for game in games:
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue

                game_pk = game["gamePk"]
                meta = {
                    "game_pk": game_pk,
                    "game_date": date_str,
                    "season": year,
                    "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                    "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                    "home_team_name": game.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                    "away_team_name": game.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                    "venue_id": game.get("venue", {}).get("id"),
                    "venue_name": game.get("venue", {}).get("name"),
                    "day_night": game.get("dayNight"),
                    "home_score": game.get("teams", {}).get("home", {}).get("score"),
                    "away_score": game.get("teams", {}).get("away", {}).get("score"),
                }

                # Probable pitchers
                home_pitcher = game.get("teams", {}).get("home", {}).get("probablePitcher", {})
                away_pitcher = game.get("teams", {}).get("away", {}).get("probablePitcher", {})
                meta["home_starter_id"] = home_pitcher.get("id")
                meta["home_starter_name"] = home_pitcher.get("fullName")
                meta["away_starter_id"] = away_pitcher.get("id")
                meta["away_starter_name"] = away_pitcher.get("fullName")

                season_games.append(meta)

            current += pd.Timedelta(days=1)
            time.sleep(0.15)  # Gentle rate limiting

        if season_games:
            season_df = pd.DataFrame(season_games)
            season_df.to_csv(cache_file, index=False)
            all_metadata.append(season_df)
            print(f"  ✓ {len(season_df)} games for {year}")

    if all_metadata:
        combined = pd.concat(all_metadata, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "game_metadata_all.csv", index=False)
        print(f"\n  ✓ Total: {len(combined):,} games with metadata")
        return combined
    return pd.DataFrame()


def collect_lineup_data():
    """
    Pull starting lineups for every game via MLB API boxscores.
    Returns each batter's lineup position, handedness, and MLB player ID
    so we can build per-lineup K-rate features in feature engineering.

    This reuses the same boxscore endpoint as umpires, so we collect
    both in one pass if umpires haven't been cached yet. If you already
    have umpire data but not lineups, this runs independently.
    """
    print("\n═══ Step 5a: Collecting Starting Lineups ═══")

    lineup_cache = OUTPUT_DIR / "game_lineups.csv"
    if lineup_cache.exists():
        print("  Loading from cache...")
        return pd.read_csv(lineup_cache)

    meta_path = OUTPUT_DIR / "game_metadata_all.csv"
    if not meta_path.exists():
        print("  ⚠ No game metadata found. Run step 4 first.")
        return pd.DataFrame()

    meta = pd.read_csv(meta_path)
    game_pks = meta["game_pk"].unique()
    lineup_records = []

    print(f"  Pulling lineups for {len(game_pks):,} games...")
    print("  (This may take a while — ~1 API call per game)")

    for i, gpk in enumerate(game_pks):
        if i % 500 == 0 and i > 0:
            print(f"    Progress: {i}/{len(game_pks)}")

        try:
            data = mlb_api_get(f"game/{gpk}/boxscore")
            if not data:
                continue

            for side in ["home", "away"]:
                team_data = data.get("teams", {}).get(side, {})
                team_info = team_data.get("team", {})
                batting_order = team_data.get("battingOrder", [])
                players_dict = team_data.get("players", {})

                for order_idx, player_id in enumerate(batting_order):
                    player_key = f"ID{player_id}"
                    player_info = players_dict.get(player_key, {})
                    person = player_info.get("person", {})
                    stats = player_info.get("stats", {})
                    batting_stats_game = stats.get("batting", {})
                    bat_side = player_info.get("batSide", {}).get("code", "R")

                    lineup_records.append({
                        "game_pk": gpk,
                        "side": side,
                        "lineup_position": order_idx + 1,  # 1-9
                        "player_id": player_id,
                        "player_name": person.get("fullName", ""),
                        "bat_side": bat_side,
                        "at_bats": batting_stats_game.get("atBats", 0),
                        "strikeouts": batting_stats_game.get("strikeOuts", 0),
                        "hits": batting_stats_game.get("hits", 0),
                        "walks": batting_stats_game.get("baseOnBalls", 0),
                    })

            time.sleep(0.1)
        except Exception:
            pass

    if lineup_records:
        lineup_df = pd.DataFrame(lineup_records)
        lineup_df.to_csv(lineup_cache, index=False)
        print(f"  ✓ {len(lineup_df):,} lineup entries collected")
        print(f"    ({lineup_df['game_pk'].nunique():,} games)")
        return lineup_df
    return pd.DataFrame()


def collect_umpire_data():
    """
    Pull home plate umpire assignments via MLB API boxscores.
    This is slower (one API call per game), so we only do it for games
    in our dataset.
    """
    print("\n═══ Step 5b: Collecting Umpire Assignments ═══")

    meta_path = OUTPUT_DIR / "game_metadata_all.csv"
    if not meta_path.exists():
        print("  ⚠ No game metadata found. Run step 4 first.")
        return

    umpire_cache = OUTPUT_DIR / "umpire_assignments.csv"
    if umpire_cache.exists():
        print("  Loading from cache...")
        return pd.read_csv(umpire_cache)

    meta = pd.read_csv(meta_path)
    game_pks = meta["game_pk"].unique()
    umpire_records = []

    print(f"  Pulling umpire data for {len(game_pks):,} games...")
    print("  (This may take a while — ~1 API call per game)")

    for i, gpk in enumerate(game_pks):
        if i % 500 == 0 and i > 0:
            print(f"    Progress: {i}/{len(game_pks)}")

        try:
            data = mlb_api_get(f"game/{gpk}/boxscore")
            if data and "officials" in data:
                for official in data["officials"]:
                    if official.get("officialType") == "Home Plate":
                        umpire_records.append({
                            "game_pk": gpk,
                            "hp_umpire_id": official.get("official", {}).get("id"),
                            "hp_umpire_name": official.get("official", {}).get("fullName"),
                        })
                        break
            time.sleep(0.1)
        except Exception:
            pass

    if umpire_records:
        ump_df = pd.DataFrame(umpire_records)
        ump_df.to_csv(umpire_cache, index=False)
        print(f"  ✓ {len(ump_df)} umpire assignments collected")
        return ump_df
    return pd.DataFrame()


# ── Step 5c: Starting Catcher per Game ──────────────────────────────────────
def collect_catcher_data():
    """
    Identify the starting catcher for each team in each game.
    Uses the boxscore API — the catcher is typically the player at
    fielding position 2 (C) in the fieldingOrder or the first player
    with position abbreviation 'C'.
    """
    print("\n═══ Step 5c: Collecting Starting Catchers ═══")

    cache_file = OUTPUT_DIR / "game_catchers.csv"
    if cache_file.exists():
        print("  Loading from cache...")
        return pd.read_csv(cache_file)

    meta_path = OUTPUT_DIR / "game_metadata_all.csv"
    if not meta_path.exists():
        print("  ⚠ No game metadata found. Run step 4 first.")
        return pd.DataFrame()

    meta = pd.read_csv(meta_path)
    game_pks = meta["game_pk"].unique()
    catcher_records = []

    print(f"  Pulling catcher data for {len(game_pks):,} games...")

    for i, gpk in enumerate(game_pks):
        if i % 500 == 0 and i > 0:
            print(f"    Progress: {i}/{len(game_pks)}")

        try:
            data = mlb_api_get(f"game/{gpk}/boxscore")
            if not data:
                continue

            for side in ["home", "away"]:
                team_data = data.get("teams", {}).get(side, {})
                players_dict = team_data.get("players", {})

                # Find the catcher: look for fielding position "C" or position code "2"
                catcher_id = None
                catcher_name = None

                for pid_key, pinfo in players_dict.items():
                    pos = pinfo.get("position", {})
                    if pos.get("abbreviation") == "C" or pos.get("code") == "2":
                        # Check this player actually played (has batting or fielding stats)
                        stats = pinfo.get("stats", {})
                        fielding = stats.get("fielding", {})
                        # Only take catchers who started (had innings at C)
                        if fielding.get("innings") or fielding.get("gamesStarted", 0) > 0:
                            catcher_id = pinfo.get("person", {}).get("id")
                            catcher_name = pinfo.get("person", {}).get("fullName", "")
                            break

                # Fallback: just use the position listed for the player
                if catcher_id is None:
                    for pid_key, pinfo in players_dict.items():
                        all_positions = pinfo.get("allPositions", [])
                        for apos in all_positions:
                            if apos.get("abbreviation") == "C":
                                catcher_id = pinfo.get("person", {}).get("id")
                                catcher_name = pinfo.get("person", {}).get("fullName", "")
                                break
                        if catcher_id:
                            break

                if catcher_id:
                    catcher_records.append({
                        "game_pk": gpk,
                        "side": side,
                        "catcher_id": catcher_id,
                        "catcher_name": catcher_name,
                    })

            time.sleep(0.1)
        except Exception:
            pass

    if catcher_records:
        catcher_df = pd.DataFrame(catcher_records)
        catcher_df.to_csv(cache_file, index=False)
        print(f"  ✓ {len(catcher_df):,} catcher assignments collected")
        print(f"    ({catcher_df['game_pk'].nunique():,} games)")
        return catcher_df
    return pd.DataFrame()


# ── Step 6: Ballpark Factors ────────────────────────────────────────────────
def collect_park_factors():
    """
    Collect ballpark factors from FanGraphs via pybaseball.
    Park factors tell us how much a venue inflates/deflates stats
    relative to league average (100 = neutral).
    """
    print("\n═══ Step 6: Collecting Ballpark Factors ═══")

    cache_file = OUTPUT_DIR / "park_factors.csv"
    if cache_file.exists():
        print("  Loading from cache...")
        return pd.read_csv(cache_file)

    all_pf = []
    for year in SEASONS:
        print(f"  Season {year}...")
        if park_factors is not None:
            try:
                pf = park_factors(year)
                if pf is not None and len(pf) > 0:
                    pf["Season"] = year
                    all_pf.append(pf)
                time.sleep(1)
                continue
            except Exception as e:
                print(f"  ⚠ Error: {e}")

        # Fallback: neutral park factors for all teams
        print("  Using fallback park factor estimates")
        fallback = pd.DataFrame({
            "Team": list(TEAM_ABBREVS.keys()),
            "Season": year,
            "Basic": 100,  # Neutral default
        })
        all_pf.append(fallback)

    if all_pf:
        combined = pd.concat(all_pf, ignore_index=True)
        combined.to_csv(cache_file, index=False)
        print(f"  ✓ Park factors saved ({len(combined)} rows)")
        return combined
    return pd.DataFrame()


# ── Step 7: Weather Data ────────────────────────────────────────────────────
def collect_weather_data():
    """
    Build a venue-based weather proxy using MLB API game data.
    The MLB API includes weather info (temp, wind, condition) in some
    game feeds. We'll also create venue-level climate defaults.

    For a more robust approach, you could use the Open-Meteo API
    (free, no key needed) to pull historical weather by venue lat/lon.
    """
    print("\n═══ Step 7: Collecting Weather/Venue Data ═══")

    # Venue coordinates for weather API lookups
    VENUE_COORDS = {
        "Angel Stadium": (33.8003, -117.8827),
        "Busch Stadium": (38.6226, -90.1928),
        "Chase Field": (33.4453, -112.0667),
        "Citi Field": (40.7571, -73.8458),
        "Citizens Bank Park": (39.9061, -75.1665),
        "Comerica Park": (42.3390, -83.0485),
        "Coors Field": (39.7559, -104.9942),
        "Dodger Stadium": (34.0739, -118.2400),
        "Fenway Park": (42.3467, -71.0972),
        "Globe Life Field": (32.7473, -97.0845),
        "Great American Ball Park": (39.0974, -84.5082),
        "Guaranteed Rate Field": (41.8299, -87.6338),
        "Kauffman Stadium": (39.0517, -94.4803),
        "LoanDepot park": (25.7781, -80.2196),
        "Minute Maid Park": (29.7573, -95.3555),
        "Nationals Park": (38.8730, -77.0074),
        "Oakland Coliseum": (37.7516, -122.2005),
        "Oracle Park": (37.7786, -122.3893),
        "Oriole Park at Camden Yards": (39.2838, -76.6218),
        "PNC Park": (40.4468, -80.0057),
        "Petco Park": (32.7076, -117.1570),
        "Progressive Field": (41.4962, -81.6852),
        "Rogers Centre": (43.6414, -79.3894),
        "T-Mobile Park": (47.5914, -122.3325),
        "Target Field": (44.9817, -93.2776),
        "Tropicana Field": (27.7682, -82.6534),  # Indoor
        "Truist Park": (33.8907, -84.4677),
        "Wrigley Field": (41.9484, -87.6553),
        "Yankee Stadium": (40.8296, -73.9262),
        "American Family Field": (43.0280, -87.9712),  # Retractable roof
    }

    # Dome/retractable roof venues (weather less relevant)
    INDOOR_VENUES = {
        "Tropicana Field", "Globe Life Field", "Chase Field",
        "LoanDepot park", "Minute Maid Park", "Rogers Centre",
        "American Family Field", "T-Mobile Park",
    }

    venue_df = pd.DataFrame([
        {
            "venue_name": name,
            "latitude": coords[0],
            "longitude": coords[1],
            "is_dome_or_retractable": name in INDOOR_VENUES,
        }
        for name, coords in VENUE_COORDS.items()
    ])

    venue_df.to_csv(OUTPUT_DIR / "venue_info.csv", index=False)
    print(f"  ✓ Venue coordinates saved ({len(venue_df)} venues)")
    print("  ℹ For historical weather, use Open-Meteo API with these coordinates:")
    print("    https://archive-api.open-meteo.com/v1/archive")
    print("    Parameters: latitude, longitude, start_date, end_date,")
    print("    hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")

    return venue_df


# ── Main ─────────────────────────────────────────────────────────────────────
def collect_earned_runs(game_pks=None, verbose=False, output_path=None):
    """
    Pull earned runs / runs / innings pitched per (pitcher × game) via the
    MLB Stats API boxscore endpoint.

    These per-game ER values feed lob_pct in features.py
    (LOB% requires runs allowed to compute). Without them, the season-
    level lob_pct_szn and downstream lob_pct_szn_blended can't be built.

    Schema written to data/pitcher_earned_runs.csv:
        game_pk, pitcher, earned_runs, runs, innings_pitched, outs

    Args:
        game_pks: Optional list of specific game_pks to fetch. If None,
            uses game_metadata_all.csv to enumerate all known games.
        verbose: Print per-batch progress.
        output_path: Optional override for the output CSV path.

    Returns:
        DataFrame of pitcher-game ER rows. Also writes/appends to disk.

    Idempotent: skips game_pks already present in the cache file.
    """
    out_path = output_path or (OUTPUT_DIR / "pitcher_earned_runs.csv")

    # Determine which game_pks we need
    if game_pks is None:
        meta_path = OUTPUT_DIR / "game_metadata_all.csv"
        if not meta_path.exists():
            print("  ⚠ No game_metadata_all.csv — run collect_game_metadata first.")
            return pd.DataFrame()
        game_pks = pd.read_csv(meta_path)["game_pk"].unique().tolist()

    # Dedup against existing cache
    existing_pks = set()
    if out_path.exists():
        try:
            existing_pks = set(pd.read_csv(out_path, usecols=["game_pk"])["game_pk"].unique())
        except Exception:
            existing_pks = set()

    pks_to_fetch = [pk for pk in game_pks if pk not in existing_pks]
    if verbose:
        print(f"  ER collection: {len(existing_pks)} cached, "
              f"{len(pks_to_fetch)} new to fetch")
    if not pks_to_fetch:
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    er_records = []
    for i, gpk in enumerate(pks_to_fetch):
        if verbose and i > 0 and i % 250 == 0:
            print(f"    Progress: {i}/{len(pks_to_fetch)}")
        try:
            data = mlb_api_get(f"game/{gpk}/boxscore")
            if not data:
                continue
            for side in ["home", "away"]:
                players = data.get("teams", {}).get(side, {}).get("players", {})
                for player_key, pinfo in players.items():
                    stats = pinfo.get("stats", {}).get("pitching", {})
                    if not stats:
                        continue
                    # Only keep pitchers who actually pitched (outs > 0)
                    outs = stats.get("outs", 0)
                    if outs == 0:
                        continue
                    pid = pinfo.get("person", {}).get("id")
                    if pid is None:
                        continue
                    ip_str = stats.get("inningsPitched", "0")
                    # IP is a string like "5.2" (5 innings, 2 outs). Convert.
                    try:
                        whole, frac = (ip_str.split(".") + ["0"])[:2]
                        ip_decimal = int(whole) + int(frac) / 3.0
                    except (ValueError, AttributeError):
                        ip_decimal = 0.0
                    er_records.append({
                        "game_pk": gpk,
                        "pitcher": pid,
                        "earned_runs":     int(stats.get("earnedRuns", 0) or 0),
                        "runs":            int(stats.get("runs", 0) or 0),
                        "innings_pitched": float(ip_decimal),
                        "outs":            int(outs),
                    })
            time.sleep(0.1)  # gentle rate limiting
        except Exception as e:
            if verbose:
                print(f"      ⚠ Failed game_pk={gpk}: {e}")

    if not er_records:
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    new_df = pd.DataFrame(er_records)
    header = not out_path.exists()
    new_df.to_csv(out_path, mode="a", header=header, index=False)
    if verbose:
        print(f"  ✓ Wrote {len(new_df)} ER rows to {out_path}")
    return new_df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MLB Pitcher Data Collection Pipeline (Enhanced)")
    print("=" * 60)
    print(f"Seasons: {SEASONS}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Step 1: Statcast pitch data
    sc_games = collect_statcast_data()

    # Step 2: FanGraphs season stats — handled separately by collect/fangraphs.py
    # (FanGraphs is behind Cloudflare; pybaseball's scraper returns 403)

    # Step 3: Opposing team context (built from step 1 data)
    if sc_games is not None and len(sc_games) > 0:
        collect_opposing_team_stats(sc_games)

    # Step 4: Game metadata (schedules, venues, probable pitchers)
    game_meta = collect_game_metadata()

    # Step 4b: Per-pitcher earned runs (one row per pitcher × game)
    # Required for lob_pct_szn and downstream blended features in 02.
    # Runs after game_meta so we have a known list of finished games.
    if game_meta is not None and len(game_meta) > 0:
        print("\n═══ Step 4b: Collecting Per-Pitcher Earned Runs ═══")
        collect_earned_runs(
            game_pks=game_meta["game_pk"].unique().tolist(),
            verbose=True,
        )

    # Step 5a: Starting lineups
    lineup_data = collect_lineup_data()

    # Step 5b: Umpire assignments
    ump_data = collect_umpire_data()

    # Step 5c: Starting catcher per game
    catcher_data = collect_catcher_data()

    # Step 6: Ballpark factors
    park_data = collect_park_factors()

    # Step 7: Weather / venue data
    venue_data = collect_weather_data()

    print("\n" + "=" * 60)
    print("═══ Collection Complete ═══")
    print(f"\nFiles in {OUTPUT_DIR.resolve()}:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:45s} ({size_mb:.1f} MB)")
    print("\nNext step: Run features.py")
