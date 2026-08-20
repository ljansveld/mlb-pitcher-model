"""
refresh.py
===================
Single-command daily refresh for the MLB pitcher model.

Instead of running collect/statcast.py → collect/weather.py →
features.py separately, this script handles everything
end-to-end for recent games:

  1. Pulls recent Statcast pitch data (default: last 3 days)
  2. Aggregates to pitcher-game, pitcher×pitch-type, batter×pitch-type
  3. Pulls game metadata from MLB Stats API (venues, starters)
  4. Pulls starting lineups + umpire assignments via boxscore API
  5. Fetches historical weather from Open-Meteo for each game
  6. Appends all new data to the existing cached CSV files
  7. Re-runs features.py to rebuild rolling averages
     (-> pitcher_model_features.csv, the base table for every model)

The historical data (2021-2025) is never re-pulled — it's already cached.
Only new 2026 games get added incrementally.

NOT HANDLED HERE: FanGraphs season stats. FanGraphs sits behind Cloudflare
and blocks automated requests, so those exports are manual — see
collect/fangraphs.py, which prints the exact URLs to download. Run it
BEFORE this script if the FanGraphs data is stale, since 02 reads it.

USAGE:
    python run.py refresh              # Pull last 3 days + rebuild features
    python run.py refresh --days 7     # Pull last 7 days
    python run.py refresh --skip-fe    # Pull/cache data but skip 02
    python run.py refresh --force      # Re-pull even if data looks fresh

    Then run any of the daily prediction scripts:
    python run.py predict strikeouts    # strikeouts
    python run.py predict hits-walks        # hits + walks
    python run.py predict outs              # outs recorded
    python run.py predict earned-runs                # earned runs

REQUIRES:
    pip install pybaseball pandas numpy requests
    Existing data/ folder with cached historical seasons from collect/statcast.py
"""

import sys
import argparse
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pitcher_model.paths import ROOT, DATA_DIR

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
CURRENT_SEASON = 2026
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Dome/retractable roof venues — weather doesn't matter here
INDOOR_VENUES = {
    "Tropicana Field", "Globe Life Field", "Chase Field",
    "LoanDepot park", "loanDepot park", "Minute Maid Park",
    "Rogers Centre", "American Family Field", "T-Mobile Park",
}

# Venue coordinates for Open-Meteo weather lookups
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
    "loanDepot park": (25.7781, -80.2196),
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
    "Tropicana Field": (27.7682, -82.6534),
    "Truist Park": (33.8907, -84.4677),
    "Wrigley Field": (41.9484, -87.6553),
    "Yankee Stadium": (40.8296, -73.9262),
    "American Family Field": (43.0280, -87.9712),
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: STATCAST DATA
# ══════════════════════════════════════════════════════════════════════════════

def pull_recent_statcast(start_date, end_date):
    """Pull recent Statcast pitch-level data via pybaseball."""
    from pybaseball import statcast

    print(f"    Pulling Statcast: {start_date} → {end_date}...")
    try:
        raw = statcast(start_dt=start_date, end_dt=end_date)
        if raw is not None and len(raw) > 0:
            print(f"    ✓ {len(raw):,} pitches")
            return raw
        else:
            print(f"    No pitches found for this date range")
            return pd.DataFrame()
    except Exception as e:
        print(f"    ✗ Error pulling Statcast: {e}")
        return pd.DataFrame()


def aggregate_pitcher_game(df):
    """
    Aggregate pitch-level data to pitcher-game level.
    Delegates to collect.statcast to guarantee identical aggregation.
    Falls back to the built-in version if that import fails.
    """
    try:
        from pitcher_model.collect import statcast
        print("    Using aggregate_pitcher_game from collect.statcast")
        result = statcast.aggregate_pitcher_game(df)
        result["season"] = CURRENT_SEASON
        return result
    except Exception as e:
        print(f"    ⚠ Could not import from collect.statcast: {e}")

    # ── Built-in fallback (matches collect.statcast logic) ──────────────
    print("    Using built-in aggregation (fallback)")
    df = df[df["pitcher"].notna()].copy()

    df["is_strike"] = df["description"].isin([
        "called_strike", "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt",
    ]).astype(int)
    df["is_whiff"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ]).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["is_ball"] = df["description"].isin(["ball", "blocked_ball"]).astype(int)
    df["is_batted_ball"] = (df["description"] == "hit_into_play").astype(int)

    if "zone" in df.columns:
        df["is_in_zone"] = df["zone"].between(1, 9).fillna(False).astype(int)
        df["is_out_of_zone"] = (~df["zone"].between(1, 9)).fillna(False).astype(int)
    else:
        df["is_in_zone"] = 0
        df["is_out_of_zone"] = 0
    df["is_chase"] = (df["is_out_of_zone"] & df["is_whiff"]).astype(int)

    if "launch_speed" in df.columns:
        df["is_barrel"] = (
            (df["launch_speed"] >= 98) & df["launch_angle"].between(26, 30)
        ).fillna(False).astype(int)
        df["is_hard_hit"] = (df["launch_speed"] >= 95).fillna(False).astype(int)
        df["is_soft_hit"] = (df["launch_speed"] < 70).fillna(False).astype(int)
    else:
        df["is_barrel"] = 0
        df["is_hard_hit"] = 0
        df["is_soft_hit"] = 0

    df["is_strikeout"] = df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    df["is_walk"] = df["events"].isin(["walk"]).astype(int)
    df["is_hbp"] = df["events"].isin(["hit_by_pitch"]).astype(int)
    df["is_single"] = (df["events"] == "single").astype(int)
    df["is_double"] = (df["events"] == "double").astype(int)
    df["is_triple"] = (df["events"] == "triple").astype(int)
    df["is_home_run"] = (df["events"] == "home_run").astype(int)
    df["is_hit"] = df["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    df["is_plate_appearance"] = df["events"].notna().astype(int)

    # Pitch type flags
    pitch_types = {
        "fastball": ["FF", "SI", "FC"],
        "breaking": ["SL", "CU", "KC", "CS", "SV", "SC"],
        "offspeed": ["CH", "FS", "FO", "KN", "EP"],
    }
    for category, codes in pitch_types.items():
        df[f"is_{category}"] = df["pitch_type"].isin(codes).astype(int)
    for pt in ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "SV", "KC"]:
        df[f"is_pt_{pt}"] = (df["pitch_type"] == pt).astype(int)

    df["is_vs_left"] = (df["stand"] == "L").astype(int)
    df["is_vs_right"] = (df["stand"] == "R").astype(int)

    group_keys = ["pitcher", "game_pk", "game_date", "home_team", "away_team"]
    if "p_throws" in df.columns:
        group_keys.append("p_throws")

    grouped = df.groupby(group_keys)

    agg = grouped.agg(
        total_pitches=("pitch_type", "count"),
        plate_appearances=("is_plate_appearance", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        strikeouts=("is_strikeout", "sum"),
        walks=("is_walk", "sum"),
        hbp=("is_hbp", "sum"),
        hits_allowed=("is_hit", "sum"),
        singles=("is_single", "sum"),
        doubles=("is_double", "sum"),
        triples=("is_triple", "sum"),
        home_runs_allowed=("is_home_run", "sum"),
        strikes=("is_strike", "sum"),
        balls=("is_ball", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        in_zone_pitches=("is_in_zone", "sum"),
        out_of_zone_pitches=("is_out_of_zone", "sum"),
        chases=("is_chase", "sum"),
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        soft_hits=("is_soft_hit", "sum"),
        avg_velocity=("release_speed", "mean"),
        max_velocity=("release_speed", "max"),
        avg_spin_rate=("release_spin_rate", "mean"),
        avg_extension=("release_extension", "mean"),
        avg_induced_vert_break=("pfx_z", "mean"),
        avg_horiz_break=("pfx_x", "mean"),
        fastball_count=("is_fastball", "sum"),
        breaking_count=("is_breaking", "sum"),
        offspeed_count=("is_offspeed", "sum"),
        ff_count=("is_pt_FF", "sum"),
        si_count=("is_pt_SI", "sum"),
        fc_count=("is_pt_FC", "sum"),
        sl_count=("is_pt_SL", "sum"),
        cu_count=("is_pt_CU", "sum"),
        ch_count=("is_pt_CH", "sum"),
        fs_count=("is_pt_FS", "sum"),
        sv_count=("is_pt_SV", "sum"),
        kc_count=("is_pt_KC", "sum"),
        pa_vs_left=("is_vs_left", "sum"),
        pa_vs_right=("is_vs_right", "sum"),
    ).reset_index()

    # Platoon whiff counts
    for side_code, side_name in [("L", "left"), ("R", "right")]:
        side_df = df[df[f"is_vs_{side_name}"] == 1]
        if len(side_df) > 0:
            side_agg = side_df.groupby(group_keys)["is_whiff"].sum().reset_index()
            side_agg = side_agg.rename(columns={"is_whiff": f"whiffs_vs_{side_name}"})
            agg = agg.merge(side_agg, on=group_keys, how="left")
            agg[f"whiffs_vs_{side_name}"] = agg[f"whiffs_vs_{side_name}"].fillna(0)

    # Derived rates
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
    agg["fastball_pct"] = agg["fastball_count"] / tp
    agg["breaking_pct"] = agg["breaking_count"] / tp
    agg["offspeed_pct"] = agg["offspeed_count"] / tp
    for pt in ["ff", "si", "fc", "sl", "cu", "ch", "fs", "sv", "kc"]:
        agg[f"{pt}_pct"] = agg[f"{pt}_count"] / tp

    pitches_vs_left = agg["pa_vs_left"].replace(0, np.nan)
    pitches_vs_right = agg["pa_vs_right"].replace(0, np.nan)
    agg["whiff_pct_vs_left"] = agg.get("whiffs_vs_left", 0) / pitches_vs_left
    agg["whiff_pct_vs_right"] = agg.get("whiffs_vs_right", 0) / pitches_vs_right

    agg["game_date"] = pd.to_datetime(agg["game_date"])
    agg["season"] = CURRENT_SEASON
    return agg


def aggregate_pitch_types(df):
    """Aggregate to pitcher × game × pitch_type level."""
    try:
        from pitcher_model.collect import statcast
        result = statcast.aggregate_pitcher_pitch_type(df)
        result["season"] = CURRENT_SEASON
        return result
    except Exception:
        pass
    return pd.DataFrame()


def aggregate_batter_pitch_types(df):
    """Aggregate to batter × game × pitch_type level."""
    try:
        from pitcher_model.collect import statcast
        result = statcast.aggregate_batter_pitch_type(df)
        result["season"] = CURRENT_SEASON
        return result
    except Exception:
        pass
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: MLB API — GAME METADATA, LINEUPS, UMPIRES
# ══════════════════════════════════════════════════════════════════════════════

def mlb_api_get(endpoint, params=None):
    """Make a request to the MLB Stats API."""
    url = f"{MLB_API_BASE}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


def get_finished_games(start_date, end_date):
    """
    Get all finished games between start_date and end_date from MLB API.
    Returns list of game dicts with metadata, lineups, and umpire info.
    """
    all_games = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current <= end:
        date_str = str(current.date())
        data = mlb_api_get("schedule", {
            "date": date_str,
            "sportId": 1,
            "hydrate": "linescore,probablePitcher,venue,officialScorer,decisions",
        })

        if data and data.get("dates"):
            for game in data["dates"][0].get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue

                game_pk = game["gamePk"]
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                venue = game.get("venue", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})

                all_games.append({
                    "game_pk": game_pk,
                    "game_date": date_str,
                    "season": CURRENT_SEASON,
                    "home_team_id": home.get("team", {}).get("id"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "day_night": game.get("dayNight"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "home_starter_id": home_pitcher.get("id"),
                    "home_starter_name": home_pitcher.get("fullName"),
                    "away_starter_id": away_pitcher.get("id"),
                    "away_starter_name": away_pitcher.get("fullName"),
                })

        current += pd.Timedelta(days=1)
        time.sleep(0.15)

    return all_games


def fetch_boxscore_data(game_pks):
    """
    Pull lineups + umpires + starting catchers from boxscore API.
    Returns (lineup_records, umpire_records, catcher_records).
    Does one API call per game but batches efficiently.
    """
    lineup_records = []
    umpire_records = []
    catcher_records = []

    for i, gpk in enumerate(game_pks):
        if i > 0 and i % 25 == 0:
            print(f"      Progress: {i}/{len(game_pks)} games")

        try:
            data = mlb_api_get(f"game/{gpk}/boxscore")
            if not data:
                continue

            # Extract lineups + catchers (both walk the same players_dict)
            for side in ["home", "away"]:
                team_data = data.get("teams", {}).get(side, {})
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
                        "lineup_position": order_idx + 1,
                        "player_id": player_id,
                        "player_name": person.get("fullName", ""),
                        "bat_side": bat_side,
                        "at_bats": batting_stats_game.get("atBats", 0),
                        "strikeouts": batting_stats_game.get("strikeOuts", 0),
                        "hits": batting_stats_game.get("hits", 0),
                        "walks": batting_stats_game.get("baseOnBalls", 0),
                    })

                # Find the starting catcher for this side. Same logic as
                # collect/statcast.collect_catcher_data — first try players
                # whose primary position is C and who recorded innings/starts
                # at C; fall back to anyone with C in their allPositions.
                catcher_id = None
                catcher_name = None
                for pid_key, pinfo in players_dict.items():
                    pos = pinfo.get("position", {})
                    if pos.get("abbreviation") == "C" or pos.get("code") == "2":
                        fielding = pinfo.get("stats", {}).get("fielding", {})
                        if fielding.get("innings") or fielding.get("gamesStarted", 0) > 0:
                            catcher_id = pinfo.get("person", {}).get("id")
                            catcher_name = pinfo.get("person", {}).get("fullName", "")
                            break
                if catcher_id is None:
                    for pid_key, pinfo in players_dict.items():
                        for apos in pinfo.get("allPositions", []):
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

            # Extract umpire
            if "officials" in data:
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

    return lineup_records, umpire_records, catcher_records


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: WEATHER (OPEN-METEO HISTORICAL API)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather_for_games(game_metadata):
    """
    Fetch historical/recent weather for each game from Open-Meteo.
    Uses the archive API for past dates and forecast API for recent dates.
    Returns DataFrame with wx_ columns matching training data format.
    """
    weather_records = []

    for _, game in game_metadata.iterrows():
        game_pk = game["game_pk"]
        venue_name = game.get("venue_name", "")
        game_date = str(game["game_date"])[:10]

        # Skip domed stadiums
        if venue_name in INDOOR_VENUES:
            weather_records.append({
                "game_pk": game_pk,
                "venue_name": venue_name,
                "wx_temperature_f": 72.0,
                "wx_humidity_pct": 50.0,
                "wx_wind_speed_mph": 0.0,
                "wx_pressure_hpa": 1013.0,
                "wx_is_dome": 1,
                "wx_is_cold": 0,
                "wx_is_hot": 0,
                "wx_is_windy": 0,
                "wx_is_humid": 0,
                "wx_air_density_proxy": 1013.0 / (287.05 * (22.22 + 273.15)),
            })
            continue

        # Get coordinates
        lat, lon = VENUE_COORDS.get(venue_name, (None, None))
        if lat is None:
            continue

        try:
            # Use archive API for dates > 5 days ago, forecast for recent
            days_ago = (datetime.now() - datetime.strptime(game_date, "%Y-%m-%d")).days
            if days_ago > 5:
                api_url = "https://archive-api.open-meteo.com/v1/archive"
            else:
                api_url = "https://api.open-meteo.com/v1/forecast"

            api_params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
                "start_date": game_date,
                "end_date": game_date,
                "temperature_unit": "fahrenheit",
            }

            # Try twice — longer timeout on retry
            resp = None
            for attempt, timeout in [(1, 10), (2, 20)]:
                try:
                    resp = requests.get(api_url, params=api_params, timeout=timeout)
                    if resp.status_code == 200:
                        break
                except requests.exceptions.Timeout:
                    if attempt == 1:
                        print(f"      ⟳ Timeout for {venue_name} — retrying...")
                        time.sleep(2)
                    else:
                        print(f"      ⚠ {venue_name} ({game_date}): timed out twice, skipping")
                except Exception:
                    break

            if resp is None or resp.status_code != 200:
                continue

            hourly = resp.json().get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            humidity = hourly.get("relative_humidity_2m", [])
            pressure = hourly.get("surface_pressure", [])
            wind = hourly.get("wind_speed_10m", [])

            if not temps:
                continue

            # Use afternoon hours (13-16 = 1-4 PM) as typical game time
            afternoon = slice(13, 17)
            temp_f = np.mean(temps[afternoon]) if len(temps) > 16 else np.mean(temps)
            humid = np.mean(humidity[afternoon]) if len(humidity) > 16 else np.mean(humidity) if humidity else 55.0
            press = np.mean(pressure[afternoon]) if len(pressure) > 16 else np.mean(pressure) if pressure else 1013.0
            wind_mph = np.mean(wind[afternoon]) if len(wind) > 16 else np.mean(wind) if wind else 6.0

            temp_c = (temp_f - 32) * 5 / 9
            air_density = press / (287.05 * (temp_c + 273.15))

            weather_records.append({
                "game_pk": game_pk,
                "venue_name": venue_name,
                "wx_temperature_f": round(temp_f, 1),
                "wx_humidity_pct": round(humid, 1),
                "wx_wind_speed_mph": round(wind_mph, 1),
                "wx_pressure_hpa": round(press, 1),
                "wx_is_dome": 0,
                "wx_is_cold": int(temp_f < 55),
                "wx_is_hot": int(temp_f > 85),
                "wx_is_windy": int(wind_mph > 12),
                "wx_is_humid": int(humid > 75),
                "wx_air_density_proxy": round(air_density, 6),
            })

            time.sleep(0.25)  # Be polite to Open-Meteo

        except Exception as e:
            print(f"      ⚠ Weather error for {venue_name} ({game_date}): {e}")
            continue

    if weather_records:
        return pd.DataFrame(weather_records)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def append_to_cache(new_data, cache_path, key_cols, label=""):
    """Append new rows to existing cache CSV, deduplicating by key columns."""
    if len(new_data) == 0:
        print(f"    {label}: no new data to append")
        return new_data

    if cache_path.exists():
        existing = pd.read_csv(cache_path, low_memory=False)
        if "game_date" in existing.columns:
            existing["game_date"] = pd.to_datetime(existing["game_date"])
        if "game_date" in new_data.columns:
            new_data["game_date"] = pd.to_datetime(new_data["game_date"])
        prev_count = len(existing)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        new_rows = len(combined) - prev_count
        print(f"    {label}: {prev_count:,} → {len(combined):,} rows ({new_rows:+,} new)")
    else:
        combined = new_data
        print(f"    {label}: created with {len(combined):,} rows")

    combined.to_csv(cache_path, index=False)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def run_catcher_features():
    """Re-run collect.catcher so the catcher feature files include the new
    pitch-level data we just appended.

    This must run BEFORE features.py because that reads
    catcher_features_asof.csv and catcher_features_prior.csv and joins
    them onto pitcher-game rows. Stale catcher files = stale features
    for any games added in this refresh.

    Run as a subprocess rather than an in-process import: both modules are
    memory-hungry (catcher features scan ~4M pitch rows, feature
    engineering builds a 1,200-column frame), and a separate process
    returns that memory to the OS on exit instead of holding it for the
    remainder of the refresh.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pitcher_model.collect.catcher"],
        cwd=ROOT, capture_output=False)
    return result.returncode == 0


def run_feature_engineering():
    """Re-run pitcher_model.features to rebuild all rolling features."""
    result = subprocess.run(
        [sys.executable, "-m", "pitcher_model.features"],
        cwd=ROOT, capture_output=False)
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end daily refresh for the MLB pitcher model"
    )
    parser.add_argument("--days", type=int, default=3,
                        help="Days of data to pull (default: 3)")
    parser.add_argument("--skip-fe", action="store_true",
                        help="Skip feature engineering (just cache data)")
    parser.add_argument("--skip-catcher-features", action="store_true",
                        help="Skip the catcher-feature rebuild "
                             "(faster, but catcher features will be stale "
                             "for new games)")
    parser.add_argument("--force", action="store_true",
                        help="Re-pull even if data looks fresh")
    args = parser.parse_args()

    today = datetime.now()
    end_date = today.strftime("%Y-%m-%d")  # Include today — catches last night's late games
    start_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"{'=' * 65}")
    print(f"  MLB PITCHER MODEL — DAILY REFRESH")
    print(f"  {today.strftime('%Y-%m-%d %H:%M')} | Pulling {start_date} → {end_date}")
    print(f"{'=' * 65}")

    DATA_DIR.mkdir(exist_ok=True)

    # ── Freshness check ─────────────────────────────────────────────────
    # Compare against YESTERDAY (the latest day with completed games), not
    # `end_date` which is today. Otherwise the script bails out the moment
    # last night's games are ingested, even if late games haven't been added
    # yet — and it never reruns feature engineering for in-progress data.
    if not args.force:
        feat_path = DATA_DIR / "pitcher_model_features.csv"
        if feat_path.exists():
            try:
                dates = pd.read_csv(feat_path, usecols=["game_date"])
                dates["game_date"] = pd.to_datetime(dates["game_date"])
                latest_game = dates["game_date"].max()
                yesterday = pd.Timestamp(today.date()) - pd.Timedelta(days=1)
                days_stale = (yesterday - latest_game).days
                if days_stale < 0:
                    print(f"\n  ℹ Features already include games through {latest_game.strftime('%Y-%m-%d')}.")
                    print(f"    No new games to add. Use --force to refresh anyway.")
                    return
                else:
                    print(f"\n  Data goes through {latest_game.strftime('%Y-%m-%d')} — "
                          f"{days_stale} day(s) behind yesterday. Refreshing...")
            except Exception:
                pass  # If we can't read the file, just proceed with refresh

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: STATCAST
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"  STEP 1: Pull recent Statcast pitch data")
    print(f"{'─' * 65}")

    raw = pull_recent_statcast(start_date, end_date)
    has_statcast = len(raw) > 0

    if has_statcast:
        # Aggregate pitcher-game
        print(f"\n    Aggregating to pitcher-game level...")
        game_level = aggregate_pitcher_game(raw)
        print(f"    ✓ {len(game_level):,} pitcher-game rows")

        # Update season cache
        season_cache = DATA_DIR / f"statcast_pitcher_games_{CURRENT_SEASON}.csv"
        append_to_cache(game_level, season_cache,
                        key_cols=["pitcher", "game_pk"],
                        label=f"Pitcher-game ({CURRENT_SEASON})")

        # Update combined all-seasons file
        all_cache = DATA_DIR / "statcast_pitcher_games_all.csv"
        if all_cache.exists():
            append_to_cache(game_level, all_cache,
                            key_cols=["pitcher", "game_pk"],
                            label="Pitcher-game (all seasons)")
        else:
            print(f"    ⚠ {all_cache.name} not found — run collect/statcast.py first for historical data")

        # ── PA-level events (for 08_per_pa_model + 09 ensemble) ───────
        # Extract PA-terminating rows and append to the rolling PA file
        # that 08/09 read. Dedupe on (game_pk, at_bat_number) so repeat
        # runs with overlapping date windows don't double-count.
        print(f"\n    Extracting PA-terminating rows for per-PA model...")
        pa_new = raw[raw["events"].notna()].copy()
        if len(pa_new) > 0:
            keep_cols = ["game_pk", "game_date", "pitcher", "batter",
                         "stand", "p_throws", "inning", "at_bat_number",
                         "home_team", "away_team", "events"]
            keep_cols = [c for c in keep_cols if c in pa_new.columns]
            pa_new = pa_new[keep_cols].copy()
            pa_new["was_K"] = pa_new["events"].isin(
                ["strikeout", "strikeout_double_play"]).astype(int)
            pa_new["was_BB"] = pa_new["events"].isin(
                ["walk", "intent_walk"]).astype(int)
            pa_new["was_HBP"] = (pa_new["events"] == "hit_by_pitch").astype(int)
            pa_new["was_hit"] = pa_new["events"].isin(
                ["single", "double", "triple", "home_run"]).astype(int)
            pa_new["was_in_play"] = (~pa_new["events"].isin([
                "strikeout", "strikeout_double_play",
                "walk", "intent_walk", "hit_by_pitch",
            ])).astype(int)

            pa_cache = DATA_DIR / "statcast_pa_events_all.csv"
            append_to_cache(pa_new, pa_cache,
                            key_cols=["game_pk", "at_bat_number"],
                            label="PA events")
        else:
            print(f"    (no PA-terminating rows in this pull)")

        # ── Pitch-level rows (for collect.catcher) ────────────────────
        # Parallels the PA events extraction above but keeps EVERY pitch,
        # not just the PA-terminating ones. The 25-column subset matches
        # PITCH_LEVEL_COLUMNS in collect/statcast.py — keep them in sync if
        # you ever change one. Dedupe on (game_pk, at_bat_number,
        # pitch_number) so re-running with overlapping windows doesn't
        # duplicate rows.
        print(f"\n    Extracting pitch-level rows for catcher features...")
        pitch_cols = [
            "game_pk", "game_date", "pitcher", "batter", "fielder_2",
            "stand", "p_throws", "inning", "at_bat_number", "pitch_number",
            "home_team", "away_team",
            "balls", "strikes",
            "pitch_type", "release_speed", "pfx_x", "pfx_z", "release_spin_rate",
            "description", "type", "zone",
            "plate_x", "plate_z",
            "events",
        ]
        keep_pitch_cols = [c for c in pitch_cols if c in raw.columns]
        pitches_new = raw[keep_pitch_cols].copy()
        if len(pitches_new) > 0:
            pitches_cache = DATA_DIR / "statcast_pitches_all.csv"
            append_to_cache(pitches_new, pitches_cache,
                            key_cols=["game_pk", "at_bat_number", "pitch_number"],
                            label="Pitch-level")
        else:
            print(f"    (no pitch rows in this pull)")

        # Pitch-type aggregations
        print(f"\n    Aggregating pitcher × pitch type...")
        pitcher_pt = aggregate_pitch_types(raw)
        if len(pitcher_pt) > 0:
            pt_season = DATA_DIR / f"pitcher_pitch_type_{CURRENT_SEASON}.csv"
            append_to_cache(pitcher_pt, pt_season,
                            key_cols=["pitcher", "game_pk", "pitch_type"],
                            label=f"Pitcher-PT ({CURRENT_SEASON})")
            pt_all = DATA_DIR / "pitcher_pitch_type_all.csv"
            if pt_all.exists():
                append_to_cache(pitcher_pt, pt_all,
                                key_cols=["pitcher", "game_pk", "pitch_type"],
                                label="Pitcher-PT (all)")

        print(f"\n    Aggregating batter × pitch type...")
        batter_pt = aggregate_batter_pitch_types(raw)
        if len(batter_pt) > 0:
            bt_season = DATA_DIR / f"batter_pitch_type_{CURRENT_SEASON}.csv"
            append_to_cache(batter_pt, bt_season,
                            key_cols=["batter", "game_pk", "pitch_type"],
                            label=f"Batter-PT ({CURRENT_SEASON})")
            bt_all = DATA_DIR / "batter_pitch_type_all.csv"
            if bt_all.exists():
                append_to_cache(batter_pt, bt_all,
                                key_cols=["batter", "game_pk", "pitch_type"],
                                label="Batter-PT (all)")
    else:
        print("    No new Statcast data (normal if no games were played).")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: GAME METADATA (MLB API)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"  STEP 2: Pull game metadata from MLB Stats API")
    print(f"{'─' * 65}")

    finished_games = get_finished_games(start_date, end_date)
    print(f"    Found {len(finished_games)} finished games")

    if finished_games:
        meta_df = pd.DataFrame(finished_games)

        # Update season + combined metadata caches
        season_meta = DATA_DIR / f"game_metadata_{CURRENT_SEASON}.csv"
        append_to_cache(meta_df, season_meta,
                        key_cols=["game_pk"],
                        label=f"Game metadata ({CURRENT_SEASON})")

        meta_all = DATA_DIR / "game_metadata_all.csv"
        if meta_all.exists():
            append_to_cache(meta_df, meta_all,
                            key_cols=["game_pk"],
                            label="Game metadata (all)")
        else:
            # First time — create with just this season
            meta_df.to_csv(meta_all, index=False)
            print(f"    Game metadata (all): created with {len(meta_df)} rows")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: LINEUPS + UMPIRES (BOXSCORE API)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"  STEP 3: Pull lineups & umpire assignments")
    print(f"{'─' * 65}")

    if finished_games:
        # Only fetch boxscores for games we don't already have
        new_game_pks = [g["game_pk"] for g in finished_games]

        # ── Earned runs collection (for train/predict earned_runs) ─────
        # Pull ER for every finished game_pk; the collector handles dedup.
        try:
            from pitcher_model.collect.statcast import collect_earned_runs
            print(f"\n    Collecting earned runs for {len(new_game_pks)} games...")
            collect_earned_runs(new_game_pks, verbose=True)
        except Exception as _e:
            print(f"    ⚠ ER collection failed: {_e}")

        lineup_cache  = DATA_DIR / "game_lineups.csv"
        umpire_cache  = DATA_DIR / "umpire_assignments.csv"
        catcher_cache = DATA_DIR / "game_catchers.csv"

        # Check which game_pks we already have for each cache type. We
        # fetch a boxscore if ANY of the three caches is missing this game,
        # since one API call covers all three extractions.
        existing_lineup_pks  = set()
        existing_umpire_pks  = set()
        existing_catcher_pks = set()
        if lineup_cache.exists():
            existing_lineup_pks = set(pd.read_csv(lineup_cache, usecols=["game_pk"])["game_pk"].unique())
        if umpire_cache.exists():
            existing_umpire_pks = set(pd.read_csv(umpire_cache, usecols=["game_pk"])["game_pk"].unique())
        if catcher_cache.exists():
            existing_catcher_pks = set(pd.read_csv(catcher_cache, usecols=["game_pk"])["game_pk"].unique())

        pks_needing_lineups  = [pk for pk in new_game_pks if pk not in existing_lineup_pks]
        pks_needing_umpires  = [pk for pk in new_game_pks if pk not in existing_umpire_pks]
        pks_needing_catchers = [pk for pk in new_game_pks if pk not in existing_catcher_pks]
        pks_to_fetch = list(set(pks_needing_lineups + pks_needing_umpires + pks_needing_catchers))

        if pks_to_fetch:
            print(f"    Fetching boxscores for {len(pks_to_fetch)} new games...")
            lineup_records, umpire_records, catcher_records = fetch_boxscore_data(pks_to_fetch)

            if lineup_records:
                lineup_df = pd.DataFrame(lineup_records)
                append_to_cache(lineup_df, lineup_cache,
                                key_cols=["game_pk", "player_id"],
                                label="Lineups")

            if umpire_records:
                umpire_df = pd.DataFrame(umpire_records)
                append_to_cache(umpire_df, umpire_cache,
                                key_cols=["game_pk"],
                                label="Umpires")

            if catcher_records:
                catcher_df = pd.DataFrame(catcher_records)
                append_to_cache(catcher_df, catcher_cache,
                                key_cols=["game_pk", "side"],
                                label="Catchers")
        else:
            print(f"    All {len(new_game_pks)} games already cached — skipping boxscore calls")
    else:
        print("    No finished games to fetch lineups/umpires for")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: WEATHER (OPEN-METEO)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print(f"  STEP 4: Fetch weather for recent games")
    print(f"{'─' * 65}")

    if finished_games:
        meta_df = pd.DataFrame(finished_games)
        weather_cache = DATA_DIR / "game_weather.csv"

        # Check which game_pks already have weather
        existing_wx_pks = set()
        if weather_cache.exists():
            existing_wx_pks = set(pd.read_csv(weather_cache, usecols=["game_pk"])["game_pk"].unique())

        games_needing_wx = meta_df[~meta_df["game_pk"].isin(existing_wx_pks)]

        if len(games_needing_wx) > 0:
            print(f"    Fetching weather for {len(games_needing_wx)} new games...")
            wx_df = fetch_weather_for_games(games_needing_wx)
            if len(wx_df) > 0:
                append_to_cache(wx_df, weather_cache,
                                key_cols=["game_pk"],
                                label="Weather")
            else:
                print(f"    No weather data retrieved (API may be slow)")
        else:
            print(f"    All games already have weather data — skipping")
    else:
        print("    No finished games to fetch weather for")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: CATCHER FEATURES + FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_fe:
        # Catcher features must rebuild BEFORE 02 runs, since 02 reads the
        # catcher_features_*.csv files. Skipping this step means 02 joins
        # stale catcher data for newly-added games.
        if not args.skip_catcher_features:
            print(f"\n{'─' * 65}")
            print(f"  STEP 5a: Rebuild catcher features (01c)")
            print(f"{'─' * 65}")
            cf_success = run_catcher_features()
            if not cf_success:
                print(f"    ⚠ Catcher feature rebuild had issues — "
                      f"02 will use stale or default values")

        print(f"\n{'─' * 65}")
        print(f"  STEP 5b: Re-run feature engineering (02)")
        print(f"{'─' * 65}")

        success = run_feature_engineering()
        if success:
            feat_path = DATA_DIR / "pitcher_model_features.csv"
            if feat_path.exists():
                df = pd.read_csv(feat_path, usecols=["game_date"])
                df["game_date"] = pd.to_datetime(df["game_date"])
                latest = df["game_date"].max()
                n_rows = len(df)
                print(f"\n    ✓ Features updated through {latest.strftime('%Y-%m-%d')}")
                print(f"      ({n_rows:,} total rows)")
        else:
            print(f"\n    ✗ Feature engineering failed — check output above")

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  ✓ REFRESH COMPLETE")

    # Sanity-check that the models the daily scripts will load actually exist.
    # Refreshing data is pointless if the trained models are missing.
    model_dir = Path("models")
    required_models = {
        "rate_model.joblib": "Strikeout rate model (train/strikeouts.py)",
        "bf_model.joblib": "Batters faced model (06 or 09)",
        "beta_binom_config.json": "Beta-Binomial config",
        "hits_rate_model.joblib": "Hits rate model (train/hits_walks.py)",
        "walks_rate_model.joblib": "Walks rate model (train/hits_walks.py)",
        "outs_rate_model.joblib": "Outs rate model (train/outs.py)",
        "er_rate_model.joblib": "Earned runs rate model (train/earned_runs.py)",
    }
    missing = [name for name in required_models if not (model_dir / name).exists()]
    if missing:
        print(f"\n  ⚠ Missing model files (run training first):")
        for name in missing:
            print(f"      - {name}  ({required_models[name]})")
        print(f"    Train with:")
        print(f"      python run.py train strikeouts     # strikeouts + BF")
        print(f"      python run.py train hits-walks        # hits/walks")
        print(f"      python run.py train outs              # outs recorded")
        print(f"      python run.py train earned-runs                # earned runs")

    if not args.skip_fe:
        print(f"\n  Next:")
        print(f"    python run.py predict strikeouts    # strikeouts")
        print(f"    python run.py predict hits-walks        # hits & walks")
        print(f"    python run.py predict outs              # outs recorded")
        print(f"    python run.py predict earned-runs                # earned runs")
    else:
        print(f"\n  Data cached. Run:")
        print(f"    python run.py features")
        print(f"    python run.py predict strikeouts")
        print(f"    python run.py predict hits-walks")
        print(f"    python run.py predict outs")
        print(f"    python run.py predict earned-runs")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
