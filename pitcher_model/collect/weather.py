"""
MLB Weather Data Collection
=============================
Pulls historical weather data for every game in the dataset using the
free Open-Meteo Archive API (no API key needed).

For each game, fetches hourly weather at the venue's coordinates for
the game date and extracts conditions around typical first-pitch times:
  - Temperature (°F)
  - Relative humidity (%)
  - Wind speed (mph)
  - Wind gusts (mph)
  - Precipitation (mm)
  - Cloud cover (%)
  - Pressure (hPa)

Requirements:
    pip install pandas requests

Usage:
    python run.py collect weather

    Run this AFTER collect/statcast.py (needs game_metadata_all.csv
    and venue_info.csv).
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pitcher_model.paths import DATA_DIR

warnings.filterwarnings("ignore")


OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Typical first-pitch hours (local approximations in UTC offset)
# Night games ~7pm local, day games ~1pm local
# We'll pull a window of hours and pick the best match
GAME_HOUR_WINDOW = list(range(17, 24))  # 5pm-11pm UTC covers most US games


def load_game_data():
    """Load game metadata and venue coordinates."""
    meta_path = DATA_DIR / "game_metadata_all.csv"
    venue_path = DATA_DIR / "venue_info.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run collect/statcast.py first.")

    meta = pd.read_csv(meta_path, parse_dates=["game_date"])
    print(f"  Games: {len(meta):,}")

    if venue_path.exists():
        venues = pd.read_csv(venue_path)
        print(f"  Venues: {len(venues)}")
    else:
        print("  ⚠ No venue_info.csv — will skip weather collection")
        venues = pd.DataFrame()

    return meta, venues


def fetch_weather_for_venue_daterange(lat, lon, start_date, end_date):
    """
    Fetch hourly weather from Open-Meteo Archive API for a coordinate
    range. The API supports date ranges up to ~1 year per call.

    Returns a DataFrame with hourly weather data.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_gusts_10m",
            "precipitation",
            "cloud_cover",
            "surface_pressure",
        ]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",  # Normalize to ET for game times
    }

    try:
        resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            return pd.DataFrame()

        hourly = data["hourly"]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "temperature_f": hourly.get("temperature_2m"),
            "humidity_pct": hourly.get("relative_humidity_2m"),
            "wind_speed_mph": hourly.get("wind_speed_10m"),
            "wind_gusts_mph": hourly.get("wind_gusts_10m"),
            "precipitation_mm": hourly.get("precipitation"),
            "cloud_cover_pct": hourly.get("cloud_cover"),
            "pressure_hpa": hourly.get("surface_pressure"),
        })
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        return df

    except Exception as e:
        print(f"    ⚠ Weather API error: {e}")
        return pd.DataFrame()


def extract_game_time_weather(hourly_df, game_date, is_night=True):
    """
    Extract weather conditions around game time from hourly data.
    Night games: average of 7pm-9pm local (hours 19-21)
    Day games: average of 1pm-3pm local (hours 13-15)
    """
    target_date = pd.Timestamp(game_date).date()
    day_data = hourly_df[hourly_df["date"] == target_date]

    if len(day_data) == 0:
        return {}

    if is_night:
        game_hours = [19, 20, 21]
    else:
        game_hours = [13, 14, 15]

    game_data = day_data[day_data["hour"].isin(game_hours)]

    # Fall back to wider window if exact hours not available
    if len(game_data) == 0:
        game_data = day_data[day_data["hour"].between(12, 23)]
    if len(game_data) == 0:
        game_data = day_data

    return {
        "wx_temperature_f": game_data["temperature_f"].mean(),
        "wx_humidity_pct": game_data["humidity_pct"].mean(),
        "wx_wind_speed_mph": game_data["wind_speed_mph"].mean(),
        "wx_wind_gusts_mph": game_data["wind_gusts_mph"].mean(),
        "wx_precipitation_mm": game_data["precipitation_mm"].sum(),
        "wx_cloud_cover_pct": game_data["cloud_cover_pct"].mean(),
        "wx_pressure_hpa": game_data["pressure_hpa"].mean(),
    }


def collect_weather():
    """
    Main collection: for each unique (venue, month-range), pull weather
    from Open-Meteo and extract game-time conditions.

    Strategy: batch by venue — one API call per venue per ~60-day chunk
    instead of one call per game. Much faster.
    """
    print("\n═══ Collecting Historical Weather Data ═══")

    cache_file = DATA_DIR / "game_weather.csv"
    if cache_file.exists():
        print("  Loading from cache...")
        return pd.read_csv(cache_file)

    meta, venues = load_game_data()

    if len(venues) == 0:
        print("  ⚠ No venue data — cannot collect weather")
        return pd.DataFrame()

    # Merge venue coordinates to games
    meta = meta.merge(venues, on="venue_name", how="left")
    meta = meta.dropna(subset=["latitude", "longitude"])

    # Group games by venue to batch API calls
    unique_venues = meta.groupby("venue_name").agg(
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        is_dome=("is_dome_or_retractable", "first"),
        min_date=("game_date", "min"),
        max_date=("game_date", "max"),
        n_games=("game_pk", "count"),
    ).reset_index()

    print(f"  {len(unique_venues)} venues, {len(meta):,} total games")

    all_weather = []

    for _, venue in unique_venues.iterrows():
        venue_name = venue["venue_name"]
        is_dome = venue["is_dome"] == 1 if pd.notna(venue["is_dome"]) else False

        if is_dome:
            # For domed/retractable venues, set neutral weather defaults
            venue_games = meta[meta["venue_name"] == venue_name]
            for _, game in venue_games.iterrows():
                all_weather.append({
                    "game_pk": game["game_pk"],
                    "venue_name": venue_name,
                    "wx_temperature_f": 72.0,  # Climate-controlled
                    "wx_humidity_pct": 50.0,
                    "wx_wind_speed_mph": 0.0,
                    "wx_wind_gusts_mph": 0.0,
                    "wx_precipitation_mm": 0.0,
                    "wx_cloud_cover_pct": 0.0,
                    "wx_pressure_hpa": 1013.0,
                    "wx_is_dome": 1,
                })
            print(f"  {venue_name}: dome/retractable — using defaults ({len(venue_games)} games)")
            continue

        # Outdoor venue — fetch from Open-Meteo in 60-day chunks
        lat, lon = venue["lat"], venue["lon"]
        venue_games = meta[meta["venue_name"] == venue_name].sort_values("game_date")

        min_dt = pd.Timestamp(venue["min_date"])
        max_dt = pd.Timestamp(venue["max_date"])

        print(f"  {venue_name}: fetching weather ({venue['n_games']} games)...")

        # Pull in 60-day chunks
        current = min_dt
        venue_hourly_chunks = []
        while current <= max_dt:
            chunk_end = min(current + pd.Timedelta(days=59), max_dt)
            hourly = fetch_weather_for_venue_daterange(
                lat, lon,
                str(current.date()),
                str(chunk_end.date()),
            )
            if len(hourly) > 0:
                venue_hourly_chunks.append(hourly)
            current = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.3)  # Rate limiting

        if not venue_hourly_chunks:
            print(f"    ⚠ No weather data returned")
            continue

        venue_hourly = pd.concat(venue_hourly_chunks, ignore_index=True)

        # Extract game-time weather for each game at this venue
        for _, game in venue_games.iterrows():
            is_night = game.get("day_night") == "night"
            wx = extract_game_time_weather(venue_hourly, game["game_date"], is_night)
            if wx:
                wx["game_pk"] = game["game_pk"]
                wx["venue_name"] = venue_name
                wx["wx_is_dome"] = 0
                all_weather.append(wx)

    if all_weather:
        weather_df = pd.DataFrame(all_weather)

        # Derived features
        weather_df["wx_is_cold"] = (weather_df["wx_temperature_f"] < 55).astype(int)
        weather_df["wx_is_hot"] = (weather_df["wx_temperature_f"] > 85).astype(int)
        weather_df["wx_is_windy"] = (weather_df["wx_wind_speed_mph"] > 12).astype(int)
        weather_df["wx_is_humid"] = (weather_df["wx_humidity_pct"] > 75).astype(int)
        weather_df["wx_has_rain"] = (weather_df["wx_precipitation_mm"] > 0.1).astype(int)

        # Air density proxy (affects ball carry — lower density = more HRs)
        # Simplified: higher temp + higher altitude + lower pressure = less dense
        weather_df["wx_air_density_proxy"] = (
            weather_df["wx_pressure_hpa"] /
            (weather_df["wx_temperature_f"].clip(lower=32) + 459.67)  # Rankine
        )

        weather_df.to_csv(cache_file, index=False)
        print(f"\n  ✓ Weather data saved: {len(weather_df):,} games")
        print(f"    Columns: {list(weather_df.columns)}")
        return weather_df

    print("  ⚠ No weather data collected")
    return pd.DataFrame()


if __name__ == "__main__":
    print("MLB Weather Data Collection")
    print("=" * 50)
    weather = collect_weather()

    if len(weather) > 0:
        print(f"\n── Weather Summary ──")
        print(f"  Games with weather: {len(weather):,}")
        print(f"  Temperature range: {weather['wx_temperature_f'].min():.0f}°F — {weather['wx_temperature_f'].max():.0f}°F")
        print(f"  Avg wind speed: {weather['wx_wind_speed_mph'].mean():.1f} mph")
        print(f"  Dome games: {weather['wx_is_dome'].sum():,}")
        print(f"  Rainy games: {weather['wx_has_rain'].sum():,}")
        print(f"\nSaved to: {DATA_DIR / 'game_weather.csv'}")
        print("\nNext: Re-run features.py and train/baseline.py")
