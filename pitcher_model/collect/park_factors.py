"""
Park Factors Collector
=======================
One-time scrape of Statcast venue park factors from Baseball Savant,
producing a real `data/park_factors.csv` to replace the placeholder
(all-100s) file. Baseball Savant publishes 3-year regressed run
indexes per team-season.

Output schema:
    Team, Season, Basic
    ATL, 2024, 102
    COL, 2024, 116
    ...

This script overwrites data/park_factors.csv.

Usage:
    python run.py collect park-factors             # all seasons 2021-current
    python run.py collect park-factors 2024 2025   # specific seasons

If Baseball Savant changes their endpoint, fall back to using the
pybaseball.statcast.park_factors function or manually downloading the
CSV from the page directly.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

from pitcher_model.paths import DATA_DIR, ensure_dirs

ensure_dirs(DATA_DIR)

OUT = DATA_DIR / "park_factors.csv"

# Baseball Savant's park factors endpoint — returns JSON
# `type=year` gives per-season factors; rolling=3 gives 3-year regressed
URL_TEMPLATE = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year={year}&batSide=&stat=index_wOBA&condition=All&rolling="
)


def fetch_year(year, headers):
    """
    Fetch park factors for a single season from Baseball Savant.
    Returns a DataFrame with columns: Team, Season, Basic
    """
    url = URL_TEMPLATE.format(year=year)
    # We append &csv=true which Savant supports for some endpoints
    csv_url = url + "&csv=true"

    print(f"  {year}: fetching from Baseball Savant...")
    for attempt in range(3):
        try:
            r = requests.get(csv_url, headers=headers, timeout=20)
            r.raise_for_status()
            text = r.text
            if not text or text.startswith("<"):
                # HTML returned — not CSV. Try the JSON-embedded variant.
                r2 = requests.get(url, headers=headers, timeout=20)
                r2.raise_for_status()
                # Savant embeds the data as JSON in a script tag — extract it
                import re
                import json
                m = re.search(r"var\s+data\s*=\s*(\[.*?\]);", r2.text, re.DOTALL)
                if not m:
                    raise ValueError("Could not locate data in page HTML")
                data = json.loads(m.group(1))
                df = pd.DataFrame(data)
            else:
                from io import StringIO
                df = pd.read_csv(StringIO(text))
            return df
        except Exception as e:
            if attempt < 2:
                print(f"    Attempt {attempt+1} failed ({e}); retrying...")
                time.sleep(2 ** attempt)
            else:
                raise


def normalize(df, year):
    """
    Normalize Savant's response to (Team, Season, Basic) format.
    Savant returns columns like: venue_id, name, year, park_factor, runs_factor, ...
    """
    # Make columns lowercase
    df.columns = [str(c).lower() for c in df.columns]

    # Find team abbreviation column — Savant uses 'team_abbrev' or similar
    team_col = next((c for c in df.columns
                     if c in ("team", "team_abbrev", "abbrev", "venue_name",
                              "name", "venue")), None)
    # Find runs factor column — prefer "runs" over "park_factor" if both exist
    runs_col = next((c for c in df.columns
                     if c in ("runs_factor", "runs", "r", "park_factor",
                              "index_r", "index_runs", "park_factor_runs")),
                    None)
    if runs_col is None:
        # Fall back to first numeric column that looks like an index (~100)
        for c in df.columns:
            try:
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                if 70 <= vals.median() <= 130:
                    runs_col = c
                    break
            except Exception:
                pass

    if not team_col or not runs_col:
        raise ValueError(
            f"Could not locate team/runs columns. Available: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "Team":   df[team_col],
        "Season": year,
        "Basic":  pd.to_numeric(df[runs_col], errors="coerce"),
    })
    out = out.dropna()
    return out


# Map full team names to 3-letter Statcast codes (in case Savant returns full names)
TEAM_ABBREV = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "A's": "OAK",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL",
    "Cubs": "CHC", "D-backs": "ARI", "Diamondbacks": "ARI", "Dodgers": "LAD",
    "Giants": "SF", "Guardians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSH", "Orioles": "BAL", "Padres": "SD",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX", "Rays": "TB",
    "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL", "Royals": "KC",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CWS", "Yankees": "NYY",
}

# Map venue names → 3-letter team abbreviations.
# Baseball Savant returns the venue (e.g. "Coors Field"), not the team
# abbreviation. We need to normalize to ATL/COL/NYY etc. so the join key
# matches the home_team column in pitcher_model_features.csv.
VENUE_TO_TEAM = {
    "Angel Stadium":                  "LAA",
    "Daikin Park":                    "HOU",  # Astros (renamed from Minute Maid Park in 2024)
    "Minute Maid Park":               "HOU",
    "Oakland Coliseum":               "OAK",
    "Sutter Health Park":             "ATH",  # 2025+ Sacramento (Athletics)
    "Sahlen Field":                   "TOR",  # 2020-21 temporary Toronto home
    "TD Ballpark":                    "TOR",  # 2021 spring training emergency home
    "Rogers Centre":                  "TOR",
    "Truist Park":                    "ATL",
    "American Family Field":          "MIL",
    "Busch Stadium":                  "STL",
    "Wrigley Field":                  "CHC",
    "Chase Field":                    "ARI",
    "Dodger Stadium":                 "LAD",
    "UNIQLO Field at Dodger Stadium": "LAD",  # sponsorship overlay, same park
    "Oracle Park":                    "SF",
    "Progressive Field":              "CLE",
    "T-Mobile Park":                  "SEA",
    "loanDepot park":                 "MIA",
    "Hard Rock Stadium":              "MIA",
    "Citi Field":                     "NYM",
    "Nationals Park":                 "WSH",
    "Oriole Park at Camden Yards":    "BAL",
    "Camden Yards":                   "BAL",
    "Petco Park":                     "SD",
    "Citizens Bank Park":             "PHI",
    "PNC Park":                       "PIT",
    "Globe Life Field":               "TEX",
    "Tropicana Field":                "TB",
    "Steinbrenner Field":             "TB",   # 2025 temp home after roof damage
    "George M. Steinbrenner Field":   "TB",
    "Fenway Park":                    "BOS",
    "Great American Ball Park":       "CIN",
    "Coors Field":                    "COL",
    "Kauffman Stadium":               "KC",
    "Comerica Park":                  "DET",
    "Target Field":                   "MIN",
    "Rate Field":                     "CWS",  # 2024+ rename of Guaranteed Rate Field
    "Guaranteed Rate Field":          "CWS",
    "Yankee Stadium":                 "NYY",
}


def main(years=None):
    if not years:
        years = list(range(2021, 2027))
    else:
        years = [int(y) for y in years]

    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36"),
        "Accept": "text/csv,application/json,*/*",
    }

    all_parts = []
    for year in years:
        try:
            raw = fetch_year(year, headers)
            normalized = normalize(raw, year)
            # Map venue names → team abbreviations first (Savant returns these
            # for the leaderboard view), then map any remaining full-team names.
            normalized["Team"] = normalized["Team"].replace(VENUE_TO_TEAM)
            normalized["Team"] = normalized["Team"].replace(TEAM_ABBREV)
            # Sanity check: warn about anything still longer than 3 chars
            unresolved = normalized[normalized["Team"].str.len() > 3]
            if len(unresolved) > 0:
                print(f"    ⚠ {year}: {len(unresolved)} rows have unresolved "
                      f"team names: {sorted(unresolved['Team'].unique())}")
                print(f"      Add these to VENUE_TO_TEAM in collect/park_factors.py")
            all_parts.append(normalized)
            print(f"    ✓ {year}: {len(normalized)} teams")
            time.sleep(1)
        except Exception as e:
            print(f"    ✗ {year}: failed ({e})")
            print(f"      You can manually download from: ")
            print(f"      https://baseballsavant.mlb.com/leaderboard/"
                  f"statcast-park-factors?type=year&year={year}")

    if not all_parts:
        print("\n✗ No data collected.")
        sys.exit(1)

    out = pd.concat(all_parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"\n✓ Saved {OUT} ({len(out)} rows)")
    print(f"\nSample:")
    print(out.groupby("Team")["Basic"].mean().sort_values(ascending=False).head(10))
    print(f"\n  Range: {out['Basic'].min():.0f} – {out['Basic'].max():.0f}")
    print(f"  Unique values: {out['Basic'].nunique()}")
    if out["Basic"].nunique() <= 1:
        print(f"\n  ⚠ Only one unique value — fetch may have failed silently.")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)
