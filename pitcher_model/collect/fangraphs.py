"""
collect/fangraphs.py
========================
Updates FanGraphs data by appending manually downloaded CSVs.

FanGraphs blocks automated scraping (Cloudflare bot challenge). This
script merges manually exported CSVs with your existing cached data.

The default FanGraphs leaderboard export is missing two stat groups
that the model needs:
  - Pitch modeling stats (Stuff+, Location+, Pitching+) — type=36 view
  - Statcast stats (Barrel%, xwOBA, etc.)                — type=24 view

So per season you may have up to FOUR manual exports:
  - fg_pitching_<YEAR>.csv             (main pitching: K%, SwStr%, etc.)
  - fg_pitching_pitchmodel_<YEAR>.csv  (Stuff+, Location+, Pitching+)
  - fg_batting_<YEAR>.csv              (main batting:  K%, BB%, wRC+)
  - fg_batting_statcast_<YEAR>.csv     (Barrel%, xwOBA, xBA, etc.)

This script joins the supplements onto the main files by PlayerId, then
appends the result to the canonical fangraphs_*_seasons.csv files.

USAGE:
    1. Open the URLs printed by `python run.py collect fangraphs`
       (one for each of the 4 export types per season you want updated)
    2. Click 'Export Data' on each FanGraphs page
    3. Save to data/ with the filenames listed above
    4. Run: python run.py collect fangraphs
"""

import re
from pathlib import Path

import pandas as pd

from pitcher_model.paths import DATA_DIR as OUTPUT_DIR


# Canonical destination files
MAIN_PITCHING = "fangraphs_pitching_seasons.csv"
MAIN_BATTING  = "fangraphs_batting_seasons.csv"

# Manual download patterns. Order matters: main first, then supplements
# (so when we de-dupe columns, supplements only contribute their unique cols).
PITCHING_PATTERNS = [
    ("main",       "fg_pitching_[0-9]*.csv"),
    ("pitchmodel", "fg_pitching_pitchmodel_*.csv"),
]
BATTING_PATTERNS = [
    ("main",     "fg_batting_[0-9]*.csv"),
    ("statcast", "fg_batting_statcast_*.csv"),
]

# Columns we expect from the model side. Used only for the diagnostic printout.
PITCHING_KEY_COLS = ["Stuff+", "Location+", "Pitching+", "SwStr%", "TTO%",
                     "O-Swing%", "Contact%", "Zone%", "K%", "BB%"]
BATTING_KEY_COLS  = ["SwStr%", "O-Swing%", "Contact%", "Z-Contact%", "TTO%",
                     "K%", "BB%", "wRC+", "xwOBA", "Barrel%"]

# Columns to keep from the *main* file when merging supplements. Anything
# present in both files: the main file's version wins. The supplement only
# contributes columns that aren't already in the main file.
SHARED_IDENTITY_COLS = {"Name", "NameASCII", "Team", "PlayerId", "MLBAMID",
                        "IP", "PA", "G", "GS", "Age", "Season"}


def extract_season_from_filename(name: str) -> int | None:
    """Pull a 4-digit year (2021–2030) out of a filename."""
    m = re.search(r"(20\d{2})", name)
    if m:
        year = int(m.group(1))
        if 2021 <= year <= 2030:
            return year
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic
# ─────────────────────────────────────────────────────────────────────────────
def check_existing():
    """Show what FanGraphs data we have on disk and what's missing."""
    print("═══ Checking Existing FanGraphs Data ═══\n")

    for label, filename, key_cols in [
        ("Pitching", MAIN_PITCHING, PITCHING_KEY_COLS),
        ("Batting",  MAIN_BATTING,  BATTING_KEY_COLS),
    ]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            print(f"  {label}: ⚠ FILE NOT FOUND ({filename})")
            continue

        df = pd.read_csv(path)
        if "Season" in df.columns:
            seasons = sorted(df["Season"].dropna().unique().tolist())
        else:
            seasons = "(no Season col)"
        print(f"  {label}: {len(df)} rows, seasons {seasons}")

        found   = [c for c in key_cols if c in df.columns]
        missing = [c for c in key_cols if c not in df.columns]
        print(f"    Present: {', '.join(found) if found else '(none)'}")
        if missing:
            print(f"    Missing: {', '.join(missing)}")

        if isinstance(seasons, list) and 2026 in seasons:
            n_2026 = len(df[df["Season"] == 2026])
            print(f"    ✓ 2026 data present ({n_2026} rows)")
        elif isinstance(seasons, list):
            print(f"    ⚠ Missing 2026 season data")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Per-season supplement merge
# ─────────────────────────────────────────────────────────────────────────────
def merge_supplement(main_df: pd.DataFrame,
                     supp_df: pd.DataFrame,
                     supp_label: str) -> pd.DataFrame:
    """Left-join supp_df onto main_df by PlayerId, adding only NEW columns.

    The main file is treated as the source of truth for any column that
    appears in both files (Name, Team, IP, PA, etc.). The supplement only
    contributes columns the main file doesn't already have.

    Falls back to a Name-based merge only if PlayerId is missing on
    either side, with a warning — this should be rare with FanGraphs CSVs.
    """
    # Pick the merge key
    if "PlayerId" in main_df.columns and "PlayerId" in supp_df.columns:
        key = "PlayerId"
    elif "MLBAMID" in main_df.columns and "MLBAMID" in supp_df.columns:
        print(f"      (using MLBAMID for {supp_label} merge — PlayerId missing)")
        key = "MLBAMID"
    elif "Name" in main_df.columns and "Name" in supp_df.columns:
        print(f"      ⚠ falling back to Name-based merge for {supp_label} "
              "— may miss diacritics/Jr.")
        key = "Name"
    else:
        print(f"      ✗ no shared key — skipping {supp_label} supplement")
        return main_df

    # Figure out which columns the supplement uniquely adds
    main_cols = set(main_df.columns)
    new_cols  = [c for c in supp_df.columns
                 if c not in main_cols and c != key]
    if not new_cols:
        print(f"      ({supp_label}: no new columns to add)")
        return main_df

    # Build the slim supplement frame: just the key + new cols
    supp_slim = supp_df[[key] + new_cols].copy()

    # Drop dupes on the key in case a player has two rows in the supplement
    # (rare — happens for traded players if the export splits them).
    if supp_slim[key].duplicated().any():
        n_dupes = supp_slim[key].duplicated().sum()
        print(f"      ({supp_label}: dropping {n_dupes} duplicate {key} row(s))")
        supp_slim = supp_slim.drop_duplicates(subset=[key], keep="first")

    merged = main_df.merge(supp_slim, on=key, how="left")
    # Count matches by checking which main rows actually got joined.
    # We do this by tracking the supp keys before merge and seeing
    # which main keys are present in that set. Don't count by column
    # nullness — the first new column may be naturally null for many
    # players (e.g. Stf+ FA is null for pitchers who don't throw a 4-seam).
    n_matched = main_df[key].isin(set(supp_slim[key])).sum()
    print(f"      ({supp_label}: +{len(new_cols)} cols, "
          f"{n_matched}/{len(merged)} players matched)")
    return merged


def assemble_season_frame(season: int,
                          patterns: list[tuple[str, str]]) -> pd.DataFrame | None:
    """Look in OUTPUT_DIR for files matching each pattern for this season,
    load the main file, and merge any supplements onto it.

    Returns None if no main file is found for this season.
    """
    main_df = None
    for label, pattern in patterns:
        # Replace the wildcard with this specific year so we don't pick up
        # other seasons' files. Patterns look like "fg_pitching_*.csv" so
        # we swap the asterisk for the year (or insert it for prefix patterns).
        # The 'main' pattern uses [0-9]* so glob it then filter by year.
        candidates = sorted(OUTPUT_DIR.glob(pattern))
        candidates = [c for c in candidates
                      if extract_season_from_filename(c.name) == season]

        if not candidates:
            if label == "main" and main_df is None:
                # No main file for this season — nothing to do
                return None
            else:
                # Supplement is optional; just note it
                continue

        # If multiple files match, take the most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        path = candidates[0]
        df = pd.read_csv(path)

        if label == "main":
            print(f"    Loaded {path.name}: {len(df)} rows, {len(df.columns)} cols")
            # .copy() defragments the (300+ column) frame loaded from CSV;
            # without it, the next column assignment triggers a noisy
            # PerformanceWarning that doesn't reflect a real problem.
            df = df.copy()
            df["Season"] = season
            main_df = df
        else:
            print(f"    Merging {path.name} ({label}): "
                  f"{len(df)} rows, {len(df.columns)} cols")
            main_df = merge_supplement(main_df, df, label)

    return main_df


# ─────────────────────────────────────────────────────────────────────────────
# Main merge into canonical files
# ─────────────────────────────────────────────────────────────────────────────
def merge_manual_downloads():
    """Find all manual-download seasons, assemble per-season frames with
    supplements joined on, and append into the canonical season files."""
    print("═══ Merging Manual Downloads ═══\n")

    any_changes = False

    for label, main_file, patterns in [
        ("Pitching", MAIN_PITCHING, PITCHING_PATTERNS),
        ("Batting",  MAIN_BATTING,  BATTING_PATTERNS),
    ]:
        # Find all main-file seasons we have downloads for
        main_pattern = patterns[0][1]
        main_files   = sorted(OUTPUT_DIR.glob(main_pattern))
        seasons      = sorted({extract_season_from_filename(f.name)
                               for f in main_files
                               if extract_season_from_filename(f.name)})

        if not seasons:
            print(f"  {label}: no manual download files found.\n")
            continue

        print(f"  {label}: found downloads for seasons {seasons}")

        # Load existing canonical file (if any)
        main_path = OUTPUT_DIR / main_file
        if main_path.exists():
            canonical_df = pd.read_csv(main_path)
        else:
            canonical_df = pd.DataFrame()

        # For each season, assemble a fresh frame and replace any existing
        # rows for that season in the canonical file.
        for season in seasons:
            print(f"\n    ── Season {season} ──")
            season_df = assemble_season_frame(season, patterns)
            if season_df is None or season_df.empty:
                print(f"    (no data assembled for {season})")
                continue

            # Replace season's rows in canonical
            if not canonical_df.empty and "Season" in canonical_df.columns:
                before = len(canonical_df)
                canonical_df = canonical_df[canonical_df["Season"] != season]
                removed = before - len(canonical_df)
                if removed > 0:
                    print(f"    Replacing {removed} existing {season} rows")

            canonical_df = pd.concat([canonical_df, season_df],
                                     ignore_index=True, sort=False)
            any_changes = True

        if any_changes:
            canonical_df.to_csv(main_path, index=False)
            print(f"\n  ✓ Wrote {main_path} "
                  f"({len(canonical_df)} total rows, "
                  f"{len(canonical_df.columns)} cols)\n")

    return any_changes


# ─────────────────────────────────────────────────────────────────────────────
# Instructions
# ─────────────────────────────────────────────────────────────────────────────
def print_instructions():
    """Print download instructions when no manual files are found."""
    print("═══ How to Download FanGraphs Data Manually ═══\n")
    print("  FanGraphs is now behind Cloudflare and blocks automated scraping.")
    print("  For each season you want to refresh, you need up to FOUR exports.\n")

    long_pitching_url = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all"
        "&qual=1&type=c," + ",".join(str(i) for i in [-1] + list(range(3, 301))) +
        "&season={Y}&month=0&season1={Y}&ind=0&team=&rost=0&age=0,100"
        "&filter=&players=&page=1_500"
    )
    long_batting_url = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all"
        "&qual=1&type=c," + ",".join(str(i) for i in [-1] + list(range(3, 301))) +
        "&season={Y}&month=0&season1={Y}&ind=0&team=&rost=0&age=0,100"
        "&filter=&players=&page=1_500"
    )
    pitchmodel_url = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all"
        "&qual=1&season={Y}&season1={Y}&ind=0&type=36"
    )
    statcast_url = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all"
        "&qual=1&season={Y}&season1={Y}&ind=0&type=24"
    )

    Y = 2026  # change this when refreshing other seasons
    print(f"  For season {Y}, open these four URLs and click Export Data on each.")
    print(f"  Save with the filenames shown.\n")

    print(f"  1. MAIN PITCHING        → data/fg_pitching_{Y}.csv")
    print(f"     {long_pitching_url.format(Y=Y)}\n")

    print(f"  2. PITCH-MODEL PITCHING → data/fg_pitching_pitchmodel_{Y}.csv")
    print(f"     (Stuff+, Location+, Pitching+)")
    print(f"     {pitchmodel_url.format(Y=Y)}\n")

    print(f"  3. MAIN BATTING         → data/fg_batting_{Y}.csv")
    print(f"     {long_batting_url.format(Y=Y)}\n")

    print(f"  4. STATCAST BATTING     → data/fg_batting_statcast_{Y}.csv")
    print(f"     (Barrel%, xwOBA, xBA, EV, HardHit%)")
    print(f"     {statcast_url.format(Y=Y)}\n")

    print("  Then re-run: python run.py collect fangraphs\n")
    print("  The script will detect all four files, join the supplements onto")
    print("  the main files by PlayerId, and append into your canonical")
    print("  fangraphs_*_seasons.csv files.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FanGraphs Data Manager")
    print("=" * 50)

    check_existing()
    merged = merge_manual_downloads()

    if not merged:
        print_instructions()
    else:
        print("✓ Data updated. Run: python run.py features")
        print("\nFinal state:")
        check_existing()
