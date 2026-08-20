"""
collect/catcher.py
=============================
Build catcher game-calling features from pitch-level Statcast data.

Two output files, both leakage-free by construction:

  data/catcher_features_asof.csv
    One row per (catcher_id, game_date) with as-of features computed using
    only pitches in the same season strictly BEFORE that date.

  data/catcher_features_prior.csv
    One row per (catcher_id, season) with FULL-SEASON features. Used by 02
    as the prior-season anchor (always season S-1 for a season-S game,
    so no leakage).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_model.paths import DATA_DIR

PITCHES_PATH = DATA_DIR / "statcast_pitches_all.csv"
ASOF_PATH    = DATA_DIR / "catcher_features_asof.csv"
PRIOR_PATH   = DATA_DIR / "catcher_features_prior.csv"

FASTBALL_TYPES = {"FF", "SI", "FT", "FA", "FC"}
BREAKING_TYPES = {"SL", "CU", "KC", "SV", "ST", "CS", "SC"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "KN"}

IN_ZONE_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9}
SHADOW_CODES  = {11, 12, 13, 14}

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "missed_bunt",
}
WHIFF_DESCRIPTIONS         = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}


def load_and_label_pitches(path: Path) -> pd.DataFrame:
    print(f"Loading {path}...")
    use_cols = ["game_date", "pitcher", "fielder_2", "stand",
                "balls", "strikes", "pitch_type", "description", "zone"]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    print(f"  {len(df):,} pitch rows")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season"]    = df["game_date"].dt.year

    before = len(df)
    df = df.dropna(subset=["fielder_2", "pitcher", "pitch_type", "stand", "season"])
    if (dropped := before - len(df)) > 0:
        print(f"  Dropped {dropped:,} rows with missing critical fields")

    df["fielder_2"] = df["fielder_2"].astype(int)
    df["pitcher"]   = df["pitcher"].astype(int)
    df["season"]    = df["season"].astype(int)

    df["pitch_family"] = "other"
    df.loc[df["pitch_type"].isin(FASTBALL_TYPES), "pitch_family"] = "fastball"
    df.loc[df["pitch_type"].isin(BREAKING_TYPES), "pitch_family"] = "breaking"
    df.loc[df["pitch_type"].isin(OFFSPEED_TYPES), "pitch_family"] = "offspeed"

    df["is_2k"]          = df["strikes"] == 2
    df["is_first_pitch"] = (df["balls"] == 0) & (df["strikes"] == 0)

    df["in_zone"]                     = df["zone"].isin(IN_ZONE_CODES)
    df["in_shadow"]                   = df["zone"].isin(SHADOW_CODES)
    df["is_swing"]                    = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"]                    = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_called_strike"]            = df["description"].isin(CALLED_STRIKE_DESCRIPTIONS)
    df["is_chase"]                    = df["is_swing"] & ~df["in_zone"]
    df["is_csw"]                      = df["is_called_strike"] | df["is_whiff"]
    df["is_borderline_called_strike"] = df["is_called_strike"] & df["in_shadow"]

    df["is_breaking"] = df["pitch_family"] == "breaking"
    df["is_fastball"] = df["pitch_family"] == "fastball"
    df["is_offspeed"] = df["pitch_family"] == "offspeed"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core LOCO math: residual catcher delta from cell sums
# ─────────────────────────────────────────────────────────────────────────────
def residuals_from_cell_sums(cell: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    """
    cell: one row per (pitcher, season, stand) with n_p, sum_p
    pc:   one row per (pitcher, season, stand, fielder_2) with n_pc, sum_pc
    Returns: (fielder_2, season, raw_delta, n) — pairs with n_other<50 dropped.
    """
    if len(pc) == 0:
        return pd.DataFrame(columns=["fielder_2", "season", "raw_delta", "n"])

    merged = pc.merge(cell, on=["pitcher", "season", "stand"], how="left")
    merged["n_other"]   = merged["n_p"]   - merged["n_pc"]
    merged["sum_other"] = merged["sum_p"] - merged["sum_pc"]

    merged = merged[(merged["n_other"] >= 50) & (merged["n_pc"] > 0)]
    if len(merged) == 0:
        return pd.DataFrame(columns=["fielder_2", "season", "raw_delta", "n"])

    merged["pair_residual"]  = (merged["sum_pc"] / merged["n_pc"]) \
                             - (merged["sum_other"] / merged["n_other"])
    merged["weighted_resid"] = merged["pair_residual"] * merged["n_pc"]

    out = merged.groupby(["fielder_2", "season"], observed=True).agg(
        sum_weighted=("weighted_resid", "sum"),
        n=("n_pc", "sum"),
    ).reset_index()
    out["raw_delta"] = out["sum_weighted"] / out["n"]
    return out[["fielder_2", "season", "raw_delta", "n"]]


def compute_full_season_residuals(df: pd.DataFrame, metric_col: str,
                                  count_filter: str | None = None) -> pd.DataFrame:
    """Used to fit shrinkage k AND populate the prior-season output file."""
    sub = df
    if count_filter == "2k":
        sub = sub[sub["is_2k"]]
    elif count_filter == "first_pitch":
        sub = sub[sub["is_first_pitch"]]
    if len(sub) == 0:
        return pd.DataFrame(columns=["fielder_2", "season", "raw_delta", "n"])

    cell = sub.groupby(["pitcher", "season", "stand"], observed=True).agg(
        n_p=(metric_col, "size"),
        sum_p=(metric_col, "sum"),
    ).reset_index()
    pc = sub.groupby(["pitcher", "season", "stand", "fielder_2"], observed=True).agg(
        n_pc=(metric_col, "size"),
        sum_pc=(metric_col, "sum"),
    ).reset_index()
    return residuals_from_cell_sums(cell, pc)


# ─────────────────────────────────────────────────────────────────────────────
# As-of computation: walk dates chronologically, snapshot at each date
# ─────────────────────────────────────────────────────────────────────────────
def compute_asof_residuals(df: pd.DataFrame, metric_col: str,
                           count_filter: str | None = None) -> pd.DataFrame:
    """
    Returns one row per (catcher, game_date, season) with raw_delta computed
    using only pitches with game_date STRICTLY BEFORE the snapshot date.

    Walks dates chronologically per season, maintaining cumulative cell
    aggregates. At each date D, snapshots residuals using cumulative-
    through-(D-1) totals. Then adds D's pitches to the running totals.
    """
    sub = df
    if count_filter == "2k":
        sub = sub[sub["is_2k"]]
    elif count_filter == "first_pitch":
        sub = sub[sub["is_first_pitch"]]
    if len(sub) == 0:
        return pd.DataFrame(columns=["fielder_2", "game_date", "season",
                                     "raw_delta", "n"])

    # Daily increments at two grain levels
    daily_cell = sub.groupby(
        ["game_date", "pitcher", "season", "stand"], observed=True
    ).agg(
        d_n_p=(metric_col, "size"),
        d_sum_p=(metric_col, "sum"),
    ).reset_index()
    daily_pc = sub.groupby(
        ["game_date", "pitcher", "season", "stand", "fielder_2"], observed=True
    ).agg(
        d_n_pc=(metric_col, "size"),
        d_sum_pc=(metric_col, "sum"),
    ).reset_index()

    out_chunks = []
    # Per-season independent walk (cumulative aggregates reset across seasons)
    for season, season_grp in daily_pc.groupby("season"):
        season_pc_daily   = season_grp
        season_cell_daily = daily_cell[daily_cell["season"] == season]
        dates = sorted(season_pc_daily["game_date"].unique())

        cell_running = pd.DataFrame(
            columns=["pitcher", "season", "stand", "n_p", "sum_p"]
        )
        pc_running = pd.DataFrame(
            columns=["pitcher", "season", "stand", "fielder_2", "n_pc", "sum_pc"]
        )

        for d in dates:
            # Snapshot BEFORE adding today's pitches
            if len(pc_running) > 0:
                snap = residuals_from_cell_sums(cell_running, pc_running)
                if len(snap) > 0:
                    snap["game_date"] = d
                    out_chunks.append(snap)

            today_cell = season_cell_daily[season_cell_daily["game_date"] == d]
            today_pc   = season_pc_daily[season_pc_daily["game_date"] == d]

            if len(today_cell) > 0:
                add = today_cell.rename(
                    columns={"d_n_p": "n_p", "d_sum_p": "sum_p"}
                )[["pitcher", "season", "stand", "n_p", "sum_p"]]
                cell_running = pd.concat([cell_running, add], ignore_index=True)
                cell_running = cell_running.groupby(
                    ["pitcher", "season", "stand"], observed=True, as_index=False
                ).agg({"n_p": "sum", "sum_p": "sum"})

            if len(today_pc) > 0:
                add = today_pc.rename(
                    columns={"d_n_pc": "n_pc", "d_sum_pc": "sum_pc"}
                )[["pitcher", "season", "stand", "fielder_2", "n_pc", "sum_pc"]]
                pc_running = pd.concat([pc_running, add], ignore_index=True)
                pc_running = pc_running.groupby(
                    ["pitcher", "season", "stand", "fielder_2"], observed=True, as_index=False
                ).agg({"n_pc": "sum", "sum_pc": "sum"})

    if not out_chunks:
        return pd.DataFrame(columns=["fielder_2", "game_date", "season",
                                     "raw_delta", "n"])

    return pd.concat(out_chunks, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical Bayes shrinkage (fit on full-season residuals — most stable)
# ─────────────────────────────────────────────────────────────────────────────
def fit_eb_shrinkage_k(catcher_resid: pd.DataFrame,
                       min_n_for_fit: int = 1000) -> float:
    fit_pool = catcher_resid[catcher_resid["n"] >= min_n_for_fit]
    if len(fit_pool) < 10:
        print(f"      (insufficient catchers with n>={min_n_for_fit}, k=2000 fallback)")
        return 2000.0

    sigma_within_per_pitch = 0.25
    mean_inv_n = (1.0 / fit_pool["n"]).mean()
    obs_var = fit_pool["raw_delta"].var()
    sigma_between = obs_var - mean_inv_n * sigma_within_per_pitch
    if sigma_between <= 0:
        return 10000.0
    k = sigma_within_per_pitch / sigma_between
    return float(np.clip(k, 200.0, 20000.0))


def apply_eb_shrinkage(catcher_resid: pd.DataFrame, k: float) -> pd.Series:
    weight = catcher_resid["n"] / (catcher_resid["n"] + k)
    return weight * catcher_resid["raw_delta"]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
METRIC_SPECS = [
    ("catcher_breaking_pct_delta",            "is_breaking",   None),
    ("catcher_breaking_pct_2k_delta",         "is_breaking",   "2k"),
    ("catcher_breaking_pct_first_pitch_delta","is_breaking",   "first_pitch"),
    ("catcher_zone_pct_delta",                "in_zone",       None),
    ("catcher_chase_pct_delta",               "is_chase",      None),
    ("catcher_csw_pct_delta",                 "is_csw",        None),
    ("catcher_borderline_strike_pct_delta",   "is_borderline_called_strike", None),
]


def build_features(df: pd.DataFrame):
    """Build prior (full-season) and as-of feature frames. Returns (asof, prior)."""

    # Sample sizes for the prior file
    full_counts = df.groupby(["fielder_2", "season"], observed=True)\
                    .size().reset_index(name="n_pitches_caught")

    # Cumulative pitch count per catcher per game-date for the as-of file
    daily_counts = df.groupby(["fielder_2", "season", "game_date"],
                              observed=True).size().reset_index(name="d_n")
    daily_counts = daily_counts.sort_values(["fielder_2", "season", "game_date"])
    # Cumulative through PRIOR date (exclusive of today). Both cumsum AND
    # shift must be per-group, otherwise the first row of a new (catcher,
    # season) group inherits the LAST row of the previous group — making
    # April 1 look like it has a full prior season's pitch count.
    cumsum = daily_counts.groupby(["fielder_2", "season"])["d_n"].cumsum()
    daily_counts["n_pitches_caught_to_date"] = (
        cumsum.groupby([daily_counts["fielder_2"], daily_counts["season"]])
              .shift(1).fillna(0).astype(int)
    )
    asof_counts = daily_counts[["fielder_2", "season", "game_date",
                                "n_pitches_caught_to_date"]]

    prior_chunks = [full_counts]
    asof_chunks  = [asof_counts]

    print("\nBuilding catcher residuals (LOCO + EB shrinkage)...")
    for out_col, metric_col, count_filter in METRIC_SPECS:
        print(f"  {out_col}...")

        full_resid = compute_full_season_residuals(df, metric_col, count_filter)
        if full_resid.empty:
            print(f"    (no full-season data for filter={count_filter})")
            continue
        k = fit_eb_shrinkage_k(full_resid)
        print(f"    k = {k:.0f}  (n catcher-seasons used for fit = {len(full_resid)})")
        full_resid[out_col] = apply_eb_shrinkage(full_resid, k)
        prior_chunks.append(full_resid[["fielder_2", "season", out_col]])

        asof_resid = compute_asof_residuals(df, metric_col, count_filter)
        if asof_resid.empty:
            continue
        asof_resid[out_col] = apply_eb_shrinkage(asof_resid, k)
        asof_chunks.append(asof_resid[["fielder_2", "season", "game_date", out_col]])

    # Combine prior chunks
    prior_out = prior_chunks[0]
    for c in prior_chunks[1:]:
        prior_out = prior_out.merge(c, on=["fielder_2", "season"], how="left")
    for col, _, _ in METRIC_SPECS:
        if col in prior_out.columns:
            prior_out[col] = prior_out[col].fillna(0.0)
    prior_out = prior_out.rename(columns={"fielder_2": "catcher_id"})
    prior_out = prior_out.sort_values(["season", "n_pitches_caught"],
                                      ascending=[True, False]).reset_index(drop=True)

    # Combine as-of chunks
    asof_out = asof_chunks[0]
    for c in asof_chunks[1:]:
        asof_out = asof_out.merge(c, on=["fielder_2", "season", "game_date"],
                                  how="left")
    for col, _, _ in METRIC_SPECS:
        if col in asof_out.columns:
            asof_out[col] = asof_out[col].fillna(0.0)
    asof_out = asof_out.rename(columns={"fielder_2": "catcher_id"})
    asof_out = asof_out.sort_values(["catcher_id", "season", "game_date"])\
                       .reset_index(drop=True)

    return asof_out, prior_out


def main():
    print("=" * 60)
    print("Catcher Game-Calling Feature Builder (as-of, leakage-free)")
    print("=" * 60)

    if not PITCHES_PATH.exists():
        raise FileNotFoundError(f"{PITCHES_PATH} not found.")

    df = load_and_label_pitches(PITCHES_PATH)
    print(f"\nSeasons: {sorted(df['season'].unique())}")
    print(f"Unique catchers: {df['fielder_2'].nunique():,}")
    print(f"Unique pitchers: {df['pitcher'].nunique():,}")
    print(f"Unique game dates: {df['game_date'].nunique():,}")

    asof, prior = build_features(df)

    print(f"\nPrior file: {len(prior):,} (catcher, season) rows")
    print(f"As-of file: {len(asof):,} (catcher, game_date) rows")

    # Sanity check: end-of-completed-season as-of should ≈ prior file.
    # (Tiny diff because last-day pitches are excluded by the strict-< cutoff.)
    seasons = sorted(asof["season"].unique())
    if len(seasons) >= 2:
        s_done = seasons[-2]
        last_d = asof[asof["season"] == s_done]["game_date"].max()
        asof_last = asof[(asof["season"] == s_done) & (asof["game_date"] == last_d)]
        prior_done = prior[prior["season"] == s_done]
        common = sorted(set(asof_last["catcher_id"]) & set(prior_done["catcher_id"]))
        if common:
            cid = common[0]
            ar = asof_last[asof_last["catcher_id"] == cid].iloc[0]
            pr = prior_done[prior_done["catcher_id"] == cid].iloc[0]
            print(f"\nSanity: catcher {cid} season {s_done}:")
            print(f"  End-of-season as-of breaking_delta: {ar['catcher_breaking_pct_delta']:+.4f}")
            print(f"  Prior file       breaking_delta:    {pr['catcher_breaking_pct_delta']:+.4f}")
            print(f"  (small diff expected — last day's pitches excluded by as-of cutoff)")

    asof.to_csv(ASOF_PATH, index=False)
    prior.to_csv(PRIOR_PATH, index=False)
    print(f"\n✓ Wrote {ASOF_PATH} ({len(asof):,} rows)")
    print(f"✓ Wrote {PRIOR_PATH} ({len(prior):,} rows)")

    # Reputational sanity check using full prior file (most stable)
    if len(seasons) >= 2:
        diag_season = seasons[-2]
        s = prior[(prior["season"] == diag_season) &
                  (prior["n_pitches_caught"] >= 2000)]
        print(f"\nTop/bottom catchers (prior-file, season {diag_season}):")
        for col in ["catcher_breaking_pct_delta", "catcher_csw_pct_delta",
                    "catcher_borderline_strike_pct_delta"]:
            if col not in s.columns:
                continue
            print(f"\n  {col}:")
            top = s.nlargest(3, col)[["catcher_id", "n_pitches_caught", col]]
            bot = s.nsmallest(3, col)[["catcher_id", "n_pitches_caught", col]]
            print("    Top 3:")
            print(top.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
            print("    Bottom 3:")
            print(bot.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\nNext: features.py joins via merge_asof on game_date "
          "for the as-of features, blends with prior file by sample size.")


if __name__ == "__main__":
    main()
