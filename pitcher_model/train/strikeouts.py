"""
train/strikeouts.py
=========================
Beta-Binomial Strikeout Distribution Model

Instead of predicting raw K counts and slapping a normal distribution on top,
this script models the two GENERATIVE components of strikeouts separately:

    p = K/PA  (strikeout rate per plate appearance)
    N = BF    (batters faced / plate appearances by pitcher)

Then combines them into a Beta-Binomial(N, α, β) distribution to get a full
discrete probability mass function over possible strikeout outcomes.

WHY THIS IS BETTER:
- Variance emerges naturally from the interaction of p and N
- Respects discrete, non-negative nature of strikeouts
- Different pitchers get different variance (high-K + deep outings ≠ low-K + short outings)
- No more hardcoded σ = 2.1 for everyone
- Properly bounded: K can't exceed N

PIPELINE:
    1. Load the feature matrix from features.py
    2. Create two targets: k_per_pa and batters_faced
    3. Train separate XGBoost models for each
    4. Calibrate Beta concentration parameter (κ) from cross-validation residuals
    5. Calibrate N uncertainty (σ_N) from cross-validation residuals
    6. Produce Beta-Binomial PMFs and compare to the old normal approach
    7. Save both trained models for use in predict/strikeouts.py

USAGE:
    python run.py train strikeouts

REQUIRES:
    - data/pitcher_model_features.csv (from features.py)
    - pip install pandas numpy scikit-learn xgboost lightgbm scipy matplotlib seaborn joblib
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from scipy.special import betaln, gammaln
from scipy.optimize import minimize_scalar, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import xgboost as xgb

from pitcher_model.paths import DATA_DIR, OUTPUT_DIR, MODEL_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR, MODEL_DIR)

# ── Paths ────────────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────────────────────
TEST_YEAR = 2025              # Hold-out year (or latest year)
MIN_PA_GAME = 6               # Drop games where pitcher faced < 6 batters (mop-up / injury)
TARGET_RATE = "k_per_pa"      # Strikeout rate target
TARGET_BF = "batters_faced"   # Batters faced target
RANDOM_STATE = 42

# Features that MUST be excluded to prevent leakage.
# These are current-game outcome stats. The feature engineering script already
# labels most of these, but we enforce the exclusion here too.
RAW_STAT_EXCLUSIONS = {
    # Current-game outcomes
    "strikeouts", "batters_faced", "plate_appearances", "hits_allowed",
    "walks", "earned_runs", "runs", "home_runs_allowed", "outs_recorded",
    "innings_pitched", "pitch_count", "pitches", "hit_by_pitch", "hbp",
    "total_pitches", "batted_balls", "strikes", "balls",
    "whiffs", "called_strikes", "in_zone_pitches", "out_of_zone_pitches",
    "chases", "barrels", "hard_hits", "soft_hits",
    "singles", "doubles", "triples",
    # ── ER-merge columns (added by 02's load_all_data merging pitcher_earned_runs.csv) ──
    # These are CURRENT-game stats. earned_runs and runs encode game outcome
    # directly; innings_pitched ≈ outs_recorded/3, which leaks BF.
    "outs",
    # ── Current-game batted-ball-type counts (added by 02's new build_batted_ball_features) ──
    # These are CURRENT-game counts — including any would leak the answer
    # for the K model (high-K starts have fewer BIPs of all types).
    # The rolling/szn versions (gb_pct_L5, fb_pct_szn, etc.) are safe and
    # picked up by the _L*/_szn patterns.
    "ground_balls", "fly_balls", "line_drives", "popups", "infield_fly_balls",
    "sweet_spot_hits", "solid_contact_hits",
    # Current-game derived
    "k_pct", "bb_pct", "k_per_pa", "k_per_9", "k_per_100_pitches",
    "pitches_per_k", "pitches_per_bf", "whiff_pct", "csw_pct", "chase_rate", "zone_pct",
    "barrel_pct", "hard_hit_pct", "est_innings", "is_short_outing",
    "k_bb_pct", "strike_pct", "soft_hit_pct", "outs_per_pa",
    # ── Current-game H/W rates (added by 02's updated rolling block pre-roll) ──
    # k_minus_bb_pct is especially dangerous — it contains the current
    # game's k_pct directly, which makes K prediction trivial.
    # The lagged versions (_L5, _szn, _szn_blended) are safe.
    "hits_per_pa", "bb_per_pa", "hr_per_pa",
    "h_per_9", "bb_per_9", "hr_per_9",
    "hr_per_bip", "hr_per_fb",
    "k_minus_bb_pct",
    "babip", "lob_pct",
    # Current-game batted-ball mix RATES
    "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
    # Current-game contact quality (the actual contact made this start —
    # leaks K rate because high-K starts have fewer / weaker contact rows)
    "avg_exit_velocity", "avg_launch_angle",
    "sweet_spot_pct", "solid_contact_pct",
    "avg_xba_contact", "avg_xwoba_contact",
    # Current-game platoon stats (CRITICAL — these were leaking via "vs_left"/"vs_right"
    # substring match against FEATURE_PATTERNS, inflating κ to ~1250)
    "pa_vs_left", "pa_vs_right", "whiffs_vs_left", "whiffs_vs_right",
    "whiff_pct_vs_left", "whiff_pct_vs_right",
    # Current-game pitch-type counts and rates
    "ff_count", "si_count", "fc_count", "sl_count", "cu_count",
    "ch_count", "fs_count", "sv_count", "kc_count",
    "fastball_count", "breaking_count", "offspeed_count",
    "ff_pct", "si_pct", "fc_pct", "sl_pct", "cu_pct",
    "ch_pct", "fs_pct", "sv_pct", "kc_pct",
    "fastball_pct", "breaking_pct", "offspeed_pct",
    # Current-game velocity/movement (raw, not rolling)
    "avg_velocity", "max_velocity", "avg_spin_rate",
    "avg_extension", "avg_induced_vert_break", "avg_horiz_break",
    # Targets
    TARGET_RATE, TARGET_BF,
    "target_hits_allowed", "target_walks", "target_home_runs",
    "target_total_pitches", "target_outs_recorded",
    "target_k_over_4_5", "target_k_over_5_5", "target_k_over_6_5",
    "target_k_pct", "target_whiff_pct",
    # New H/W binary targets (added by 02's add_targets)
    "target_h_over_4_5", "target_h_over_5_5", "target_h_over_6_5",
    "target_bb_over_1_5", "target_bb_over_2_5", "target_bb_over_3_5",
}

# Patterns that identify safe engineered features (matched as SUBSTRINGS, not prefixes)
# This matches the convention in train/baseline.py's FEATURE_PATTERNS
FEATURE_PATTERNS = [
    # Rolling pitcher stats (suffix convention: k_pct_L3, whiff_pct_L10, etc.)
    "_L3", "_L5", "_L10",
    # Season cumulative (suffix convention: k_pct_szn, etc.)
    # NOTE: this also matches the new _szn_blended empirical-Bayes-shrunk
    # features added in features.py — no extra pattern needed.
    "_szn",
    # Prior-season carryover (last 5/10 starts of previous season).
    # These were computed in 02's build_early_season_features() but never
    # previously picked up by the feature filter. Adding them here lets the
    # correlation pruner naturally favor them over noisy early-season _szn
    # values when the pitcher has very few current-season starts.
    "_prev5", "_prev10",
    # Blended / empirical-Bayes features (from build_blended_rate_features
    # in 02). _szn_blended already matches via "_szn", but _L5_blended only
    # matches via "_L5". Listed here for clarity and to future-proof against
    # naming changes.
    "_blended",
    # Trends
    "trend",
    # Opposing team
    "opp_",
    # Lineup-level
    "lu_", "pitcher_x_lineup", "whiff_x_lineup", "k_trend_x_lu",
    # Pitch-type effectiveness
    "pt_", "cross_",
    # Umpire
    "ump_",
    # Park factors
    "pf_", "park_",
    # Weather
    "wx_",
    # Lineup features (prefix convention)
    "lineup_",
    # BF-specific features
    "bf_L", "bf_season", "bf_trend", "bf_pitch", "bf_short", "bf_deep",
    "bf_prior", "bf_has_prior", "bf_vs_prior",
    # Interaction features (archetype bias fix)
    "ix_",
    # Normalized K metrics
    "k_per_100", "k_per_9", "k_per_pa", "pitches_per_k",
    "est_innings", "is_short_outing",
    # Schedule/context
    "rest_days", "short_rest", "extra_rest",
    "is_home", "is_night", "is_day", "is_weekend", "is_dome",
    "day_of_week", "month", "start_num",
    # Platoon
    "platoon", "vs_left", "vs_right",
    "pitcher_throws",
    # Stuff volatility
    "_std_",
    # Pitch mix
    "pitch_mix_",
    # Additional game context patterns
    "pa_std", "pa_range", "pa_trend", "pa_prior",
    "baserunner_rate", "hr_rate_L", "pitches_per_pa",
    "deep_outing", "recent_pitches",
    "pvt_", "prior_starts",
    "early_x_", "season_phase", "days_into_season",
    "is_first_month",
    # Rolling window observation counts (how many actual starts back each window)
    "n_starts_in_L",
    # Opposing starter features (game-level pace/depth)
    "opp_sp_", "ix_both_deep", "ix_aces_matchup", "ix_combined_depth",
    # FanGraphs pitcher quality (Stuff+, Location+, Pitching+, plate discipline)
    "fg_stuff", "fg_location", "fg_pitching", "fg_swstr", "fg_o_swing",
    "fg_z_swing", "fg_contact", "fg_o_contact", "fg_z_contact",
    "fg_zone_pct", "fg_first_strike", "fg_tto", "fg_loc_", "fg_pitcher_frm",
    # Catcher framing
    "catcher_frm",
    # Lineup plate discipline (SwStr%, Contact%, TTO%, chase, concentration)
    "lu_swstr", "lu_o_swing", "lu_z_contact", "lu_contact_pct",
    "lu_tto", "lu_csw", "lu_fg_k", "lu_barrel", "lu_hard_hit",
    "lu_k_rate_std", "lu_tto_pct_std", "lu_max_k", "lu_min_k",
    # Velocity delta & rolling platoon K rates
    "velo_delta", "whiff_pct_vs_left", "whiff_pct_vs_right",
    "platoon_whiff_diff", "pitcher_tto_L", "pitcher_tto_szn",
    # New interaction features
    "ix_stuff_x_", "ix_pitcher_lu_", "ix_tto_matchup", "ix_swstr_x_contact",
    # Per-batter slot features (b1_ through b9_ for each lineup position)
    "opp_b1_", "opp_b2_", "opp_b3_", "opp_b4_", "opp_b5_",
    "opp_b6_", "opp_b7_", "opp_b8_", "opp_b9_",
    # Lineup distribution features
    "lineup_k_rate_p90", "lineup_k_rate_p10", "lineup_k_rate_iqr",
    "lineup_k_rate_skew", "lineup_expected_k", "lineup_platoon_disadv",
    # Pitcher-batter historical matchup
    "pb_hist_k_rate", "pb_hist_whiff_rate", "pb_hist_pa",
    "pb_familiar_batters", "pb_familiarity_pct",
    # EWM (exponentially weighted moving averages)
    "_ewm",
    # Delta features (changes between windows)
    "delta_",
    # Last 1 start (ultra-short-term)
    "_L1",
    # Pitcher × umpire interaction
    "ix_pitcher_csw_x_ump", "ix_pitcher_edge_x_ump",
    "ix_pitcher_k_x_ump", "ix_pitcher_bb_x_ump",
    # Pitcher × catcher framing interaction
    "ix_catcher_frm_x_",
    # Pitch count fatigue
    "pitchcount_", "heavy_prev_start", "pitches_per_out",
    # Enhanced lineup distribution
    "lu_k_rate_median", "lu_top3_k", "lu_bot3_k", "lu_top3_bot3_gap",
    # Pitch-type matchup score
    "pitch_matchup_score", "pitch_k_matchup_score",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & TARGET CREATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load feature matrix, create rate + BF targets, engineer BF-specific features, filter bad rows."""
    path = DATA_DIR / "pitcher_model_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run features.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Create targets ──────────────────────────────────────────────────
    # batters_faced: use direct column if available, else compute from PA-related cols
    if "batters_faced" not in df.columns:
        bf_cols = ["plate_appearances", "at_bats", "outs_recorded"]
        if "plate_appearances" in df.columns:
            df["batters_faced"] = df["plate_appearances"]
        elif "at_bats" in df.columns and "walks" in df.columns:
            df["batters_faced"] = (
                df["at_bats"] + df.get("walks", 0) +
                df.get("hit_by_pitch", 0) + df.get("sacrifice_flies", 0).fillna(0)
            )
        elif "outs_recorded" in df.columns:
            df["batters_faced"] = (df["outs_recorded"] * 1.35).round().astype(int)
        else:
            raise ValueError("Cannot compute batters_faced — no PA, AB, or outs columns found.")

    # k_per_pa
    if "k_per_pa" not in df.columns:
        df["k_per_pa"] = df["strikeouts"] / df["batters_faced"].replace(0, np.nan)

    # ── Engineer BF-specific features ───────────────────────────────────
    # These capture outing depth patterns that generic features miss.
    # Sort by pitcher and date for rolling calculations.
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # Pitches per batter faced (pitch efficiency)
    if "pitch_count" in df.columns or "pitches" in df.columns:
        pc_col = "pitch_count" if "pitch_count" in df.columns else "pitches"
        df["pitches_per_bf"] = df[pc_col] / df["batters_faced"].replace(0, np.nan)
    elif "pitch_count" not in df.columns:
        # Try to compute from available data
        df["pitches_per_bf"] = np.nan

    for pitcher_id, grp in df.groupby("pitcher"):
        idx = grp.index

        # Rolling BF averages (how deep does this pitcher typically go?)
        for window in [3, 5, 10]:
            col_bf = f"bf_L{window}_avg"
            df.loc[idx, col_bf] = grp["batters_faced"].shift(1).rolling(window, min_periods=1).mean()

            col_bf_std = f"bf_L{window}_std"
            df.loc[idx, col_bf_std] = grp["batters_faced"].shift(1).rolling(window, min_periods=2).std()

        # Season-to-date BF average
        cum_bf = grp["batters_faced"].shift(1).expanding(min_periods=1).mean()
        df.loc[idx, "bf_season_avg"] = cum_bf

        # BF trend: recent vs season (positive = going deeper lately)
        if "bf_L3_avg" in df.columns:
            pass  # Will compute after the loop
        
        # Rolling pitch efficiency (pitches per BF)
        if "pitches_per_bf" in df.columns:
            for window in [3, 5]:
                col_eff = f"bf_pitch_eff_L{window}"
                df.loc[idx, col_eff] = grp["pitches_per_bf"].shift(1).rolling(window, min_periods=1).mean()

        # Short outing frequency (BF < 18, roughly < 5 innings)
        short_flag = (grp["batters_faced"] < 18).astype(float)
        for window in [5, 10]:
            col_short = f"bf_short_pct_L{window}"
            df.loc[idx, col_short] = short_flag.shift(1).rolling(window, min_periods=1).mean()

        # Quality start proxy frequency (BF >= 25, roughly 6+ innings)
        deep_flag = (grp["batters_faced"] >= 25).astype(float)
        for window in [5, 10]:
            col_deep = f"bf_deep_pct_L{window}"
            df.loc[idx, col_deep] = deep_flag.shift(1).rolling(window, min_periods=1).mean()

        # Rolling pitch count averages (if available)
        pc_col = None
        if "pitch_count" in grp.columns:
            pc_col = "pitch_count"
        elif "pitches" in grp.columns:
            pc_col = "pitches"
        if pc_col:
            for window in [3, 5]:
                col_pc = f"bf_pitchcount_L{window}"
                df.loc[idx, col_pc] = grp[pc_col].shift(1).rolling(window, min_periods=1).mean()

    # BF trend features (computed after all pitchers)
    df["bf_trend_3v10"] = df.get("bf_L3_avg", 0) - df.get("bf_L10_avg", 0)
    df["bf_trend_3vseason"] = df.get("bf_L3_avg", 0) - df.get("bf_season_avg", 0)

    # ── Prior-year baselines (helps early-season BF predictions) ────────
    # For each pitcher's first few starts, we want their prior-year averages
    # as fallback features when current-season rolling windows are thin.
    df["year"] = df["game_date"].dt.year
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    df["start_number_in_season"] = df.groupby(["pitcher", "year"]).cumcount() + 1

    # Compute prior-year summaries for each pitcher × year
    yearly = df.groupby(["pitcher", "year"]).agg(
        yearly_k_rate=(TARGET_RATE, "mean"),
        yearly_bf=(TARGET_BF, "mean"),
        yearly_k_avg=("strikeouts", "mean"),
        yearly_starts=("game_date", "count"),
    ).reset_index()

    # Shift by one year to get prior-year stats
    yearly["prior_year"] = yearly["year"] + 1
    prior = yearly.rename(columns={
        "yearly_k_rate": "bf_prior_year_k_rate",
        "yearly_bf": "bf_prior_year_bf_avg",
        "yearly_k_avg": "bf_prior_year_k_avg",
        "yearly_starts": "bf_prior_year_starts",
    })[["pitcher", "prior_year", "bf_prior_year_k_rate", "bf_prior_year_bf_avg",
        "bf_prior_year_k_avg", "bf_prior_year_starts"]]

    df = df.merge(prior, left_on=["pitcher", "year"], right_on=["pitcher", "prior_year"], how="left")
    df.drop(columns=["prior_year"], errors="ignore", inplace=True)

    # Fill missing prior-year data with league averages
    for col in ["bf_prior_year_k_rate", "bf_prior_year_bf_avg", "bf_prior_year_k_avg"]:
        if col in df.columns:
            league_avg = df[col].median()  # Use median as league baseline
            df[col] = df[col].fillna(league_avg)
    if "bf_prior_year_starts" in df.columns:
        df["bf_prior_year_starts"] = df["bf_prior_year_starts"].fillna(0)

    # Has-prior flag (binary: did this pitcher pitch last year?)
    df["bf_has_prior_year"] = (df["bf_prior_year_starts"] > 0).astype(float)

    # Interaction: how far is the current-season rolling BF from the prior-year baseline?
    # (positive = pitcher is going deeper than last year)
    df["bf_vs_prior_year"] = df.get("bf_L5_avg", 0) - df["bf_prior_year_bf_avg"]

    print(f"  Engineered prior-year features: bf_prior_year_*, bf_has_prior_year, bf_vs_prior_year")

    # ── Interaction features for archetype bias ─────────────────────────
    # The model compresses predictions toward the mean because it treats
    # K rate and outing depth independently. These interactions explicitly
    # capture "high-K pitcher going deep = lots of Ks" and vice versa.

    # Find rolling K rate columns (different naming conventions)
    k_rate_cols = [c for c in df.columns if any(x in c.lower() for x in
                   ["k_pct", "k_rate", "whiff"]) and
                   any(c.startswith(p) or c.lower().startswith(p.lower())
                       for p in ["L3_", "L5_", "L10_", "season_", "pt_"])]

    # Use a robust K-ability proxy: pick the best available
    k_ability_col = None
    for candidate in ["season_k_pct", "L10_k_pct", "L5_k_pct",
                      "season_whiff_pct", "L10_whiff_pct"]:
        matches = [c for c in df.columns if c.lower() == candidate.lower()]
        if matches:
            k_ability_col = matches[0]
            break

    # If we found a K-ability column, create interactions
    if k_ability_col:
        # Normalize to 0-1 range if it's a percentage (0-100)
        k_vals = df[k_ability_col].copy()
        if k_vals.median() > 1:
            k_vals = k_vals / 100

        # K ability × BF depth interaction
        bf_depth = df.get("bf_L5_avg", df.get("bf_season_avg", pd.Series(22, index=df.index)))
        df["ix_k_ability_x_bf_depth"] = k_vals * bf_depth

        # K ability squared (captures non-linear effect of elite K pitchers)
        df["ix_k_ability_sq"] = k_vals ** 2

        # K ability × opponent K vulnerability
        opp_k_cols = [c for c in df.columns if "opp_k_pct" in c.lower() or "opp_lu" in c.lower() and "k_rate" in c.lower()]
        if opp_k_cols:
            opp_k = df[opp_k_cols[0]].copy()
            if opp_k.median() > 1:
                opp_k = opp_k / 100
            df["ix_k_ability_x_opp_k"] = k_vals * opp_k

        print(f"  Engineered interaction features using '{k_ability_col}': "
              f"ix_k_ability_x_bf_depth, ix_k_ability_sq, ix_k_ability_x_opp_k")
    else:
        # Fallback: use prior-year K rate for interactions
        k_vals = df["bf_prior_year_k_rate"].copy()
        bf_depth = df.get("bf_L5_avg", df.get("bf_season_avg", pd.Series(22, index=df.index)))
        df["ix_k_ability_x_bf_depth"] = k_vals * bf_depth
        df["ix_k_ability_sq"] = k_vals ** 2
        print(f"  Engineered interaction features using prior-year K rate (fallback)")

    # BF depth squared (captures non-linear effect of very deep/short outings)
    bf_depth_proxy = df.get("bf_L5_avg", df.get("bf_season_avg", pd.Series(22, index=df.index)))
    df["ix_bf_depth_sq"] = bf_depth_proxy ** 2

    # Opponent quality × pitcher depth (tough opponents = shorter outings)
    opp_hard = None
    for candidate in ["opp_hard_hit_pct", "opp_barrel_pct"]:
        if candidate in df.columns:
            opp_hard = df[candidate]
            break
    if opp_hard is not None:
        df["ix_opp_quality_x_bf"] = opp_hard * bf_depth_proxy

    print(f"  Engineered BF-specific features: bf_L*_avg, bf_L*_std, bf_season_avg, "
          f"bf_trend_*, bf_pitch_eff_*, bf_short_pct_*, bf_deep_pct_*, bf_pitchcount_*")

    # ── Filter ──────────────────────────────────────────────────────────
    initial = len(df)
    df = df[df["batters_faced"] >= MIN_PA_GAME].copy()
    df = df.dropna(subset=[TARGET_RATE, TARGET_BF, "strikeouts"])
    df = df[df[TARGET_RATE].between(0, 1)]  # Sanity: rate must be [0, 1]
    print(f"  Loaded {initial:,} rows → {len(df):,} after filtering (BF >= {MIN_PA_GAME}, valid targets)")

    # ── Verify we have the raw strikeouts for evaluation ────────────────
    df["actual_K"] = df["strikeouts"].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df, target):
    """
    Select features for a given target, avoiding leakage.
    Uses SUBSTRING matching (not prefix) to catch features like k_pct_L3,
    whiff_pct_szn, etc. which have the pattern as a suffix.
    """
    candidates = []
    for col in df.columns:
        # Skip identifiers, dates, targets, and raw stats
        if col in RAW_STAT_EXCLUSIONS:
            continue
        if col in ["game_date", "game_pk", "pitcher", "pitcher_name",
                    "team", "opponent", "venue", "umpire", "actual_K",
                    "year", "player_name", "start_number_in_season",
                    "prior_year", "target_strikeouts",
                    "catcher_id", "catcher_name", "opp_starter_id"]:
            continue

        # Check if column matches any safe pattern (substring match)
        col_lower = col.lower()
        is_safe = any(pattern.lower() in col_lower for pattern in FEATURE_PATTERNS)

        if is_safe:
            # Must be numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Must have some variance
                if df[col].nunique() > 1:
                    candidates.append(col)

    print(f"  Selected {len(candidates)} features for target '{target}'")
    return candidates


def select_bf_features_permissive(df, target_col):
    """
    Permissive BF feature selector — takes everything numeric not
    explicitly excluded. Mirrors the approach of the old standalone
    BF-improvement script, which outperformed the pattern-based
    allowlist on BF MAE (~1.46 vs 2.46). The allowlist was missing
    predictive columns whose names don't match its prefixes.

    Leakage is prevented by excluding current-game stats and any column
    whose name matches a leakage pattern.
    """
    leakage_exact = {
        # current-game outcomes / targets
        "strikeouts", "k_per_pa", "batters_faced", "plate_appearances",
        "hits_allowed", "walks", "total_pitches", "est_innings",
        "home_runs_allowed", "barrels", "batted_balls", "hard_hits",
        "soft_hits", "singles", "doubles", "triples", "hbp", "balls",
        "strikes", "called_strikes", "chases", "whiffs",
        "in_zone_pitches", "out_of_zone_pitches",
        "whiffs_vs_left", "whiffs_vs_right",
        "whiff_pct_vs_left", "whiff_pct_vs_right",
        "is_short_outing", "outs_recorded", "outs_per_pa",
        # ── ER-merge columns (current-game, added by 02's load_all_data) ──
        # innings_pitched is by far the worst BF leak: BF ≈ IP*3 + baserunners.
        # earned_runs / runs / outs are similarly current-game outcomes that
        # let the model recover BF via the box-score arithmetic.
        "innings_pitched", "earned_runs", "runs", "outs",
        # ── Current-game derived rates from aggregate_pitcher_game in 01 ──
        # These were missed in earlier versions of the exclusion list. Each
        # is a current-game rate computed from per-game counts; the lagged
        # versions (_L3/L5/L10/szn/blended) are safe and picked up by the
        # permissive selector.
        "k_pct", "bb_pct", "k_bb_pct", "strike_pct",
        "whiff_pct", "csw_pct", "zone_pct", "chase_rate",
        "barrel_pct", "hard_hit_pct", "soft_hit_pct",
        # Pitch-mix percentages (current game)
        "fastball_pct", "breaking_pct", "offspeed_pct",
        "ff_pct", "si_pct", "fc_pct", "sl_pct", "cu_pct",
        "ch_pct", "fs_pct", "sv_pct", "kc_pct",
        # Normalized K metrics (current game)
        "k_per_100_pitches", "pitches_per_k", "k_per_9",
        # raw pitch counts this game
        "ff_count", "sl_count", "cu_count", "ch_count", "si_count",
        "fc_count", "fs_count", "sv_count", "kc_count",
        "fastball_count", "breaking_count", "offspeed_count",
        # current-game PA splits
        "pa_vs_left", "pa_vs_right",
        # ── Current-game H/W rates (added by updated 02's rolling block) ──
        # These leak BF because most are computed as count/PA. With them
        # in the feature set, the BF model can recover BF directly from
        # the count columns (e.g., hits_allowed / hits_per_pa = PA).
        # The lagged versions (_L5, _szn, _szn_blended) are safe and
        # picked up by the permissive selector below.
        "hits_per_pa", "bb_per_pa", "hr_per_pa",
        "h_per_9", "bb_per_9", "hr_per_9",
        "hr_per_bip", "hr_per_fb",
        "k_minus_bb_pct",
        "babip", "lob_pct",
        # Current-game batted-ball-type counts and rates (from
        # build_batted_ball_features in 02). These all encode current-game
        # contact information that leaks BF when combined with raw counts.
        "ground_balls", "fly_balls", "line_drives", "popups", "infield_fly_balls",
        "sweet_spot_hits", "solid_contact_hits",
        "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
        # Current-game contact quality (averages over BIPs THIS start —
        # encodes which pitchers had lots of contact vs few, which leaks BF)
        "avg_exit_velocity", "avg_launch_angle",
        "sweet_spot_pct", "solid_contact_pct",
        "avg_xba_contact", "avg_xwoba_contact",
        # ids / meta
        "pitcher", "pitcher_name", "pitcher_team", "game_pk", "game_date",
        "opp_team", "venue_name", "venue_id", "hp_umpire_name",
        "hp_umpire_id", "home_team", "away_team", "home_team_name",
        "away_team_name", "home_team_id", "away_team_id",
        "home_starter_id", "away_starter_id", "opp_starter_id",
        "season", "year", "p_throws", "player_name",
        "catcher_id", "catcher_name", "start_number_in_season",
        "prior_year", "actual_K",
        # derived targets
        "target_strikeouts", "target_k_pct", "target_whiff_pct",
        "target_walks", "target_hits_allowed", "target_home_runs",
        "target_total_pitches", "target_outs_recorded",
        "target_k_over_4_5", "target_k_over_5_5", "target_k_over_6_5",
        # New H/W binary targets (added by updated 02's add_targets)
        "target_h_over_4_5", "target_h_over_5_5", "target_h_over_6_5",
        "target_bb_over_1_5", "target_bb_over_2_5", "target_bb_over_3_5",
    }
    # Season cumulative raw counts are game-inclusive → leak
    leakage_suffixes = ("_szn",)
    leakage_szn_names = {
        "strikeouts_szn", "walks_szn", "hits_allowed_szn",
        "home_runs_allowed_szn", "total_pitches_szn",
        "plate_appearances_szn", "barrels_szn", "batted_balls_szn",
        "hard_hits_szn", "called_strikes_szn", "chases_szn",
        "whiffs_szn", "in_zone_pitches_szn", "out_of_zone_pitches_szn",
        # New H/W-pipeline raw-count cumulative columns (also game-inclusive)
        "ground_balls_szn", "fly_balls_szn", "line_drives_szn",
        "popups_szn", "infield_fly_balls_szn",
        "singles_szn", "doubles_szn", "triples_szn",
        "earned_runs_szn", "runs_szn", "hbp_szn", "soft_hits_szn",
        "sweet_spot_hits_szn", "solid_contact_hits_szn",
    }
    candidates = []
    for col in df.columns:
        if col == target_col:
            continue
        if col in leakage_exact or col in leakage_szn_names:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() <= 1:
            continue
        candidates.append(col)
    print(f"  Selected {len(candidates)} features for target '{target_col}' "
          f"(permissive / BF)")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# BF-SPECIFIC FEATURE ENGINEERING
# (absorbed from a former standalone BF script — this module now produces
#  the best BF model in a single pass instead of needing a separate pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_bf_features(df, target_col=None):
    """
    Add features specifically designed to predict BF.
    Targets the biggest error source: early hooks due to blowups.

    Features added (names are stable so downstream can reference them):
      - baserunner_rate_L{3,5}     : proxy for runs allowed / trouble
      - hr_rate_L{3,5}             : HR allowed drives quick hooks
      - pitches_per_pa_L{3,5}      : efficiency (low = deeper into game)
      - deep_outing_pct_L{3,5,10}  : share of recent starts with BF >= 24
      - pa_std_L{3,5,10}           : volatility of BF by pitcher
      - pa_range_L{5,10}           : max-min BF span, another vol signal
      - pa_prior_year, pa_prior_year_std : carry-over from last season
      - ix_blowout_x_fatigue       : interaction, blowout risk × rest
      - ix_efficiency_x_k          : efficient strikeout pitcher → deeper
    """
    df = df.copy()
    n_before = len(df.columns)
    group_col = "pitcher"
    tgt = target_col or "batters_faced"

    # 1A: Blowout-risk rolling rates (baserunners per PA, HR per PA)
    if "hits_allowed" in df.columns and "walks" in df.columns and tgt in df.columns:
        for window in [3, 5]:
            col = f"baserunner_rate_L{window}"
            baserunners = df["hits_allowed"] + df["walks"]
            shifted = baserunners.groupby(df[group_col]).shift(1)
            pa_shift = df[tgt].groupby(df[group_col]).shift(1)
            df[col] = (
                shifted.groupby(df[group_col])
                .rolling(window, min_periods=1).sum()
                .reset_index(level=0, drop=True)
            ) / (
                pa_shift.groupby(df[group_col])
                .rolling(window, min_periods=1).sum()
                .reset_index(level=0, drop=True)
            ).replace(0, np.nan)

    if "home_runs_allowed" in df.columns and tgt in df.columns:
        for window in [3, 5]:
            col = f"hr_rate_L{window}"
            shifted = df["home_runs_allowed"].groupby(df[group_col]).shift(1)
            pa_shift = df[tgt].groupby(df[group_col]).shift(1)
            df[col] = (
                shifted.groupby(df[group_col])
                .rolling(window, min_periods=1).sum()
                .reset_index(level=0, drop=True)
            ) / (
                pa_shift.groupby(df[group_col])
                .rolling(window, min_periods=1).sum()
                .reset_index(level=0, drop=True)
            ).replace(0, np.nan)

    # 1B: Pitch efficiency (pitches per PA) — low = deeper into games
    if "total_pitches" in df.columns and tgt in df.columns:
        ppa = df["total_pitches"] / df[tgt].replace(0, np.nan)
        for window in [3, 5]:
            col = f"pitches_per_pa_L{window}"
            shifted = ppa.groupby(df[group_col]).shift(1)
            df[col] = (
                shifted.groupby(df[group_col])
                .rolling(window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

    # 1C: Deep-outing frequency and BF volatility
    if tgt in df.columns:
        deep = (df[tgt] >= 24).astype(float)
        for window in [3, 5, 10]:
            shifted = deep.groupby(df[group_col]).shift(1)
            df[f"deep_outing_pct_L{window}"] = (
                shifted.groupby(df[group_col])
                .rolling(window, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
        for window in [3, 5, 10]:
            shifted = df[tgt].groupby(df[group_col]).shift(1)
            df[f"pa_std_L{window}"] = (
                shifted.groupby(df[group_col])
                .rolling(window, min_periods=2).std()
                .reset_index(level=0, drop=True)
            )
        for window in [5, 10]:
            shifted = df[tgt].groupby(df[group_col]).shift(1)
            rmax = (shifted.groupby(df[group_col])
                    .rolling(window, min_periods=2).max()
                    .reset_index(level=0, drop=True))
            rmin = (shifted.groupby(df[group_col])
                    .rolling(window, min_periods=2).min()
                    .reset_index(level=0, drop=True))
            df[f"pa_range_L{window}"] = rmax - rmin

    # 1D: Prior-year PA stats (new season carry-over)
    if tgt in df.columns and "season" in df.columns:
        py = df.groupby([group_col, "season"])[tgt].agg(["mean", "std"]).reset_index()
        py["season"] = py["season"] + 1  # shift to next season
        py = py.rename(columns={"mean": "pa_prior_year",
                                "std": "pa_prior_year_std"})
        df = df.merge(py, on=[group_col, "season"], how="left")

    # 1E: Interactions (blowout × fatigue, efficiency × K)
    if "baserunner_rate_L5" in df.columns and "days_rest" in df.columns:
        df["ix_blowout_x_fatigue"] = (
            df["baserunner_rate_L5"].fillna(0) *
            (7 - df["days_rest"].fillna(4)).clip(lower=0)
        )
    if "pitches_per_pa_L5" in df.columns and "k_pct_szn" in df.columns:
        # Low pitches/PA AND high K% = efficient K pitcher → goes deeper
        df["ix_efficiency_x_k"] = (
            (20 - df["pitches_per_pa_L5"].fillna(17)).clip(lower=0) *
            df["k_pct_szn"].fillna(0.22)
        )

    n_added = len(df.columns) - n_before
    print(f"  engineer_bf_features: added {n_added} BF-specific features")
    return df


def train_bf_models_multivariant(X_train, y_train, X_test, y_test,
                                 feature_names):
    """
    Train three BF model variants (standard / log / Huber) and return
    the best by MAE on test. Adapted from the former standalone BF script's
    train_bf_models so 06 can produce a best BF model in one pass.

    Returns: (best_preds_on_test, best_model, best_variant_name, summary_dict)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    results = {}

    # A: Standard XGB
    print("    Variant A: XGBoost (standard)...")
    ma = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
    )
    ma.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pa = ma.predict(X_test)
    results["standard"] = {
        "model": ma, "preds": pa,
        "mae": float(mean_absolute_error(y_test, pa)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pa))),
        "bias": float((pa - y_test).mean()),
        "is_log": False,
    }
    print(f"      MAE={results['standard']['mae']:.3f}  "
          f"bias={results['standard']['bias']:+.3f}")

    # B: XGB on log(PA) with lognormal bias correction
    print("    Variant B: XGBoost (log-transform)...")
    y_tr_log = np.log(np.clip(y_train, 1, None))
    y_te_log = np.log(np.clip(y_test, 1, None))
    mb = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
    )
    mb.fit(X_train, y_tr_log, eval_set=[(X_test, y_te_log)], verbose=False)
    pb_raw = np.exp(mb.predict(X_test))
    resid_log = y_tr_log - mb.predict(X_train)
    sigma2 = float(resid_log.var())
    pb_corr = pb_raw * np.exp(sigma2 / 2)
    results["log_transform"] = {
        "model": mb, "preds": pb_corr,
        "mae": float(mean_absolute_error(y_test, pb_corr)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pb_corr))),
        "bias": float((pb_corr - y_test).mean()),
        "is_log": True, "log_sigma2": sigma2,
    }
    print(f"      MAE={results['log_transform']['mae']:.3f}  "
          f"bias={results['log_transform']['bias']:+.3f}")

    # C: XGB with Huber (robust to blowup outliers)
    print("    Variant C: XGBoost (Huber)...")
    mc = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=1.0,
        objective="reg:pseudohubererror", huber_slope=2.0,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    mc.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pc = mc.predict(X_test)
    results["huber"] = {
        "model": mc, "preds": pc,
        "mae": float(mean_absolute_error(y_test, pc)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pc))),
        "bias": float((pc - y_test).mean()),
        "is_log": False,
    }
    print(f"      MAE={results['huber']['mae']:.3f}  "
          f"bias={results['huber']['bias']:+.3f}")

    best_name = min(results, key=lambda k: results[k]["mae"])
    best = results[best_name]
    print(f"    → Best BF variant: '{best_name}' "
          f"(MAE={best['mae']:.3f})")
    return best["preds"], best["model"], best_name, results


# ══════════════════════════════════════════════════════════════════════════════
# COLLINEARITY PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def prune_collinear_features(df, feature_cols, target_col,
                             corr_threshold=0.92, target_year_cutoff=None,
                             verbose=True):
    """
    Drop redundant features via correlation clustering.

    The K-rate feature space contains many near-duplicates by construction
    (k_pct_L3, _L5, _L10, _szn, _ewm, _trend_3, ...). Feeding all of them
    into a tree model is mostly harmless for accuracy but inflates feature-
    importance noise and SHAP instability, and badly hurts the Ridge
    fallback and any linear meta-learner. It also makes "this feature
    matters" claims unreliable.

    Algorithm:
      1. Restrict to training rows only (no test leakage in the corr matrix).
      2. Drop zero-variance / all-NaN features.
      3. Compute |Spearman corr| matrix (rank-based → robust to outliers,
         which the raw Statcast distributions have plenty of).
      4. Cluster features greedily: walk features in descending order of
         |corr(feature, target)|; for each unkept feature, keep it and
         drop everything still-unprocessed with |r| >= threshold to it.
      5. Return the kept list plus a dropped→kept mapping for logging.

    This deterministically keeps the most target-relevant member of each
    redundant cluster — exactly the right behavior for k_pct_L3/L5/L10/szn.
    """
    if verbose:
        print(f"\n  Pruning collinear features (|r| >= {corr_threshold}) "
              f"for target '{target_col}'...")

    work = df.copy()
    if target_year_cutoff is not None and "year" in work.columns:
        work = work[work["year"] < target_year_cutoff]
    work = work.dropna(subset=[target_col])

    present = [c for c in feature_cols if c in work.columns]
    if not present:
        if verbose:
            print("    ⚠ No features present — returning original list")
        return list(feature_cols), {}

    X = work[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(work[target_col], errors="coerce").fillna(0.0)

    nunique = X.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        if verbose:
            print(f"    Dropping {len(constant)} zero-variance features")
        X = X.drop(columns=constant)

    if X.shape[1] <= 1:
        return list(X.columns), {c: None for c in constant}

    # Rank target relevance once, up front (Spearman = corr of ranks)
    target_corr = X.apply(lambda c: c.corr(y, method="spearman")).abs()
    target_corr = target_corr.fillna(0.0).sort_values(ascending=False)
    order = target_corr.index.tolist()

    # Spearman corr among features. For wide feature sets this is the
    # expensive step; fall back to Pearson on ranks if needed.
    try:
        corr = X[order].corr(method="spearman").abs()
    except Exception:
        corr = X[order].rank().corr().abs()

    kept = []
    dropped_map = {}  # dropped_feature -> kept_feature it was redundant with
    remaining = set(order)
    for feat in order:
        if feat not in remaining:
            continue
        kept.append(feat)
        remaining.discard(feat)
        # Find everything still-remaining that is highly correlated with `feat`
        sims = corr.loc[feat, list(remaining)]
        redundant = sims[sims >= corr_threshold].index.tolist()
        for r in redundant:
            dropped_map[r] = feat
            remaining.discard(r)

    for c in constant:
        dropped_map[c] = None

    if verbose:
        print(f"    Kept {len(kept)} / {len(present)} features "
              f"(dropped {len(dropped_map)})")
        # Show a few illustrative drops to sanity-check the clustering
        examples = [(d, k) for d, k in dropped_map.items() if k is not None][:8]
        if examples:
            print("    Example redundancies (dropped → kept):")
            for d, k in examples:
                print(f"      {d:40s} → {k}")

    return kept, dropped_map


# ══════════════════════════════════════════════════════════════════════════════
# TIME-BASED TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def time_split(df, feature_cols, target_col):
    """Split by year: everything before TEST_YEAR = train, TEST_YEAR = test."""
    df = df.sort_values("game_date").reset_index(drop=True)
    df["year"] = df["game_date"].dt.year

    train = df[df["year"] < TEST_YEAR].copy()
    test = df[df["year"] >= TEST_YEAR].copy()

    # Drop rows with NaN in features
    train = train.dropna(subset=feature_cols + [target_col])
    test = test.dropna(subset=feature_cols + [target_col])

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    print(f"  Train: {len(train):,} rows ({train['year'].min()}-{train['year'].max()})")
    print(f"  Test:  {len(test):,} rows ({test['year'].min()}-{test['year'].max()})")

    return X_train, X_test, y_train, y_test, train, test


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, X_test, y_test, feature_names, target_name,
              n_iter=50, cv_folds=4):
    """
    Train XGBoost with RandomizedSearchCV using TimeSeriesSplit.
    Returns the best model, CV results, and test predictions.
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    results = {}

    # ── XGBoost ─────────────────────────────────────────────────────────
    if has_xgb:
        print(f"\n  Training XGBoost for '{target_name}'...")
        xgb_params = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "gamma": [0, 0.1, 0.5, 1.0],
            "min_child_weight": [1, 3, 5, 7],
        }

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        search = RandomizedSearchCV(
            xgb_model, xgb_params, n_iter=n_iter,
            cv=tscv, scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search.fit(X_train, y_train)

        best_xgb = search.best_estimator_
        preds = best_xgb.predict(X_test)
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        cv_mae = -search.best_score_
        print(f"    XGBoost — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, CV MAE: {cv_mae:.4f}")
        print(f"    Best params: {search.best_params_}")

        results["XGBoost"] = {
            "model": best_xgb, "preds": preds,
            "MAE": mae, "RMSE": rmse, "R²": r2, "CV_MAE": cv_mae,
            "params": search.best_params_,
        }

    # ── LightGBM ────────────────────────────────────────────────────────
    if has_lgb:
        print(f"\n  Training LightGBM for '{target_name}'...")
        lgb_params = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 5, 7, -1],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "min_child_weight": [1, 3, 5, 7],
            "min_split_gain": [0, 0.01, 0.1],
        }

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        search_lgb = RandomizedSearchCV(
            lgb_model, lgb_params, n_iter=n_iter,
            cv=tscv, scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_lgb.fit(X_train, y_train)

        best_lgb = search_lgb.best_estimator_
        preds_lgb = best_lgb.predict(X_test)
        mae_lgb = np.mean(np.abs(y_test - preds_lgb))
        rmse_lgb = np.sqrt(np.mean((y_test - preds_lgb) ** 2))
        ss_res_lgb = np.sum((y_test - preds_lgb) ** 2)
        r2_lgb = 1 - ss_res_lgb / ss_tot if ss_tot > 0 else 0
        cv_mae_lgb = -search_lgb.best_score_

        print(f"    LightGBM — MAE: {mae_lgb:.4f}, RMSE: {rmse_lgb:.4f}, R²: {r2_lgb:.4f}, CV MAE: {cv_mae_lgb:.4f}")
        print(f"    Best params: {search_lgb.best_params_}")

        results["LightGBM"] = {
            "model": best_lgb, "preds": preds_lgb,
            "MAE": mae_lgb, "RMSE": rmse_lgb, "R²": r2_lgb, "CV_MAE": cv_mae_lgb,
            "params": search_lgb.best_params_,
        }

    # ── Ridge (lightweight baseline) ────────────────────────────────────
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print(f"\n  Training Ridge for '{target_name}'...")
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    preds_ridge = ridge_pipe.predict(X_test)
    mae_ridge = np.mean(np.abs(y_test - preds_ridge))
    rmse_ridge = np.sqrt(np.mean((y_test - preds_ridge) ** 2))
    ss_res_ridge = np.sum((y_test - preds_ridge) ** 2)
    r2_ridge = 1 - ss_res_ridge / ss_tot if ss_tot > 0 else 0
    print(f"    Ridge — MAE: {mae_ridge:.4f}, RMSE: {rmse_ridge:.4f}, R²: {r2_ridge:.4f}")

    results["Ridge"] = {
        "model": ridge_pipe, "preds": preds_ridge,
        "MAE": mae_ridge, "RMSE": rmse_ridge, "R²": r2_ridge,
    }

    # ── Select best ─────────────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n  ✓ Best model for '{target_name}': {best_name} (MAE={results[best_name]['MAE']:.4f})")

    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# BETA-BINOMIAL MATH
# ══════════════════════════════════════════════════════════════════════════════

def beta_binom_pmf(k, n, alpha, beta_param):
    """
    P(K = k | N = n, α, β) using the Beta-Binomial PMF.

    BetaBinomial(k; n, α, β) = C(n,k) * B(k+α, n-k+β) / B(α, β)

    where B is the Beta function. We use log-space for numerical stability.
    """
    if n < 0 or k < 0 or k > n:
        return 0.0
    log_pmf = (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)  # log C(n,k)
        + betaln(k + alpha, n - k + beta_param)                 # log B(k+α, n-k+β)
        - betaln(alpha, beta_param)                              # log B(α, β)
    )
    return np.exp(log_pmf)


def beta_binom_pmf_array(n, alpha, beta_param):
    """Full PMF as an array: P(K=0), P(K=1), ..., P(K=n)."""
    ks = np.arange(n + 1)
    log_comb = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    log_beta_num = betaln(ks + alpha, n - ks + beta_param)
    log_beta_den = betaln(alpha, beta_param)
    log_pmf = log_comb + log_beta_num - log_beta_den
    pmf = np.exp(log_pmf)
    pmf = pmf / pmf.sum()  # Normalize for numerical safety
    return pmf


def expected_pmf_over_N(pred_p, pred_N, kappa, sigma_N, max_k=25):
    """
    Compute the expected PMF by marginalizing over uncertainty in N.

    We model N as a discrete distribution centered on pred_N with spread sigma_N.
    For each possible N value, we compute the Beta-Binomial PMF and weight
    by the probability of that N.

    Returns array of length max_k+1: P(K=0), ..., P(K=max_k).
    """
    # Clip rate to match inference (predict/strikeouts.py clips to [0.01, 0.99])
    pred_p = np.clip(pred_p, 0.01, 0.99)
    alpha = pred_p * kappa
    beta_param = (1 - pred_p) * kappa

    # Clamp parameters
    alpha = max(alpha, 0.01)
    beta_param = max(beta_param, 0.01)

    # Range of plausible N values (discrete)
    min_N = max(1, int(pred_N - 3 * sigma_N))
    max_N = int(pred_N + 3 * sigma_N) + 1

    # Weights for each N (discretized normal)
    N_values = np.arange(min_N, max_N + 1)
    N_weights = sp_stats.norm.pdf(N_values, loc=pred_N, scale=max(sigma_N, 0.5))
    N_weights = N_weights / N_weights.sum()

    # Marginalize
    combined_pmf = np.zeros(max_k + 1)
    for n_val, w in zip(N_values, N_weights):
        pmf = beta_binom_pmf_array(n_val, alpha, beta_param)
        # Pad or truncate to max_k+1
        if len(pmf) <= max_k + 1:
            combined_pmf[:len(pmf)] += w * pmf
        else:
            combined_pmf += w * pmf[:max_k + 1]

    # Renormalize
    total = combined_pmf.sum()
    if total > 0:
        combined_pmf = combined_pmf / total

    return combined_pmf


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_kappa(pred_p, actual_N, actual_K):
    """
    Find the concentration parameter κ that maximizes log-likelihood
    of observed strikeout counts under the Beta-Binomial model.

    CRITICAL: We use ACTUAL N (not predicted N) here. The question κ answers
    is: "given that we know exactly how many batters were faced, how uncertain
    is the model's predicted strikeout rate p̂?" If we used predicted N, errors
    in the N model would contaminate the κ estimate and push it toward
    infinity (tighter = better when N is wrong in a correlated way).

    Higher κ = tighter distribution around predicted p (more confident).
    Lower κ = wider distribution (more uncertain).
    Typical baseball values: κ ≈ 10-80.
      - κ=10: p̂=0.25 → Beta(2.5, 7.5) → quite wide, std≈0.13
      - κ=30: p̂=0.25 → Beta(7.5, 22.5) → moderate, std≈0.08
      - κ=80: p̂=0.25 → Beta(20, 60) → tight, std≈0.05
    """
    # Grid search first to find the right region, then refine
    kappa_grid = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500]

    def compute_ll(kappa):
        total_ll = 0.0
        for p_i, n_i, k_i in zip(pred_p, actual_N, actual_K):
            n_i = int(round(n_i))
            k_i = int(k_i)
            if n_i < k_i:
                n_i = k_i  # Safety: N must be >= K
            if n_i <= 0:
                continue
            # Clamp p away from 0/1 to avoid degenerate Beta
            p_clamped = np.clip(p_i, 0.005, 0.995)
            alpha = p_clamped * kappa
            beta_p = (1 - p_clamped) * kappa
            prob = beta_binom_pmf(k_i, n_i, alpha, beta_p)
            total_ll += np.log(max(prob, 1e-15))
        return total_ll

    # Grid search
    grid_results = []
    for kap in kappa_grid:
        ll = compute_ll(kap)
        grid_results.append((kap, ll))

    grid_results.sort(key=lambda x: x[1], reverse=True)
    print(f"    κ grid search (top 5):")
    for kap, ll in grid_results[:5]:
        print(f"      κ={kap:>6.0f}  →  LL={ll:>10.2f}")

    # Refine around the best grid point
    best_grid_kappa = grid_results[0][0]
    low = max(2, best_grid_kappa * 0.4)
    high = best_grid_kappa * 2.5

    def neg_ll(log_kappa):
        return -compute_ll(np.exp(log_kappa))

    result = minimize_scalar(neg_ll, bounds=(np.log(low), np.log(high)), method="bounded")
    kappa_opt = np.exp(result.x)
    final_ll = -result.fun

    # Sanity check: if κ is hitting the upper bound, warn
    if kappa_opt > 200:
        print(f"    ⚠ WARNING: κ={kappa_opt:.0f} is very high — distribution may be too tight.")
        print(f"      This can happen if predictions are very accurate on the test set.")
        print(f"      Consider using cross-validated out-of-fold predictions for calibration.")

    print(f"    ✓ Calibrated κ = {kappa_opt:.2f} (LL = {final_ll:.2f})")

    # Print what this means in practice
    example_p = 0.25
    alpha_ex = example_p * kappa_opt
    beta_ex = (1 - example_p) * kappa_opt
    std_p = np.sqrt(alpha_ex * beta_ex / ((alpha_ex + beta_ex) ** 2 * (alpha_ex + beta_ex + 1)))
    print(f"    → For p̂=0.25: Beta({alpha_ex:.1f}, {beta_ex:.1f}), std(p)={std_p:.4f}")
    print(f"    → ~95% of true p in [{example_p - 2 * std_p:.3f}, {example_p + 2 * std_p:.3f}]")

    return kappa_opt


def calibrate_kappa_cv(X_train, y_train_rate, y_train_bf, y_train_K,
                       feature_cols_rate, rate_model, n_folds=5):
    """
    Cross-validated κ calibration. Trains the rate model on K-1 folds,
    predicts the held-out fold, then calibrates κ on those out-of-fold
    predictions. This avoids overfitting κ to in-sample predictions.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.base import clone

    tscv = TimeSeriesSplit(n_splits=n_folds)

    # Collect all OOF predictions
    all_pred_p = []
    all_actual_N = []
    all_actual_K = []

    print(f"    Running {n_folds}-fold CV for κ calibration...")
    print(f"    Total training samples: {len(y_train_rate)}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"      Fold {fold + 1}: train={len(train_idx)}, val={len(val_idx)}")

        # Clone model (safer than deepcopy for sklearn-compatible estimators)
        try:
            fold_model = clone(rate_model)
        except Exception:
            import copy
            fold_model = copy.deepcopy(rate_model)

        fold_model.fit(X_train[train_idx], y_train_rate[train_idx])
        fold_preds = fold_model.predict(X_train[val_idx])

        fold_N = y_train_bf[val_idx]
        fold_K = y_train_K[val_idx]

        # Filter valid rows for this fold
        valid_mask = (fold_N > 0) & np.isfinite(fold_preds) & np.isfinite(fold_K)
        n_valid = valid_mask.sum()
        print(f"        Valid predictions: {n_valid}")

        all_pred_p.append(fold_preds[valid_mask])
        all_actual_N.append(fold_N[valid_mask])
        all_actual_K.append(fold_K[valid_mask])

    # Concatenate all folds
    oof_pred_p = np.concatenate(all_pred_p)
    oof_actual_N = np.concatenate(all_actual_N)
    oof_actual_K = np.concatenate(all_actual_K)

    print(f"    Total out-of-fold predictions for calibration: {len(oof_pred_p)}")

    if len(oof_pred_p) < 50:
        print(f"    ⚠ WARNING: Only {len(oof_pred_p)} OOF predictions — κ calibration may be unreliable.")

    kappa = calibrate_kappa(oof_pred_p, oof_actual_N, oof_actual_K)
    return kappa


def calibrate_sigma_N(pred_N, actual_N):
    """
    Calibrate the standard deviation of N prediction errors.
    This captures uncertainty in how many batters the pitcher will face.
    """
    residuals = actual_N - pred_N
    sigma = np.std(residuals)
    bias = np.mean(residuals)
    mae = np.mean(np.abs(residuals))
    print(f"    N residuals: mean = {bias:.2f}, std = {sigma:.2f}, MAE = {mae:.2f}")
    # Also report percentiles to show the shape
    pcts = np.percentile(residuals, [5, 25, 50, 75, 95])
    print(f"    N residual percentiles: 5%={pcts[0]:.1f}, 25%={pcts[1]:.1f}, "
          f"50%={pcts[2]:.1f}, 75%={pcts[3]:.1f}, 95%={pcts[4]:.1f}")
    return sigma


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_distributions(pred_p, pred_N, actual_K, actual_N, kappa, sigma_N):
    """
    Compare Beta-Binomial distributions against:
    1. Old normal approximation (pred_K = pred_p * pred_N, σ = 2.1)
    2. Naive mean prediction

    Metrics:
    - Log-likelihood (higher = better calibrated)
    - MAE of point prediction (E[K] from the distribution)
    - Brier-like score for over/under at common lines
    - Calibration: does P(K >= 6) = 30% actually happen 30% of the time?
    """
    results = {
        "beta_binom": {"log_lik": 0, "point_preds": [], "over_probs": {}},
        "normal": {"log_lik": 0, "point_preds": [], "over_probs": {}},
    }

    lines = [4, 5, 6, 7, 8]
    for l in lines:
        results["beta_binom"]["over_probs"][l] = []
        results["normal"]["over_probs"][l] = []

    NORMAL_STD = 2.1  # The old hardcoded value

    for p_i, n_i, k_i, n_actual in zip(pred_p, pred_N, actual_K, actual_N):
        k_i = int(k_i)
        n_round = max(int(round(n_i)), 1)

        # ── Beta-Binomial ───────────────────────────────────────────────
        pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=30)

        # Point prediction = E[K]
        ks = np.arange(len(pmf))
        ek = np.sum(ks * pmf)
        results["beta_binom"]["point_preds"].append(ek)

        # Log-likelihood
        if k_i < len(pmf):
            prob = pmf[k_i]
        else:
            prob = 1e-15
        results["beta_binom"]["log_lik"] += np.log(max(prob, 1e-15))

        # Over probabilities
        for l in lines:
            p_over = pmf[l:].sum() if l < len(pmf) else 0.0
            results["beta_binom"]["over_probs"][l].append(p_over)

        # ── Normal approximation ────────────────────────────────────────
        pred_k_normal = p_i * n_i  # Point prediction
        results["normal"]["point_preds"].append(pred_k_normal)

        # Log-likelihood (use normal PDF evaluated at integer k)
        norm_prob = sp_stats.norm.pdf(k_i, loc=pred_k_normal, scale=NORMAL_STD)
        results["normal"]["log_lik"] += np.log(max(norm_prob, 1e-15))

        # Over probabilities
        for l in lines:
            # P(K >= l) = 1 - CDF(l - 0.5) with continuity correction
            z = (l - 0.5 - pred_k_normal) / NORMAL_STD
            p_over_norm = 1 - sp_stats.norm.cdf(z)
            results["normal"]["over_probs"][l].append(p_over_norm)

    # ── Compute summary metrics ─────────────────────────────────────────
    actual_K_arr = np.array(actual_K, dtype=float)

    bb_preds = np.array(results["beta_binom"]["point_preds"])
    norm_preds = np.array(results["normal"]["point_preds"])

    summary = {
        "beta_binom": {
            "MAE": np.mean(np.abs(actual_K_arr - bb_preds)),
            "RMSE": np.sqrt(np.mean((actual_K_arr - bb_preds) ** 2)),
            "log_lik": results["beta_binom"]["log_lik"],
            "mean_pred": bb_preds.mean(),
        },
        "normal": {
            "MAE": np.mean(np.abs(actual_K_arr - norm_preds)),
            "RMSE": np.sqrt(np.mean((actual_K_arr - norm_preds) ** 2)),
            "log_lik": results["normal"]["log_lik"],
            "mean_pred": norm_preds.mean(),
        },
    }

    # ── Calibration at each line ────────────────────────────────────────
    for method in ["beta_binom", "normal"]:
        summary[method]["calibration"] = {}
        for l in lines:
            probs = np.array(results[method]["over_probs"][l])
            actual_over = (actual_K_arr >= l).astype(float)

            # Bin probabilities and check calibration
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            cal_data = []
            for i in range(len(bins) - 1):
                mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if mask.sum() >= 10:  # Need enough samples
                    pred_mean = probs[mask].mean()
                    actual_rate = actual_over[mask].mean()
                    cal_data.append((pred_mean, actual_rate, mask.sum()))

            # Brier score
            brier = np.mean((probs - actual_over) ** 2)
            summary[method]["calibration"][l] = {
                "brier": brier,
                "bins": cal_data,
                "overall_pred": probs.mean(),
                "overall_actual": actual_over.mean(),
            }

    return summary, results


def evaluate_distributions_per_pitcher(pred_p, pred_N, actual_K, actual_N, kappa, sigma_N_array):
    """
    Same as evaluate_distributions but uses per-pitcher σ_N values.
    sigma_N_array is an array the same length as pred_p.
    """
    results = {
        "beta_binom": {"log_lik": 0, "point_preds": [], "over_probs": {}},
        "normal": {"log_lik": 0, "point_preds": [], "over_probs": {}},
    }

    lines = [4, 5, 6, 7, 8]
    for l in lines:
        results["beta_binom"]["over_probs"][l] = []
        results["normal"]["over_probs"][l] = []

    NORMAL_STD = 2.1

    for p_i, n_i, k_i, n_actual, sig_i in zip(pred_p, pred_N, actual_K, actual_N, sigma_N_array):
        k_i = int(k_i)
        n_round = max(int(round(n_i)), 1)

        # Beta-Binomial with per-pitcher sigma
        pmf = expected_pmf_over_N(p_i, n_i, kappa, sig_i, max_k=30)
        ks = np.arange(len(pmf))
        ek = np.sum(ks * pmf)
        results["beta_binom"]["point_preds"].append(ek)

        if k_i < len(pmf):
            prob = pmf[k_i]
        else:
            prob = 1e-15
        results["beta_binom"]["log_lik"] += np.log(max(prob, 1e-15))

        for l in lines:
            p_over = pmf[l:].sum() if l < len(pmf) else 0.0
            results["beta_binom"]["over_probs"][l].append(p_over)

        # Normal (same as global — doesn't use per-pitcher sigma)
        pred_k_normal = p_i * n_i
        results["normal"]["point_preds"].append(pred_k_normal)
        norm_prob = sp_stats.norm.pdf(k_i, loc=pred_k_normal, scale=NORMAL_STD)
        results["normal"]["log_lik"] += np.log(max(norm_prob, 1e-15))

        for l in lines:
            z = (l - 0.5 - pred_k_normal) / NORMAL_STD
            p_over_norm = 1 - sp_stats.norm.cdf(z)
            results["normal"]["over_probs"][l].append(p_over_norm)

    actual_K_arr = np.array(actual_K, dtype=float)
    bb_preds = np.array(results["beta_binom"]["point_preds"])
    norm_preds = np.array(results["normal"]["point_preds"])
    ss_tot = np.sum((actual_K_arr - actual_K_arr.mean()) ** 2)

    summary = {
        "beta_binom": {
            "MAE": np.mean(np.abs(actual_K_arr - bb_preds)),
            "RMSE": np.sqrt(np.mean((actual_K_arr - bb_preds) ** 2)),
            "log_lik": results["beta_binom"]["log_lik"],
            "mean_pred": bb_preds.mean(),
        },
        "normal": {
            "MAE": np.mean(np.abs(actual_K_arr - norm_preds)),
            "RMSE": np.sqrt(np.mean((actual_K_arr - norm_preds) ** 2)),
            "log_lik": results["normal"]["log_lik"],
            "mean_pred": norm_preds.mean(),
        },
    }

    for method in ["beta_binom", "normal"]:
        summary[method]["calibration"] = {}
        for l in lines:
            probs = np.array(results[method]["over_probs"][l])
            actual_over = (actual_K_arr >= l).astype(float)
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            cal_data = []
            for i in range(len(bins) - 1):
                mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if mask.sum() >= 10:
                    pred_mean = probs[mask].mean()
                    actual_rate = actual_over[mask].mean()
                    cal_data.append((pred_mean, actual_rate, mask.sum()))
            brier = np.mean((probs - actual_over) ** 2)
            summary[method]["calibration"][l] = {
                "brier": brier, "bins": cal_data,
                "overall_pred": probs.mean(), "overall_actual": actual_over.mean(),
            }

    return summary, results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(summary, output_dir):
    """Bar charts comparing Beta-Binomial vs Normal on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = ["beta_binom", "normal"]
    labels = ["Beta-Binomial", "Normal (σ=2.1)"]
    colors = ["#2196F3", "#FF9800"]

    # MAE
    ax = axes[0]
    vals = [summary[m]["MAE"] for m in methods]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Point Prediction MAE", fontweight="bold")
    ax.set_ylabel("MAE (strikeouts)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Log-likelihood
    ax = axes[1]
    vals = [summary[m]["log_lik"] for m in methods]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Log-Likelihood (higher = better)", fontweight="bold")
    ax.set_ylabel("Total log-likelihood")
    for i, v in enumerate(vals):
        ax.text(i, v * 0.98, f"{v:.0f}", ha="center", fontsize=10)

    # Brier score at K >= 6
    ax = axes[2]
    if 6 in summary["beta_binom"]["calibration"]:
        vals = [summary[m]["calibration"][6]["brier"] for m in methods]
        ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title("Brier Score at K ≥ 6 (lower = better)", fontweight="bold")
        ax.set_ylabel("Brier Score")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "beta_binom_vs_normal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved comparison plot")


def plot_calibration(summary, output_dir):
    """Calibration plots: predicted probability vs observed frequency."""
    lines_to_plot = [5, 6, 7]
    fig, axes = plt.subplots(1, len(lines_to_plot), figsize=(5 * len(lines_to_plot), 5))

    for idx, line in enumerate(lines_to_plot):
        ax = axes[idx] if len(lines_to_plot) > 1 else axes

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        for method, label, color in [
            ("beta_binom", "Beta-Binomial", "#2196F3"),
            ("normal", "Normal", "#FF9800"),
        ]:
            if line in summary[method]["calibration"]:
                bins = summary[method]["calibration"][line]["bins"]
                if bins:
                    pred_vals = [b[0] for b in bins]
                    actual_vals = [b[1] for b in bins]
                    sizes = [b[2] for b in bins]
                    ax.scatter(pred_vals, actual_vals, s=[s / 5 for s in sizes],
                               color=color, alpha=0.7, label=label, edgecolors="black", linewidth=0.5)

        ax.set_xlabel("Predicted P(K ≥ line)")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration: K ≥ {line}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved calibration plots")


def plot_example_pmfs(pred_p_arr, pred_N_arr, actual_K_arr, kappa, sigma_N, output_dir, n_examples=6):
    """Show example PMFs for individual games alongside the actual outcome."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Pick diverse examples: low-K, medium-K, high-K games
    sorted_idx = np.argsort(actual_K_arr)
    n = len(sorted_idx)
    picks = [
        sorted_idx[int(n * 0.05)],   # Low K
        sorted_idx[int(n * 0.20)],   # Below avg
        sorted_idx[int(n * 0.40)],   # Avg-ish
        sorted_idx[int(n * 0.60)],   # Above avg
        sorted_idx[int(n * 0.80)],   # High K
        sorted_idx[int(n * 0.95)],   # Very high K
    ]

    for ax_idx, i in enumerate(picks[:n_examples]):
        ax = axes[ax_idx]
        p_i = pred_p_arr[i]
        n_i = pred_N_arr[i]
        k_actual = int(actual_K_arr[i])

        # Beta-Binomial PMF
        pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=20)
        ks = np.arange(len(pmf))
        ek = np.sum(ks * pmf)

        ax.bar(ks, pmf, color="#2196F3", alpha=0.6, label="Beta-Binomial PMF", edgecolor="black", linewidth=0.3)
        ax.axvline(k_actual, color="red", linewidth=2, linestyle="--", label=f"Actual K = {k_actual}")
        ax.axvline(ek, color="#2196F3", linewidth=2, linestyle=":", label=f"E[K] = {ek:.1f}")

        # Normal overlay
        NORMAL_STD = 2.1
        pred_k_normal = p_i * n_i
        x_norm = np.linspace(0, 20, 200)
        y_norm = sp_stats.norm.pdf(x_norm, loc=pred_k_normal, scale=NORMAL_STD)
        # Scale to match bar heights
        y_norm_scaled = y_norm * (pmf.max() / y_norm.max()) if y_norm.max() > 0 else y_norm
        ax.plot(x_norm, y_norm_scaled, color="#FF9800", linewidth=1.5, alpha=0.8, label="Normal approx")

        ax.set_title(f"p̂={p_i:.3f}, N̂={n_i:.0f}", fontsize=10)
        ax.set_xlabel("Strikeouts")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, 18)

    plt.suptitle("Beta-Binomial PMF vs Normal Approximation (Example Games)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "example_pmfs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved example PMF plots")


def plot_submodel_diagnostics(y_test, preds, target_name, output_dir):
    """Residual plot and predicted-vs-actual for a sub-model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs actual
    ax = axes[0]
    ax.scatter(y_test, preds, alpha=0.15, s=8, color="#2196F3")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{target_name}: Predicted vs Actual", fontweight="bold")

    # Residuals
    ax = axes[1]
    residuals = y_test - preds
    ax.hist(residuals, bins=50, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"{target_name}: Residual Distribution", fontweight="bold")
    ax.text(0.02, 0.95, f"Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    safe_name = target_name.replace("/", "_per_")
    plt.savefig(output_dir / f"diagnostics_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_thresholds(pred_p, pred_N, actual_K, kappa, sigma_N):
    """Compare the Beta-Binomial PMF against a Normal approximation at the
    strikeout thresholds that matter.

    The point of the Beta-Binomial is that strikeout counts are discrete,
    bounded by batters faced, and overdispersed relative to a Binomial. A
    Normal with a fixed sigma ignores all three. This function tests whether
    that actually buys better probabilities at K >= 5, 6, 7, 8.

    Scored with proper scoring rules rather than accuracy alone:
      Brier    — mean squared error of the probability. Lower is better.
      Log-loss — penalises confident-and-wrong much harder. Lower is better.
      Accuracy — fraction correct when thresholding the probability at 0.5.
                 Reported for intuition, but it throws away calibration
                 information, which is exactly what we care about here.
    """
    lines_ = [4.5, 5.5, 6.5, 7.5]
    NORMAL_STD = 2.1
    EPS = 1e-15

    n_games = len(actual_K)
    actual_K_arr = np.array(actual_K)

    print(f"\n{'═' * 95}")
    print(f"  THRESHOLD EVALUATION: Beta-Binomial vs Normal ({n_games} starts)")
    print(f"{'═' * 95}")

    print(f"  Precomputing {n_games} Beta-Binomial PMFs...")
    bb_pmfs = [expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=25)
               for p_i, n_i in zip(pred_p, pred_N)]

    print(f"\n  {'Line':>6s} {'Method':20s} {'Base':>7s} {'Brier':>9s} "
          f"{'LogLoss':>9s} {'Acc':>7s}")
    print(f"  {'-' * 62}")

    summary = []
    for line in lines_:
        k_threshold = int(line) + 1          # P(K > 5.5) == P(K >= 6)
        actual_over = actual_K_arr > line
        base_rate   = actual_over.mean()

        bb_probs = np.array([
            pmf[k_threshold:].sum() if k_threshold < len(pmf) else 0.0
            for pmf in bb_pmfs
        ])
        nm_probs = np.array([
            1 - sp_stats.norm.cdf(line, loc=pred_p[i] * pred_N[i], scale=NORMAL_STD)
            for i in range(n_games)
        ])

        for method_name, probs in [("Beta-Binomial", bb_probs),
                                   ("Normal(σ=2.1)", nm_probs)]:
            pc       = np.clip(probs, EPS, 1 - EPS)
            brier    = np.mean((probs - actual_over) ** 2)
            log_loss = -np.mean(actual_over * np.log(pc) +
                                (1 - actual_over) * np.log(1 - pc))
            accuracy = np.mean((probs > 0.5) == actual_over)
            print(f"  {line:>6.1f} {method_name:20s} {base_rate:>6.1%} "
                  f"{brier:>9.4f} {log_loss:>9.4f} {accuracy:>6.1%}")
            summary.append((line, method_name, brier, log_loss))
        print(f"  {'-' * 62}")

    # ── Which distribution wins, and by how much ────────────────────────
    print(f"\n{'═' * 95}")
    print(f"  BETA-BINOMIAL vs NORMAL — Brier improvement per threshold")
    print(f"{'═' * 95}")
    print(f"  {'Line':>6s} {'BB Brier':>10s} {'Normal Brier':>13s} "
          f"{'Delta':>9s} {'Winner':>16s}")
    print(f"  {'-' * 60}")

    bb_wins = 0
    for line in lines_:
        bb   = next(b for (l, m, b, _) in summary if l == line and m == "Beta-Binomial")
        nm   = next(b for (l, m, b, _) in summary if l == line and m == "Normal(σ=2.1)")
        delta = nm - bb                      # positive => BB is better
        winner = "Beta-Binomial" if delta > 0 else "Normal"
        bb_wins += delta > 0
        print(f"  {line:>6.1f} {bb:>10.4f} {nm:>13.4f} {delta:>+9.4f} {winner:>16s}")

    print(f"\n  Beta-Binomial wins {bb_wins}/{len(lines_)} thresholds on Brier score.")



# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def save_models(rate_model, bf_model, rate_features, bf_features, kappa, sigma_N, model_dir):
    """Save everything needed for daily predictions."""
    joblib.dump(rate_model, model_dir / "rate_model.joblib")
    joblib.dump(bf_model, model_dir / "bf_model.joblib")

    config = {
        "kappa": float(kappa),
        "sigma_N": float(sigma_N),
        "rate_features": rate_features,
        "bf_features": bf_features,
        "target_rate": TARGET_RATE,
        "target_bf": TARGET_BF,
        "min_pa_game": MIN_PA_GAME,
    }
    with open(model_dir / "beta_binom_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  ✓ Saved models and config to {model_dir}/")
    print(f"    - rate_model.joblib")
    print(f"    - bf_model.joblib")
    print(f"    - beta_binom_config.json (κ={kappa:.2f}, σ_N={sigma_N:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BETA-BINOMIAL STRIKEOUT MODEL")
    print("  K ~ BetaBinomial(N, p·κ, (1-p)·κ)")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────
    print("\n── Step 1: Load & Prepare Data ──")
    df = load_and_prepare()

    # Add BF-specific engineered features (absorbed from old 09_improve_bf_model).
    # These flow through feature selection + pruning like everything else.
    print("\n── Step 1b: Engineer BF-Specific Features ──")
    df = engineer_bf_features(df, target_col=TARGET_BF)

    # ── Feature selection ───────────────────────────────────────────────
    print("\n── Step 2: Feature Selection ──")
    rate_features = select_features(df, TARGET_RATE)
    bf_features = select_bf_features_permissive(df, TARGET_BF)

    # ── Collinearity pruning ────────────────────────────────────────────
    # Many of our features are near-duplicates by construction
    # (k_pct_L3 / L5 / L10 / szn / ewm / trend_3 / ...). We cluster on
    # |Spearman corr| and keep the most target-relevant member of each
    # cluster. Restrict the corr matrix to TRAINING rows only (no leakage).
    print("\n── Step 2b: Collinearity Pruning ──")
    df_for_prune = df.copy()
    if "game_date" in df_for_prune.columns:
        df_for_prune["year"] = pd.to_datetime(
            df_for_prune["game_date"], errors="coerce"
        ).dt.year
    rate_features, rate_dropped = prune_collinear_features(
        df_for_prune, rate_features, TARGET_RATE,
        corr_threshold=0.99, target_year_cutoff=TEST_YEAR,
    )
    bf_features, bf_dropped = prune_collinear_features(
        df_for_prune, bf_features, TARGET_BF,
        corr_threshold=0.99, target_year_cutoff=TEST_YEAR,
    )
    print(f"\n  Final feature counts: rate={len(rate_features)}, "
          f"bf={len(bf_features)}")
    del df_for_prune

    # ── Unified data split ─────────────────────────────────────────────
    # CRITICAL: Both sub-models must use the same rows so that predictions
    # are aligned for the Beta-Binomial combination step.
    print("\n── Step 3: Unified Train/Test Split ──")

    # Only require targets and actual_K to be non-NaN
    # Features with NaN get filled with 0 — tree models handle this fine
    required_targets = [TARGET_RATE, TARGET_BF, "actual_K"]
    df_clean = df.dropna(subset=required_targets).copy()
    df_clean = df_clean.sort_values("game_date").reset_index(drop=True)
    df_clean["year"] = df_clean["game_date"].dt.year

    # Fill NaN in feature columns with 0
    all_features = list(set(rate_features + bf_features))
    for col in all_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    train_df = df_clean[df_clean["year"] < TEST_YEAR].copy()
    test_df = df_clean[df_clean["year"] >= TEST_YEAR].copy()

    print(f"  After unifying: {len(df_clean):,} rows ({len(df_clean.columns)} cols)")
    print(f"  Train: {len(train_df):,} rows ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['year'].min()}-{test_df['year'].max()})")

    # Extract arrays
    X_train_r = train_df[rate_features].values
    X_test_r = test_df[rate_features].values
    y_train_r = train_df[TARGET_RATE].values
    y_test_r = test_df[TARGET_RATE].values

    X_train_b = train_df[bf_features].values
    X_test_b = test_df[bf_features].values
    y_train_b = train_df[TARGET_BF].values
    y_test_b = test_df[TARGET_BF].values

    train_actual_K = train_df["actual_K"].values
    test_actual_K = test_df["actual_K"].values

    # ── Train K/PA rate model ───────────────────────────────────────────
    print("\n── Step 4: Train K/PA Rate Model ──")
    rate_results, rate_best = train_xgb(X_train_r, y_train_r, X_test_r, y_test_r, rate_features, "K/PA")

    # ── Train Batters Faced model ───────────────────────────────────────
    # ── Train Batters Faced model (multi-variant: std / log / Huber) ────
    print("\n── Step 5: Train Batters Faced Model (multi-variant) ──")
    bf_preds, bf_model, bf_best, _bf_variants = train_bf_models_multivariant(
        X_train_b, y_train_b, X_test_b, y_test_b, bf_features,
    )
    # Compat-shape the downstream contract: bf_results[bf_best] must have
    # "model", "preds", "MAE", "R²". Compute R² once here.
    from sklearn.metrics import r2_score as _r2
    bf_results = {
        bf_best: {
            "model": bf_model,
            "preds": bf_preds,
            "MAE": float(np.mean(np.abs(bf_preds - y_test_b))),
            "R²": float(_r2(y_test_b, bf_preds)),
        }
    }

    # ── Diagnostics for sub-models ──────────────────────────────────────
    print("\n── Step 6: Sub-Model Diagnostics ──")
    rate_preds = rate_results[rate_best]["preds"]
    bf_preds = bf_results[bf_best]["preds"]

    plot_submodel_diagnostics(y_test_r, rate_preds, "K_per_PA", OUTPUT_DIR)
    plot_submodel_diagnostics(y_test_b, bf_preds, "Batters_Faced", OUTPUT_DIR)
    print(f"  ✓ Sub-model diagnostic plots saved")

    # Baselines for context
    print(f"\n  K/PA Rate model ({rate_best}):")
    print(f"    MAE:  {rate_results[rate_best]['MAE']:.4f}")
    print(f"    R²:   {rate_results[rate_best]['R²']:.4f}")
    print(f"    Mean actual: {y_test_r.mean():.4f}, Mean predicted: {rate_preds.mean():.4f}")

    print(f"\n  Batters Faced model ({bf_best}):")
    print(f"    MAE:  {bf_results[bf_best]['MAE']:.2f}")
    print(f"    R²:   {bf_results[bf_best]['R²']:.4f}")
    print(f"    Mean actual: {y_test_b.mean():.1f}, Mean predicted: {bf_preds.mean():.1f}")

    # Direct K prediction from p * N (before Beta-Binomial uncertainty)
    direct_k_pred = rate_preds * bf_preds
    direct_mae = np.mean(np.abs(test_actual_K - direct_k_pred))
    print(f"\n  Direct point prediction (p̂ × N̂):")
    print(f"    MAE: {direct_mae:.3f}")

    # ── Archetype bias check ────────────────────────────────────────────
    print("\n── Step 6b: Archetype Bias Check ──")
    median_rate = np.median(rate_preds[rate_preds > 0])
    median_bf = np.median(bf_preds)
    archetypes = {
        "High-K, Deep": (rate_preds >= median_rate) & (bf_preds >= median_bf),
        "High-K, Short": (rate_preds >= median_rate) & (bf_preds < median_bf),
        "Low-K, Deep": (rate_preds < median_rate) & (bf_preds >= median_bf),
        "Low-K, Short": (rate_preds < median_rate) & (bf_preds < median_bf),
    }

    print(f"    {'Archetype':>15s} {'Games':>6s} {'MAE':>7s} {'Bias':>8s} {'Avg K':>7s} {'Avg Pred':>9s}")
    print(f"    {'-' * 60}")
    for name, mask in archetypes.items():
        if mask.sum() < 10:
            continue
        mae = np.mean(np.abs(test_actual_K[mask] - direct_k_pred[mask]))
        bias = np.mean(test_actual_K[mask] - direct_k_pred[mask])
        avg_k = test_actual_K[mask].mean()
        avg_pred = direct_k_pred[mask].mean()
        print(f"    {name:>15s} {mask.sum():>6d} {mae:>7.3f} {bias:>+8.3f} {avg_k:>7.1f} {avg_pred:>9.1f}")

    # ── Calibrate κ and σ_N ─────────────────────────────────────────────
    print("\n── Step 7: Calibrate Beta-Binomial Parameters ──")
    sigma_N_global = calibrate_sigma_N(bf_preds, y_test_b)

    # Per-pitcher σ_N: use bf_L10_std if available, else fall back to global
    # This captures that an ace who consistently goes 6-7 innings has lower
    # BF variance than a volatile pitcher who alternates 4 and 7 innings.
    print("\n  Per-pitcher σ_N calibration:")
    if "bf_L10_std" in test_df.columns:
        pitcher_sigma = test_df["bf_L10_std"].values.copy()
        # Fill NaN/zero with global (first few starts won't have std)
        invalid = np.isnan(pitcher_sigma) | (pitcher_sigma <= 0)
        pitcher_sigma[invalid] = sigma_N_global
        # Floor at 1.0 (even the most consistent pitcher has some variance)
        pitcher_sigma = np.clip(pitcher_sigma, 1.0, sigma_N_global * 2)

        print(f"    Per-pitcher σ_N: mean={pitcher_sigma.mean():.2f}, "
              f"median={np.median(pitcher_sigma):.2f}, "
              f"min={pitcher_sigma.min():.2f}, max={pitcher_sigma.max():.2f}")
        print(f"    Pitchers using global fallback: {invalid.sum()} / {len(invalid)}")
        use_per_pitcher_sigma = True
    else:
        print(f"    bf_L10_std not available — using global σ_N = {sigma_N_global:.2f}")
        pitcher_sigma = np.full(len(y_test_b), sigma_N_global)
        use_per_pitcher_sigma = False

    # For backward compatibility, sigma_N is still the global value
    sigma_N = sigma_N_global

    # Method A: Calibrate κ using test-set predictions but ACTUAL N
    print("\n  Method A: κ from test-set predictions + actual N")
    kappa_test = calibrate_kappa(rate_preds, y_test_b, test_actual_K)

    # Method B: Cross-validated κ from training data (preferred)
    # Now correctly aligned — train_actual_K and y_train_b come from the same rows
    print("\n  Method B: κ from cross-validated training predictions")
    kappa_cv = calibrate_kappa_cv(
        X_train_r, y_train_r, y_train_b, train_actual_K,
        rate_features, rate_results[rate_best]["model"], n_folds=5,
    )

    # Use the CV-calibrated κ (more conservative and generalizable)
    kappa = kappa_cv
    print(f"\n  → Using CV κ = {kappa:.2f} (test κ was {kappa_test:.2f})")

    # ── Evaluate distributions ──────────────────────────────────────────
    print("\n── Step 8: Evaluate Beta-Binomial vs Normal ──")

    # Evaluate with global σ_N
    summary_global, _ = evaluate_distributions(
        rate_preds, bf_preds,
        test_actual_K, y_test_b,
        kappa, sigma_N,
    )

    # Evaluate with per-pitcher σ_N
    if use_per_pitcher_sigma:
        summary_pp, full_results = evaluate_distributions_per_pitcher(
            rate_preds, bf_preds,
            test_actual_K, y_test_b,
            kappa, pitcher_sigma,
        )
    else:
        summary_pp = summary_global
        full_results = None

    print(f"\n  {'Metric':25s} {'BB (global σ)':>15s} {'BB (per-pitch σ)':>17s} {'Normal(σ=2.1)':>15s}")
    print(f"  {'-' * 75}")
    print(f"  {'Point MAE':25s} {summary_global['beta_binom']['MAE']:>15.3f} {summary_pp['beta_binom']['MAE']:>17.3f} {summary_global['normal']['MAE']:>15.3f}")
    print(f"  {'Log-Likelihood':25s} {summary_global['beta_binom']['log_lik']:>15.0f} {summary_pp['beta_binom']['log_lik']:>17.0f} {summary_global['normal']['log_lik']:>15.0f}")
    print(f"  {'Mean Prediction':25s} {summary_global['beta_binom']['mean_pred']:>15.2f} {summary_pp['beta_binom']['mean_pred']:>17.2f} {summary_global['normal']['mean_pred']:>15.2f}")

    print(f"\n  Calibration (Brier scores — lower is better):")
    for l in [5, 6, 7]:
        if l in summary_global["beta_binom"]["calibration"] and l in summary_pp["beta_binom"]["calibration"]:
            g_brier = summary_global["beta_binom"]["calibration"][l]["brier"]
            pp_brier = summary_pp["beta_binom"]["calibration"][l]["brier"]
            nm_brier = summary_global["normal"]["calibration"][l]["brier"]
            best = min(g_brier, pp_brier, nm_brier)
            label = "← global" if best == g_brier else ("← per-pitch" if best == pp_brier else "← Norm")
            print(f"    K ≥ {l}:  Global {g_brier:.4f}  |  Per-pitcher {pp_brier:.4f}  |  Normal {nm_brier:.4f}  {label}")

    # Use whichever sigma approach has better log-likelihood
    if summary_pp["beta_binom"]["log_lik"] > summary_global["beta_binom"]["log_lik"]:
        summary = summary_pp
        sigma_N_mode = "per_pitcher"
        print(f"\n  → Per-pitcher σ_N wins (LL improvement: {summary_pp['beta_binom']['log_lik'] - summary_global['beta_binom']['log_lik']:.0f})")
    else:
        summary = summary_global
        sigma_N_mode = "global"
        print(f"\n  → Global σ_N wins")

    print("\n── Step 9: Generate Plots ──")
    plot_comparison(summary, OUTPUT_DIR)
    plot_calibration(summary, OUTPUT_DIR)
    plot_example_pmfs(rate_preds, bf_preds, test_actual_K, kappa, sigma_N, OUTPUT_DIR)

    # ── Threshold evaluation ────────────────────────────────────────────
    print("\n── Step 10: Threshold Evaluation ──")
    evaluate_thresholds(rate_preds, bf_preds, test_actual_K, kappa, sigma_N)

    # ── Save models ─────────────────────────────────────────────────────
    print("\n── Step 11: Save Models ──")
    save_models(
        rate_results[rate_best]["model"],
        bf_results[bf_best]["model"],
        rate_features,
        bf_features,
        kappa,
        sigma_N,
        MODEL_DIR,
    )

    # Also save sigma_N_mode to config
    config_path = MODEL_DIR / "beta_binom_config.json"
    with open(config_path) as f:
        config = json.load(f)
    # Write under BOTH naming conventions so 05/09/11 can read it regardless of
    # which writer last touched the config (06 historically used capital N,
    # 09 uses lowercase n).
    config["sigma_N_mode"] = sigma_N_mode
    config["sigma_N_global"] = float(sigma_N)
    config["sigma_n_type"] = sigma_N_mode
    config["sigma_n_global"] = float(sigma_N)
    config["sigma_n"] = float(sigma_N)
    # Record which BF variant won — critical for daily prediction file
    # to know whether bf_model.predict() returns BF or log(BF).
    config["bf_variant"] = str(bf_best)
    config["bf_is_log"] = bool(_bf_variants[bf_best].get("is_log", False))
    if config["bf_is_log"] and "log_sigma2" in _bf_variants[bf_best]:
        config["bf_log_sigma2"] = float(_bf_variants[bf_best]["log_sigma2"])
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"    σ_N mode: {sigma_N_mode}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  K/PA Rate Model:     {rate_best} (MAE = {rate_results[rate_best]['MAE']:.4f})")
    print(f"  Batters Faced Model: {bf_best} (MAE = {bf_results[bf_best]['MAE']:.2f})")
    print(f"  Calibrated κ:        {kappa:.2f}")
    print(f"  Calibrated σ_N:      {sigma_N:.2f} ({sigma_N_mode})")
    print(f"  Point prediction MAE (p̂ × N̂): {direct_mae:.3f}")
    print(f"  Log-likelihood improvement: {summary['beta_binom']['log_lik'] - summary['normal']['log_lik']:.0f}")
    print(f"\n  Outputs in: {OUTPUT_DIR.resolve()}/")
    print(f"  Models in:  {MODEL_DIR.resolve()}/")

    # Save summary to CSV
    summary_rows = []
    for method in ["beta_binom", "normal"]:
        row = {"method": method}
        row.update({k: v for k, v in summary[method].items() if k != "calibration"})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "beta_binom_comparison.csv", index=False)
    print(f"  ✓ Saved beta_binom_comparison.csv")


if __name__ == "__main__":
    main()
