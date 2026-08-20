"""
train/hits_walks.py
======================
Beta-Binomial Models for Hits Allowed and Walks

Follows the same generative decomposition as the strikeout model in
train/strikeouts.py:
    Hits:  h = H/PA rate × N  →  BetaBinomial(N, α_h, β_h)
    Walks: w = BB/PA rate × N →  BetaBinomial(N, α_w, β_w)

Reuses the existing BF (plate_appearances) model from 06 for N predictions —
there is no point training a separate BF model since BF depends on the
pitcher and game context, not on the stat type. Only trains new rate
models for H/PA and BB/PA.

WHY THIS MIRRORS 06 SO CLOSELY:
- Variance emerges naturally from the interaction of p and N
- Different pitchers get different variance (deep starter ≠ short starter)
- Properly bounded: count can't exceed N
- Identical feature pipeline, training, evaluation, diagnostics
  → easy comparison and consistent operational behavior

PIPELINE:
    1. Load the feature matrix from features.py
    2. Create two targets: h_per_pa, bb_per_pa
    3. Train separate XGBoost rate models for each
       (RandomizedSearchCV + TimeSeriesSplit, same as 06)
    4. Load existing BF model from 06 to predict N
    5. Calibrate Beta concentration parameter (κ) per stat
    6. Produce Beta-Binomial PMFs and compare to the old normal approach
    7. Run archetype bias check and threshold evaluation
    8. Save both rate models + per-stat configs for predict/hits_walks.py

OUTPUTS:
    - models/hits_rate_model.joblib   — H/PA rate model
    - models/walks_rate_model.joblib  — BB/PA rate model
    - models/hits_config.json         — κ_h, features, etc.
    - models/walks_config.json        — κ_w, features, etc.
    - output/hits_walks_comparison.csv
    - output/<stat>_*.png             — diagnostic plots

USAGE:
    python run.py train hits-walks

REQUIRES:
    - data/pitcher_model_features.csv (from features.py)
    - models/bf_model.joblib          (from train/strikeouts.py, reused)
    - models/beta_binom_config.json   (from 06, for bf_features + σ_N)
    - pip install pandas numpy scikit-learn xgboost lightgbm scipy matplotlib seaborn joblib
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from scipy.special import betaln, gammaln
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import xgboost as xgb

# Shared with the strikeout model — the collinearity pruner is identical
# across stats, so it lives there and is imported rather than duplicated.
# strikeouts.py gates its training behind __main__, so importing is safe.
from pitcher_model.train.strikeouts import prune_collinear_features

# ── Paths ────────────────────────────────────────────────────────────────────
from pitcher_model.paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR, MODEL_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
TEST_YEAR    = 2025
MIN_PA_GAME  = 6
RANDOM_STATE = 42

# Features that MUST be excluded to prevent leakage.
# These are current-game outcome stats. The feature engineering script
# already labels most of these, but we enforce the exclusion here too.
# (Mirrored from train/strikeouts.py's RAW_STAT_EXCLUSIONS — kept in
# sync so the H/W rate models see the same safe feature set as the K rate
# model. Targets h_per_pa / bb_per_pa are added below.)
RAW_STAT_EXCLUSIONS = {
    # Current-game outcomes
    "strikeouts", "batters_faced", "plate_appearances", "hits_allowed",
    "walks", "earned_runs", "runs", "home_runs_allowed", "outs_recorded",
    "innings_pitched", "pitch_count", "pitches", "hit_by_pitch", "hbp",
    "total_pitches", "batted_balls", "strikes", "balls",
    "whiffs", "called_strikes", "in_zone_pitches", "out_of_zone_pitches",
    "chases", "barrels", "hard_hits", "soft_hits",
    "singles", "doubles", "triples",
    # ── Current-game batted-ball mix counts (added by build_batted_ball_features) ──
    # These are CURRENT-game counts — including them would leak the answer.
    # The rolling/season versions (gb_pct_L5, fb_pct_szn, etc.) are safe and
    # picked up by the _L*/_szn patterns.
    "ground_balls", "fly_balls", "line_drives", "popups", "infield_fly_balls",
    # ── Sweet-spot / solid-contact counts (added by updated aggregate_pitcher_game) ──
    # Current-game count flags from 01's batted-ball classification.
    "sweet_spot_hits", "solid_contact_hits",
    # ── Outs (added by ER-merge via collect_earned_runs in 01) ──
    # outs = 3 * IP (rounded), leaks BF directly.
    "outs",
    # Current-game derived
    "k_pct", "bb_pct", "k_per_pa", "k_per_9", "k_per_100_pitches",
    "pitches_per_k", "pitches_per_bf", "whiff_pct", "csw_pct", "chase_rate", "zone_pct",
    "barrel_pct", "hard_hit_pct", "est_innings", "is_short_outing",
    "k_bb_pct", "strike_pct", "soft_hit_pct", "outs_per_pa",
    # ── Current-game H/W rates (added by build_rolling_features pre-roll) ──
    # Same idea as k_pct above — these are the per-game rates that derive
    # directly from the target columns. The lagged versions are safe.
    "hits_per_pa", "bb_per_pa", "hr_per_pa",
    "h_per_9", "bb_per_9", "hr_per_9",
    "hr_per_bip", "hr_per_fb",
    "k_minus_bb_pct",
    "babip", "lob_pct",
    "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
    "avg_exit_velocity", "avg_launch_angle",
    "sweet_spot_pct", "solid_contact_pct",
    "avg_xba_contact", "avg_xwoba_contact",
    # Current-game platoon stats
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
    # Targets (for hits/walks model — h_per_pa/bb_per_pa added at runtime
    # by load_and_prepare; also include K targets so they don't leak into
    # the H/W rate models)
    "h_per_pa", "bb_per_pa",
    "target_strikeouts", "target_k_pct", "target_whiff_pct",
    "target_hits_allowed", "target_walks", "target_home_runs",
    "target_total_pitches", "target_outs_recorded",
    "target_k_over_4_5", "target_k_over_5_5", "target_k_over_6_5",
    # New H/W binary targets (added by add_targets in 02)
    "target_h_over_4_5", "target_h_over_5_5", "target_h_over_6_5",
    "target_bb_over_1_5", "target_bb_over_2_5", "target_bb_over_3_5",
}

# Patterns that identify safe engineered features (matched as SUBSTRINGS, not prefixes).
# Mirrored from train/strikeouts.py's FEATURE_PATTERNS — kept in sync so
# H/W and K rate models pick up the same engineered features.
FEATURE_PATTERNS = [
    # Rolling pitcher stats (suffix convention: bb_pct_L3, hits_per_pa_L10, etc.)
    "_L3", "_L5", "_L10",
    # Season cumulative (also matches new _szn_blended empirical-Bayes features)
    "_szn",
    # Prior-season carryover (last 5/10 starts of previous season).
    # These were computed in 02's build_early_season_features() but never
    # previously picked up by the feature filter.
    "_prev5", "_prev10",
    # Blended / empirical-Bayes features (from build_blended_rate_features
    # in 02). _szn_blended already matches via "_szn", but _L5_blended only
    # matches via "_L5". Listed here for clarity and to future-proof.
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
    # BF-specific features (kept since the rate models may benefit from BF context)
    "bf_L", "bf_season", "bf_trend", "bf_pitch", "bf_short", "bf_deep",
    "bf_prior", "bf_has_prior", "bf_vs_prior",
    # Interaction features (archetype bias fix)
    "ix_",
    # Normalized K metrics (useful proxies for stuff/command even in H/W context)
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
    # Rolling window observation counts
    "n_starts_in_L",
    # Opposing starter features
    "opp_sp_", "ix_both_deep", "ix_aces_matchup", "ix_combined_depth",
    # FanGraphs pitcher quality
    "fg_stuff", "fg_location", "fg_pitching", "fg_swstr", "fg_o_swing",
    "fg_z_swing", "fg_contact", "fg_o_contact", "fg_z_contact",
    "fg_zone_pct", "fg_first_strike", "fg_tto", "fg_loc_", "fg_pitcher_frm",
    # Catcher framing
    "catcher_frm",
    # Lineup plate discipline
    "lu_swstr", "lu_o_swing", "lu_z_contact", "lu_contact_pct",
    "lu_tto", "lu_csw", "lu_fg_k", "lu_barrel", "lu_hard_hit",
    "lu_k_rate_std", "lu_tto_pct_std", "lu_max_k", "lu_min_k",
    # Velocity delta & rolling platoon rates
    "velo_delta", "platoon_whiff_diff", "pitcher_tto_L", "pitcher_tto_szn",
    # New interaction features
    "ix_stuff_x_", "ix_pitcher_lu_", "ix_tto_matchup", "ix_swstr_x_contact",
    # Per-batter slot features (b1_ through b9_)
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
    # Delta features
    "delta_",
    # Last 1 start (ultra-short-term)
    "_L1",
    # Pitcher × umpire interactions
    "ix_pitcher_csw_x_ump", "ix_pitcher_edge_x_ump",
    "ix_pitcher_k_x_ump", "ix_pitcher_bb_x_ump",
    # Pitcher × catcher framing
    "ix_catcher_frm_x_",
    # Pitch count fatigue
    "pitchcount_", "heavy_prev_start", "pitches_per_out",
    # Enhanced lineup distribution
    "lu_k_rate_median", "lu_top3_k", "lu_bot3_k", "lu_top3_bot3_gap",
    # Pitch-type matchup score
    "pitch_matchup_score", "pitch_k_matchup_score",
    # ── HITS / WALKS pipeline additions ──
    # Standalone hit-rate stats (no _L/_szn suffix on the FG ones since
    # they're season-level only). The rolling/cumulative versions
    # (hits_per_pa_L5, babip_szn, etc.) are already covered by the _L*
    # and _szn patterns above.
    "hits_per_pa", "bb_per_pa", "hr_per_pa",
    "h_per_9", "bb_per_9", "hr_per_9",
    "hr_per_bip", "hr_per_fb",
    "k_minus_bb_pct",
    "babip", "lob_pct",
    # Batted-ball mix (pitcher's allowed batted-ball type rates)
    "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
    # Contact quality (what the BIP looked like when hit)
    "avg_exit_velocity", "avg_launch_angle",
    "sweet_spot_pct", "solid_contact_pct",
    "avg_xba_contact", "avg_xwoba_contact",
    "soft_hit_pct",
    # FanGraphs hits-side metrics (FIP/xFIP/SIERA family + batted-ball mix
    # + BABIP allowed + hard-contact-allowed). These are the season-level
    # talent indicators that K models don't need but hit-suppression
    # prediction lives or dies on.
    "fg_fip", "fg_xfip", "fg_siera", "fg_tera", "fg_xera",
    "fg_era", "fg_whip", "fg_lob_pct", "fg_hr_per_fb",
    "fg_k_minus_bb_pct", "fg_k_per_9", "fg_bb_per_9", "fg_hr_per_9",
    "fg_gb_pct", "fg_fb_pct", "fg_ld_pct", "fg_iffb_pct",
    "fg_babip_allowed", "fg_soft_pct", "fg_med_pct",
    "fg_hard_pct_allowed", "fg_barrel_pct_allowed", "fg_hard_hit_pct_allowed",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & TARGET CREATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load feature matrix, create rate targets for hits and walks, filter bad rows.

    Returns a DataFrame with h_per_pa, bb_per_pa, actual_H, actual_BB, and
    batters_faced columns added (rate models use BF only as denominator at
    evaluation time — they do NOT use BF as a feature).
    """
    path = DATA_DIR / "pitcher_model_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run features.py first.")

    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Create BF column (matches 06's logic exactly) ───────────────────
    if "batters_faced" not in df.columns:
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

    # ── Create rate targets ─────────────────────────────────────────────
    bf = df["batters_faced"].replace(0, np.nan)
    df["h_per_pa"]  = df["hits_allowed"] / bf
    df["bb_per_pa"] = df["walks"] / bf

    # ── Filter ──────────────────────────────────────────────────────────
    initial = len(df)
    df = df[df["batters_faced"] >= MIN_PA_GAME].copy()
    df = df.dropna(subset=["h_per_pa", "bb_per_pa", "hits_allowed", "walks"])
    df = df[df["h_per_pa"].between(0, 1)]
    df = df[df["bb_per_pa"].between(0, 1)]
    print(f"  Loaded {initial:,} rows → {len(df):,} after filtering "
          f"(BF >= {MIN_PA_GAME}, valid targets)")

    # ── Verify we have the raw counts for evaluation ────────────────────
    df["actual_H"]  = df["hits_allowed"].astype(int)
    df["actual_BB"] = df["walks"].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df, target):
    """
    Select features for a given target, avoiding leakage.
    Uses SUBSTRING matching (not prefix) to catch features like bb_pct_L3,
    hits_per_pa_szn, etc. which have the pattern as a suffix.

    Mirrors train/strikeouts.py's select_features() exactly.
    """
    candidates = []
    skip_cols = {"game_date", "game_pk", "pitcher", "pitcher_name",
                 "team", "opponent", "venue", "umpire",
                 "actual_K", "actual_H", "actual_BB",
                 "year", "player_name", "start_number_in_season",
                 "prior_year", "h_per_pa", "bb_per_pa",
                 "batters_faced",
                 "pitcher_team", "opp_team", "venue_name", "venue_id",
                 "hp_umpire_name", "hp_umpire_id",
                 "home_team", "away_team", "home_team_name", "away_team_name",
                 "home_team_id", "away_team_id",
                 "home_starter_id", "away_starter_id", "opp_starter_id",
                 "season", "p_throws", "day_night",
                 "latitude", "longitude",
                 "catcher_id", "catcher_name", "target_strikeouts"}

    for col in df.columns:
        # Skip identifiers, dates, targets, and raw stats
        if col in RAW_STAT_EXCLUSIONS or col in skip_cols:
            continue
        if col == target:
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


# ══════════════════════════════════════════════════════════════════════════════
# TIME-BASED TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def time_split(df, feature_cols, target_col):
    """Split by year: everything before TEST_YEAR = train, TEST_YEAR = test.

    Mirrors train/strikeouts.py's time_split() — uses NaN→0 fill for
    features (tree models handle this fine) while still dropping any row
    missing the target.
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    df["year"] = df["game_date"].dt.year

    train = df[df["year"] < TEST_YEAR].copy()
    test  = df[df["year"] >= TEST_YEAR].copy()

    train = train.dropna(subset=[target_col])
    test  = test.dropna(subset=[target_col])

    X_train = train[feature_cols].fillna(0).values
    y_train = train[target_col].values
    X_test  = test[feature_cols].fillna(0).values
    y_test  = test[target_col].values

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

    Mirrors train/strikeouts.py's train_xgb() exactly. Also trains
    LightGBM and Ridge as fallbacks so we can pick the best by MAE.
    Returns (results_dict, best_name).
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Safety: ensure enough samples per fold
    cv_folds = min(cv_folds, max(2, len(X_train) // 100))

    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    results = {}

    # ── XGBoost ──────────────────────────────────────────────────────────
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
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
    )
    search = RandomizedSearchCV(
        xgb_model, xgb_params, n_iter=n_iter,
        cv=tscv, scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    preds = best_xgb.predict(X_test)
    mae  = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    r2   = float(1 - np.sum((y_test - preds) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    cv_mae = float(-search.best_score_)
    print(f"    XGBoost — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, CV MAE: {cv_mae:.4f}")
    results["XGBoost"] = {
        "model": best_xgb, "preds": preds,
        "MAE": mae, "RMSE": rmse, "R²": r2, "CV_MAE": cv_mae,
    }

    # ── LightGBM ─────────────────────────────────────────────────────────
    try:
        import lightgbm as lgb
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
        }
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        lgb_model = lgb.LGBMRegressor(
            objective="regression", random_state=RANDOM_STATE, verbosity=-1,
        )
        search_lgb = RandomizedSearchCV(
            lgb_model, lgb_params, n_iter=n_iter,
            cv=tscv, scoring="neg_mean_absolute_error",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        )
        search_lgb.fit(X_train, y_train)
        best_lgb = search_lgb.best_estimator_
        preds_lgb = best_lgb.predict(X_test)
        mae_lgb  = float(np.mean(np.abs(y_test - preds_lgb)))
        rmse_lgb = float(np.sqrt(np.mean((y_test - preds_lgb) ** 2)))
        r2_lgb   = float(1 - np.sum((y_test - preds_lgb) ** 2) / ss_tot) if ss_tot > 0 else 0.0
        cv_mae_lgb = float(-search_lgb.best_score_)
        print(f"    LightGBM — MAE: {mae_lgb:.4f}, RMSE: {rmse_lgb:.4f}, R²: {r2_lgb:.4f}, CV MAE: {cv_mae_lgb:.4f}")
        results["LightGBM"] = {
            "model": best_lgb, "preds": preds_lgb,
            "MAE": mae_lgb, "RMSE": rmse_lgb, "R²": r2_lgb, "CV_MAE": cv_mae_lgb,
        }
    except ImportError:
        print("  ⚠ lightgbm not installed, skipping")

    # ── Ridge ────────────────────────────────────────────────────────────
    print(f"\n  Training Ridge for '{target_name}'...")
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    preds_r = ridge.predict(X_test)
    mae_r  = float(np.mean(np.abs(y_test - preds_r)))
    rmse_r = float(np.sqrt(np.mean((y_test - preds_r) ** 2)))
    r2_r   = float(1 - np.sum((y_test - preds_r) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    print(f"    Ridge — MAE: {mae_r:.4f}, RMSE: {rmse_r:.4f}, R²: {r2_r:.4f}")
    results["Ridge"] = {"model": ridge, "preds": preds_r,
                        "MAE": mae_r, "RMSE": rmse_r, "R²": r2_r}

    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n  ✓ Best for '{target_name}': {best_name} "
          f"(MAE={results[best_name]['MAE']:.4f})")
    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# BETA-BINOMIAL MATH (mirrored from 06)
# ══════════════════════════════════════════════════════════════════════════════

def beta_binom_pmf(k, n, alpha, beta_param):
    if n < 0 or k < 0 or k > n:
        return 0.0
    log_pmf = (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta_param)
        - betaln(alpha, beta_param)
    )
    return float(np.exp(log_pmf))


def beta_binom_pmf_array(n, alpha, beta_param):
    ks = np.arange(n + 1)
    log_comb = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    log_beta_num = betaln(ks + alpha, n - ks + beta_param)
    log_beta_den = betaln(alpha, beta_param)
    log_pmf = log_comb + log_beta_num - log_beta_den
    pmf = np.exp(log_pmf)
    pmf = pmf / pmf.sum()
    return pmf


def expected_pmf_over_N(pred_p, pred_N, kappa, sigma_N, max_k=20):
    """Marginalize Beta-Binomial PMF over a Normal(pred_N, sigma_N) on N."""
    pred_p = np.clip(pred_p, 0.01, 0.99)
    alpha = max(pred_p * kappa, 0.01)
    beta_param = max((1 - pred_p) * kappa, 0.01)

    min_N = max(1, int(pred_N - 3 * sigma_N))
    max_N = int(pred_N + 3 * sigma_N) + 1
    N_values  = np.arange(min_N, max_N + 1)
    N_weights = sp_stats.norm.pdf(N_values, loc=pred_N, scale=max(sigma_N, 0.5))
    N_weights = N_weights / N_weights.sum()

    combined_pmf = np.zeros(max_k + 1)
    for n_val, w in zip(N_values, N_weights):
        pmf = beta_binom_pmf_array(n_val, alpha, beta_param)
        if len(pmf) <= max_k + 1:
            combined_pmf[:len(pmf)] += w * pmf
        else:
            combined_pmf += w * pmf[:max_k + 1]

    total = combined_pmf.sum()
    if total > 0:
        combined_pmf /= total
    return combined_pmf


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION (mirrored from 06)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_kappa(pred_p, actual_N, actual_count, label=""):
    """Find κ that maximizes log-likelihood using actual N.

    Mirrors train/strikeouts.py's calibrate_kappa(): grid search →
    bounded fine-tune in log-space, with a hard cap at κ=500 (raised from
    200 because hits/walks rates are lower-noise than the K rate and the
    old cap was forcing the BB distribution to look like a Normal). If κ
    actually wants 400+, that's diagnostic — it tells us we've squeezed
    out most of the genuine variance in the rate predictions.
    """
    kappa_grid = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500]

    def compute_ll(kappa):
        total_ll = 0.0
        for p_i, n_i, c_i in zip(pred_p, actual_N, actual_count):
            n_i = int(round(n_i))
            c_i = int(c_i)
            if n_i < c_i:
                n_i = c_i
            if n_i <= 0:
                continue
            p_clamped = np.clip(p_i, 0.005, 0.995)
            alpha = p_clamped * kappa
            beta_p = (1 - p_clamped) * kappa
            prob = beta_binom_pmf(c_i, n_i, alpha, beta_p)
            total_ll += np.log(max(prob, 1e-15))
        return total_ll

    grid_results = [(kap, compute_ll(kap)) for kap in kappa_grid]
    grid_results.sort(key=lambda x: x[1], reverse=True)

    print(f"    κ grid search ({label}, top 5):")
    for kap, ll in grid_results[:5]:
        print(f"      κ={kap:>6.0f}  →  LL={ll:>10.2f}")

    best_grid = grid_results[0][0]
    low  = max(2, best_grid * 0.4)
    high = min(500.0, best_grid * 2.5)  # raised cap to 500

    result = minimize_scalar(
        lambda lk: -compute_ll(np.exp(lk)),
        bounds=(np.log(low), np.log(high)), method="bounded",
    )
    kappa_opt = float(np.exp(result.x))
    final_ll = float(-result.fun)

    # Hard cap for safety
    KAPPA_MAX = 500.0
    if kappa_opt > KAPPA_MAX:
        print(f"    ⚠ κ={kappa_opt:.0f} exceeded cap; clamping to {KAPPA_MAX}")
        kappa_opt = KAPPA_MAX
    if kappa_opt > 300:
        print(f"    ⚠ WARNING: κ={kappa_opt:.0f} is very high — distribution very tight.")
        print(f"             Check for residual leakage if this seems implausible.")

    print(f"    ✓ Calibrated κ = {kappa_opt:.2f} (LL = {final_ll:.2f})")
    return kappa_opt


def calibrate_kappa_cv(X_train, y_train_rate, y_train_bf, y_train_count,
                       feature_cols, rate_model, label="", n_folds=5):
    """Cross-validated κ calibration. Mirrors 06's version."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.base import clone

    tscv = TimeSeriesSplit(n_splits=n_folds)
    all_pred_p, all_actual_N, all_actual_count = [], [], []

    print(f"    Running {n_folds}-fold CV for κ calibration ({label})...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        try:
            fold_model = clone(rate_model)
        except Exception:
            import copy
            fold_model = copy.deepcopy(rate_model)

        fold_model.fit(X_train[train_idx], y_train_rate[train_idx])
        fold_preds = fold_model.predict(X_train[val_idx])

        fold_N = y_train_bf[val_idx]
        fold_count = y_train_count[val_idx]

        valid = (fold_N > 0) & np.isfinite(fold_preds) & np.isfinite(fold_count)
        all_pred_p.append(fold_preds[valid])
        all_actual_N.append(fold_N[valid])
        all_actual_count.append(fold_count[valid])

    oof_p = np.concatenate(all_pred_p)
    oof_N = np.concatenate(all_actual_N)
    oof_count = np.concatenate(all_actual_count)
    print(f"    Total OOF predictions: {len(oof_p)}")

    return calibrate_kappa(oof_p, oof_N, oof_count, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION (mirrored from 06's evaluate_distributions)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_distributions(pred_p, pred_N, actual_count, actual_N,
                           kappa, sigma_N, lines, label=""):
    """Compare Beta-Binomial vs Normal point + calibration metrics.

    Mirrors train/strikeouts.py's evaluate_distributions() but
    parameterized for any stat (label / lines).
    """
    NORMAL_STD = float(np.std(actual_count - pred_p * pred_N))

    bb_ll, norm_ll = 0.0, 0.0
    bb_preds, norm_preds = [], []
    bb_over = {l: [] for l in lines}
    norm_over = {l: [] for l in lines}

    for p_i, n_i, c_i, n_act in zip(pred_p, pred_N, actual_count, actual_N):
        c_i = int(c_i)
        pred_count = p_i * n_i

        # Beta-Binomial
        pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=20)
        ek  = sum(i * pmf[i] for i in range(len(pmf)))
        bb_preds.append(ek)
        prob = pmf[c_i] if c_i < len(pmf) else 1e-15
        bb_ll += np.log(max(prob, 1e-15))
        for l in lines:
            bb_over[l].append(pmf[l:].sum() if l < len(pmf) else 0.0)

        # Normal
        norm_preds.append(pred_count)
        norm_prob = sp_stats.norm.pdf(c_i, loc=pred_count, scale=max(NORMAL_STD, 0.5))
        norm_ll += np.log(max(norm_prob, 1e-15))
        for l in lines:
            z = (l - 0.5 - pred_count) / max(NORMAL_STD, 0.5)
            norm_over[l].append(1 - sp_stats.norm.cdf(z))

    actual  = np.array(actual_count, dtype=float)
    bb_arr  = np.array(bb_preds)
    norm_arr = np.array(norm_preds)

    print(f"\n  {label} — Model Comparison:")
    print(f"  {'Metric':25s} {'Beta-Binomial':>15s} {'Normal':>15s}")
    print(f"  {'-' * 58}")
    print(f"  {'Point MAE':25s} {np.mean(np.abs(actual - bb_arr)):>15.3f} {np.mean(np.abs(actual - norm_arr)):>15.3f}")
    print(f"  {'Point RMSE':25s} {np.sqrt(np.mean((actual - bb_arr)**2)):>15.3f} {np.sqrt(np.mean((actual - norm_arr)**2)):>15.3f}")
    print(f"  {'Log-Likelihood':25s} {bb_ll:>15.0f} {norm_ll:>15.0f}")
    print(f"  {'Mean Prediction':25s} {bb_arr.mean():>15.2f} {norm_arr.mean():>15.2f}")
    print(f"  {'Mean Actual':25s} {actual.mean():>15.2f} {'':>15s}")

    print(f"\n  Calibration (Brier scores — lower is better):")
    bb_calibration   = {}
    norm_calibration = {}
    for l in lines:
        bb_probs    = np.array(bb_over[l])
        norm_probs  = np.array(norm_over[l])
        actual_over = (actual >= l).astype(float)
        bb_brier   = float(np.mean((bb_probs   - actual_over) ** 2))
        norm_brier = float(np.mean((norm_probs - actual_over) ** 2))
        winner = "← BB" if bb_brier <= norm_brier else "← Normal"
        print(f"    {label} ≥ {l}:  BB {bb_brier:.4f}  |  Normal {norm_brier:.4f}  {winner}")

        # Bin reliability data (predicted vs observed by probability bucket).
        # Same bucketing as 06's evaluate_distributions so plot_calibration
        # produces comparable reliability diagrams.
        bb_bins   = _compute_calibration_bins(bb_probs,   actual_over)
        norm_bins = _compute_calibration_bins(norm_probs, actual_over)
        bb_calibration[l] = {
            "brier": bb_brier, "bins": bb_bins,
            "overall_pred":   float(bb_probs.mean()),
            "overall_actual": float(actual_over.mean()),
        }
        norm_calibration[l] = {
            "brier": norm_brier, "bins": norm_bins,
            "overall_pred":   float(norm_probs.mean()),
            "overall_actual": float(actual_over.mean()),
        }

    summary = {
        "beta_binom": {
            "MAE":   float(np.mean(np.abs(actual - bb_arr))),
            "RMSE":  float(np.sqrt(np.mean((actual - bb_arr) ** 2))),
            "log_lik": float(bb_ll),
            "mean_pred": float(bb_arr.mean()),
            "calibration": bb_calibration,
            "over_probs": {l: np.array(bb_over[l]).tolist() for l in lines},
        },
        "normal": {
            "MAE":   float(np.mean(np.abs(actual - norm_arr))),
            "RMSE":  float(np.sqrt(np.mean((actual - norm_arr) ** 2))),
            "log_lik": float(norm_ll),
            "mean_pred": float(norm_arr.mean()),
            "calibration": norm_calibration,
            "over_probs": {l: np.array(norm_over[l]).tolist() for l in lines},
        },
        "actual_mean": float(actual.mean()),
        "normal_std": NORMAL_STD,
        "lines": list(lines),
    }
    return summary


def _compute_calibration_bins(probs, actual_over):
    """Bin predicted probabilities and return (pred_mean, actual_rate, n) per bin.

    Matches 06's bucketing exactly: 10 buckets from [0, 0.1, ..., 1.01] and a
    minimum of 10 samples per bucket to report. Returns a list of tuples so
    downstream plotters can do reliability scatter plots.
    """
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    cal_data = []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() >= 10:
            cal_data.append((
                float(probs[mask].mean()),
                float(actual_over[mask].mean()),
                int(mask.sum()),
            ))
    return cal_data


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING (mirrored from 06)
# ══════════════════════════════════════════════════════════════════════════════

def plot_submodel_diagnostics(y_test, preds, target_name, output_dir):
    """Pred vs actual scatter + residual histogram for a single sub-model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(y_test, preds, alpha=0.3, s=8)
    lim_lo = float(min(y_test.min(), preds.min()))
    lim_hi = float(max(y_test.max(), preds.max()))
    axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=1)
    axes[0].set_xlabel(f"Actual {target_name}")
    axes[0].set_ylabel(f"Predicted {target_name}")
    axes[0].set_title(f"{target_name}: predicted vs actual")
    axes[0].grid(alpha=0.3)

    resid = preds - y_test
    axes[1].hist(resid, bins=40, alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="r", ls="--", lw=1)
    axes[1].set_xlabel(f"Residual (pred - actual)")
    axes[1].set_title(f"{target_name}: residual distribution "
                      f"(mean={resid.mean():+.4f}, std={resid.std():.4f})")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Sub-model diagnostics: {target_name}", y=1.02)
    plt.tight_layout()
    out = output_dir / f"{target_name.lower().replace(' ', '_')}_submodel_diagnostics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_calibration(summary, stat_label, output_dir):
    """Reliability diagrams: predicted probability vs observed frequency.

    Mirrors 06's plot_calibration: scatter of (predicted, actual) by
    probability bucket, with marker size proportional to sample size and
    a 45° perfect-calibration reference line. Drawn one panel per line.
    """
    lines = summary["lines"]
    if not lines:
        return
    bb_cal   = summary["beta_binom"]["calibration"]
    norm_cal = summary["normal"]["calibration"]

    # Reliability scatter (one panel per line)
    n_lines = len(lines)
    fig, axes = plt.subplots(1, n_lines, figsize=(4.5 * n_lines, 4.5))
    if n_lines == 1:
        axes = [axes]

    for ax, l in zip(axes, lines):
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        for method_data, label_txt, color in [
            (bb_cal[l],   "Beta-Binomial", "#2196F3"),
            (norm_cal[l], "Normal",        "#FF9800"),
        ]:
            bins = method_data.get("bins", [])
            if bins:
                pred_vals   = [b[0] for b in bins]
                actual_vals = [b[1] for b in bins]
                sizes       = [b[2] for b in bins]
                ax.scatter(pred_vals, actual_vals,
                           s=[max(15, s / 5) for s in sizes],
                           color=color, alpha=0.7, label=label_txt,
                           edgecolors="black", linewidth=0.5)

        ax.set_xlabel(f"Predicted P({stat_label[0].upper()} ≥ {l})")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{stat_label}: calibration at ≥ {l}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"{stat_label.lower().replace(' ', '_')}_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Bar-chart summary (Brier per line, BB vs Normal) — kept for at-a-glance
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(lines))
    w = 0.35
    bb_briers   = [bb_cal[l]["brier"]   for l in lines]
    norm_briers = [norm_cal[l]["brier"] for l in lines]
    ax.bar(x - w/2, bb_briers,   w, label="Beta-Binomial", color="#2196F3")
    ax.bar(x + w/2, norm_briers, w, label="Normal (fixed σ)", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{stat_label[0].upper()}≥{l}" for l in lines])
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title(f"{stat_label}: BB vs Normal Brier", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = output_dir / f"{stat_label.lower().replace(' ', '_')}_brier_bars.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(summary, stat_label, output_dir):
    """Bar charts: MAE, log-likelihood, and median-line Brier (BB vs Normal).

    Mirrors 06's plot_comparison but parameterized per stat. The middle Brier
    bar uses the median requested line (e.g. H≥5 for hits, BB≥3 for walks)
    so the at-a-glance comparison covers the most common prop level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ["Beta-Binomial", "Normal (fixed σ)"]
    colors = ["#2196F3", "#FF9800"]

    # MAE
    ax = axes[0]
    vals = [summary["beta_binom"]["MAE"], summary["normal"]["MAE"]]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"{stat_label}: Point Prediction MAE", fontweight="bold")
    ax.set_ylabel(f"MAE ({stat_label.lower()})")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Log-likelihood
    ax = axes[1]
    vals = [summary["beta_binom"]["log_lik"], summary["normal"]["log_lik"]]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"{stat_label}: Log-Likelihood (higher = better)", fontweight="bold")
    ax.set_ylabel("Total log-likelihood")
    for i, v in enumerate(vals):
        ax.text(i, v * 0.98 if v < 0 else v * 1.02, f"{v:.0f}",
                ha="center", fontsize=10)

    # Brier at the median line
    ax = axes[2]
    lines = summary["lines"]
    if lines:
        mid_line = lines[len(lines) // 2]
        bb_brier   = summary["beta_binom"]["calibration"][mid_line]["brier"]
        norm_brier = summary["normal"]["calibration"][mid_line]["brier"]
        vals = [bb_brier, norm_brier]
        ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{stat_label}: Brier @ ≥{mid_line} (lower = better)",
                     fontweight="bold")
        ax.set_ylabel("Brier Score")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    out = output_dir / f"{stat_label.lower().replace(' ', '_')}_bb_vs_normal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_example_pmfs(pred_p_arr, pred_N_arr, actual_count_arr,
                      kappa, sigma_N, stat_label, output_dir, n_examples=6):
    """Show example PMFs alongside the actual outcome (six diverse picks).

    Mirrors 06's plot_example_pmfs: picks games at the 5th / 20th / 40th /
    60th / 80th / 95th percentiles of the actual count so the panel covers
    low, medium, and high outings.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    sorted_idx = np.argsort(actual_count_arr)
    n = len(sorted_idx)
    if n < 6:
        plt.close()
        return
    picks = [
        sorted_idx[int(n * 0.05)],
        sorted_idx[int(n * 0.20)],
        sorted_idx[int(n * 0.40)],
        sorted_idx[int(n * 0.60)],
        sorted_idx[int(n * 0.80)],
        sorted_idx[int(n * 0.95)],
    ]

    # x-range: large enough to cover typical hits/walks
    max_count = max(int(np.percentile(actual_count_arr, 99)) + 3, 15)

    for ax_idx, i in enumerate(picks[:n_examples]):
        ax = axes[ax_idx]
        p_i = pred_p_arr[i]
        n_i = pred_N_arr[i]
        c_actual = int(actual_count_arr[i])

        pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=max_count)
        ks = np.arange(len(pmf))
        ek = float(np.sum(ks * pmf))

        ax.bar(ks, pmf, color="#2196F3", alpha=0.6,
               label="Beta-Binomial PMF", edgecolor="black", linewidth=0.3)
        ax.axvline(c_actual, color="red", linewidth=2, linestyle="--",
                   label=f"Actual = {c_actual}")
        ax.axvline(ek, color="#2196F3", linewidth=2, linestyle=":",
                   label=f"E[{stat_label[0].upper()}] = {ek:.1f}")

        # Normal overlay (using the residual std as σ — same baseline used
        # in evaluate_distributions for the fixed-σ comparison)
        NORMAL_STD = float(np.std(actual_count_arr - pred_p_arr * pred_N_arr))
        pred_norm = p_i * n_i
        x_norm = np.linspace(0, max_count, 200)
        y_norm = sp_stats.norm.pdf(x_norm, loc=pred_norm, scale=max(NORMAL_STD, 0.5))
        # Scale to match bar heights
        if y_norm.max() > 0 and pmf.max() > 0:
            y_norm_scaled = y_norm * (pmf.max() / y_norm.max())
            ax.plot(x_norm, y_norm_scaled, color="#FF9800",
                    linewidth=1.5, alpha=0.8, label="Normal approx")

        ax.set_title(f"p̂={p_i:.3f}, N̂={n_i:.0f}", fontsize=10)
        ax.set_xlabel(stat_label)
        ax.set_ylabel("Probability")
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, max_count - 1)

    plt.suptitle(f"Beta-Binomial PMF vs Normal Approximation — {stat_label} "
                 f"(Example Games)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = output_dir / f"{stat_label.lower().replace(' ', '_')}_example_pmfs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ARCHETYPE BIAS CHECK (mirrored from 06)
# ══════════════════════════════════════════════════════════════════════════════

def archetype_bias_check(rate_preds, bf_preds, actual_count, stat_label):
    """Split test set into rate × depth quadrants and report MAE / bias.

    Mirrors 06's check: do high-rate-deep pitchers (lots of expected stat)
    actually get more than the model predicts? Compression toward the mean
    shows up as systematic bias here.
    """
    direct = rate_preds * bf_preds
    median_rate = float(np.median(rate_preds[rate_preds > 0])) if (rate_preds > 0).any() else float(np.median(rate_preds))
    median_bf   = float(np.median(bf_preds))

    archetypes = {
        f"High-{stat_label}, Deep":  (rate_preds >= median_rate) & (bf_preds >= median_bf),
        f"High-{stat_label}, Short": (rate_preds >= median_rate) & (bf_preds <  median_bf),
        f"Low-{stat_label}, Deep":   (rate_preds <  median_rate) & (bf_preds >= median_bf),
        f"Low-{stat_label}, Short":  (rate_preds <  median_rate) & (bf_preds <  median_bf),
    }

    print(f"    {'Archetype':>22s} {'Games':>6s} {'MAE':>7s} {'Bias':>8s} "
          f"{'Avg ' + stat_label:>9s} {'Avg Pred':>9s}")
    print(f"    {'-' * 68}")
    for name, mask in archetypes.items():
        if mask.sum() < 10:
            continue
        mae  = float(np.mean(np.abs(actual_count[mask] - direct[mask])))
        bias = float(np.mean(actual_count[mask] - direct[mask]))
        avg_act  = float(actual_count[mask].mean())
        avg_pred = float(direct[mask].mean())
        print(f"    {name:>22s} {mask.sum():>6d} {mae:>7.3f} {bias:>+8.3f} "
              f"{avg_act:>9.2f} {avg_pred:>9.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD EVALUATION (mirrored from 06's evaluate_thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_thresholds(pred_p, pred_N, actual_count, kappa, sigma_N,
                        stat_label, lines):
    """Score the Beta-Binomial PMF at each count threshold.

    For every line L, the model's P(count >= L) is scored against the
    realised outcome with proper scoring rules:

      Brier    — mean squared probability error. Lower is better.
      Log-loss — punishes confident-and-wrong. Lower is better.
      Accuracy — fraction correct at a 0.5 cutoff, for intuition only.

    Base rate is printed alongside so a threshold where the outcome is
    nearly always one-sided (e.g. walks >= 1) isn't mistaken for skill:
    a model that always predicts the base rate scores well on accuracy
    while carrying no information.
    """
    EPS = 1e-15
    print(f"\n  {stat_label} threshold evaluation")
    print(f"    {'Line':>4s} {'N':>6s} {'Base':>7s} {'Brier':>9s} "
          f"{'LogLoss':>9s} {'Acc':>7s}")
    print(f"    {'-' * 48}")

    for l in lines:
        probs, actuals = [], []
        for p_i, n_i, c_i in zip(pred_p, pred_N, actual_count):
            pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=20)
            probs.append(float(pmf[l:].sum()) if l < len(pmf) else 0.0)
            actuals.append(1 if c_i >= l else 0)

        if not probs:
            print(f"    {l:>4d} {0:>6d}       —         —         —       —")
            continue

        probs   = np.array(probs)
        actuals = np.array(actuals)
        pc      = np.clip(probs, EPS, 1 - EPS)

        base     = actuals.mean()
        brier    = np.mean((probs - actuals) ** 2)
        log_loss = -np.mean(actuals * np.log(pc) + (1 - actuals) * np.log(1 - pc))
        accuracy = np.mean((probs > 0.5) == actuals)

        print(f"    {l:>4d} {len(probs):>6d} {base:>6.1%} {brier:>9.4f} "
              f"{log_loss:>9.4f} {accuracy:>6.1%}")



# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS (mirrored from 06)
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, features, kappa, sigma_N, eval_summary,
               stat_name, model_dir):
    """Save a rate model and its config.

    Config schema matches what predict/hits_walks.py expects:
      - rate_features: list of feature names
      - kappa:         per-stat Beta concentration
      - sigma_n:       BF uncertainty (inherited from 06's BF model)
      - bb_mae:        point-prediction MAE on test
      - normal_std:    Normal-baseline σ for comparison
      - actual_mean:   test-set mean of the actual count
    """
    model_path  = model_dir / f"{stat_name}_rate_model.joblib"
    config_path = model_dir / f"{stat_name}_config.json"

    joblib.dump(model, model_path)

    config = {
        "stat":           stat_name,
        "rate_features":  features,
        "kappa":          float(kappa),
        "sigma_n":        float(sigma_N),
        "bb_mae":         eval_summary["beta_binom"]["MAE"],
        "normal_std":     eval_summary["normal_std"],
        "actual_mean":    eval_summary["actual_mean"],
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✓ Saved {model_path}")
    print(f"  ✓ Saved {config_path}  (κ={kappa:.2f}, σ_N={sigma_N:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# PER-STAT TRAINING WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_stat(df, rate_target, count_col, stat_label, lines,
                            bf_model, bf_features, bf_is_log, bf_log_sigma2,
                            sigma_N):
    """End-to-end training + evaluation for one stat (hits or walks).

    Returns: (best_model, features, kappa, eval_summary, best_name)
    """
    print(f"\n{'=' * 70}")
    print(f"  {stat_label.upper()} MODEL  ({rate_target} rate)")
    print(f"{'=' * 70}")

    # ── Feature selection ───────────────────────────────────────────────
    print(f"\n── Step 2: Feature Selection ({stat_label}) ──")
    rate_features = select_features(df, rate_target)

    # ── Collinearity pruning ────────────────────────────────────────────
    if prune_collinear_features is not None:
        print(f"\n── Step 2b: Collinearity Pruning ({stat_label}) ──")
        df_for_prune = df.copy()
        if "year" not in df_for_prune.columns and "game_date" in df_for_prune.columns:
            df_for_prune["year"] = pd.to_datetime(
                df_for_prune["game_date"], errors="coerce"
            ).dt.year
        rate_features, _ = prune_collinear_features(
            df_for_prune, rate_features, rate_target,
            corr_threshold=0.99, target_year_cutoff=TEST_YEAR,
        )
        del df_for_prune

    print(f"\n  Final feature count for {stat_label}: {len(rate_features)}")

    # ── Split ───────────────────────────────────────────────────────────
    print(f"\n── Step 3: Train/Test Split ({stat_label}) ──")
    X_train, X_test, y_train, y_test, train_df, test_df = time_split(
        df, rate_features, rate_target
    )

    # ── Train ───────────────────────────────────────────────────────────
    print(f"\n── Step 4: Train Rate Model ({stat_label}) ──")
    rate_results, best_name = train_xgb(
        X_train, y_train, X_test, y_test, rate_features, rate_target
    )
    best_model = rate_results[best_name]["model"]
    rate_preds = rate_results[best_name]["preds"]

    # ── BF predictions (test set) ───────────────────────────────────────
    test_actual_N  = test_df["batters_faced"].values.astype(float)
    test_actual_C  = test_df[count_col].values
    train_actual_N = train_df["batters_faced"].values.astype(float)
    train_actual_C = train_df[count_col].values

    if bf_model is not None and bf_features:
        avail = [f for f in bf_features if f in test_df.columns]
        if avail and len(avail) == len(bf_features):
            bf_preds_test = bf_model.predict(test_df[bf_features].fillna(0).values)
            if bf_is_log:
                bf_preds_test = np.exp(bf_preds_test) * np.exp(bf_log_sigma2 / 2)
        else:
            print(f"  ⚠ BF features missing in test rows — using actual N for evaluation")
            bf_preds_test = test_actual_N.copy()
    else:
        bf_preds_test = test_actual_N.copy()

    direct = rate_preds * bf_preds_test
    direct_mae = float(np.mean(np.abs(test_actual_C - direct)))
    print(f"\n  Direct point prediction (p̂ × N̂):")
    print(f"    MAE: {direct_mae:.3f}")
    print(f"    Mean actual: {test_actual_C.mean():.2f}, "
          f"Mean predicted: {direct.mean():.2f}")

    # ── Sub-model diagnostics ───────────────────────────────────────────
    print(f"\n── Step 5: Sub-Model Diagnostics ({stat_label}) ──")
    plot_submodel_diagnostics(y_test, rate_preds, f"{stat_label} rate", OUTPUT_DIR)
    print(f"  ✓ Diagnostic plot saved")

    # ── Archetype bias check ────────────────────────────────────────────
    print(f"\n── Step 5b: Archetype Bias Check ({stat_label}) ──")
    archetype_bias_check(rate_preds, bf_preds_test, test_actual_C, stat_label)

    # ── Calibrate κ ─────────────────────────────────────────────────────
    print(f"\n── Step 6: Calibrate Beta-Binomial κ ({stat_label}) ──")

    print(f"\n  Method A: κ from test-set predictions + actual N")
    kappa_test = calibrate_kappa(rate_preds, test_actual_N, test_actual_C,
                                 label=stat_label)

    print(f"\n  Method B: κ from cross-validated training predictions")
    kappa_cv = calibrate_kappa_cv(
        X_train, y_train, train_actual_N, train_actual_C,
        rate_features, best_model, label=stat_label, n_folds=5,
    )

    kappa = kappa_cv
    print(f"\n  → Using CV κ = {kappa:.2f} (test κ was {kappa_test:.2f})")

    # ── Evaluate distributions ──────────────────────────────────────────
    print(f"\n── Step 7: Evaluate Beta-Binomial vs Normal ({stat_label}) ──")
    eval_summary = evaluate_distributions(
        rate_preds, bf_preds_test, test_actual_C, test_actual_N,
        kappa, sigma_N, lines=lines, label=stat_label,
    )

    # ── Plots ───────────────────────────────────────────────────────────
    print(f"\n── Step 8: Diagnostic Plots ({stat_label}) ──")
    plot_calibration(eval_summary, stat_label, OUTPUT_DIR)
    print(f"  ✓ Calibration plots saved")
    plot_comparison(eval_summary, stat_label, OUTPUT_DIR)
    print(f"  ✓ BB-vs-Normal comparison plot saved")
    plot_example_pmfs(rate_preds, bf_preds_test, test_actual_C,
                      kappa, sigma_N, stat_label, OUTPUT_DIR)
    print(f"  ✓ Example PMF plots saved")

    # ── Threshold evaluation ────────────────────────────────────────────
    print(f"\n── Step 9: Threshold Evaluation ({stat_label}) ──")
    evaluate_thresholds(rate_preds, bf_preds_test, test_actual_C,
                        kappa, sigma_N, stat_label, lines)

    return best_model, rate_features, kappa, eval_summary, best_name


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  HITS ALLOWED & WALKS — BETA-BINOMIAL MODELS")
    print("  H/PA, BB/PA ~ Beta(p·κ, (1-p)·κ);  count ~ BetaBinomial(N, α, β)")
    print("=" * 70)

    # ── Load & prepare ───────────────────────────────────────────────────
    print("\n── Step 1: Load & Prepare Data ──")
    df = load_and_prepare()

    # ── Load BF model + config from 06 ───────────────────────────────────
    bb_config_path = MODEL_DIR / "beta_binom_config.json"
    bf_model_path  = MODEL_DIR / "bf_model.joblib"

    if not bb_config_path.exists():
        raise FileNotFoundError(
            f"{bb_config_path} not found. Run train/strikeouts.py first."
        )
    with open(bb_config_path) as f:
        bb_config = json.load(f)

    # σ_N: read whichever convention 06 wrote (capital N is older, lower-n newer)
    sigma_N = float(
        bb_config.get("sigma_n") or bb_config.get("sigma_n_global") or
        bb_config.get("sigma_N") or bb_config.get("sigma_N_global") or 2.5
    )
    print(f"  Using σ_N = {sigma_N:.2f} from 06's BF config")

    bf_features   = bb_config.get("bf_features", None)
    bf_is_log     = bool(bb_config.get("bf_is_log", False))
    bf_log_sigma2 = float(bb_config.get("bf_log_sigma2", 0.0))

    if bf_model_path.exists():
        bf_model = joblib.load(bf_model_path)
        scale_note = " (log scale)" if bf_is_log else ""
        print(f"  ✓ Loaded BF model from 06 for N predictions{scale_note}")
    else:
        bf_model = None
        print(f"  ⚠ No BF model found at {bf_model_path} — "
              "will use actual N for calibration only")

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN BOTH STATS
    # ══════════════════════════════════════════════════════════════════════
    h_model, h_features, kappa_h, h_eval, h_best = train_and_evaluate_stat(
        df, rate_target="h_per_pa", count_col="actual_H",
        stat_label="Hits", lines=[3, 4, 5, 6, 7, 8],
        bf_model=bf_model, bf_features=bf_features,
        bf_is_log=bf_is_log, bf_log_sigma2=bf_log_sigma2,
        sigma_N=sigma_N,
    )

    w_model, w_features, kappa_w, w_eval, w_best = train_and_evaluate_stat(
        df, rate_target="bb_per_pa", count_col="actual_BB",
        stat_label="Walks", lines=[1, 2, 3, 4, 5],
        bf_model=bf_model, bf_features=bf_features,
        bf_is_log=bf_is_log, bf_log_sigma2=bf_log_sigma2,
        sigma_N=sigma_N,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SAVE MODELS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  SAVING MODELS")
    print(f"{'=' * 70}")
    save_model(h_model, h_features, kappa_h, sigma_N, h_eval, "hits",  MODEL_DIR)
    save_model(w_model, w_features, kappa_w, sigma_N, w_eval, "walks", MODEL_DIR)

    # ── Save comparison CSV ──────────────────────────────────────────────
    comparison = pd.DataFrame([
        {"stat": "hits",  "model": h_best,
         "point_MAE_bb":  h_eval["beta_binom"]["MAE"],
         "point_MAE_norm": h_eval["normal"]["MAE"],
         "log_lik_bb":    h_eval["beta_binom"]["log_lik"],
         "log_lik_norm":  h_eval["normal"]["log_lik"],
         "kappa": kappa_h, "mean_actual": h_eval["actual_mean"]},
        {"stat": "walks", "model": w_best,
         "point_MAE_bb":  w_eval["beta_binom"]["MAE"],
         "point_MAE_norm": w_eval["normal"]["MAE"],
         "log_lik_bb":    w_eval["beta_binom"]["log_lik"],
         "log_lik_norm":  w_eval["normal"]["log_lik"],
         "kappa": kappa_w, "mean_actual": w_eval["actual_mean"]},
    ])
    comparison.to_csv(OUTPUT_DIR / "hits_walks_comparison.csv", index=False)
    print(f"  ✓ Saved {OUTPUT_DIR / 'hits_walks_comparison.csv'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Hits:")
    print(f"    Rate Model:  {h_best}")
    print(f"    κ_h:         {kappa_h:.2f}")
    print(f"    BB Point MAE:{h_eval['beta_binom']['MAE']:.3f}  "
          f"(Normal: {h_eval['normal']['MAE']:.3f})")
    print(f"    LL improvement vs Normal: "
          f"{h_eval['beta_binom']['log_lik'] - h_eval['normal']['log_lik']:+.0f}")
    print(f"\n  Walks:")
    print(f"    Rate Model:  {w_best}")
    print(f"    κ_w:         {kappa_w:.2f}")
    print(f"    BB Point MAE:{w_eval['beta_binom']['MAE']:.3f}  "
          f"(Normal: {w_eval['normal']['MAE']:.3f})")
    print(f"    LL improvement vs Normal: "
          f"{w_eval['beta_binom']['log_lik'] - w_eval['normal']['log_lik']:+.0f}")
    print(f"\n  Outputs in: {OUTPUT_DIR.resolve()}/")
    print(f"  Models in:  {MODEL_DIR.resolve()}/")
    print(f"\n  Next step: python run.py predict hits-walks")


if __name__ == "__main__":
    main()
