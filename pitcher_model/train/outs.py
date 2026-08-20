"""
train/outs.py
================
Beta-Binomial Model for Pitcher Outs Recorded

Follows the same generative decomposition as the strikeout/hits/walks models:
    Outs:  o = O/PA rate × N  →  BetaBinomial(N, α_o, β_o)

Reuses the existing BF (plate_appearances) model for N predictions.
Only trains a new rate model for O/PA.

OUTPUTS:
    - models/outs_rate_model.joblib     — O/PA rate model
    - models/outs_config.json           — κ_o, features, etc.
    - output/outs_comparison.csv        — Evaluation metrics

USAGE:
    python run.py train outs

REQUIRES:
    - data/pitcher_model_features.csv  (with outs_recorded column — run 02 first)
    - models/bf_model.joblib (existing BF model, reused)
    - models/beta_binom_config.json
    - pip install pandas numpy scikit-learn xgboost lightgbm scipy joblib
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
import joblib
import json

from pitcher_model.train.strikeouts import prune_collinear_features

# ── Paths ────────────────────────────────────────────────────────────────────
from pitcher_model.paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR, MODEL_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
TEST_YEAR = 2025
MIN_PA_GAME = 6
RANDOM_STATE = 42

# Features that MUST be excluded to prevent leakage (current-game outcomes)
RAW_STAT_EXCLUSIONS = {
    # Direct outcome counts
    "strikeouts", "batters_faced", "plate_appearances", "hits_allowed",
    "walks", "earned_runs", "runs", "home_runs_allowed", "outs_recorded",
    "innings_pitched", "pitch_count", "pitches", "hit_by_pitch",
    # Current-game rates
    "k_pct", "bb_pct", "k_per_pa", "k_per_9", "k_per_100_pitches",
    "pitches_per_k", "pitches_per_bf", "whiff_pct", "csw_pct", "chase_rate",
    "zone_pct", "barrel_pct", "hard_hit_pct", "est_innings", "is_short_outing",
    "k_bb_pct", "strike_pct", "soft_hit_pct", "outs_per_pa",
    # Raw counts
    "singles", "doubles", "triples", "hbp", "balls", "strikes",
    "called_strikes", "chases", "whiffs", "in_zone_pitches",
    "out_of_zone_pitches", "whiffs_vs_left", "whiffs_vs_right",
    "barrels", "batted_balls", "hard_hits", "soft_hits", "total_pitches",
    # ── Hits/walks pipeline current-game additions (added by 02 v2) ──
    # All of these are current-game rates derived from outcome counts.
    # Including any of them would leak the answer (outs ↔ K + BIP-outs,
    # so hits_per_pa and bb_per_pa are mechanically constrained by outs_per_pa).
    "hits_per_pa", "bb_per_pa", "hr_per_pa",
    "h_per_9", "bb_per_9", "hr_per_9",
    "hr_per_bip", "hr_per_fb",
    "k_minus_bb_pct",
    "babip", "lob_pct",
    # Current-game batted-ball mix counts (from build_batted_ball_features in 02)
    "ground_balls", "fly_balls", "line_drives", "popups", "infield_fly_balls",
    # Current-game batted-ball mix RATES
    "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
    # Current-game contact quality
    "avg_exit_velocity", "avg_launch_angle",
    "sweet_spot_pct", "solid_contact_pct",
    "avg_xba_contact", "avg_xwoba_contact",
    # Raw pitch-type counts
    "ff_count", "sl_count", "cu_count", "ch_count", "si_count",
    "fc_count", "fs_count", "sv_count", "kc_count",
    "fastball_count", "breaking_count", "offspeed_count",
    # Targets
    "target_strikeouts", "target_k_pct", "target_whiff_pct",
    "target_walks", "target_hits_allowed", "target_home_runs",
    "target_total_pitches", "target_k_over_4_5", "target_k_over_5_5",
    "target_k_over_6_5", "target_outs_recorded",
    # New H/W binary targets
    "target_h_over_4_5", "target_h_over_5_5", "target_h_over_6_5",
    "target_bb_over_1_5", "target_bb_over_2_5", "target_bb_over_3_5",
    # PA-related
    "pa_vs_left", "pa_vs_right",
}

# Patterns that identify safe engineered features (substring match)
FEATURE_PATTERNS = [
    "_L3", "_L5", "_L10", "_szn", "trend",
    # ── Prior-season carryover and empirical-Bayes blends ──
    # _prev5/_prev10 are last-N-starts rates from previous season; _blended
    # is the empirical-Bayes shrinkage to a prior. The K model (06) uses
    # these via the same substring trick. Outs benefits equally because
    # early-season outs predictions need a stable prior on both K rate
    # and BIP-out rate.
    "_prev5", "_prev10", "_blended",
    "opp_", "lu_", "pitcher_x_lineup", "whiff_x_lineup", "k_trend_x_lu",
    "pt_", "cross_", "ump_", "pf_", "park_", "wx_", "lineup_",
    "bf_L", "bf_season", "bf_trend", "bf_pitch", "bf_short", "bf_deep",
    "bf_prior", "bf_has_prior", "bf_vs_prior", "ix_",
    "k_per_100", "k_per_9", "k_per_pa", "pitches_per_k",
    "est_innings", "is_short_outing", "outs_per_pa", "outs_recorded",
    "rest_days", "short_rest", "extra_rest",
    "is_home", "is_night", "is_day", "is_weekend", "is_dome",
    "day_of_week", "month", "start_num",
    "platoon", "vs_left", "vs_right", "pitcher_throws",
    "_std_", "pitch_mix_",
    "pa_std", "pa_range", "pa_trend", "pa_prior",
    "baserunner_rate", "hr_rate_L", "pitches_per_pa",
    "deep_outing", "recent_pitches",
    "pvt_", "prior_starts",
    "early_x_", "season_phase", "days_into_season",
    "is_first_month",
    "n_starts_in_L",
    # Opposing starter features (game-level pace/depth)
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
    # Velocity delta & platoon
    "velo_delta", "whiff_pct_vs_left", "whiff_pct_vs_right",
    "platoon_whiff_diff", "pitcher_tto_L", "pitcher_tto_szn",
    # New interactions
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
    # ══════════════════════════════════════════════════════════════════
    # HITS / WALKS PIPELINE ADDITIONS
    # ══════════════════════════════════════════════════════════════════
    # Outs depends on BOTH K rate and BIP-conversion. Hit-suppression
    # features (BABIP, batted-ball mix, contact quality) directly affect
    # the BIP-out rate, which is the second component of outs alongside K.
    # GB pitchers in particular convert more outs (double plays + ground
    # outs > air outs in conversion rate). So the outs model benefits from
    # the FULL hits/walks feature set, not just the K-side stats.
    #
    # Hit-rate stats (rolling/szn versions caught by _L*/_szn patterns above
    # already; these explicit names also catch the base columns when present
    # in safe lagged contexts):
    "hits_per_pa", "bb_per_pa", "hr_per_pa",
    "h_per_9", "bb_per_9", "hr_per_9",
    "hr_per_bip", "hr_per_fb",
    "k_minus_bb_pct",
    "babip", "lob_pct",
    # Batted-ball mix (GB/FB pitcher classification — huge for outs)
    "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
    # Contact quality
    "avg_exit_velocity", "avg_launch_angle",
    "sweet_spot_pct", "solid_contact_pct",
    "avg_xba_contact", "avg_xwoba_contact",
    "soft_hit_pct",
    # FanGraphs hits-side metrics — SIERA in particular is built around
    # K-BB% and GB% which is a direct outs predictor.
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
    """Load feature matrix, create rate target for outs recorded."""
    path = DATA_DIR / "pitcher_model_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run features.py first.")

    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Create BF column ─────────────────────────────────────────────────
    if "batters_faced" not in df.columns:
        if "plate_appearances" in df.columns:
            df["batters_faced"] = df["plate_appearances"]
        else:
            raise ValueError("No plate_appearances or batters_faced column found.")

    # ── Derive outs_recorded if not present ──────────────────────────────
    if "outs_recorded" not in df.columns:
        hbp = df["hbp"] if "hbp" in df.columns else 0
        df["outs_recorded"] = (
            df["plate_appearances"] - df["hits_allowed"] - df["walks"] - hbp
        ).clip(lower=0).astype(int)
        print("  ⚠ Derived outs_recorded from PA - H - BB - HBP (re-run 02 for native column)")

    # ── Create rate target ───────────────────────────────────────────────
    bf = df["batters_faced"].replace(0, np.nan)
    df["o_per_pa"] = (df["outs_recorded"] / bf).clip(lower=0, upper=1)

    # ── Filter ───────────────────────────────────────────────────────────
    initial = len(df)
    df = df[df["batters_faced"] >= MIN_PA_GAME].copy()
    df = df.dropna(subset=["o_per_pa", "outs_recorded"])
    df = df[df["o_per_pa"].between(0, 1)]
    print(f"  Loaded {initial:,} rows → {len(df):,} after filtering")

    # Store actual count for evaluation
    df["actual_O"] = df["outs_recorded"].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df, target):
    """Select features avoiding leakage, using substring matching."""
    candidates = []
    skip_cols = {"game_date", "game_pk", "pitcher", "pitcher_name",
                 "team", "opponent", "venue", "umpire", "actual_K",
                 "actual_H", "actual_BB", "actual_O", "year", "player_name",
                 "start_number_in_season", "prior_year",
                 "h_per_pa", "bb_per_pa", "o_per_pa", "batters_faced",
                 "pitcher_team", "opp_team", "venue_name", "venue_id",
                 "hp_umpire_name", "hp_umpire_id",
                 "home_team", "away_team", "home_team_name", "away_team_name",
                 "home_team_id", "away_team_id", "home_starter_id", "away_starter_id",
                 "season", "p_throws", "day_night",
                 "latitude", "longitude",
                 "catcher_id", "catcher_name", "opp_starter_id"}

    for col in df.columns:
        if col in RAW_STAT_EXCLUSIONS or col in skip_cols:
            continue
        if col == target:
            continue
        col_lower = col.lower()
        is_safe = any(pattern.lower() in col_lower for pattern in FEATURE_PATTERNS)
        if is_safe and pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            candidates.append(col)

    print(f"  Selected {len(candidates)} features for target '{target}'")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def time_split(df, feature_cols, target_col):
    """Split by year: everything before TEST_YEAR = train, TEST_YEAR+ = test."""
    df = df.sort_values("game_date").reset_index(drop=True)
    df["year"] = df["game_date"].dt.year

    train = df[df["year"] < TEST_YEAR].copy()
    test = df[df["year"] >= TEST_YEAR].copy()

    train = train.dropna(subset=[target_col])
    test = test.dropna(subset=[target_col])

    X_train = train[feature_cols].fillna(0).values
    y_train = train[target_col].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test[target_col].values

    print(f"  Train: {len(train):,} rows ({train['year'].min()}-{train['year'].max()})")
    print(f"  Test:  {len(test):,} rows ({test['year'].min()}-{test['year'].max()})")

    return X_train, X_test, y_train, y_test, train, test


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_models(X_train, y_train, X_test, y_test, feature_names, target_name,
                 n_iter=50, cv_folds=4):
    """Train XGBoost + LightGBM + Ridge, pick the best."""
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    cv_folds = min(cv_folds, max(2, len(X_train) // 100))
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    results = {}

    # ── XGBoost ──────────────────────────────────────────────────────────
    try:
        import xgboost as xgb
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
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        r2 = 1 - np.sum((y_test - preds) ** 2) / ss_tot if ss_tot > 0 else 0
        cv_mae = -search.best_score_
        print(f"    XGBoost — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, CV MAE: {cv_mae:.4f}")
        results["XGBoost"] = {
            "model": best_xgb, "preds": preds,
            "MAE": mae, "RMSE": rmse, "R²": r2, "CV_MAE": cv_mae,
        }
    except ImportError:
        print("  ⚠ xgboost not installed, skipping")

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
        mae_lgb = np.mean(np.abs(y_test - preds_lgb))
        rmse_lgb = np.sqrt(np.mean((y_test - preds_lgb) ** 2))
        r2_lgb = 1 - np.sum((y_test - preds_lgb) ** 2) / ss_tot if ss_tot > 0 else 0
        cv_mae_lgb = -search_lgb.best_score_
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
    mae_r = np.mean(np.abs(y_test - preds_r))
    rmse_r = np.sqrt(np.mean((y_test - preds_r) ** 2))
    r2_r = 1 - np.sum((y_test - preds_r) ** 2) / ss_tot if ss_tot > 0 else 0
    print(f"    Ridge — MAE: {mae_r:.4f}, RMSE: {rmse_r:.4f}, R²: {r2_r:.4f}")
    results["Ridge"] = {"model": ridge, "preds": preds_r,
                        "MAE": mae_r, "RMSE": rmse_r, "R²": r2_r}

    best_name = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n  ✓ Best for '{target_name}': {best_name} (MAE={results[best_name]['MAE']:.4f})")
    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# BETA-BINOMIAL MATH (reused from 06/10)
# ══════════════════════════════════════════════════════════════════════════════

def beta_binom_pmf(k, n, alpha, beta_param):
    if n < 0 or k < 0 or k > n:
        return 0.0
    log_pmf = (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta_param)
        - betaln(alpha, beta_param)
    )
    return np.exp(log_pmf)


def beta_binom_pmf_array(n, alpha, beta_param):
    ks = np.arange(n + 1)
    log_comb = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    log_beta_num = betaln(ks + alpha, n - ks + beta_param)
    log_beta_den = betaln(alpha, beta_param)
    log_pmf = log_comb + log_beta_num - log_beta_den
    pmf = np.exp(log_pmf)
    pmf = pmf / pmf.sum()
    return pmf


def expected_pmf_over_N(pred_p, pred_N, kappa, sigma_N, max_k=30):
    """PMF marginalized over BF uncertainty using ±3σ range.

    max_k=30 because outs can go up to ~27 (9 full innings).
    """
    pred_p = np.clip(pred_p, 0.01, 0.99)
    alpha = max(pred_p * kappa, 0.01)
    beta_param = max((1 - pred_p) * kappa, 0.01)
    min_N = max(1, int(pred_N - 3 * sigma_N))
    max_N = int(pred_N + 3 * sigma_N) + 1
    N_values = np.arange(min_N, max_N + 1)
    N_weights = sp_stats.norm.pdf(N_values, loc=pred_N, scale=max(sigma_N, 0.5))
    N_weights = N_weights / N_weights.sum()
    combined = np.zeros(max_k + 1)
    for n_val, w in zip(N_values, N_weights):
        pmf = beta_binom_pmf_array(n_val, alpha, beta_param)
        if len(pmf) <= max_k + 1:
            combined[:len(pmf)] += w * pmf
        else:
            combined += w * pmf[:max_k + 1]
    total = combined.sum()
    if total > 0:
        combined /= total
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# KAPPA CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_kappa(pred_p, actual_N, actual_count, label=""):
    """Find κ that maximizes log-likelihood using actual N."""
    kappa_grid = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300]

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
    low = max(2, best_grid * 0.4)
    high = min(200.0, best_grid * 2.5)  # hard cap: κ>200 produces unrealistically tight distributions

    result = minimize_scalar(
        lambda lk: -compute_ll(np.exp(lk)),
        bounds=(np.log(low), np.log(high)), method="bounded",
    )
    kappa_opt = np.exp(result.x)
    final_ll = -result.fun

    # Hard cap for safety — high κ collapses BB to binomial and creates huge spurious edges
    KAPPA_MAX = 200.0
    if kappa_opt > KAPPA_MAX:
        print(f"    ⚠ κ={kappa_opt:.0f} exceeded cap; clamping to {KAPPA_MAX}")
        kappa_opt = KAPPA_MAX
    if kappa_opt > 150:
        print(f"    ⚠ WARNING: κ={kappa_opt:.0f} is high — distribution may be too tight.")

    print(f"    ✓ Calibrated κ = {kappa_opt:.2f} (LL = {final_ll:.2f})")
    return kappa_opt


def calibrate_kappa_cv(X_train, y_train_rate, y_train_bf, y_train_count,
                       feature_cols, rate_model, label="", n_folds=5):
    """Cross-validated κ calibration."""
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
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(pred_p, pred_N, actual_count, actual_N, kappa, sigma_N,
                   label, lines=None):
    """Evaluate Beta-Binomial vs Normal for outs."""
    if lines is None:
        lines = [12, 14, 15, 16, 17, 18, 19, 20, 21]

    n_eval = len(actual_count)
    actual = actual_count

    bb_arr = np.zeros(n_eval)
    norm_arr = np.zeros(n_eval)
    bb_ll = 0.0
    norm_ll = 0.0

    all_pred_N = pred_N if isinstance(pred_N, np.ndarray) else np.full(n_eval, pred_N)
    NORMAL_STD = np.std(actual_count - pred_p * all_pred_N)

    for i in range(n_eval):
        p_i = np.clip(pred_p[i], 0.01, 0.99)
        n_i = all_pred_N[i]
        c_i = int(actual[i])
        n_actual = int(actual_N[i])

        # BB expected value
        pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=30)
        bb_arr[i] = sum(k * pmf[k] for k in range(len(pmf)))

        # BB log-likelihood (using actual N, not predicted N)
        alpha = max(p_i * kappa, 0.01)
        beta_p = max((1 - p_i) * kappa, 0.01)
        if n_actual >= c_i and n_actual > 0:
            prob = beta_binom_pmf(c_i, n_actual, alpha, beta_p)
            bb_ll += np.log(max(prob, 1e-15))

        # Normal comparison
        mu = p_i * n_i
        norm_arr[i] = mu
        norm_ll += sp_stats.norm.logpdf(c_i, loc=mu, scale=max(NORMAL_STD, 0.1))

    bb_mae = np.mean(np.abs(actual - bb_arr))
    norm_mae = np.mean(np.abs(actual - norm_arr))
    print(f"\n  {label} Evaluation ({n_eval} games):")
    print(f"    BB  MAE: {bb_mae:.3f}  |  LL: {bb_ll:.1f}")
    print(f"    Norm MAE: {norm_mae:.3f}  |  LL: {norm_ll:.1f}")
    if bb_ll > norm_ll:
        print(f"    ✓ Beta-Binomial wins on log-likelihood (+{bb_ll - norm_ll:.1f})")
    else:
        print(f"    ⚠ Normal wins on log-likelihood (+{norm_ll - bb_ll:.1f})")

    # Brier scores at each line
    brier_results = {}
    for line in lines:
        actual_over = (actual >= line).astype(float)
        bb_probs = []
        for i in range(n_eval):
            p_i = np.clip(pred_p[i], 0.01, 0.99)
            n_i = all_pred_N[i]
            pmf = expected_pmf_over_N(p_i, n_i, kappa, sigma_N, max_k=30)
            p_over = pmf[line:].sum() if line < len(pmf) else 0.0
            bb_probs.append(p_over)
        bb_probs = np.array(bb_probs)
        brier = np.mean((bb_probs - actual_over) ** 2)
        base_rate = actual_over.mean()
        brier_results[line] = {"brier": float(brier), "base_rate": float(base_rate)}
        print(f"    O≥{line:2d}: Brier={brier:.4f}  base_rate={base_rate:.1%}  avg_pred={bb_probs.mean():.1%}")

    return {
        "bb_mae": float(np.mean(np.abs(actual - bb_arr))),
        "norm_mae": float(np.mean(np.abs(actual - norm_arr))),
        "bb_ll": float(bb_ll),
        "norm_ll": float(norm_ll),
        "bb_mean_pred": float(bb_arr.mean()),
        "actual_mean": float(actual.mean()),
        "brier": brier_results,
        "normal_std": float(NORMAL_STD),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, features, kappa, sigma_N, eval_results, model_dir):
    """Save the outs rate model and its config."""
    model_path = model_dir / "outs_rate_model.joblib"
    config_path = model_dir / "outs_config.json"

    joblib.dump(model, model_path)

    config = {
        "stat": "outs",
        "rate_features": features,
        "kappa": float(kappa),
        "sigma_n": float(sigma_N),
        "sigma_N": float(sigma_N),
        "sigma_N_global": float(sigma_N),
        "bb_mae": eval_results["bb_mae"],
        "normal_std": eval_results["normal_std"],
        "actual_mean": eval_results["actual_mean"],
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✓ Saved {model_path}")
    print(f"  ✓ Saved {config_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PITCHER OUTS RECORDED — BETA-BINOMIAL MODEL")
    print("=" * 70)

    # ── Load & prepare ───────────────────────────────────────────────────
    print("\n── Step 1: Load & Prepare Data ──")
    df = load_and_prepare()

    # ── Load existing BF config for sigma_N ──────────────────────────────
    bb_config_path = MODEL_DIR / "beta_binom_config.json"
    if bb_config_path.exists():
        with open(bb_config_path) as f:
            bb_config = json.load(f)
        sigma_N = bb_config.get("sigma_n", bb_config.get("sigma_n_global", 2.5))
        print(f"  Using σ_N = {sigma_N:.2f} from existing BF config")
    else:
        sigma_N = 2.5
        print(f"  No BF config found, using default σ_N = {sigma_N}")

    # ── Load existing BF model for N predictions ─────────────────────────
    bf_model_path = MODEL_DIR / "bf_model.joblib"
    if bf_model_path.exists():
        bf_model = joblib.load(bf_model_path)
        bf_features = bb_config.get("bf_features", None) if bb_config_path.exists() else None
        bf_is_log = bool(bb_config.get("bf_is_log", False)) if bb_config_path.exists() else False
        bf_log_sigma2 = float(bb_config.get("bf_log_sigma2", 0.0)) if bb_config_path.exists() else 0.0
        print(f"  Loaded BF model for N predictions{' (log-scale)' if bf_is_log else ''}")
    else:
        bf_model = None
        bf_features = None
        bf_is_log = False
        bf_log_sigma2 = 0.0
        print(f"  ⚠ No BF model found — will use actual N for calibration only")

    def _bf_predict(X):
        """Wrapper that applies log-inverse if 06 trained on log(BF)."""
        p = bf_model.predict(X)
        if bf_is_log:
            return np.exp(p) * np.exp(bf_log_sigma2 / 2)
        return p

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN OUTS MODEL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  OUTS RECORDED MODEL (O/PA rate)")
    print(f"{'=' * 70}")

    print("\n── Step 2: Select Features ──")
    outs_features = select_features(df, "o_per_pa")
    if prune_collinear_features is not None:
        df_for_prune = df.copy()
        if "year" not in df_for_prune.columns and "game_date" in df_for_prune.columns:
            df_for_prune["year"] = pd.to_datetime(df_for_prune["game_date"], errors="coerce").dt.year
        outs_features, _ = prune_collinear_features(
            df_for_prune, outs_features, "o_per_pa",
            corr_threshold=0.99, target_year_cutoff=TEST_YEAR,
        )

    print("\n── Step 3: Train/Test Split ──")
    X_tr, X_te, y_tr, y_te, train_df, test_df = time_split(
        df, outs_features, "o_per_pa"
    )

    print("\n── Step 4: Train Models ──")
    results, best_name = train_models(
        X_tr, y_tr, X_te, y_te, outs_features, "o_per_pa"
    )

    # Get aligned BF and actual counts for test set
    y_test_bf = test_df["batters_faced"].values
    actual_O = test_df["actual_O"].values

    # Rate predictions
    rate_preds = results[best_name]["preds"]

    # BF predictions (from BF model or use actual)
    if bf_model and bf_features:
        bf_available = [f for f in bf_features if f in test_df.columns]
        if bf_available and len(bf_available) == len(bf_features):
            bf_preds = _bf_predict(test_df[bf_features].fillna(0).values)
        else:
            bf_preds = y_test_bf.copy()
    else:
        bf_preds = y_test_bf.copy()

    # Direct prediction
    direct = rate_preds * bf_preds
    direct_mae = np.mean(np.abs(actual_O - direct))
    print(f"\n  Direct point prediction (ô × N̂): MAE = {direct_mae:.3f}")
    print(f"  Mean actual outs: {actual_O.mean():.2f}")

    # ── Calibrate κ ──────────────────────────────────────────────────────
    print("\n── Step 5: Calibrate κ (Outs) ──")
    print("\n  Method A: κ from test-set predictions + actual N")
    kappa_test = calibrate_kappa(rate_preds, y_test_bf, actual_O, label="Outs")

    print("\n  Method B: κ from cross-validated training predictions")
    train_actual_O = train_df["actual_O"].values
    train_bf = train_df["batters_faced"].values
    kappa_cv = calibrate_kappa_cv(
        X_tr, y_tr, train_bf, train_actual_O,
        outs_features, results[best_name]["model"], label="Outs",
    )
    kappa = kappa_cv
    print(f"\n  → Using CV κ = {kappa:.2f} (test κ was {kappa_test:.2f})")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n── Step 6: Evaluate Outs Model ──")
    eval_results = evaluate_model(
        rate_preds, bf_preds, actual_O, y_test_bf,
        kappa, sigma_N, label="Outs",
        lines=[12, 14, 15, 16, 17, 18, 19, 20, 21],
    )

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SAVING MODEL")
    print(f"{'=' * 70}")
    save_model(results[best_name]["model"], outs_features, kappa, sigma_N,
               eval_results, MODEL_DIR)

    # ── Save comparison CSV ──────────────────────────────────────────────
    comparison = pd.DataFrame([{
        "stat": "outs",
        "model": best_name,
        "rate_MAE": results[best_name]["MAE"],
        "point_MAE": eval_results["bb_mae"],
        "kappa": kappa,
        "mean_actual": eval_results["actual_mean"],
        "log_lik_bb": eval_results["bb_ll"],
        "log_lik_norm": eval_results["norm_ll"],
    }])
    comparison.to_csv(OUTPUT_DIR / "outs_comparison.csv", index=False)
    print(f"  ✓ Saved outs_comparison.csv")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Outs Recorded:")
    print(f"    Rate Model:  {best_name} (O/PA MAE = {results[best_name]['MAE']:.4f})")
    print(f"    κ:           {kappa:.2f}")
    print(f"    Point MAE:   {eval_results['bb_mae']:.3f} (mean actual: {eval_results['actual_mean']:.2f})")
    print(f"\n  Next: run predict/outs.py to generate daily predictions")


if __name__ == "__main__":
    main()
