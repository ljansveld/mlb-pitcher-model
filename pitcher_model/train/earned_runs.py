"""
train/earned_runs.py
==============
Earned runs allowed model. Same Beta-Binomial architecture as the K,
hits, walks, and outs models:
  - Rate target: er_per_pa = earned_runs / batters_faced
  - Trained via XGB with time-split CV
  - Uses the saved BF model from train/strikeouts.py for N predictions
    (inverse-log if needed)
  - Calibrates κ for ER-specific BB dispersion
  - Saves er_rate_model.joblib + er_config.json

PREREQUISITES:
  1. collect/statcast.py has been run (produces data/earned_runs_by_pitcher.csv)
  2. train/strikeouts.py has been run (produces bf_model.joblib + config)

Reuses helper functions from train/hits_walks.py rather than duplicating the
Beta-Binomial fitting path: select_features, time_split, train_xgb,
calibrate_kappa_cv, expected_pmf_over_N, evaluate_distributions, save_model.
The last two are aliased locally to train_models / evaluate_model.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

from pitcher_model.paths import DATA_DIR, MODEL_DIR, ensure_dirs

ensure_dirs(MODEL_DIR)

TEST_YEAR = 2025
MIN_PA_GAME = 6
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# Shared machinery
# ──────────────────────────────────────────────────────────────────────────────
# Earned runs use the same Beta-Binomial fitting path as hits and walks — the
# only differences are the target column, kappa, and the ER-specific feature
# exclusions below. So the training/evaluation helpers are imported from
# hits_walks rather than duplicated. Both modules gate their training behind
# __main__, so importing runs no work.
from pitcher_model.train.hits_walks import (
    select_features,
    time_split,
    train_xgb as train_models,
    calibrate_kappa_cv,
    expected_pmf_over_N,
    evaluate_distributions as evaluate_model,
    save_model,
)
from pitcher_model.train.strikeouts import prune_collinear_features

# Thresholds the ER distribution is scored at. Earned runs are low-count
# (test mean ~2.4), so the ladder sits lower than the hits/walks one.
ER_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]


# ──────────────────────────────────────────────────────────────────────────────
# LOAD WITH ER TARGET
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prepare_er():
    feat_path = DATA_DIR / "pitcher_model_features.csv"
    er_path = DATA_DIR / "earned_runs_by_pitcher.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run 02 first.")
    if not er_path.exists():
        raise FileNotFoundError(
            f"{er_path} not found. Run collect/statcast.py first "
            f"(backfill) and/or refresh.py (daily)."
        )

    df = pd.read_csv(feat_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    er = pd.read_csv(er_path)
    er["game_pk"] = er["game_pk"].astype("int64")
    er["pitcher"] = er["pitcher"].astype("int64")

    # Join ER onto the feature frame
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    before = len(df)
    df = df.merge(er, on=["game_pk", "pitcher"], how="left")
    matched = df["earned_runs"].notna().sum()
    print(f"  Joined ER: {matched:,}/{before:,} rows matched "
          f"({100*matched/before:.1f}%)")

    # Drop rows with no ER info
    df = df.dropna(subset=["earned_runs"]).copy()
    df["earned_runs"] = df["earned_runs"].astype(int)

    # BF column
    if "batters_faced" not in df.columns:
        if "plate_appearances" in df.columns:
            df["batters_faced"] = df["plate_appearances"]
        else:
            raise ValueError("No plate_appearances column.")

    # Rate target — ER per batter faced. A bit unusual (ER is a rare event
    # so the rate is low ~0.03–0.08, Beta-Binomial still handles it fine)
    bf = df["batters_faced"].replace(0, np.nan)
    df["er_per_pa"] = df["earned_runs"] / bf

    # Filter
    df = df[df["batters_faced"] >= MIN_PA_GAME].copy()
    df = df.dropna(subset=["er_per_pa"]).copy()
    df["year"] = df["game_date"].dt.year

    print(f"  After filter: {len(df):,} rows. "
          f"ER per start: mean={df['earned_runs'].mean():.2f}, "
          f"ER/PA: mean={df['er_per_pa'].mean():.4f}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  EARNED RUNS ALLOWED MODEL (Beta-Binomial)")
    print("=" * 70)

    print("\n── Step 1: Load & Prepare Data ──")
    df = load_and_prepare_er()

    # Load BF config + model
    bb_config_path = MODEL_DIR / "beta_binom_config.json"
    if not bb_config_path.exists():
        raise FileNotFoundError("beta_binom_config.json not found. Run 06 first.")
    with open(bb_config_path) as f:
        bb_config = json.load(f)
    sigma_N = float(bb_config.get("sigma_n", bb_config.get("sigma_n_global", 2.5)))
    bf_model = joblib.load(MODEL_DIR / "bf_model.joblib")
    bf_features = bb_config.get("bf_features", None)
    bf_is_log = bool(bb_config.get("bf_is_log", False))
    bf_log_sigma2 = float(bb_config.get("bf_log_sigma2", 0.0))
    print(f"  Using σ_N = {sigma_N:.2f}, "
          f"BF model{' (log-scale)' if bf_is_log else ''}")

    def _bf_predict(X):
        p = bf_model.predict(X)
        if bf_is_log:
            return np.exp(p) * np.exp(bf_log_sigma2 / 2)
        return p

    # Feature selection + pruning
    print("\n── Step 2: Select Features ──")
    er_features = select_features(df, "er_per_pa")
    if prune_collinear_features is not None:
        df_for_prune = df.copy()
        if "year" not in df_for_prune.columns:
            df_for_prune["year"] = df_for_prune["game_date"].dt.year
        er_features, _ = prune_collinear_features(
            df_for_prune, er_features, "er_per_pa",
            corr_threshold=0.92, target_year_cutoff=TEST_YEAR,
        )

    # Time split — note: 10's time_split returns (X_tr, X_te, y_tr, y_te, train_df, test_df)
    print("\n── Step 3: Train/Test Split ──")
    X_tr, X_te, y_tr, y_te, train_df, test_df = time_split(
        df, er_features, "er_per_pa"
    )
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # Train
    print("\n── Step 4: Train ER Rate Model ──")
    results, best = train_models(
        X_tr, y_tr, X_te, y_te, er_features, "er_per_pa",
    )
    er_rate_preds = results[best]["preds"]

    # BF predictions on test
    bf_avail = [f for f in bf_features if f in test_df.columns]
    if bf_avail and len(bf_avail) == len(bf_features):
        bf_preds = _bf_predict(test_df[bf_features].fillna(0).values)
    else:
        # Fallback: use actual BF if BF-feature set doesn't match
        print(f"  ⚠ BF feature mismatch ({len(bf_avail)}/{len(bf_features)}) — "
              f"using actual BF on test for calibration only")
        bf_preds = test_df["batters_faced"].values.astype(float)

    # Direct point-prediction diagnostic
    actual_ER = test_df["earned_runs"].values.astype(int)
    direct = er_rate_preds * bf_preds
    print(f"\n  Direct point prediction (ê × N̂): "
          f"MAE = {np.mean(np.abs(actual_ER - direct)):.3f}")
    print(f"  Mean actual ER: {actual_ER.mean():.2f}  "
          f"Mean predicted: {direct.mean():.2f}")

    # Calibrate κ via CV on training set (uses the same CV helper as 10)
    print("\n── Step 5: Calibrate κ ──")
    y_train_bf = train_df["batters_faced"].values.astype(float)
    y_train_er = train_df["earned_runs"].values.astype(int)
    if bf_avail and len(bf_avail) == len(bf_features):
        # Build OOF BF predictions to feed calibrator correctly (same as 10 does).
        # Simplest: use actual BF on train, which is what 10 falls back to too.
        pass
    kappa = calibrate_kappa_cv(
        X_tr, y_tr, y_train_bf, y_train_er,
        er_features, results[best]["model"], n_folds=5,
    )
    print(f"  Calibrated κ = {kappa:.2f}")

    # Evaluate
    print("\n── Step 6: Evaluate on Test ──")
    eval_results = evaluate_model(
        er_rate_preds, bf_preds, actual_ER,
        test_df["batters_faced"].values, kappa, sigma_N,
        lines=ER_LINES, label="Earned Runs",
    )

    # Save
    print("\n── Step 7: Save ──")
    save_model(
        results[best]["model"], er_features, kappa, sigma_N,
        eval_results, "er", MODEL_DIR,
    )
    print(f"\n  ✓ Saved models/er_rate_model.joblib + er_config.json")
    print("\n" + "=" * 70)
    print("  DONE. Run predict/earned_runs.py for daily predictions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
