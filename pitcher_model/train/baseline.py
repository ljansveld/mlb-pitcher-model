"""
MLB Pitcher Model Training & Evaluation (Enhanced)
====================================================
Trains and evaluates models with comprehensive diagnostics:
  - Multiple model types (linear, tree-based, boosting)
  - Proper time-based train/test splitting
  - Naive baselines for benchmarking
  - Feature importance (built-in + SHAP)
  - Error analysis by pitcher tier, opposing team quality, etc.
  - Threshold classification (does the model separate over/under?)

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
    pip install shap  (optional, for SHAP analysis)

Usage:
    python run.py train baseline
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pitcher_model.paths import DATA_DIR, OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")


# ── Configuration ────────────────────────────────────────────────────────────
TARGET = "target_strikeouts"

FEATURE_PATTERNS = [
    # Rolling pitcher stats
    "_L3", "_L5", "_L10",
    # Season cumulative
    "_szn",
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
    "pf_",
    # Weather
    "wx_",
    # Schedule/context
    "rest_days", "short_rest", "extra_rest",
    "is_home", "is_night", "is_weekend", "day_of_week", "month",
    "start_num", "is_dome",
    # Normalized K metrics
    "k_per_100", "k_per_9", "k_per_pa", "pitches_per_k",
    "est_innings", "is_short_outing",
    # Early-season context
    "days_into", "season_phase", "is_first_month",
    "prior_starts", "_prev5", "_prev10", "pvt_",
    "_x_reliability", "early_x_",
    # Platoon
    "platoon", "vs_left", "vs_right",
    "pitcher_throws_R",
    # Stuff volatility
    "_std_",
]

# Train on 2021-2024, test on 2025
TEST_SEASONS = [2025]

# For within-season split (alternative)
WITHIN_SEASON_TRAIN_PCT = 0.70


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_features():
    path = DATA_DIR / "pitcher_model_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run features.py first.")
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df


def select_features(df):
    """Select feature columns, excluding targets and identifiers."""
    exclude_prefixes = ("target_", "game_pk", "game_date", "pitcher", "season",
                        "home_team", "away_team", "opp_team", "pitcher_team",
                        "venue_name", "hp_umpire", "day_night", "p_throws",
                        "home_team_name", "away_team_name",
                        "home_starter", "away_starter", "home_team_id", "away_team_id",
                        "venue_id", "latitude", "longitude")

    # Also exclude raw game-level stats (these are what we're predicting)
    raw_stats = {
        "strikeouts", "walks", "hits_allowed", "home_runs_allowed",
        "plate_appearances", "total_pitches", "batted_balls",
        "strikes", "balls", "whiffs", "called_strikes",
        "in_zone_pitches", "out_of_zone_pitches", "chases",
        "barrels", "hard_hits", "soft_hits", "hbp",
        "singles", "doubles", "triples",
        "fastball_count", "breaking_count", "offspeed_count",
        "ff_count", "si_count", "fc_count", "sl_count", "cu_count",
        "ch_count", "fs_count", "sv_count", "kc_count",
        "pa_vs_left", "pa_vs_right", "whiffs_vs_left", "whiffs_vs_right",
        "avg_velocity", "max_velocity", "avg_spin_rate",
        "avg_extension", "avg_induced_vert_break", "avg_horiz_break",
        "k_pct", "bb_pct", "k_bb_pct", "whiff_pct", "csw_pct",
        "zone_pct", "chase_rate", "strike_pct",
        "barrel_pct", "hard_hit_pct", "soft_hit_pct",
        "fastball_pct", "breaking_pct", "offspeed_pct",
        "ff_pct", "si_pct", "fc_pct", "sl_pct", "cu_pct",
        "ch_pct", "fs_pct", "sv_pct", "kc_pct",
        "whiff_pct_vs_left", "whiff_pct_vs_right",
        "k_per_100_pitches", "k_per_9", "k_per_pa", "pitches_per_k",
        "est_innings", "is_short_outing",
    }

    feature_cols = []
    for col in df.columns:
        # Skip identifiers and targets
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        # Skip raw current-game stats
        if col in raw_stats:
            continue
        # Must match at least one feature pattern
        if any(pattern in col for pattern in FEATURE_PATTERNS):
            feature_cols.append(col)

    # Remove columns that are all NaN or constant
    valid_cols = []
    for c in feature_cols:
        if df[c].notna().sum() > 100 and df[c].nunique() > 1:
            valid_cols.append(c)

    print(f"  Selected {len(valid_cols)} features (from {len(df.columns)} total columns)")
    return valid_cols


def time_split(df, feature_cols):
    """Chronological train/test split."""
    train = df[~df["season"].isin(TEST_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()

    print(f"  Train: {sorted(train['season'].unique())} ({len(train):,} rows)")
    print(f"  Test:  {sorted(test['season'].unique())} ({len(test):,} rows)")

    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    y_train = train[TARGET]
    y_test = test[TARGET]

    return X_train, X_test, y_train, y_test, train, test


# ── Baselines ────────────────────────────────────────────────────────────────
def compute_baselines(X_test, y_test, feature_cols):
    """Naive baselines to beat."""
    baselines = {}

    # Always predict mean
    mean_val = y_test.mean()
    baselines["Always Mean"] = {
        "preds": np.full(len(y_test), mean_val),
        "MAE": mean_absolute_error(y_test, np.full(len(y_test), mean_val)),
        "RMSE": np.sqrt(mean_squared_error(y_test, np.full(len(y_test), mean_val))),
        "R²": 0.0,
    }

    # Last 3 rolling average
    for suffix, label in [("strikeouts_L3", "Last 3 Avg"), ("strikeouts_L5", "Last 5 Avg"),
                          ("strikeouts_L10", "Last 10 Avg")]:
        if suffix in feature_cols:
            preds = X_test[suffix].values
            baselines[label] = {
                "preds": preds,
                "MAE": mean_absolute_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R²": r2_score(y_test, preds),
            }

    print("\n── Baselines ──")
    for name, m in baselines.items():
        print(f"  {name:20s}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R²']:.3f}")

    return baselines


# ── Model Training ───────────────────────────────────────────────────────────
def train_default_models(X_train, y_train, X_test, y_test):
    """Train models with sensible defaults (fast, used as comparison baseline)."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Ridge": (Ridge(alpha=1.0), True),
        "Lasso": (Lasso(alpha=0.05), True),
        "ElasticNet": (ElasticNet(alpha=0.05, l1_ratio=0.5), True),
        "Random Forest": (RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            random_state=42, n_jobs=-1,
        ), False),
        "Gradient Boosting": (GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            min_samples_leaf=15, subsample=0.8, random_state=42,
        ), False),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        ), False)
    except ImportError:
        print("  ℹ XGBoost not installed")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = (LGBMRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        ), False)
    except ImportError:
        print("  ℹ LightGBM not installed")

    results = {}
    for name, (model, use_scaled) in models.items():
        print(f"  Training {name} (defaults)...")
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc if use_scaled else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        results[name] = {
            "model": model, "predictions": preds, "use_scaled": use_scaled,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R²": r2_score(y_test, preds),
        }
        print(f"    MAE={results[name]['MAE']:.3f}  R²={results[name]['R²']:.3f}")

    return results, scaler


# ── Hyperparameter Tuning ────────────────────────────────────────────────────
def tune_hyperparameters(X_train, y_train, X_test, y_test, train_df):
    """
    Bayesian-style hyperparameter tuning using RandomizedSearchCV with
    TimeSeriesSplit cross-validation.

    TimeSeriesSplit ensures we never train on future data during CV:
      Fold 1: train [0..T1], validate [T1..T2]
      Fold 2: train [0..T2], validate [T2..T3]
      ...

    We tune the top 3 model families: Gradient Boosting, XGBoost, LightGBM.
    Linear models get a lighter Ridge/ElasticNet alpha sweep.
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

    print("\n── Hyperparameter Tuning ──")
    print("  Using TimeSeriesSplit (4 folds) + RandomizedSearchCV")

    # TimeSeriesSplit for proper temporal CV
    tscv = TimeSeriesSplit(n_splits=4)

    tuned_results = {}

    # ── Ridge/ElasticNet (quick alpha sweep) ──
    print("\n  Tuning Ridge...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    ridge_params = {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
    ridge_search = RandomizedSearchCV(
        Ridge(), ridge_params, n_iter=8, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    ridge_search.fit(X_train_sc, y_train)
    ridge_preds = ridge_search.predict(X_test_sc)
    tuned_results["Ridge (tuned)"] = {
        "model": ridge_search.best_estimator_,
        "predictions": ridge_preds,
        "use_scaled": True,
        "MAE": mean_absolute_error(y_test, ridge_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "R²": r2_score(y_test, ridge_preds),
        "best_params": ridge_search.best_params_,
        "cv_score": -ridge_search.best_score_,
    }
    print(f"    Best: alpha={ridge_search.best_params_['alpha']}")
    print(f"    CV MAE={-ridge_search.best_score_:.3f}  Test MAE={tuned_results['Ridge (tuned)']['MAE']:.3f}")

    print("\n  Tuning ElasticNet...")
    enet_params = {
        "alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    enet_search = RandomizedSearchCV(
        ElasticNet(max_iter=5000), enet_params, n_iter=20, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    enet_search.fit(X_train_sc, y_train)
    enet_preds = enet_search.predict(X_test_sc)
    tuned_results["ElasticNet (tuned)"] = {
        "model": enet_search.best_estimator_,
        "predictions": enet_preds,
        "use_scaled": True,
        "MAE": mean_absolute_error(y_test, enet_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, enet_preds)),
        "R²": r2_score(y_test, enet_preds),
        "best_params": enet_search.best_params_,
        "cv_score": -enet_search.best_score_,
    }
    print(f"    Best: {enet_search.best_params_}")
    print(f"    CV MAE={-enet_search.best_score_:.3f}  Test MAE={tuned_results['ElasticNet (tuned)']['MAE']:.3f}")

    # ── Random Forest ──
    print("\n  Tuning Random Forest (30 random combos)...")
    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 6, 8, 10, 12, None],
        "min_samples_leaf": [5, 10, 15, 20, 30],
        "min_samples_split": [5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, n_iter=30, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf_preds = rf_search.predict(X_test)
    tuned_results["Random Forest (tuned)"] = {
        "model": rf_search.best_estimator_,
        "predictions": rf_preds,
        "use_scaled": False,
        "MAE": mean_absolute_error(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "R²": r2_score(y_test, rf_preds),
        "best_params": rf_search.best_params_,
        "cv_score": -rf_search.best_score_,
    }
    print(f"    Best: {rf_search.best_params_}")
    print(f"    CV MAE={-rf_search.best_score_:.3f}  Test MAE={tuned_results['Random Forest (tuned)']['MAE']:.3f}")

    # ── Gradient Boosting ──
    print("\n  Tuning Gradient Boosting (40 random combos)...")
    gb_params = {
        "n_estimators": [200, 300, 400, 500, 700],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "min_samples_leaf": [5, 10, 15, 20, 30],
        "subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_params, n_iter=40, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)
    gb_preds = gb_search.predict(X_test)
    tuned_results["Gradient Boosting (tuned)"] = {
        "model": gb_search.best_estimator_,
        "predictions": gb_preds,
        "use_scaled": False,
        "MAE": mean_absolute_error(y_test, gb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, gb_preds)),
        "R²": r2_score(y_test, gb_preds),
        "best_params": gb_search.best_params_,
        "cv_score": -gb_search.best_score_,
    }
    print(f"    Best: {gb_search.best_params_}")
    print(f"    CV MAE={-gb_search.best_score_:.3f}  Test MAE={tuned_results['Gradient Boosting (tuned)']['MAE']:.3f}")

    # ── XGBoost (if available) ──
    try:
        from xgboost import XGBRegressor
        print("\n  Tuning XGBoost (50 random combos)...")
        xgb_params = {
            "n_estimators": [200, 300, 400, 500, 700, 1000],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
            "min_child_weight": [5, 10, 15, 20, 30],
            "subsample": [0.7, 0.8, 0.85, 0.9],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "gamma": [0, 0.1, 0.5, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
            xgb_params, n_iter=50, cv=tscv,
            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
        )
        xgb_search.fit(X_train, y_train)
        xgb_preds = xgb_search.predict(X_test)
        tuned_results["XGBoost (tuned)"] = {
            "model": xgb_search.best_estimator_,
            "predictions": xgb_preds,
            "use_scaled": False,
            "MAE": mean_absolute_error(y_test, xgb_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
            "R²": r2_score(y_test, xgb_preds),
            "best_params": xgb_search.best_params_,
            "cv_score": -xgb_search.best_score_,
        }
        print(f"    Best: {xgb_search.best_params_}")
        print(f"    CV MAE={-xgb_search.best_score_:.3f}  Test MAE={tuned_results['XGBoost (tuned)']['MAE']:.3f}")
    except ImportError:
        print("  ℹ XGBoost not installed — skipping")

    # ── LightGBM (if available) ──
    try:
        from lightgbm import LGBMRegressor
        print("\n  Tuning LightGBM (50 random combos)...")
        lgbm_params = {
            "n_estimators": [200, 300, 400, 500, 700, 1000],
            "max_depth": [3, 4, 5, 6, 7, 8, -1],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
            "num_leaves": [15, 20, 31, 40, 50, 63],
            "min_child_weight": [5, 10, 15, 20, 30],
            "subsample": [0.7, 0.8, 0.85, 0.9],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "min_split_gain": [0, 0.01, 0.1, 0.5],
        }
        lgbm_search = RandomizedSearchCV(
            LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
            lgbm_params, n_iter=50, cv=tscv,
            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
        )
        lgbm_search.fit(X_train, y_train)
        lgbm_preds = lgbm_search.predict(X_test)
        tuned_results["LightGBM (tuned)"] = {
            "model": lgbm_search.best_estimator_,
            "predictions": lgbm_preds,
            "use_scaled": False,
            "MAE": mean_absolute_error(y_test, lgbm_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, lgbm_preds)),
            "R²": r2_score(y_test, lgbm_preds),
            "best_params": lgbm_search.best_params_,
            "cv_score": -lgbm_search.best_score_,
        }
        print(f"    Best: {lgbm_search.best_params_}")
        print(f"    CV MAE={-lgbm_search.best_score_:.3f}  Test MAE={tuned_results['LightGBM (tuned)']['MAE']:.3f}")
    except ImportError:
        print("  ℹ LightGBM not installed — skipping")

    # ── Save tuning results ──
    tuning_rows = []
    for name, res in tuned_results.items():
        tuning_rows.append({
            "model": name,
            "cv_mae": res.get("cv_score", np.nan),
            "test_mae": res["MAE"],
            "test_rmse": res["RMSE"],
            "test_r2": res["R²"],
            "best_params": str(res.get("best_params", {})),
        })
    pd.DataFrame(tuning_rows).to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    print(f"\n  ✓ Tuning results saved to tuning_results.csv")

    return tuned_results, scaler


# ── Feature Importance ───────────────────────────────────────────────────────
def analyze_feature_importance(results, feature_cols):
    """Plot feature importance from the best tree model."""
    tree_names = ["Gradient Boosting", "XGBoost", "LightGBM", "Random Forest"]
    best = None
    for name in tree_names:
        if name in results:
            if best is None or results[name]["MAE"] < results[best]["MAE"]:
                best = name

    if best is None:
        return

    model = results[best]["model"]
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_cols, "importance": importances,
    }).sort_values("importance", ascending=True).tail(25)

    # Color by feature category
    def categorize(name):
        if name.startswith("cross_"): return "Pitch Matchup"
        if "lu_vs_" in name: return "Lineup vs Pitch"
        if name.startswith("pt_") and "_L" in name: return "Pitch-Type Stats"
        if any(x in name for x in ["_prev5","_prev10","pvt_","days_into","season_phase",
                                     "is_first_month","prior_starts","_x_reliability","early_x_"]): return "Early-Season"
        if "wx_" in name: return "Weather"
        if "lu_" in name or "lineup" in name: return "Lineup"
        if "opp_" in name: return "Opposing Team"
        if "ump_" in name: return "Umpire"
        if "pf_" in name or "dome" in name: return "Ballpark"
        if "trend" in name: return "Trend"
        if "_szn" in name: return "Season Cum."
        if "platoon" in name or "vs_left" in name or "vs_right" in name: return "Platoon"
        if any(x in name for x in ["rest", "night", "weekend", "month", "start_num", "is_home"]): return "Schedule"
        if "_std_" in name: return "Volatility"
        return "Rolling Stats"

    imp_df["category"] = imp_df["feature"].apply(categorize)
    colors = {
        "Rolling Stats": "#2563eb", "Season Cum.": "#7c3aed",
        "Trend": "#059669", "Opposing Team": "#dc2626",
        "Lineup": "#f59e0b", "Pitch-Type Stats": "#e11d48",
        "Lineup vs Pitch": "#7c2d12", "Pitch Matchup": "#b91c1c",
        "Early-Season": "#16a34a", "Weather": "#06b6d4",
        "Umpire": "#d97706", "Ballpark": "#0891b2",
        "Platoon": "#be185d", "Schedule": "#4b5563",
        "Volatility": "#9333ea",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(imp_df["feature"], imp_df["importance"],
                   color=[colors.get(c, "#6b7280") for c in imp_df["category"]])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 25 Features — {best}")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=colors[c], label=c)
                    for c in sorted(imp_df["category"].unique()) if c in colors]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  ✓ Feature importance plot saved")

    # Save full importance table
    full_imp = pd.DataFrame({
        "feature": feature_cols, "importance": importances,
    }).sort_values("importance", ascending=False)
    full_imp["category"] = full_imp["feature"].apply(categorize)
    full_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Print category contribution
    cat_imp = full_imp.groupby("category")["importance"].sum().sort_values(ascending=False)
    print("\n  Feature importance by category:")
    for cat, imp in cat_imp.items():
        print(f"    {cat:20s}  {imp:.3f}  ({imp/cat_imp.sum()*100:.1f}%)")


def run_shap_analysis(results, X_test, feature_cols):
    """SHAP analysis for interpretability (if shap is installed)."""
    try:
        import shap
    except ImportError:
        print("  ℹ SHAP not installed — skipping (pip install shap)")
        return

    # Use best tree model
    for name in ["XGBoost", "LightGBM", "Gradient Boosting"]:
        if name in results:
            model = results[name]["model"]
            print(f"\n  Running SHAP analysis on {name}...")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:500])  # Sample for speed

            fig, ax = plt.subplots(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test.iloc[:500], feature_names=feature_cols,
                              show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ SHAP summary plot saved")
            break


# ── Diagnostic Plots ─────────────────────────────────────────────────────────
def plot_predictions_vs_actual(results, y_test):
    best = min(results, key=lambda k: results[k]["MAE"])
    preds = results[best]["predictions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    axes[0].scatter(y_test, preds, alpha=0.2, s=10, c="#2563eb")
    mx = max(y_test.max(), preds.max()) + 1
    axes[0].plot([0, mx], [0, mx], "r--", lw=1.5)
    axes[0].set_xlabel("Actual Strikeouts")
    axes[0].set_ylabel("Predicted Strikeouts")
    axes[0].set_title(f"Predicted vs Actual — {best}")

    # Residual distribution
    residuals = y_test.values - preds
    axes[1].hist(residuals, bins=40, color="#2563eb", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Error (Actual - Predicted)")
    axes[1].set_title("Residual Distribution")
    axes[1].text(0.02, 0.95, f"Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}",
                 transform=axes[1].transAxes, va="top", fontsize=10,
                 bbox=dict(facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostic_plots.png", dpi=150)
    plt.close()
    print(f"  ✓ Diagnostic plots saved")


def plot_error_by_actual(results, y_test):
    """Show MAE broken down by actual strikeout count."""
    best = min(results, key=lambda k: results[k]["MAE"])
    preds = results[best]["predictions"]

    error_df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
    error_df["abs_error"] = np.abs(error_df["actual"] - error_df["predicted"])

    binned = error_df.groupby("actual").agg(
        mae=("abs_error", "mean"),
        count=("abs_error", "count"),
    ).reset_index()
    binned = binned[binned["count"] >= 10]  # Min sample size

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(binned["actual"], binned["mae"], color="#2563eb", alpha=0.7, label="MAE")
    ax1.set_xlabel("Actual Strikeouts")
    ax1.set_ylabel("Mean Absolute Error", color="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(binned["actual"], binned["count"], "ro-", markersize=4, label="Sample Size")
    ax2.set_ylabel("Number of Games", color="red")

    ax1.set_title(f"Prediction Error by Actual K Count — {best}")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_by_actual.png", dpi=150)
    plt.close()
    print(f"  ✓ Error by actual plot saved")


# ── Threshold Classification ─────────────────────────────────────────────────
def evaluate_threshold_classification(results, y_test):
    """How well does the point-estimate model separate starts above and below
    each strikeout threshold?

    This is a regression model, so it has no native probability. The only
    signal available is the distance between the prediction and the line: if
    pred > line + margin, the model is implicitly calling "over". Sweeping
    the margin trades coverage for precision — a wide margin fires rarely but
    should be right more often.

    Precision here is exactly that: of the starts where the model committed
    to a side, what fraction did it get right? A margin whose precision does
    not climb above the base rate means the model's confidence is not
    informative, which is the main limitation this script exists to expose
    and the reason the Beta-Binomial model in 06 predicts a full PMF instead.
    """
    best  = min(results, key=lambda k: results[k]["MAE"])
    preds = results[best]["predictions"]

    lines_     = [4.5, 5.5, 6.5]
    margins    = [0.3, 0.5, 0.75, 1.0]

    print(f"\n── Threshold Classification ({best}) ──")
    print(f"  {'Side':>6}  {'Line':>5}  {'Margin':>7}  {'N':>6}  "
          f"{'Correct':>8}  {'Precision':>10}  {'Base':>7}")

    rows = []
    for line in lines_:
        base_over  = (y_test.values > line).mean()
        base_under = 1 - base_over

        for margin in margins:
            for side, mask, correct, base in [
                ("over",  preds > (line + margin), y_test.values > line,  base_over),
                ("under", preds < (line - margin), y_test.values < line,  base_under),
            ]:
                n = int(mask.sum())
                if n == 0:
                    continue
                n_correct = int(correct[mask].sum())
                precision = n_correct / n
                print(f"  {side:>6}  {line:>5.1f}  {margin:>7.2f}  {n:>6}  "
                      f"{n_correct:>8}  {precision:>9.1%}  {base:>6.1%}")
                rows.append({
                    "side": side, "line": line, "margin": margin,
                    "n": n, "n_correct": n_correct,
                    "precision": precision, "base_rate": base,
                    "lift_over_base": precision - base,
                })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_DIR / "threshold_classification.csv", index=False)
        print(f"\n  ✓ Threshold classification saved")
        mean_lift = df["lift_over_base"].mean()
        print(f"  Mean lift over base rate: {mean_lift:+.1%} "
              f"({'informative' if mean_lift > 0.02 else 'weak — see 06 for the distributional model'})")


# ── Comparison Table ─────────────────────────────────────────────────────────
def generate_comparison(default_results, tuned_results, baselines):
    rows = []
    for name, m in baselines.items():
        rows.append({"Model": f"⬜ {name}", "Type": "baseline",
                      "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R²"]})
    for name, m in default_results.items():
        rows.append({"Model": f"🟦 {name}", "Type": "default",
                      "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R²"]})
    for name, m in tuned_results.items():
        cv_mae = m.get("cv_score", np.nan)
        rows.append({"Model": f"🟩 {name}", "Type": "tuned", "CV MAE": cv_mae,
                      "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R²"]})

    comp = pd.DataFrame(rows).sort_values("MAE")
    comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n═══ Full Model Comparison (sorted by Test MAE) ═══")
    print(comp.to_string(index=False, float_format="%.3f"))

    # Show improvement from tuning
    print("\n── Tuning Improvement ──")
    for tuned_name, tuned_m in tuned_results.items():
        base_name = tuned_name.replace(" (tuned)", "")
        if base_name in default_results:
            default_mae = default_results[base_name]["MAE"]
            tuned_mae = tuned_m["MAE"]
            improvement = default_mae - tuned_mae
            pct = improvement / default_mae * 100
            print(f"  {base_name:25s}  {default_mae:.3f} → {tuned_mae:.3f}  "
                  f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.3f}, {abs(pct):.1f}%)")

    return comp


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MLB Pitcher Model Training (with Lineup Features & Hyperparameter Tuning)")
    print("=" * 75)

    # Load
    print("\n── Loading features ──")
    df = load_features()
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Features
    feature_cols = select_features(df)

    # Split
    print("\n── Splitting data ──")
    X_train, X_test, y_train, y_test, train_df, test_df = time_split(df, feature_cols)

    # Baselines
    print("\n── Baselines ──")
    baselines = compute_baselines(X_test, y_test, feature_cols)

    # Phase 1: Train with defaults (fast, gives a comparison point)
    print("\n── Phase 1: Training with Default Hyperparameters ──")
    default_results, scaler = train_default_models(X_train, y_train, X_test, y_test)

    # Phase 2: Hyperparameter tuning (slower, should improve results)
    print("\n── Phase 2: Hyperparameter Tuning ──")
    print("  This will take 10-30 minutes depending on dataset size and hardware.")
    tuned_results, scaler = tune_hyperparameters(X_train, y_train, X_test, y_test, train_df)

    # Combine all results for comparison
    all_model_results = {**default_results, **tuned_results}

    # Full comparison
    comparison = generate_comparison(default_results, tuned_results, baselines)

    # Analysis (use best tuned model)
    print("\n── Analysis & Plots ──")
    analyze_feature_importance(all_model_results, feature_cols)
    run_shap_analysis(all_model_results, X_test, feature_cols)
    plot_predictions_vs_actual(all_model_results, y_test)
    plot_error_by_actual(all_model_results, y_test)

    # Threshold classification with best tuned model
    evaluate_threshold_classification(all_model_results, y_test)

    # ── Early-season vs full-season accuracy breakdown ──
    best = min(all_model_results, key=lambda k: all_model_results[k]["MAE"])
    test_df = test_df.copy()
    test_df["predicted_K"] = all_model_results[best]["predictions"]
    test_df["error"] = test_df[TARGET] - test_df["predicted_K"]

    print("\n── Accuracy by Season Phase ──")
    if "days_into_season" in test_df.columns:
        phases = [
            ("First 2 weeks (days 0-14)", test_df["days_into_season"] <= 14),
            ("Weeks 3-4 (days 15-30)", test_df["days_into_season"].between(15, 30)),
            ("Month 2 (days 31-60)", test_df["days_into_season"].between(31, 60)),
            ("Month 3+ (days 61+)", test_df["days_into_season"] > 60),
            ("Full season", pd.Series(True, index=test_df.index)),
        ]
        for label, mask in phases:
            subset = test_df[mask]
            if len(subset) > 10:
                mae = mean_absolute_error(subset[TARGET], subset["predicted_K"])
                r2 = r2_score(subset[TARGET], subset["predicted_K"]) if len(subset) > 2 else 0
                print(f"  {label:35s}  MAE={mae:.3f}  R²={r2:.3f}  (n={len(subset):,})")
    elif "month" in test_df.columns:
        for m in sorted(test_df["month"].dropna().unique()):
            subset = test_df[test_df["month"] == m]
            if len(subset) > 10:
                mae = mean_absolute_error(subset[TARGET], subset["predicted_K"])
                month_names = {3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October"}
                print(f"  {month_names.get(int(m), f'Month {int(m)}'):35s}  MAE={mae:.3f}  (n={len(subset):,})")

    if "start_num" in test_df.columns:
        print("\n── Accuracy by Pitcher's Start Number ──")
        for start_range, label in [((1, 3), "Starts 1-3"), ((4, 6), "Starts 4-6"),
                                    ((7, 12), "Starts 7-12"), ((13, 35), "Starts 13+")]:
            subset = test_df[test_df["start_num"].between(*start_range)]
            if len(subset) > 10:
                mae = mean_absolute_error(subset[TARGET], subset["predicted_K"])
                print(f"  {label:35s}  MAE={mae:.3f}  (n={len(subset):,})")

    # Save predictions
    test_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Save best tuned model params for reproduction
    best_tuned = min(tuned_results, key=lambda k: tuned_results[k]["MAE"])
    print(f"\n{'='*75}")
    print(f"Best overall model:  {best} (MAE={all_model_results[best]['MAE']:.3f})")
    print(f"Best tuned model:    {best_tuned} (MAE={tuned_results[best_tuned]['MAE']:.3f})")
    if "best_params" in tuned_results[best_tuned]:
        print(f"Best tuned params:   {tuned_results[best_tuned]['best_params']}")

    # Save the best model, scaler, and feature list for daily predictions
    import pickle
    model_bundle = {
        "model": all_model_results[best]["model"],
        "scaler": scaler,
        "feature_cols": feature_cols,
        "use_scaled": all_model_results[best].get("use_scaled", False),
        "model_name": best,
        "target": TARGET,
        "mae": all_model_results[best]["MAE"],
    }
    with open(OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n  ✓ Saved model bundle to best_model.pkl")

    print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name}")
