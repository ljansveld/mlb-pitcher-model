"""
train/hits_walks_per_pa.py
==============================
Per-plate-appearance binary models for hits allowed and walks, trained on
TRUE BINARY OUTCOMES.

Mirrors the architecture of train/strikeouts_per_pa.py (strikeouts) but trains
two separate XGBoost binary classifiers in one pass:

    was_hit  = 1 if the PA ended in a hit (single/double/triple/HR), 0 otherwise
    was_BB   = 1 if the PA ended in a walk, 0 otherwise

Both models are trained on true PA-level binary outcomes from the
statcast_pa_events_all.csv file (the same source as 08). The inference
pattern in predict/hits_walks.py is identical to 08's:
  1. Score P(hit) and P(BB) for each projected PA in a start using
     the actual posted lineup (real per-batter features).
  2. Sum probabilities across all PAs → Poisson-Binomial PMF.

INPUTS
------
- data/statcast_pa_events_all.csv       (from collect/statcast.py patch)
- data/pitcher_model_features.csv       (from features.py)

OUTPUTS
-------
- models/per_pa_hit_model.joblib
- models/per_pa_hit_config.json
- models/per_pa_bb_model.joblib
- models/per_pa_bb_config.json

RUN ORDER
---------
1. Run collect/statcast.py (with PA events patch applied)
2. Run features.py
3. Run train/strikeouts.py    (K BB models)
4. Run train/strikeouts_per_pa.py           (K per-PA model)
5. Run train/hits_walks.py       (H/W BB models)
6. Run train/hits_walks_per_pa.py   ← this file
7. Run predict/hits_walks.py       (uses both BB and per-PA, side by side)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss

from pitcher_model.paths import DATA_DIR, MODEL_DIR, ensure_dirs

ensure_dirs(MODEL_DIR)


PA_EVENTS_PATH        = DATA_DIR / "statcast_pa_events_all.csv"
PITCHER_FEATURES_PATH = DATA_DIR / "pitcher_model_features.csv"

TEST_YEAR    = 2025
RANDOM_STATE = 42

# Pitcher-side features to attach to each PA. These come from your
# start-level features CSV and describe the pitcher's recent form.
#
# RECENCY-BIAS FIX (April 2026): We dropped raw k_pct_L3 / bb_pct_L3 /
# whiff_pct_L3 / hits_per_pa_L3. They were an empirical liability — in
# the first 4-6 weeks of a season, a single bad start drops a 3-start
# rolling rate by ~10 percentage points and there's no shrinkage because
# the 3-start window is "full". This caused mispredictions where one
# bad start flipped the model's view of an elite pitcher.
#
# What replaces them:
#   - *_szn_blended: empirical-Bayes-shrunk season rate (100-PA prior toward
#     last year's final-10-start rate). Stable in April, converges to the
#     raw season rate by June.
#   - *_L5_blended:  L5 with a 5-start prior toward last year's rate.
#   - *_prev10:      hard anchor to last year (unchanged within a season).
#   - prior_starts_this_season: lets the model learn "early April → trust
#     prev10 more" natively.
PITCHER_FEATS_KEEP = [
    # Season-to-date rates — BLENDED versions (prefer over raw _szn)
    "k_pct_szn_blended", "bb_pct_szn_blended", "whiff_pct_szn_blended",
    "csw_pct_szn_blended", "barrel_pct_szn_blended",
    "hard_hit_pct_szn_blended",
    # Raw season rates kept for training-time comparison (pruner may
    # drop them naturally if the blended ones correlate better with y)
    "k_pct_szn", "bb_pct_szn", "whiff_pct_szn", "csw_pct_szn",
    "hits_per_pa_szn", "barrel_pct_szn", "hard_hit_pct_szn",
    # L5 windows — both raw and blended
    "k_pct_L5", "k_pct_L5_blended",
    "bb_pct_L5", "whiff_pct_L5", "whiff_pct_L5_blended",
    "csw_pct_L5", "hits_per_pa_L5",
    "barrel_pct_L5", "barrel_pct_L5_blended",
    "hard_hit_pct_L5", "hard_hit_pct_L5_blended",
    # L10 is stable enough to keep raw (≈ end of April for a full-season vet)
    "k_pct_L10", "bb_pct_L10", "hits_per_pa_L10",
    # Hit-quality allowed
    "hr_rate_L5",
    # Prior-year anchors (the BIG win — these were computed but never used)
    "k_pct_prev10", "k_pct_prev5",
    "bb_pct_prev10", "bb_pct_prev5",
    "whiff_pct_prev10", "csw_pct_prev10",
    "barrel_pct_prev10", "hard_hit_pct_prev10",
    # Reliability signals — lets the model weight recent vs prior differently
    "prior_starts_this_season", "prior_starts_available",
    "n_starts_in_L5", "n_starts_in_L10",
    # Pitch mix (seasonal, stable)
    "fastball_pct_szn", "sl_pct_szn", "cu_pct_szn", "ch_pct_szn",
    # Velocity
    "avg_velocity_L5", "avg_velocity_szn",
    # FanGraphs quality grades (very stable year-to-year for established pitchers)
    "fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
    # Game context
    "rest_days", "is_home", "is_night_game", "wx_is_dome",
    "wx_temperature_f", "wx_effective_wind",
    "pitcher_throws_R",
    "catcher_frm",
    "ump_k_pct",
    "pf_Basic",
    # ══════════════════════════════════════════════════════════════════
    # HITS / WALKS PIPELINE ADDITIONS
    # ══════════════════════════════════════════════════════════════════
    # Per-PA models for hits and walks need direct rate signals for the
    # outcome they're predicting. The K-only feature set above gave the
    # per-PA hits model a blind spot for hit-rate-specific signal.
    #
    # Blended (preferred when available)
    "hits_per_pa_szn_blended", "bb_per_pa_szn_blended",
    "hits_per_pa_L5_blended",  "bb_per_pa_L5_blended",
    "babip_szn_blended", "babip_L5_blended",
    "lob_pct_szn_blended",
    "gb_pct_szn_blended", "fb_pct_szn_blended", "ld_pct_szn_blended",
    # Quality of contact (allowed) — predicts hit conversion on BIP
    "avg_exit_velocity_szn_blended", "avg_xwoba_contact_szn_blended",
    "sweet_spot_pct_szn_blended",
    # Raw season versions (training-time comparison)
    "bb_per_pa_szn", "hr_per_pa_szn", "babip_szn", "lob_pct_szn",
    "gb_pct_szn", "fb_pct_szn", "ld_pct_szn",
    "avg_exit_velocity_szn", "avg_xwoba_contact_szn",
    # Prior-year anchors for H/W (early-season stabilizers)
    "hits_per_pa_prev10", "bb_per_pa_prev10", "hr_per_pa_prev10",
    "babip_prev10", "lob_pct_prev10",
    "gb_pct_prev10", "fb_pct_prev10", "ld_pct_prev10",
    "avg_exit_velocity_prev10", "avg_xwoba_contact_prev10",
    # FanGraphs hits-side talent metrics (season-level, very stable)
    "fg_siera_prev", "fg_xfip_prev", "fg_fip_prev",
    "fg_lob_pct_prev", "fg_hr_per_fb_prev", "fg_k_minus_bb_pct_prev",
    "fg_gb_pct_prev", "fg_fb_pct_prev", "fg_ld_pct_prev",
    "fg_babip_allowed_prev", "fg_barrel_pct_allowed_prev", "fg_hard_hit_pct_allowed_prev",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & JOIN
# ══════════════════════════════════════════════════════════════════════════════

def load_and_join():
    """Build the PA-level training frame: events → pitcher features → batter features."""
    print("── Step 1: Load PA events ──")
    if not PA_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"{PA_EVENTS_PATH} not found. Run `python run.py collect statcast` "
            f"to produce it."
        )
    pa = pd.read_csv(PA_EVENTS_PATH)
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa["year"] = pa["game_date"].dt.year

    # Ensure the binary outcome columns we need exist.
    # was_hit and was_BB should already be in the events CSV from collect/statcast.py.
    required = ["was_K", "was_BB", "was_hit", "was_in_play"]
    missing_cols = [c for c in required if c not in pa.columns]
    if missing_cols:
        # Reconstruct from events column if available
        if "events" in pa.columns:
            print(f"  Building missing columns from events: {missing_cols}")
            if "was_hit" in missing_cols:
                pa["was_hit"] = pa["events"].isin(
                    ["single", "double", "triple", "home_run"]
                ).astype(int)
            if "was_BB" in missing_cols:
                pa["was_BB"] = pa["events"].isin(["walk"]).astype(int)
            if "was_K" in missing_cols:
                pa["was_K"] = pa["events"].isin(
                    ["strikeout", "strikeout_double_play"]
                ).astype(int)
            if "was_in_play" in missing_cols:
                pa["was_in_play"] = pa["events"].isin(
                    ["single", "double", "triple", "home_run",
                     "field_out", "grounded_into_double_play",
                     "force_out", "double_play", "triple_play",
                     "sac_fly", "sac_bunt", "fielders_choice",
                     "fielders_choice_out"]
                ).astype(int)
        else:
            raise ValueError(
                f"PA events CSV is missing columns {missing_cols} and no "
                "'events' column to reconstruct from."
            )

    print(f"  Loaded {len(pa):,} PAs across "
          f"{pa['year'].min()}–{pa['year'].max()}")
    print(f"  League hit rate: {pa['was_hit'].mean():.3f}   "
          f"walk rate: {pa['was_BB'].mean():.3f}")

    print("\n── Step 2: Join pitcher-game features ──")
    pf = pd.read_csv(PITCHER_FEATURES_PATH)
    pf["game_date"] = pd.to_datetime(pf["game_date"], errors="coerce")
    available_pf = [c for c in PITCHER_FEATS_KEEP if c in pf.columns]
    missing = [c for c in PITCHER_FEATS_KEEP if c not in pf.columns]
    if missing:
        print(f"  (skipping {len(missing)} missing pitcher cols: {missing[:5]}...)")
    pf_slim = pf[["game_pk", "pitcher"] + available_pf].drop_duplicates(
        subset=["game_pk", "pitcher"]
    )
    pa = pa.merge(pf_slim, on=["game_pk", "pitcher"], how="inner")
    print(f"  After pitcher join: {len(pa):,} PAs "
          f"({len(available_pf)} pitcher features attached)")

    print("\n── Step 3: Build Statcast-derived batter features ──")
    # We skip FanGraphs (scraper is blocked) and derive batter-side
    # features directly from the PA events CSV. For each (batter, PA)
    # we compute season-to-date and rolling-L100 rates using ONLY PAs
    # strictly before that row — achieved via shift(1) inside groupby.
    pa = pa.sort_values(["batter", "year", "game_date", "at_bat_number"]).reset_index(drop=True)

    # Season-to-date: expanding mean of prior outcomes within (batter, year)
    g = pa.groupby(["batter", "year"], sort=False)
    for src, dst in [("was_K", "batter_k_rate_std"),
                     ("was_BB", "batter_bb_rate_std"),
                     ("was_hit", "batter_hit_rate_std"),
                     ("was_in_play", "batter_bip_rate_std")]:
        shifted = g[src].shift(1)
        pa[dst] = shifted.groupby([pa["batter"], pa["year"]]).expanding().mean()\
                         .reset_index(level=[0, 1], drop=True)

    # Rolling L100 PAs (a more recent-form signal) — within batter only, not reset by year
    g_all = pa.groupby(["batter"], sort=False)
    for src, dst in [("was_K", "batter_k_rate_L100"),
                     ("was_BB", "batter_bb_rate_L100"),
                     ("was_hit", "batter_hit_rate_L100")]:
        shifted = g_all[src].shift(1)
        pa[dst] = shifted.groupby(pa["batter"]).rolling(100, min_periods=20).mean()\
                         .reset_index(level=0, drop=True)

    # Prior-year rates (mirror the spirit of the old FG prior-year join —
    # stable baseline stats that aren't affected by current-season form).
    # Computed as the full-season mean per (batter, year), then shifted
    # to the next year so year N gets year-(N-1) stats.
    prior = (pa.groupby(["batter", "year"])
               .agg(batter_k_rate_prior=("was_K", "mean"),
                    batter_bb_rate_prior=("was_BB", "mean"),
                    batter_hit_rate_prior=("was_hit", "mean"),
                    batter_bip_rate_prior=("was_in_play", "mean"),
                    batter_pa_prior=("was_K", "size"))
               .reset_index())
    prior["year"] = prior["year"].astype(int) + 1
    pa = pa.merge(prior, on=["batter", "year"], how="left")

    # Fill missing (rookies, early-season, small L100 windows) with league means.
    # These are gentle priors — the model will learn to discount them.
    league_k   = pa["was_K"].mean()
    league_bb  = pa["was_BB"].mean()
    league_hit = pa["was_hit"].mean()
    league_bip = pa["was_in_play"].mean()
    fill_map = {
        "batter_k_rate_std": league_k, "batter_k_rate_L100": league_k,
        "batter_k_rate_prior": league_k,
        "batter_bb_rate_std": league_bb, "batter_bb_rate_L100": league_bb,
        "batter_bb_rate_prior": league_bb,
        "batter_hit_rate_std": league_hit, "batter_hit_rate_L100": league_hit,
        "batter_hit_rate_prior": league_hit,
        "batter_bip_rate_std": league_bip,
        "batter_bip_rate_prior": league_bip,
        "batter_pa_prior": 0.0,  # zero prior PAs flags "no prior year data"
    }
    for col, val in fill_map.items():
        if col in pa.columns:
            pa[col] = pa[col].fillna(val)

    batter_feats = list(fill_map.keys())
    print(f"  Built {len(batter_feats)} batter-side features from "
          f"{len(pa):,} prior PAs (leakage-free via shift(1))")

    print(f"\n  Final frame: {len(pa):,} rows, {len(pa.columns)} cols")
    return pa, available_pf, batter_feats


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_one(pa, pitcher_feats, batter_feats, target_col, label,
              model_path, config_path):
    """
    Train a single binary XGBoost classifier for `target_col` (was_hit or was_BB).

    Mirrors train/strikeouts_per_pa.py's train() function exactly — same XGBoost
    hyperparameters, same calibration & diagnostics, same artifact format.
    The only difference is the interaction-feature set, which is tailored
    to each outcome (hits care about contact quality; walks care about
    command).

    Returns (model, features, metrics).
    """
    print(f"\n{'=' * 70}")
    print(f"  PER-PA MODEL: {label}  (target = {target_col})")
    print(f"{'=' * 70}")

    # Interaction features: simple products of pitcher signal × batter signal.
    # We now prefer the BLENDED pitcher rate over the raw _szn for interactions,
    # for the same reason as the direct features: in April the raw _szn is
    # dominated by the last 1-2 starts, which makes interaction features spike.
    pitcher_k_src = ("k_pct_szn_blended" if "k_pct_szn_blended" in pa.columns
                     else "k_pct_szn")
    pitcher_bb_src = ("bb_pct_szn_blended" if "bb_pct_szn_blended" in pa.columns
                      else "bb_pct_szn")
    pitcher_whiff_src = ("whiff_pct_szn_blended" if "whiff_pct_szn_blended" in pa.columns
                         else "whiff_pct_szn")
    pitcher_barrel_src = ("barrel_pct_szn_blended" if "barrel_pct_szn_blended" in pa.columns
                          else "barrel_pct_szn")

    # K interaction (shared across both models — useful contact signal)
    if pitcher_k_src in pa.columns and "batter_k_rate_prior" in pa.columns:
        pa["ix_pitcher_k_x_batter_k"] = pa[pitcher_k_src] * pa["batter_k_rate_prior"]
    if pitcher_k_src in pa.columns and "batter_k_rate_L100" in pa.columns:
        pa["ix_pitcher_k_x_batter_k_recent"] = pa[pitcher_k_src] * pa["batter_k_rate_L100"]

    if target_col == "was_hit":
        # Pitcher hit-rate × batter hit-rate (current + recent)
        if "hits_per_pa_szn" in pa.columns and "batter_hit_rate_prior" in pa.columns:
            pa["ix_pitcher_hit_x_batter_hit"] = (
                pa["hits_per_pa_szn"] * pa["batter_hit_rate_prior"]
            )
        if "hits_per_pa_szn" in pa.columns and "batter_hit_rate_L100" in pa.columns:
            pa["ix_pitcher_hit_x_batter_hit_recent"] = (
                pa["hits_per_pa_szn"] * pa["batter_hit_rate_L100"]
            )
        # Hard-contact proxy: pitcher barrel allowed × batter BIP rate
        if pitcher_barrel_src in pa.columns and "batter_bip_rate_std" in pa.columns:
            pa["ix_barrel_x_bip"] = (
                pa[pitcher_barrel_src] * pa["batter_bip_rate_std"]
            )
        # Inverse-of-contact (high pitcher whiff × low batter BIP = harder to hit)
        if pitcher_whiff_src in pa.columns and "batter_bip_rate_std" in pa.columns:
            pa["ix_whiff_x_contact"] = (
                pa[pitcher_whiff_src] * (1 - pa["batter_bip_rate_std"])
            )
        interaction_feats = [
            "ix_pitcher_k_x_batter_k",
            "ix_pitcher_k_x_batter_k_recent",
            "ix_pitcher_hit_x_batter_hit",
            "ix_pitcher_hit_x_batter_hit_recent",
            "ix_barrel_x_bip",
            "ix_whiff_x_contact",
        ]

    else:  # was_BB
        # Pitcher walk-rate × batter walk-rate (current + recent)
        if pitcher_bb_src in pa.columns and "batter_bb_rate_prior" in pa.columns:
            pa["ix_pitcher_bb_x_batter_bb"] = (
                pa[pitcher_bb_src] * pa["batter_bb_rate_prior"]
            )
        if pitcher_bb_src in pa.columns and "batter_bb_rate_L100" in pa.columns:
            pa["ix_pitcher_bb_x_batter_bb_recent"] = (
                pa[pitcher_bb_src] * pa["batter_bb_rate_L100"]
            )
        # Whiff × contact (command/discipline proxy)
        if pitcher_whiff_src in pa.columns and "batter_bip_rate_std" in pa.columns:
            pa["ix_whiff_x_contact"] = (
                pa[pitcher_whiff_src] * (1 - pa["batter_bip_rate_std"])
            )
        interaction_feats = [
            "ix_pitcher_k_x_batter_k",
            "ix_pitcher_k_x_batter_k_recent",
            "ix_pitcher_bb_x_batter_bb",
            "ix_pitcher_bb_x_batter_bb_recent",
            "ix_whiff_x_contact",
        ]

    # Platoon flag
    if "p_throws" in pa.columns and "stand" in pa.columns:
        pa["same_handed"] = (pa["p_throws"] == pa["stand"]).astype(int)

    # Build feature list — preserve order, dedup, drop missing
    base_feats = list(dict.fromkeys(
        pitcher_feats + batter_feats + interaction_feats +
        ["same_handed", "inning"]
    ))
    feats = [c for c in base_feats if c in pa.columns]
    for c in feats:
        pa[c] = pd.to_numeric(pa[c], errors="coerce").fillna(0.0)

    train_df = pa[pa["year"] < TEST_YEAR].copy()
    test_df  = pa[pa["year"] >= TEST_YEAR].copy()
    print(f"\n── Train/Test Split ──")
    print(f"  Train PAs: {len(train_df):,}  Test PAs: {len(test_df):,}")
    print(f"  Feature count: {len(feats)}")
    tr_rate = train_df[target_col].mean()
    te_rate = test_df[target_col].mean()
    print(f"  Train {label} rate: {tr_rate:.3f}  Test: {te_rate:.3f}")

    X_tr = train_df[feats].values
    y_tr = train_df[target_col].values
    X_te = test_df[feats].values
    y_te = test_df[target_col].values

    # XGBoost binary:logistic — same hyper-parameter choices as 08.
    # min_child_weight=50 prevents overfitting on noisy PA-level labels.
    model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=50,  # higher for PA-level (lots of rows, noisy labels)
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    p_te = model.predict_proba(X_te)[:, 1]
    ll = log_loss(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)

    # Baseline: predict the train mean for every PA. Gives log-loss floor
    # for "no features."
    baseline = np.full_like(y_te, y_tr.mean(), dtype=float)
    ll_base = log_loss(y_te, baseline)
    brier_base = brier_score_loss(y_te, baseline)

    print(f"\n  Test log-loss:   {ll:.4f}   (baseline: {ll_base:.4f}  "
          f"Δ={ll-ll_base:+.4f})")
    print(f"  Test Brier:      {brier:.4f}  (baseline: {brier_base:.4f}  "
          f"Δ={brier-brier_base:+.4f})")
    if ll >= ll_base:
        print(f"  ⚠ Model does not beat the constant-mean baseline for {label}. "
              "Check feature joins and leakage.")

    # Calibration by deciles
    print(f"\n  Calibration (predicted p vs realized rate, by decile):")
    bins = pd.qcut(p_te, q=10, labels=False, duplicates="drop")
    for b in sorted(np.unique(bins)):
        mask = bins == b
        print(f"    bin {b}: n={mask.sum():>6}  "
              f"pred={p_te[mask].mean():.3f}  actual={y_te[mask].mean():.3f}")

    # Top features
    try:
        imp = model.feature_importances_
        ranked = sorted(zip(feats, imp), key=lambda t: t[1], reverse=True)
        print(f"\n  Top 15 features ({label}):")
        for name, v in ranked[:15]:
            print(f"    {v:.4f}  {name}")
    except Exception:
        pass

    # Save
    joblib.dump(model, model_path)
    metrics = {
        "log_loss": float(ll), "brier": float(brier),
        "baseline_log_loss": float(ll_base),
        "baseline_brier": float(brier_base),
    }
    config = {
        "target": target_col,
        "features": feats,
        "test_year": TEST_YEAR,
        "metrics": metrics,
        "league_rate": float(te_rate),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✓ Saved {model_path}")
    print(f"  ✓ Saved {config_path}")

    return model, feats, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PER-PA HITS & WALKS MODELS (true binary labels)")
    print("=" * 70)

    pa, pitcher_feats, batter_feats = load_and_join()

    # Train hits model — work on a copy so interaction columns don't bleed
    pa_hit = pa.copy()
    train_one(
        pa_hit, pitcher_feats, batter_feats,
        target_col="was_hit",
        label="Hits Allowed",
        model_path=MODEL_DIR / "per_pa_hit_model.joblib",
        config_path=MODEL_DIR / "per_pa_hit_config.json",
    )

    # Train walks model
    pa_bb = pa.copy()
    train_one(
        pa_bb, pitcher_feats, batter_feats,
        target_col="was_BB",
        label="Walks",
        model_path=MODEL_DIR / "per_pa_bb_model.joblib",
        config_path=MODEL_DIR / "per_pa_bb_config.json",
    )

    print("\n" + "=" * 70)
    print("  DONE. Next step: use these models inside a Poisson-Binomial")
    print("  aggregator (predict P(hit)/P(BB) for each projected PA in a")
    print("  start, convolve into distribution, compare log-loss vs 10's BB).")
    print("  → python run.py predict hits-walks")
    print("=" * 70)


if __name__ == "__main__":
    main()
