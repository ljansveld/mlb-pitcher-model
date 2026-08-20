"""
train/strikeouts_per_pa.py
==================
Per-plate-appearance strikeout model trained on TRUE BINARY OUTCOMES.

Unlike 07 (which uses season K rates as soft labels), this model trains
on PA-level events from Statcast: was_K = 1 if the PA ended in a K, 0
otherwise. This is the "right" way to build a per-batter K model — it
lets the learner pick up on actual matchup interactions (batter X's
whiff rate on sliders × pitcher Y's slider usage, etc.) rather than
being capped by season-rate coupling.

INPUTS
------
- data/statcast_pa_events_all.csv
  Produced by collect/statcast.py.
  One row per PA with was_K, stand, p_throws, game context.

- data/pitcher_model_features.csv
  Your existing start-level feature table from features.py.
  Used to attach pitcher-side rolling features to each PA.

- data/fangraphs_batting_seasons.csv (optional)
  Season batter stats used for batter-side features.

OUTPUTS
-------
- models/per_pa_k_model.joblib
- models/per_pa_k_config.json  (feature list + calibration + metrics)

RUN ORDER
---------
1. Run collect/statcast.py with the patch applied (creates statcast_pa_events_all.csv)
2. Run features.py
3. Run train/strikeouts.py
4. Run train/strikeouts_per_pa.py   ← this file
5. (Optional) Run 07 for the season-rate PB path as a baseline
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


PA_EVENTS_PATH = DATA_DIR / "statcast_pa_events_all.csv"
PITCHER_FEATURES_PATH = DATA_DIR / "pitcher_model_features.csv"
FG_BATTING_PATH = DATA_DIR / "fangraphs_batting_seasons.csv"

TEST_YEAR = 2025
RANDOM_STATE = 42

# Pitcher-side features to attach to each PA. These come from your
# start-level features CSV and describe the pitcher's recent form.
#
# RECENCY-BIAS FIX (April 2026): We dropped raw k_pct_L3 and whiff_pct_L3.
# They were an empirical liability — in the first 4-6 weeks of a season,
# a single 2-K outing drops k_pct_L3 by ~10 percentage points and there's
# no shrinkage because the 3-start window is "full". This caused the
# Crochet-type mispredictions where one bad start flipped the model's
# view of an elite pitcher.
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
    # Season-to-date rates — BLENDED versions (prefer these over raw _szn)
    "k_pct_szn_blended", "whiff_pct_szn_blended", "csw_pct_szn_blended",
    # Raw season rates kept for training-time comparison (pruner may
    # drop them naturally if the blended ones correlate better with y)
    "k_pct_szn", "whiff_pct_szn", "csw_pct_szn",
    # L5 windows — both raw and blended
    "k_pct_L5", "k_pct_L5_blended",
    "whiff_pct_L5", "whiff_pct_L5_blended",
    "csw_pct_L5",
    # L10 is stable enough to keep raw (10 starts ≈ end of April for a full-season vet)
    "k_pct_L10",
    # Prior-year anchors (the BIG win — these were computed but never used)
    "k_pct_prev10", "k_pct_prev5",
    "whiff_pct_prev10", "csw_pct_prev10",
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
]

# Batter-side features from FanGraphs (prefixed fg_ after merge).
BATTER_FG_FEATS = [
    "K%", "BB%", "SwStr%", "Contact%", "O-Swing%", "Z-Contact%",
    "Hard%", "Barrel%", "wRC+", "xwOBA",
]


# ──────────────────────────────────────────────────────────────────────────────
# LOAD & JOIN
# ──────────────────────────────────────────────────────────────────────────────

def load_and_join():
    """Build the PA-level training frame by joining events → pitcher → batter."""
    print("── Step 1: Load PA events ──")
    if not PA_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"{PA_EVENTS_PATH} not found. Run `python run.py collect statcast` "
            f"to produce it."
        )
    pa = pd.read_csv(PA_EVENTS_PATH)
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa["year"] = pa["game_date"].dt.year
    print(f"  Loaded {len(pa):,} PAs across "
          f"{pa['year'].min()}–{pa['year'].max()}")
    print(f"  League K rate: {pa['was_K'].mean():.3f}")

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
                    batter_pa_prior=("was_K", "size"))
               .reset_index())
    prior["year"] = prior["year"].astype(int) + 1
    pa = pa.merge(prior, on=["batter", "year"], how="left")

    # Fill missing (rookies, early-season, small L100 windows) with league means.
    # These are gentle priors — the model will learn to discount them.
    league_k = pa["was_K"].mean()
    league_bb = pa["was_BB"].mean()
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


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def train(pa, pitcher_feats, batter_feats):
    """Train XGBoost classifier with binary:logistic on was_K."""
    # Interaction features: simple products of pitcher K signal × batter K signal.
    # We now prefer the BLENDED pitcher rate over the raw _szn for interactions,
    # for the same reason as the direct features: in April the raw _szn is
    # dominated by the last 1-2 starts, which makes interaction features spike.
    pitcher_k_src = ("k_pct_szn_blended" if "k_pct_szn_blended" in pa.columns
                     else "k_pct_szn")
    pitcher_whiff_src = ("whiff_pct_szn_blended" if "whiff_pct_szn_blended" in pa.columns
                         else "whiff_pct_szn")

    if pitcher_k_src in pa.columns and "batter_k_rate_prior" in pa.columns:
        pa["ix_pitcher_k_x_batter_k"] = pa[pitcher_k_src] * pa["batter_k_rate_prior"]
    if pitcher_k_src in pa.columns and "batter_k_rate_L100" in pa.columns:
        pa["ix_pitcher_k_x_batter_k_recent"] = pa[pitcher_k_src] * pa["batter_k_rate_L100"]
    if pitcher_whiff_src in pa.columns and "batter_bip_rate_std" in pa.columns:
        # Inverse of contact proxy — high pitcher whiff × low batter BIP = K risk
        pa["ix_whiff_x_contact"] = pa[pitcher_whiff_src] * (1 - pa["batter_bip_rate_std"])
    if "p_throws" in pa.columns and "stand" in pa.columns:
        # 1 if pitcher and batter are same-handed (platoon advantage to pitcher)
        pa["same_handed"] = (pa["p_throws"] == pa["stand"]).astype(int)

    # Build feature list
    base_feats = list(dict.fromkeys(
        pitcher_feats + batter_feats +
        ["same_handed", "ix_pitcher_k_x_batter_k",
         "ix_pitcher_k_x_batter_k_recent", "ix_whiff_x_contact", "inning"]
    ))
    feats = [c for c in base_feats if c in pa.columns]
    for c in feats:
        pa[c] = pd.to_numeric(pa[c], errors="coerce").fillna(0.0)

    train_df = pa[pa["year"] < TEST_YEAR].copy()
    test_df = pa[pa["year"] >= TEST_YEAR].copy()
    print(f"\n── Step 4: Train ──")
    print(f"  Train PAs: {len(train_df):,}  Test PAs: {len(test_df):,}")
    print(f"  Feature count: {len(feats)}")
    print(f"  Train K rate: {train_df['was_K'].mean():.3f}  "
          f"Test K rate: {test_df['was_K'].mean():.3f}")

    X_tr = train_df[feats].values
    y_tr = train_df["was_K"].values
    X_te = test_df[feats].values
    y_te = test_df["was_K"].values

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
        print("  ⚠ Model does not beat the constant-mean baseline. "
              "Check feature joins and leakage.")

    # Calibration by deciles
    print("\n  Calibration (predicted p vs realized rate, by decile):")
    bins = pd.qcut(p_te, q=10, labels=False, duplicates="drop")
    for b in sorted(np.unique(bins)):
        mask = bins == b
        print(f"    bin {b}: n={mask.sum():>6}  "
              f"pred={p_te[mask].mean():.3f}  actual={y_te[mask].mean():.3f}")

    # Top features
    try:
        imp = model.feature_importances_
        ranked = sorted(zip(feats, imp), key=lambda t: t[1], reverse=True)
        print("\n  Top 15 features:")
        for name, v in ranked[:15]:
            print(f"    {v:.4f}  {name}")
    except Exception:
        pass

    return model, feats, {"log_loss": float(ll), "brier": float(brier),
                          "baseline_log_loss": float(ll_base)}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PER-PA STRIKEOUT MODEL (true binary labels)")
    print("=" * 70)

    pa, pitcher_feats, batter_feats = load_and_join()
    model, feats, metrics = train(pa, pitcher_feats, batter_feats)

    print("\n── Step 5: Save ──")
    joblib.dump(model, MODEL_DIR / "per_pa_k_model.joblib")
    config = {
        "features": feats,
        "test_year": TEST_YEAR,
        "metrics": metrics,
    }
    with open(MODEL_DIR / "per_pa_k_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ models/per_pa_k_model.joblib")
    print(f"  ✓ models/per_pa_k_config.json")

    print("\n" + "=" * 70)
    print("  DONE. Next step: use this model inside a Poisson-Binomial")
    print("  aggregator (predict P(K) for each projected PA in a start,")
    print("  convolve into distribution, compare log-loss vs 06's BB).")
    print("=" * 70)


if __name__ == "__main__":
    main()
