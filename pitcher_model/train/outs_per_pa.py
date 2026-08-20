"""
train/outs_per_pa.py
========================
Per-plate-appearance binary model for outs recorded.

Trains a single XGBoost binary classifier:
    was_out = 1 if the PA ended in an out (any type), 0 otherwise

A PA ends in an out if it is NOT a hit, walk, HBP, strikeout, or error
that awards a base. In practice: was_out = was_in_play AND NOT was_hit,
plus strikeouts (which are also outs).

Actually, from the pitcher's perspective:
    was_out = 1  if events IN {field_out, grounded_into_double_play,
                                force_out, double_play, triple_play,
                                sac_fly, sac_bunt, fielders_choice,
                                fielders_choice_out, strikeout,
                                strikeout_double_play, caught_stealing_*,
                                pickoff_*, etc.}
    was_out = 0  otherwise (hit, walk, HBP, error reaching base)

The exact reconstruction is done from the events column. Note that
double plays record 2 outs for 1 PA — we count them as was_out=1 and
handle the double-count separately (most per-PA models do the same;
the Poisson-Binomial will slightly underestimate outs in GDP-heavy
situations, but this is a second-order effect).

INPUTS
------
- data/statcast_pa_events_all.csv
- data/pitcher_model_features.csv

OUTPUTS
-------
- models/per_pa_out_model.joblib
- models/per_pa_out_config.json

RUN ORDER
---------
1. collect/statcast.py  (with PA events patch)
2. features.py
3. train/outs.py             (trains BB baseline)
4. train/outs_per_pa.py     ← this file
5. predict/outs.py             (now shows BB vs PP side by side)
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

# PA-ending events that result in at least one out recorded for the pitcher
OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "force_out",
    "double_play", "triple_play", "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "strikeout", "strikeout_double_play",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "other_out",
}

# Pitcher-side features relevant to out-recording ability.
# Outs are driven by: K ability (direct outs), soft-contact induction,
# ground-ball tendency (more double plays), and control (limiting free passes).
# Mirrors the BLENDED-feature treatment used in train/strikeouts_per_pa.py and
# train/hits_walks_per_pa.py (April 2026 recency-bias fix), and pulls in
# the new hits-pipeline features for hit-suppression signal.
PITCHER_FEATS_KEEP = [
    # ── Season-to-date rates — BLENDED versions (prefer over raw _szn) ──
    "k_pct_szn_blended", "bb_pct_szn_blended", "whiff_pct_szn_blended",
    "csw_pct_szn_blended", "barrel_pct_szn_blended",
    "hard_hit_pct_szn_blended",
    # Raw season rates (training-time alternates; pruner may keep blended)
    "k_pct_szn", "bb_pct_szn", "whiff_pct_szn", "csw_pct_szn",
    "hits_per_pa_szn", "barrel_pct_szn", "hard_hit_pct_szn",
    # L5 windows — both raw and blended
    "k_pct_L5", "k_pct_L5_blended",
    "bb_pct_L5", "whiff_pct_L5", "whiff_pct_L5_blended",
    "csw_pct_L5", "hits_per_pa_L5",
    "barrel_pct_L5", "barrel_pct_L5_blended",
    "hard_hit_pct_L5", "hard_hit_pct_L5_blended",
    # L10
    "k_pct_L10", "bb_pct_L10", "hits_per_pa_L10",
    # ── Outs-specific rolling rates ──
    "outs_per_pa_L3", "outs_per_pa_L5", "outs_per_pa_L10", "outs_per_pa_szn",
    # Hit-quality allowed (negative signal for outs)
    # NOTE: the previous version had "hr_rate_L5" but that column is not
    # produced by features.py. The actual rolling HR rates are
    # hr_per_pa_L5 / hr_per_9_L5 / hr_per_bip_L5 (built by 02's roll_cols loop).
    "hr_per_pa_L5", "hr_per_9_L5",
    # ── Prior-year anchors (the BIG win — these were computed but never used) ──
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
    # FanGraphs quality grades
    "fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
    # Game context
    "rest_days", "is_home", "is_night_game", "wx_is_dome",
    "wx_temperature_f", "wx_effective_wind",
    "pitcher_throws_R",
    # Correct column names from features.py
    "catcher_frm",
    "ump_k_pct",
    "pf_Basic",
    # ══════════════════════════════════════════════════════════════════
    # HITS / WALKS PIPELINE ADDITIONS — direct BIP-out signal
    # ══════════════════════════════════════════════════════════════════
    # A pitcher's BIP-out rate (the second half of "outs") depends on
    # batted-ball mix, contact quality, and BABIP. These features tell
    # the per-PA model which pitchers convert BIPs to outs better.
    #
    # Blended (preferred when available)
    "hits_per_pa_szn_blended", "bb_per_pa_szn_blended",
    "hits_per_pa_L5_blended",  "bb_per_pa_L5_blended",
    "babip_szn_blended", "babip_L5_blended",
    "lob_pct_szn_blended",
    "gb_pct_szn_blended", "fb_pct_szn_blended", "ld_pct_szn_blended",
    # GB% is especially valuable: a GB pitcher converts more outs/BIP via
    # double plays and limits HR damage.
    "avg_exit_velocity_szn_blended", "avg_xwoba_contact_szn_blended",
    "sweet_spot_pct_szn_blended",
    # Raw season versions
    "bb_per_pa_szn", "hr_per_pa_szn", "babip_szn", "lob_pct_szn",
    "gb_pct_szn", "fb_pct_szn", "ld_pct_szn",
    "avg_exit_velocity_szn", "avg_xwoba_contact_szn",
    # Prior-year anchors for H/W (early-season stabilizers)
    "hits_per_pa_prev10", "bb_per_pa_prev10", "hr_per_pa_prev10",
    "babip_prev10", "lob_pct_prev10",
    "gb_pct_prev10", "fb_pct_prev10", "ld_pct_prev10",
    "avg_exit_velocity_prev10", "avg_xwoba_contact_prev10",
    # FanGraphs hits-side talent metrics — SIERA is built around K-BB% + GB%
    # and is the single most predictive metric for true outs-per-PA talent.
    "fg_siera", "fg_xfip", "fg_fip",
    "fg_lob_pct", "fg_hr_per_fb", "fg_k_minus_bb_pct",
    "fg_gb_pct", "fg_fb_pct", "fg_ld_pct",
    "fg_babip_allowed", "fg_barrel_pct_allowed", "fg_hard_hit_pct_allowed",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & JOIN
# ══════════════════════════════════════════════════════════════════════════════

def load_and_join():
    print("── Step 1: Load PA events ──")
    if not PA_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"{PA_EVENTS_PATH} not found. Run `python run.py collect statcast` "
            "to produce it."
        )
    pa = pd.read_csv(PA_EVENTS_PATH)
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce")
    pa["year"] = pa["game_date"].dt.year

    # Build was_out from the events column
    if "was_out" not in pa.columns:
        if "events" in pa.columns:
            pa["was_out"] = pa["events"].isin(OUT_EVENTS).astype(int)
            print(f"  Built was_out from events column")
        else:
            raise ValueError("PA events CSV needs either 'was_out' or 'events' column.")
    else:
        pa["was_out"] = pa["was_out"].astype(int)

    # Ensure other binary columns exist (used for batter features)
    for col, event_set in [
        ("was_K",   {"strikeout", "strikeout_double_play"}),
        ("was_BB",  {"walk"}),
        ("was_hit", {"single", "double", "triple", "home_run"}),
        ("was_in_play", {"single","double","triple","home_run","field_out",
                         "grounded_into_double_play","force_out","double_play",
                         "triple_play","sac_fly","sac_bunt","fielders_choice",
                         "fielders_choice_out"}),
    ]:
        if col not in pa.columns and "events" in pa.columns:
            pa[col] = pa["events"].isin(event_set).astype(int)

    print(f"  Loaded {len(pa):,} PAs  ({pa['year'].min()}–{pa['year'].max()})")
    print(f"  League out rate: {pa['was_out'].mean():.3f}")

    print("\n── Step 2: Join pitcher-game features ──")
    pf = pd.read_csv(PITCHER_FEATURES_PATH)
    pf["game_date"] = pd.to_datetime(pf["game_date"], errors="coerce")
    available_pf = [c for c in PITCHER_FEATS_KEEP if c in pf.columns]
    missing_pf   = [c for c in PITCHER_FEATS_KEEP if c not in pf.columns]
    if missing_pf:
        print(f"  (skipping {len(missing_pf)} missing pitcher cols: "
              f"{missing_pf[:5]}{'...' if len(missing_pf) > 5 else ''})")
    pf_slim = pf[["game_pk", "pitcher"] + available_pf].drop_duplicates(
        subset=["game_pk", "pitcher"]
    )
    pa = pa.merge(pf_slim, on=["game_pk", "pitcher"], how="inner")
    print(f"  After pitcher join: {len(pa):,} PAs ({len(available_pf)} pitcher features)")

    print("\n── Step 3: Build batter features (leakage-free) ──")
    pa = pa.sort_values(
        ["batter", "year", "game_date", "at_bat_number"]
    ).reset_index(drop=True)

    g_sy = pa.groupby(["batter", "year"], sort=False)
    for src, dst in [
        ("was_K",      "batter_k_rate_std"),
        ("was_BB",     "batter_bb_rate_std"),
        ("was_hit",    "batter_hit_rate_std"),
        ("was_out",    "batter_out_rate_std"),
        ("was_in_play","batter_bip_rate_std"),
    ]:
        if src in pa.columns:
            shifted = g_sy[src].shift(1)
            pa[dst] = (
                shifted.groupby([pa["batter"], pa["year"]])
                       .expanding().mean()
                       .reset_index(level=[0, 1], drop=True)
            )

    g_all = pa.groupby(["batter"], sort=False)
    for src, dst in [
        ("was_K",   "batter_k_rate_L100"),
        ("was_BB",  "batter_bb_rate_L100"),
        ("was_hit", "batter_hit_rate_L100"),
        ("was_out", "batter_out_rate_L100"),
    ]:
        if src in pa.columns:
            shifted = g_all[src].shift(1)
            pa[dst] = (
                shifted.groupby(pa["batter"])
                       .rolling(100, min_periods=20).mean()
                       .reset_index(level=0, drop=True)
            )

    prior = (
        pa.groupby(["batter", "year"])
          .agg(
              batter_k_rate_prior   =("was_K",   "mean"),
              batter_bb_rate_prior  =("was_BB",  "mean"),
              batter_hit_rate_prior =("was_hit", "mean"),
              batter_out_rate_prior =("was_out", "mean"),
              batter_bip_rate_prior =("was_in_play","mean"),
              batter_pa_prior       =("was_K",   "size"),
          )
          .reset_index()
    )
    prior["year"] = prior["year"].astype(int) + 1
    pa = pa.merge(prior, on=["batter", "year"], how="left")

    league_k   = pa["was_K"].mean()
    league_bb  = pa["was_BB"].mean()
    league_hit = pa["was_hit"].mean()
    league_out = pa["was_out"].mean()
    league_bip = pa["was_in_play"].mean() if "was_in_play" in pa.columns else 0.68

    fill_map = {
        "batter_k_rate_std":    league_k,   "batter_k_rate_L100":   league_k,
        "batter_k_rate_prior":  league_k,
        "batter_bb_rate_std":   league_bb,  "batter_bb_rate_L100":  league_bb,
        "batter_bb_rate_prior": league_bb,
        "batter_hit_rate_std":  league_hit, "batter_hit_rate_L100": league_hit,
        "batter_hit_rate_prior":league_hit,
        "batter_out_rate_std":  league_out, "batter_out_rate_L100": league_out,
        "batter_out_rate_prior":league_out,
        "batter_bip_rate_std":  league_bip, "batter_bip_rate_prior":league_bip,
        "batter_pa_prior": 0.0,
    }
    for col, val in fill_map.items():
        if col in pa.columns:
            pa[col] = pa[col].fillna(val)

    batter_feats = list(fill_map.keys())
    print(f"  Built {len(batter_feats)} batter features (leakage-free)")
    print(f"  Final frame: {len(pa):,} rows, {len(pa.columns)} cols")
    return pa, available_pf, batter_feats


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(pa, pitcher_feats, batter_feats):
    print(f"\n{'=' * 70}")
    print(f"  PER-PA MODEL: Outs Recorded  (target = was_out)")
    print(f"{'=' * 70}")

    # Interaction features
    # NOTE: Prefer BLENDED versions of pitcher rate stats over raw _szn here.
    # This is the April 2026 recency-bias fix mirrored from 10b — early in
    # the season the raw _szn rate is wildly unstable; the empirical-Bayes
    # blend shrinks it to a prior, making the interaction more meaningful
    # during the first ~10 starts of each season.
    k_src = "k_pct_szn_blended" if "k_pct_szn_blended" in pa.columns else "k_pct_szn"
    out_src = "outs_per_pa_szn"  # no blended version of outs (it's not in rate_configs)
    barrel_src = ("barrel_pct_szn_blended"
                  if "barrel_pct_szn_blended" in pa.columns else "barrel_pct_szn")

    if k_src in pa.columns and "batter_k_rate_prior" in pa.columns:
        pa["ix_pitcher_k_x_batter_k"] = pa[k_src] * pa["batter_k_rate_prior"]
    # Outs: pitcher out-rate × batter out-rate (the "easy out" matchup signal)
    if out_src in pa.columns and "batter_out_rate_prior" in pa.columns:
        pa["ix_pitcher_out_x_batter_out"] = (
            pa[out_src] * pa["batter_out_rate_prior"]
        )
    if out_src in pa.columns and "batter_out_rate_L100" in pa.columns:
        pa["ix_pitcher_out_x_batter_out_recent"] = (
            pa[out_src] * pa["batter_out_rate_L100"]
        )
    # Harder contact proxy: pitcher barrel × batter BIP (more BIP = more out chances, but also hits)
    if barrel_src in pa.columns and "batter_bip_rate_std" in pa.columns:
        pa["ix_barrel_x_bip"] = pa[barrel_src] * pa["batter_bip_rate_std"]

    # ── H/W pipeline interactions ──
    # Ground-ball pitcher × batter contact propensity → DP-prone matchups
    if "gb_pct_szn_blended" in pa.columns and "batter_bip_rate_prior" in pa.columns:
        pa["ix_gb_x_bip"] = (pa["gb_pct_szn_blended"]
                              * pa["batter_bip_rate_prior"])
    # Hit-suppression × batter hit rate → "tough out" matchup
    if "hits_per_pa_szn_blended" in pa.columns and "batter_hit_rate_prior" in pa.columns:
        pa["ix_hits_x_batter_hits"] = (pa["hits_per_pa_szn_blended"]
                                        * pa["batter_hit_rate_prior"])

    if "p_throws" in pa.columns and "stand" in pa.columns:
        pa["same_handed"] = (pa["p_throws"] == pa["stand"]).astype(int)

    interaction_feats = [
        "ix_pitcher_k_x_batter_k",
        "ix_pitcher_out_x_batter_out",
        "ix_pitcher_out_x_batter_out_recent",
        "ix_barrel_x_bip",
        "ix_gb_x_bip",
        "ix_hits_x_batter_hits",
    ]

    base_feats = list(dict.fromkeys(
        pitcher_feats + batter_feats + interaction_feats + ["same_handed", "inning"]
    ))
    feats = [c for c in base_feats if c in pa.columns]
    for c in feats:
        pa[c] = pd.to_numeric(pa[c], errors="coerce").fillna(0.0)

    train_df = pa[pa["year"] < TEST_YEAR].copy()
    test_df  = pa[pa["year"] >= TEST_YEAR].copy()

    print(f"\n── Train/Test Split ──")
    print(f"  Train PAs: {len(train_df):,}  Test PAs: {len(test_df):,}")
    print(f"  Features:  {len(feats)}")
    tr_rate = train_df["was_out"].mean()
    te_rate = test_df["was_out"].mean()
    print(f"  Train out rate: {tr_rate:.3f}  Test: {te_rate:.3f}")

    X_tr = train_df[feats].values
    y_tr = train_df["was_out"].values
    X_te = test_df[feats].values
    y_te = test_df["was_out"].values

    model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=50,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    p_te     = model.predict_proba(X_te)[:, 1]
    ll       = log_loss(y_te, p_te)
    brier    = brier_score_loss(y_te, p_te)
    baseline = np.full_like(y_te, y_tr.mean(), dtype=float)
    ll_base  = log_loss(y_te, baseline)
    brier_base = brier_score_loss(y_te, baseline)

    print(f"\n── Test Metrics ──")
    print(f"  Log-loss: {ll:.4f}  (baseline: {ll_base:.4f}  Δ={ll-ll_base:+.4f})")
    print(f"  Brier:    {brier:.4f}  (baseline: {brier_base:.4f}  Δ={brier-brier_base:+.4f})")
    if ll >= ll_base:
        print("  ⚠ Model does not beat the constant-mean baseline.")

    print(f"\n  Calibration (pred vs actual, by decile):")
    bins = pd.qcut(p_te, q=10, labels=False, duplicates="drop")
    for b in sorted(np.unique(bins)):
        mask = bins == b
        print(f"    bin {b}: n={mask.sum():>6}  "
              f"pred={p_te[mask].mean():.3f}  actual={y_te[mask].mean():.3f}")

    try:
        imp    = model.feature_importances_
        ranked = sorted(zip(feats, imp), key=lambda t: t[1], reverse=True)
        print(f"\n  Top 15 features:")
        for name, v in ranked[:15]:
            print(f"    {v:.4f}  {name}")
    except Exception:
        pass

    metrics = {
        "log_loss": float(ll), "brier": float(brier),
        "baseline_log_loss": float(ll_base),
        "league_out_rate": float(te_rate),
    }
    joblib.dump(model, MODEL_DIR / "per_pa_out_model.joblib")
    config = {
        "target": "was_out",
        "features": feats,
        "test_year": TEST_YEAR,
        "metrics": metrics,
        "league_rate": float(te_rate),
    }
    with open(MODEL_DIR / "per_pa_out_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✓ Saved models/per_pa_out_model.joblib")
    print(f"  ✓ Saved models/per_pa_out_config.json")
    return model, feats, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PER-PA OUTS RECORDED MODEL (true binary labels)")
    print("=" * 70)
    pa, pitcher_feats, batter_feats = load_and_join()
    train(pa.copy(), pitcher_feats, batter_feats)
    print("\n" + "=" * 70)
    print("  DONE. Run predict/outs.py — it will show BB vs Per-PA side by side.")
    print("=" * 70)


if __name__ == "__main__":
    main()
