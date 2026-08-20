"""
train/earned_runs_per_pa.py
======================
Per-plate-appearance proxy model for earned runs allowed.

Unlike strikeouts, hits, walks, and outs — which are clean binary PA
outcomes — earned runs are a GAME-LEVEL statistic driven by sequencing
(runners on base, scoring opportunities). There is no "was_ER = 1" column
at the PA level in Statcast.

APPROACH: Train on a run-scoring proxy
---------------------------------------
We use was_run_scored = 1 if a run was scored on or due to this PA as a
target when available. If not, we construct a strong run-value proxy:

    was_run_proxy = 1 if the PA resulted in a home run, OR
                    was_hit AND batter_is_cleanup_slot (positions 3-5), OR
                    (was_hit AND prior_runners_on heuristic)

Actually, the cleanest approach is to use the 'post_bat_score' and
'bat_score' columns from Statcast (if available) to flag PAs where the
batting team scored. This directly gives us was_run_scored.

If score columns are not present, we fall back to using xwOBA as a
continuous proxy target via a regression-to-binary approach.

The Poisson-Binomial aggregation then gives:
    Expected ER ≈ Σ P(run_scored_or_proxied_i) × adjustment_factor

The adjustment factor maps from raw run-scoring probability back to
expected earned runs (earned ≈ 0.85-0.90 × total runs in practice).

OUTPUTS
-------
- models/per_pa_er_model.joblib
- models/per_pa_er_config.json

RUN ORDER
---------
1. collect/statcast.py  (with PA events patch)
2. features.py
3. collect/statcast.py  (for the BB baseline)
4. train/earned_runs.py            (trains BB baseline)
5. train/earned_runs_per_pa.py    ← this file
6. predict/earned_runs.py            (now shows BB vs PP side by side)
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

from pitcher_model.paths import DATA_DIR, MODEL_DIR, ensure_dirs

ensure_dirs(MODEL_DIR)


PA_EVENTS_PATH        = DATA_DIR / "statcast_pa_events_all.csv"
PITCHER_FEATURES_PATH = DATA_DIR / "pitcher_model_features.csv"

TEST_YEAR    = 2025
RANDOM_STATE = 42

# Earned runs ≈ 87% of total runs historically (unearned ~13%)
EARNED_RUNS_FRACTION = 0.87

# Pitcher-side features most predictive of run-scoring.
# ERA/run-prevention is driven by: avoiding hard contact, limiting walks,
# getting outs efficiently, and pitch quality.
PITCHER_FEATS_KEEP = [
    # Strikeout / walk (directly limits baserunners)
    "k_pct_L3", "k_pct_L5", "k_pct_L10", "k_pct_szn",
    "bb_pct_L3", "bb_pct_L5", "bb_pct_L10", "bb_pct_szn",
    # Contact quality allowed
    "barrel_pct_L3", "barrel_pct_L5", "barrel_pct_szn",
    "hard_hit_pct_L5", "hard_hit_pct_szn",
    "hr_rate_L5", "hr_rate_szn",
    # Hit rate (BABIP-type signal)
    "hits_per_pa_L5", "hits_per_pa_szn",
    # Whiff / command
    "whiff_pct_L5", "whiff_pct_szn",
    "csw_pct_L5", "csw_pct_szn",
    # ERA-adjacent rolling metrics
    "er_per_pa_L3", "er_per_pa_L5", "er_per_pa_L10", "er_per_pa_szn",
    # Pitch mix & velocity
    "fastball_pct_szn", "sl_pct_szn", "cu_pct_szn", "ch_pct_szn",
    "avg_velocity_L5", "avg_velocity_szn",
    # Stuff / command grades
    "fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
    # Game context
    "rest_days", "is_home", "is_night", "is_dome",
    "wx_temperature_f", "wx_effective_wind",
    "pitcher_throws_R",
    "catcher_frm_runs",
    "ump_k_rate_L30",
    "pf_k_rate",
]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD TARGET: was_run_scored (or proxy)
# ══════════════════════════════════════════════════════════════════════════════

def build_run_target(pa):
    """
    Build a binary run-scoring indicator for each PA.

    Priority:
    1. If Statcast bat_score / post_bat_score columns exist, use actual
       run-scoring PAs: was_run_scored = (post_bat_score > bat_score)
    2. Fall back to a run-value proxy based on outcome type:
         HR → 1 (guaranteed at least 1 run)
         Other hits → weighted probability based on base-hit type
         Walks with bases loaded → treat as likely run
         Everything else → 0

    Returns the pa DataFrame with 'was_run_target' column added.
    """
    if "post_bat_score" in pa.columns and "bat_score" in pa.columns:
        # Use actual scoring data
        pa["post_bat_score"] = pd.to_numeric(pa["post_bat_score"], errors="coerce")
        pa["bat_score"]      = pd.to_numeric(pa["bat_score"],      errors="coerce")
        pa["was_run_target"] = (
            (pa["post_bat_score"] > pa["bat_score"]) &
            pa["post_bat_score"].notna() &
            pa["bat_score"].notna()
        ).astype(int)
        rate = pa["was_run_target"].mean()
        print(f"  ✓ Using actual run-scoring indicator  "
              f"(rate: {rate:.4f}, ~{rate*27:.2f} runs/9-inn equivalent)")
        return pa, "actual"

    # Fallback: construct proxy from event type
    print("  ⚠ bat_score/post_bat_score not found — using run-value proxy")
    if "events" in pa.columns:
        # Assign run-scoring probability weight by event type.
        # These are approximate RE24-derived weights scaled to binary:
        #   HR=1.0, triple≈0.5, double≈0.35, single≈0.25, walk≈0.12
        # We threshold at 0.5 to get a binary label, which biases toward
        # HR only — but it's the most defensible without full game state.
        run_weight = {
            "home_run": 1.0,
            "triple":   0.50,
            "double":   0.30,
            # Singles and walks below 0.5 → won't flip binary, so
            # we use a soft approach: sample based on weight
        }
        # Binary: only home runs are guaranteed runs at the PA level
        # without game state. For a per-PA model this is the honest signal.
        pa["was_run_target"] = pa["events"].isin(["home_run"]).astype(int)

        # Softer proxy: also include extra-base hits with downweighted probability
        # We encode as 1 for HR, and use a random sample for 2B/3B based on
        # their approximate run-scoring frequency (~30% for doubles, ~50% for triples)
        rng = np.random.default_rng(42)
        is_double = (pa["events"] == "double")
        is_triple = (pa["events"] == "triple")
        pa.loc[is_double, "was_run_target"] = (rng.random(is_double.sum()) < 0.30).astype(int)
        pa.loc[is_triple, "was_run_target"] = (rng.random(is_triple.sum()) < 0.50).astype(int)

        rate = pa["was_run_target"].mean()
        print(f"  Proxy run target rate: {rate:.4f}  "
              f"(HR + probabilistic 2B/3B)")
    else:
        raise ValueError(
            "PA events CSV needs 'events', 'post_bat_score', or 'bat_score' "
            "to build a run-scoring target."
        )

    return pa, "proxy"


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

    # Ensure baseline binary columns exist
    for col, event_set in [
        ("was_K",   {"strikeout", "strikeout_double_play"}),
        ("was_BB",  {"walk"}),
        ("was_hit", {"single", "double", "triple", "home_run"}),
        ("was_in_play", {"single","double","triple","home_run","field_out",
                         "grounded_into_double_play","force_out","double_play",
                         "sac_fly","sac_bunt","fielders_choice","fielders_choice_out"}),
    ]:
        if col not in pa.columns and "events" in pa.columns:
            pa[col] = pa["events"].isin(event_set).astype(int)

    pa, target_type = build_run_target(pa)
    print(f"  Loaded {len(pa):,} PAs  ({pa['year'].min()}–{pa['year'].max()})")

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
        ("was_K",          "batter_k_rate_std"),
        ("was_BB",         "batter_bb_rate_std"),
        ("was_hit",        "batter_hit_rate_std"),
        ("was_in_play",    "batter_bip_rate_std"),
        ("was_run_target", "batter_run_rate_std"),
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
        ("was_K",          "batter_k_rate_L100"),
        ("was_BB",         "batter_bb_rate_L100"),
        ("was_hit",        "batter_hit_rate_L100"),
        ("was_run_target", "batter_run_rate_L100"),
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
              batter_k_rate_prior   =("was_K",          "mean"),
              batter_bb_rate_prior  =("was_BB",         "mean"),
              batter_hit_rate_prior =("was_hit",        "mean"),
              batter_run_rate_prior =("was_run_target", "mean"),
              batter_bip_rate_prior =("was_in_play",    "mean"),
              batter_pa_prior       =("was_K",          "size"),
          )
          .reset_index()
    )
    prior["year"] = prior["year"].astype(int) + 1
    pa = pa.merge(prior, on=["batter", "year"], how="left")

    league_k   = pa["was_K"].mean()
    league_bb  = pa["was_BB"].mean()
    league_hit = pa["was_hit"].mean()
    league_run = pa["was_run_target"].mean()
    league_bip = pa["was_in_play"].mean() if "was_in_play" in pa.columns else 0.68

    fill_map = {
        "batter_k_rate_std":    league_k,   "batter_k_rate_L100":   league_k,
        "batter_k_rate_prior":  league_k,
        "batter_bb_rate_std":   league_bb,  "batter_bb_rate_L100":  league_bb,
        "batter_bb_rate_prior": league_bb,
        "batter_hit_rate_std":  league_hit, "batter_hit_rate_L100": league_hit,
        "batter_hit_rate_prior":league_hit,
        "batter_run_rate_std":  league_run, "batter_run_rate_L100": league_run,
        "batter_run_rate_prior":league_run,
        "batter_bip_rate_std":  league_bip, "batter_bip_rate_prior":league_bip,
        "batter_pa_prior": 0.0,
    }
    for col, val in fill_map.items():
        if col in pa.columns:
            pa[col] = pa[col].fillna(val)

    batter_feats = list(fill_map.keys())
    print(f"  Built {len(batter_feats)} batter features (leakage-free)")
    print(f"  Final frame: {len(pa):,} rows, {len(pa.columns)} cols")
    return pa, available_pf, batter_feats, target_type


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(pa, pitcher_feats, batter_feats, target_type):
    print(f"\n{'=' * 70}")
    print(f"  PER-PA MODEL: Earned Runs Proxy  (target = was_run_target)")
    print(f"  Target type: {target_type}")
    print(f"{'=' * 70}")

    # Interaction features
    if "k_pct_szn" in pa.columns and "batter_k_rate_prior" in pa.columns:
        pa["ix_pitcher_k_x_batter_k"] = pa["k_pct_szn"] * pa["batter_k_rate_prior"]
    # Run-scoring: pitcher ER rate × batter run-scoring tendency
    er_col = next((c for c in ["er_per_pa_szn", "er_per_pa_L5"] if c in pa.columns), None)
    if er_col and "batter_run_rate_prior" in pa.columns:
        pa["ix_pitcher_er_x_batter_run"] = pa[er_col] * pa["batter_run_rate_prior"]
    if er_col and "batter_run_rate_L100" in pa.columns:
        pa["ix_pitcher_er_x_batter_run_recent"] = pa[er_col] * pa["batter_run_rate_L100"]
    # Hard contact: pitcher barrel rate × batter hit tendency (barrel often = run scored)
    if "barrel_pct_szn" in pa.columns and "batter_hit_rate_prior" in pa.columns:
        pa["ix_barrel_x_hit"] = pa["barrel_pct_szn"] * pa["batter_hit_rate_prior"]
    # Walk × batter walk tendency (free baserunners drive runs)
    if "bb_pct_szn" in pa.columns and "batter_bb_rate_prior" in pa.columns:
        pa["ix_pitcher_bb_x_batter_bb"] = pa["bb_pct_szn"] * pa["batter_bb_rate_prior"]

    if "p_throws" in pa.columns and "stand" in pa.columns:
        pa["same_handed"] = (pa["p_throws"] == pa["stand"]).astype(int)

    interaction_feats = [
        "ix_pitcher_k_x_batter_k",
        "ix_pitcher_er_x_batter_run",
        "ix_pitcher_er_x_batter_run_recent",
        "ix_barrel_x_hit",
        "ix_pitcher_bb_x_batter_bb",
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
    tr_rate = train_df["was_run_target"].mean()
    te_rate = test_df["was_run_target"].mean()
    print(f"  Train run-proxy rate: {tr_rate:.4f}  Test: {te_rate:.4f}")
    print(f"  → Σ p_i per 27 PAs ≈ {tr_rate*27:.2f} raw runs  "
          f"× {EARNED_RUNS_FRACTION:.2f} ≈ {tr_rate*27*EARNED_RUNS_FRACTION:.2f} ER/game")

    X_tr = train_df[feats].values
    y_tr = train_df["was_run_target"].values
    X_te = test_df[feats].values
    y_te = test_df["was_run_target"].values

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

    print(f"\n── Test Metrics ──")
    print(f"  Log-loss: {ll:.4f}  (baseline: {ll_base:.4f}  Δ={ll-ll_base:+.4f})")
    print(f"  Brier:    {brier:.4f}")
    if ll >= ll_base:
        print("  ⚠ Model does not beat the constant-mean baseline.")

    print(f"\n  Calibration (pred vs actual, by decile):")
    bins = pd.qcut(p_te, q=10, labels=False, duplicates="drop")
    for b in sorted(np.unique(bins)):
        mask = bins == b
        print(f"    bin {b}: n={mask.sum():>6}  "
              f"pred={p_te[mask].mean():.4f}  actual={y_te[mask].mean():.4f}")

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
        "target_type": target_type,
        "league_run_proxy_rate": float(te_rate),
        "earned_runs_fraction": EARNED_RUNS_FRACTION,
        "note": (
            "Poisson-Binomial sum of P(run_proxy) gives expected raw runs. "
            f"Multiply by {EARNED_RUNS_FRACTION} to convert to expected ER."
        ),
    }

    joblib.dump(model, MODEL_DIR / "per_pa_er_model.joblib")
    config = {
        "target": "was_run_target",
        "features": feats,
        "test_year": TEST_YEAR,
        "metrics": metrics,
        "league_rate": float(te_rate),
        "earned_runs_fraction": EARNED_RUNS_FRACTION,
    }
    with open(MODEL_DIR / "per_pa_er_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✓ Saved models/per_pa_er_model.joblib")
    print(f"  ✓ Saved models/per_pa_er_config.json")
    return model, feats, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PER-PA EARNED RUNS MODEL (run-scoring proxy)")
    print("=" * 70)
    pa, pitcher_feats, batter_feats, target_type = load_and_join()
    train(pa.copy(), pitcher_feats, batter_feats, target_type)
    print("\n" + "=" * 70)
    print("  DONE. Run predict/earned_runs.py — it will show BB vs Per-PA side by side.")
    print(f"  NOTE: Per-PA ER uses a run-scoring proxy. The Poisson-Binomial")
    print(f"  sum is scaled by {EARNED_RUNS_FRACTION} to convert raw runs to ER.")
    print("=" * 70)


if __name__ == "__main__":
    main()
