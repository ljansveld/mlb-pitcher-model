"""
evaluate/feature_importance.py
===================
Quick feature-importance inspector. Loads each saved model and prints
the top N features by importance. Useful for spotting leakage: if a
"current-game" or full-season-aggregate feature is in the top 5, that's
your leak.

USAGE
-----
    python run.py evaluate features                # all models, top 20
    python run.py evaluate features --top 50       # top 50
    python run.py evaluate features k_rate         # just K rate model
    python run.py evaluate features bf hits        # multiple

WHAT TO LOOK FOR
----------------
SAFE features:
  - anything ending in _L3, _L5, _L10  (rolling, properly lagged)
  - anything ending in _szn, _szn_blended  (cumulative, shifted)
  - anything ending in _prev5, _prev10, _prev  (previous season)
  - opp_*, lu_*, ump_*, pf_*, wx_*, fg_stuff_*, fg_loc_*  (process metrics)

SUSPICIOUS features (likely current-game leak):
  - bare names like 'hits_per_pa', 'babip', 'gb_pct' (no suffix)
  - 'k_per_pa', 'k_per_9', 'pitches_per_pa'
  - any rate that could be inverted to recover the target

SUSPICIOUS features (likely same-season-aggregate leak):
  - 'fg_fip', 'fg_siera', 'fg_k_per_9', etc. without _prev suffix
  - 'fg_lob_pct', 'fg_babip_allowed', 'fg_gb_pct' without _prev
"""

import sys
import json
import joblib
import numpy as np
from pathlib import Path

from pitcher_model.paths import MODEL_DIR


# Map model nickname → (joblib path, config path, config-key for features)
MODELS = {
    "k_rate":  ("rate_model.joblib",        "beta_binom_config.json", "rate_features"),
    "bf":      ("bf_model.joblib",          "beta_binom_config.json", "bf_features"),
    "k_pp":    ("per_pa_k_model.joblib",   "per_pa_k_config.json",   "features"),
    "hits":    ("hits_rate_model.joblib",   "hits_config.json",       "rate_features"),
    "walks":   ("walks_rate_model.joblib",  "walks_config.json",      "rate_features"),
    "hit_pp":  ("per_pa_hit_model.joblib",  "per_pa_hit_config.json", "features"),
    "bb_pp":   ("per_pa_bb_model.joblib",   "per_pa_bb_config.json",  "features"),
    "outs":    ("outs_rate_model.joblib",   "outs_config.json",       "rate_features"),
    "outs_pp": ("per_pa_out_model.joblib",  "per_pa_out_config.json", "features"),
}


def get_importances(model):
    """Return importance array for various sklearn-API regressors/classifiers."""
    # XGBoost / LightGBM
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    # sklearn Pipeline → check final step
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                return np.asarray(step.feature_importances_, dtype=float)
            if hasattr(step, "coef_"):
                c = np.asarray(step.coef_, dtype=float)
                return np.abs(c.flatten())
    # Linear models (Ridge etc.)
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=float)
        return np.abs(c.flatten())
    return None


def inspect(nickname, top_n):
    if nickname not in MODELS:
        print(f"  ✗ Unknown model '{nickname}'. Options: {list(MODELS)}")
        return
    model_file, config_file, feat_key = MODELS[nickname]
    mpath = MODEL_DIR / model_file
    cpath = MODEL_DIR / config_file
    if not mpath.exists() or not cpath.exists():
        print(f"  (skipping {nickname} — file not found)")
        return

    model = joblib.load(mpath)
    with open(cpath) as f:
        cfg = json.load(f)
    feats = cfg.get(feat_key)
    if not feats:
        print(f"  ✗ {nickname}: config has no '{feat_key}' key")
        return

    imp = get_importances(model)
    if imp is None:
        print(f"  ✗ {nickname}: model type doesn't expose importances")
        return

    if len(imp) != len(feats):
        print(f"  ⚠ {nickname}: importance length ({len(imp)}) "
              f"!= features length ({len(feats)}). Showing anyway.")

    n = min(len(imp), len(feats), top_n)
    order = np.argsort(-imp)[:n]
    total = imp.sum() if imp.sum() > 0 else 1.0

    print(f"\n══ {nickname.upper()} — top {n} of {len(feats)} features ══")
    print(f"  {'#':<3} {'importance':>10}  {'pct':>5}  feature")
    print(f"  {'-'*3} {'-'*10:>10}  {'-'*5}  {'-'*40}")
    for rank, i in enumerate(order, 1):
        pct = imp[i] / total * 100
        flag = ""
        name = feats[i]
        # Heuristic flags for likely-leak features
        leak_words = [
            "hits_per_pa", "bb_per_pa", "hr_per_pa", "k_per_pa",
            "babip", "lob_pct", "k_minus_bb_pct",
            "h_per_9", "bb_per_9", "hr_per_9",
            "k_per_9", "k_per_100",
            "hr_per_bip", "hr_per_fb",
            "gb_pct", "fb_pct", "ld_pct", "pop_pct", "iffb_pct",
            "avg_exit_velocity", "avg_launch_angle",
            "sweet_spot_pct", "solid_contact_pct",
            "avg_xba_contact", "avg_xwoba_contact",
            "ground_balls", "fly_balls", "line_drives", "popups",
            "soft_hit_pct", "hard_hit_pct", "barrel_pct",
            "outs_per_pa", "outs_recorded", "est_innings",
            "whiff_pct", "csw_pct", "chase_rate",
            "fg_fip", "fg_siera", "fg_xfip", "fg_era", "fg_xera",
            "fg_whip", "fg_lob_pct", "fg_hr_per_fb", "fg_k_minus_bb",
            "fg_k_per_9", "fg_bb_per_9", "fg_hr_per_9",
            "fg_gb_pct", "fg_fb_pct", "fg_ld_pct", "fg_iffb_pct",
            "fg_babip", "fg_soft", "fg_med", "fg_hard",
            "fg_barrel", "fg_hard_hit",
        ]
        safe_suffixes = ("_L1", "_L3", "_L5", "_L10", "_szn", "_blended",
                         "_prev", "_prev5", "_prev10", "_trend",
                         "_ewm", "_std", "_p90", "_p10", "_iqr", "_skew")
        # Prefixes that mean "this is from another entity" (opposing batter,
        # opposing lineup, this batter, etc.) — safe even if the substring
        # matches a leak word.
        safe_prefixes = (
            "opp_b1_", "opp_b2_", "opp_b3_", "opp_b4_", "opp_b5_",
            "opp_b6_", "opp_b7_", "opp_b8_", "opp_b9_",
            "opp_lu_", "opp_batting_", "opp_team_",
            "lu_", "lineup_",
            "batter_", "vs_cat_", "cat_",
            "fg_loc_", "fg_stuff_",  # per-pitch-type Stuff+/Location+
            "fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
            "fg_first_strike", "fg_zone_pct",
            "bf_", "pf_", "wx_", "ump_", "pvt_", "pb_",
            "park_", "venue_", "catcher_",
            "ix_",  # interaction features — should be safe by construction
            # ── Always-lagged delta/trend prefixes ──
            # delta_*_3v10 = stat_L3 - stat_L10; trend_*_3v10 = same.
            # Both inputs are themselves lagged rolling means, so the
            # output is doubly safe regardless of which stat is in the middle.
            "delta_", "trend_",
        )
        if any(lw in name for lw in leak_words) and not name.endswith(safe_suffixes) \
                and not any(name.startswith(p) for p in safe_prefixes):
            # Bare leak-word name with no safe suffix or prefix → flag it
            flag = "  ⚠ POSSIBLE LEAK"
        print(f"  {rank:<3d} {imp[i]:>10.4f}  {pct:>4.1f}%  {name}{flag}")


def main():
    args = sys.argv[1:]
    top_n = 20
    targets = []
    for a in args:
        if a == "--top":
            continue
        if a.isdigit():
            top_n = int(a)
            continue
        if a.startswith("--top="):
            top_n = int(a.split("=")[1])
            continue
        targets.append(a)
    if not targets:
        targets = list(MODELS)

    for t in targets:
        inspect(t, top_n)


if __name__ == "__main__":
    main()
