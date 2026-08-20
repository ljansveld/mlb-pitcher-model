"""
predict/outs.py
================
Daily predictions for pitcher outs recorded — two models side by side.

BB  = Beta-Binomial (from train/outs.py)
PP  = Per-PA XGBoost (from train/outs_per_pa.py)
      Scores P(out) for each projected PA using the actual posted lineup
      with real per-batter features, then convolves via Poisson-Binomial.

Both models run independently and are shown side by side.

USAGE:
    python run.py predict outs               # Today
    python run.py predict outs 2026-04-15    # Specific date

REQUIRES:
    - models/outs_rate_model.joblib, outs_config.json
    - models/bf_model.joblib, beta_binom_config.json
    - models/per_pa_out_model.joblib, per_pa_out_config.json  (optional)
    - data/pitcher_model_features.csv
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.special import betaln, gammaln
from scipy import stats as sp_stats
import joblib
import requests

from pitcher_model.paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR)


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ── Smart feature defaults ──────────────────────────────────────────────────
_PLUS_FEATURES = {"fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
                  "fg_stuff_fa", "fg_stuff_sl", "fg_stuff_ch", "fg_stuff_cu",
                  "fg_stuff_si", "fg_stuff_fc", "fg_stuff_fs",
                  "fg_loc_fa", "fg_loc_sl", "fg_loc_ch", "fg_loc_cu",
                  "fg_loc_si", "fg_loc_fc", "fg_loc_fs"}
_FEATURE_DEFAULTS = {
    "fg_swstr_pct": 0.11, "fg_o_swing_pct": 0.30, "fg_z_swing_pct": 0.65,
    "fg_contact_pct": 0.78, "fg_o_contact_pct": 0.65, "fg_z_contact_pct": 0.87,
    "fg_zone_pct_szn": 0.45, "fg_first_strike_pct": 0.60,
    "fg_tto_pct": 0.33, "fg_pitcher_frm": 0.0, "catcher_frm": 0.0,
    "opp_lu_swstr_pct": 0.11, "opp_lu_o_swing_pct": 0.30,
    "opp_lu_z_contact_pct": 0.87, "opp_lu_contact_pct": 0.78,
    "opp_lu_tto_pct": 0.33, "opp_lu_csw_pct": 0.30,
    "opp_lu_fg_k_pct": 0.22, "opp_lu_barrel_pct": 0.07,
    "opp_lu_hard_hit_pct": 0.35, "opp_lu_k_rate_std": 0.06,
    "opp_lu_tto_pct_std": 0.05, "opp_lu_max_k_rate": 0.30,
    "opp_lu_min_k_rate": 0.12, "velo_delta_L3_vs_szn": 0.0,
    "platoon_whiff_diff_L5": 0.0,
    "pitcher_tto_L5": 0.33, "pitcher_tto_L10": 0.33, "pitcher_tto_szn": 0.33,
    "ix_stuff_x_lu_k": 0.22, "ix_swstr_x_contact": 0.024,
    "ix_stuff_x_contact": 0.22, "ix_stuff_x_chase": 0.30,
    "ix_pitcher_lu_swstr": 0.012, "ix_tto_matchup": 0.11,
    "ix_pitcher_csw_x_ump_k": 0.066, "ix_pitcher_edge_x_ump_zone": 0.12,
    "ix_pitcher_k_x_ump_k": 0.048, "ix_pitcher_bb_x_ump_bb": 0.006,
    "ix_catcher_frm_x_csw": 0.0, "ix_catcher_frm_x_strike": 0.0,
    "ix_catcher_frm_x_k": 0.0,
    "pitchcount_L1": 90.0, "pitchcount_L2_total": 180.0,
    "heavy_prev_start": 0.2, "pitches_per_out_L1": 5.5, "pitches_per_out_L3": 5.5,
    "opp_lu_k_rate_median": 0.22, "opp_lu_top3_k_mean": 0.28,
    "opp_lu_bot3_k_mean": 0.16, "opp_lu_top3_bot3_gap": 0.12,
    "pitch_matchup_score": 0.05, "pitch_k_matchup_score": 0.04,
    "pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
    "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0,
    # ── Blended / empirical-Bayes season rates (April 2026 recency-bias fix) ──
    # Mirrored from predict/hits_walks.py so the outs model gets the same
    # league-average fallbacks when blended columns are missing or null.
    "k_pct_szn_blended": 0.22,
    "whiff_pct_szn_blended": 0.25,
    "csw_pct_szn_blended": 0.30,
    "bb_pct_szn_blended": 0.085,
    "barrel_pct_szn_blended": 0.07,
    "hard_hit_pct_szn_blended": 0.35,
    "chase_rate_szn_blended": 0.30,
    "k_pct_L5_blended": 0.22,
    "whiff_pct_L5_blended": 0.25,
    "csw_pct_L5_blended": 0.30,
    "barrel_pct_L5_blended": 0.07,
    "hard_hit_pct_L5_blended": 0.35,
    # Prior-season carryover anchors
    "k_pct_prev5": 0.22, "k_pct_prev10": 0.22,
    "bb_pct_prev5": 0.085, "bb_pct_prev10": 0.085,
    "whiff_pct_prev5": 0.25, "whiff_pct_prev10": 0.25,
    "csw_pct_prev5": 0.30, "csw_pct_prev10": 0.30,
    "barrel_pct_prev5": 0.07, "barrel_pct_prev10": 0.07,
    "hard_hit_pct_prev5": 0.35, "hard_hit_pct_prev10": 0.35,
    "chase_rate_prev5": 0.30, "chase_rate_prev10": 0.30,
    "avg_velocity_prev5": 93.0, "avg_velocity_prev10": 93.0,
    "avg_spin_rate_prev5": 2300.0, "avg_spin_rate_prev10": 2300.0,
    "strikeouts_prev5": 5.5, "strikeouts_prev10": 5.5,
    "k_per_100_pitches_prev5": 6.0, "k_per_100_pitches_prev10": 6.0,
    "k_per_9_prev5": 8.5, "k_per_9_prev10": 8.5,
    # Reliability signals
    "prior_starts_this_season": 0,
    "prior_starts_available": 0,
    "n_starts_in_L5": 0,
    "n_starts_in_L10": 0,
    # ══════════════════════════════════════════════════════════════════
    # HITS / WALKS PIPELINE DEFAULTS (mirrored from predict/hits_walks.py)
    # ══════════════════════════════════════════════════════════════════
    # Outs depends on both K rate AND BIP-out rate, so the daily outs
    # script needs every hits-side default that the H/W daily uses.
    "hits_per_pa":        0.235, "hits_per_pa_L3":  0.235, "hits_per_pa_L5":  0.235,
    "hits_per_pa_L10":    0.235, "hits_per_pa_szn": 0.235,
    "hits_per_pa_szn_blended": 0.235, "hits_per_pa_L5_blended": 0.235,
    "hits_per_pa_prev5":  0.235, "hits_per_pa_prev10": 0.235,
    "bb_per_pa":          0.085, "bb_per_pa_L3":    0.085, "bb_per_pa_L5":    0.085,
    "bb_per_pa_L10":      0.085, "bb_per_pa_szn":   0.085,
    "bb_per_pa_szn_blended": 0.085, "bb_per_pa_L5_blended": 0.085,
    "bb_per_pa_prev5":    0.085, "bb_per_pa_prev10": 0.085,
    "hr_per_pa":          0.030, "hr_per_pa_L3":    0.030, "hr_per_pa_L5":    0.030,
    "hr_per_pa_L10":      0.030, "hr_per_pa_szn":   0.030,
    "hr_per_pa_szn_blended": 0.030, "hr_per_pa_L5_blended": 0.030,
    "hr_per_pa_prev5":    0.030, "hr_per_pa_prev10": 0.030,
    "h_per_9":            8.5,   "h_per_9_L5":      8.5,   "h_per_9_szn":   8.5,
    "h_per_9_prev5":      8.5,   "h_per_9_prev10":  8.5,
    "bb_per_9":           3.1,   "bb_per_9_L5":     3.1,   "bb_per_9_szn":  3.1,
    "bb_per_9_prev5":     3.1,   "bb_per_9_prev10": 3.1,
    "hr_per_9":           1.15,  "hr_per_9_L5":     1.15,  "hr_per_9_szn":  1.15,
    "hr_per_9_prev5":     1.15,  "hr_per_9_prev10": 1.15,
    "hr_per_bip":         0.045, "hr_per_bip_L5":   0.045, "hr_per_bip_szn": 0.045,
    "hr_per_fb":          0.115, "hr_per_fb_L5":    0.115, "hr_per_fb_szn":  0.115,
    "k_minus_bb_pct":     0.135, "k_minus_bb_pct_L5":   0.135,
    "k_minus_bb_pct_szn": 0.135, "k_minus_bb_pct_prev5": 0.135,
    "k_minus_bb_pct_prev10": 0.135,
    # BABIP / LOB% — heavy luck stats
    "babip":              0.295, "babip_L3":         0.295, "babip_L5":      0.295,
    "babip_L10":          0.295, "babip_szn":        0.295,
    "babip_szn_blended":  0.295, "babip_L5_blended": 0.295,
    "babip_prev5":        0.295, "babip_prev10":     0.295,
    "lob_pct":            0.72,  "lob_pct_L3":       0.72,  "lob_pct_L5":    0.72,
    "lob_pct_L10":        0.72,  "lob_pct_szn":      0.72,
    "lob_pct_szn_blended": 0.72, "lob_pct_L5_blended": 0.72,
    "lob_pct_prev5":      0.72,  "lob_pct_prev10":   0.72,
    # Batted-ball mix (huge for outs — GB pitchers convert more outs)
    "gb_pct":             0.43,  "gb_pct_L3":        0.43,  "gb_pct_L5":     0.43,
    "gb_pct_L10":         0.43,  "gb_pct_szn":       0.43,
    "gb_pct_szn_blended": 0.43,  "gb_pct_L5_blended": 0.43,
    "gb_pct_prev5":       0.43,  "gb_pct_prev10":    0.43,
    "fb_pct":             0.36,  "fb_pct_L3":        0.36,  "fb_pct_L5":     0.36,
    "fb_pct_L10":         0.36,  "fb_pct_szn":       0.36,
    "fb_pct_szn_blended": 0.36,  "fb_pct_L5_blended": 0.36,
    "fb_pct_prev5":       0.36,  "fb_pct_prev10":    0.36,
    "ld_pct":             0.21,  "ld_pct_L3":        0.21,  "ld_pct_L5":     0.21,
    "ld_pct_L10":         0.21,  "ld_pct_szn":       0.21,
    "ld_pct_szn_blended": 0.21,  "ld_pct_L5_blended": 0.21,
    "ld_pct_prev5":       0.21,  "ld_pct_prev10":    0.21,
    "pop_pct":            0.10,  "pop_pct_L5":       0.10,  "pop_pct_szn":   0.10,
    "pop_pct_szn_blended": 0.10, "pop_pct_prev5":    0.10,  "pop_pct_prev10": 0.10,
    "iffb_pct":           0.10,  "iffb_pct_L5":      0.10,  "iffb_pct_szn":  0.10,
    "iffb_pct_szn_blended": 0.10,
    "soft_hit_pct":       0.17,  "soft_hit_pct_L5":  0.17,  "soft_hit_pct_szn": 0.17,
    "soft_hit_pct_szn_blended": 0.17,
    "soft_hit_pct_prev5": 0.17,  "soft_hit_pct_prev10": 0.17,
    # Contact quality
    "avg_exit_velocity":         88.5,  "avg_exit_velocity_L5":  88.5,
    "avg_exit_velocity_szn":     88.5,  "avg_exit_velocity_szn_blended": 88.5,
    "avg_exit_velocity_L5_blended": 88.5,
    "avg_exit_velocity_prev5":   88.5,  "avg_exit_velocity_prev10": 88.5,
    "avg_launch_angle":          12.0,  "avg_launch_angle_L5":   12.0,
    "avg_launch_angle_szn":      12.0,  "avg_launch_angle_szn_blended": 12.0,
    "avg_launch_angle_prev5":    12.0,  "avg_launch_angle_prev10": 12.0,
    "sweet_spot_pct":            0.34,  "sweet_spot_pct_L5":     0.34,
    "sweet_spot_pct_szn":        0.34,  "sweet_spot_pct_szn_blended": 0.34,
    "sweet_spot_pct_L5_blended": 0.34,
    "sweet_spot_pct_prev5":      0.34,  "sweet_spot_pct_prev10": 0.34,
    "solid_contact_pct":         0.08,  "solid_contact_pct_L5":  0.08,
    "solid_contact_pct_szn":     0.08,  "solid_contact_pct_szn_blended": 0.08,
    "solid_contact_pct_prev5":   0.08,  "solid_contact_pct_prev10": 0.08,
    "avg_xba_contact":           0.250, "avg_xba_contact_L5":    0.250,
    "avg_xba_contact_szn":       0.250, "avg_xba_contact_szn_blended": 0.250,
    "avg_xba_contact_prev5":     0.250, "avg_xba_contact_prev10": 0.250,
    "avg_xwoba_contact":         0.345, "avg_xwoba_contact_L5":  0.345,
    "avg_xwoba_contact_szn":     0.345, "avg_xwoba_contact_szn_blended": 0.345,
    "avg_xwoba_contact_L5_blended": 0.345,
    "avg_xwoba_contact_prev5":   0.345, "avg_xwoba_contact_prev10": 0.345,
    # FanGraphs hits-side season stats
    "fg_fip":                 4.20, "fg_xfip":              4.20, "fg_siera": 4.20,
    "fg_tera":                4.20, "fg_xera":              4.20, "fg_era":   4.20,
    "fg_whip":                1.30, "fg_lob_pct":           0.72,
    "fg_hr_per_fb":           0.115, "fg_k_minus_bb_pct":    0.135,
    "fg_k_per_9":             8.5,  "fg_bb_per_9":          3.1,  "fg_hr_per_9": 1.15,
    "fg_gb_pct":              0.43, "fg_fb_pct":            0.36, "fg_ld_pct":   0.21,
    "fg_iffb_pct":            0.10, "fg_babip_allowed":     0.295,
    "fg_soft_pct":            0.17, "fg_med_pct":           0.49,
    "fg_hard_pct_allowed":    0.34, "fg_barrel_pct_allowed":0.07,
    "fg_hard_hit_pct_allowed":0.35,
}


def smart_feature_get(latest, f, fallback=0):
    val = latest.get(f, None)
    if val is not None and pd.notna(val):
        return val
    if f in _PLUS_FEATURES:
        return 100.0
    return _FEATURE_DEFAULTS.get(f, fallback)


# ══════════════════════════════════════════════════════════════════════════════
# MATH
# ══════════════════════════════════════════════════════════════════════════════

def beta_binom_pmf_array(n, alpha, beta_param):
    ks = np.arange(n + 1)
    log_comb     = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    log_beta_num = betaln(ks + alpha, n - ks + beta_param)
    log_beta_den = betaln(alpha, beta_param)
    pmf = np.exp(log_comb + log_beta_num - log_beta_den)
    return pmf / pmf.sum()


def expected_pmf_over_N(pred_p, pred_N, kappa, sigma_N, max_k=30):
    alpha      = max(pred_p * kappa, 0.01)
    beta_param = max((1 - pred_p) * kappa, 0.01)
    min_N      = max(1, int(pred_N - 3 * sigma_N))
    max_N      = int(pred_N + 3 * sigma_N) + 1
    N_values   = np.arange(min_N, max_N + 1)
    N_weights  = sp_stats.norm.pdf(N_values, loc=pred_N, scale=max(sigma_N, 0.5))
    N_weights /= N_weights.sum()
    combined   = np.zeros(max_k + 1)
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


def poisson_binomial_pmf(probs, max_k=30):
    probs  = np.asarray(probs, dtype=float)
    pmf    = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0]  = pmf[0] * (1 - p)
    out = np.zeros(max_k + 1)
    if len(pmf) > max_k + 1:
        out[:max_k] = pmf[:max_k]
        out[max_k]  = pmf[max_k:].sum()
    else:
        out[:len(pmf)] = pmf
    return out


def pmf_expected(pmf):
    return sum(i * pmf[i] for i in range(len(pmf)))


def _allocate_pas_to_slots(n_pa, n_slots):
    """Allocate n_pa total PAs across n_slots lineup positions.

    Mirrors predict/hits_walks.py — empirical PA-share weights so leadoff
    gets ~15% more PAs than slot 9. Returns a list of integer PA counts
    summing to n_pa. This matters for outs because slot 1 sees more PAs
    against the starter, and slot 1's K/out rate differs systematically
    from slot 9's; the naive `pa_idx % 9` cycling treats all slots equally.
    """
    base_weights = np.array(
        [1.15, 1.10, 1.08, 1.05, 1.02, 0.98, 0.95, 0.90, 0.85]
    )
    if n_slots <= len(base_weights):
        weights = base_weights[:n_slots]
    else:
        weights = np.concatenate([base_weights,
                                  np.full(n_slots - len(base_weights), 0.85)])
    weights = weights / weights.sum()

    fractional = weights * n_pa
    floors = np.floor(fractional).astype(int)
    remainder = n_pa - floors.sum()
    if remainder > 0:
        order = np.argsort(-(fractional - floors))
        for i in range(int(remainder)):
            floors[order[i]] += 1
    return floors.tolist()


def synthesize_blended_features(latest, pdf):
    """Backward-compat synthesis for blended-feature columns.

    Mirrors predict/hits_walks.py's _synth_blend / _synth_blend_l5 helpers.
    If the feature CSV was built with the updated features.py,
    the blended columns are already present and this is a no-op. If the
    CSV is older, this synthesizes the same empirical-Bayes blend on the fly.
    """
    def _synth_blend(szn_col, prev_col_10, prev_col_5, league_default,
                     denom_col, prior_n):
        out_col = f"{szn_col}_blended"
        if out_col in latest.index and pd.notna(latest.get(out_col)):
            return
        szn_val = latest.get(szn_col)
        if pd.isna(szn_val):
            szn_val = league_default
        n_szn = latest.get(denom_col, 0.0)
        if pd.isna(n_szn):
            n_szn = 0.0
        prior_val = None
        for pc in (prev_col_10, prev_col_5):
            if pc and pc in latest.index:
                v = latest.get(pc)
                if pd.notna(v):
                    prior_val = float(v)
                    break
        if prior_val is None and szn_col in pdf.columns:
            hist = pdf[szn_col].iloc[:-1]
            if len(hist) and hist.notna().any():
                prior_val = float(hist.dropna().mean())
        if prior_val is None:
            prior_val = league_default
        try:
            n = max(0.0, float(n_szn))
            blended = (n * float(szn_val) + prior_n * prior_val) / (n + prior_n)
        except (TypeError, ValueError):
            blended = league_default
        latest[out_col] = blended

    def _synth_blend_l5(l5_col, prev_col, league_default):
        out_col = f"{l5_col}_blended"
        if out_col in latest.index and pd.notna(latest.get(out_col)):
            return
        l5_val = latest.get(l5_col)
        if pd.isna(l5_val):
            l5_val = league_default
        n_actual = latest.get("n_starts_in_L5", 5.0)
        if pd.isna(n_actual):
            n_actual = 5.0
        n_actual = max(0.0, min(5.0, float(n_actual)))
        prior_val = None
        if prev_col and prev_col in latest.index:
            v = latest.get(prev_col)
            if pd.notna(v):
                prior_val = float(v)
        if prior_val is None and l5_col.replace("_L5", "_szn") in pdf.columns:
            base_col = l5_col.replace("_L5", "_szn")
            hist = pdf[base_col].iloc[:-1]
            if len(hist) and hist.notna().any():
                prior_val = float(hist.dropna().mean())
        if prior_val is None:
            prior_val = league_default
        L5_PRIOR = 5.0
        try:
            blended = (n_actual * float(l5_val) + L5_PRIOR * prior_val) / \
                      (n_actual + L5_PRIOR)
        except (TypeError, ValueError):
            blended = league_default
        latest[out_col] = blended

    # K-side season blends (already used by K models, but also helpful when
    # outs model picks them up via FEATURE_PATTERNS)
    _synth_blend("k_pct_szn",        "k_pct_prev10",        "k_pct_prev5",
                 0.22,  "plate_appearances_szn",     100.0)
    _synth_blend("whiff_pct_szn",    "whiff_pct_prev10",    None,
                 0.25,  "total_pitches_szn",         400.0)
    _synth_blend("csw_pct_szn",      "csw_pct_prev10",      None,
                 0.30,  "total_pitches_szn",         400.0)
    _synth_blend("bb_pct_szn",       "bb_pct_prev10",       "bb_pct_prev5",
                 0.085, "plate_appearances_szn",     100.0)
    _synth_blend("barrel_pct_szn",   "barrel_pct_prev10",   None,
                 0.07,  "batted_balls_szn",           80.0)
    _synth_blend("hard_hit_pct_szn", "hard_hit_pct_prev10", None,
                 0.35,  "batted_balls_szn",           80.0)
    _synth_blend("chase_rate_szn",   "chase_rate_prev10",   None,
                 0.30,  "out_of_zone_pitches_szn",   400.0)

    # H/W blends (mirrored from predict/hits_walks.py)
    _synth_blend("hits_per_pa_szn",  "hits_per_pa_prev10",  "hits_per_pa_prev5",
                 0.235, "plate_appearances_szn",     100.0)
    _synth_blend("bb_per_pa_szn",    "bb_per_pa_prev10",    "bb_per_pa_prev5",
                 0.085, "plate_appearances_szn",     100.0)
    _synth_blend("hr_per_pa_szn",    "hr_per_pa_prev10",    "hr_per_pa_prev5",
                 0.030, "plate_appearances_szn",     100.0)
    _synth_blend("babip_szn",        "babip_prev10",        "babip_prev5",
                 0.295, "plate_appearances_szn",     300.0)
    _synth_blend("lob_pct_szn",      "lob_pct_prev10",      "lob_pct_prev5",
                 0.72,  "plate_appearances_szn",     100.0)
    _synth_blend("gb_pct_szn",       "gb_pct_prev10",       "gb_pct_prev5",
                 0.43,  "batted_balls_szn",           80.0)
    _synth_blend("fb_pct_szn",       "fb_pct_prev10",       "fb_pct_prev5",
                 0.36,  "batted_balls_szn",           80.0)
    _synth_blend("ld_pct_szn",       "ld_pct_prev10",       "ld_pct_prev5",
                 0.21,  "batted_balls_szn",           80.0)
    _synth_blend("avg_exit_velocity_szn", "avg_exit_velocity_prev10", None,
                 88.5,  "batted_balls_szn",           80.0)
    _synth_blend("sweet_spot_pct_szn",    "sweet_spot_pct_prev10",    None,
                 0.34,  "batted_balls_szn",           80.0)
    _synth_blend("avg_xwoba_contact_szn", "avg_xwoba_contact_prev10", None,
                 0.345, "batted_balls_szn",           80.0)

    # L5 blends
    _synth_blend_l5("k_pct_L5",         "k_pct_prev10",         0.22)
    _synth_blend_l5("whiff_pct_L5",     "whiff_pct_prev10",     0.25)
    _synth_blend_l5("csw_pct_L5",       "csw_pct_prev10",       0.30)
    _synth_blend_l5("barrel_pct_L5",    "barrel_pct_prev10",    0.07)
    _synth_blend_l5("hard_hit_pct_L5",  "hard_hit_pct_prev10",  0.35)
    _synth_blend_l5("hits_per_pa_L5",   "hits_per_pa_prev10",   0.235)
    _synth_blend_l5("bb_per_pa_L5",     "bb_per_pa_prev10",     0.085)
    _synth_blend_l5("babip_L5",         "babip_prev10",         0.295)
    _synth_blend_l5("gb_pct_L5",        "gb_pct_prev10",        0.43)
    _synth_blend_l5("fb_pct_L5",        "fb_pct_prev10",        0.36)
    _synth_blend_l5("ld_pct_L5",        "ld_pct_prev10",        0.21)
    _synth_blend_l5("avg_exit_velocity_L5", "avg_exit_velocity_prev10", 88.5)
    _synth_blend_l5("sweet_spot_pct_L5",    "sweet_spot_pct_prev10",    0.34)
    _synth_blend_l5("avg_xwoba_contact_L5", "avg_xwoba_contact_prev10", 0.345)

    return latest


# ══════════════════════════════════════════════════════════════════════════════
# PER-PA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_per_pa_model(model_file, config_file):
    try:
        model = joblib.load(MODEL_DIR / model_file)
        with open(MODEL_DIR / config_file) as f:
            cfg = json.load(f)
        print(f"  ✓ Per-PA outs model loaded ({len(cfg['features'])} features)")
        return model, cfg["features"]
    except Exception as e:
        print(f"  (per-PA outs model not loaded: {e})")
        return None, None


def build_per_pa_pmf(latest, lineup, bf_pred, pp_model, pp_features):
    """Score per-PA out model and convolve to Poisson-Binomial PMF."""
    if pp_model is None or not lineup or bf_pred is None or bf_pred < 1:
        return None

    n_pa = max(1, min(int(round(float(bf_pred))), 40))

    p_throws = "R"
    if "p_throws" in latest.index and pd.notna(latest["p_throws"]):
        p_throws = str(latest["p_throws"]).upper()[:1]

    league_defaults = {
        "batter_k_rate_std":    0.22,  "batter_k_rate_L100":   0.22,
        "batter_k_rate_prior":  0.22,
        "batter_bb_rate_std":   0.085, "batter_bb_rate_L100":  0.085,
        "batter_bb_rate_prior": 0.085,
        "batter_hit_rate_std":  0.24,  "batter_hit_rate_L100": 0.24,
        "batter_hit_rate_prior":0.24,
        "batter_out_rate_std":  0.68,  "batter_out_rate_L100": 0.68,
        "batter_out_rate_prior":0.68,
        "batter_bip_rate_std":  0.68,  "batter_bip_rate_prior":0.68,
        "batter_pa_prior": 400.0,
    }

    pitcher_k     = float(smart_feature_get(latest, "k_pct_szn",       0.22))
    pitcher_out   = float(smart_feature_get(latest, "outs_per_pa_szn",  0.68))
    pitcher_barrel= float(smart_feature_get(latest, "barrel_pct_szn",   0.07))

    lineup_sorted = sorted(lineup, key=lambda b: b.get("lineup_position", 99))

    # Allocate PAs across the order using empirical PA-share weighting
    # (matches predict/hits_walks.py — leadoff sees ~15% more PAs than
    # slot 9 over a typical start). The naive `pa_idx % len(lineup)`
    # cycling treats all slots equally, which biases outs estimates upward
    # when slot 1's K-rate differs from slot 9's.
    pa_per_slot = _allocate_pas_to_slots(n_pa, len(lineup_sorted))

    rows = []
    pa_idx_global = 0
    for slot_idx, n_pa_for_slot in enumerate(pa_per_slot):
        if n_pa_for_slot == 0:
            continue
        slot     = lineup_sorted[slot_idx]
        pos      = slot.get("lineup_position", slot_idx + 1)
        bat_side = str(slot.get("bat_side", "R")).upper()[:1]

        for _ in range(n_pa_for_slot):
            pa_idx = pa_idx_global
            pa_idx_global += 1

            def slot_val(stat, fallback):
                key = f"opp_b{pos}_{stat}"
                v   = latest.get(key, None)
                if v is not None and pd.notna(v):
                    return float(v)
                return fallback

            batter_k_rate   = slot_val("k_rate",     league_defaults["batter_k_rate_std"])
            batter_k_rate_L = slot_val("k_rate_L10", league_defaults["batter_k_rate_L100"])
            batter_bb_rate  = slot_val("bb_rate",     league_defaults["batter_bb_rate_std"])
            batter_hit_rate = slot_val("hit_rate",    league_defaults["batter_hit_rate_std"])

            # out_rate proxy: 1 - hit_rate - bb_rate (approximately)
            batter_out_rate = max(0.30, min(0.90, 1.0 - batter_hit_rate - batter_bb_rate))

            swstr = slot_val("swstr_pct", None)
            batter_bip = (1.0 - float(swstr)) if swstr is not None else max(0.30, min(0.90, 1.0 - batter_k_rate - batter_bb_rate))
            batter_bip = max(0.30, min(0.90, batter_bip))

            row = {}
            for feat in pp_features:
                if feat == "batter_k_rate_std":
                    row[feat] = batter_k_rate
                elif feat == "batter_k_rate_L100":
                    row[feat] = batter_k_rate_L
                elif feat == "batter_k_rate_prior":
                    row[feat] = batter_k_rate
                elif feat == "batter_bb_rate_std":
                    row[feat] = batter_bb_rate
                elif feat == "batter_bb_rate_L100":
                    row[feat] = batter_bb_rate
                elif feat == "batter_bb_rate_prior":
                    row[feat] = batter_bb_rate
                elif feat == "batter_hit_rate_std":
                    row[feat] = batter_hit_rate
                elif feat == "batter_hit_rate_L100":
                    row[feat] = batter_hit_rate
                elif feat == "batter_hit_rate_prior":
                    row[feat] = batter_hit_rate
                elif feat == "batter_out_rate_std":
                    row[feat] = batter_out_rate
                elif feat == "batter_out_rate_L100":
                    row[feat] = batter_out_rate
                elif feat == "batter_out_rate_prior":
                    row[feat] = batter_out_rate
                elif feat == "batter_bip_rate_std":
                    row[feat] = batter_bip
                elif feat == "batter_bip_rate_prior":
                    row[feat] = batter_bip
                elif feat == "batter_pa_prior":
                    row[feat] = slot_val("pa_prior", 400.0)
                elif feat == "ix_pitcher_k_x_batter_k":
                    row[feat] = pitcher_k * batter_k_rate
                elif feat == "ix_pitcher_out_x_batter_out":
                    row[feat] = pitcher_out * batter_out_rate
                elif feat == "ix_pitcher_out_x_batter_out_recent":
                    row[feat] = pitcher_out * batter_out_rate
                elif feat == "ix_barrel_x_bip":
                    row[feat] = pitcher_barrel * batter_bip
                # ── H/W pipeline interaction additions ──
                elif feat == "ix_gb_x_bip":
                    pitcher_gb = float(smart_feature_get(latest, "gb_pct_szn_blended", 0.43))
                    row[feat] = pitcher_gb * batter_bip
                elif feat == "ix_hits_x_batter_hits":
                    pitcher_hits = float(smart_feature_get(latest, "hits_per_pa_szn_blended", 0.235))
                    row[feat] = pitcher_hits * batter_hit_rate
                elif feat == "same_handed":
                    row[feat] = 1 if p_throws == bat_side else 0
                elif feat == "inning":
                    row[feat] = (pa_idx // 3) + 1
                elif feat in latest.index and pd.notna(latest[feat]):
                    row[feat] = latest[feat]
                else:
                    row[feat] = smart_feature_get(latest, feat, 0.0)
            rows.append(row)

    X = pd.DataFrame(rows, columns=pp_features).fillna(0.0).values
    try:
        probs = pp_model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"    (per-PA outs scoring failed: {e})")
        return None
    return poisson_binomial_pmf(probs, max_k=30)


# ══════════════════════════════════════════════════════════════════════════════
# MLB API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_game_lineup(game_pk, side="away"):
    try:
        r    = requests.get(f"{MLB_API_BASE}/game/{game_pk}/boxscore", timeout=15)
        data = r.json()
    except Exception:
        return []
    team_data     = data.get("teams", {}).get(side, {})
    batting_order = team_data.get("battingOrder", [])
    players       = team_data.get("players", {})
    lineup = []
    for pos_idx, player_id in enumerate(batting_order[:9], 1):
        pinfo  = players.get(f"ID{player_id}", {})
        person = pinfo.get("person", {})
        lineup.append({
            "player_id":       player_id,
            "player_name":     person.get("fullName", ""),
            "bat_side":        pinfo.get("batSide", {}).get("code", "R"),
            "lineup_position": pos_idx,
        })
    return lineup


def build_per_batter_live_features(lineup, pitcher_hand, features_df):
    if not lineup:
        return {}
    lu_hist = None
    lineups_path = Path("data/game_lineups.csv")
    if lineups_path.exists():
        lu_hist = pd.read_csv(lineups_path)
    fg = None
    fg_path = Path("data/fangraphs_batting_seasons.csv")
    if fg_path.exists():
        fg = pd.read_csv(fg_path)

    batter_stats = {}
    if lu_hist is not None:
        lu_hist = lu_hist.sort_values(["player_id", "game_pk"])
        for pid in [b["player_id"] for b in lineup]:
            rows = lu_hist[lu_hist["player_id"] == pid]
            if len(rows) == 0:
                continue
            tab = rows["at_bats"].sum()
            if tab > 0:
                batter_stats[pid] = {
                    "k_rate":   rows["strikeouts"].sum() / tab,
                    "hit_rate": rows["hits"].sum() / tab,
                    "bb_rate":  rows["walks"].sum() / (tab + rows["walks"].sum())
                                if (tab + rows["walks"].sum()) > 0 else 0,
                }
                rec = rows.tail(10)
                rab = rec["at_bats"].sum()
                if rab > 0:
                    batter_stats[pid]["k_rate_L10"]  = rec["strikeouts"].sum() / rab
                    batter_stats[pid]["bb_rate_L10"] = (
                        rec["walks"].sum() / (rab + rec["walks"].sum())
                        if (rab + rec["walks"].sum()) > 0 else 0
                    )
                    batter_stats[pid]["hit_rate_L10"] = rec["hits"].sum() / rab

    fg_lookup = {}
    if fg is not None:
        def norm_name(n):
            return str(n).strip().lower().replace(".", "").replace(",", "") if pd.notna(n) else ""
        fg_latest = fg.sort_values("Season").groupby("Name").last().reset_index()
        fg_stat_cols = {
            "SwStr%": "swstr_pct", "Contact%": "contact_pct",
            "O-Swing%": "o_swing_pct", "Z-Contact%": "z_contact_pct",
            "K%+": "k_pct_plus", "wRC+": "wrc_plus", "xwOBA": "xwoba",
            "Barrel%": "barrel_pct", "HardHit%": "hard_hit_pct",
            "BB%": "bb_pct", "ISO": "iso", "CSW%": "csw_pct", "TTO%": "tto_pct",
        }
        for _, row in fg_latest.iterrows():
            nm = norm_name(row["Name"])
            stats_d = {}
            for src, dst in fg_stat_cols.items():
                if src in row.index:
                    v = pd.to_numeric(row[src], errors="coerce")
                    if pd.notna(v):
                        stats_d[dst] = v
            fg_lookup[nm] = stats_d

    result  = {}
    k_rates = []
    for batter in lineup:
        pos      = batter["lineup_position"]
        pid      = batter["player_id"]
        bat_side = batter.get("bat_side", "R")
        prefix   = f"opp_b{pos}_"
        bs       = batter_stats.get(pid, {})
        result[prefix + "k_rate"]       = bs.get("k_rate",      0.22)
        result[prefix + "k_rate_L10"]   = bs.get("k_rate_L10",  0.22)
        result[prefix + "bb_rate"]      = bs.get("bb_rate",      0.08)
        result[prefix + "bb_rate_L10"]  = bs.get("bb_rate_L10",  0.08)
        result[prefix + "hit_rate"]     = bs.get("hit_rate",     0.25)
        result[prefix + "hit_rate_L10"] = bs.get("hit_rate_L10", 0.25)
        kr = bs.get("k_rate", np.nan)
        if pd.notna(kr):
            k_rates.append(kr)
        result[prefix + "platoon_disadv"] = (
            0.0 if bat_side == "S" else
            (1.0 if bat_side == pitcher_hand else 0.0)
        )
        nm   = str(batter.get("player_name", "")).strip().lower().replace(".", "").replace(",", "")
        fg_s = fg_lookup.get(nm, {})
        for dst, dflt in [
            ("swstr_pct", 0.11), ("contact_pct", 0.78), ("o_swing_pct", 0.30),
            ("z_contact_pct", 0.87), ("k_pct_plus", 100.0), ("wrc_plus", 100.0),
            ("xwoba", 0.320), ("barrel_pct", 0.07), ("hard_hit_pct", 0.35),
            ("bb_pct", 0.08), ("csw_pct", 0.30), ("tto_pct", 0.33),
        ]:
            result[prefix + dst] = fg_s.get(dst, dflt)

    if len(k_rates) >= 5:
        k_arr = np.array(k_rates)
        result["opp_lineup_k_rate_p90"]  = float(np.percentile(k_arr, 90))
        result["opp_lineup_k_rate_p10"]  = float(np.percentile(k_arr, 10))
        result["opp_lineup_k_rate_iqr"]  = float(np.percentile(k_arr, 75) - np.percentile(k_arr, 25))
        result["opp_lineup_k_rate_skew"] = float(pd.Series(k_arr).skew())
        pa_w = np.array([1.15,1.10,1.08,1.05,1.02,0.98,0.95,0.90,0.85])[:len(k_arr)]
        result["opp_lineup_expected_k_total"] = float((k_arr * pa_w / pa_w.sum()).sum())
    platoon_vals = [result.get(f"opp_b{i}_platoon_disadv", 0) for i in range(1, 10)]
    result["opp_lineup_platoon_disadv_count"] = sum(v for v in platoon_vals if pd.notna(v))
    return result


def build_pitcher_batter_history_live(pitcher_id, lineup, features_df):
    if not lineup:
        return {}
    lineups_path = Path("data/game_lineups.csv")
    bpt_path     = Path("data/batter_pitch_type_all.csv")
    if not lineups_path.exists() or not bpt_path.exists():
        return {"pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
                "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0}
    lu_hist = pd.read_csv(lineups_path)
    bpt     = pd.read_csv(bpt_path)
    today_batter_ids = set(b["player_id"] for b in lineup)
    pitcher_games    = features_df[features_df["pitcher_id"] == pitcher_id][["game_pk"]].drop_duplicates()
    pitcher_game_pks = set(pitcher_games["game_pk"])
    hist_batters = lu_hist[
        (lu_hist["game_pk"].isin(pitcher_game_pks)) &
        (lu_hist["player_id"].isin(today_batter_ids))
    ]
    if len(hist_batters) == 0:
        return {"pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
                "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0}
    hist_bpt = bpt[
        (bpt["game_pk"].isin(pitcher_game_pks)) &
        (bpt["batter"].isin(today_batter_ids))
    ]
    familiar = hist_batters["player_id"].nunique()
    if len(hist_bpt) == 0:
        return {"pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
                "pb_hist_pa": 0, "pb_familiar_batters": familiar,
                "pb_familiarity_pct": familiar / max(len(lineup), 1)}
    total_ks     = hist_bpt["bpt_strikeouts"].sum()
    total_pa     = hist_bpt["bpt_pa"].sum()
    total_whiffs = hist_bpt["bpt_whiffs"].sum()
    total_pitches= hist_bpt["bpt_pitches_seen"].sum()
    return {
        "pb_hist_k_rate":      total_ks / total_pa if total_pa > 0 else 0.22,
        "pb_hist_whiff_rate":  total_whiffs / total_pitches if total_pitches > 0 else 0.25,
        "pb_hist_pa":          int(total_pa),
        "pb_familiar_batters": familiar,
        "pb_familiarity_pct":  familiar / max(len(lineup), 1),
    }


def get_schedule(date_str):
    params = {"date": date_str, "sportId": 1, "hydrate": "probablePitcher,team"}
    try:
        r    = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=15)
        data = r.json()
    except Exception:
        return []
    games = []
    for game in data.get("dates", [{}])[0].get("games", []):
        home = game.get("teams", {}).get("home", {})
        away = game.get("teams", {}).get("away", {})
        games.append({
            "game_pk":           game["gamePk"],
            "home_team":         home.get("team", {}).get("abbreviation", ""),
            "away_team":         away.get("team", {}).get("abbreviation", ""),
            "home_pitcher_name": home.get("probablePitcher", {}).get("fullName", "TBD"),
            "home_pitcher_id":   home.get("probablePitcher", {}).get("id"),
            "away_pitcher_name": away.get("probablePitcher", {}).get("fullName", "TBD"),
            "away_pitcher_id":   away.get("probablePitcher", {}).get("id"),
        })
    return games


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def predict_slate(date_str):
    print(f"\n{'=' * 110}")
    print(f"  MLB PITCHER OUTS RECORDED PREDICTIONS — {date_str}")
    print(f"  BB = Beta-Binomial  |  PP = Per-PA XGBoost (real lineup features)")
    print(f"{'=' * 110}")

    # Load BF model
    try:
        bf_model = joblib.load(MODEL_DIR / "bf_model.joblib")
        with open(MODEL_DIR / "beta_binom_config.json") as f:
            bb_config = json.load(f)
        bf_features   = bb_config.get("bf_features", None)
        sigma_n_bf    = float(bb_config.get("sigma_n") or bb_config.get("sigma_n_global") or
                              bb_config.get("sigma_N") or bb_config.get("sigma_N_global") or 2.0)
        bf_is_log     = bool(bb_config.get("bf_is_log", False))
        bf_log_sigma2 = float(bb_config.get("bf_log_sigma2", 0.0))
    except Exception as e:
        print(f"  ✗ Could not load BF model: {e}")
        return

    # Load BB outs model
    try:
        outs_model = joblib.load(MODEL_DIR / "outs_rate_model.joblib")
        with open(MODEL_DIR / "outs_config.json") as f:
            outs_config = json.load(f)
        outs_sigma    = float(outs_config.get("sigma_n") or outs_config.get("sigma_n_global") or
                              outs_config.get("sigma_N") or outs_config.get("sigma_N_global") or sigma_n_bf)
        outs_features = outs_config["rate_features"]
        outs_kappa    = outs_config["kappa"]
        print(f"  ✓ BB outs model loaded (κ={outs_kappa:.1f}, σ_N={outs_sigma:.2f})")
    except Exception as e:
        print(f"  ✗ Could not load outs model: {e}")
        return

    # Load per-PA model (optional)
    pp_model, pp_features = load_per_pa_model(
        "per_pa_out_model.joblib", "per_pa_out_config.json"
    )

    # Load features
    feat_path = DATA_DIR / "pitcher_model_features.csv"
    if not feat_path.exists():
        print(f"  ✗ Feature file not found: {feat_path}")
        return
    features_df = pd.read_csv(feat_path)
    if "pitcher_id" not in features_df.columns:
        features_df["pitcher_id"] = features_df.get("pitcher", features_df.index)
    features_df["pitcher_id"] = pd.to_numeric(
        features_df["pitcher_id"], errors="coerce"
    ).astype("Int64")
    print(f"  ✓ Features loaded ({len(features_df)} rows)")

    games = get_schedule(date_str)
    if not games:
        print(f"  ℹ No games found for {date_str}")
        return
    print(f"  ✓ {len(games)} games")

    predictions = []
    skipped     = []

    for game in games:
        for side in ["away", "home"]:
            pname = game[f"{side}_pitcher_name"]
            pid   = game[f"{side}_pitcher_id"]
            if not pid or pname == "TBD":
                continue

            own_team = game[f"{side}_team"]
            opp_team = game["home_team"] if side == "away" else game["away_team"]
            opp_side = "home" if side == "away" else "away"
            opp_pid  = game.get(f"{opp_side}_pitcher_id")

            pitcher_rows = features_df[features_df["pitcher_id"] == pid].copy()
            if pitcher_rows.empty:
                pitcher_rows = features_df[features_df["pitcher_id"] == int(pid)].copy()
            if pitcher_rows.empty:
                for name_col in ["player_name", "pitcher_name", "pitcher"]:
                    if name_col in features_df.columns:
                        pitcher_rows = features_df[features_df[name_col] == pname].copy()
                        if not pitcher_rows.empty:
                            break
            if pitcher_rows.empty:
                skipped.append(f"{pname} (id={pid})")
                continue

            pitcher_rows = pitcher_rows.sort_values("game_date")
            latest       = pitcher_rows.iloc[-1].copy()

            # ── Synthesize blended features if missing (backward-compat) ──
            # If the feature CSV was built before 02 added the _blended columns,
            # this fills them in on-the-fly using the same empirical-Bayes math.
            # No-op if the columns are already present.
            latest = synthesize_blended_features(latest, pitcher_rows)

            if opp_pid is not None:
                opp_rows = features_df[features_df["pitcher_id"] == opp_pid]
                if opp_rows.empty:
                    opp_rows = features_df[features_df["pitcher_id"] == int(opp_pid)]
                if not opp_rows.empty:
                    opp_latest = opp_rows.sort_values("game_date").iloc[-1]
                    opp_map = {
                        "k_pct_L5": "opp_sp_k_pct", "bb_pct_L5": "opp_sp_bb_pct",
                        "whiff_pct_L5": "opp_sp_whiff_pct", "csw_pct_L5": "opp_sp_csw_pct",
                        "barrel_pct_L5": "opp_sp_barrel_pct", "hard_hit_pct_L5": "opp_sp_hard_hit_pct",
                        "plate_appearances_L5": "opp_sp_bf_avg", "est_innings_L5": "opp_sp_ip_avg",
                        "is_short_outing_L5": "opp_sp_short_pct",
                        "outs_recorded_L5": "opp_sp_outs_avg", "outs_per_pa_L5": "opp_sp_outs_rate",
                        "avg_velocity_L5": "opp_sp_velo",
                        "plate_appearances_L10": "opp_sp_bf_avg_L10",
                        "est_innings_L10": "opp_sp_ip_avg_L10",
                        "is_short_outing_L10": "opp_sp_short_pct_L10",
                        "outs_recorded_L10": "opp_sp_outs_avg_L10",
                    }
                    OPP_DEFAULTS = {
                        "k_pct_L5": 0.22, "bb_pct_L5": 0.08, "whiff_pct_L5": 0.25,
                        "csw_pct_L5": 0.30, "barrel_pct_L5": 0.07, "hard_hit_pct_L5": 0.35,
                        "plate_appearances_L5": 22.0, "est_innings_L5": 5.5,
                        "is_short_outing_L5": 0.20, "outs_recorded_L5": 16.0,
                        "outs_per_pa_L5": 0.72, "avg_velocity_L5": 93.0,
                        "plate_appearances_L10": 22.0, "est_innings_L10": 5.5,
                        "is_short_outing_L10": 0.20, "outs_recorded_L10": 16.0,
                    }
                    for src, dst in opp_map.items():
                        val = opp_latest.get(src, OPP_DEFAULTS.get(src, 0))
                        latest[dst] = val if pd.notna(val) else OPP_DEFAULTS.get(src, 0)
                    bf5 = latest.get("plate_appearances_L5", 22)
                    latest["ix_both_deep"]      = (bf5 * latest.get("opp_sp_bf_avg", 22)) / 500.0
                    latest["ix_aces_matchup"]   = latest.get("k_pct_L5", 0) * latest.get("opp_sp_k_pct", 0)
                    latest["ix_combined_depth"] = latest.get("est_innings_L5", 5) + latest.get("opp_sp_ip_avg", 5)

            opp_lineup_side = "home" if side == "away" else "away"
            lineup = fetch_game_lineup(game["game_pk"], side=opp_lineup_side)
            if lineup:
                p_throws = latest.get("p_throws", "R")
                if pd.isna(p_throws):
                    p_throws = "R"
                pb_features = build_per_batter_live_features(lineup, str(p_throws), features_df)
                for k, v in pb_features.items():
                    latest[k] = v
                pb_hist = build_pitcher_batter_history_live(pid, lineup, features_df)
                for k, v in pb_hist.items():
                    latest[k] = v

            current_season = pitcher_rows["season"].max() if "season" in pitcher_rows.columns else None
            start_num = len(pitcher_rows[pitcher_rows["season"] == current_season]) if current_season else len(pitcher_rows)

            try:
                if bf_features:
                    bf_vec = [smart_feature_get(latest, f) for f in bf_features]
                    n_hat  = bf_model.predict([bf_vec])[0]
                else:
                    n_hat = 22.0
                if bf_is_log:
                    n_hat = float(np.exp(n_hat) * np.exp(bf_log_sigma2 / 2))
                n_hat = max(n_hat, 6)
            except Exception:
                n_hat = 22.0

            # ── BB prediction ────────────────────────────────────────────
            try:
                feat_vec = [smart_feature_get(latest, f) for f in outs_features]
                p_hat_bb = np.clip(outs_model.predict([feat_vec])[0], 0.01, 0.99)
            except Exception:
                continue

            pmf_bb    = expected_pmf_over_N(p_hat_bb, n_hat, outs_kappa, outs_sigma, max_k=30)
            exp_bb    = pmf_expected(pmf_bb)
            est_ip_bb = exp_bb / 3

            row = {
                "pitcher": pname, "team": own_team, "opponent": opp_team,
                "N_hat": round(n_hat, 1), "start_num": start_num,
                "bb_outs_rate": round(p_hat_bb, 4),
                "bb_outs_pred": round(exp_bb, 1),
                "bb_est_IP":    round(est_ip_bb, 1),
            }

            for line in range(max(9, int(exp_bb) - 4), int(exp_bb) + 5):
                p_over = pmf_bb[line:].sum() if line < len(pmf_bb) else 0
                row[f"bb_P(O>={line})"] = round(float(p_over), 3)

            half_center = round(exp_bb - 0.5) + 0.5
            half_lines  = [half_center + d for d in [-2, -1, 0, 1, 2]]
            while half_lines[0] < 0.5:
                half_lines = [h + 1 for h in half_lines]
            for hl in half_lines:
                over_k  = int(hl) + 1
                p_over  = float(pmf_bb[over_k:].sum()) if over_k < len(pmf_bb) else 0.0
                p_under = 1.0 - p_over
                row[f"O{hl:g}_bb_over_prob"]  = round(p_over,  3)
                row[f"O{hl:g}_bb_under_prob"] = round(p_under, 3)

            # ── PP prediction ─────────────────────────────────────────────
            if lineup and pp_model is not None:
                try:
                    pmf_pp = build_per_pa_pmf(latest, lineup, n_hat, pp_model, pp_features)
                    if pmf_pp is not None:
                        exp_pp    = pmf_expected(pmf_pp)
                        est_ip_pp = exp_pp / 3
                        row["pp_outs_pred"] = round(exp_pp, 1)
                        row["pp_est_IP"]    = round(est_ip_pp, 1)

                        for line in range(max(9, int(exp_pp) - 4), int(exp_pp) + 5):
                            p_over = pmf_pp[line:].sum() if line < len(pmf_pp) else 0
                            row[f"pp_P(O>={line})"] = round(float(p_over), 3)

                        half_center = round(exp_pp - 0.5) + 0.5
                        half_lines_pp = [half_center + d for d in [-2, -1, 0, 1, 2]]
                        while half_lines_pp[0] < 0.5:
                            half_lines_pp = [h + 1 for h in half_lines_pp]
                        for hl in half_lines_pp:
                            over_k  = int(hl) + 1
                            p_over  = float(pmf_pp[over_k:].sum()) if over_k < len(pmf_pp) else 0.0
                            p_under = 1.0 - p_over
                            row[f"O{hl:g}_pp_over_prob"]  = round(p_over,  3)
                            row[f"O{hl:g}_pp_under_prob"] = round(p_under, 3)
                except Exception as e:
                    print(f"    (per-PA outs failed for {pname}: {e})")

            predictions.append(row)

    if not predictions:
        print("  ℹ No predictions generated.")
        if skipped:
            print(f"  ⚠ {len(skipped)} skipped:")
            for s in skipped[:10]:
                print(f"    - {s}")
        return

    pdf = pd.DataFrame(predictions)
    has_pp = "pp_outs_pred" in pdf.columns and pdf["pp_outs_pred"].notna().any()

    # ── Side-by-side summary table ────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  PITCHER OUTS RECORDED PREDICTIONS  (BB = Beta-Binomial | PP = Per-PA XGBoost)")
    print(f"{'=' * 120}")

    bb_prob_cols = sorted([c for c in pdf.columns if c.startswith("bb_P(O>=")])
    pp_prob_cols = sorted([c for c in pdf.columns if c.startswith("pp_P(O>=")])

    hdr = f"  {'Pitcher':25s} {'Team':>4s} {'Opp':>4s} {'BB Outs':>7s} {'BB ~IP':>6s}"
    for c in bb_prob_cols[:4]:
        hdr += f" {c[3:]:>10s}"
    if has_pp:
        hdr += f"  {'PP Outs':>7s} {'PP ~IP':>6s}"
        for c in pp_prob_cols[:4]:
            hdr += f" {c[3:]:>10s}"
    print(hdr)
    print(f"  {'-' * 110}")

    for _, r in pdf.sort_values("bb_outs_pred", ascending=False).iterrows():
        line = (f"  {r['pitcher']:25s} {r['team']:>4s} {r['opponent']:>4s} "
                f"{r['bb_outs_pred']:>7.1f} {r['bb_est_IP']:>6.1f}")
        for c in bb_prob_cols[:4]:
            v = r.get(c, np.nan)
            line += f" {v:>9.1%}" if pd.notna(v) else f" {'—':>9s}"
        if has_pp:
            pp_pred = r.get("pp_outs_pred", np.nan)
            pp_ip   = r.get("pp_est_IP", np.nan)
            if pd.notna(pp_pred):
                line += f"  {pp_pred:>7.1f} {pp_ip:>6.1f}"
                for c in pp_prob_cols[:4]:
                    v = r.get(c, np.nan)
                    line += f" {v:>9.1%}" if pd.notna(v) else f" {'—':>9s}"
            else:
                line += "  (no lineup yet)"
        print(line)

    # ── Threshold probability ladders ─────────────────────────────────────
    # For each pitcher, show P(outs > line) and P(outs < line) across five
    # half-run thresholds centred on the point prediction.
    for model_tag, pred_col in ([("BB", "bb_outs_pred")] +
                                 ([("PP", "pp_outs_pred")] if has_pp else [])):
        print(f"\n  THRESHOLD PROBABILITIES — OUTS [{model_tag} MODEL]")
        print(f"  {'-' * 120}")
        hdr = f"  {'Pitcher':25s} {'Outs':>5s}"
        for slot in ["L1", "L2", "L3", "L4", "L5"]:
            hdr += f"  {slot+' line':>7s} {'P(over)':>8s} {'P(under)':>8s}"
        print(hdr)
        print(f"  {'-' * 120}")

        tag = model_tag.lower()
        for _, r in pdf.sort_values(pred_col, ascending=False).iterrows():
            pred_val = r.get(pred_col, np.nan)
            if pd.isna(pred_val):
                continue
            half_center = round(pred_val - 0.5) + 0.5
            half_lines  = sorted([half_center + d for d in [-2, -1, 0, 1, 2]])
            while half_lines[0] < 0.5:
                half_lines = [h + 1 for h in half_lines]
            row_line = f"  {r['pitcher']:25s} {pred_val:>5.1f}"
            for hl in half_lines:
                key = f"O{hl:g}_{tag}"
                p_o = r.get(f"{key}_over_prob")
                p_u = r.get(f"{key}_under_prob")
                p_o_s = f"{p_o:>7.1%}" if pd.notna(p_o) else "      —"
                p_u_s = f"{p_u:>7.1%}" if pd.notna(p_u) else "      —"
                row_line += f"  {hl:>7.1f} {p_o_s:>8s} {p_u_s:>8s}"
            print(row_line)

    if skipped:
        print(f"\n  ⚠ {len(skipped)} pitchers skipped:")
        for s in skipped[:10]:
            print(f"    - {s}")

    out_path = OUTPUT_DIR / f"outs_{date_str}.csv"
    pdf.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved to {out_path}")
    if has_pp:
        print(f"    BB columns: bb_outs_pred, bb_P(O>=N), O*_bb_*")
        print(f"    PP columns: pp_outs_pred, pp_P(O>=N), O*_pp_*  (when lineup posted)")


if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    predict_slate(date_str)
