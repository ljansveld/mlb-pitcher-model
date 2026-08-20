"""
MLB Daily Pitcher Strikeout Predictions
=============================================================================
Pulls today's probable starters from the MLB Stats API and produces a full
probability distribution over strikeouts for each one — not just a point
estimate. Output is P(K >= k) for every threshold k, which is what you
actually want when the question is "how likely is 7+?" rather than
"what's the average?".

TWO MODELS RUN SIDE BY SIDE:
  BB  = Beta-Binomial (train/strikeouts.py) — start-level rate model.
        Predicts the per-PA strikeout rate p and batters faced N, then
        combines them into a Beta-Binomial(N, alpha, beta) PMF. The Beta
        prior absorbs start-to-start dispersion that a plain Binomial
        would understate.
  PP  = Per-PA XGBoost (train/strikeouts_per_pa.py) — plate-appearance level model
        with real per-batter features from the actual posted lineup.
        Scores P(K) for each projected PA, then convolves the Bernoullis
        via Poisson-Binomial. Captures lineup heterogeneity the BB model
        averages away, but requires the lineup to be posted (~2-4 hrs
        before first pitch).

Both models run independently and are reported side by side, so their
disagreement is visible rather than hidden behind a blend.

USAGE:
    python run.py predict strikeouts               # Today's games
    python run.py predict strikeouts 2026-04-15     # Specific date

REQUIRES:
    - models/rate_model.joblib (from train/strikeouts.py)
    - models/bf_model.joblib
    - models/beta_binom_config.json
    - data/pitcher_model_features.csv
    - models/per_pa_k_model.joblib  (from train/strikeouts_per_pa.py)
    - models/per_pa_k_config.json
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

# ── Paths ────────────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────────────────────
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ── Data-sufficiency flag (FLAG-ONLY — never drops a prediction) ─────────────
# The Beta-Binomial produces overconfident probabilities when a pitcher has
# little real data: it falls back on the career/league prior and dresses it
# up as a confident estimate. These thresholds flag those predictions so you
# know which to distrust. Tune freely; set FLAG_THIN_DATA = False to silence.
FLAG_THIN_DATA       = True
MIN_STARTS_IN_L5     = 4    # last-5 window must be backed by >= this many REAL starts
MIN_STARTS_THIS_SZN  = 4    # pitcher must have >= this many COMPLETED starts THIS season
MAX_DAYS_SINCE_START = 21   # >= this many days since the pitcher's last start
                            # (measured to the slate date) = long layoff / injury return


def data_sufficiency_flag(n_starts_l5, starts_this_szn, days_since_last):
    """Return (is_sufficient: bool, reasons: list[str]).

    Flag-only: a False result NEVER drops the prediction, it just marks it as
    low-confidence. Missing/NaN start-counts are treated as 0 (i.e.
    insufficient, the conservative choice); a missing days-since-last value is
    treated as a normal 5 so we don't false-flag when the date can't be parsed.
    """
    def _num(v, default):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return default if pd.isna(f) else f

    n5   = _num(n_starts_l5, 0.0)
    pszn = _num(starts_this_szn, 0.0)
    gap  = _num(days_since_last, 5.0)

    reasons = []
    if n5 < MIN_STARTS_IN_L5:
        reasons.append(f"only {n5:.0f}/5 recent starts with data")
    if pszn < MIN_STARTS_THIS_SZN:
        reasons.append(f"only {pszn:.0f} start{'' if pszn == 1 else 's'} this season")
    if gap >= MAX_DAYS_SINCE_START:
        reasons.append(f"long layoff ({gap:.0f}d since last start)")
    return (len(reasons) == 0), reasons


# ══════════════════════════════════════════════════════════════════════════════
# SMART FEATURE DEFAULTS (league-average fallbacks for missing values)
# ══════════════════════════════════════════════════════════════════════════════

# Features centered at 100 (Stuff+, Location+, Pitching+ style)
_PLUS_FEATURES = {"fg_stuff_plus", "fg_location_plus", "fg_pitching_plus",
                  "fg_stuff_fa", "fg_stuff_sl", "fg_stuff_ch", "fg_stuff_cu",
                  "fg_stuff_si", "fg_stuff_fc", "fg_stuff_fs",
                  "fg_loc_fa", "fg_loc_sl", "fg_loc_ch", "fg_loc_cu",
                  "fg_loc_si", "fg_loc_fc", "fg_loc_fs"}

# League-average defaults for rate features
_FEATURE_DEFAULTS = {
    "fg_swstr_pct": 0.11, "fg_o_swing_pct": 0.30, "fg_z_swing_pct": 0.65,
    "fg_contact_pct": 0.78, "fg_o_contact_pct": 0.65, "fg_z_contact_pct": 0.87,
    "fg_zone_pct_szn": 0.45, "fg_first_strike_pct": 0.60,
    "fg_tto_pct": 0.33, "fg_pitcher_frm": 0.0,
    "catcher_frm": 0.0,
    "opp_lu_swstr_pct": 0.11, "opp_lu_o_swing_pct": 0.30,
    "opp_lu_z_contact_pct": 0.87, "opp_lu_contact_pct": 0.78,
    "opp_lu_tto_pct": 0.33, "opp_lu_csw_pct": 0.30,
    "opp_lu_fg_k_pct": 0.22, "opp_lu_barrel_pct": 0.07,
    "opp_lu_hard_hit_pct": 0.35,
    "opp_lu_k_rate_std": 0.06, "opp_lu_tto_pct_std": 0.05,
    "opp_lu_max_k_rate": 0.30, "opp_lu_min_k_rate": 0.12,
    "velo_delta_L3_vs_szn": 0.0,
    "platoon_whiff_diff_L5": 0.0,
    "pitcher_tto_L5": 0.33, "pitcher_tto_L10": 0.33, "pitcher_tto_szn": 0.33,
    "ix_stuff_x_lu_k": 0.22, "ix_swstr_x_contact": 0.024,
    "ix_stuff_x_contact": 0.22, "ix_stuff_x_chase": 0.30,
    "ix_pitcher_lu_swstr": 0.012, "ix_tto_matchup": 0.11,
    # Pitcher × umpire interactions (default = pitcher avg × league avg ump)
    "ix_pitcher_csw_x_ump_k": 0.066, "ix_pitcher_edge_x_ump_zone": 0.12,
    "ix_pitcher_k_x_ump_k": 0.048, "ix_pitcher_bb_x_ump_bb": 0.006,
    # Pitcher × catcher framing (default = 0, framing is centered at 0)
    "ix_catcher_frm_x_csw": 0.0, "ix_catcher_frm_x_strike": 0.0,
    "ix_catcher_frm_x_k": 0.0,
    # Pitch count fatigue
    "pitchcount_L1": 90.0, "pitchcount_L2_total": 180.0,
    "heavy_prev_start": 0.2, "pitches_per_out_L1": 5.5, "pitches_per_out_L3": 5.5,
    # Enhanced lineup distribution
    "opp_lu_k_rate_median": 0.22, "opp_lu_top3_k_mean": 0.28,
    "opp_lu_bot3_k_mean": 0.16, "opp_lu_top3_bot3_gap": 0.12,
    # Pitch-type matchup score
    "pitch_matchup_score": 0.05, "pitch_k_matchup_score": 0.04,
    # Pitcher-batter history
    "pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
    "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0,
    # ── New (April 2026 recency-bias fix) ──────────────────────────────────
    # Blended / empirical-Bayes season rates (from build_blended_rate_features
    # in features.py). Defaults = league averages, same as their
    # raw _szn counterparts. Falling back to these when the feature is missing
    # is safe: it matches what the blend would produce for a pitcher with
    # zero history (prior = league mean, n_szn = 0 → blended = league mean).
    "k_pct_szn_blended": 0.22,
    "whiff_pct_szn_blended": 0.25,
    "csw_pct_szn_blended": 0.30,
    "bb_pct_szn_blended": 0.085,
    "barrel_pct_szn_blended": 0.07,
    "hard_hit_pct_szn_blended": 0.35,
    "chase_rate_szn_blended": 0.30,
    # L5 blended — same defaults as L5 raw
    "k_pct_L5_blended": 0.22,
    "whiff_pct_L5_blended": 0.25,
    "csw_pct_L5_blended": 0.30,
    "barrel_pct_L5_blended": 0.07,
    "hard_hit_pct_L5_blended": 0.35,
    # Prior-season carryover (last-N-starts rates from previous season).
    # Defaults = league averages; the feature is NaN for rookies and year-1
    # pitchers, and league average is the best neutral guess.
    "k_pct_prev5": 0.22, "k_pct_prev10": 0.22,
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
    # Reliability signals — default to 0 for a brand-new pitcher (is safe:
    # XGBoost will read this as "no data, lean on prior").
    "prior_starts_this_season": 0,
    "prior_starts_available": 0,
    "n_starts_in_L5": 0,
    "n_starts_in_L10": 0,
}


def smart_feature_get(latest, feature_name, fallback=0):
    """Get a feature value with a sensible league-average default."""
    val = latest.get(feature_name, None)
    if val is not None and pd.notna(val):
        return val
    if feature_name in _PLUS_FEATURES:
        return 100.0
    if feature_name in _FEATURE_DEFAULTS:
        return _FEATURE_DEFAULTS[feature_name]
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# PER-PA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_per_pa_artifacts():
    """Load per-PA model and feature list.

    Returns (model, features) or (None, None) if artifacts are missing.
    """
    try:
        model = joblib.load(MODEL_DIR / "per_pa_k_model.joblib")
        with open(MODEL_DIR / "per_pa_k_config.json") as f:
            pp_cfg = json.load(f)
        features = pp_cfg["features"]
        return model, features
    except Exception as e:
        print(f"  (per-PA model not loaded: {e} — BB-only path)")
        return None, None


def poisson_binomial_pmf(probs, max_k=25):
    """PMF of sum of independent Bernoulli(p_i) via standard recursion.

    Returns length-(max_k+1) array with overflow mass collected in the
    final bin so the array aligns with the BB pmf for comparison at a
    given K threshold.
    """
    probs = np.asarray(probs, dtype=float)
    pmf = np.zeros(len(probs) + 1)
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


def _allocate_pas_to_slots(n_pa, n_slots):
    """Allocate n_pa total PAs across n_slots lineup positions.

    Uses the same empirical weights as features.py's
    lineup_expected_k_total: leadoff slot gets ~15% more PAs than slot 9
    on average, decreasing monotonically through the order.

    The previous implementation rotated `pa_idx % n_slots` which gave
    earlier slots one extra PA only when n_pa wasn't a multiple of 9.
    For a typical 25-PA start that meant slot 1 got 3 PAs and slot 9
    got 2 — a 50% premium. The empirical weights produce more like a
    35% premium (1.15 / 0.85), better matching real top-of-order
    advantage.

    Returns a list of integer PA counts of length n_slots, summing to n_pa.
    """
    base_weights = np.array(
        [1.15, 1.10, 1.08, 1.05, 1.02, 0.98, 0.95, 0.90, 0.85]
    )
    # Truncate or pad weights to actual lineup size (NL games can have <9)
    if n_slots <= len(base_weights):
        weights = base_weights[:n_slots]
    else:
        weights = np.concatenate([base_weights,
                                  np.full(n_slots - len(base_weights), 0.85)])
    weights = weights / weights.sum()

    # Fractional PAs per slot, then largest-remainder rounding so the
    # integer counts sum exactly to n_pa.
    fractional = weights * n_pa
    floors = np.floor(fractional).astype(int)
    remainder = n_pa - floors.sum()
    if remainder > 0:
        # Distribute the leftover PAs to the slots with largest fractional parts
        # (which, given the descending weights, will favor the top of the order)
        order = np.argsort(-(fractional - floors))
        for i in range(int(remainder)):
            floors[order[i]] += 1
    return floors.tolist()


def build_per_pa_pmf(latest, lineup, bf_pred, pp_model, pp_features):
    """Score per-PA model for each projected PA and convolve to a PMF.

    Allocates bf_pred PAs across the batting order using empirical PA-share
    weights (leadoff ~15% more PAs than 9-hole), then scores each projected
    PA with the per-PA XGBoost model and convolves via Poisson-Binomial.

    Batter features are pulled from `latest` which was populated by
    build_per_batter_live_features() for the actual posted lineup. Each
    lineup slot has stats stored as opp_b{pos}_k_rate, opp_b{pos}_k_rate_L10,
    opp_b{pos}_swstr_pct, etc. League-average fallbacks are only used when
    a specific batter has no historical data.

    Returns a length-26 PMF, or None if inputs are insufficient.
    """
    if pp_model is None or not lineup or bf_pred is None or bf_pred < 1:
        return None

    n_pa = max(1, min(int(round(float(bf_pred))), 40))

    p_throws = "R"
    if "p_throws" in latest.index and pd.notna(latest["p_throws"]):
        p_throws = str(latest["p_throws"]).upper()[:1]

    # League fallbacks — last-resort only when a batter has no historical data
    league_defaults = {
        "batter_k_rate_std":   0.22,  "batter_k_rate_L100":  0.22,  "batter_k_rate_prior": 0.22,
        "batter_bb_rate_std":  0.085, "batter_bb_rate_L100": 0.085, "batter_bb_rate_prior":0.085,
        "batter_hit_rate_std": 0.22,  "batter_hit_rate_L100":0.22,  "batter_hit_rate_prior":0.22,
        "batter_bip_rate_std": 0.68,  "batter_pa_prior": 400.0,
    }

    lineup_sorted = sorted(lineup, key=lambda b: b.get("lineup_position", 99))

    # Allocate the n_pa total PAs across slots using empirical weights.
    # pa_per_slot[i] = how many PAs to score for the batter in slot i
    # (0-indexed within lineup_sorted, so index 0 = leadoff).
    pa_per_slot = _allocate_pas_to_slots(n_pa, len(lineup_sorted))

    # Track which inning each PA represents — preserves the previous
    # (pa_idx // 3) + 1 mapping so the model gets a comparable inning signal.
    rows = []
    pa_idx_global = 0
    for slot_idx, n_pa_for_slot in enumerate(pa_per_slot):
        if n_pa_for_slot == 0:
            continue
        slot     = lineup_sorted[slot_idx]
        pos      = slot.get("lineup_position", slot_idx + 1)
        bat_side = str(slot.get("bat_side", "R")).upper()[:1]

        for _ in range(n_pa_for_slot):
            pa_idx = pa_idx_global  # for the inning calc below
            pa_idx_global += 1

            # ── Pull real batter stats from `latest` for this lineup slot ────
            # build_per_batter_live_features() stores per-slot stats as opp_b{pos}_*.
            def slot_val(stat, fallback):
                key = f"opp_b{pos}_{stat}"
                v = latest.get(key, None)
                if v is not None and pd.notna(v):
                    return float(v)
                return fallback

            batter_k_rate    = slot_val("k_rate",     league_defaults["batter_k_rate_std"])
            batter_k_rate_L  = slot_val("k_rate_L10", league_defaults["batter_k_rate_L100"])
            batter_bb_rate   = slot_val("bb_rate",     league_defaults["batter_bb_rate_std"])
            batter_bb_rate_L = slot_val("bb_rate_L10", league_defaults["batter_bb_rate_L100"])
            batter_hit_rate  = slot_val("hit_rate",    league_defaults["batter_hit_rate_std"])

            # bip_rate_std: use (1 - swstr_pct) as contact proxy if available,
            # otherwise estimate from K/BB rates.
            swstr = slot_val("swstr_pct", None)
            if swstr is not None:
                batter_bip = 1.0 - float(swstr)
            else:
                batter_bip = max(0.0, 1.0 - batter_k_rate - batter_bb_rate)
            batter_bip = max(0.30, min(0.90, batter_bip))  # clamp to sane range

            # Pitcher-side scalars used in interaction features
            pitcher_k     = float(smart_feature_get(latest, "k_pct_szn",    0.22))
            pitcher_whiff = float(smart_feature_get(latest, "whiff_pct_szn", 0.11))

            # ── Build the feature row for this PA ────────────────────────────
            row = {}
            for feat in pp_features:
                # Batter-side features — use real per-slot stats
                if feat == "batter_k_rate_std":
                    row[feat] = batter_k_rate
                elif feat == "batter_k_rate_L100":
                    row[feat] = batter_k_rate_L
                elif feat == "batter_k_rate_prior":
                    row[feat] = batter_k_rate          # best proxy for prior year
                elif feat == "batter_bb_rate_std":
                    row[feat] = batter_bb_rate
                elif feat == "batter_bb_rate_L100":
                    row[feat] = batter_bb_rate_L
                elif feat == "batter_bb_rate_prior":
                    row[feat] = batter_bb_rate
                elif feat == "batter_hit_rate_std":
                    row[feat] = batter_hit_rate
                elif feat == "batter_hit_rate_L100":
                    row[feat] = batter_hit_rate
                elif feat == "batter_hit_rate_prior":
                    row[feat] = batter_hit_rate
                elif feat == "batter_bip_rate_std":
                    row[feat] = batter_bip
                elif feat == "batter_pa_prior":
                    row[feat] = slot_val("pa_prior", 400.0)

                # Interaction features — now use real per-batter K rate
                elif feat == "ix_pitcher_k_x_batter_k":
                    row[feat] = pitcher_k * batter_k_rate
                elif feat == "ix_pitcher_k_x_batter_k_recent":
                    row[feat] = pitcher_k * batter_k_rate_L
                elif feat == "ix_whiff_x_contact":
                    row[feat] = pitcher_whiff * (1.0 - batter_bip)

                # Matchup context
                elif feat == "same_handed":
                    row[feat] = 1 if p_throws == bat_side else 0
                elif feat == "inning":
                    row[feat] = (pa_idx // 3) + 1

                # Pitcher-side and everything else — pull from latest
                elif feat in latest.index and pd.notna(latest[feat]):
                    row[feat] = latest[feat]
                else:
                    row[feat] = smart_feature_get(latest, feat, 0.0)

            rows.append(row)

    X = pd.DataFrame(rows, columns=pp_features).fillna(0.0).values
    try:
        probs = pp_model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"    (per-PA scoring failed: {e})")
        return None
    return poisson_binomial_pmf(probs, max_k=25)


def split_p_over(pmf_bb, pmf_pp, line):
    """Return P(K >= line) from EACH model separately.

    Returns (p_bb, p_pp). p_pp is None if per-PA model didn't fire.
    No blending — models are compared independently.
    """
    p_bb = float(pmf_bb[line:].sum()) if line < len(pmf_bb) else 0.0
    if pmf_pp is None:
        return p_bb, None
    p_pp = float(pmf_pp[line:].sum()) if line < len(pmf_pp) else 0.0
    return p_bb, p_pp


# ══════════════════════════════════════════════════════════════════════════════
# BETA-BINOMIAL MATH
# ══════════════════════════════════════════════════════════════════════════════

def beta_binom_pmf_array(n, alpha, beta_param):
    """Full PMF: P(K=0), ..., P(K=n)."""
    n = int(n)
    k = np.arange(n + 1)
    log_pmf = (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta_param)
        - betaln(alpha, beta_param)
    )
    pmf = np.exp(log_pmf)
    pmf /= pmf.sum()
    return pmf


def predict_beta_binom_pmf(p_hat, n_hat, kappa, sigma_n=0.0):
    """
    Given predicted rate p_hat, predicted batters faced n_hat,
    and concentration kappa, return the Beta-Binomial PMF.

    Marginalizes over N uncertainty using ±3σ_N to match the
    expected_pmf_over_N implementation used in training.
    """
    p_hat   = float(np.clip(p_hat, 0.01, 0.99))
    n_hat_f = float(max(n_hat, 1.0))

    alpha      = max(p_hat * kappa, 0.01)
    beta_param = max((1 - p_hat) * kappa, 0.01)

    if sigma_n > 0:
        min_N    = max(1, int(n_hat_f - 3 * sigma_n))
        max_N    = int(n_hat_f + 3 * sigma_n) + 1
        N_values = np.arange(min_N, max_N + 1)
        N_weights = sp_stats.norm.pdf(N_values, loc=n_hat_f, scale=max(sigma_n, 0.5))
        N_weights = N_weights / N_weights.sum()

        max_k        = int(max_N)
        combined_pmf = np.zeros(max_k + 1)
        for n_val, w in zip(N_values, N_weights):
            pmf = beta_binom_pmf_array(int(n_val), alpha, beta_param)
            combined_pmf[:len(pmf)] += w * pmf

        total = combined_pmf.sum()
        if total > 0:
            combined_pmf /= total
        return combined_pmf

    return beta_binom_pmf_array(int(round(n_hat_f)), alpha, beta_param)


# ══════════════════════════════════════════════════════════════════════════════
# MLB API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mlb_api_get(endpoint, params=None):
    url = f"{MLB_API_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ⚠ MLB API error: {e}")
        return None


def get_schedule(date_str):
    """Get today's games with probable pitchers."""
    data = mlb_api_get("schedule", {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team,linescore",
    })
    if not data or not data.get("dates"):
        return []

    games = []
    for game in data["dates"][0].get("games", []):
        state    = game.get("status", {}).get("abstractGameState", "")
        detailed = game.get("status", {}).get("detailedState", "")
        if state == "Final":
            continue

        away = game.get("teams", {}).get("away", {})
        home = game.get("teams", {}).get("home", {})

        away_pitcher = away.get("probablePitcher", {})
        home_pitcher = home.get("probablePitcher", {})

        games.append({
            "game_pk":           game["gamePk"],
            "game_time":         game.get("gameDate", ""),
            "game_status":       detailed,
            "away_team":         away.get("team", {}).get("abbreviation", ""),
            "home_team":         home.get("team", {}).get("abbreviation", ""),
            "away_pitcher_id":   away_pitcher.get("id"),
            "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
            "home_pitcher_id":   home_pitcher.get("id"),
            "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
        })
    return games


def check_lineup_posted(game_pk):
    """Check if lineups have been posted for a game."""
    data = mlb_api_get(f"game/{game_pk}/boxscore")
    if not data:
        return False
    away_batters = data.get("teams", {}).get("away", {}).get("batters", [])
    home_batters = data.get("teams", {}).get("home", {}).get("batters", [])
    return len(away_batters) >= 9 and len(home_batters) >= 9


def fetch_game_lineup(game_pk, side="away"):
    """
    Fetch the batting lineup for a specific game and side from the MLB API.
    Returns list of dicts with player_id, player_name, bat_side, lineup_position.
    """
    data = mlb_api_get(f"game/{game_pk}/boxscore")
    if not data:
        return []

    team_data     = data.get("teams", {}).get(side, {})
    batting_order = team_data.get("battingOrder", [])
    players       = team_data.get("players", {})

    lineup = []
    for pos_idx, player_id in enumerate(batting_order[:9], 1):
        pkey  = f"ID{player_id}"
        pinfo = players.get(pkey, {})
        person = pinfo.get("person", {})
        lineup.append({
            "player_id":       player_id,
            "player_name":     person.get("fullName", ""),
            "bat_side":        pinfo.get("batSide", {}).get("code", "R"),
            "lineup_position": pos_idx,
        })
    return lineup


def build_per_batter_live_features(lineup, pitcher_hand, features_df, fg_batting_path=None):
    """
    Build per-batter slot features for today's lineup at inference time.
    Uses historical lineup data from game_lineups.csv to look up each
    batter's stats, and FanGraphs batting data for advanced metrics.

    Returns dict of per-batter features to inject into the feature row.
    """
    if not lineup:
        return {}

    fg = None
    fg_path = fg_batting_path or Path("data/fangraphs_batting_seasons.csv")
    if (fg_path if isinstance(fg_path, Path) else Path(fg_path)).exists():
        fg = pd.read_csv(fg_path)

    lineups_path = Path("data/game_lineups.csv")
    lu_hist = None
    if lineups_path.exists():
        lu_hist = pd.read_csv(lineups_path)

    batter_stats = {}
    if lu_hist is not None:
        lu_hist = lu_hist.sort_values(["player_id", "game_pk"])
        for pid in [b["player_id"] for b in lineup]:
            batter_rows = lu_hist[lu_hist["player_id"] == pid]
            if len(batter_rows) == 0:
                continue
            total_ab = batter_rows["at_bats"].sum()
            total_k  = batter_rows["strikeouts"].sum()
            total_h  = batter_rows["hits"].sum()
            total_bb = batter_rows["walks"].sum()
            if total_ab > 0:
                batter_stats[pid] = {
                    "k_rate":   total_k / total_ab,
                    "hit_rate": total_h / total_ab,
                    "bb_rate":  total_bb / (total_ab + total_bb) if (total_ab + total_bb) > 0 else 0,
                }
                recent = batter_rows.tail(10)
                rab = recent["at_bats"].sum()
                if rab > 0:
                    batter_stats[pid]["k_rate_L10"] = recent["strikeouts"].sum() / rab
                    batter_stats[pid]["bb_rate_L10"] = recent["walks"].sum() / (rab + recent["walks"].sum())

    fg_lookup = {}
    if fg is not None:
        def normalize_name(name):
            return str(name).strip().lower().replace(".", "").replace(",", "") if pd.notna(name) else ""

        fg_latest = fg.sort_values("Season").groupby("Name").last().reset_index()
        fg_stat_cols = {
            "SwStr%": "swstr_pct", "Contact%": "contact_pct", "O-Swing%": "o_swing_pct",
            "Z-Contact%": "z_contact_pct", "K%+": "k_pct_plus", "wRC+": "wrc_plus",
            "xwOBA": "xwoba", "Barrel%": "barrel_pct", "HardHit%": "hard_hit_pct",
            "BB%": "bb_pct", "ISO": "iso", "BABIP": "babip", "CSW%": "csw_pct",
            "O-Contact%": "o_contact_pct", "TTO%": "tto_pct", "Hard%": "hard_pct",
            "Pull%": "pull_pct",
        }
        for _, row in fg_latest.iterrows():
            nm    = normalize_name(row["Name"])
            stats = {}
            for src, dst in fg_stat_cols.items():
                if src in row.index:
                    val = pd.to_numeric(row[src], errors="coerce")
                    if pd.notna(val):
                        stats[dst] = val
            fg_lookup[nm] = stats

    result   = {}
    k_rates  = []

    for batter in lineup:
        pos      = batter["lineup_position"]
        pid      = batter["player_id"]
        pname    = batter.get("player_name", "")
        bat_side = batter.get("bat_side", "R")
        prefix   = f"opp_b{pos}_"

        bstats = batter_stats.get(pid, {})
        result[prefix + "k_rate"]     = bstats.get("k_rate",     0.22)
        result[prefix + "k_rate_L10"] = bstats.get("k_rate_L10", 0.22)
        result[prefix + "bb_rate"]    = bstats.get("bb_rate",    0.08)
        result[prefix + "bb_rate_L10"]= bstats.get("bb_rate_L10",0.08)
        result[prefix + "hit_rate"]   = bstats.get("hit_rate",   0.25)

        kr = bstats.get("k_rate", np.nan)
        if pd.notna(kr):
            k_rates.append(kr)

        if bat_side == "S":
            result[prefix + "platoon_disadv"] = 0.0
        elif bat_side == pitcher_hand:
            result[prefix + "platoon_disadv"] = 1.0
        else:
            result[prefix + "platoon_disadv"] = 0.0

        nm_norm  = str(pname).strip().lower().replace(".", "").replace(",", "")
        fg_stats = fg_lookup.get(nm_norm, {})
        for dst_name, default in [
            ("swstr_pct", 0.11), ("contact_pct", 0.78), ("o_swing_pct", 0.30),
            ("z_contact_pct", 0.87), ("k_pct_plus", 100.0), ("wrc_plus", 100.0),
            ("xwoba", 0.320), ("barrel_pct", 0.07), ("hard_hit_pct", 0.35),
            ("bb_pct", 0.08), ("iso", 0.150), ("babip", 0.300), ("csw_pct", 0.30),
            ("o_contact_pct", 0.65), ("tto_pct", 0.33), ("hard_pct", 0.35),
            ("pull_pct", 0.40),
        ]:
            result[prefix + dst_name] = fg_stats.get(dst_name, default)

    if len(k_rates) >= 5:
        k_arr = np.array(k_rates)
        result["opp_lineup_k_rate_p90"]    = float(np.percentile(k_arr, 90))
        result["opp_lineup_k_rate_p10"]    = float(np.percentile(k_arr, 10))
        result["opp_lineup_k_rate_iqr"]    = float(np.percentile(k_arr, 75) - np.percentile(k_arr, 25))
        result["opp_lineup_k_rate_skew"]   = float(pd.Series(k_arr).skew())
        pa_weights = np.array([1.15, 1.10, 1.08, 1.05, 1.02, 0.98, 0.95, 0.90, 0.85])[:len(k_arr)]
        pa_weights = pa_weights / pa_weights.sum()
        result["opp_lineup_expected_k_total"] = float((k_arr * pa_weights).sum())

    platoon_vals = [result.get(f"opp_b{i}_platoon_disadv", 0) for i in range(1, 10)]
    result["opp_lineup_platoon_disadv_count"] = sum(v for v in platoon_vals if pd.notna(v))

    return result


def build_pitcher_batter_history_live(pitcher_id, lineup, features_df):
    """
    Build pitcher-batter history features at inference time.
    Looks at historical feature data to find previous games where this
    pitcher faced any of the batters in today's lineup.
    """
    if not lineup:
        return {}

    lineups_path = Path("data/game_lineups.csv")
    bpt_path     = Path("data/batter_pitch_type_all.csv")

    if not lineups_path.exists() or not bpt_path.exists():
        return {
            "pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
            "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0,
        }

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
        return {
            "pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
            "pb_hist_pa": 0, "pb_familiar_batters": 0, "pb_familiarity_pct": 0.0,
        }

    hist_bpt = bpt[
        (bpt["game_pk"].isin(pitcher_game_pks)) &
        (bpt["batter"].isin(today_batter_ids))
    ]

    if len(hist_bpt) == 0:
        familiar = hist_batters["player_id"].nunique()
        return {
            "pb_hist_k_rate": 0.22, "pb_hist_whiff_rate": 0.25,
            "pb_hist_pa": 0, "pb_familiar_batters": familiar,
            "pb_familiarity_pct": familiar / max(len(lineup), 1),
        }

    total_ks      = hist_bpt["bpt_strikeouts"].sum()
    total_pa      = hist_bpt["bpt_pa"].sum()
    total_whiffs  = hist_bpt["bpt_whiffs"].sum()
    total_pitches = hist_bpt["bpt_pitches_seen"].sum()
    familiar      = hist_batters["player_id"].nunique()

    return {
        "pb_hist_k_rate":      total_ks / total_pa if total_pa > 0 else 0.22,
        "pb_hist_whiff_rate":  total_whiffs / total_pitches if total_pitches > 0 else 0.25,
        "pb_hist_pa":          int(total_pa),
        "pb_familiar_batters": familiar,
        "pb_familiarity_pct":  familiar / max(len(lineup), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING (for daily predictions)
# ══════════════════════════════════════════════════════════════════════════════

def build_daily_features(pitcher_id, pitcher_name, features_df, opp_team=None,
                         opp_pitcher_id=None):
    """
    Build feature row for a pitcher using the latest data from
    pitcher_model_features.csv.

    Returns (feature_row, season_start_num).
    """
    pdf = features_df[features_df["pitcher_id"] == pitcher_id].copy()
    if pdf.empty:
        return None, 0

    pdf    = pdf.sort_values("game_date")
    latest = pdf.iloc[-1].copy()

    current_season = pdf["season"].max() if "season" in pdf.columns else None
    if current_season is not None:
        start_num = len(pdf[pdf["season"] == current_season])
    else:
        start_num = len(pdf)

    # ── On-the-fly blended-feature synthesis (backward compatibility) ──
    # If the features CSV was built before features.py's
    # build_blended_rate_features() existed, the _blended columns will be
    # missing. Synthesize them here from whatever IS available so the
    # per-PA and BB models see the stable features they were trained on.
    # This is a no-op if the CSV already has the blended columns.
    #
    # Uses the same empirical-Bayes blend as the training-time function:
    #   blended = (n_szn * szn_rate + PRIOR_N * prior_rate) / (n_szn + PRIOR_N)
    #
    # Priority for prior_rate: _prev10 > _prev5 > pitcher's own career mean
    # from pdf > league average default.
    def _synth_blend(szn_col, prev_col_10, prev_col_5, league_default,
                     denom_col, prior_n):
        """Compute a blended rate and write it back to `latest`.

        Only runs if the target _blended column is missing/NaN.
        """
        out_col = f"{szn_col}_blended"
        # If training pipeline already produced this column, trust it
        if out_col in latest.index and pd.notna(latest.get(out_col)):
            return
        szn_val = latest.get(szn_col)
        if pd.isna(szn_val):
            szn_val = league_default
        n_szn = latest.get(denom_col, 0.0)
        if pd.isna(n_szn):
            n_szn = 0.0
        # Find the best prior
        prior_val = None
        for pc in (prev_col_10, prev_col_5):
            if pc and pc in latest.index:
                v = latest.get(pc)
                if pd.notna(v):
                    prior_val = float(v)
                    break
        # Fallback: this pitcher's own historical mean in the features CSV
        if prior_val is None and szn_col in pdf.columns:
            # Use prior rows only (exclude today's row) — simulates shift(1).
            # pdf is already sorted by game_date; iloc[-1] is today.
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

    # Season-to-date rates — 100 PA prior for PA-backed rates,
    # 400 pitch prior for pitch-backed rates, 80 BIP prior for BIP-backed.
    _synth_blend("k_pct_szn",        "k_pct_prev10",        "k_pct_prev5",
                 0.22, "plate_appearances_szn", 100.0)
    _synth_blend("whiff_pct_szn",    "whiff_pct_prev10",    None,
                 0.25, "total_pitches_szn",     400.0)
    _synth_blend("csw_pct_szn",      "csw_pct_prev10",      None,
                 0.30, "total_pitches_szn",     400.0)
    _synth_blend("bb_pct_szn",       None,                  None,
                 0.085, "plate_appearances_szn", 100.0)
    _synth_blend("barrel_pct_szn",   "barrel_pct_prev10",   None,
                 0.07, "batted_balls_szn",       80.0)
    _synth_blend("hard_hit_pct_szn", "hard_hit_pct_prev10", None,
                 0.35, "batted_balls_szn",       80.0)
    _synth_blend("chase_rate_szn",   "chase_rate_prev10",   None,
                 0.30, "out_of_zone_pitches_szn", 400.0)

    # L5 blended — uses n_starts_in_L5 as the denom in "start units",
    # with a 5-start-equivalent prior. Same logic as 02's function.
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

    _synth_blend_l5("k_pct_L5",        "k_pct_prev10",        0.22)
    _synth_blend_l5("whiff_pct_L5",    "whiff_pct_prev10",    0.25)
    _synth_blend_l5("csw_pct_L5",      "csw_pct_prev10",      0.30)
    _synth_blend_l5("barrel_pct_L5",   "barrel_pct_prev10",   0.07)
    _synth_blend_l5("hard_hit_pct_L5", "hard_hit_pct_prev10", 0.35)

    # Ensure prior_starts_this_season exists — models trained on the
    # updated feature set use this. Derive from pdf if the column is missing.
    if ("prior_starts_this_season" not in latest.index or
            pd.isna(latest.get("prior_starts_this_season"))):
        if current_season is not None:
            # Prior starts = today's start number minus 1 (today is the
            # start being predicted, not yet played).
            latest["prior_starts_this_season"] = max(0, start_num - 1)
        else:
            latest["prior_starts_this_season"] = 0

    if opp_pitcher_id is not None:
        opp_pdf = features_df[features_df["pitcher_id"] == opp_pitcher_id].copy()
        if not opp_pdf.empty:
            opp_pdf    = opp_pdf.sort_values("game_date")
            opp_latest = opp_pdf.iloc[-1]

            opp_mapping = {
                "k_pct_L5":             "opp_sp_k_pct",
                "bb_pct_L5":            "opp_sp_bb_pct",
                "whiff_pct_L5":         "opp_sp_whiff_pct",
                "csw_pct_L5":           "opp_sp_csw_pct",
                "barrel_pct_L5":        "opp_sp_barrel_pct",
                "hard_hit_pct_L5":      "opp_sp_hard_hit_pct",
                "plate_appearances_L5": "opp_sp_bf_avg",
                "est_innings_L5":       "opp_sp_ip_avg",
                "is_short_outing_L5":   "opp_sp_short_pct",
                "outs_recorded_L5":     "opp_sp_outs_avg",
                "outs_per_pa_L5":       "opp_sp_outs_rate",
                "avg_velocity_L5":      "opp_sp_velo",
                "plate_appearances_L10":"opp_sp_bf_avg_L10",
                "est_innings_L10":      "opp_sp_ip_avg_L10",
                "is_short_outing_L10":  "opp_sp_short_pct_L10",
                "outs_recorded_L10":    "opp_sp_outs_avg_L10",
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
            for src, dst in opp_mapping.items():
                default = OPP_DEFAULTS.get(src, 0)
                val     = opp_latest.get(src, default)
                latest[dst] = val if pd.notna(val) else default

            bf_5     = latest.get("plate_appearances_L5", 22)
            opp_bf_5 = latest.get("opp_sp_bf_avg", 22)
            latest["ix_both_deep"]      = (bf_5 * opp_bf_5) / 500.0
            latest["ix_aces_matchup"]   = latest.get("k_pct_L5", 0) * latest.get("opp_sp_k_pct", 0)
            latest["ix_combined_depth"] = latest.get("est_innings_L5", 5) + latest.get("opp_sp_ip_avg", 5)

    return latest, start_num


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_probability_table(valid):
    """Print probability breakdown for all pitchers — both BB and Per-PA."""
    print(f"\n{'=' * 120}")
    print(f"  PROBABILITY BREAKDOWN  (BB = Beta-Binomial | PP = Per-PA XGBoost)")
    print(f"{'=' * 120}")
    print(f"  {'Pitcher':25s} "
          f"{'BB Pred':>7s} {'BB P(5+)':>9s} {'BB P(6+)':>9s} {'BB P(7+)':>9s}  "
          f"{'PP Pred':>7s} {'PP P(5+)':>9s} {'PP P(6+)':>9s} {'PP P(7+)':>9s}")
    print(f"  {'-' * 105}")

    for _, r in valid.sort_values("predicted_K", ascending=False).iterrows():
        pmf_bb = r["pmf"]
        pmf_pp = r.get("pmf_pp") if "pmf_pp" in r.index else None

        bb_pred = r["predicted_K"]
        bb_p5   = pmf_bb[5:].sum() if len(pmf_bb) > 5 else 0
        bb_p6   = pmf_bb[6:].sum() if len(pmf_bb) > 6 else 0
        bb_p7   = pmf_bb[7:].sum() if len(pmf_bb) > 7 else 0

        if pmf_pp is not None:
            pp_pred = sum(i * pmf_pp[i] for i in range(len(pmf_pp)))
            pp_p5   = pmf_pp[5:].sum() if len(pmf_pp) > 5 else 0
            pp_p6   = pmf_pp[6:].sum() if len(pmf_pp) > 6 else 0
            pp_p7   = pmf_pp[7:].sum() if len(pmf_pp) > 7 else 0
            pp_str  = f"{pp_pred:>7.1f} {pp_p5:>8.1%} {pp_p6:>8.1%} {pp_p7:>8.1%}"
        else:
            pp_str  = "   n/a (no lineup yet)"

        print(f"  {r['pitcher']:25s} "
              f"{bb_pred:>7.1f} {bb_p5:>8.1%} {bb_p6:>8.1%} {bb_p7:>8.1%}  "
              f"{pp_str}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def predict_slate(date_str):
    print(f"\n{'=' * 100}")
    print(f"  MLB PITCHER STRIKEOUT PREDICTIONS — {date_str}")
    print(f"  BB = Beta-Binomial  |  PP = Per-PA XGBoost (posted lineup)")
    print(f"{'=' * 100}")

    # ── Load BB models ───────────────────────────────────────────────────
    try:
        rate_model = joblib.load(MODEL_DIR / "rate_model.joblib")
        bf_model   = joblib.load(MODEL_DIR / "bf_model.joblib")
        with open(MODEL_DIR / "beta_binom_config.json") as f:
            bb_config = json.load(f)
        kappa    = bb_config["kappa"]
        sigma_n  = float(
            bb_config.get("sigma_n")
            or bb_config.get("sigma_n_global")
            or bb_config.get("sigma_N")
            or bb_config.get("sigma_N_global")
            or 0.0
        )
        rate_features = bb_config.get("rate_features", None)
        bf_features   = bb_config.get("bf_features",   None)
        bf_is_log     = bool(bb_config.get("bf_is_log", False))
        bf_log_sigma2 = float(bb_config.get("bf_log_sigma2", 0.0))
        print(f"  ✓ BB models loaded (κ={kappa:.1f}, σ_N={sigma_n:.1f}"
              f"{', BF on log scale' if bf_is_log else ''})")
    except Exception as e:
        print(f"  ✗ Could not load BB models: {e}")
        return

    # ── Load Per-PA model ────────────────────────────────────────────────
    pp_model, pp_features = load_per_pa_artifacts()
    if pp_model is not None:
        print(f"  ✓ Per-PA model loaded ({len(pp_features)} features) — "
              f"running both models side by side")
    else:
        print(f"  ⚠ Per-PA model not available — BB only")

    # ── Load feature data ────────────────────────────────────────────────
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
    print(f"  ✓ Feature data loaded ({len(features_df)} rows)")

    # ── Get today's schedule ─────────────────────────────────────────────
    games = get_schedule(date_str)
    if not games:
        print(f"\n  ℹ No upcoming games found for {date_str}.")
        return
    print(f"  ✓ {len(games)} games on schedule")

    # ── Build predictions ────────────────────────────────────────────────
    predictions = []
    for game in games:
        for side in ["away", "home"]:
            pid   = game[f"{side}_pitcher_id"]
            pname = game[f"{side}_pitcher_name"]
            if not pid or pname == "TBD":
                continue

            opp_team = game["home_team"] if side == "away" else game["away_team"]
            own_team = game["away_team"] if side == "away" else game["home_team"]
            opp_side = "home" if side == "away" else "away"
            opp_pid  = game.get(f"{opp_side}_pitcher_id")

            latest, start_num = build_daily_features(
                pid, pname, features_df, opp_team, opp_pitcher_id=opp_pid
            )
            if latest is None:
                predictions.append({
                    "pitcher": pname, "pitcher_id": pid, "team": own_team,
                    "opponent": opp_team, "game_pk": game["game_pk"],
                    "game_status": game.get("game_status", ""),
                    "predicted_K": None, "pmf": None, "pmf_pp": None,
                    "start_num": 0, "note": "No feature data",
                })
                continue

            # ── Inject per-batter lineup features for TODAY ──────────────
            opp_lineup_side = "home" if side == "away" else "away"
            lineup = fetch_game_lineup(game["game_pk"], side=opp_lineup_side)
            if lineup:
                p_throws = latest.get("p_throws", "R")
                if pd.isna(p_throws):
                    p_throws = "R"
                pb_features = build_per_batter_live_features(
                    lineup, str(p_throws), features_df
                )
                for k, v in pb_features.items():
                    latest[k] = v

                pb_hist = build_pitcher_batter_history_live(pid, lineup, features_df)
                for k, v in pb_hist.items():
                    latest[k] = v

            # ── Predict rate and batters faced ───────────────────────────
            try:
                if rate_features:
                    rf    = [smart_feature_get(latest, f) for f in rate_features]
                    p_hat = rate_model.predict([rf])[0]
                else:
                    p_hat = rate_model.predict(latest.to_frame().T)[0]

                if bf_features:
                    bf    = [smart_feature_get(latest, f) for f in bf_features]
                    n_hat = bf_model.predict([bf])[0]
                else:
                    n_hat = bf_model.predict(latest.to_frame().T)[0]

                if bf_is_log:
                    n_hat = float(np.exp(n_hat) * np.exp(bf_log_sigma2 / 2))
            except Exception as e:
                predictions.append({
                    "pitcher": pname, "pitcher_id": pid, "team": own_team,
                    "opponent": opp_team, "game_pk": game["game_pk"],
                    "game_status": game.get("game_status", ""),
                    "predicted_K": None, "pmf": None, "pmf_pp": None,
                    "start_num": start_num, "note": f"Prediction error: {e}",
                })
                continue

            p_hat = np.clip(p_hat, 0.01, 0.99)
            n_hat = max(n_hat, 1)

            # ── BB PMF ───────────────────────────────────────────────────
            pmf        = predict_beta_binom_pmf(p_hat, n_hat, kappa, sigma_n)
            predicted_k = sum(i * pmf[i] for i in range(len(pmf)))

            # ── Per-PA PMF ───────────────────────────────────────────────
            # Requires lineup to be posted. If not yet, pmf_pp stays None
            # and the PP column will show "n/a" in the output.
            pmf_pp = None
            try:
                if pp_model is not None and lineup:
                    pmf_pp = build_per_pa_pmf(
                        latest, lineup, n_hat, pp_model, pp_features
                    )
            except Exception as e:
                print(f"    (per-PA pmf failed for {pname}: {e})")
                pmf_pp = None

            # Days since this pitcher's last recorded start, measured to the
            # SLATE date. The features-CSV 'rest_days' column measures the gap
            # *before* the last start (and for a pitcher's first start of the
            # year it spans the whole offseason → clips to 21 → false layoff
            # flag), so we compute the real "days since last pitched" here.
            try:
                _days_since_last = (pd.to_datetime(date_str)
                                    - pd.to_datetime(latest.get("game_date"))).days
            except Exception:
                _days_since_last = None
            predictions.append({
                "pitcher": pname, "pitcher_id": pid, "team": own_team,
                "opponent": opp_team, "game_pk": game["game_pk"],
                "game_status": game.get("game_status", ""),
                "predicted_K": predicted_k, "pmf": pmf, "pmf_pp": pmf_pp,
                "start_num": start_num, "rate": p_hat, "n_hat": n_hat, "note": "",
                # Data-sufficiency signals (used for the THIN-DATA flag below)
                "n_starts_L5":           latest.get("n_starts_in_L5"),
                "days_since_last_start": _days_since_last,
            })

    pw    = pd.DataFrame(predictions)
    valid = pw[pw["predicted_K"].notna()]

    if valid.empty:
        print("\n  ℹ No predictions could be made (missing feature data).")
        return

    # ── Print prediction summary table ──────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  PREDICTIONS ({len(valid)} pitchers)")
    print(f"{'=' * 100}")
    print(f"  {'Pitcher':25s} {'Team':>4s} {'Opp':>4s} {'Pred K':>7s} "
          f"{'P(5+)':>7s} {'P(6+)':>7s} {'P(7+)':>7s} {'#':>3s}")
    print(f"  {'-' * 70}")

    for _, r in valid.sort_values("predicted_K", ascending=False).iterrows():
        pmf_r = r["pmf"]
        p5    = pmf_r[5:].sum() if len(pmf_r) > 5 else 0
        p6    = pmf_r[6:].sum() if len(pmf_r) > 6 else 0
        p7    = pmf_r[7:].sum() if len(pmf_r) > 7 else 0
        print(f"  {r['pitcher']:25s} {r['team']:>4s} {r['opponent']:>4s} "
              f"{r['predicted_K']:>6.1f}  {p5:>6.1%} {p6:>6.1%} {p7:>6.1%} "
              f"{r['start_num']:>3d}")

    # ── Threshold probability breakdown ──────────────────────────────────
    print_probability_table(valid)

    # ── Lineup status (PP predictions require a posted lineup) ───────────
    PRE_GAME_STATES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start", ""}
    pre_game = valid[valid["game_status"].isin(PRE_GAME_STATES)].copy()

    if len(pre_game) > 0:
        lineup_status = {}
        for _, r in pre_game.iterrows():
            gpk = r["game_pk"]
            if gpk not in lineup_status:
                lineup_status[gpk] = check_lineup_posted(gpk)

        n_posted  = sum(1 for v in lineup_status.values() if v)
        n_pending = len(lineup_status) - n_posted
        print(f"\n  Lineups: {n_posted} game(s) posted, {n_pending} pending")
        if n_pending > 0:
            print(f"  Rerun closer to first pitch for per-PA (PP) predictions "
                  f"on the pending games.")

    # ── Save predictions ─────────────────────────────────────────────────
    out_rows = []
    for _, r in valid.sort_values("predicted_K", ascending=False).iterrows():
        pmf_bb = r["pmf"]
        pmf_pp = r.get("pmf_pp") if "pmf_pp" in r.index else None

        row = {
            "pitcher":     r["pitcher"],
            "team":        r["team"],
            "opponent":    r["opponent"],
            "game_status": r["game_status"],
            "start_num":   r["start_num"],
            "bb_pred_K":   round(float(r["predicted_K"]), 2),
        }
        for k in range(3, 13):
            row[f"bb_P(K>={k})"] = round(
                float(pmf_bb[k:].sum()) if len(pmf_bb) > k else 0.0, 4)

        if pmf_pp is not None:
            row["pp_pred_K"] = round(
                float(sum(i * pmf_pp[i] for i in range(len(pmf_pp)))), 2)
            for k in range(3, 13):
                row[f"pp_P(K>={k})"] = round(
                    float(pmf_pp[k:].sum()) if len(pmf_pp) > k else 0.0, 4)

        # Flag predictions the Beta-Binomial is likely to be overconfident on:
        # thin recent data means it leans on the career/league prior.
        if FLAG_THIN_DATA:
            ok, reasons = data_sufficiency_flag(
                r.get("n_starts_L5"), r.get("start_num"),
                r.get("days_since_last_start"))
            row["data_sufficient"] = ok
            row["data_flags"]      = "; ".join(reasons)

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df["run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = OUTPUT_DIR / f"strikeouts_{date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(out_df)} pitcher predictions to {out_path}")

    if FLAG_THIN_DATA and "data_sufficient" in out_df.columns:
        n_thin = int((~out_df["data_sufficient"]).sum())
        if n_thin:
            print(f"  ⚠ {n_thin} prediction(s) flagged THIN DATA — the "
                  f"Beta-Binomial leans on its prior for these, so treat the "
                  f"probabilities as low-confidence.")



# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    predict_slate(date_str)
