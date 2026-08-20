"""
predict/earned_runs.py
==============
Daily predictions for earned runs allowed — two models side by side.

BB  = Beta-Binomial (from train/earned_runs.py)
PP  = Per-PA XGBoost (from train/earned_runs_per_pa.py)
      Scores P(run_proxy) for each projected PA using the actual posted
      lineup with real per-batter features, sums via Poisson-Binomial,
      then scales by earned_runs_fraction (~0.87) to convert raw run
      probability to expected earned runs.

NOTE ON THE PP MODEL FOR ER:
  Earned runs are not a clean per-PA binary outcome. The per-PA model
  uses a run-scoring proxy (was_run_scored from bat_score columns if
  available, or HR/2B/3B proxy otherwise). The Poisson-Binomial sum
  approximates raw runs; multiply by er_fraction to get expected ER.
  This is an approximation — sequencing effects (runners on base) are
  not captured. Use BB as the primary model and PP as a secondary signal.

USAGE:
    python run.py predict earned-runs               # Today
    python run.py predict earned-runs 2026-04-15    # Specific date

REQUIRES:
    - models/er_rate_model.joblib, er_config.json
    - models/bf_model.joblib, beta_binom_config.json
    - models/per_pa_er_model.joblib, per_pa_er_config.json  (optional)
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


def expected_pmf_over_N(pred_p, pred_N, kappa, sigma_N, max_k=15):
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


def poisson_binomial_pmf(probs, max_k=15):
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


def scale_pmf_by_fraction(pmf, fraction):
    """
    Scale a Poisson-Binomial (raw runs) PMF to an earned-runs PMF.

    Each raw-run outcome k is mapped to expected ER = k × fraction.
    We convolve this into an ER PMF by assigning probability mass
    proportionally across integer ER values.
    """
    max_k  = len(pmf) - 1
    er_pmf = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        er_expected = k * fraction
        lo = int(er_expected)
        hi = lo + 1
        frac_hi = er_expected - lo
        if lo <= max_k:
            er_pmf[lo] += pmf[k] * (1 - frac_hi)
        if hi <= max_k:
            er_pmf[hi] += pmf[k] * frac_hi
    total = er_pmf.sum()
    if total > 0:
        er_pmf /= total
    return er_pmf


def pmf_to_over_probs(pmf, lines):
    """Return dict of line → P(outcome >= line) for a list of integer lines."""
    return {
        l: float(pmf[l:].sum()) if l < len(pmf) else 0.0
        for l in lines
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-PA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_per_pa_model():
    try:
        model = joblib.load(MODEL_DIR / "per_pa_er_model.joblib")
        with open(MODEL_DIR / "per_pa_er_config.json") as f:
            cfg = json.load(f)
        er_fraction = float(cfg.get("earned_runs_fraction", 0.87))
        print(f"  ✓ Per-PA ER model loaded ({len(cfg['features'])} features, "
              f"ER fraction={er_fraction:.2f})")
        return model, cfg["features"], er_fraction
    except Exception as e:
        print(f"  (per-PA ER model not loaded: {e})")
        return None, None, 0.87


def build_per_pa_pmf(latest, lineup, bf_pred, pp_model, pp_features, er_fraction):
    """Score per-PA ER proxy model and convolve to a scaled ER PMF."""
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
        "batter_run_rate_std":  0.04,  "batter_run_rate_L100": 0.04,
        "batter_run_rate_prior":0.04,
        "batter_bip_rate_std":  0.68,  "batter_bip_rate_prior":0.68,
        "batter_pa_prior": 400.0,
    }

    er_col     = next((c for c in ["er_per_pa_szn", "er_per_pa_L5"] if c in latest.index
                       and pd.notna(latest.get(c))), None)
    pitcher_er = float(latest.get(er_col, 0.04)) if er_col else 0.04
    pitcher_k  = float(smart_feature_get(latest, "k_pct_szn",      0.22))
    pitcher_bb = float(smart_feature_get(latest, "bb_pct_szn",      0.085))
    pitcher_barrel = float(smart_feature_get(latest, "barrel_pct_szn", 0.07))

    lineup_sorted = sorted(lineup, key=lambda b: b.get("lineup_position", 99))
    rows = []
    for pa_idx in range(n_pa):
        slot     = lineup_sorted[pa_idx % len(lineup_sorted)]
        pos      = slot.get("lineup_position", (pa_idx % 9) + 1)
        bat_side = str(slot.get("bat_side", "R")).upper()[:1]

        def slot_val(stat, fallback):
            key = f"opp_b{pos}_{stat}"
            v   = latest.get(key, None)
            if v is not None and pd.notna(v):
                return float(v)
            return fallback

        batter_k_rate    = slot_val("k_rate",     league_defaults["batter_k_rate_std"])
        batter_k_rate_L  = slot_val("k_rate_L10", league_defaults["batter_k_rate_L100"])
        batter_bb_rate   = slot_val("bb_rate",     league_defaults["batter_bb_rate_std"])
        batter_bb_rate_L = slot_val("bb_rate_L10", league_defaults["batter_bb_rate_L100"])
        batter_hit_rate  = slot_val("hit_rate",    league_defaults["batter_hit_rate_std"])

        # Run proxy: use xwOBA as run-scoring signal if available
        batter_xwoba = slot_val("xwoba", 0.320)
        # Normalize xwOBA to a ~run-scoring rate (league ~0.320 → ~4% run rate)
        batter_run_proxy = max(0.0, (batter_xwoba - 0.200) / (0.500 - 0.200) * 0.08)

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
                row[feat] = batter_bb_rate_L
            elif feat == "batter_bb_rate_prior":
                row[feat] = batter_bb_rate
            elif feat == "batter_hit_rate_std":
                row[feat] = batter_hit_rate
            elif feat == "batter_hit_rate_L100":
                row[feat] = batter_hit_rate
            elif feat == "batter_hit_rate_prior":
                row[feat] = batter_hit_rate
            elif feat == "batter_run_rate_std":
                row[feat] = batter_run_proxy
            elif feat == "batter_run_rate_L100":
                row[feat] = batter_run_proxy
            elif feat == "batter_run_rate_prior":
                row[feat] = batter_run_proxy
            elif feat == "batter_bip_rate_std":
                row[feat] = batter_bip
            elif feat == "batter_bip_rate_prior":
                row[feat] = batter_bip
            elif feat == "batter_pa_prior":
                row[feat] = slot_val("pa_prior", 400.0)
            elif feat == "ix_pitcher_k_x_batter_k":
                row[feat] = pitcher_k * batter_k_rate
            elif feat == "ix_pitcher_er_x_batter_run":
                row[feat] = pitcher_er * batter_run_proxy
            elif feat == "ix_pitcher_er_x_batter_run_recent":
                row[feat] = pitcher_er * batter_run_proxy
            elif feat == "ix_barrel_x_hit":
                row[feat] = pitcher_barrel * batter_hit_rate
            elif feat == "ix_pitcher_bb_x_batter_bb":
                row[feat] = pitcher_bb * batter_bb_rate
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
        print(f"    (per-PA ER scoring failed: {e})")
        return None

    # Raw run PMF → scale to ER PMF
    raw_pmf = poisson_binomial_pmf(probs, max_k=15)
    er_pmf  = scale_pmf_by_fraction(raw_pmf, er_fraction)
    return er_pmf


# ══════════════════════════════════════════════════════════════════════════════
# MLB API / HELPERS (same pattern as 13/11/15)
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
                    batter_stats[pid]["k_rate_L10"]   = rec["strikeouts"].sum() / rab
                    batter_stats[pid]["bb_rate_L10"]  = (
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
    print(f"\n{'=' * 100}")
    print(f"  MLB PITCHER EARNED RUNS PREDICTIONS — {date_str}")
    print(f"  BB = Beta-Binomial  |  PP = Per-PA XGBoost (run-scoring proxy, scaled by ER fraction)")
    print(f"{'=' * 100}")

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

    # Load BB ER model
    try:
        er_model = joblib.load(MODEL_DIR / "er_rate_model.joblib")
        with open(MODEL_DIR / "er_config.json") as f:
            er_config = json.load(f)
        er_sigma    = float(er_config.get("sigma_n") or er_config.get("sigma_n_global") or
                            er_config.get("sigma_N") or er_config.get("sigma_N_global") or sigma_n_bf)
        er_features = er_config["rate_features"]
        er_kappa    = er_config["kappa"]
        print(f"  ✓ BB ER model loaded (κ={er_kappa:.1f}, σ_N={er_sigma:.2f})")
    except Exception as e:
        print(f"  ✗ Could not load ER model: {e}")
        return

    # Load per-PA model (optional)
    pp_model, pp_features, er_fraction = load_per_pa_model()

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
                feat_vec = [smart_feature_get(latest, f) for f in er_features]
                p_hat_bb = np.clip(er_model.predict([feat_vec])[0], 0.01, 0.99)
            except Exception:
                continue

            pmf_bb = expected_pmf_over_N(p_hat_bb, n_hat, er_kappa, er_sigma, max_k=15)
            exp_bb = pmf_expected(pmf_bb)

            row = {
                "pitcher": pname, "team": own_team, "opponent": opp_team,
                "N_hat": round(n_hat, 1), "start_num": start_num,
                "bb_er_rate": round(p_hat_bb, 4),
                "bb_er_pred": round(exp_bb, 1),
            }

            for line in range(max(0, int(exp_bb) - 2), int(exp_bb) + 5):
                p_over = pmf_bb[line:].sum() if line < len(pmf_bb) else 0
                row[f"bb_P(ER>={line})"] = round(float(p_over), 3)

            half_center = round(exp_bb - 0.5) + 0.5
            half_lines  = [half_center + d for d in [-1, 0, 1]]
            while half_lines[0] < 0.5:
                half_lines = [h + 1 for h in half_lines]
            for hl in half_lines:
                over_k  = int(hl) + 1
                p_over  = float(pmf_bb[over_k:].sum()) if over_k < len(pmf_bb) else 0.0
                p_under = 1.0 - p_over
                row[f"ER{hl:g}_bb_over_prob"]      = round(p_over,  3)
                row[f"ER{hl:g}_bb_under_prob"]     = round(p_under, 3)

            # ── PP prediction ─────────────────────────────────────────────
            if lineup and pp_model is not None:
                try:
                    pmf_pp = build_per_pa_pmf(
                        latest, lineup, n_hat, pp_model, pp_features, er_fraction
                    )
                    if pmf_pp is not None:
                        exp_pp = pmf_expected(pmf_pp)
                        row["pp_er_pred"] = round(exp_pp, 1)

                        for line in range(max(0, int(exp_pp) - 2), int(exp_pp) + 5):
                            p_over = pmf_pp[line:].sum() if line < len(pmf_pp) else 0
                            row[f"pp_P(ER>={line})"] = round(float(p_over), 3)

                        half_center = round(exp_pp - 0.5) + 0.5
                        half_lines_pp = [half_center + d for d in [-1, 0, 1]]
                        while half_lines_pp[0] < 0.5:
                            half_lines_pp = [h + 1 for h in half_lines_pp]
                        for hl in half_lines_pp:
                            over_k  = int(hl) + 1
                            p_over  = float(pmf_pp[over_k:].sum()) if over_k < len(pmf_pp) else 0.0
                            p_under = 1.0 - p_over
                            row[f"ER{hl:g}_pp_over_prob"]      = round(p_over,  3)
                            row[f"ER{hl:g}_pp_under_prob"]     = round(p_under, 3)
                except Exception as e:
                    print(f"    (per-PA ER failed for {pname}: {e})")

            predictions.append(row)

    if not predictions:
        print("  ℹ No predictions generated.")
        if skipped:
            print(f"  ⚠ {len(skipped)} skipped:")
            for s in skipped[:10]:
                print(f"    - {s}")
        return

    pdf     = pd.DataFrame(predictions)
    has_pp  = "pp_er_pred" in pdf.columns and pdf["pp_er_pred"].notna().any()

    # ── Side-by-side summary table ────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  EARNED RUNS PREDICTIONS  (BB = Beta-Binomial | PP = Per-PA XGBoost)")
    if pp_model is not None:
        print(f"  PP NOTE: ER proxy model — run-scoring events scaled by "
              f"ER fraction ({er_fraction:.2f})")
    print(f"{'=' * 110}")

    bb_prob_cols = sorted([c for c in pdf.columns if c.startswith("bb_P(ER>=")])
    pp_prob_cols = sorted([c for c in pdf.columns if c.startswith("pp_P(ER>=")])

    hdr = f"  {'Pitcher':25s} {'Team':>4s} {'Opp':>4s} {'BB ER':>6s}"
    for c in bb_prob_cols[:4]:
        hdr += f" {c[3:]:>10s}"
    if has_pp:
        hdr += f"  {'PP ER':>6s}"
        for c in pp_prob_cols[:4]:
            hdr += f" {c[3:]:>10s}"
    print(hdr)
    print(f"  {'-' * 110}")

    for _, r in pdf.sort_values("bb_er_pred", ascending=False).iterrows():
        line = f"  {r['pitcher']:25s} {r['team']:>4s} {r['opponent']:>4s} {r['bb_er_pred']:>6.1f}"
        for c in bb_prob_cols[:4]:
            v = r.get(c, np.nan)
            line += f" {v:>9.1%}" if pd.notna(v) else f" {'—':>9s}"
        if has_pp:
            pp_pred = r.get("pp_er_pred", np.nan)
            if pd.notna(pp_pred):
                line += f"  {pp_pred:>6.1f}"
                for c in pp_prob_cols[:4]:
                    v = r.get(c, np.nan)
                    line += f" {v:>9.1%}" if pd.notna(v) else f" {'—':>9s}"
            else:
                line += "  (no lineup yet)"
        print(line)

    # ── Threshold probability ladders ─────────────────────────────────────
    for model_tag, pred_col in ([("BB", "bb_er_pred")] +
                                  ([("PP", "pp_er_pred")] if has_pp else [])):
        print(f"\n  THRESHOLD PROBABILITIES — EARNED RUNS [{model_tag} MODEL]")
        print(f"  {'-' * 80}")
        hdr = f"  {'Pitcher':25s} {'ER':>5s}"
        for slot in ["L1", "L2", "L3"]:
            hdr += f"  {slot+' line':>7s} {'P(over)':>8s} {'P(under)':>8s}"
        print(hdr)
        print(f"  {'-' * 80}")

        tag = model_tag.lower()
        for _, r in pdf.sort_values(pred_col, ascending=False).iterrows():
            pred_val = r.get(pred_col, np.nan)
            if pd.isna(pred_val):
                continue
            half_center = round(pred_val - 0.5) + 0.5
            half_lines  = [half_center + d for d in [-1, 0, 1]]
            while half_lines[0] < 0.5:
                half_lines = [h + 1 for h in half_lines]
            row_line = f"  {r['pitcher']:25s} {pred_val:>5.1f}"
            for hl in half_lines:
                key = f"ER{hl:g}_{tag}"
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

    out_path = OUTPUT_DIR / f"er_{date_str}.csv"
    pdf.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved to {out_path}")
    if has_pp:
        print(f"    BB columns: bb_er_pred, bb_P(ER>=N), ER*_bb_*")
        print(f"    PP columns: pp_er_pred, pp_P(ER>=N), ER*_pp_*  (when lineup posted)")


if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    predict_slate(date_str)
