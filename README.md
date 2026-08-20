# MLB Pitcher Outcome Model

Predicts **full probability distributions** over what a starting pitcher will do
tonight — strikeouts, hits allowed, walks, outs recorded, and earned runs — from
1,272 engineered features spanning six seasons of Statcast, MLB Stats API,
FanGraphs, weather, park, umpire, and catcher data.

The point is the distribution, not just the point estimate. "Gausman is projected to get
5.6 strikeouts" is less useful than knowing P(K ≥ 5) = 66%, P(K ≥ 6) = 49%,
P(K ≥ 7) = 33%, with the correct distribution shape. A mean
with an implied Normal around it gets that shape wrong.

```
  Pitcher                   BB Pred  BB P(5+)  BB P(6+)  BB P(7+)  PP Pred  PP P(5+)  PP P(6+)  PP P(7+)
  ---------------------------------------------------------------------------------------------------------
  Jacob deGrom                  6.1    74.1%    58.2%    41.7%     n/a (no lineup yet)
  Taj Bradley                   5.9    70.1%    53.5%    37.2%      5.5    67.9%    48.6%    30.3%
  Kevin Gausman                 5.6    66.3%    49.1%    33.0%      4.9    56.6%    36.7%    20.4%
```

## Modeling approach

Every stat is modeled **twice**, by two methods that fail in different ways. Both
run independently and are reported side by side, so disagreement stays visible
rather than getting averaged away behind a blend.

### 1. Beta-Binomial (start level)

Rather than regressing on the raw count, this decomposes the outcome into its two
generative parts and models each separately:

```
p = stat / PA        per-plate-appearance rate     (XGBoost)
N = batters faced    workload                      (XGBoost, log scale)

outcome ~ BetaBinomial(N, α, β)      where α = p·κ, β = (1-p)·κ
```

Why this shape rather than a Normal:

- Counts are **discrete and bounded** by batters faced. A Normal puts mass on
  7.3 strikeouts and on outcomes exceeding the PAs actually thrown.
- Real starts are **overdispersed** relative to a Binomial — the same pitcher
  against the same lineup varies more than fixed-p sampling allows. The Beta
  prior absorbs exactly that, with concentration κ fit per stat.
- Uncertainty in **N itself** matters, so the final PMF is marginalized over a
  Normal prior on batters faced (σ_N ≈ 3.2).

Fitted concentrations, which encode how much extra dispersion each stat carries
(lower κ = more overdispersed):

| Stat | κ | Test MAE | Normal-baseline σ | Mean |
|---|---|---|---|---|
| Strikeouts | 165.7 | — | — | — |
| Hits | 487.3 | 1.57 | 1.95 | 4.85 |
| Walks | 500.0 | 0.99 | 1.24 | 1.74 |
| Outs | 200.0 | 1.75 | 2.16 | 15.09 |
| Earned runs | 23.3 | 1.57 | 1.94 | 2.39 |

Earned runs stand out at κ = 23 — an order of magnitude more overdispersed than
hits or walks. That is the sequencing problem showing up in the fit: ER depends
on when baserunners arrive, not just how many, so a per-PA rate model captures
much less of it. ER probabilities are the weakest of the five.

### 2. Per-PA XGBoost + Poisson-Binomial (plate-appearance level)

The Beta-Binomial assumes every PA in a start carries the same probability, with
all within-game variation absorbed into κ. That's the wrong shape: a lineup with
three high-strikeout hitters and six contact hitters produces a
**fatter-shouldered** distribution than a uniform lineup with the same mean.

So the second model scores each projected plate appearance individually against
that batter's real features, then combines the resulting Bernoullis:

```
P(K) per PA  →  XGBoost on (pitcher × batter × context) features
PMF          →  Poisson-Binomial convolution over all projected PAs
```

Trained on true PA-level binary outcomes from Statcast (`was_K`, `was_hit`,
`was_BB`, `was_out`), not on season rates as soft labels — so it can learn actual
matchup interactions rather than being capped by season-rate coupling.

The tradeoff: it needs the **posted lineup**, which lands ~2–4 hours before first
pitch. Before that, only the Beta-Binomial can run. That's why the output above
shows `n/a (no lineup yet)` for some pitchers.

## Feature engineering

`features.py` builds the 1,272-column table that every model reads.
All features are lagged with `shift(1)` — a row for a given start contains only
information available *before* that start.

- **Rolling performance** — 3 / 5 / 10-start windows, plus season-to-date
- **Empirical-Bayes blended rates** — early-season rates shrunk toward league
  mean by sample size, so a pitcher's April 2nd start doesn't inherit a wild
  small-sample K rate
- **Pitch mix & stuff** — per-pitch-type usage, velocity, whiff; FanGraphs
  Stuff+ / Location+ / Pitching+
- **Opposing lineup** — per-slot batter quality (`b1_` … `b9_`), K rate spread
  across the lineup, top-3 vs bottom-3 gap
- **Platoon splits** — L/R matchup deltas
- **Catcher** — framing plus game-calling tendencies, built leak-free from
  pitch-level data (as-of features use only prior-date pitches)
- **Umpire** — strike-zone tendencies by home plate umpire
- **Park & weather** — Statcast park factors, Open-Meteo conditions, dome flags,
  wind decomposed relative to center field bearing
- **Pitcher×batter history** — prior matchup results and lineup familiarity
- **Fatigue** — pitch counts, rest days, pitches per out in recent starts

## Pipeline

Every command goes through `run.py`. Run `python run.py` to see all targets.

### One-time / full rebuild

```bash
python run.py collect statcast        # Statcast + MLB API -> data/ (slow, hours)
python run.py collect fangraphs       # FanGraphs merge (manual step, see below)
python run.py collect catcher         # catcher game-calling features
python run.py collect batters         # per-batter rolling stats
python run.py collect weather         # Open-Meteo historical weather
python run.py collect park-factors    # Baseball Savant park factors
python run.py features                # -> data/pitcher_model_features.csv
```

### Training

```bash
python run.py train-all              # everything below, in order
```

Or individually:

```bash
python run.py train baseline             # point-estimate baseline + diagnostics
python run.py train strikeouts           # K rate + batters-faced models
python run.py train strikeouts-per-pa
python run.py train hits-walks           # hits + walks rate models
python run.py train hits-walks-per-pa
python run.py train outs                 # outs rate model
python run.py train outs-per-pa
python run.py train earned-runs          # earned runs rate model
python run.py train earned-runs-per-pa
```

### Daily use

```bash
python run.py refresh                # incremental refresh + feature rebuild
python run.py predict-all            # all four stats
```

Or one at a time:

```bash
python run.py predict strikeouts
python run.py predict hits-walks
python run.py predict outs
python run.py predict earned-runs
```

Each accepts an optional date: `python run.py predict outs 2026-06-15`.
Arguments pass straight through, so `python run.py refresh --days 60 --force`
works too.

After a long gap, two caches need an explicit nudge — `refresh` doesn't rebuild
them, and neither errors when stale:

```bash
python run.py collect catcher
python run.py collect batters --force
python run.py features
```

If the FanGraphs data is also stale, refresh it (see below) **before** running
`features.py`, since that's what reads it.

## Retraining

You almost never need to. The daily refresh and retraining do different jobs,
and only the first has to happen often:

- **The refresh updates feature *values*** — a pitcher's last-5-start K rate,
  the opposing lineup's current chase rate, tonight's weather. The model sees
  current form automatically. This is the daily job.
- **Retraining updates the *mapping*** — how much last-5 K rate should count
  toward tonight's prediction. That only moves when baseball moves: rule
  changes, ball construction, league-wide offensive environment. Season
  timescale, not weekly.

Every model splits on `TEST_YEAR = 2025`: **train on 2021–2024, evaluate on
2025 onward**. In-season refreshes add rows to the *test* side only, so the
training window is already complete — retraining mid-season reproduces the
same weights and burns hours doing it.

The retrain worth scheduling is at the season boundary: roll `TEST_YEAR`
forward (to 2026 once 2026 is complete), which gives five training seasons and
a clean untouched holdout, then re-run the training scripts above.

Resist the temptation to train on everything to squeeze out accuracy. The
time-based holdout is what makes the evaluation numbers mean anything.

### Evaluation

```bash
python run.py evaluate per-pa        # per-PA vs Beta-Binomial on log-loss
python run.py evaluate features      # feature importance / leakage check
```

## Setup

```bash
pip install -r requirements.txt
```

Developed on Python 3.13.

## Data

`data/`, `models/`, `output/` are **not** in version control — the raw Statcast
pitch table alone is ~575 MB, well past GitHub's 100 MB per-file limit. The full
local footprint is roughly 1.6 GB.

Everything regenerates from public sources via the pipeline above. Sources:

| Source | Access | Contents |
|---|---|---|
| Statcast (via `pybaseball`) | automated | pitch-level data, 2021– |
| MLB Stats API | automated | schedules, lineups, umpires, venues, catchers |
| Open-Meteo Archive | automated, no key | historical weather per venue |
| Baseball Savant | automated | park factors |
| FanGraphs | **manual** | season stats, Stuff+/Location+/Pitching+ |

### The FanGraphs manual step

FanGraphs sits behind Cloudflare and blocks automated requests, so these exports
have to be downloaded by hand. Run:

```bash
python run.py collect fangraphs
```

With no new files present it prints four URLs. Open each, click **Export Data**,
and save into `data/` with these exact names:

| # | Export | Filename |
|---|---|---|
| 1 | Main pitching (K%, SwStr%, plate discipline) | `data/fg_pitching_<YEAR>.csv` |
| 2 | Pitch model (Stuff+, Location+, Pitching+) | `data/fg_pitching_pitchmodel_<YEAR>.csv` |
| 3 | Main batting (K%, BB%, wRC+) | `data/fg_batting_<YEAR>.csv` |
| 4 | Statcast batting (Barrel%, xwOBA, xBA) | `data/fg_batting_statcast_<YEAR>.csv` |

Then re-run `python run.py collect fangraphs`. It joins the supplements onto the main
files by `PlayerId` and replaces that season's rows in
`data/fangraphs_*_seasons.csv`.

Two things worth knowing:

- **Filenames use underscores.** `fg_pitching_pitchmodel.2026.csv` (a dot) will
  not match the glob and gets silently skipped.
- **These are season-to-date snapshots, not append-only.** Re-exporting mid-season
  *replaces* that season's rows rather than adding to them, so the file has to be
  refreshed periodically or the season stats freeze at whatever date you last
  pulled. In practice this is the piece most likely to go stale.

## Repository layout

```
run.py                        single entry point (see `python run.py` for all commands)
pitcher_model/
├── paths.py                  data/models/output locations, resolved from repo root
├── features.py               builds the 1,272-column modeling table
├── refresh.py                incremental daily refresh orchestrator
├── collect/
│   ├── statcast.py           Statcast pitches + MLB Stats API (games, lineups,
│   │                         umpires, catchers, earned runs)
│   ├── fangraphs.py          FanGraphs manual-export merge
│   ├── catcher.py            catcher framing + game-calling features
│   ├── batters.py            per-batter rolling stats (--force to rebuild)
│   ├── weather.py            Open-Meteo historical weather
│   └── park_factors.py       Baseball Savant park factors
├── train/
│   ├── baseline.py           point-estimate regression + diagnostics
│   ├── strikeouts.py         K rate + batters-faced models (shared by all stats)
│   ├── strikeouts_per_pa.py  per-PA strikeout model
│   ├── hits_walks.py         hits + walks rate models (shared fitting helpers)
│   ├── hits_walks_per_pa.py  per-PA hits + walks models
│   ├── outs.py               outs rate model
│   ├── outs_per_pa.py        per-PA outs model
│   ├── earned_runs.py        earned runs rate model
│   └── earned_runs_per_pa.py per-PA earned runs proxy model
├── predict/
│   ├── strikeouts.py         daily strikeouts
│   ├── hits_walks.py         daily hits + walks
│   ├── outs.py               daily outs recorded
│   └── earned_runs.py        daily earned runs
└── evaluate/
    ├── per_pa.py             per-PA vs Beta-Binomial on log-loss
    └── feature_importance.py feature importance / leakage check
```

Two modules are shared rather than duplicated: `train/strikeouts.py` owns the
collinearity pruner, and `train/hits_walks.py` owns the Beta-Binomial fitting
and evaluation helpers that `train/outs.py` and `train/earned_runs.py` import.
Both gate their training behind `__main__`, so importing them runs no work.

`paths.py` resolves `data/`, `models/`, and `output/` from the repository root
rather than the working directory, so any command works from anywhere:

```bash
cd /anywhere && python /path/to/repo/run.py predict outs
```

## Known limitations

- **Earned runs are weakly modeled.** ER is driven by sequencing, which a per-PA
  rate model can't see. κ = 23 reflects this honestly. The per-PA ER model trains
  on a run-scoring *proxy* because Statcast has no PA-level "was_ER" outcome.
- **The Beta-Binomial gets overconfident on thin data.** With few recent starts it
  falls back on the career/league prior while still reporting a confident-looking
  probability. `predict/strikeouts.py` flags these rows (`data_sufficient`,
  `data_flags`) rather than dropping them.
- **Per-PA predictions need a posted lineup**, so they're unavailable until
  roughly 2–4 hours before first pitch.
- **Some caches go stale silently.** The FanGraphs season snapshot and the
  `01c` / `collect/batters.py` outputs are all read without a freshness check, so nothing
  errors if they're months old — the features just quietly drift. `collect/batters.py` now
  prints the date its cache is frozen at; the others don't. Refresh them
  explicitly after any long gap (see *Daily use*).
