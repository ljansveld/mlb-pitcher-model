#!/usr/bin/env python3
"""
run.py — single entry point for the MLB pitcher outcome model.

    python run.py refresh --days 7
    python run.py predict strikeouts
    python run.py predict outs 2026-06-15
    python run.py train hits-walks
    python run.py collect fangraphs
    python run.py features
    python run.py evaluate per-pa

Every target is dispatched as `python -m pitcher_model.<module>` in a
subprocess. That's deliberate rather than importing and calling:

  * Each module keeps its own `if __name__ == "__main__"` block, which
    varies in shape across the pipeline (some define main(), some run
    inline, some parse their own argv). Dispatching by module means none
    of that had to be normalised.
  * Extra arguments pass straight through to the target module, so
    `run.py predict outs 2026-06-15` and `run.py refresh --days 60 --force`
    work without run.py knowing anything about those flags.
  * Training and feature engineering are memory-hungry. A subprocess
    returns that memory to the OS on exit instead of holding it.

`python -m pitcher_model.predict.outs 2026-06-15` remains equivalent for
anyone who prefers the long form.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# group -> {friendly target name: module path}
TARGETS = {
    "collect": {
        "statcast":     "pitcher_model.collect.statcast",
        "fangraphs":    "pitcher_model.collect.fangraphs",
        "catcher":      "pitcher_model.collect.catcher",
        "batters":      "pitcher_model.collect.batters",
        "weather":      "pitcher_model.collect.weather",
        "park-factors": "pitcher_model.collect.park_factors",
    },
    "train": {
        "baseline":            "pitcher_model.train.baseline",
        "strikeouts":          "pitcher_model.train.strikeouts",
        "strikeouts-per-pa":   "pitcher_model.train.strikeouts_per_pa",
        "hits-walks":          "pitcher_model.train.hits_walks",
        "hits-walks-per-pa":   "pitcher_model.train.hits_walks_per_pa",
        "outs":                "pitcher_model.train.outs",
        "outs-per-pa":         "pitcher_model.train.outs_per_pa",
        "earned-runs":         "pitcher_model.train.earned_runs",
        "earned-runs-per-pa":  "pitcher_model.train.earned_runs_per_pa",
    },
    "predict": {
        "strikeouts":  "pitcher_model.predict.strikeouts",
        "hits-walks":  "pitcher_model.predict.hits_walks",
        "outs":        "pitcher_model.predict.outs",
        "earned-runs": "pitcher_model.predict.earned_runs",
    },
    "evaluate": {
        "per-pa":   "pitcher_model.evaluate.per_pa",
        "features": "pitcher_model.evaluate.feature_importance",
    },
}

# Groups that take no target — the group *is* the command.
SINGLETONS = {
    "features": "pitcher_model.features",
    "refresh":  "pitcher_model.refresh",
}

# Convenience: run several targets in sequence.
CHAINS = {
    "train-all": [
        "pitcher_model.train.strikeouts",
        "pitcher_model.train.strikeouts_per_pa",
        "pitcher_model.train.hits_walks",
        "pitcher_model.train.hits_walks_per_pa",
        "pitcher_model.train.outs",
        "pitcher_model.train.outs_per_pa",
        "pitcher_model.train.earned_runs",
        "pitcher_model.train.earned_runs_per_pa",
    ],
    "predict-all": [
        "pitcher_model.predict.strikeouts",
        "pitcher_model.predict.hits_walks",
        "pitcher_model.predict.outs",
        "pitcher_model.predict.earned_runs",
    ],
}


def usage(problem=None):
    if problem:
        print(f"error: {problem}\n", file=sys.stderr)
    print(__doc__.strip().split("\n\n")[0])
    print("\nCommands:")
    for name in SINGLETONS:
        print(f"  {name}")
    for group, targets in TARGETS.items():
        print(f"  {group} <target>")
        width = max(len(t) for t in targets)
        for t in targets:
            print(f"      {t:<{width}}")
    print("  " + "  ".join(CHAINS) + "   (run a whole group in sequence)")
    print("\nExtra arguments are passed through to the target, e.g.")
    print("  python run.py predict outs 2026-06-15")
    print("  python run.py refresh --days 60 --force")
    return 2


def run_module(module, extra):
    """Dispatch one module; return its exit code."""
    cmd = [sys.executable, "-m", module, *extra]
    print(f"\n\033[1m▶ {module}\033[0m" if sys.stdout.isatty() else f"\n▶ {module}")
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()

    cmd, rest = argv[0], argv[1:]

    if cmd in SINGLETONS:
        return run_module(SINGLETONS[cmd], rest)

    if cmd in CHAINS:
        for module in CHAINS[cmd]:
            code = run_module(module, rest)
            if code != 0:
                print(f"\n✗ {module} exited {code} — stopping chain.", file=sys.stderr)
                return code
        print(f"\n✓ {cmd} complete.")
        return 0

    if cmd in TARGETS:
        if not rest:
            return usage(f"'{cmd}' needs a target: "
                         f"{', '.join(TARGETS[cmd])}")
        target, extra = rest[0], rest[1:]
        if target not in TARGETS[cmd]:
            return usage(f"unknown {cmd} target '{target}'. "
                         f"Options: {', '.join(TARGETS[cmd])}")
        return run_module(TARGETS[cmd][target], extra)

    return usage(f"unknown command '{cmd}'")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
