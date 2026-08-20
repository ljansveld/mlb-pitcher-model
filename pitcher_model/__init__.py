"""
MLB pitcher outcome model.

Predicts full probability distributions over starting-pitcher outcomes —
strikeouts, hits allowed, walks, outs recorded, earned runs — rather than
point estimates.

Subpackages:
    collect   raw data acquisition (Statcast, MLB Stats API, FanGraphs, weather)
    features  the engineered modeling table
    train     model fitting, one module per stat
    predict   daily slate predictions
    evaluate  scoring and diagnostics
"""

__version__ = "1.0.0"
