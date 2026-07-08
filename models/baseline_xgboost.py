"""Gradient-boosted trees over score state and live context.

A flexible reference that learns the map from score to win probability directly from data
instead of assuming the Markov structure. It also consumes the entropy-weighted psychological
momentum and other running context features. Early stopping is tuned on the validation season.
"""
import xgboost as xgb

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, split, report

MATCH_FRACTION = 0.50

FEATURES = [
    "sets_diff", "games_diff", "score_diff", "p1_serving", "set_no", "best_of",
    "tiebreak", "pts_played", "p1_sets", "p2_sets", "p1_games", "p2_games",
    "p1_serve_rate", "p2_serve_rate", "p1_serve_n", "p2_serve_n",
    "rally_avg", "recent_rally_avg", "p1_serve_rally_avg", "p2_serve_rally_avg",
    "p1_ace_rate", "p2_ace_rate", "p1_recent_ace_rate", "p2_recent_ace_rate",
    "p1_momentum", "p2_momentum", "momentum_diff",
    "p1_pm_positive", "p2_pm_positive", "pm_positive_diff",
    "p1_pm_negative", "p2_pm_negative", "pm_negative_diff",
]

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.08,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "lambda": 1.0,
    "seed": 42,
    "verbosity": 0,
}

def main():
    train, val, test = split(load(with_elo=False, match_fraction=MATCH_FRACTION))
    train_dm = xgb.DMatrix(train[FEATURES].values, label=train.y.values, feature_names=FEATURES)
    val_dm = xgb.DMatrix(val[FEATURES].values, label=val.y.values, feature_names=FEATURES)
    test_dm = xgb.DMatrix(test[FEATURES].values, feature_names=FEATURES)
    model = xgb.train(
        PARAMS,
        train_dm,
        num_boost_round=2000,
        evals=[(val_dm, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    pred = model.predict(test_dm, iteration_range=(0, model.best_iteration + 1))
    report("xgboost (score+PM)", test.y.values, pred)


if __name__ == "__main__":
    main()
