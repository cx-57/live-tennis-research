"""XGBoost live win-probability baseline.

This model learns win probability directly from score state and live match features instead of
using the Markov recursion. It is used as a flexible machine-learning comparison against the
structured Markov models.
"""

import sys
from pathlib import Path

import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, report, split


MATCH_FRACTION = 0.50

FEATURES = [
    "sets_diff", "games_diff", "score_diff", "p1_serving", "set_no", "best_of",
    "tiebreak", "pts_played", "p1_sets", "p2_sets", "p1_games", "p2_games",
    "p1_serve_rate", "p2_serve_rate", "p1_serve_n", "p2_serve_n",
    "rally_avg", "recent_rally_avg", "p1_serve_rally_avg", "p2_serve_rally_avg",
    "p1_ace_rate", "p2_ace_rate", "p1_recent_ace_rate", "p2_recent_ace_rate",
    "p1_points_won", "p2_points_won", "points_won_diff",
    "p1_break_points", "p2_break_points", "break_point_diff",
    "p1_break_points_won", "p2_break_points_won", "break_point_won_diff",
    "p1_first_serve_points_won", "p2_first_serve_points_won", "first_srv_won_diff",
    "p1_double_faults", "p2_double_faults", "double_fault_diff",
    "p1_momentum", "p2_momentum", "momentum_diff",
    "live_serve_diff", "serve_points_total", "serve_points_balance",
    "serve_sample_weight",
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

    missing = [feature for feature in FEATURES if feature not in train.columns]
    if missing:
        raise ValueError(
            "Missing prepared features: "
            f"{missing}. Rerun: python3 scripts/prepare_data.py"
        )

    train_dm = xgb.DMatrix(
        train[FEATURES].values,
        label=train.y.values,
        feature_names=FEATURES,
    )
    val_dm = xgb.DMatrix(
        val[FEATURES].values,
        label=val.y.values,
        feature_names=FEATURES,
    )
    test_dm = xgb.DMatrix(
        test[FEATURES].values,
        feature_names=FEATURES,
    )

    model = xgb.train(
        PARAMS,
        train_dm,
        num_boost_round=2000,
        evals=[(val_dm, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    pred = model.predict(test_dm, iteration_range=(0, model.best_iteration + 1))
    report("xgboost (score+live)", test.y.values, pred)


if __name__ == "__main__":
    main()
