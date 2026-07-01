"""Gradient-boosted trees over the raw score state (XGBoost version).

A flexible reference that learns the map from score to win probability directly from data
instead of assuming the Markov structure. Early stopping is tuned on the validation season.
"""
import xgboost as xgb

from common import load, split, report

FEATURES = [
    "sets_diff", "games_diff", "score_diff", "p1_serving", "set_no", "best_of",
    "tiebreak", "pts_played", "p1_sets", "p2_sets", "p1_games", "p2_games"
]

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 200,
    "seed": 42
}


def main():
    train, val, test = split(load(with_elo=False, match_fraction=0.25))

    dtrain = xgb.DMatrix(train[FEATURES], label=train.y)
    dval = xgb.DMatrix(val[FEATURES], label=val.y)
    dtest = xgb.DMatrix(test[FEATURES])

    model = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dtest)

    report("xgboost (score)", test.y.values, preds)


if __name__ == "__main__":
    main()
