"""Gradient-boosted trees over the raw score state.

A flexible reference that learns the map from score to win probability directly from data
instead of assuming the Markov structure. Early stopping is tuned on the validation season.
"""
import lightgbm as lgb

from common import load, split, report

FEATURES = ["sets_diff", "games_diff", "score_diff", "p1_serving", "set_no", "best_of",
            "tiebreak", "pts_played", "p1_sets", "p2_sets", "p1_games", "p2_games"]

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=63, max_depth=7, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=200, verbose=-1, seed=42)


def main():
    train, val, test = split(load(with_elo=False))
    model = lgb.train(
        PARAMS,
        lgb.Dataset(train[FEATURES].values, train.y.values),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(val[FEATURES].values, val.y.values)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    report("gradient boosting (score)", test.y.values, model.predict(test[FEATURES].values))


if __name__ == "__main__":
    main()
