"""Gradient-boosting baseline using score state and live match features.

Unlike the residual model, this baseline has no Elo or Markov prediction. It
learns match-winner probability directly from the prepared live feature row.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, split, val_logloss
from src.live_features import UNIVERSAL_LIVE_FEATURES


MATCH_FRACTIONS = (0.25, 0.50, 0.75)
OUTPUT_PATH = ROOT / "images" / "baseline_gbm_accuracy.csv"

FEATURES = list(dict.fromkeys([
    "sets_diff", "games_diff", "score_diff", "p1_serving", "set_no",
    "best_of", "tiebreak", "pts_played", "p1_sets", "p2_sets",
    "p1_games", "p2_games", "p1_serve_rate", "p2_serve_rate",
    "p1_serve_n", "p2_serve_n", "rally_avg", "recent_rally_avg",
    "p1_serve_rally_avg", "p2_serve_rally_avg", "p1_ace_rate",
    "p2_ace_rate", "p1_recent_ace_rate", "p2_recent_ace_rate",
] + UNIVERSAL_LIVE_FEATURES))

MODEL_GRID = [
    {
        "learning_rate": 0.03,
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "l2_regularization": 3.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "l2_regularization": 3.0,
    },
]


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(
            "Missing prepared features: "
            f"{missing}. Rerun: python3 src/prepare_data.py"
        )
    matrix = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def fit_model(x, y, params):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=42,
        **params,
    ).fit(x, y)


def evaluate_fraction(fraction: float) -> dict:
    train, validation, test = split(
        load(with_elo=False, match_fraction=fraction)
    )
    x_train = feature_matrix(train)
    x_validation = feature_matrix(validation)
    x_test = feature_matrix(test)

    best_params = None
    best_validation_loss = float("inf")
    for params in MODEL_GRID:
        model = fit_model(x_train, train.y.to_numpy(), params)
        probability = model.predict_proba(x_validation)[:, 1]
        validation_loss = val_logloss(validation.y.to_numpy(), probability)
        if validation_loss < best_validation_loss:
            best_params = params
            best_validation_loss = validation_loss

    train_validation = pd.concat([train, validation], ignore_index=True)
    model = fit_model(
        feature_matrix(train_validation),
        train_validation.y.to_numpy(),
        best_params,
    )
    probability = model.predict_proba(x_test)[:, 1]
    y_test = test.y.to_numpy()
    return {
        "match_fraction": fraction,
        "percent": int(round(100 * fraction)),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_logloss": best_validation_loss,
        "test_logloss": log_loss(y_test, probability, labels=[0, 1]),
        "test_accuracy": accuracy_score(y_test, probability >= 0.5),
        **best_params,
    }


def main():
    rows = []
    for fraction in MATCH_FRACTIONS:
        print(f"evaluating baseline GBM at {fraction:.0%}", flush=True)
        result = evaluate_fraction(fraction)
        rows.append(result)
        print(
            f"  accuracy={result['test_accuracy']:.2%} "
            f"logloss={result['test_logloss']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print("\n" + results.to_string(index=False), flush=True)
    print(f"saved: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
