"""Gradient-boosting baseline using Elo, score state, and live match features.

Unlike the residual model, this baseline has no Markov prediction. It learns
match-winner probability directly from the pre-match Elo ratings and prepared
live feature row.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUTPUT_PATH = ROOT / "experiments" / "outputs" / "hgbm_accuracy.csv"
PLOT_PATH = ROOT / "experiments" / "outputs" / "hgbm_elo_accuracy.png"

FEATURES = list(dict.fromkeys([
    "elo_p1", "elo_p2", "elo_diff", "elo_prob_p1",
    "sets_diff", "games_diff", "score_diff", "p1_serving", "set_no",
    "best_of", "tiebreak", "pts_played", "p1_sets", "p2_sets",
    "p1_games", "p2_games", "p1_serve_rate", "p2_serve_rate",
    "p1_serve_n", "p2_serve_n", "rally_avg", "recent_rally_avg",
    "p1_serve_rally_avg", "p2_serve_rally_avg", "p1_ace_rate",
    "p2_ace_rate", "p1_recent_ace_rate", "p2_recent_ace_rate",
] + UNIVERSAL_LIVE_FEATURES))
ELO_FEATURES = {"elo_p1", "elo_p2", "elo_diff", "elo_prob_p1"}

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
    features = [
        feature for feature in FEATURES
        if feature in frame.columns or feature not in ELO_FEATURES
    ]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(
            "Missing prepared features: "
            f"{missing}. Rerun: python3 src/prepare_data.py"
        )
    matrix = frame[features].replace([np.inf, -np.inf], np.nan)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def fit_model(x, y, params):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=42,
        **params,
    ).fit(x, y)


def evaluate_fraction(
    fraction: float, tour: str | None = None, with_elo: bool = True
) -> dict:
    data = load(with_elo=with_elo, match_fraction=fraction)
    if tour is not None:
        tour = tour.upper()
        if tour not in {"ATP", "WTA"}:
            raise ValueError("tour must be 'ATP', 'WTA', or None")
        match_number = pd.to_numeric(
            data["match_id"].str.rsplit("-", n=1).str[-1], errors="raise"
        )
        data = data[(match_number < 2000) if tour == "ATP" else (match_number >= 2000)]

    train, validation, test = split(data)
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
        "tour": tour or "Combined",
        "percent": int(round(100 * fraction)),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_logloss": best_validation_loss,
        "test_logloss": log_loss(y_test, probability, labels=[0, 1]),
        "test_accuracy": accuracy_score(y_test, probability >= 0.5),
        **best_params,
    }


def plot_results(results: pd.DataFrame) -> None:
    results = results.sort_values("percent")
    accuracy = 100 * results["test_accuracy"]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        results["percent"], accuracy,
        color="#1f77b4", marker="o", markersize=9, linewidth=3,
    )
    for percent, value in zip(results["percent"], accuracy):
        axis.annotate(
            f"{value:.2f}%", (percent, value), xytext=(0, 10),
            textcoords="offset points", ha="center", fontsize=12,
        )

    axis.set_title("HGBM Baseline with Elo", fontsize=20)
    axis.set_xlabel("Match Progress (%)", fontsize=15)
    axis.set_ylabel("Test Accuracy (%)", fontsize=15)
    axis.set_xticks(results["percent"])
    axis.set_xlim(20, 80)
    axis.set_ylim(max(50, accuracy.min() - 5), min(100, accuracy.max() + 5))
    axis.grid(alpha=0.25)
    axis.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    plot_results(results)
    print("\n" + results.to_string(index=False), flush=True)
    print(f"saved: {OUTPUT_PATH}", flush=True)
    print(f"saved: {PLOT_PATH}", flush=True)


if __name__ == "__main__":
    main()
