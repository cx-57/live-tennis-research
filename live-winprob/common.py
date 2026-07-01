"""Shared paths, data loading and metrics for the live win-probability models."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TENNIS_DATA", os.path.join(ROOT, "data"))
ART = os.environ.get("TENNIS_ARTIFACTS", os.path.join(ROOT, "artifacts"))

SLAM_DIR = os.path.join(DATA, "slam")
ATP_DIR = os.path.join(DATA, "atp")
WTA_DIR = os.path.join(DATA, "wta")

POINTS = os.path.join(ART, "points.parquet")
ELO = os.path.join(ART, "elo.parquet")

# the score state passed to the Markov recursion
STATE = [
    "best_of", "p1_sets", "p2_sets",
    "p1_games", "p2_games",
    "p1_serving", "p1_score", "p2_score",
    "tiebreak"
]


def load(with_elo=True, match_fraction=None):
    df = pd.read_parquet(POINTS)

    if with_elo:
        df = df.merge(pd.read_parquet(ELO), on="match_id", how="inner")

    if match_fraction is not None:
        # total points in each match
        df["match_len"] = df.groupby("match_id")["point_no"].transform("max")

        # point corresponding to desired percentage
        df["target_point"] = (
            np.ceil(match_fraction * df["match_len"])
            .clip(lower=1)
            .astype(int)
        )

        # keep one row per match only
        df = df[df["point_no"] == df["target_point"]].copy()

    return df


def split(df):
    """Time split, no match crosses a boundary: train up to 2021, validate 2022, test 2023 on."""
    return (
        df[df.year <= 2021],
        df[df.year == 2022],
        df[df.year >= 2023]
    )


def val_logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def report(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print(
        f"{name:34s} logloss={log_loss(y, p, labels=[0, 1]):.4f} "
        f"brier={brier_score_loss(y, p):.4f} "
        f"accuracy={accuracy_score(y, p > 0.5):.4f}"
    )