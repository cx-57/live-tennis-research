"""Shared paths, data loading, splitting, and metrics for live win-probability models."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("TENNIS_DATA", os.path.join(ROOT, "data"))
ART = os.environ.get("TENNIS_ARTIFACTS", os.path.join(ROOT, "artifacts"))

SLAM_DIR = os.path.join(DATA, "slam")
ATP_DIR = os.path.join(DATA, "atp")
WTA_DIR = os.path.join(DATA, "wta")
POINTS = os.path.join(ART, "points.parquet")
ELO = os.path.join(ART, "elo.parquet")

# Score state passed to the Markov recursion
STATE = [
    "best_of",
    "p1_sets",
    "p2_sets",
    "p1_games",
    "p2_games",
    "p1_serving",
    "p1_score",
    "p2_score",
    "tiebreak",
]


def load(with_elo=True, match_fraction=None):
    """Load processed point data, optionally adding Elo and selecting match progress."""
    df = pd.read_parquet(POINTS)

    if with_elo:
        df = df.merge(pd.read_parquet(ELO), on="match_id", how="inner")

    if match_fraction is not None:
        df = at_match_fraction(df, match_fraction)

    return df


def at_match_fraction(df, fraction):
    """Keep the point closest to a fixed fraction of each match."""
    match_len = df.groupby("match_id").point_no.transform("max")
    target_point = np.ceil(fraction * match_len).clip(lower=1).astype(int)

    return df[df.point_no == target_point].copy()


def split(df):
    """Time split with no match crossing a boundary."""
    train = df[df.year <= 2021]
    val = df[df.year == 2022]
    test = df[df.year >= 2023]

    return train, val, test


def val_logloss(y, p):
    """Compute clipped binary log loss."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def report(name, y, p):
    """Print the main evaluation metrics for a probability model."""
    p = np.clip(p, 1e-6, 1 - 1e-6)

    print(
        f"{name:34s} logloss={log_loss(y, p, labels=[0, 1]):.4f}  "
        f"brier={brier_score_loss(y, p):.4f}  "
        f"accuracy={accuracy_score(y, p > 0.5):.4f}"
    )
