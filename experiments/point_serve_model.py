"""Leakage-safe next-point serve model used as input to the Markov recursion."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ELO, POINTS, val_logloss


MODEL_GRID = [
    {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15,
     "min_samples_leaf": 100, "l2_regularization": 3.0},
    {"learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31,
     "min_samples_leaf": 100, "l2_regularization": 5.0},
    {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 31,
     "min_samples_leaf": 200, "l2_regularization": 10.0},
]


def load_point_data():
    points = pd.read_parquet(POINTS)
    elo = pd.read_parquet(ELO)
    return points.merge(elo, on="match_id", how="inner")


def _numeric(frame, column, default=0.0):
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy()


def feature_matrix(frame, forced_server=None):
    """Express every row from the server's perspective."""
    if forced_server is None:
        server = _numeric(frame, "server").astype(int)
    else:
        server = np.full(len(frame), int(forced_server))
    p1 = server == 1

    def side(p1_column, p2_column, default=0.0):
        return np.where(
            p1, _numeric(frame, p1_column, default),
            _numeric(frame, p2_column, default),
        )

    def opponent(p1_column, p2_column, default=0.0):
        return np.where(
            p1, _numeric(frame, p2_column, default),
            _numeric(frame, p1_column, default),
        )

    elo_diff = _numeric(frame, "elo_diff")
    output = pd.DataFrame({
        "server_elo_advantage": np.where(p1, elo_diff, -elo_diff),
        "server_live_serve_rate": side("p1_serve_rate", "p2_serve_rate", 0.5),
        "server_live_serve_n": side("p1_serve_n", "p2_serve_n"),
        "opponent_live_serve_rate": opponent("p1_serve_rate", "p2_serve_rate", 0.5),
        "opponent_live_serve_n": opponent("p1_serve_n", "p2_serve_n"),
        "server_hist_serve": side("p1_hist_serve", "p2_hist_serve", 0.60),
        "opponent_hist_return": opponent("p1_hist_return", "p2_hist_return", 0.40),
        "server_hist_serve_n": side("p1_hist_serve_n", "p2_hist_serve_n"),
        "opponent_hist_return_n": opponent("p1_hist_return_n", "p2_hist_return_n"),
        "server_ace_rate": side("p1_ace_rate", "p2_ace_rate"),
        "server_recent_ace_rate": side("p1_recent_ace_rate", "p2_recent_ace_rate"),
        "server_rally_avg": side("p1_serve_rally_avg", "p2_serve_rally_avg"),
        "recent_rally_avg": _numeric(frame, "recent_rally_avg"),
        "log_server_live_n": np.log1p(side("p1_serve_n", "p2_serve_n")),
    })
    slam = frame["slam"].astype(str) if "slam" in frame else pd.Series("", index=frame.index)
    for name in ("ausopen", "frenchopen", "wimbledon", "usopen"):
        output[f"slam_{name}"] = (slam.to_numpy() == name).astype(float)
    return output.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_model(train, params):
    return HistGradientBoostingClassifier(
        loss="log_loss", early_stopping=False, random_state=42, **params
    ).fit(feature_matrix(train), train["serve_won"].to_numpy())


def tune_model(train, validation):
    best = None
    for params in MODEL_GRID:
        model = fit_model(train, params)
        probability = model.predict_proba(feature_matrix(validation))[:, 1]
        loss = val_logloss(validation["serve_won"].to_numpy(), probability)
        if best is None or loss < best[0]:
            best = (loss, params, model)
    return best


def predict_serve_probs(model, states):
    """Return P1-on-serve and P2-on-serve next-point win probabilities."""
    pa = model.predict_proba(feature_matrix(states, forced_server=1))[:, 1]
    pb = model.predict_proba(feature_matrix(states, forced_server=2))[:, 1]
    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)
