"""Shared, model-independent live-match feature definitions."""

import numpy as np
import pandas as pd


UNIVERSAL_LIVE_FEATURES = [
    "p1_points_won",
    "p2_points_won",
    "points_won_diff",
    "p1_break_points",
    "p2_break_points",
    "break_point_diff",
    "p1_break_points_won",
    "p2_break_points_won",
    "break_point_won_diff",
    "p1_first_serve_points_won",
    "p2_first_serve_points_won",
    "first_srv_won_diff",
    "p1_double_faults",
    "p2_double_faults",
    "double_fault_diff",
    "p1_momentum",
    "p2_momentum",
    "momentum_diff",
    "live_serve_diff",
    "serve_points_total",
    "serve_points_balance",
    "serve_sample_weight",
]


def numeric_col(df, column):
    """Return one column as finite numeric values."""
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def add_universal_live_features(df):
    """Add model-independent difference and sample-size features."""
    x = df.copy()

    pairs = {
        "points_won_diff": ("p1_points_won", "p2_points_won"),
        "break_point_diff": ("p1_break_points", "p2_break_points"),
        "break_point_won_diff": (
            "p1_break_points_won",
            "p2_break_points_won",
        ),
        "first_srv_won_diff": (
            "p1_first_serve_points_won",
            "p2_first_serve_points_won",
        ),
        "double_fault_diff": ("p1_double_faults", "p2_double_faults"),
        "momentum_diff": ("p1_momentum", "p2_momentum"),
        "live_serve_diff": ("p1_serve_rate", "p2_serve_rate"),
        "serve_points_balance": ("p1_serve_n", "p2_serve_n"),
    }

    for output, (p1_column, p2_column) in pairs.items():
        x[output] = numeric_col(x, p1_column) - numeric_col(x, p2_column)

    x["serve_points_total"] = (
        numeric_col(x, "p1_serve_n") + numeric_col(x, "p2_serve_n")
    )
    x["serve_sample_weight"] = np.log1p(x["serve_points_total"])

    return x
