"""Residual machine-learning model built on top of the Markov tennis model.

This file compares three live win-probability approaches across different match stages:
an Elo-based Markov model, a serve-shrink Markov model, and a residual gradient-boosting model.
The residual model uses the Markov predictions plus live match features to learn corrections
that the structured Markov model may miss.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import STATE, load, report, split, val_logloss
from markov import predict


MATCH_FRACTION = 0.75

# Match fractions used to evaluate how accuracy changes as more of the match is known
PLOT_ACCURACY_CURVE = True
PLOT_FRACTIONS = np.linspace(0.05, 0.95, 19)

# Output files for the accuracy graph and the underlying results table
IMAGE_DIR = "images"
PLOT_PATH = os.path.join(IMAGE_DIR, "residual_markov_live_features_accuracy.png")
CSV_PATH = os.path.join(IMAGE_DIR, "residual_markov_live_features_accuracy.csv")

# Hyperparameter grids for the Markov prior, serve-shrink strength, and residual ML model
BASE_GRID = [0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
SLOPE_GRID = [4e-5, 6e-5, 9e-5, 1.3e-4, 1.8e-4, 2.2e-4]
KAPPA_GRID = [40, 80, 160, 320, 640]

MODEL_GRIDS = [
    {"learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 3.0},
]

RAW_FEATURES = [
    "elo_diff",
    "p1_serve_rate",
    "p1_serve_n",
    "p2_serve_rate",
    "p2_serve_n",
]

LIVE_EXTRA_FEATURES = [
    "P1PointsWon",
    "P2PointsWon",
    "P1BreakPoint",
    "P2BreakPoint",
    "P1BreakPointWon",
    "P2BreakPointWon",
    "P1FirstSrvWon",
    "P2FirstSrvWon",
    "P1DoubleFault",
    "P2DoubleFault",
    "P1Momentum",
    "P2Momentum",
    "Rally",
    "Speed_KMH",
]

DERIVED_FEATURES = [
    "markov_prob",
    "markov_logit",
    "markov_uncertainty",
    "serve_shrink_prob",
    "serve_shrink_logit",
    "serve_shrink_diff",
    "serve_shrink_abs_diff",
    "prior_serve_diff",
    "live_serve_diff",
    "p1_live_serve_edge",
    "p2_live_serve_edge",
    "live_serve_edge_diff",
    "serve_points_total",
    "serve_points_balance",
    "serve_sample_weight",
    "points_won_diff",
    "break_point_diff",
    "break_point_won_diff",
    "first_srv_won_diff",
    "double_fault_diff",
    "momentum_diff",
]


def asymmetric_serve_probs(df, base, slope):
    # Convert Elo difference into separate serve probabilities for each player
    edge = np.clip(slope * df.elo_diff.to_numpy(), -0.15, 0.15)

    pa = base + edge
    pb = base - edge

    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def numeric_col(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def add_diff_feature(df, new_col, p1_col, p2_col):
    if p1_col in df.columns and p2_col in df.columns:
        df[new_col] = numeric_col(df, p1_col) - numeric_col(df, p2_col)


def accuracy(y_true, pred):
    return accuracy_score(y_true, pred >= 0.5)


def tune_markov_params(val):
    best_params = None
    best_ll = float("inf")

    for base in BASE_GRID:
        for slope in SLOPE_GRID:
            pa, pb = asymmetric_serve_probs(val, base, slope)
            pred = predict(val, pa, pb, STATE)
            ll = val_logloss(val.y.values, pred)

            if np.isfinite(ll) and ll < best_ll:
                best_params = (base, slope)
                best_ll = ll

    if best_params is None:
        raise ValueError("Could not tune asymmetric Markov params.")

    return best_params, best_ll


def markov_prediction(df, base, slope):
    pa, pb = asymmetric_serve_probs(df, base, slope)
    return predict(df, pa, pb, STATE)


def serve_shrink_probs(df, base, slope, kappa):
    # Blend pre-match serve priors with live serve performance from the match so far
    prior_a, prior_b = asymmetric_serve_probs(df, base, slope)

    p1_rate = df.p1_serve_rate.to_numpy()
    p2_rate = df.p2_serve_rate.to_numpy()
    p1_n = df.p1_serve_n.to_numpy()
    p2_n = df.p2_serve_n.to_numpy()

    pa = (p1_n * p1_rate + kappa * prior_a) / (p1_n + kappa)
    pb = (p2_n * p2_rate + kappa * prior_b) / (p2_n + kappa)

    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def serve_shrink_prediction(df, base, slope, kappa):
    pa, pb = serve_shrink_probs(df, base, slope, kappa)
    return predict(df, pa, pb, STATE)


def tune_serve_shrink_kappa(val, base, slope):
    best_kappa = None
    best_ll = float("inf")

    for kappa in KAPPA_GRID:
        pred = serve_shrink_prediction(val, base, slope, kappa)
        ll = val_logloss(val.y.values, pred)

        if np.isfinite(ll) and ll < best_ll:
            best_kappa = kappa
            best_ll = ll

    if best_kappa is None:
        raise ValueError("Could not tune serve-shrink kappa.")

    return best_kappa, best_ll


def add_ml_features(df, base, slope, kappa):
    # Add Markov outputs and live-match differences as features for the residual model
    x = df.copy()

    prior_a, prior_b = asymmetric_serve_probs(x, base, slope)

    markov_prob = predict(x, prior_a, prior_b, STATE)
    serve_shrink_prob = serve_shrink_prediction(x, base, slope, kappa)

    p1_rate = x.p1_serve_rate.to_numpy()
    p2_rate = x.p2_serve_rate.to_numpy()
    p1_n = x.p1_serve_n.to_numpy()
    p2_n = x.p2_serve_n.to_numpy()

    serve_points_total = p1_n + p2_n

    x["markov_prob"] = markov_prob
    x["markov_logit"] = logit(markov_prob)
    x["markov_uncertainty"] = 1.0 - np.abs(2.0 * markov_prob - 1.0)

    x["serve_shrink_prob"] = serve_shrink_prob
    x["serve_shrink_logit"] = logit(serve_shrink_prob)
    x["serve_shrink_diff"] = serve_shrink_prob - markov_prob
    x["serve_shrink_abs_diff"] = np.abs(x["serve_shrink_diff"])

    x["prior_serve_diff"] = prior_a - prior_b
    x["live_serve_diff"] = p1_rate - p2_rate
    x["p1_live_serve_edge"] = p1_rate - prior_a
    x["p2_live_serve_edge"] = p2_rate - prior_b
    x["live_serve_edge_diff"] = (
        x["p1_live_serve_edge"] - x["p2_live_serve_edge"]
    )

    x["serve_points_total"] = serve_points_total
    x["serve_points_balance"] = p1_n - p2_n
    x["serve_sample_weight"] = np.log1p(serve_points_total)

    add_diff_feature(x, "points_won_diff", "P1PointsWon", "P2PointsWon")
    add_diff_feature(x, "break_point_diff", "P1BreakPoint", "P2BreakPoint")
    add_diff_feature(x, "break_point_won_diff", "P1BreakPointWon", "P2BreakPointWon")
    add_diff_feature(x, "first_srv_won_diff", "P1FirstSrvWon", "P2FirstSrvWon")
    add_diff_feature(x, "double_fault_diff", "P1DoubleFault", "P2DoubleFault")
    add_diff_feature(x, "momentum_diff", "P1Momentum", "P2Momentum")

    return x


def feature_columns(df):
    # Keep only feature columns that actually exist in the current dataset
    cols = [c for c in DERIVED_FEATURES if c in df.columns]
    cols += [c for c in RAW_FEATURES if c in df.columns]
    cols += [c for c in LIVE_EXTRA_FEATURES if c in df.columns]
    cols += [c for c in STATE if c in df.columns]

    return list(dict.fromkeys(cols))


def make_features(df, base, slope, kappa, columns=None):
    # Clean features so the ML model only receives finite numeric values
    x = add_ml_features(df, base, slope, kappa)

    if columns is None:
        columns = feature_columns(x)

    x = x[columns]
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)

    return x, columns


def fit_model(x_train, y_train, params):
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=250,
        early_stopping=True,
        random_state=42,
        **params,
    )

    model.fit(x_train, y_train)
    return model


def tune_residual_model(train, val, base, slope, kappa):
    x_train, columns = make_features(train, base, slope, kappa)
    x_val, _ = make_features(val, base, slope, kappa, columns)

    best_params = None
    best_ll = float("inf")

    for params in MODEL_GRIDS:
        model = fit_model(x_train, train.y.values, params)
        pred = model.predict_proba(x_val)[:, 1]
        ll = val_logloss(val.y.values, pred)

        if np.isfinite(ll) and ll < best_ll:
            best_params = params
            best_ll = ll

    if best_params is None:
        raise ValueError("Could not tune residual model.")

    return best_params, best_ll, columns


def refit_on_train_val(train, val, base, slope, kappa, columns, params):
    train_val = pd.concat([train, val], ignore_index=True)
    x_train_val, _ = make_features(train_val, base, slope, kappa, columns)

    return fit_model(x_train_val, train_val.y.values, params)


def evaluate_fraction(match_fraction):
    # Evaluate Markov, serve-shrink, and residual models at one match fraction
    train, val, test = split(load(with_elo=True, match_fraction=match_fraction))

    (base, slope), markov_ll = tune_markov_params(val)
    markov_test_pred = markov_prediction(test, base, slope)

    kappa, serve_shrink_ll = tune_serve_shrink_kappa(val, base, slope)
    serve_shrink_test_pred = serve_shrink_prediction(test, base, slope, kappa)

    params, residual_ll, columns = tune_residual_model(
        train,
        val,
        base,
        slope,
        kappa,
    )

    model = refit_on_train_val(
        train,
        val,
        base,
        slope,
        kappa,
        columns,
        params,
    )

    x_test, _ = make_features(test, base, slope, kappa, columns)
    residual_test_pred = model.predict_proba(x_test)[:, 1]

    return {
        "match_fraction": match_fraction,
        "percent": int(round(match_fraction * 100)),
        "base": base,
        "slope": slope,
        "kappa": kappa,
        "markov_val_logloss": markov_ll,
        "serve_shrink_val_logloss": serve_shrink_ll,
        "residual_val_logloss": residual_ll,
        "markov_accuracy": accuracy(test.y.values, markov_test_pred),
        "serve_shrink_accuracy": accuracy(test.y.values, serve_shrink_test_pred),
        "residual_accuracy": accuracy(test.y.values, residual_test_pred),
    }


def plot_accuracy_curve():
    # Run all match fractions and save the accuracy curve
    rows = []

    for fraction in PLOT_FRACTIONS:
        print(f"evaluating match_fraction={fraction:.2f}")
        rows.append(evaluate_fraction(float(fraction)))

    results = pd.DataFrame(rows)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    results.to_csv(CSV_PATH, index=False)

    print("\nAccuracy by match progress:")
    print(
        results[
            [
                "percent",
                "markov_accuracy",
                "serve_shrink_accuracy",
                "residual_accuracy",
                "markov_val_logloss",
                "serve_shrink_val_logloss",
                "residual_val_logloss",
            ]
        ].round(4).to_string(index=False)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        results.percent,
        results.markov_accuracy,
        marker="o",
        label="Asymmetric Markov",
    )

    plt.plot(
        results.percent,
        results.serve_shrink_accuracy,
        marker="o",
        label="Serve-shrink Markov",
    )

    plt.plot(
        results.percent,
        results.residual_accuracy,
        marker="o",
        label="Residual + serve-shrink GBM + live features",
    )

    plt.xlabel("Match Progress (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Live Win-Probability Accuracy by Match Progress")
    plt.ylim(0.45, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=200)

    print(f"\nsaved graph to {PLOT_PATH}")
    print(f"saved data to {CSV_PATH}")


def main():
    if PLOT_ACCURACY_CURVE:
        plot_accuracy_curve()
        return

    train, val, test = split(load(with_elo=True, match_fraction=MATCH_FRACTION))

    print(f"match_fraction={MATCH_FRACTION}")

    (base, slope), markov_ll = tune_markov_params(val)

    print(
        f"asymmetric Markov params: "
        f"base={base} slope={slope} val_logloss={markov_ll:.4f}"
    )

    markov_test_pred = markov_prediction(test, base, slope)
    report("asymmetric Markov", test.y.values, markov_test_pred)

    kappa, serve_shrink_ll = tune_serve_shrink_kappa(val, base, slope)

    print(f"serve-shrink kappa={kappa} val_logloss={serve_shrink_ll:.4f}")

    serve_shrink_test_pred = serve_shrink_prediction(test, base, slope, kappa)
    report("serve-shrink Markov", test.y.values, serve_shrink_test_pred)

    params, residual_ll, columns = tune_residual_model(
        train,
        val,
        base,
        slope,
        kappa,
    )

    print(f"residual params={params} val_logloss={residual_ll:.4f}")
    print(f"features: {columns}")

    model = refit_on_train_val(
        train,
        val,
        base,
        slope,
        kappa,
        columns,
        params,
    )

    x_test, _ = make_features(test, base, slope, kappa, columns)
    residual_test_pred = model.predict_proba(x_test)[:, 1]

    report(
        "residual Markov + serve-shrink GBM + selected live features",
        test.y.values,
        residual_test_pred,
    )


if __name__ == "__main__":
    main()
