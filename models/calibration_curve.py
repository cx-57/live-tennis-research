"""Calibration curves for the live tennis win-probability models.

This script evaluates whether predicted win probabilities match observed win rates.
It saves reliability diagrams and calibration tables into the images folder.

Outputs:
    images/calibration_curve_raw_25.png
    images/calibration_curve_raw_50.png
    images/calibration_curve_raw_75.png
    images/calibration_curve_calibrated_25.png
    images/calibration_curve_calibrated_50.png
    images/calibration_curve_calibrated_75.png
    images/calibration_summary.csv
    images/calibration_bins.csv
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import STATE, load, split, val_logloss
from src.markov import predict


IMAGE_DIR = ROOT / "images"
MATCH_FRACTIONS = [0.25, 0.50, 0.75]
N_BINS = 10
EPS = 1e-6

SYMMETRIC_P_GRID = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
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


def clipped(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p):
    p = clipped(p)
    return np.log(p / (1.0 - p))


def numeric_array(df, col, default=0.0):
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def numeric_col(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def add_diff_feature(df, new_col, p1_col, p2_col):
    if p1_col in df.columns and p2_col in df.columns:
        df[new_col] = numeric_col(df, p1_col) - numeric_col(df, p2_col)


def asymmetric_serve_probs(df, base, slope):
    elo_diff = numeric_array(df, "elo_diff", default=0.0)
    edge = np.clip(slope * elo_diff, -0.15, 0.15)

    pa = base + edge
    pb = base - edge

    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def serve_shrink_probs(df, base, slope, kappa):
    prior_a, prior_b = asymmetric_serve_probs(df, base, slope)

    p1_rate = numeric_array(df, "p1_serve_rate", default=np.nan)
    p2_rate = numeric_array(df, "p2_serve_rate", default=np.nan)
    p1_n = numeric_array(df, "p1_serve_n", default=0.0)
    p2_n = numeric_array(df, "p2_serve_n", default=0.0)

    # If an early state has no live serve rate yet, fall back to the prior.
    p1_rate = np.where(np.isfinite(p1_rate), p1_rate, prior_a)
    p2_rate = np.where(np.isfinite(p2_rate), p2_rate, prior_b)

    pa = (p1_n * p1_rate + kappa * prior_a) / (p1_n + kappa)
    pb = (p2_n * p2_rate + kappa * prior_b) / (p2_n + kappa)

    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def markov_prediction(df, base, slope):
    pa, pb = asymmetric_serve_probs(df, base, slope)
    return predict(df, pa, pb, STATE)


def serve_shrink_prediction(df, base, slope, kappa):
    pa, pb = serve_shrink_probs(df, base, slope, kappa)
    return predict(df, pa, pb, STATE)


def tune_symmetric_p(val):
    best_p = None
    best_ll = float("inf")

    for p in SYMMETRIC_P_GRID:
        pred = predict(val, p, p, STATE)
        ll = val_logloss(val.y.values, pred)

        if np.isfinite(ll) and ll < best_ll:
            best_p = p
            best_ll = ll

    if best_p is None:
        raise ValueError("Could not tune symmetric Markov probability.")

    return best_p, best_ll


def tune_markov_params(val):
    best_params = None
    best_ll = float("inf")

    for base in BASE_GRID:
        for slope in SLOPE_GRID:
            pred = markov_prediction(val, base, slope)
            ll = val_logloss(val.y.values, pred)

            if np.isfinite(ll) and ll < best_ll:
                best_params = (base, slope)
                best_ll = ll

    if best_params is None:
        raise ValueError("Could not tune asymmetric Markov parameters.")

    return best_params, best_ll


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
    x = df.copy()

    prior_a, prior_b = asymmetric_serve_probs(x, base, slope)
    markov_prob = predict(x, prior_a, prior_b, STATE)
    serve_shrink_prob = serve_shrink_prediction(x, base, slope, kappa)

    p1_rate = numeric_array(x, "p1_serve_rate", default=np.nan)
    p2_rate = numeric_array(x, "p2_serve_rate", default=np.nan)
    p1_n = numeric_array(x, "p1_serve_n", default=0.0)
    p2_n = numeric_array(x, "p2_serve_n", default=0.0)

    p1_rate = np.where(np.isfinite(p1_rate), p1_rate, prior_a)
    p2_rate = np.where(np.isfinite(p2_rate), p2_rate, prior_b)

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
    x["live_serve_edge_diff"] = x["p1_live_serve_edge"] - x["p2_live_serve_edge"]

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
    cols = [c for c in DERIVED_FEATURES if c in df.columns]
    cols += [c for c in RAW_FEATURES if c in df.columns]
    cols += [c for c in LIVE_EXTRA_FEATURES if c in df.columns]
    cols += [c for c in STATE if c in df.columns]

    return list(dict.fromkeys(cols))


def make_features(df, base, slope, kappa, columns=None):
    x = add_ml_features(df, base, slope, kappa)

    if columns is None:
        columns = feature_columns(x)

    x = x[columns]
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)

    return x, columns


def fit_residual_model(x_train, y_train, params):
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
        model = fit_residual_model(x_train, train.y.values, params)
        pred = model.predict_proba(x_val)[:, 1]
        ll = val_logloss(val.y.values, pred)

        if np.isfinite(ll) and ll < best_ll:
            best_params = params
            best_ll = ll

    if best_params is None:
        raise ValueError("Could not tune residual model.")

    return best_params, best_ll, columns


def raw_predictions_for_fraction(match_fraction):
    train, val, test = split(load(with_elo=True, match_fraction=match_fraction))

    y_val = val.y.values
    y_test = test.y.values

    val_pred = {}
    test_pred = {}
    tuning = {"match_fraction": match_fraction, "percent": int(round(match_fraction * 100))}

    symmetric_p, symmetric_ll = tune_symmetric_p(val)
    val_pred["Symmetric Markov"] = predict(val, symmetric_p, symmetric_p, STATE)
    test_pred["Symmetric Markov"] = predict(test, symmetric_p, symmetric_p, STATE)
    tuning["symmetric_p"] = symmetric_p
    tuning["symmetric_val_logloss"] = symmetric_ll

    (base, slope), markov_ll = tune_markov_params(val)
    val_pred["Asymmetric Markov"] = markov_prediction(val, base, slope)
    test_pred["Asymmetric Markov"] = markov_prediction(test, base, slope)
    tuning["base"] = base
    tuning["slope"] = slope
    tuning["markov_val_logloss"] = markov_ll

    kappa, serve_shrink_ll = tune_serve_shrink_kappa(val, base, slope)
    val_pred["Serve-shrink Markov"] = serve_shrink_prediction(val, base, slope, kappa)
    test_pred["Serve-shrink Markov"] = serve_shrink_prediction(test, base, slope, kappa)
    tuning["kappa"] = kappa
    tuning["serve_shrink_val_logloss"] = serve_shrink_ll

    params, residual_ll, columns = tune_residual_model(train, val, base, slope, kappa)

    x_train, _ = make_features(train, base, slope, kappa, columns)
    x_val, _ = make_features(val, base, slope, kappa, columns)
    residual_val_model = fit_residual_model(x_train, train.y.values, params)
    val_pred["Residual Markov"] = residual_val_model.predict_proba(x_val)[:, 1]

    train_val = pd.concat([train, val], ignore_index=True)
    x_train_val, _ = make_features(train_val, base, slope, kappa, columns)
    x_test, _ = make_features(test, base, slope, kappa, columns)
    residual_test_model = fit_residual_model(x_train_val, train_val.y.values, params)
    test_pred["Residual Markov"] = residual_test_model.predict_proba(x_test)[:, 1]

    tuning["residual_val_logloss"] = residual_ll
    tuning["residual_params"] = str(params)
    tuning["n_features"] = len(columns)

    return y_val, y_test, val_pred, test_pred, tuning


def fit_logistic_calibrator(y_val, p_val):
    """Fit a Platt-style logistic calibrator from validation predictions."""
    y_val = np.asarray(y_val, dtype=int)
    p_val = clipped(p_val)

    if len(np.unique(y_val)) < 2:
        return None

    model = LogisticRegression(max_iter=1000)
    model.fit(logit(p_val).reshape(-1, 1), y_val)
    return model


def apply_logistic_calibrator(calibrator, p):
    p = clipped(p)

    if calibrator is None:
        return p

    return clipped(calibrator.predict_proba(logit(p).reshape(-1, 1))[:, 1])


def calibration_table(y_true, pred, model_name, variant, match_fraction):
    y_true = np.asarray(y_true, dtype=float)
    pred = clipped(pred)

    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_ids = np.digitize(pred, bins[1:-1], right=False)

    rows = []
    for bin_id in range(N_BINS):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        mean_pred = float(pred[mask].mean())
        observed_rate = float(y_true[mask].mean())
        abs_error = abs(observed_rate - mean_pred)

        rows.append(
            {
                "match_fraction": match_fraction,
                "percent": int(round(match_fraction * 100)),
                "model": model_name,
                "variant": variant,
                "bin": bin_id + 1,
                "bin_low": bins[bin_id],
                "bin_high": bins[bin_id + 1],
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_win_rate": observed_rate,
                "abs_calibration_error": abs_error,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(bin_df, total_count):
    if bin_df.empty or total_count == 0:
        return np.nan
    return float((bin_df["count"] / total_count * bin_df["abs_calibration_error"]).sum())


def maximum_calibration_error(bin_df):
    if bin_df.empty:
        return np.nan
    return float(bin_df["abs_calibration_error"].max())


def metric_row(y_true, pred, model_name, variant, match_fraction, bin_df):
    pred = clipped(pred)

    return {
        "match_fraction": match_fraction,
        "percent": int(round(match_fraction * 100)),
        "model": model_name,
        "variant": variant,
        "logloss": log_loss(y_true, pred, labels=[0, 1]),
        "brier": brier_score_loss(y_true, pred),
        "accuracy": accuracy_score(y_true, pred >= 0.5),
        "ece": expected_calibration_error(bin_df, len(y_true)),
        "mce": maximum_calibration_error(bin_df),
    }


def plot_reliability_curve(bin_results, summary, match_fraction, variant):
    percent = int(round(match_fraction * 100))
    subset = bin_results[
        (bin_results["match_fraction"] == match_fraction) & (bin_results["variant"] == variant)
    ]
    summary_subset = summary[
        (summary["match_fraction"] == match_fraction) & (summary["variant"] == variant)
    ]

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for model_name in summary_subset["model"]:
        model_bins = subset[subset["model"] == model_name]
        if model_bins.empty:
            continue

        plt.plot(
            model_bins["mean_predicted_probability"],
            model_bins["observed_win_rate"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Mean Predicted Win Probability")
    plt.ylabel("Observed Win Rate")
    plt.title(f"Reliability Diagram at {percent}% Match Progress ({variant})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = IMAGE_DIR / f"calibration_curve_{variant}_{percent}.png"
    plt.savefig(path, dpi=200)
    plt.close()

    return path


def evaluate_calibration():
    IMAGE_DIR.mkdir(exist_ok=True)

    all_bins = []
    all_summary = []
    tuning_rows = []

    for match_fraction in MATCH_FRACTIONS:
        percent = int(round(match_fraction * 100))
        print(f"\nevaluating calibration at {percent}% match progress")

        y_val, y_test, val_raw, test_raw, tuning = raw_predictions_for_fraction(match_fraction)
        tuning_rows.append(tuning)

        for model_name, raw_test_pred in test_raw.items():
            raw_bin_df = calibration_table(
                y_test,
                raw_test_pred,
                model_name,
                "raw",
                match_fraction,
            )
            all_bins.append(raw_bin_df)
            all_summary.append(
                metric_row(y_test, raw_test_pred, model_name, "raw", match_fraction, raw_bin_df)
            )

            calibrator = fit_logistic_calibrator(y_val, val_raw[model_name])
            calibrated_test_pred = apply_logistic_calibrator(calibrator, raw_test_pred)

            calibrated_bin_df = calibration_table(
                y_test,
                calibrated_test_pred,
                model_name,
                "calibrated",
                match_fraction,
            )
            all_bins.append(calibrated_bin_df)
            all_summary.append(
                metric_row(
                    y_test,
                    calibrated_test_pred,
                    model_name,
                    "calibrated",
                    match_fraction,
                    calibrated_bin_df,
                )
            )

    bins = pd.concat(all_bins, ignore_index=True)
    summary = pd.DataFrame(all_summary)
    tuning = pd.DataFrame(tuning_rows)

    bins_path = IMAGE_DIR / "calibration_bins.csv"
    summary_path = IMAGE_DIR / "calibration_summary.csv"
    tuning_path = IMAGE_DIR / "calibration_tuning.csv"

    bins.to_csv(bins_path, index=False)
    summary.to_csv(summary_path, index=False)
    tuning.to_csv(tuning_path, index=False)

    for match_fraction in MATCH_FRACTIONS:
        for variant in ["raw", "calibrated"]:
            path = plot_reliability_curve(bins, summary, match_fraction, variant)
            print(f"saved graph to {path}")

    print(f"saved calibration bins to {bins_path}")
    print(f"saved calibration summary to {summary_path}")
    print(f"saved tuning details to {tuning_path}")

    print("\nCalibration summary:")
    print(
        summary[
            ["percent", "model", "variant", "logloss", "brier", "accuracy", "ece", "mce"]
        ]
        .sort_values(["percent", "variant", "model"])
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    evaluate_calibration()
