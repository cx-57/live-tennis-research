"""Calibration curves for the live tennis win-probability models.

This script evaluates whether predicted win probabilities match observed win rates.
It saves reliability diagrams in images/ and calibration tables in results/.

Outputs:
    images/figure_3_trace_calibration.png
    images/figure_3_trace_calibration.pdf
    results/calibration_summary.csv
    results/calibration_bins.csv
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load, split


IMAGE_DIR = ROOT / "images"
RESULT_DIR = ROOT / "results"
MATCH_FRACTIONS = [0.25, 0.50, 0.75]
N_BINS = 10
EPS = 1e-6

def clipped(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def predictions_for_fraction(match_fraction):
    """Evaluate the paper's Trace model at a held-out checkpoint."""
    from models import trace
    train, validation, test = split(load(with_elo=True, match_fraction=match_fraction))
    random_state = trace.CHECKPOINT_RANDOM_STATES.get(round(match_fraction, 2), 42)
    (base, slope), markov_loss = trace.tune_markov_params(validation)
    kappa, shrink_loss = trace.tune_serve_shrink_kappa(validation, base, slope)
    params, trace_loss, columns = trace.tune_trace_model(
        train, validation, base, slope, kappa, random_state
    )
    model = trace.refit_on_train_val(
        train, validation, base, slope, kappa, columns, params, random_state
    )
    features, _ = trace.make_features(test, base, slope, kappa, columns)
    tuning = {
        "match_fraction": match_fraction, "percent": round(100 * match_fraction),
        "base": base, "slope": slope, "kappa": kappa,
        "markov_val_logloss": markov_loss, "serve_shrink_val_logloss": shrink_loss,
        "trace_val_logloss": trace_loss, "trace_params": str(params),
        "n_features": len(columns), "random_state": random_state,
    }
    return test.y.values, {"Trace": model.predict_proba(features)[:, 1]}, tuning


def calibration_table(y_true, pred, model_name, match_fraction):
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


def metric_row(y_true, pred, model_name, match_fraction, bin_df):
    pred = clipped(pred)

    return {
        "match_fraction": match_fraction,
        "percent": int(round(match_fraction * 100)),
        "model": model_name,
        "logloss": log_loss(y_true, pred, labels=[0, 1]),
        "brier": brier_score_loss(y_true, pred),
        "accuracy": accuracy_score(y_true, pred >= 0.5),
        "ece": expected_calibration_error(bin_df, len(y_true)),
        "mce": maximum_calibration_error(bin_df),
    }


def plot_reliability_curve(bin_results):
    plt.figure(figsize=(12, 8))
    plt.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=3,
        label="Perfect calibration",
    )

    for match_fraction in MATCH_FRACTIONS:
        model_bins = bin_results[bin_results["match_fraction"] == match_fraction]
        if model_bins.empty:
            continue

        plt.plot(
            100 * model_bins["mean_predicted_probability"],
            100 * model_bins["observed_win_rate"],
            marker="o",
            markersize=8,
            linewidth=3,
            label=f"{int(round(100 * match_fraction))}%",
        )

    plt.xlabel("Predicted Probability of Winning (%)", fontsize=20)
    plt.ylabel("Actual Win Percentage (%)", fontsize=20)
    plt.title("Trace Calibration by Match Progress", fontsize=22)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()

    path = IMAGE_DIR / "figure_3_trace_calibration.png"
    pdf_path = IMAGE_DIR / "figure_3_trace_calibration.pdf"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    return path


def evaluate_calibration():
    IMAGE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    all_bins = []
    all_summary = []
    tuning_rows = []

    for match_fraction in MATCH_FRACTIONS:
        percent = int(round(match_fraction * 100))
        print(f"\nevaluating calibration at {percent}% match progress")

        y_test, test_pred, tuning = predictions_for_fraction(match_fraction)
        tuning_rows.append(tuning)

        model_name = "Trace"
        probability = test_pred[model_name]
        bin_df = calibration_table(
            y_test,
            probability,
            model_name,
            match_fraction,
        )
        all_bins.append(bin_df)
        all_summary.append(
            metric_row(y_test, probability, model_name, match_fraction, bin_df)
        )

    bins = pd.concat(all_bins, ignore_index=True)
    summary = pd.DataFrame(all_summary)
    tuning = pd.DataFrame(tuning_rows)

    bins_path = RESULT_DIR / "calibration_bins.csv"
    summary_path = RESULT_DIR / "calibration_summary.csv"
    tuning_path = RESULT_DIR / "calibration_tuning.csv"

    bins.to_csv(bins_path, index=False)
    summary.to_csv(summary_path, index=False)
    tuning.to_csv(tuning_path, index=False)

    path = plot_reliability_curve(bins)
    print(f"saved graph to {path}")

    print(f"saved calibration bins to {bins_path}")
    print(f"saved calibration summary to {summary_path}")
    print(f"saved tuning details to {tuning_path}")

    print("\nCalibration summary:")
    print(
        summary[
            ["percent", "model", "logloss", "brier", "accuracy", "ece", "mce"]
        ]
        .sort_values(["percent", "model"])
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    evaluate_calibration()
