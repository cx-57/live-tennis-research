"""Compare core models with train 2011-2019, dev 2017, and test 2014.

The training split explicitly excludes the development and test seasons. The
benchmark evaluates match snapshots at 25%, 50%, and 75%, plus every point in
the 2014 test matches. Results are stored separately from the main time split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from models import baseline_gbm
from models import markov_ensemble as residual
from src.common import STATE, at_match_fraction, load, val_logloss
from src.markov import predict


TRAIN_YEARS = tuple(year for year in range(2011, 2020) if year not in (2014, 2017))
DEV_YEAR = 2017
TEST_YEAR = 2014
CHECKPOINTS = (0.25, 0.50, 0.75, None)
OUTPUT_PATH = ROOT / "images" / "year_split_2011_2019_dev2017_test2014.csv"


def experiment_split(frame: pd.DataFrame):
    train = frame[frame.year.isin(TRAIN_YEARS)]
    validation = frame[frame.year == DEV_YEAR]
    test = frame[frame.year == TEST_YEAR]
    return train, validation, test


def evaluation_frame(frame: pd.DataFrame, fraction: float | None):
    return frame if fraction is None else at_match_fraction(frame, fraction)


def metrics(model: str, fraction: float | None, test, probability, **params):
    y = test.y.to_numpy()
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return {
        "evaluation": "all_points" if fraction is None else f"{round(100*fraction)}%",
        "match_fraction": np.nan if fraction is None else fraction,
        "model": model,
        "accuracy": accuracy_score(y, probability >= 0.5),
        "logloss": log_loss(y, probability, labels=[0, 1]),
        "test_rows": len(test),
        "test_matches": test.match_id.nunique(),
        "train_years": ",".join(map(str, TRAIN_YEARS)),
        "dev_year": DEV_YEAR,
        "test_year": TEST_YEAR,
        "parameters": json.dumps(params, sort_keys=True),
    }


def tune_symmetric(validation):
    best_p, best_loss = None, float("inf")
    for serve_probability in (0.60, 0.61, 0.62, 0.63, 0.64, 0.65):
        probability = predict(
            validation, serve_probability, serve_probability, STATE
        )
        loss = val_logloss(validation.y.to_numpy(), probability)
        if loss < best_loss:
            best_p, best_loss = serve_probability, loss
    return best_p, best_loss


def evaluate_no_elo(full: pd.DataFrame, fraction: float | None):
    frame = evaluation_frame(full, fraction)
    train, validation, test = experiment_split(frame)
    rows = []

    serve_probability, _ = tune_symmetric(validation)
    probability = predict(test, serve_probability, serve_probability, STATE)
    rows.append(
        metrics(
            "symmetric_markov", fraction, test, probability,
            serve_probability=serve_probability,
        )
    )

    x_train = baseline_gbm.feature_matrix(train)
    x_validation = baseline_gbm.feature_matrix(validation)
    best_params, best_loss = None, float("inf")
    for params in baseline_gbm.MODEL_GRID:
        model = baseline_gbm.fit_model(x_train, train.y.to_numpy(), params)
        probability = model.predict_proba(x_validation)[:, 1]
        loss = val_logloss(validation.y.to_numpy(), probability)
        if loss < best_loss:
            best_params, best_loss = params, loss

    train_validation = pd.concat([train, validation], ignore_index=True)
    model = baseline_gbm.fit_model(
        baseline_gbm.feature_matrix(train_validation),
        train_validation.y.to_numpy(),
        best_params,
    )
    probability = model.predict_proba(baseline_gbm.feature_matrix(test))[:, 1]
    rows.append(metrics("baseline_gbm", fraction, test, probability, **best_params))
    return rows


def evaluate_with_elo(full: pd.DataFrame, fraction: float | None):
    frame = evaluation_frame(full, fraction)
    train, validation, test = experiment_split(frame)
    rows = []

    (base, slope), _ = residual.tune_markov_params(validation)
    markov_probability = residual.markov_prediction(test, base, slope)
    rows.append(
        metrics(
            "asymmetric_markov", fraction, test, markov_probability,
            base=base, slope=slope,
        )
    )

    kappa, _ = residual.tune_serve_shrink_kappa(validation, base, slope)
    shrink_probability = residual.serve_shrink_prediction(
        test, base, slope, kappa
    )
    rows.append(
        metrics(
            "serve_shrink_markov", fraction, test, shrink_probability,
            base=base, slope=slope, kappa=kappa,
        )
    )

    model_params, _, columns = residual.tune_residual_model(
        train, validation, base, slope, kappa, random_state=42
    )
    model = residual.refit_on_train_val(
        train, validation, base, slope, kappa, columns, model_params,
        random_state=42,
    )
    x_test, _ = residual.make_features(
        test, base, slope, kappa, columns
    )
    residual_probability = model.predict_proba(x_test)[:, 1]
    rows.append(
        metrics(
            "markov_ensemble", fraction, test, residual_probability,
            base=base, slope=slope, kappa=kappa, **model_params,
        )
    )
    return rows


def print_rows(rows):
    for row in rows:
        print(
            f"  {row['model']:22s} accuracy={row['accuracy']:.2%} "
            f"logloss={row['logloss']:.4f} rows={row['test_rows']:,}",
            flush=True,
        )


def main():
    rows = []

    print("Loading non-Elo data for symmetric Markov and baseline GBM...", flush=True)
    no_elo = load(with_elo=False)
    for fraction in CHECKPOINTS:
        label = "all points" if fraction is None else f"{fraction:.0%}"
        print(f"\n{label}: non-Elo models", flush=True)
        result = evaluate_no_elo(no_elo, fraction)
        rows.extend(result)
        print_rows(result)
    del no_elo

    print("\nLoading Elo data for structured and residual models...", flush=True)
    with_elo = load(with_elo=True)
    for fraction in CHECKPOINTS:
        label = "all points" if fraction is None else f"{fraction:.0%}"
        print(f"\n{label}: Elo/Markov models", flush=True)
        result = evaluate_with_elo(with_elo, fraction)
        rows.extend(result)
        print_rows(result)

    results = pd.DataFrame(rows)
    order = pd.Categorical(
        results.evaluation,
        categories=["25%", "50%", "75%", "all_points"],
        ordered=True,
    )
    results = results.assign(_order=order).sort_values(
        ["_order", "model"]
    ).drop(columns="_order")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
