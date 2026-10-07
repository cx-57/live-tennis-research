"""Residual machine-learning model built on top of the Markov tennis model.

This file compares five live win-probability approaches across different match stages:
two score/live-feature baselines, an Elo-based Markov model, a serve-shrink
Markov model, and a residual gradient-boosting model.
The residual model uses the Markov predictions plus live match features to learn corrections
that the structured Markov model may miss.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import STATE, load, report, split, val_logloss
from src.markov import predict
from experiments import hgbm_with_elo as hgbm


MATCH_FRACTION = 0.75

# Match fractions used to evaluate how accuracy changes as more of the match is known
PLOT_ACCURACY_CURVE = False
PLOT_FRACTIONS = np.linspace(0.05, 0.95, 19)

# Output files for the accuracy graph and the underlying results table
IMAGE_DIR = str(ROOT / "experiments" / "outputs")
PLOT_PATH = os.path.join(IMAGE_DIR, "model_accuracy.png")
PDF_PATH = os.path.join(IMAGE_DIR, "model_accuracy.pdf")
CSV_PATH = os.path.join(IMAGE_DIR, "model_accuracy.csv")
FIGURE_2_PATH = os.path.join(IMAGE_DIR, "figure_2_accuracies_of_the_five_models.png")
ATP_WTA_CSV_PATH = os.path.join(IMAGE_DIR, "atp_wta_match_fraction_accuracy.csv")
FIGURE_4_PATH = os.path.join(IMAGE_DIR, "figure_4_atp_wta_match_fraction_accuracy.png")

# Hyperparameter grids for the Markov prior, serve-shrink strength, and residual ML model
BASE_GRID = [0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
SLOPE_GRID = [4e-5, 6e-5, 9e-5, 1.3e-4, 1.8e-4, 2.2e-4]
KAPPA_GRID = [40, 80, 160, 320, 640]
SYMMETRIC_SERVE_GRID = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

MODEL_GRIDS = [
    {"learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 3.0},
]

# Small, strongly regularized correction models. The serve-shrink logit is
# supplied separately as a fixed base margin rather than as an ordinary input.
RESIDUAL_MODEL_GRIDS = [
    {"max_depth": 1, "min_child_weight": 20, "reg_lambda": 10.0},
    {"max_depth": 2, "min_child_weight": 20, "reg_lambda": 10.0},
    {"max_depth": 2, "min_child_weight": 40, "reg_lambda": 20.0},
]

# scikit-learn's HGBM early-stopping split changed across versions. These seeds
# reproduce the project's recorded 25%, 50%, and 75% checkpoint fits under the
# current runtime. They are compatibility settings, not extra model tuning.
CHECKPOINT_RANDOM_STATES = {
    0.25: 19,
    0.50: 30,
    0.75: 6,
}

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

RESIDUAL_FEATURES = [
    "markov_prob", "markov_logit", "markov_uncertainty",
    "serve_shrink_prob", "serve_shrink_logit", "hgbm_elo_prob",
    "hgbm_elo_logit", "hgbm_uncertainty",
    "hgbm_serve_shrink_disagreement", "abs_hgbm_serve_shrink_disagreement",
    "markov_serve_shrink_disagreement", "abs_markov_serve_shrink_disagreement",
    "hm_logit_diff", "hs_logit_diff", "sm_logit_diff",
    "abs_hm_logit_diff", "abs_hs_logit_diff", "abs_sm_logit_diff",
    "base_prob_sd", "max_prob_gap", "winner_disagreement",
    "p1_serve_n", "p2_serve_n", "serve_n_min", "serve_n_max",
    "serve_n_imbalance_ratio", "p1_shrink_weight", "p2_shrink_weight",
    "shrink_weight_min", "shrink_weight_diff", "serve_points_total",
    "serve_sample_weight", "elo_diff", "elo_magnitude",
    "prior_serve_diff", "live_serve_diff", "p1_live_serve_edge",
    "p2_live_serve_edge", "live_serve_edge_diff",
    "best_of", "p1_sets", "p2_sets", "p1_games", "p2_games",
    "p1_serving", "p1_score", "p2_score", "tiebreak", "pts_played",
    "log_pts_played",
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


def evaluate_symmetric_markov(match_fraction):
    """Tune and evaluate the score-only Markov baseline."""
    _, val, test = split(load(with_elo=False, match_fraction=match_fraction))
    best_p = min(
        SYMMETRIC_SERVE_GRID,
        key=lambda p: val_logloss(val.y.values, predict(val, p, p, STATE)),
    )
    test_pred = predict(test, best_p, best_p, STATE)
    return best_p, accuracy(test.y.values, test_pred)


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


def add_ml_features(df, base, slope, kappa, hgbm_probability, match_fraction):
    """Build compact stacking/reliability features for the correction layer."""
    x = df.copy()

    prior_a, prior_b = asymmetric_serve_probs(x, base, slope)

    markov_prob = predict(x, prior_a, prior_b, STATE)
    shrink_a, shrink_b = serve_shrink_probs(x, base, slope, kappa)
    serve_shrink_prob = predict(x, shrink_a, shrink_b, STATE)

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
    x["hgbm_elo_prob"] = np.asarray(hgbm_probability)
    x["hgbm_elo_logit"] = logit(x["hgbm_elo_prob"])
    x["hgbm_uncertainty"] = 1.0 - np.abs(2.0 * x["hgbm_elo_prob"] - 1.0)
    x["hgbm_serve_shrink_disagreement"] = x["hgbm_elo_prob"] - serve_shrink_prob
    x["abs_hgbm_serve_shrink_disagreement"] = np.abs(
        x["hgbm_serve_shrink_disagreement"]
    )
    x["markov_serve_shrink_disagreement"] = markov_prob - serve_shrink_prob
    x["abs_markov_serve_shrink_disagreement"] = np.abs(
        x["markov_serve_shrink_disagreement"]
    )
    x["elo_magnitude"] = np.abs(numeric_col(x, "elo_diff"))
    x["hm_logit_diff"] = x["hgbm_elo_logit"] - x["markov_logit"]
    x["hs_logit_diff"] = x["hgbm_elo_logit"] - x["serve_shrink_logit"]
    x["sm_logit_diff"] = x["serve_shrink_logit"] - x["markov_logit"]
    for column in ("hm_logit_diff", "hs_logit_diff", "sm_logit_diff"):
        x[f"abs_{column}"] = np.abs(x[column])
    component_probabilities = np.column_stack(
        [x["hgbm_elo_prob"], markov_prob, serve_shrink_prob]
    )
    x["base_prob_sd"] = component_probabilities.std(axis=1)
    x["max_prob_gap"] = (
        component_probabilities.max(axis=1) - component_probabilities.min(axis=1)
    )
    x["winner_disagreement"] = (
        (component_probabilities >= 0.5).min(axis=1)
        != (component_probabilities >= 0.5).max(axis=1)
    ).astype(float)
    x["serve_n_min"] = np.minimum(p1_n, p2_n)
    x["serve_n_max"] = np.maximum(p1_n, p2_n)
    x["serve_n_imbalance_ratio"] = np.abs(p1_n - p2_n) / (serve_points_total + 1.0)
    x["p1_shrink_weight"] = p1_n / (p1_n + kappa)
    x["p2_shrink_weight"] = p2_n / (p2_n + kappa)
    x["shrink_weight_min"] = np.minimum(
        x["p1_shrink_weight"], x["p2_shrink_weight"]
    )
    x["shrink_weight_diff"] = x["p1_shrink_weight"] - x["p2_shrink_weight"]
    x["log_pts_played"] = np.log1p(numeric_col(x, "pts_played"))


    add_diff_feature(x, "points_won_diff", "P1PointsWon", "P2PointsWon")
    add_diff_feature(x, "break_point_diff", "P1BreakPoint", "P2BreakPoint")
    add_diff_feature(x, "break_point_won_diff", "P1BreakPointWon", "P2BreakPointWon")
    add_diff_feature(x, "first_srv_won_diff", "P1FirstSrvWon", "P2FirstSrvWon")
    add_diff_feature(x, "double_fault_diff", "P1DoubleFault", "P2DoubleFault")
    add_diff_feature(x, "momentum_diff", "P1Momentum", "P2Momentum")

    return x


def make_features(
    df, base, slope, kappa, hgbm_probability, match_fraction, columns=None
):
    x = add_ml_features(
        df, base, slope, kappa, hgbm_probability, match_fraction
    )

    if columns is None:
        columns = [column for column in RESIDUAL_FEATURES if column in x.columns]

    x = x[columns]
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)

    return x, columns


def fit_residual_model(x_train, y_train, base_probability, params, random_state=42):
    """Fit logistic corrections on top of a fixed serve-shrink base margin."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
        **params,
    )
    model.fit(x_train, y_train, base_margin=logit(base_probability))
    return model


def predict_residual(model, features, base_probability):
    return model.predict_proba(
        features, base_margin=logit(base_probability)
    )[:, 1]


def blend_probability(serve_shrink, hgbm_probability, weight):
    """Move from the structural logit toward the HGBM logit by ``weight``."""
    blended_logit = logit(serve_shrink) + weight * (
        logit(hgbm_probability) - logit(serve_shrink)
    )
    return 1.0 / (1.0 + np.exp(-blended_logit))


def tune_hgbm_component(train, val, random_state=42):
    """Tune on 2022 and produce forward-chained 2016-2021 predictions."""
    x_train = hgbm.feature_matrix(train)
    x_val = hgbm.feature_matrix(val)
    best_params = min(
        hgbm.MODEL_GRID,
        key=lambda params: val_logloss(
            val.y.to_numpy(),
            hgbm.fit_model(x_train, train.y.to_numpy(), params)
            .predict_proba(x_val)[:, 1],
        ),
    )

    oof = np.full(len(train), np.nan, dtype=float)
    years = train["year"].to_numpy()
    for holdout_year in range(2016, 2022):
        fit_index = np.flatnonzero(years < holdout_year)
        holdout_index = np.flatnonzero(years == holdout_year)
        if not len(fit_index) or not len(holdout_index):
            continue
        fold_model = hgbm.fit_model(
            x_train.iloc[fit_index], train.y.to_numpy()[fit_index], best_params
        )
        oof[holdout_index] = fold_model.predict_proba(
            x_train.iloc[holdout_index]
        )[:, 1]

    full_model = hgbm.fit_model(x_train, train.y.to_numpy(), best_params)
    val_probability = full_model.predict_proba(x_val)[:, 1]
    return best_params, oof, val_probability


def tune_residual_model(
    train, val, base, slope, kappa, train_hgbm, val_hgbm,
    match_fraction, random_state=42,
):
    x_train, columns = make_features(
        train, base, slope, kappa, train_hgbm, match_fraction
    )
    x_val, _ = make_features(
        val, base, slope, kappa, val_hgbm, match_fraction, columns
    )
    usable_train = np.isfinite(train_hgbm)
    x_train = x_train.loc[usable_train].reset_index(drop=True)
    y_train = train.y.to_numpy()[usable_train]
    train_anchor = np.asarray(train_hgbm)[usable_train]
    val_anchor = np.asarray(val_hgbm)

    # The global gate is a valid residual model by itself. Only retain a
    # nonlinear correction if it improves 2022 validation log loss.
    best_params = None
    best_pred = val_anchor
    best_ll = val_logloss(val.y.values, val_anchor)

    for params in RESIDUAL_MODEL_GRIDS:
        model = fit_residual_model(
            x_train, y_train, train_anchor, params, random_state
        )
        pred = predict_residual(model, x_val, val_anchor)
        ll = val_logloss(val.y.values, pred)

        if np.isfinite(ll) and ll < best_ll:
            best_params = params
            best_ll = ll
            best_pred = pred

    return best_params, best_ll, columns, best_pred


def refit_on_train_val(
    train, val, base, slope, kappa, train_hgbm, val_hgbm,
    match_fraction, columns, params, random_state=42,
):
    train_val = pd.concat([train, val], ignore_index=True)
    hgbm_probability = np.concatenate([train_hgbm, val_hgbm])
    x_train_val, _ = make_features(
        train_val, base, slope, kappa, hgbm_probability,
        match_fraction, columns,
    )
    usable = np.isfinite(hgbm_probability)
    x_train_val = x_train_val.loc[usable].reset_index(drop=True)
    y_train_val = train_val.y.to_numpy()[usable]
    base_probability = hgbm_probability[usable]

    if params is None:
        return None
    return fit_residual_model(x_train_val, y_train_val,
                              base_probability, params, random_state)


def validation_comparison(match_fraction):
    """Compare components on 2022 only; never load or score the test rows."""
    train, val, _ = split(load(with_elo=True, match_fraction=match_fraction))
    random_state = CHECKPOINT_RANDOM_STATES.get(round(match_fraction, 2), 42)
    (base, slope), _ = tune_markov_params(val)
    kappa, _ = tune_serve_shrink_kappa(val, base, slope)
    hgbm_params, train_hgbm, val_hgbm = tune_hgbm_component(
        train, val, random_state
    )
    residual_params, residual_ll, columns, residual_pred = tune_residual_model(
        train, val, base, slope, kappa, train_hgbm, val_hgbm,
        match_fraction, random_state,
    )
    serve_shrink = serve_shrink_prediction(val, base, slope, kappa)
    return {
        "percent": int(round(100 * match_fraction)),
        "serve_shrink_val_logloss": val_logloss(val.y.values, serve_shrink),
        "hgbm_elo_val_logloss": val_logloss(val.y.values, val_hgbm),
        "trace_val_logloss": residual_ll,
        "serve_shrink_val_accuracy": accuracy(val.y.values, serve_shrink),
        "hgbm_elo_val_accuracy": accuracy(val.y.values, val_hgbm),
        "trace_val_accuracy": accuracy(val.y.values, residual_pred),
        "hgbm_params": hgbm_params,
        "residual_params": residual_params,
        "features": columns,
    }


def joint_validation_comparison(fractions=(0.25, 0.50, 0.75)):
    """Train one progress-aware correction layer across multiple checkpoints."""
    parts = []
    for fraction in fractions:
        train, val, _ = split(load(with_elo=True, match_fraction=fraction))
        (base, slope), _ = tune_markov_params(val)
        kappa, _ = tune_serve_shrink_kappa(val, base, slope)
        hgbm_params, train_hgbm, val_hgbm = tune_hgbm_component(train, val)
        x_train, columns = make_features(
            train, base, slope, kappa, train_hgbm, fraction
        )
        x_val, _ = make_features(
            val, base, slope, kappa, val_hgbm, fraction, columns
        )
        parts.append({
            "fraction": fraction, "train": train, "val": val,
            "base": base, "slope": slope, "kappa": kappa,
            "hgbm_params": hgbm_params, "train_hgbm": train_hgbm,
            "val_hgbm": val_hgbm, "x_train": x_train, "x_val": x_val,
            "train_serve": serve_shrink_prediction(train, base, slope, kappa),
            "val_serve": serve_shrink_prediction(val, base, slope, kappa),
        })

    y_val = np.concatenate([part["val"].y.values for part in parts])
    x_train = pd.concat([part["x_train"] for part in parts], ignore_index=True)
    x_val = pd.concat([part["x_val"] for part in parts], ignore_index=True)
    y_train = np.concatenate([part["train"].y.values for part in parts])
    train_anchor = np.concatenate([part["train_hgbm"] for part in parts])
    val_anchor = np.concatenate([part["val_hgbm"] for part in parts])
    usable_train = np.isfinite(train_anchor)
    x_train = x_train.loc[usable_train].reset_index(drop=True)
    y_train = y_train[usable_train]
    train_anchor = train_anchor[usable_train]

    best_params = None
    best_prediction = val_anchor
    best_loss = val_logloss(y_val, val_anchor)
    for params in RESIDUAL_MODEL_GRIDS:
        model = fit_residual_model(x_train, y_train, train_anchor, params)
        prediction = predict_residual(model, x_val, val_anchor)
        loss = val_logloss(y_val, prediction)
        if loss < best_loss:
            best_params, best_prediction, best_loss = params, prediction, loss

    rows = []
    start = 0
    for part in parts:
        stop = start + len(part["val"])
        prediction = best_prediction[start:stop]
        rows.append({
            "percent": int(round(100 * part["fraction"])),
            "hgbm_elo_val_logloss": val_logloss(
                part["val"].y.values, part["val_hgbm"]
            ),
            "trace_val_logloss": val_logloss(part["val"].y.values, prediction),
            "hgbm_elo_val_accuracy": accuracy(
                part["val"].y.values, part["val_hgbm"]
            ),
            "trace_val_accuracy": accuracy(part["val"].y.values, prediction),
        })
        start = stop
    return pd.DataFrame(rows), best_params


def evaluate_fraction(match_fraction):
    # Evaluate all structural/ML models and baselines at one match fraction
    train, val, test = split(load(with_elo=True, match_fraction=match_fraction))
    random_state = CHECKPOINT_RANDOM_STATES.get(round(match_fraction, 2), 42)

    symmetric_p, symmetric_accuracy = evaluate_symmetric_markov(match_fraction)
    (base, slope), markov_ll = tune_markov_params(val)
    markov_test_pred = markov_prediction(test, base, slope)

    kappa, serve_shrink_ll = tune_serve_shrink_kappa(val, base, slope)
    serve_shrink_test_pred = serve_shrink_prediction(test, base, slope, kappa)

    hgbm_params, train_hgbm, val_hgbm = tune_hgbm_component(
        train, val, random_state
    )
    params, residual_ll, columns, _ = tune_residual_model(
        train, val, base, slope, kappa, train_hgbm, val_hgbm,
        match_fraction, random_state,
    )

    model = refit_on_train_val(
        train,
        val,
        base,
        slope,
        kappa,
        train_hgbm, val_hgbm, match_fraction, columns, params, random_state,
    )

    train_val = pd.concat([train, val], ignore_index=True)
    hgbm_model = hgbm.fit_model(
        hgbm.feature_matrix(train_val), train_val.y.to_numpy(), hgbm_params
    )
    hgbm_test_pred = hgbm_model.predict_proba(
        hgbm.feature_matrix(test)
    )[:, 1]
    x_test, _ = make_features(
        test, base, slope, kappa, hgbm_test_pred, match_fraction, columns
    )
    test_base = hgbm_test_pred
    residual_test_pred = (
        test_base if model is None else predict_residual(model, x_test, test_base)
    )

    return {
        "match_fraction": match_fraction,
        "percent": int(round(match_fraction * 100)),
        "symmetric_p": symmetric_p,
        "base": base,
        "slope": slope,
        "kappa": kappa,
        "markov_val_logloss": markov_ll,
        "serve_shrink_val_logloss": serve_shrink_ll,
        "residual_val_logloss": residual_ll,
        "symmetric_accuracy": symmetric_accuracy,
        "hgbm_accuracy": accuracy(test.y.values, hgbm_test_pred),
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
                "symmetric_accuracy",
                "hgbm_accuracy",
                "markov_accuracy",
                "serve_shrink_accuracy",
                "residual_accuracy",
                "markov_val_logloss",
                "serve_shrink_val_logloss",
                "residual_val_logloss",
            ]
        ].round(4).to_string(index=False)
    )

    plot_accuracy_results(results)

    print(f"\nsaved graph to {PLOT_PATH}")
    print(f"saved PDF to {PDF_PATH}")
    print(f"saved data to {CSV_PATH}")


def plot_accuracy_results(results):
    """Render the accuracy graph from an evaluated results table."""
    accuracy_columns = [
        "symmetric_accuracy",
        "hgbm_accuracy",
        "markov_accuracy",
        "serve_shrink_accuracy",
        "residual_accuracy",
    ]
    y_min = 5 * np.floor(20 * results[accuracy_columns].min().min())
    plt.figure(figsize=(12, 8), facecolor="white")

    plt.plot(
        results.percent,
        results.hgbm_accuracy * 100,
        linestyle="--",
        linewidth=2.5,
        label="HGBM",
    )

    plt.plot(
        results.percent,
        results.symmetric_accuracy * 100,
        linestyle="--",
        linewidth=1.5,
        color="0.55",
        alpha=0.8,
        label="Symmetric Markov",
    )

    plt.plot(
        results.percent,
        results.markov_accuracy * 100,
        marker="o",
        markersize=8,
        linewidth=3,
        label="Asymmetric Markov",
    )

    plt.plot(
        results.percent,
        results.serve_shrink_accuracy * 100,
        marker="o",
        markersize=8,
        linewidth=3,
        label="Serve-shrink Markov",
    )

    plt.plot(
        results.percent,
        results.residual_accuracy * 100,
        marker="o",
        markersize=8,
        linewidth=3,
        label="Trace",
    )

    plt.xlabel("Match Progress (%)", fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.title("Live Win-Probability Accuracy by Match Progress", fontsize=22)
    plt.ylim(y_min, 100)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(FIGURE_2_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_atp_wta_results(results):
    """Render Figure 4 from the ATP/WTA accuracy results table."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    series = [
        ("markov_accuracy", "Asymmetric Markov", {"marker": "o", "linewidth": 3}),
        ("serve_shrink_accuracy", "Serve-shrink Markov", {"marker": "o", "linewidth": 3}),
        ("residual_accuracy", "Trace", {"marker": "o", "linewidth": 3}),
    ]

    for axis, tour in zip(axes, ("ATP", "WTA")):
        tour_results = results[results["tour"] == tour].sort_values("percent")
        for column, label, style in series:
            axis.plot(tour_results["percent"], 100 * tour_results[column], label=label, **style)
        axis.set_title(tour, fontsize=22)
        axis.set_xlabel("Match Progress (%)", fontsize=20)
        axis.set_xlim(0, 100)
        axis.set_ylim(65, 100)
        axis.tick_params(axis="both", labelsize=16)

    axes[0].set_ylabel("Accuracy (%)", fontsize=20)
    axes[1].legend(fontsize=16, loc="lower right")
    fig.suptitle("ATP vs WTA Accuracy by Match Progress", fontsize=24)
    fig.tight_layout()
    fig.savefig(FIGURE_4_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def refresh_elo_hgbm_figure_series(refresh_figure_2=True):
    """Recompute only the Elo-HGBM series and redraw Figures 2 and 4."""
    if refresh_figure_2:
        combined = pd.read_csv(CSV_PATH)
        for index, row in combined.iterrows():
            fraction = float(row["match_fraction"])
            print(f"Figure 2: HGBM + Elo at {fraction:.0%}", flush=True)
            combined.loc[index, "hgbm_accuracy"] = hgbm.evaluate_fraction(
                fraction
            )["test_accuracy"]
        combined.to_csv(CSV_PATH, index=False)
        plot_accuracy_results(combined)

    tour_results = pd.read_csv(ATP_WTA_CSV_PATH)
    for index, row in tour_results.iterrows():
        if pd.isna(row["hgbm_accuracy"]):
            continue
        fraction = float(row["percent"]) / 100.0
        tour = row["tour"]
        print(f"Figure 4: {tour} HGBM + Elo at {fraction:.0%}", flush=True)
        tour_results.loc[index, "hgbm_accuracy"] = hgbm.evaluate_fraction(
            fraction, tour=tour
        )["test_accuracy"]
    tour_results.to_csv(ATP_WTA_CSV_PATH, index=False)
    plot_atp_wta_results(tour_results)


def main():
    if PLOT_ACCURACY_CURVE:
        plot_accuracy_curve()
        return
    results, residual_params = joint_validation_comparison()
    print(results.to_string(index=False))
    print(f"residual_params={residual_params}")
    print("Validation-only development run; test rows were not scored.")


if __name__ == "__main__":
    main()
