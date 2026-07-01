"""Live win-probability model combining the pre-match prior with in-match serving.

Each player's serve-win probability is a blend of the Elo-implied prior and the serve-won rate
observed so far in the match. The blend is a Bayesian shrink with a pseudo-count kappa: early in
the match, when few serve points have been seen, the prior dominates; as serve points accumulate
the observed rate takes over. The Markov recursion turns the two serve probabilities and the
current score into a win probability. The base and slope come from the asymmetric-Markov
calibration; kappa is calibrated here on the validation season.
"""
import numpy as np
import json
import joblib
import os

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import load, split, report, val_logloss, STATE, ART
from markov import predict

BASE, SLOPE = 0.61, 1.3e-4
MODEL_PATH = os.path.join(ART, "serve_shrink_model.json")
CALIBRATOR_PATH = os.path.join(ART, "serve_shrink_calibrator.joblib")
IMPORTANCE_PATH = os.path.join(ART, "serve_shrink_feature_importance.csv")
MODEL_FEATURES = [
    "elo_diff",
    "p1_serve_rate", "p1_serve_n",
    "p2_serve_rate", "p2_serve_n",
]
LIVE_CONTEXT_FEATURES = [
    "rally_avg", "rally_n", "recent_rally_avg",
    "p1_serve_rally_avg", "p2_serve_rally_avg",
    "p1_aces", "p2_aces",
    "p1_ace_rate", "p1_ace_serve_n",
    "p2_ace_rate", "p2_ace_serve_n",
    "p1_recent_ace_rate", "p2_recent_ace_rate",
]
FEATURES = MODEL_FEATURES + LIVE_CONTEXT_FEATURES + STATE
CALIBRATION_FEATURES = ["base_markov_logit"] + MODEL_FEATURES + LIVE_CONTEXT_FEATURES


def serve_probs(df, base, slope, kappa):
    edge = np.clip(slope * df.elo_diff.to_numpy(), -0.15, 0.15)
    prior_a, prior_b = base + edge, base - edge
    r1, n1 = df.p1_serve_rate.to_numpy(), df.p1_serve_n.to_numpy()
    r2, n2 = df.p2_serve_rate.to_numpy(), df.p2_serve_n.to_numpy()
    pa = (n1 * r1 + kappa * prior_a) / (n1 + kappa)
    pb = (n2 * r2 + kappa * prior_b) / (n2 + kappa)
    return np.clip(pa, 0.45, 0.88), np.clip(pb, 0.45, 0.88)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def base_predict(df, params):
    return predict(
        df,
        *serve_probs(df, params["base"], params["slope"], params["kappa"]),
        STATE,
    )


def calibration_matrix(df, params):
    base = base_predict(df, params)
    return np.column_stack([logit(base), df[MODEL_FEATURES + LIVE_CONTEXT_FEATURES].to_numpy()])


def model_predict(df, params, calibrator=None):
    if calibrator is None:
        return base_predict(df, params)
    return calibrator.predict_proba(calibration_matrix(df, params))[:, 1]


def fit_calibrator(train, val, params):
    best, best_ll = None, float("inf")
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        calibrator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=c, max_iter=2000, random_state=42)),
        ])
        calibrator.fit(calibration_matrix(train, params), train.y.values)
        ll = val_logloss(val.y.values, model_predict(val, params, calibrator))
        if ll < best_ll:
            best, best_ll = c, ll

    final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=best, max_iter=2000, random_state=42)),
    ])
    train_val = np.concatenate([
        calibration_matrix(train, params),
        calibration_matrix(val, params),
    ])
    y = np.concatenate([train.y.values, val.y.values])
    final.fit(train_val, y)
    print(f"calibrated logistic layer C = {best}")
    return final, {"type": "logistic_regression", "C": best, "val_logloss": best_ll}


def save_model(params, calibrator, calibration_meta, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    payload = {
        "model": "serve-shrink-calibrated",
        "params": params,
        "features": FEATURES,
        "state_features": STATE,
        "serve_probability_features": MODEL_FEATURES,
        "available_live_context_features": LIVE_CONTEXT_FEATURES,
        "calibration_features": CALIBRATION_FEATURES,
        "calibration": {
            **calibration_meta,
            "artifact": CALIBRATOR_PATH,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved model to {path}")
    print(f"saved calibration layer to {CALIBRATOR_PATH}")


def permutation_importance(df, y, params, calibrator=None, features=FEATURES, repeats=5, seed=7):
    rng = np.random.default_rng(seed)
    baseline = val_logloss(y, model_predict(df, params, calibrator))
    rows = []

    for feature in features:
        losses = []
        for _ in range(repeats):
            shuffled = df.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            losses.append(val_logloss(y, model_predict(shuffled, params, calibrator)))

        losses = np.array(losses)
        rows.append({
            "feature": feature,
            "baseline_logloss": baseline,
            "permuted_logloss_mean": losses.mean(),
            "importance_logloss_increase": losses.mean() - baseline,
            "permuted_logloss_std": losses.std(ddof=0),
        })

    rows.sort(key=lambda row: row["importance_logloss_increase"], reverse=True)
    return rows


def save_feature_importance(rows, path=IMPORTANCE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "feature,baseline_logloss,permuted_logloss_mean,"
            "importance_logloss_increase,permuted_logloss_std\n"
        )
        for row in rows:
            f.write(
                f"{row['feature']},"
                f"{row['baseline_logloss']:.8f},"
                f"{row['permuted_logloss_mean']:.8f},"
                f"{row['importance_logloss_increase']:.8f},"
                f"{row['permuted_logloss_std']:.8f}\n"
            )
    print(f"saved feature importance to {path}")


def print_feature_importance(rows, top_n=12):
    print("\nfeature importance (test permutation log-loss increase)")
    for row in rows[:top_n]:
        print(f"{row['feature']:16s} {row['importance_logloss_increase']:.6f}")


def main():
    train, val, test = split(load(with_elo=True, match_fraction=0.70))

    best_k, best_ll = None, float("inf")
    for kappa in [40, 80, 160, 320, 640]:
        ll = val_logloss(val.y.values, predict(val, *serve_probs(val, BASE, SLOPE, kappa), STATE))
        if ll < best_ll:
            best_k, best_ll = kappa, ll
    print(f"calibrated kappa = {best_k}")

    params = {"base": BASE, "slope": SLOPE, "kappa": best_k}
    report("serve-shrink model", test.y.values,
           base_predict(test, params))

    calibrator, calibration_meta = fit_calibrator(train, val, params)
    report("serve-shrink calibrated", test.y.values,
           model_predict(test, params, calibrator))

    save_model(params, calibrator, calibration_meta)
    importance = permutation_importance(test, test.y.values, params, calibrator)
    save_feature_importance(importance)
    print_feature_importance(importance)


if __name__ == "__main__":
    main()
