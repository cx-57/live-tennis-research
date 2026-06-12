"""Symmetric Markov anchor.

Points are treated as independent and the two players as equally strong, so the live win
probability is a pure function of the score. The single serve-win probability is calibrated on
the validation season. This is the structural baseline the other models are measured against.
"""
from common import load, split, report, val_logloss, STATE
from markov import predict


def main():
    train, val, test = split(load(with_elo=False))

    best_p, best_ll = None, float("inf")
    for p in [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]:
        ll = val_logloss(val.y.values, predict(val, p, p, STATE))
        if ll < best_ll:
            best_p, best_ll = p, ll
    print(f"calibrated serve-win probability p = {best_p}")

    report("symmetric Markov", test.y.values, predict(test, best_p, best_p, STATE))


if __name__ == "__main__":
    main()
