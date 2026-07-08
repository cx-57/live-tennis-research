"""Hierarchical point-game-set-match win-probability recursion

Player 1 wins a point on its own serve with probability pa, player 2 with probability pb
Win probability is always reported from player 1's perspective
The symmetric case pa == pb recovers the equal-players anchor
"""
from functools import lru_cache
import numpy as np


def make_solver(pa, pb):
    """Build one cached Markov solver for a fixed pair of serve probabilities"""

    @lru_cache(None)
    def game(s, i, j):
        # Server game-win probability from point score i:j
        if i >= 4 and i - j >= 2:
            return 1.0
        if j >= 4 and j - i >= 2:
            return 0.0
        if i >= 3 and j >= 3:
            deuce = s * s / (s * s + (1 - s) * (1 - s))
            if i == j:
                return deuce
            return s + (1 - s) * deuce if i > j else s * deuce
        return s * game(s, i + 1, j) + (1 - s) * game(s, i, j + 1)

    holdA = game(pa, 0, 0)
    holdB = game(pb, 0, 0)

    def tb_server(k):
        return 1 if (((k + 1) // 2) % 2 == 0) else 2

    @lru_cache(None)
    def tb(a, b):
        # Player 1 tiebreak-win probability from tiebreak score a:b
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0

        k = a + b
        pt_win = lambda kk: pa if tb_server(kk) == 1 else (1 - pb)

        if a >= 6 and b >= 6 and a == b:
            w1, w2 = pt_win(k), pt_win(k + 1)
            return w1 * w2 / (w1 * w2 + (1 - w1) * (1 - w2))

        w = pt_win(k)
        return w * tb(a + 1, b) + (1 - w) * tb(a, b + 1)

    def tbwp(a, b, p1_served_first):
        return tb(a, b) if p1_served_first else 1.0 - tb(b, a)

    @lru_cache(None)
    def setwp(g1, g2, p1serve):
        # Player 1 set-win probability from game score g1:g2
        if g1 >= 6 and g1 - g2 >= 2:
            return 1.0
        if g2 >= 6 and g2 - g1 >= 2:
            return 0.0
        if g1 == 6 and g2 == 6:
            return tbwp(0, 0, bool(p1serve))

        if g1 >= 7 and g2 >= 7 and g1 == g2:
            wA, wB = holdA, 1 - holdB
            return wA * wB / (wA * wB + (1 - wA) * (1 - wB))

        p1_game = holdA if p1serve else (1 - holdB)
        return p1_game * setwp(g1 + 1, g2, not p1serve) + (1 - p1_game) * setwp(
            g1, g2 + 1, not p1serve
        )

    set0 = 0.5 * (setwp(0, 0, True) + setwp(0, 0, False))

    @lru_cache(None)
    def matchwp(s1, s2, best_of, setp):
        # Player 1 match-win probability from set score s1:s2
        need = (best_of + 1) // 2
        if s1 >= need:
            return 1.0
        if s2 >= need:
            return 0.0
        return setp * matchwp(s1 + 1, s2, best_of, set0) + (1 - setp) * matchwp(
            s1, s2 + 1, best_of, set0
        )

    def wp(best_of, s1, s2, g1, g2, p1serve, p1pts, p2pts, tie):
        if tie:
            k = p1pts + p2pts
            p1_served_first = (tb_server(k) == 1) == bool(p1serve)
            setp = tbwp(p1pts, p2pts, p1_served_first)
        else:
            if p1serve:
                p1_game = game(pa, p1pts, p2pts)
            else:
                p1_game = 1 - game(pb, p2pts, p1pts)

            setp = p1_game * setwp(g1 + 1, g2, not p1serve) + (1 - p1_game) * setwp(
                g1, g2 + 1, not p1serve
            )

        return matchwp(s1, s2, best_of, setp)

    return wp


_solvers = {}


def solver(pa, pb):
    # Reuse solvers for rounded serve probabilities
    key = (round(pa, 3), round(pb, 3))
    s = _solvers.get(key)

    if s is None:
        s = make_solver(*key)
        _solvers[key] = s

    return s


def predict(df, pa, pb, state_cols):
    """Win probability for every row in df"""
    st = df[state_cols].to_numpy()

    pa = np.round(np.broadcast_to(pa, len(df)), 3)
    pb = np.round(np.broadcast_to(pb, len(df)), 3)

    out = np.empty(len(df))
    memo = {}

    # Memoize repeated score states so each unique state is solved once
    for i in range(len(df)):
        bo, s1, s2, g1, g2, p1s, q1, q2, tie = st[i]

        key = (
            pa[i],
            pb[i],
            int(bo),
            int(s1),
            int(s2),
            int(g1),
            int(g2),
            bool(p1s),
            int(q1),
            int(q2),
            bool(tie),
        )

        v = memo.get(key)

        if v is None:
            try:
                v = solver(pa[i], pb[i])(
                    int(bo),
                    int(s1),
                    int(s2),
                    int(g1),
                    int(g2),
                    bool(p1s),
                    int(q1),
                    int(q2),
                    bool(tie),
                )
            except RecursionError:
                need = (int(bo) + 1) // 2
                v = 1.0 if s1 >= need else 0.0 if s2 >= need else 0.5

            memo[key] = v

        out[i] = v

    return np.clip(out, 1e-6, 1 - 1e-6)
