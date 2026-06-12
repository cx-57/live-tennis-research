"""Build the point-state table from the Grand Slam point-by-point files.

One row per point played, with the score state after that point (which is known live) and a
label for whether player 1 went on to win the match. The match winner and the best-of length
are derived from the cumulative set winners in the points, because the matches-file metadata is
missing for several recent years. Serve-won rates are accumulated using only earlier points, so
every feature on a row is available at that moment in the match.
"""
import glob
import os
import re
import numpy as np
import pandas as pd

from common import SLAM_DIR, POINTS, ART

SCOREMAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def conv_score(v):
    s = str(v).strip()
    if s in SCOREMAP:
        return SCOREMAP[s]
    try:
        return int(s)            # tiebreak points are already numeric
    except ValueError:
        return np.nan


def process_match(g, mid, year, slam):
    g = g.reset_index(drop=True)
    sw = pd.to_numeric(g.SetWinner, errors="coerce").fillna(0).astype(int)
    p1_sets = (sw == 1).cumsum().values
    p2_sets = (sw == 2).cumsum().values
    f1, f2 = int((sw == 1).sum()), int((sw == 2).sum())
    if f1 == f2:                                       # incomplete or unscored match
        return None
    winner = 1 if f1 > f2 else 2
    sets_to_win = max(f1, f2)
    if sets_to_win not in (2, 3):                      # retirement before a full result
        return None
    best_of = 2 * sets_to_win - 1

    server = pd.to_numeric(g.PointServer, errors="coerce").fillna(0).astype(int).values
    pw = pd.to_numeric(g.PointWinner, errors="coerce").fillna(0).astype(int).values
    p1g = pd.to_numeric(g.P1GamesWon, errors="coerce").fillna(0).astype(int).values
    p2g = pd.to_numeric(g.P2GamesWon, errors="coerce").fillna(0).astype(int).values
    setno = pd.to_numeric(g.SetNo, errors="coerce").fillna(1).astype(int).values
    s1 = g.P1Score.map(conv_score).values
    s2 = g.P2Score.map(conv_score).values
    tiebreak = ((p1g == 6) & (p2g == 6)).astype(int)
    point_no = np.arange(1, len(g) + 1)

    # running serve-won rate per player, using only points before the current one
    def run_rate(is_serve_pt, is_serve_win):
        n = np.concatenate([[0], np.cumsum(is_serve_pt.astype(float))[:-1]])
        w = np.concatenate([[0], np.cumsum(is_serve_win.astype(float))[:-1]])
        return np.where(n > 0, w / np.maximum(n, 1), 0.5), n

    p1_serve_rate, p1_serve_n = run_rate(server == 1, (server == 1) & (pw == 1))
    p2_serve_rate, p2_serve_n = run_rate(server == 2, (server == 2) & (pw == 2))

    keep = (server > 0) & (pw > 0)
    df = pd.DataFrame(dict(
        match_id=mid, year=year, slam=slam, best_of=best_of, winner=winner,
        y=float(winner == 1), set_no=setno, p1_sets=p1_sets, p2_sets=p2_sets,
        p1_games=p1g, p2_games=p2g, server=server, p1_score=s1, p2_score=s2,
        tiebreak=tiebreak, point_no=point_no,
        p1_serve_rate=p1_serve_rate, p1_serve_n=p1_serve_n,
        p2_serve_rate=p2_serve_rate, p2_serve_n=p2_serve_n,
    ))
    return df[keep]


def main():
    os.makedirs(ART, exist_ok=True)
    out = []
    for pf in sorted(glob.glob(f"{SLAM_DIR}/*-points.csv")):
        base = os.path.basename(pf).replace("-points.csv", "")
        m = re.match(r"(\d{4})-(.+)", base)
        year, slam = int(m.group(1)), m.group(2)
        p = pd.read_csv(pf, low_memory=False)
        if "match_id" not in p.columns:
            continue
        for mid, g in p.groupby("match_id"):
            r = process_match(g, mid, year, slam)
            if r is not None and len(r) > 20:
                out.append(r)
    df = pd.concat(out, ignore_index=True)

    df["sets_diff"] = df.p1_sets - df.p2_sets
    df["games_diff"] = df.p1_games - df.p2_games
    df["p1_serving"] = (df.server == 1).astype(int)
    df["score_diff"] = df.p1_score - df.p2_score
    df["pts_played"] = df.point_no
    df.to_parquet(POINTS)
    print(f"wrote {POINTS}: {df.shape}, matches={df.match_id.nunique()}, "
          f"years {df.year.min()}-{df.year.max()}")

    print("player 1 win rate:", round(df.groupby('match_id').y.first().mean(), 4))
    print("win prob by set lead:")
    print(df.groupby('sets_diff').y.mean().round(3).to_string())


if __name__ == "__main__":
    main()
