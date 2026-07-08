"""Build the point-state table from the Grand Slam point-by-point files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import os
import re

import numpy as np
import pandas as pd

from src.common import SLAM_DIR, POINTS, ART

# Convert tennis point text into numbers 
SCOREMAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def conv_score(v):
    s = str(v).strip()

    if s in SCOREMAP:
        return SCOREMAP[s]

    try:
        return int(s)
    except ValueError:
        return np.nan


"""Convert one raw match into point-level model features.

Each row of the output represents the state before/at a point in the match.
The function builds score-state features, live serve statistics, rally/serve
features, and the final match-winner label used for supervised learning.
 """

def process_match(g, mid, year, slam):
    g = g.reset_index(drop=True)

    # Helpers to encode numeric and categorical values 
    def numeric_values(col, default=0.0):
        if col not in g:
            return np.full(len(g), default)

        return pd.to_numeric(g[col], errors="coerce").fillna(default).values

    def category_codes(col):
        if col not in g:
            return np.full(len(g), -1)

        s = g[col].astype("string").fillna("__missing__")
        return (pd.util.hash_pandas_object(s, index=False).to_numpy() % 1024).astype(int)

    # Determine completed sets, match winner, and match format.
    sw = pd.to_numeric(g.SetWinner, errors="coerce").fillna(0).astype(int)

    p1_sets = (sw == 1).cumsum().values
    p2_sets = (sw == 2).cumsum().values

    f1 = int((sw == 1).sum())
    f2 = int((sw == 2).sum())

    if f1 == f2:
        return None

    winner = 1 if f1 > f2 else 2
    sets_to_win = max(f1, f2)

    if sets_to_win not in (2, 3):
        return None

    best_of = 2 * sets_to_win - 1

    # Extract point-level score state from dataset
    server = pd.to_numeric(g.PointServer, errors="coerce").fillna(0).astype(int).values
    pw = pd.to_numeric(g.PointWinner, errors="coerce").fillna(0).astype(int).values

    p1g = pd.to_numeric(g.P1GamesWon, errors="coerce").fillna(0).astype(int).values
    p2g = pd.to_numeric(g.P2GamesWon, errors="coerce").fillna(0).astype(int).values

    setno = pd.to_numeric(g.SetNo, errors="coerce").fillna(1).astype(int).values

    s1 = g.P1Score.map(conv_score).values
    s2 = g.P2Score.map(conv_score).values

    # Extract optional live-match features if they exist in dataset
    rally = numeric_values("RallyCount", np.nan)

    if np.isnan(rally).all():
        rally = numeric_values("Rally", np.nan)

    speed_kmh = numeric_values("Speed_KMH", np.nan)
    serve_number = numeric_values("ServeNumber", 0)

    tiebreak = ((p1g == 6) & (p2g == 6)).astype(int)
    point_no = np.arange(1, len(g) + 1)

    # Prior-stat helpers to compute live stats using only points before the current point
    # avoids data leakage 
    def prior_sum(values):
        return np.concatenate([[0], np.cumsum(values.astype(float))[:-1]])

    def run_rate(is_serve_pt, is_serve_win):
        n = prior_sum(is_serve_pt)
        w = prior_sum(is_serve_win)

        return np.where(n > 0, w / np.maximum(n, 1), 0.5), n

    def run_mean(values, is_sample=None, default=0.0):
        if is_sample is None:
            is_sample = np.ones(len(values), dtype=bool)

        valid = is_sample & np.isfinite(values)

        n = prior_sum(valid)
        total = prior_sum(np.where(valid, values, 0.0))

        return np.where(n > 0, total / np.maximum(n, 1), default), n

    def rolling_prior_mean(values, window, default=0.0):
        prior = pd.Series(values).shift(1)
        return prior.rolling(window, min_periods=1).mean().fillna(default).to_numpy()

    def rolling_prior_rate(is_sample, is_success, window, default=0.5):
        sample = pd.Series(is_sample.astype(float)).shift(1)
        success = pd.Series(is_success.astype(float)).shift(1)

        n = sample.rolling(window, min_periods=1).sum()
        w = success.rolling(window, min_periods=1).sum()

        return (w / n.replace(0, np.nan)).fillna(default).to_numpy()

    # Live serve performance so far for each player
    # These are the main inputs for the serve-shrink model
    p1_serve_rate, p1_serve_n = run_rate(server == 1, (server == 1) & (pw == 1))
    p2_serve_rate, p2_serve_n = run_rate(server == 2, (server == 2) & (pw == 2))

    p1_ace = numeric_values("P1Ace", 0).astype(int)
    p2_ace = numeric_values("P2Ace", 0).astype(int)

    p1_ace_rate, p1_ace_serve_n = run_rate(server == 1, p1_ace == 1)
    p2_ace_rate, p2_ace_serve_n = run_rate(server == 2, p2_ace == 1)

    p1_aces = prior_sum(p1_ace == 1)
    p2_aces = prior_sum(p2_ace == 1)

    rally_avg, rally_n = run_mean(rally)

    p1_serve_rally_avg, _ = run_mean(rally, server == 1)
    p2_serve_rally_avg, _ = run_mean(rally, server == 2)

    recent_rally_avg = rolling_prior_mean(rally, window=12)

    p1_recent_ace_rate = rolling_prior_rate(server == 1, p1_ace == 1, window=24)
    p2_recent_ace_rate = rolling_prior_rate(server == 2, p2_ace == 1, window=24)

    # Keep only valid points with known server and point winner
    keep = (server > 0) & (pw > 0)

    # Extra live features that are extracted from the GS point-by-point dataset
    df = pd.DataFrame(
        dict(
            match_id=mid,
            year=year,
            slam=slam,
            best_of=best_of,
            winner=winner,
            y=float(winner == 1),
            set_no=setno,
            p1_sets=p1_sets,
            p2_sets=p2_sets,
            p1_games=p1g,
            p2_games=p2g,
            server=server,
            p1_score=s1,
            p2_score=s2,
            tiebreak=tiebreak,
            point_no=point_no,
            point_winner=pw,
            serve_won=(server == pw).astype(int),
            speed_kmh=speed_kmh,
            serve_number=serve_number,
            serve_width=category_codes("ServeWidth"),
            serve_depth=category_codes("ServeDepth"),
            return_depth=category_codes("ReturnDepth"),
            p1_serve_rate=p1_serve_rate,
            p1_serve_n=p1_serve_n,
            p2_serve_rate=p2_serve_rate,
            p2_serve_n=p2_serve_n,
            rally_avg=rally_avg,
            rally_n=rally_n,
            recent_rally_avg=recent_rally_avg,
            p1_serve_rally_avg=p1_serve_rally_avg,
            p2_serve_rally_avg=p2_serve_rally_avg,
            p1_aces=p1_aces,
            p2_aces=p2_aces,
            p1_ace_rate=p1_ace_rate,
            p1_ace_serve_n=p1_ace_serve_n,
            p2_ace_rate=p2_ace_rate,
            p2_ace_serve_n=p2_ace_serve_n,
            p1_recent_ace_rate=p1_recent_ace_rate,
            p2_recent_ace_rate=p2_recent_ace_rate,
        )
    )

    return df[keep]


def main():
    os.makedirs(ART, exist_ok=True)

    out = []

    for pf in sorted(glob.glob(f"{SLAM_DIR}/*-points.csv")):
        base = os.path.basename(pf).replace("-points.csv", "")
        m = re.match(r"(\d{4})-(.+)", base)

        year = int(m.group(1))
        slam = m.group(2)

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

    print(
        f"wrote {POINTS}: {df.shape}, "
        f"matches={df.match_id.nunique()}, "
        f"years {df.year.min()}-{df.year.max()}"
    )

    print("player 1 win rate:", round(df.groupby("match_id").y.first().mean(), 4))

    print("win prob by set lead:")
    print(df.groupby("sets_diff").y.mean().round(3).to_string())


if __name__ == "__main__":
    main()
