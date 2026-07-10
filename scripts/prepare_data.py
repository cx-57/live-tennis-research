"""Build the point-state table from the Grand Slam point-by-point files."""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ART, POINTS, SLAM_DIR


SCOREMAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def conv_score(value):
    """Convert tennis point-score text to a numeric state."""
    score = str(value).strip()

    if score in SCOREMAP:
        return SCOREMAP[score]

    try:
        return int(score)
    except ValueError:
        return np.nan


def numeric_col(df, column):
    """Return one column as finite numeric values."""
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def add_universal_live_features(df):
    """Add model-independent difference and sample-size features."""
    pairs = {
        "points_won_diff": ("p1_points_won", "p2_points_won"),
        "break_point_diff": ("p1_break_points", "p2_break_points"),
        "break_point_won_diff": (
            "p1_break_points_won",
            "p2_break_points_won",
        ),
        "first_srv_won_diff": (
            "p1_first_serve_points_won",
            "p2_first_serve_points_won",
        ),
        "double_fault_diff": ("p1_double_faults", "p2_double_faults"),
        "momentum_diff": ("p1_momentum", "p2_momentum"),
        "live_serve_diff": ("p1_serve_rate", "p2_serve_rate"),
        "serve_points_balance": ("p1_serve_n", "p2_serve_n"),
    }

    for output, (p1_column, p2_column) in pairs.items():
        df[output] = numeric_col(df, p1_column) - numeric_col(df, p2_column)

    df["serve_points_total"] = (
        numeric_col(df, "p1_serve_n") + numeric_col(df, "p2_serve_n")
    )
    df["serve_sample_weight"] = np.log1p(df["serve_points_total"])

    return df


def process_match(g, mid, year, slam):
    """Convert one raw match into leakage-safe point-level model features."""
    g = g.reset_index(drop=True)

    def numeric_values(column, default=0.0):
        if column not in g:
            return np.full(len(g), default)

        return pd.to_numeric(g[column], errors="coerce").fillna(default).values

    def category_codes(column):
        if column not in g:
            return np.full(len(g), -1)

        values = g[column].astype("string").fillna("__missing__")
        return (
            pd.util.hash_pandas_object(values, index=False).to_numpy() % 1024
        ).astype(int)

    def prior_sum(values):
        """Cumulative total using only points before the current point."""
        return np.concatenate([[0], np.cumsum(values.astype(float))[:-1]])

    def run_rate(is_sample, is_success):
        count = prior_sum(is_sample)
        wins = prior_sum(is_success)
        return np.where(count > 0, wins / np.maximum(count, 1), 0.5), count

    def run_mean(values, is_sample=None, default=0.0):
        if is_sample is None:
            is_sample = np.ones(len(values), dtype=bool)

        valid = is_sample & np.isfinite(values)
        count = prior_sum(valid)
        total = prior_sum(np.where(valid, values, 0.0))
        return np.where(count > 0, total / np.maximum(count, 1), default), count

    def rolling_prior_mean(values, window, default=0.0):
        prior = pd.Series(values).shift(1)
        return prior.rolling(window, min_periods=1).mean().fillna(default).to_numpy()

    def rolling_prior_rate(is_sample, is_success, window, default=0.5):
        sample = pd.Series(is_sample.astype(float)).shift(1)
        success = pd.Series(is_success.astype(float)).shift(1)
        count = sample.rolling(window, min_periods=1).sum()
        wins = success.rolling(window, min_periods=1).sum()
        return (wins / count.replace(0, np.nan)).fillna(default).to_numpy()

    set_winner = pd.to_numeric(g.SetWinner, errors="coerce").fillna(0).astype(int)
    p1_sets = (set_winner == 1).cumsum().values
    p2_sets = (set_winner == 2).cumsum().values

    final_p1_sets = int((set_winner == 1).sum())
    final_p2_sets = int((set_winner == 2).sum())

    if final_p1_sets == final_p2_sets:
        return None

    winner = 1 if final_p1_sets > final_p2_sets else 2
    sets_to_win = max(final_p1_sets, final_p2_sets)

    if sets_to_win not in (2, 3):
        return None

    best_of = 2 * sets_to_win - 1

    server = pd.to_numeric(g.PointServer, errors="coerce").fillna(0).astype(int).values
    point_winner = (
        pd.to_numeric(g.PointWinner, errors="coerce").fillna(0).astype(int).values
    )
    p1_games = pd.to_numeric(g.P1GamesWon, errors="coerce").fillna(0).astype(int).values
    p2_games = pd.to_numeric(g.P2GamesWon, errors="coerce").fillna(0).astype(int).values
    set_no = pd.to_numeric(g.SetNo, errors="coerce").fillna(1).astype(int).values
    p1_score = g.P1Score.map(conv_score).values
    p2_score = g.P2Score.map(conv_score).values

    rally = numeric_values("RallyCount", np.nan)
    if np.isnan(rally).all():
        rally = numeric_values("Rally", np.nan)

    speed_kmh = numeric_values("Speed_KMH", np.nan)
    serve_number = numeric_values("ServeNumber", 0)
    tiebreak = ((p1_games == 6) & (p2_games == 6)).astype(int)
    point_no = np.arange(1, len(g) + 1)

    p1_serve_rate, p1_serve_n = run_rate(
        server == 1,
        (server == 1) & (point_winner == 1),
    )
    p2_serve_rate, p2_serve_n = run_rate(
        server == 2,
        (server == 2) & (point_winner == 2),
    )

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

    p1_points_won = prior_sum(point_winner == 1)
    p2_points_won = prior_sum(point_winner == 2)

    p1_break_point_event = numeric_values("P1BreakPoint", 0) > 0
    p2_break_point_event = numeric_values("P2BreakPoint", 0) > 0
    p1_break_points = prior_sum(p1_break_point_event)
    p2_break_points = prior_sum(p2_break_point_event)
    p1_break_points_won = prior_sum(p1_break_point_event & (point_winner == 1))
    p2_break_points_won = prior_sum(p2_break_point_event & (point_winner == 2))

    p1_first_serve_won = (
        (server == 1) & (serve_number == 1) & (point_winner == 1)
    )
    p2_first_serve_won = (
        (server == 2) & (serve_number == 1) & (point_winner == 2)
    )
    p1_first_serve_points_won = prior_sum(p1_first_serve_won)
    p2_first_serve_points_won = prior_sum(p2_first_serve_won)

    p1_double_fault_event = numeric_values("P1DoubleFault", 0) > 0
    p2_double_fault_event = numeric_values("P2DoubleFault", 0) > 0
    p1_double_faults = prior_sum(p1_double_fault_event)
    p2_double_faults = prior_sum(p2_double_fault_event)

    all_points = np.ones(len(g), dtype=bool)
    p1_momentum = rolling_prior_rate(all_points, point_winner == 1, window=12)
    p2_momentum = rolling_prior_rate(all_points, point_winner == 2, window=12)

    keep = (server > 0) & (point_winner > 0)

    df = pd.DataFrame(
        {
            "match_id": mid,
            "year": year,
            "slam": slam,
            "best_of": best_of,
            "winner": winner,
            "y": float(winner == 1),
            "set_no": set_no,
            "p1_sets": p1_sets,
            "p2_sets": p2_sets,
            "p1_games": p1_games,
            "p2_games": p2_games,
            "server": server,
            "p1_score": p1_score,
            "p2_score": p2_score,
            "tiebreak": tiebreak,
            "point_no": point_no,
            "point_winner": point_winner,
            "serve_won": (server == point_winner).astype(int),
            "speed_kmh": speed_kmh,
            "serve_number": serve_number,
            "serve_width": category_codes("ServeWidth"),
            "serve_depth": category_codes("ServeDepth"),
            "return_depth": category_codes("ReturnDepth"),
            "p1_serve_rate": p1_serve_rate,
            "p1_serve_n": p1_serve_n,
            "p2_serve_rate": p2_serve_rate,
            "p2_serve_n": p2_serve_n,
            "rally_avg": rally_avg,
            "rally_n": rally_n,
            "recent_rally_avg": recent_rally_avg,
            "p1_serve_rally_avg": p1_serve_rally_avg,
            "p2_serve_rally_avg": p2_serve_rally_avg,
            "p1_aces": p1_aces,
            "p2_aces": p2_aces,
            "p1_ace_rate": p1_ace_rate,
            "p1_ace_serve_n": p1_ace_serve_n,
            "p2_ace_rate": p2_ace_rate,
            "p2_ace_serve_n": p2_ace_serve_n,
            "p1_recent_ace_rate": p1_recent_ace_rate,
            "p2_recent_ace_rate": p2_recent_ace_rate,
            "p1_points_won": p1_points_won,
            "p2_points_won": p2_points_won,
            "p1_break_points": p1_break_points,
            "p2_break_points": p2_break_points,
            "p1_break_points_won": p1_break_points_won,
            "p2_break_points_won": p2_break_points_won,
            "p1_first_serve_points_won": p1_first_serve_points_won,
            "p2_first_serve_points_won": p2_first_serve_points_won,
            "p1_double_faults": p1_double_faults,
            "p2_double_faults": p2_double_faults,
            "p1_momentum": p1_momentum,
            "p2_momentum": p2_momentum,
        }
    )

    return df[keep]


def main():
    os.makedirs(ART, exist_ok=True)
    matches = []

    for point_file in sorted(glob.glob(f"{SLAM_DIR}/*-points.csv")):
        filename = os.path.basename(point_file).replace("-points.csv", "")
        match = re.match(r"(\d{4})-(.+)", filename)

        if match is None:
            continue

        year = int(match.group(1))
        slam = match.group(2)
        points = pd.read_csv(point_file, low_memory=False)

        if "match_id" not in points.columns:
            continue

        for match_id, group in points.groupby("match_id"):
            processed = process_match(group, match_id, year, slam)

            if processed is not None and len(processed) > 20:
                matches.append(processed)

    if not matches:
        raise ValueError(f"No valid point files were found in {SLAM_DIR}.")

    df = pd.concat(matches, ignore_index=True)
    df["sets_diff"] = df.p1_sets - df.p2_sets
    df["games_diff"] = df.p1_games - df.p2_games
    df["p1_serving"] = (df.server == 1).astype(int)
    df["score_diff"] = df.p1_score - df.p2_score
    df["pts_played"] = df.point_no
    df = add_universal_live_features(df)

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
