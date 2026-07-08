"""Pre-match Elo rating built from the full tour match history (ATP and WTA, 2005 onward)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from src.common import ATP_DIR, WTA_DIR, SLAM_DIR, POINTS, ELO

SLAM_NAME = {"ausopen": "australian open", "frenchopen": "roland garros",
             "wimbledon": "wimbledon", "usopen": "us open"}
ROUND_ORDER = {"RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
               "QF": 5, "SF": 6, "BR": 6, "F": 7}


def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def last_name(s):
    parts = s.split()
    return parts[-1] if parts else s


def load_tour():
    frames = []
    for folder in (ATP_DIR, WTA_DIR):
        for f in sorted(glob.glob(f"{folder}/*_matches_20*.csv")):
            year = int(re.search(r"(\d{4})\.csv$", f).group(1))
            if not 2005 <= year <= 2024:
                continue
            cols = ("tourney_date", "tourney_name", "winner_name", "loser_name", "round")
            c = pd.read_csv(f, usecols=lambda x: x in cols, low_memory=False)
            c["year"] = year
            frames.append(c)
    tour = pd.concat(frames, ignore_index=True).dropna(
        subset=["winner_name", "loser_name", "tourney_date"])
    tour["rord"] = tour["round"].map(ROUND_ORDER).fillna(3).astype(int)
    return tour.sort_values(["tourney_date", "rord"]).reset_index(drop=True)


def run_elo(tour):
    elo = defaultdict(lambda: 1500.0)
    games = defaultdict(int)
    pre_w = np.empty(len(tour))
    pre_l = np.empty(len(tour))
    for i, r in enumerate(tour.itertuples()):
        w, l = r.winner_name, r.loser_name
        ew, el = elo[w], elo[l]
        pre_w[i], pre_l[i] = ew, el

        expected_w = 1.0 / (1 + 10 ** ((el - ew) / 400))
        kw = 250.0 / ((games[w] + 5) ** 0.4)
        kl = 250.0 / ((games[l] + 5) ** 0.4)
        elo[w] = ew + kw * (1 - expected_w)
        elo[l] = el + kl * (expected_w - 1)
        games[w] += 1
        games[l] += 1
    tour["pre_w"], tour["pre_l"] = pre_w, pre_l
    return tour, len(elo)


def main():
    tour = load_tour()
    tour, n_players = run_elo(tour)
    print(f"tour matches 2005-2024: {len(tour)}  players rated: {n_players}")

    tour["sc"] = tour.tourney_name.map(norm)
    slam = tour[tour.sc.isin(set(SLAM_NAME.values()))].copy()
    slam["nw"] = slam.winner_name.map(norm)
    slam["nl"] = slam.loser_name.map(norm)

    by_full = {}
    by_last = {}
    last_clash = set()
    for r in slam.itertuples():
        by_full[(r.year, r.sc, frozenset((r.nw, r.nl)))] = (r.pre_w, r.pre_l, r.nw)
        key = (r.year, r.sc, frozenset((last_name(r.nw), last_name(r.nl))))
        if key in by_last and by_last[key][2] != last_name(r.nw):
            last_clash.add(key)
        by_last[key] = (r.pre_w, r.pre_l, last_name(r.nw))
    for key in last_clash:
        by_last.pop(key, None)

    names = {}
    for f in sorted(glob.glob(f"{SLAM_DIR}/*-matches.csv")):
        m = pd.read_csv(f)
        for r in m.itertuples():
            if isinstance(r.player1, str) and isinstance(r.player2, str):
                names[r.match_id] = (r.player1.strip(), r.player2.strip())

    meta = pd.read_parquet(POINTS, columns=["match_id", "year", "slam"]).groupby(
        "match_id").agg(year=("year", "first"), slam=("slam", "first")).reset_index()
    meta = meta[meta.match_id.isin(names)].copy()

    rows = []
    missed = 0
    for r in meta.itertuples():
        n1, n2 = norm(names[r.match_id][0]), norm(names[r.match_id][1])
        sc = SLAM_NAME.get(r.slam)
        hit = by_full.get((r.year, sc, frozenset((n1, n2))))
        by_full_match = hit is not None
        if hit is None:
            hit = by_last.get((r.year, sc, frozenset((last_name(n1), last_name(n2)))))
        if hit is None:
            missed += 1
            continue
        pre_w, pre_l, winner_key = hit
        p1_is_winner = (n1 == winner_key) if by_full_match else (last_name(n1) == winner_key)
        e1, e2 = (pre_w, pre_l) if p1_is_winner else (pre_l, pre_w)
        rows.append((r.match_id, e1, e2))

    ep = pd.DataFrame(rows, columns=["match_id", "elo_p1", "elo_p2"])
    ep["elo_diff"] = ep.elo_p1 - ep.elo_p2
    ep["elo_prob_p1"] = 1.0 / (1 + 10 ** (-ep.elo_diff / 400))
    ep.to_parquet(ELO)
    print(f"wrote {ELO}: matched {len(ep)}/{len(meta)} matches (missed {missed})")


if __name__ == "__main__":
    main()
