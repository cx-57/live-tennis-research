"""Tests for the live match-state data source.

No network: matches are constructed to the Live Tennis API's published live
score shape (sets / games-per-set / points / server / is_tiebreak). The final
test proves a mapped row feeds the existing Markov recursion unchanged.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import STATE
from src.live_source import (
    best_of_from_format,
    derive_break_point,
    map_state,
    state_frame,
)
from src.markov import predict


# A live match mid-third-game: set 1 went 4-6 (p2), set 2 in progress 3-4,
# p1 serving at 15, p2 receiving at 40 -> a break point against p1.
LIVE_MATCH = {
    "id": 90211,
    "format": "BO3",
    "score": {
        "sets": [0, 1],
        "games": [[4, 3], [6, 4]],
        "points": ["15", "40"],
        "server": 1,
        "is_tiebreak": False,
        "timestamp": "2026-08-18T02:31:04Z",
    },
}

BO5_TIEBREAK_MATCH = {
    "id": 90999,
    "format": "BO5",
    "score": {
        "sets": [1, 1],
        "games": [[6, 6], [6, 6]],
        "points": ["5", "6"],
        "server": 2,
        "is_tiebreak": True,
        "timestamp": "2026-08-18T03:00:00Z",
    },
}

COMPLETED_MATCH = {
    "id": 90212,
    "format": "BO3",
    "score": {
        "sets": [2, 0],
        "games": [[6, 6], [4, 2]],
        "points": [None, None],
        "server": None,
        "is_tiebreak": False,
        "timestamp": "2026-08-18T04:00:00Z",
    },
}


def test_best_of_from_format():
    assert best_of_from_format("BO3") == 3
    assert best_of_from_format("BO5") == 5
    assert best_of_from_format(None) == 3
    assert best_of_from_format("weird") == 3


def test_map_state_live_match():
    row = map_state(LIVE_MATCH)
    assert row is not None
    assert row["match_id"] == 90211
    assert row["best_of"] == 3
    assert row["p1_sets"] == 0 and row["p2_sets"] == 1
    # current (second) set is 3-4
    assert row["p1_games"] == 3 and row["p2_games"] == 4
    assert row["p1_serving"] == 1
    # "15" -> 1, "40" -> 3 via SCOREMAP
    assert row["p1_score"] == 1 and row["p2_score"] == 3
    assert row["tiebreak"] == 0
    # server p1 at 15, receiver p2 at 40 -> break point
    assert row["break_point"] is True


def test_map_state_has_exactly_state_columns():
    row = map_state(LIVE_MATCH)
    for column in STATE:
        assert column in row


def test_break_point_truth_table():
    def score(points, server=1, tiebreak=False):
        return {"points": points, "server": server, "is_tiebreak": tiebreak}

    assert derive_break_point(score(["15", "40"], server=1)) is True
    assert derive_break_point(score(["40", "AD"], server=1)) is True
    assert derive_break_point(score(["40", "40"], server=1)) is False  # deuce
    assert derive_break_point(score(["40", "0"], server=1)) is False  # game point
    # server 2, receiver p1 at AD
    assert derive_break_point(score(["AD", "30"], server=2)) is True
    # never in a tiebreak
    assert derive_break_point(score(["6", "7"], server=1, tiebreak=True)) is False
    # undefined on null server / points
    assert derive_break_point(score(["30", "40"], server=None)) is None
    assert derive_break_point(score([None, None], server=1)) is None
    assert derive_break_point(None) is None


def test_tiebreak_match_maps():
    row = map_state(BO5_TIEBREAK_MATCH)
    assert row is not None
    assert row["best_of"] == 5
    assert row["tiebreak"] == 1
    assert row["p1_games"] == 6 and row["p2_games"] == 6
    assert row["p1_serving"] == 0  # server == 2
    assert row["p1_score"] == 5 and row["p2_score"] == 6
    assert row["break_point"] is False  # suppressed in tiebreak


def test_completed_match_is_not_scorable():
    assert map_state(COMPLETED_MATCH) is None
    assert map_state({}) is None
    assert map_state({"score": {}}) is None


def test_state_frame_feeds_markov():
    frame = state_frame([LIVE_MATCH, COMPLETED_MATCH, BO5_TIEBREAK_MATCH])
    # completed match dropped; two scorable rows remain
    assert len(frame) == 2
    assert list(frame.columns)[1 : 1 + len(STATE)] == STATE

    probs = predict(frame, 0.63, 0.63, STATE)
    assert len(probs) == 2
    for p in probs:
        assert 0.0 < p < 1.0


def test_state_frame_empty():
    frame = state_frame([COMPLETED_MATCH])
    assert frame.empty
    assert "break_point" in frame.columns


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all live_source tests passed")
