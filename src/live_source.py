"""Live match-state data source for the Markov recursion.

Everything else in this repository reads its in-match state from the offline
``points.parquet`` table (see :mod:`src.prepare_data`). That table has one row
per pre-point state with the columns listed in :data:`src.common.STATE`, and
every model scores a match by handing those columns to
:func:`src.markov.predict`.

This module produces the *same* ``STATE`` row from a match that is happening
right now, so the trained models can forecast a live match instead of only a
historical point-by-point file. It maps the Live Tennis API in-play score
object into the exact ``STATE`` schema, reusing
:data:`src.prepare_data.SCOREMAP` / :func:`src.prepare_data.conv_score` for the
``0/15/30/40/AD`` -> integer point encoding so a mapped row is schema-identical
to one produced by the offline pipeline.

Disclosure
----------
The Live Tennis API (https://livetennisapi.com) maintains this adapter. It uses
only free-tier keyed endpoints -- ``/matches?status=live`` and
``/matches/{id}`` (30 requests/minute, 100 requests/day). A free key:
https://livetennisapi.com/subscribe/free. Fetching uses the standard-library
``urllib`` so this file adds no third-party dependency to the project.

The live win-probability and market-price fields the API also serves are
deliberately *not* used: this adapter supplies only score / server / games /
sets so that this repository's own Markov model computes the probability.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import STATE
from src.prepare_data import conv_score

LIVE_BASE_URL = "https://api.livetennisapi.com/api/public/v1"
API_KEY_ENV = "LIVETENNIS_API_KEY"
_USER_AGENT = "live-tennis-research/live_source (+https://livetennisapi.com)"


class MissingAPIKeyError(RuntimeError):
    """Raised when no Live Tennis API key is available."""

    def __init__(self) -> None:
        super().__init__(
            f"No Live Tennis API key found. Set the {API_KEY_ENV} environment "
            "variable. Free keys: https://livetennisapi.com/subscribe/free"
        )


def best_of_from_format(fmt) -> int:
    """Map the API ``format`` field (``"BO3"``/``"BO5"``) to ``best_of``.

    Defaults to 3 when the field is missing or unrecognised, matching the
    dominant format in the offline data.
    """
    digits = "".join(ch for ch in str(fmt) if ch.isdigit())
    return 5 if digits == "5" else 3


def derive_break_point(score):
    """Three-valued break-point flag for the current point.

    ``True``  -- the current point is a break point (the receiver can win the
    game and break serve).
    ``False`` -- it is not a break point.
    ``None``  -- undefined: no score, no known server, or missing point values
    (the API returns null points on a completed match).

    Break point rule: receiver at ``AD``, or receiver at ``40`` while the
    server is at ``0``/``15``/``30``; never during a tiebreak.
    """
    if not score:
        return None
    if score.get("is_tiebreak"):
        return False
    server = score.get("server")
    if server not in (1, 2):
        return None
    points = score.get("points") or []
    if len(points) != 2 or points[0] is None or points[1] is None:
        return None
    receiver_points = str(points[1] if server == 1 else points[0])
    server_points = str(points[0] if server == 1 else points[1])
    if receiver_points == "AD":
        return True
    return receiver_points == "40" and server_points in ("0", "15", "30")


def _current_set_games(games):
    """Return ``(p1_games, p2_games)`` for the set in progress, or ``None``.

    The API encodes games as ``[p1_games_per_set, p2_games_per_set]`` (e.g.
    ``[[4, 3], [6, 4]]`` -> set 1 was 4-6, set 2 is 3-4). The set in progress is
    the last entry of each side's list.
    """
    if not isinstance(games, (list, tuple)) or len(games) != 2:
        return None
    p1_sets, p2_sets = games[0], games[1]
    if not isinstance(p1_sets, (list, tuple)) or not isinstance(p2_sets, (list, tuple)):
        return None
    if not p1_sets or not p2_sets:
        return None
    try:
        return int(p1_sets[-1]), int(p2_sets[-1])
    except (TypeError, ValueError):
        return None


def map_state(match):
    """Map one Live Tennis API match object to a ``STATE`` row.

    Returns a dict carrying every :data:`src.common.STATE` column (ready for
    :func:`src.markov.predict`) plus ``match_id``, ``break_point`` (the
    three-valued flag) and ``score_timestamp``. Returns ``None`` when the match
    is not in a scorable in-play state -- no score object, no known server,
    missing/complete point values, or no games yet -- so callers can skip it.
    """
    if not isinstance(match, dict):
        return None

    score = match.get("score")
    if not isinstance(score, dict):
        return None

    server = score.get("server")
    if server not in (1, 2):
        return None

    current_games = _current_set_games(score.get("games"))
    if current_games is None:
        return None
    p1_games, p2_games = current_games

    points = score.get("points") or []
    if len(points) != 2 or points[0] is None or points[1] is None:
        return None
    p1_score = conv_score(points[0])
    p2_score = conv_score(points[1])
    if pd.isna(p1_score) or pd.isna(p2_score):
        return None

    sets = score.get("sets") or [0, 0]
    try:
        p1_sets, p2_sets = int(sets[0]), int(sets[1])
    except (TypeError, ValueError, IndexError):
        return None

    return {
        "match_id": match.get("id"),
        "best_of": best_of_from_format(match.get("format")),
        "p1_sets": p1_sets,
        "p2_sets": p2_sets,
        "p1_games": p1_games,
        "p2_games": p2_games,
        "p1_serving": 1 if server == 1 else 0,
        "p1_score": int(p1_score),
        "p2_score": int(p2_score),
        "tiebreak": 1 if score.get("is_tiebreak") else 0,
        "break_point": derive_break_point(score),
        "score_timestamp": score.get("timestamp"),
    }


def state_frame(matches):
    """Build a DataFrame of ``STATE`` rows from an iterable of API matches.

    Non-scorable matches are dropped. The returned frame always has exactly the
    :data:`src.common.STATE` columns (plus ``match_id``, ``break_point``,
    ``score_timestamp``) and is ready to pass straight to
    :func:`src.markov.predict` with per-row serve probabilities.
    """
    rows = [row for row in (map_state(m) for m in matches) if row is not None]
    columns = ["match_id", *STATE, "break_point", "score_timestamp"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def _get(path, api_key, base_url, params=None, timeout=15.0):
    """GET one JSON path from the Live Tennis API (stdlib, keyed)."""
    if not api_key:
        raise MissingAPIKeyError()
    url = base_url.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Live Tennis API rate limit hit (free tier: 30 req/min, "
                "100 req/day). Slow the polling cadence."
            ) from exc
        raise
    if isinstance(payload, dict):
        return payload.get("data", payload)
    return payload


def fetch_live_matches(api_key=None, base_url=LIVE_BASE_URL, tour=None, timeout=15.0):
    """Fetch the currently live matches (free-tier ``/matches?status=live``)."""
    api_key = api_key or os.environ.get(API_KEY_ENV) or ""
    params = {"status": "live", "limit": 50}
    if tour:
        params["tour"] = tour
    data = _get("/matches", api_key, base_url, params, timeout)
    return data if isinstance(data, list) else []


def main():
    """Fetch live matches and print each one's mapped state + live win prob.

    Uses a single neutral serve-win probability so the demo needs no Elo
    artifact; a real forecast plugs each player's model-derived serve
    probability into :func:`src.markov.predict` instead.
    """
    from src.markov import predict

    neutral_serve_prob = 0.63  # ~ baseline_markov's calibrated value

    try:
        matches = fetch_live_matches()
    except MissingAPIKeyError as exc:
        print(exc)
        return

    frame = state_frame(matches)
    if frame.empty:
        print("No scorable live matches right now.")
        return

    probs = predict(frame, neutral_serve_prob, neutral_serve_prob, STATE)
    for row, p1_win in zip(frame.itertuples(index=False), probs):
        bp = {True: "BP", False: "--", None: "?"}[row.break_point]
        print(
            f"match {row.match_id}: sets {row.p1_sets}-{row.p2_sets} "
            f"games {row.p1_games}-{row.p2_games} "
            f"pts {row.p1_score}-{row.p2_score} "
            f"serve={'p1' if row.p1_serving else 'p2'} {bp}  "
            f"P(p1 wins)={p1_win:.3f}"
        )


if __name__ == "__main__":
    main()
