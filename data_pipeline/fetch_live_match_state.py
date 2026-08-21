"""
Live in-match state for ATP matches (Live Tennis API).

This is a data-only companion to fetch_kalshi.py. Kalshi tells us the market
price and *eventually* the settled result; it does not tell us what is
happening inside a match right now. The scheduler polls hourly (see README),
so two things the bot currently can't see are:

  1. A match that has already finished but whose Kalshi market hasn't settled
     yet - the result is knowable from live state before the hourly Kalshi
     poll catches the settlement.
  2. A retirement or walkover mid-match. The README notes the retirement/injury
     skip rule only consults a manual watchlist (data/injury_watchlist.csv);
     live state carries an explicit event_status of "Retired"/"Walkover" plus a
     "withdrew" player index, so an in-progress withdrawal is observable.

It also exposes the live score, who is serving, and a three-valued break-point
flag (on / off / undefined) for any in-progress match, which the pre-match and
monitoring flows can surface.

Source / disclosure: this reads the Live Tennis API's free keyed tier
(https://api.livetennisapi.com/api/public/v1 - 30 req/min, 100 req/day). This
module is contributed by the Live Tennis API team. Everything called here
(GET /matches?status=live, GET /matches/{id}) is on the free tier; set an API
key in the LIVETENNIS_API_KEY environment variable (free key:
https://livetennisapi.com/subscribe/free). It is observe-only: it reads match
state and never places or influences a trade.

Break-point rule (documented behaviour of the score object): a break point is
ON when the RECEIVER is at AD, or the receiver is at 40 while the server is at
0/15/30. It is never on during a tiebreak, and it is UNDEFINED (None) whenever
the server or either point is null - completed matches carry null points.

Run with: python3 data_pipeline/fetch_live_match_state.py
"""

import os
import re
import unicodedata
from datetime import datetime, timezone

import requests

try:  # optional - matches notifications/telegram_bot.py's .env convention
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional for pure functions
    pass

LIVETENNIS_BASE = "https://api.livetennisapi.com/api/public/v1"
API_KEY_ENV = "LIVETENNIS_API_KEY"
UTC = timezone.utc

# Values the API puts in event_status for a match that ended without a normal
# final score. "Completed" is the ordinary case and is deliberately absent.
WITHDRAWAL_STATUSES = {"retired", "walkover"}


def _api_key():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} not set - live match state needs a Live Tennis API "
            "key. Free key: https://livetennisapi.com/subscribe/free"
        )
    return key


def _get(path, params=None, timeout=20):
    resp = requests.get(
        f"{LIVETENNIS_BASE}{path}",
        params=params,
        headers={"X-API-Key": _api_key()},
        timeout=timeout,
    )
    if resp.status_code == 429:
        raise RuntimeError(
            "Live Tennis API rate limit hit (free tier: 30 req/min, 100 "
            "req/day) - slow the polling cadence."
        )
    resp.raise_for_status()
    return resp.json()


def fetch_live_matches(tour="atp", limit=50):
    """All in-progress matches for a tour ("atp"/"wta"). One dict per match."""
    payload = _get("/matches", params={"status": "live", "tour": tour, "limit": limit})
    return payload.get("data", []) if isinstance(payload, dict) else []


def fetch_match_state(match_id):
    """Single match by id, regardless of status; None if the id is unknown."""
    try:
        payload = _get(f"/matches/{match_id}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    if isinstance(payload, dict):
        return payload.get("data", payload)
    return None


def _player_name(match, index):
    """Name of player 1 or 2 from a match's players object; '' if absent."""
    players = match.get("players") or {}
    side = players.get("p1" if index == 1 else "p2") or {}
    return side.get("name") or ""


def derive_break_point(score):
    """Three-valued break-point flag for the current point.

    Returns True (break point on), False (not a break point), or None
    (undefined - a tiebreak, or the server/points aren't known, e.g. a match
    that has ended). None is deliberately distinct from False so callers can
    tell "no break point" apart from "we can't say".
    """
    if not score:
        return None
    if score.get("is_tiebreak"):
        return None
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


def score_line(score):
    """Render a score object as "6-4 3-4 (15-40)" from player 1's perspective."""
    if not score:
        return ""
    games = score.get("games") or []
    parts = []
    if len(games) == 2 and games[0] and len(games[0]) == len(games[1]):
        parts = [f"{a}-{b}" for a, b in zip(games[0], games[1])]
    points = score.get("points") or []
    if len(points) == 2 and points[0] is not None and points[1] is not None:
        parts.append(f"({points[0]}-{points[1]})")
    return " ".join(parts)


def _parse_iso(value):
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def summarize_match_state(match):
    """Flatten a raw match into the fields the bot cares about.

    Keys: match_id, tour, status, event_status, player_1, player_2,
    server_name (who is serving, or None), score_line, is_tiebreak,
    break_point (three-valued), winner_name/withdrew_name (None until known),
    is_withdrawal (True on a retirement/walkover), is_final, live_as_of.
    """
    score = match.get("score") or {}
    server = score.get("server")
    server_name = _player_name(match, server) if server in (1, 2) else None

    winner = match.get("winner")
    winner_name = _player_name(match, winner) if winner in (1, 2) else None
    withdrew = match.get("withdrew")
    withdrew_name = _player_name(match, withdrew) if withdrew in (1, 2) else None

    event_status = match.get("event_status")
    status_lower = (event_status or "").strip().lower()

    return {
        "match_id": match.get("id"),
        "tour": match.get("tour"),
        "status": match.get("status"),
        "event_status": event_status,
        "player_1": _player_name(match, 1),
        "player_2": _player_name(match, 2),
        "server_name": server_name,
        "score_line": score_line(score),
        "is_tiebreak": bool(score.get("is_tiebreak")),
        "break_point": derive_break_point(score),
        "winner_name": winner_name,
        "withdrew_name": withdrew_name,
        "is_withdrawal": status_lower in WITHDRAWAL_STATUSES or withdrew in (1, 2),
        "is_final": match.get("status") == "completed" or winner in (1, 2),
        "live_as_of": _parse_iso(score.get("timestamp")),
    }


def _normalize_name(name):
    """Lowercase, strip diacritics, keep a-z and spaces.

    Same shape as models/predict_upcoming.normalize_name, duplicated here so
    this module stays importable without pulling in the ensemble's heavy
    torch/xgboost imports.
    """
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def _names_agree(a, b):
    """Conservative one-vs-one name agreement (Kalshi form vs feed form)."""
    na, nb = _normalize_name(a), _normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True  # "J Lehecka" vs "Jiri Lehecka"
    return na.split()[-1] == nb.split()[-1]  # surname agreement


def find_live_match_for_players(matches, name_a, name_b):
    """Find the live match whose two players match a Kalshi matchup's names.

    Requires BOTH sides to agree (in either order) and returns a unique match
    or None - never a guess when two matches could fit. `matches` is a list of
    raw match dicts (e.g. from fetch_live_matches).
    """
    hits = []
    for match in matches:
        p1 = _player_name(match, 1)
        p2 = _player_name(match, 2)
        direct = _names_agree(name_a, p1) and _names_agree(name_b, p2)
        reversed_ = _names_agree(name_a, p2) and _names_agree(name_b, p1)
        if direct or reversed_:
            hits.append(match)
    return hits[0] if len(hits) == 1 else None


def _fmt_break_point(flag):
    return {True: "yes", False: "no", None: "undef"}[flag]


if __name__ == "__main__":
    print("Fetching live ATP match state from the Live Tennis API...")
    matches = fetch_live_matches(tour="atp")
    print(f"\nLive ATP matches right now: {len(matches)}")

    for match in matches:
        s = summarize_match_state(match)
        flags = []
        if s["break_point"] is True:
            flags.append("BREAK POINT")
        if s["is_tiebreak"]:
            flags.append("tiebreak")
        if s["is_withdrawal"]:
            flags.append((s["event_status"] or "withdrawal").upper())
        elif s["event_status"]:
            flags.append(s["event_status"])
        serving = f" - serving {s['server_name']}" if s["server_name"] else ""
        flag_text = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  {s['player_1']} vs {s['player_2']}  {s['score_line']}"
            f"{serving}  (bp={_fmt_break_point(s['break_point'])}){flag_text}"
        )
        if s["is_final"] and s["winner_name"]:
            note = f" ({s['event_status']})" if s["event_status"] else ""
            print(f"      -> final: {s['winner_name']} won{note}")
