"""
STEP 1 - Kalshi tennis market data.

Pulls live tennis markets from the public Kalshi API (no authentication
required for market data - only placing trades needs an API key, which we
don't need yet). Docs: https://trading-api.readme.io

Kalshi lists one event per match, with two YES/NO markets underneath it
(one per player). We fetch both ATP and WTA match-winner markets, pair them
up into one row per match, and convert the match time to US Central Time so
we can flag "early morning" (before 8am CT) matches - the whole reason the
picks bot needs to run the night before instead of the morning of.

Kalshi's series tickers for match winners:
- KXATPMATCH - ATP (men's) match winner
- KXWTAMATCH - WTA (women's) match winner
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CENTRAL = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")

SERIES_TICKERS = {
    "ATP": "KXATPMATCH",
    "WTA": "KXWTAMATCH",
}


def _get(path, params=None, timeout=20):
    resp = requests.get(f"{KALSHI_BASE}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_event_by_ticker(event_ticker):
    """Single-event lookup, regardless of status - used to check whether a match has resolved yet."""
    payload = _get(f"/events/{event_ticker}", params={"with_nested_markets": "true"})
    return payload.get("event"), payload.get("markets", [])


def fetch_events_with_markets(series_ticker, status="open", max_pages=20):
    """Page through all events (with nested markets) for a series ticker."""
    events = []
    cursor = None
    for _ in range(max_pages):
        params = {
            "series_ticker": series_ticker,
            "status": status,
            "with_nested_markets": "true",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        payload = _get("/events", params=params)
        events.extend(payload.get("events", []))
        cursor = payload.get("cursor")
        if not cursor:
            break
    return events


def _to_central(iso_ts):
    if not iso_ts:
        return None
    dt_utc = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return dt_utc.astimezone(CENTRAL)


def events_to_matches(events, tour):
    """Turn Kalshi events (2 markets each) into one row per match."""
    matches = []
    for event in events:
        markets = event.get("markets", [])
        if len(markets) != 2:
            continue  # skip anything that isn't a simple two-outcome match market

        m1, m2 = markets
        start_ct = _to_central(m1.get("occurrence_datetime"))

        match = {
            "tour": tour,
            "event_ticker": event.get("event_ticker"),
            "matchup": event.get("title"),
            "player_1": m1.get("yes_sub_title"),
            "player_1_yes_bid": m1.get("yes_bid_dollars"),
            "player_1_yes_ask": m1.get("yes_ask_dollars"),
            "player_1_volume": m1.get("volume_fp"),
            "player_2": m2.get("yes_sub_title"),
            "player_2_yes_bid": m2.get("yes_bid_dollars"),
            "player_2_yes_ask": m2.get("yes_ask_dollars"),
            "player_2_volume": m2.get("volume_fp"),
            "start_time_utc": m1.get("occurrence_datetime"),
            "start_datetime_ct": start_ct,
            "start_time_ct": start_ct.strftime("%Y-%m-%d %I:%M %p %Z") if start_ct else None,
            "start_hour_ct": start_ct.hour if start_ct else None,
            "round_and_tournament": m1.get("title", "").split(": ")[-1].rstrip("? match") if m1.get("title") else None,
            "rules_primary": m1.get("rules_primary"),
        }
        matches.append(match)
    return matches


def fetch_all_tennis_matches():
    """Fetch all open ATP + WTA match markets, paired into one row per match."""
    all_matches = []
    for tour, ticker in SERIES_TICKERS.items():
        events = fetch_events_with_markets(ticker)
        all_matches.extend(events_to_matches(events, tour))
    return all_matches


def matches_for_date(matches, target_date_ct):
    """Filter to matches whose Central-Time start date matches target_date_ct (a date object)."""
    out = []
    for m in matches:
        if not m["start_datetime_ct"]:
            continue
        if m["start_datetime_ct"].date() == target_date_ct:
            out.append(m)
    return out


if __name__ == "__main__":
    print("Fetching live Kalshi tennis markets (ATP + WTA)...")
    matches = fetch_all_tennis_matches()
    print(f"\nTotal open tennis match markets on Kalshi: {len(matches)}")

    today_ct = datetime.now(CENTRAL).date()
    tomorrow_ct = today_ct + timedelta(days=1)

    tomorrow_matches = matches_for_date(matches, tomorrow_ct)
    print(f"Matches scheduled tomorrow ({tomorrow_ct}) CT: {len(tomorrow_matches)}")

    early_morning = [m for m in tomorrow_matches if m["start_hour_ct"] is not None and m["start_hour_ct"] < 8]
    print(f"  Of which early morning (before 8am CT): {len(early_morning)}")

    print("\nSample of 5 matches with current prices:")
    for m in matches[:5]:
        print(
            f"  [{m['tour']}] {m['matchup']} - starts {m['start_time_ct']} - "
            f"{m['player_1']} YES ask {m['player_1_yes_ask']} / "
            f"{m['player_2']} YES ask {m['player_2_yes_ask']} - "
            f"vol {m['player_1_volume']}"
        )
