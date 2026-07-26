"""
Sends Message 3 - a reminder 30 minutes before each flagged match that
starts after 8am CT (early-morning matches were already bet the night
before, so they don't get this one). Meant to be polled every few minutes
by Step 8's scheduler, which passes in "now" - each run only sends for
matches whose start time is between 25 and 35 minutes away, so it fires
once per match rather than every poll.

Run with: python3 notifications/send_pre_match_alert.py
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from fetch_kalshi import fetch_event_by_ticker, CENTRAL  # noqa: E402
from format_messages import format_pre_match_alert  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import TRADES_CSV_PATH  # noqa: E402


def matches_due_soon(trades_df, now_ct, window=(25, 35)):
    due = []
    for _, row in trades_df.iterrows():
        if row["early_morning_flag"]:
            continue  # already bet the night before, no same-day reminder needed
        # strptime's %Z doesn't reliably parse abbreviations like "CDT" - strip
        # it and attach CENTRAL directly instead (the abbreviation is always
        # whatever CENTRAL's current offset already is).
        naive_part = row["match_start_time_ct"].rsplit(" ", 1)[0]
        start = datetime.strptime(naive_part, "%Y-%m-%d %I:%M %p").replace(tzinfo=CENTRAL)
        minutes_until = (start - now_ct).total_seconds() / 60
        if window[0] <= minutes_until <= window[1]:
            due.append(row)
    return due


def main():
    if not os.path.exists(TRADES_CSV_PATH):
        print("No trades.csv yet.")
        return
    trades = pd.read_csv(TRADES_CSV_PATH)
    unresolved = trades[trades["match_result_winner"].isna()]

    now_ct = datetime.now(CENTRAL)
    due = matches_due_soon(unresolved, now_ct)
    if not due:
        print("No matches starting in the next ~30 minutes.")
        return

    for row in due:
        event, markets = fetch_event_by_ticker(row["event_ticker"])
        market = next((m for m in markets if m.get("yes_sub_title") == row["recommended_player"]), None)
        current_price = float(market["yes_ask_dollars"]) if market and market.get("yes_ask_dollars") else row["evening_kalshi_price"]
        edge_still_valid = (row["model_predicted_probability"] - current_price) * 100 >= 5

        message = format_pre_match_alert({
            "recommended_side": row["recommended_player"],
            "opponent": row["player_2"] if row["player_1"] == row["recommended_player"] else row["player_1"],
            "current_price": current_price,
            "edge_still_valid": edge_still_valid,
            "bet_size": row["recommended_bet_size"],
        })
        print(message)
        send_telegram_message(message)


if __name__ == "__main__":
    main()
