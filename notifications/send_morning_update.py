"""
Sends Message 2 - the 8am morning confirmation. Only covers matches
starting after 8am CT that were already flagged in last night's evening
message; refreshes their Kalshi price and says whether the edge still
holds. Run every day at 8am CT (Step 8 wires up the schedule).

Run with: python3 notifications/send_morning_update.py
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from fetch_kalshi import fetch_all_tennis_matches, CENTRAL  # noqa: E402
from format_messages import format_morning_update  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import TRADES_CSV_PATH, update_morning_price  # noqa: E402


def main():
    today_ct = datetime.now(CENTRAL).date()
    if not os.path.exists(TRADES_CSV_PATH):
        print("No trades.csv yet - nothing to confirm this morning.")
        return

    trades = pd.read_csv(TRADES_CSV_PATH)
    todays = trades[
        (trades["date"] == today_ct.isoformat()) & (trades["early_morning_flag"] == False)  # noqa: E712
    ]
    if todays.empty:
        print("No standard-morning matches flagged for today - nothing to send.")
        return

    print("Fetching current Kalshi prices...")
    live_matches = {m["event_ticker"]: m for m in fetch_all_tennis_matches()}

    matches_after_8am = []
    for _, row in todays.iterrows():
        live = live_matches.get(row["event_ticker"])
        if live is None:
            print(f"  {row['player_1']} vs {row['player_2']}: market not found live (may have closed)")
            continue
        current_price = (
            live["player_1_yes_ask"] if live["player_1"] == row["recommended_player"] else live["player_2_yes_ask"]
        )
        if current_price is None:
            continue
        current_price = float(current_price)
        matches_after_8am.append({
            "recommended_side": row["recommended_player"],
            "opponent": row["player_2"] if row["player_1"] == row["recommended_player"] else row["player_1"],
            "start_time_ct": row["match_start_time_ct"],
            "evening_price": row["evening_kalshi_price"],
            "current_price": current_price,
            "model_prob": row["model_predicted_probability"],
            "edge_pp": row["edge_pct"],
        })
        update_morning_price(row["match_start_time_ct"], row["recommended_player"], current_price)

    message = format_morning_update(matches_after_8am)
    print("\n--- Message to send ---")
    print(message)
    send_telegram_message(message)
    print("\nSent to Telegram.")


if __name__ == "__main__":
    main()
