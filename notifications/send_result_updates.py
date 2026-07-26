"""
Sends Message 4 - a result update as soon as each flagged match resolves
on Kalshi. Meant to be polled periodically (Step 8) rather than run once;
each run only sends for matches that just settled since the last check
(trades.csv rows where match_result_winner is still blank).

Run with: python3 notifications/send_result_updates.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from fetch_kalshi import fetch_event_by_ticker  # noqa: E402
from format_messages import format_result_update  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import TRADES_CSV_PATH, record_result, compute_model_record  # noqa: E402


def main():
    if not os.path.exists(TRADES_CSV_PATH):
        print("No trades.csv yet.")
        return

    trades = pd.read_csv(TRADES_CSV_PATH)
    unresolved = trades[trades["match_result_winner"].isna()]
    if unresolved.empty:
        print("No unresolved trades to check.")
        return

    running_pnl_today = 0.0
    record_today = {"wins": 0, "losses": 0}

    for _, row in unresolved.iterrows():
        event, markets = fetch_event_by_ticker(row["event_ticker"])
        settled_market = next((m for m in markets if m.get("result")), None)
        if settled_market is None:
            continue  # still in progress

        recommended_is_yes_side = settled_market.get("yes_sub_title") == row["recommended_player"]
        recommended_won = (
            (settled_market["result"] == "yes") == recommended_is_yes_side
        )
        actual_winner = row["recommended_player"] if recommended_won else row["player_2"] if row["player_1"] == row["recommended_player"] else row["player_1"]

        df = record_result(row["match_start_time_ct"], row["recommended_player"], actual_winner)
        updated_row = df[
            (df["match_start_time_ct"] == row["match_start_time_ct"])
            & (df["recommended_player"] == row["recommended_player"])
        ].iloc[0]

        running_pnl_today += updated_row["profit_or_loss"]
        record_today["wins" if updated_row["prediction_correct"] else "losses"] += 1

        message = format_result_update({
            "player_1": row["player_1"], "player_2": row["player_2"],
            "predicted_winner": row["recommended_player"], "actual_winner": actual_winner,
            "model_prob": row["model_predicted_probability"],
            "pnl": updated_row["profit_or_loss"],
            "running_pnl_today": running_pnl_today,
            "record_today_wins": record_today["wins"], "record_today_losses": record_today["losses"],
        })
        print(message)
        send_telegram_message(message)


if __name__ == "__main__":
    main()
