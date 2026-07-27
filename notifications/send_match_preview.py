"""
Sends the 9pm match preview message - added per user request (2026-07-27):
an hour before the 10pm evening picks message, show every ATP match the
model analyzed for tomorrow, with each player's model probability, Kalshi
price, and edge - not just the ones that clear the betting thresholds.
Run every night at 9pm CT (an hour before send_evening_picks.py).

Run with: python3 notifications/send_match_preview.py
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))

from predict_upcoming import generate_recommendations  # noqa: E402
from fetch_kalshi import CENTRAL  # noqa: E402
from format_messages import format_match_preview_message  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402


def main():
    print("Generating tomorrow's full match analysis...")
    recommendations, skipped, tomorrow_matches, all_analysis = generate_recommendations(verbose=True)

    tomorrow_date = datetime.now(CENTRAL).date() + timedelta(days=1)
    flagged_matchups = {r["matchup"] for r in recommendations}
    message = format_match_preview_message(all_analysis, tomorrow_date, flagged_matchups=flagged_matchups)

    print("\n--- Message to send ---")
    print(message)

    send_telegram_message(message)
    print("\nSent to Telegram.")


if __name__ == "__main__":
    main()
