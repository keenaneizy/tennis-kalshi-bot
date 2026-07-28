"""
Sends Message 1 - the 10pm evening picks message. This is the most
critical message in the whole system: it's sent the night before so
early-morning matches (which start before there's time to bet after
waking up) are never missed. Run every night at 10pm CT (Step 8 wires up
the actual schedule; this script is what that schedule calls).

Run with: python3 notifications/send_evening_picks.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from predict_upcoming import generate_recommendations, ARTIFACT_DIR  # noqa: E402
from fetch_kalshi import CENTRAL  # noqa: E402
from format_messages import format_evening_message  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import append_evening_recommendations, compute_model_record  # noqa: E402
from send_markers import write_sent_marker  # noqa: E402

EVENING_SEND_MARKER = "data/last_evening_send.txt"


def load_eval_metrics():
    with open(f"{ARTIFACT_DIR}/eval_metrics.json") as f:
        return json.load(f)


def main():
    print("Generating tomorrow's recommendations...")
    recommendations, skipped, tomorrow_matches, all_analysis = generate_recommendations(verbose=True)
    eval_metrics = load_eval_metrics()

    tomorrow_date = datetime.now(CENTRAL).date() + timedelta(days=1)
    model_record = compute_model_record()
    message = format_evening_message(recommendations, tomorrow_date, eval_metrics, model_record=model_record)

    print("\n--- Message to send ---")
    print(message)

    send_telegram_message(message)
    print("\nSent to Telegram.")

    append_evening_recommendations(recommendations, tomorrow_date)
    print(f"Logged {len(recommendations)} trade(s) to data/trades.csv")

    # Marker for the safety-net retry - written only after a confirmed
    # successful send, so a stalled/failed run (whatever the cause) is
    # detectable even when there were 0 recommendations that night (which
    # would otherwise leave no trace in trades.csv either way).
    write_sent_marker(EVENING_SEND_MARKER, datetime.now(CENTRAL).date())


if __name__ == "__main__":
    main()
