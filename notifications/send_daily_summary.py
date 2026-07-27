"""
Sends Message 5 - the 9pm daily summary: today's record/P&L, running
totals, bankroll, and a preview of tomorrow's early-morning count so the
user knows to watch for the 10pm picks. Run every day at 9pm CT.

Run with: python3 notifications/send_daily_summary.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from predict_upcoming import generate_recommendations, ARTIFACT_DIR  # noqa: E402
from fetch_kalshi import CENTRAL  # noqa: E402
from format_messages import format_daily_summary  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import TRADES_CSV_PATH, compute_model_record, STARTING_BANKROLL  # noqa: E402


def main():
    today_ct = datetime.now(CENTRAL).date()

    if os.path.exists(TRADES_CSV_PATH):
        trades = pd.read_csv(TRADES_CSV_PATH)
        settled = trades[trades["prediction_correct"].notna()]
        today_settled = settled[settled["date"] == today_ct.isoformat()]
    else:
        settled = pd.DataFrame()
        today_settled = pd.DataFrame()

    with open(f"{ARTIFACT_DIR}/eval_metrics.json") as f:
        eval_metrics = json.load(f)

    total_record = compute_model_record(settled) if not settled.empty else None
    yesterday_brier = None
    if not settled.empty:
        yesterday = settled[settled["date"] == (today_ct - timedelta(days=1)).isoformat()]
        if not yesterday["brier_score_that_day"].dropna().empty:
            yesterday_brier = yesterday["brier_score_that_day"].dropna().iloc[-1]

    print("Checking tomorrow's early-morning count...")
    tomorrow_recs, _, _, _ = generate_recommendations(verbose=False)
    tomorrow_early_count = sum(1 for r in tomorrow_recs if r["start_hour_ct"] is not None and r["start_hour_ct"] < 8)

    summary = {
        "today_wins": int(today_settled["prediction_correct"].sum()) if not today_settled.empty else 0,
        "today_losses": int((~today_settled["prediction_correct"].astype(bool)).sum()) if not today_settled.empty else 0,
        "today_pnl": float(today_settled["profit_or_loss"].sum()) if not today_settled.empty else 0.0,
        "total_wins": total_record["wins"] if total_record else 0,
        "total_losses": total_record["losses"] if total_record else 0,
        "total_accuracy": total_record["accuracy"] if total_record else eval_metrics["accuracy"],
        "total_pnl": total_record["total_pnl"] if total_record else 0.0,
        "bankroll": total_record["bankroll"] if total_record else STARTING_BANKROLL,
        "brier_today": today_settled["brier_score_that_day"].dropna().iloc[-1] if not today_settled.empty and not today_settled["brier_score_that_day"].dropna().empty else eval_metrics["brier"],
        "brier_yesterday": yesterday_brier if yesterday_brier is not None else eval_metrics["brier"],
        "tomorrow_early_morning_count": tomorrow_early_count,
    }

    message = format_daily_summary(summary)
    print("\n--- Message to send ---")
    print(message)
    send_telegram_message(message)
    print("\nSent to Telegram.")


if __name__ == "__main__":
    main()
