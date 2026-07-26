"""
STEP 7 - Monthly report: accuracy by surface/tier/round, favorites vs
underdogs, early morning vs normal, ROI by confidence tier, and average
edge on winning vs losing picks. Meant to run on the 1st of each month
(Step 8 wires up the schedule).

Run with: python3 notifications/send_monthly_report.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))

from analytics import (  # noqa: E402
    accuracy_by_surface, accuracy_by_tier, accuracy_favorites_vs_underdogs,
    accuracy_early_morning_vs_normal, roi_by_confidence_tier, avg_edge_winning_vs_losing,
    _settled,
)
from format_messages import format_monthly_report  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402
from trade_tracker import TRADES_CSV_PATH  # noqa: E402


def main():
    if not os.path.exists(TRADES_CSV_PATH):
        print("No trades.csv yet - nothing to report.")
        return
    settled = _settled()
    if settled.empty:
        print("No settled trades yet - nothing to report.")
        return

    stats = {
        "by_surface": accuracy_by_surface(settled),
        "by_tier": accuracy_by_tier(settled),
        "favorites_vs_underdogs": accuracy_favorites_vs_underdogs(settled),
        "early_vs_normal": accuracy_early_morning_vs_normal(settled),
        "roi_by_tier": roi_by_confidence_tier(settled),
        "avg_edge": avg_edge_winning_vs_losing(settled),
    }
    message = format_monthly_report(stats)
    print(message)
    send_telegram_message(message)
    print("\nSent to Telegram.")


if __name__ == "__main__":
    main()
