"""
Safety net for the three time-critical nightly messages: 9pm match
preview, 9pm daily summary, and 10pm evening picks (the most critical -
see send_evening_picks.py).

Built after this exact pattern repeated twice in the first two nights of
running: a Routine's trigger fires on schedule (confirmed via its
last_fired_at timestamp) but the underlying script doesn't complete - no
Telegram send, no marker written - and since each firing runs in an
isolated fresh session, there's nothing to inspect afterward to find out
why. Rather than chase that root cause, each send_*.py script writes a
"data/last_*_send.txt" marker (today's date) only after a CONFIRMED
successful Telegram send (see send_markers.py). This script checks all
three markers on an hourly cadence and retries whichever is missing, once
it's had a reasonable window to complete on its own:

  - match preview / daily summary (target 9pm CT): retry-eligible from
    10pm CT onward (the first hourly check after their 9pm target)
  - evening picks (target 10pm CT): retry-eligible from 11pm CT onward

Meant to be called from the same hourly Routine that already does
pre-match alerts and result checks, so it gets picked up automatically
without needing its own schedule slot.

Run with: python3 notifications/check_and_retry_critical_messages.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))

from fetch_kalshi import CENTRAL  # noqa: E402
from send_markers import sent_today  # noqa: E402
import send_match_preview  # noqa: E402
import send_daily_summary  # noqa: E402
import send_evening_picks  # noqa: E402

JOBS = [
    ("match preview", send_match_preview, 22),
    ("daily summary", send_daily_summary, 22),
    ("evening picks", send_evening_picks, 23),
]


def main():
    now_ct = datetime.now(CENTRAL)
    today = now_ct.date()

    for label, module, retry_hour in JOBS:
        if now_ct.hour < retry_hour:
            print(f"{label}: not yet {retry_hour}:00 CT - too early to check.")
            continue
        if sent_today(module.SENT_MARKER if hasattr(module, "SENT_MARKER") else module.EVENING_SEND_MARKER, today):
            print(f"{label}: already confirmed sent today - nothing to do.")
            continue
        print(f"{label}: no confirmed send found for today despite it being "
              f"{now_ct.strftime('%I:%M %p')} CT - retrying now.")
        module.main()


if __name__ == "__main__":
    main()
