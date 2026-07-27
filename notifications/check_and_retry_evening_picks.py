"""
Safety net for the 10pm evening picks job (the most critical message in
the system - see send_evening_picks.py). Built after the very first real
scheduled firing silently failed to complete: the Routine fired on time,
but the script never reached its final steps (no Telegram send, no
trades.csv update), and since that ran in an isolated fresh session there
was no way to inspect what went wrong after the fact.

Rather than trying to guess the root cause, this adds a detectable retry:
send_evening_picks.py writes data/last_evening_send.txt (today's date)
only after a CONFIRMED successful send. If it's late evening Central time
and that marker doesn't show today's date, the 10pm job evidently didn't
complete - so just run it again. Meant to be called from the same hourly
Routine that already does pre-match alerts and result checks, so it gets
picked up automatically without needing its own schedule slot.

Run with: python3 notifications/check_and_retry_evening_picks.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))

from fetch_kalshi import CENTRAL  # noqa: E402
import send_evening_picks  # noqa: E402

RETRY_WINDOW_START_HOUR_CT = 23  # only retry from 11pm CT onward - give the 10pm job an hour of slack first


def main():
    now_ct = datetime.now(CENTRAL)
    if now_ct.hour < RETRY_WINDOW_START_HOUR_CT:
        print(f"Not yet {RETRY_WINDOW_START_HOUR_CT}:00 CT - too early to check for a missed evening send.")
        return

    today_str = now_ct.date().isoformat()
    marker_ok = (
        os.path.exists(send_evening_picks.EVENING_SEND_MARKER)
        and open(send_evening_picks.EVENING_SEND_MARKER).read().strip() == today_str
    )
    if marker_ok:
        print(f"Evening picks were already sent successfully today ({today_str}) - nothing to do.")
        return

    print(f"No confirmed evening-picks send found for today ({today_str}) despite it being "
          f"{now_ct.strftime('%I:%M %p')} CT - the 10pm job appears to have failed. Retrying now.")
    send_evening_picks.main()


if __name__ == "__main__":
    main()
