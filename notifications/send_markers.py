"""
Shared "confirmed sent today" marker helper for the time-critical
messages (9pm match preview, 9pm daily summary, 10pm evening picks).

Built after the pattern repeated: a Routine's trigger fires on schedule
(confirmed via its last_fired_at timestamp) but the actual script doesn't
complete - no Telegram send, no side effects - and since each firing runs
in an isolated fresh session, there's nothing to inspect afterward to find
out why. Rather than chase that root cause, every time-critical send
writes a marker file only after a CONFIRMED successful Telegram send, and
a periodic check (see check_and_retry_critical_messages.py) detects a
missing marker and retries.
"""

import os
from datetime import date


def write_sent_marker(path, today):
    with open(path, "w") as f:
        f.write(today.isoformat())


def sent_today(path, today):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return f.read().strip() == today.isoformat()
