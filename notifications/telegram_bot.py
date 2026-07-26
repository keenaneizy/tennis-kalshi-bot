"""
Thin wrapper around the Telegram Bot API. Credentials come from .env (never
committed - see .gitignore) so the bot token isn't sitting in the repo.

Every message this project sends is prefixed so it's unmistakably separate
from the MLB bot's picks, per the "completely separate project" requirement.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = "https://api.telegram.org/bot{token}"


def send_telegram_message(text, parse_mode=None):
    """
    parse_mode defaults to plain text. None of our messages use deliberate
    Markdown (bold/italic) - they're plain lines with emoji - and several
    (the weekly retrain message, in particular) include raw feature names
    like "rank_diff_p1_minus_p2", which Telegram's Markdown parser reads as
    unmatched italic markers and rejects with a 400. Plain text sidesteps
    that entirely; pass parse_mode="Markdown" explicitly if a future
    message actually needs formatting AND is guaranteed not to contain
    stray _ or * characters.
    """
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - check .env"
        )
    url = f"{API_BASE.format(token=BOT_TOKEN)}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode
    resp = requests.post(url, data=data, timeout=20)
    resp.raise_for_status()
    result = resp.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API rejected the message: {result}")
    return result


if __name__ == "__main__":
    send_telegram_message(
        "🎾 TENNIS PICKS bot connected - this is a one-time test message. "
        "If you can read this, the bot is wired up correctly."
    )
    print("Test message sent - check Telegram.")
