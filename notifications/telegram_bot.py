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


def send_telegram_message(text, parse_mode="Markdown"):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - check .env"
        )
    url = f"{API_BASE.format(token=BOT_TOKEN)}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode},
        timeout=20,
    )
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
