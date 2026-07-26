"""
STEP 7 - Weekly retraining, scheduled for Sunday 3am Central Time (Step 8
wires up the actual cron-style trigger; this script is what it calls).

Retrains from scratch on all currently-available data rather than
incrementally fine-tuning: fetch_historical_matches() already pulls
everything through "now" (CSV mirror + live-results supplement), so a full
retrain is simply re-running Steps 1-3 end to end. Simpler and safer than
incremental updates, and fast enough (a few minutes) to do weekly without
issue.

Run with: python3 models/retrain_weekly.py
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trades"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notifications"))

from fetch_historical_matches import fetch_all_historical_matches  # noqa: E402
from feature_engineering import build_training_features  # noqa: E402
from train_ensemble import main as train_ensemble_main, ARTIFACT_DIR  # noqa: E402
from analytics import (  # noqa: E402
    last_n_days_accuracy, accuracy_by_surface, accuracy_early_morning_vs_normal,
    flag_systematic_errors,
)
from format_messages import format_weekly_retrain_message  # noqa: E402
from telegram_bot import send_telegram_message  # noqa: E402

RAW_DATA_PATH = "data/raw/historical_matches.csv"
FEATURES_PATH = "data/processed_features.csv"


def main():
    old_metrics = None
    old_metrics_path = f"{ARTIFACT_DIR}/eval_metrics.json"
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path) as f:
            old_metrics = json.load(f)

    print("Step 1: re-pulling historical matches (CSV mirror + live-results supplement)...")
    matches, _ = fetch_all_historical_matches(verbose=True)
    matches.to_csv(RAW_DATA_PATH, index=False)

    print("\nStep 2: rebuilding features...")
    features, _players, _h2h = build_training_features(matches, verbose=True)
    features.to_csv(FEATURES_PATH, index=False)

    print("\nStep 3: retraining the ensemble...")
    train_ensemble_main()

    with open(old_metrics_path) as f:
        new_metrics = json.load(f)
    with open(f"{ARTIFACT_DIR}/top_features.json") as f:
        top_features = json.load(f)

    surface_acc = accuracy_by_surface()
    best_surface = max(surface_acc.items(), key=lambda kv: kv[1]["accuracy"]) if surface_acc else None
    worst_surface = min(surface_acc.items(), key=lambda kv: kv[1]["accuracy"]) if surface_acc else None
    if best_surface:
        best_surface = (best_surface[0], best_surface[1]["accuracy"])
    if worst_surface:
        worst_surface = (worst_surface[0], worst_surface[1]["accuracy"])

    early_stats = accuracy_early_morning_vs_normal()
    early_morning_acc = early_stats.get(True, {}).get("accuracy", float("nan"))

    stats = {
        "brier_old": old_metrics["brier"] if old_metrics else new_metrics["brier"],
        "brier_new": new_metrics["brier"],
        "acc_7d": last_n_days_accuracy(days=7),
        "acc_30d": last_n_days_accuracy(days=30),
        "early_morning_acc": early_morning_acc,
        "best_surface": best_surface,
        "worst_surface": worst_surface,
        "top_features": top_features,
        "systematic_error_flags": flag_systematic_errors(),
    }

    message = format_weekly_retrain_message(stats)
    print("\n--- Message to send ---")
    print(message)
    send_telegram_message(message)
    print("\nSent to Telegram.")


if __name__ == "__main__":
    main()
