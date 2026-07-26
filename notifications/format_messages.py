"""
Formats the 5 Telegram message types from Step 5 of the spec. Every message
is prefixed "TENNIS PICKS" (or "TENNIS ...") so it's never confused with the
separate MLB bot's messages.

Message 1 (evening picks, 10pm CT) is the most important - it's the one
that has to fire every single night without exception, since it covers
matches that start before there's time to bet in the morning.
"""

from datetime import datetime


def _opp_line(r, tier_label):
    lines = [
        f"{tier_label}",
        f"{r['recommended_side']} vs {r['opponent']} — starts {r['start_time_ct'].split(' ', 1)[1]}",
        f"My model: {r['recommended_side']} wins at {r['model_prob']:.0%}",
        f"Kalshi price: {r['kalshi_price']*100:.0f}¢ — implied {r['implied_prob']:.0%}",
        f"Edge: +{r['edge_pp']:.1f} percentage points",
        f"Recommended: BUY YES on {r['recommended_side']}",
        f"Bet size: ${r['bet_size']:.0f}",
        f"Expected profit: ${r['expected_profit']:.2f}",
        f"Volume: ${r['volume']:,.0f}",
    ]
    if r.get("exposure_note"):
        lines.append(f"Note: {r['exposure_note']}")
    return "\n".join(lines)


def format_evening_message(recommendations, tomorrow_date, eval_metrics, model_record=None):
    """Message 1 - the primary 10pm picks message."""
    early = [r for r in recommendations if r["start_hour_ct"] is not None and r["start_hour_ct"] < 8]
    high_conviction = [r for r in recommendations if r["tier"] == "HIGH_CONVICTION"]
    standard = [r for r in recommendations if r["tier"] == "STANDARD"]

    tournaments = sorted({r["tournament"] for r in recommendations}) or ["No flagged matches"]
    header = f"🎾 TENNIS PICKS — {' / '.join(tournaments)} — {tomorrow_date:%B %d, %Y}"

    parts = [header, ""]

    if early:
        parts.append("🌙 EARLY MORNING — BET TONIGHT")
        for r in early:
            parts.append(_opp_line(r, "🌙 EARLY MORNING — BET TONIGHT"))
            parts.append("⚠️ Match starts before 8am — place this bet tonight")
            if r["volume"] < 1000:
                parts.append("⚠️ Kalshi volume is thin tonight — this may improve by match time, but there's no time to recheck in the morning")
            parts.append("")

    if high_conviction or standard:
        parts.append("— STANDARD MORNING GAMES (starts after 8am CT) —")
        parts.append("")
        for r in high_conviction:
            parts.append(_opp_line(r, "⭐ HIGH CONVICTION"))
            parts.append("Note: Recheck Kalshi price in morning before betting — price may move overnight")
            if r["volume"] < 1000:
                parts.append("⚠️ Kalshi volume is thin tonight — recheck in the morning")
            parts.append("")
        for r in standard:
            parts.append(_opp_line(r, "▶ STANDARD PICK"))
            parts.append("Note: Recheck Kalshi price in morning before betting — price may move overnight")
            if r["volume"] < 1000:
                parts.append("⚠️ Kalshi volume is thin tonight — recheck in the morning")
            parts.append("")

    if not recommendations:
        parts.append("No opportunities met the edge/volume thresholds tonight.")
        parts.append("")

    total_expected_profit = sum(r["expected_profit"] for r in recommendations)
    parts.append(
        f"Total opportunities tomorrow: {len(recommendations)} "
        f"({len(early)} early morning, {len(high_conviction)} high conviction, {len(standard)} standard)"
    )
    parts.append(f"Total expected profit if all hit: ${total_expected_profit:.2f}")
    if model_record:
        parts.append(f"Model current record: {model_record['wins']}-{model_record['losses']} "
                      f"— {model_record['accuracy']:.1%} accuracy")
    else:
        parts.append(f"Model current record: no settled bets yet — "
                      f"held-out test accuracy {eval_metrics['accuracy']:.1%}")
    parts.append(f"Current Brier score: {eval_metrics['brier']:.4f}")

    return "\n".join(parts)


def format_morning_update(matches_after_8am, results_so_far=None):
    """Message 2 - 8am confirmation, only matches starting after 8am."""
    header = f"🎾 TENNIS MORNING UPDATE — {datetime.now():%B %d, %Y}"
    parts = [header, ""]
    for m in matches_after_8am:
        movement = "unchanged"
        if m["current_price"] > m["evening_price"]:
            movement = "up"
        elif m["current_price"] < m["evening_price"]:
            movement = "down"
        current_edge = (m["model_prob"] - m["current_price"]) * 100
        if current_edge >= 5:
            action = "STILL BET" if current_edge <= m["edge_pp"] else "BETTER VALUE — price moved in our favor"
        else:
            action = "SKIP — edge gone"
        parts.append(f"{m['recommended_side']} vs {m['opponent']} — starts {m['start_time_ct']}")
        parts.append(f"Last night's Kalshi price: {m['evening_price']*100:.0f}¢")
        parts.append(f"Current Kalshi price: {m['current_price']*100:.0f}¢")
        parts.append(f"Price movement: {movement}")
        parts.append(f"Edge update: {current_edge:+.1f} percentage points")
        parts.append(f"Action: {action}")
        parts.append("")
    if results_so_far:
        parts.append(f"Early morning results so far today: {results_so_far['wins']}-{results_so_far['losses']}")
    parts.append(f"Confirmed bets for today: {len(matches_after_8am)} matches")
    return "\n".join(parts)


def format_pre_match_alert(rec, minutes_until=30):
    return "\n".join([
        "🎾 MATCH STARTING SOON",
        f"{rec['recommended_side']} vs {rec['opponent']} in {minutes_until} minutes",
        f"Current Kalshi price: {rec['current_price']*100:.0f}¢",
        f"Edge still valid: {'yes' if rec['edge_still_valid'] else 'no'}",
        f"Reminder: Bet size ${rec['bet_size']:.0f}",
    ])


def format_result_update(result):
    correct = result["predicted_winner"] == result["actual_winner"]
    return "\n".join([
        f"🎾 RESULT: {result['player_1']} vs {result['player_2']}",
        "✅ CORRECT" if correct else "❌ WRONG",
        f"Predicted: {result['predicted_winner']} at {result['model_prob']:.0%}",
        f"Actual winner: {result['actual_winner']}",
        f"Kalshi profit or loss: ${result['pnl']:+.2f}",
        f"Today's running P&L: ${result['running_pnl_today']:+.2f}",
        f"Model record today: {result['record_today_wins']}-{result['record_today_losses']}",
    ])


def format_daily_summary(summary):
    brier_trend = (
        "improving" if summary["brier_today"] < summary["brier_yesterday"]
        else "declining" if summary["brier_today"] > summary["brier_yesterday"]
        else "flat"
    )
    return "\n".join([
        f"🎾 TENNIS DAILY SUMMARY — {datetime.now():%B %d, %Y}",
        f"Today's record: {summary['today_wins']}-{summary['today_losses']}",
        f"Today's P&L: ${summary['today_pnl']:+.2f}",
        f"Running model record: {summary['total_wins']}-{summary['total_losses']} "
        f"— {summary['total_accuracy']:.1%} accuracy",
        f"Running P&L: ${summary['total_pnl']:+.2f}",
        f"Current bankroll: ${summary['bankroll']:,.2f}",
        f"Brier score: {summary['brier_today']:.4f} — {brier_trend} versus yesterday",
        f"Tomorrow's early morning games: {summary['tomorrow_early_morning_count']} matches flagged "
        f"— watch for 10pm picks tonight",
    ])


def format_weekly_retrain_message(stats):
    lines = [
        f"🎾 TENNIS MODEL UPDATE — {datetime.now():%B %d, %Y}",
        "Weekly retraining complete",
        f"Brier score: {stats['brier_old']:.4f} → {stats['brier_new']:.4f}",
        f"Last 7 days accuracy: {stats['acc_7d']:.1%}" if stats["acc_7d"] == stats["acc_7d"] else "Last 7 days accuracy: n/a (not enough settled picks yet)",
        f"Last 30 days accuracy: {stats['acc_30d']:.1%}" if stats["acc_30d"] == stats["acc_30d"] else "Last 30 days accuracy: n/a (not enough settled picks yet)",
        f"Early morning match accuracy: {stats['early_morning_acc']:.1%} — tracking separately"
        if stats["early_morning_acc"] == stats["early_morning_acc"] else "Early morning match accuracy: n/a yet",
    ]
    if stats.get("best_surface"):
        lines.append(f"Best performing surface: {stats['best_surface'][0]} at {stats['best_surface'][1]:.1%}")
    if stats.get("worst_surface"):
        lines.append(f"Worst performing surface: {stats['worst_surface'][0]} at {stats['worst_surface'][1]:.1%}")
    lines.append("Top 5 most predictive features this week:")
    for name, importance in stats["top_features"]:
        lines.append(f"  {name}: {importance:.4f}")
    if stats.get("systematic_error_flags"):
        lines.append("")
        lines.append("⚠️ Possible systematic errors:")
        for flag in stats["systematic_error_flags"]:
            lines.append(f"  {flag}")
    return "\n".join(lines)


def format_monthly_report(stats):
    lines = [f"🎾 TENNIS MONTHLY REPORT — {datetime.now():%B %Y}", ""]

    lines.append("Accuracy by surface:")
    for surface, s in stats["by_surface"].items():
        lines.append(f"  {surface}: {s['accuracy']:.1%} ({s['n']} picks)")

    lines.append("\nAccuracy by tournament tier:")
    for tier, s in stats["by_tier"].items():
        lines.append(f"  {tier}: {s['accuracy']:.1%} ({s['n']} picks)")

    lines.append("\nAccuracy — favorites vs underdogs:")
    for is_fav, s in stats["favorites_vs_underdogs"].items():
        label = "Favorites" if is_fav else "Underdogs"
        lines.append(f"  {label}: {s['accuracy']:.1%} ({s['n']} picks)")

    lines.append("\nAccuracy — early morning vs normal hours:")
    for is_early, s in stats["early_vs_normal"].items():
        label = "Early morning" if is_early else "Normal hours"
        lines.append(f"  {label}: {s['accuracy']:.1%} ({s['n']} picks)")

    lines.append("\nROI — HIGH CONVICTION vs STANDARD:")
    for tier, s in stats["roi_by_tier"].items():
        lines.append(f"  {tier}: {s['roi']:+.1%} ROI ({s['n']} picks)")

    lines.append("\nAverage edge — winning vs losing picks:")
    lines.append(f"  Winning picks: +{stats['avg_edge']['winning']:.1f}pp")
    lines.append(f"  Losing picks: +{stats['avg_edge']['losing']:.1f}pp")

    return "\n".join(lines)
