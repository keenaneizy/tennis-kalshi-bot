"""
STEP 1 CONFIRMATION REPORT.

Runs the historical-data pull and the live Kalshi pull, then prints exactly
the checks the user asked for before Step 2 (feature engineering) is allowed
to start:
  - Total historical matches pulled
  - Sample of 5 recent matches with full statistics
  - Tomorrow's scheduled ATP/WTA matches, including early morning start times
  - Current Kalshi tennis markets with YES/NO prices

Run with: python3 data_pipeline/confirm_step1.py
"""

from datetime import timedelta

from fetch_historical_matches import fetch_all_historical_matches
from fetch_kalshi import fetch_all_tennis_matches, matches_for_date, CENTRAL
from datetime import datetime


def main():
    print("=" * 70)
    print("STEP 1 CONFIRMATION - HISTORICAL DATA")
    print("=" * 70)
    matches, wta_error = fetch_all_historical_matches(verbose=False)
    print(f"\nTotal historical matches pulled: {len(matches):,}")
    print(matches["tour"].value_counts().to_string())
    print(f"Years covered: {matches['tourney_date'].astype(str).str[:4].min()} - "
          f"{matches['tourney_date'].astype(str).str[:4].max()}")
    n_retirements = matches["retirement_or_walkover"].sum()
    print(f"Retirement/walkover matches flagged (excluded from training later): {n_retirements:,}")
    if wta_error:
        print(f"\n*** WTA WARNING: {wta_error}")
        print("*** ATP-only until a working WTA data source is confirmed.")

    print("\n--- Sample of 5 recent matches with full statistics ---")
    recent = matches[matches["tour"] == "ATP"].sort_values("tourney_date").tail(5)
    cols_to_show = [
        "tourney_name", "surface", "tourney_level", "round", "tourney_date",
        "winner_name", "winner_rank", "loser_name", "loser_rank", "score",
        "minutes", "indoor",
        "w_ace", "w_df", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
    ]
    for _, row in recent.iterrows():
        print(f"\n{row['tourney_name']} ({row['surface']}, {row['tourney_level']}, "
              f"{'indoor' if row['indoor'] else 'outdoor'}) - {row['round']} - {row['tourney_date']}")
        print(f"  {row['winner_name']} (rank {row['winner_rank']}) def. "
              f"{row['loser_name']} (rank {row['loser_rank']})  {row['score']}  ({row['minutes']} min)")
        print(f"  Winner: {row['w_ace']} aces, {row['w_df']} DFs, "
              f"1st in {row['w_1stIn']}, 1st won {row['w_1stWon']}, 2nd won {row['w_2ndWon']}, "
              f"BP saved {row['w_bpSaved']}/{row['w_bpFaced']}")
        print(f"  Loser:  {row['l_ace']} aces, {row['l_df']} DFs, "
              f"1st in {row['l_1stIn']}, 1st won {row['l_1stWon']}, 2nd won {row['l_2ndWon']}, "
              f"BP saved {row['l_bpSaved']}/{row['l_bpFaced']}")

    print("\n" + "=" * 70)
    print("STEP 1 CONFIRMATION - LIVE KALSHI + SCHEDULE DATA")
    print("=" * 70)
    kalshi_matches = fetch_all_tennis_matches()
    print(f"\nTotal open Kalshi tennis match markets right now: {len(kalshi_matches)}")

    today_ct = datetime.now(CENTRAL).date()
    tomorrow_ct = today_ct + timedelta(days=1)
    tomorrow_matches = matches_for_date(kalshi_matches, tomorrow_ct)
    tomorrow_matches.sort(key=lambda m: m["start_datetime_ct"])

    print(f"\n--- Tomorrow's scheduled matches ({tomorrow_ct}), derived from Kalshi listings ---")
    early = [m for m in tomorrow_matches if m["start_hour_ct"] is not None and m["start_hour_ct"] < 8]
    standard = [m for m in tomorrow_matches if m not in early]

    print(f"\nEARLY MORNING (before 8am CT) - {len(early)} match(es):")
    for m in early:
        print(f"  [{m['tour']}] {m['matchup']} - {m['start_time_ct']} - "
              f"{m['player_1']} ask {m['player_1_yes_ask']} / {m['player_2']} ask {m['player_2_yes_ask']} "
              f"- vol {m['player_1_volume']}")

    print(f"\nSTANDARD MORNING/DAY (8am CT or later) - {len(standard)} match(es), showing first 10:")
    for m in standard[:10]:
        print(f"  [{m['tour']}] {m['matchup']} - {m['start_time_ct']} - "
              f"{m['player_1']} ask {m['player_1_yes_ask']} / {m['player_2']} ask {m['player_2_yes_ask']} "
              f"- vol {m['player_1_volume']}")

    print("\n--- Current Kalshi tennis markets (YES/NO prices), sample of 10 ---")
    for m in kalshi_matches[:10]:
        yes_no_p1 = f"YES {m['player_1_yes_ask']} / NO {round(1 - float(m['player_1_yes_ask']), 2) if m['player_1_yes_ask'] else '?'}"
        print(f"  [{m['tour']}] {m['matchup']}: {m['player_1']} -> {yes_no_p1}, vol {m['player_1_volume']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
