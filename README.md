# Tennis Kalshi Bot

Tennis match prediction + Kalshi edge-identification system. Separate project
from the MLB bot - different repo, different data, different everything.

## Status: Step 1 of 8 (data pipeline) built and confirmed with live data

See the task plan for the full 8-step build. Only Step 1 (pull historical
match data + connect to live Kalshi tennis markets) is done so far, per
instructions to validate each step before moving to the next.

## What's working right now

- `data_pipeline/fetch_historical_matches.py` - pulls 10 years of ATP match
  history (2017-2026, ~25,000 matches) from Tennismylife/TML-Database on
  GitHub, with full stats: aces, double faults, serve %, break points,
  rankings, surface, tier, round, indoor/outdoor, duration, and a
  retirement/walkover flag for excluding those matches from training later.
- `data_pipeline/fetch_kalshi.py` - pulls live ATP + WTA match markets from
  the public Kalshi API (no auth needed for market data), pairs up the
  YES/NO markets for each match, and converts start times to US Central Time
  so early-morning (before 8am CT) matches can be flagged.
- `data_pipeline/confirm_step1.py` - runs both of the above and prints the
  Step 1 confirmation report (total matches, a sample of 5, tomorrow's
  schedule split into early-morning vs. standard, and current Kalshi prices).

**Known gap:** WTA historical data (JeffSackmann/tennis_wta) is currently
unreachable - GitHub is returning 404 on every file in that repo, verified
independently via raw.githubusercontent.com, jsdelivr, and statically.io.
This looks like GitHub-side throttling of a very heavily-scraped repo, not a
real deletion. `fetch_wta_matches()` is fully implemented and will start
working the moment that access is restored - no code changes needed, just
re-run it. Live WTA *markets* on Kalshi work fine already (KXWTAMATCH);
it's specifically the historical WTA match stats that are blocked right now.

## Running Step 1 yourself

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 data_pipeline/confirm_step1.py
```
