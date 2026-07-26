# Tennis Kalshi Bot

Tennis match prediction + Kalshi edge-identification system. Separate project
from the MLB bot - different repo, different data, different everything.

## Status: Steps 1-4 of 8 built and confirmed with live data

Data pipeline, feature engineering, the 4-model ensemble, and Kalshi edge
identification are all working end to end. Telegram alerts, the trade
tracker, and the continuous-learning/scheduling steps (5-8) are next.

By decision (2026-07-26): **ATP only for now** - see "Known gaps" below.

## What's working right now

**Step 1 - data pipeline**
- `data_pipeline/fetch_historical_matches.py` - pulls ~10 years of ATP match
  history from Tennismylife/TML-Database (GitHub), then patches in everything
  more recent from Tennismylife's own live-website API (see "Known gaps" -
  the GitHub mirror turned out to be 6+ months stale). Full stats: aces,
  double faults, serve %, break points, rankings, surface, tier, round,
  indoor/outdoor, duration, retirement/walkover flag.
- `data_pipeline/fetch_kalshi.py` - live ATP + WTA match markets from the
  public Kalshi API, paired into one row per match with start times
  converted to Central Time for early-morning flagging.
- `data_pipeline/confirm_step1.py` - prints the Step 1 confirmation report.

**Step 2 - feature engineering**
- `data_pipeline/feature_engineering.py` - walks every match in chronological
  order, building surface/form/H2H/ranking-Elo/tournament/serve-return/
  physical-scheduling features from only what was known before that match
  (no leakage), with a randomized player_1/player_2 assignment per row.
- `data_pipeline/tournament_metadata.py` - hand-curated venue lookup
  (country/coordinates/altitude) for the ~60 tournaments that repeat yearly.

**Step 3 - the 4-model ensemble**
- `models/prepare_model_data.py` - encodes categoricals, chronological
  68/12/20 train/validation/test split.
- `models/train_ensemble.py` - logistic regression, 500-tree random forest,
  XGBoost (tuned via 5-fold `TimeSeriesSplit`), and a 64/32 dropout neural
  net (torch); weights the ensemble by validation Brier score, calibrates
  with isotonic regression, and reports accuracy/Brier/log-loss/ROI/
  calibration curve on the held-out test set. Artifacts saved to
  `models/artifacts/` (gitignored - regenerate by re-running the script).

**Step 4 - Kalshi edge identification**
- `models/predict_upcoming.py` - for each upcoming ATP match on Kalshi:
  matches player names to historical player IDs, infers surface/tournament
  context, builds live features from each player's current state, runs the
  trained ensemble, compares to the Kalshi price, and applies the HIGH
  CONVICTION / STANDARD / EARLY MORNING PRIORITY / skip rules with 25% Kelly
  bet sizing (capped at $144, $500 total exposure).

## Known gaps (disclosed, not silently worked around)

- **WTA is not included.** JeffSackmann/tennis_wta (GitHub) returns 404 on
  every file - verified independently via raw.githubusercontent.com,
  jsdelivr, and statically.io, most likely GitHub-side throttling of a very
  heavily-scraped repo. Decision (2026-07-26): proceed ATP-only rather than
  block the project on it. `fetch_wta_matches()` is fully implemented and
  will work the moment that access clears. Live WTA *markets* on Kalshi
  already work fine (KXWTAMATCH) - it's specifically the historical WTA
  stats that are missing, so there's no trained WTA model yet.
- **No Challenger/ITF-level results.** Both the GitHub mirror and its
  live-results supplement only cover ATP tour-level matches. Players whose
  recent record is mostly Challenger/ITF (common for early-morning matches
  outside the top tiers) won't be found and are skipped rather than guessed.
- **No live rankings/injury-news feed.** Current ranking is inferred from
  each player's most recent tracked match rather than a live rankings API.
  The retirement/injury skip rule only checks a manual watchlist
  (`data/injury_watchlist.csv`, empty by default) - there's no connected
  news source yet.
- **Tournament venue metadata (altitude/coordinates/home country) only
  covers ~60 recurring tournaments** - one-off events fall back to unknown
  rather than a guessed location.

## Running it yourself

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 data_pipeline/confirm_step1.py       # Step 1 check
python3 data_pipeline/feature_engineering.py # Step 2: builds data/processed_features.csv
python3 models/train_ensemble.py             # Step 3: trains + saves models/artifacts/
python3 models/predict_upcoming.py           # Step 4: tomorrow's Kalshi edge report
```
