"""
STEP 1 - Historical match data pipeline.

Downloads professional tennis match history (last 10 years) and standardizes
it into one clean table with the columns our prediction model needs:
winner/loser, tournament (name/surface/tier), rankings, serve & return stats,
score, round, indoor/outdoor, match duration, and a retirement/walkover flag.

Data sources:
- ATP: Tennismylife/TML-Database (GitHub). This is a community-maintained
  mirror/superset of Jeff Sackmann's tennis_atp dataset. We use it instead of
  JeffSackmann/tennis_atp directly because that repo's raw files are
  currently returning 404 from GitHub's CDN in this environment (confirmed
  with several independent services - GitHub itself appears to be
  throttling/blocking raw access to that specific repo, most likely because
  it is one of the most heavily-scraped tennis data sources on GitHub).
  The Tennismylife mirror has the same column layout plus an indoor/outdoor
  flag Sackmann's original doesn't have as its own column.
- WTA: JeffSackmann/tennis_wta. Same GitHub throttling issue applies here as
  well, so this currently fails. See fetch_wta_matches() - it is written to
  work the moment that repo becomes reachable again, and raises a clear
  error in the meantime instead of silently returning bad data.

Decision (2026-07-26): proceed on ATP data only for now rather than block
the whole project on WTA access. fetch_all_historical_matches() defaults to
ATP-only; pass include_wta=True to retry WTA once it's reachable again.
"""

import io
import sys
import time
import pandas as pd
import requests

CURRENT_YEAR = 2026
YEARS_BACK = 10
YEARS = list(range(CURRENT_YEAR - YEARS_BACK + 1, CURRENT_YEAR + 1))  # 2017-2026

ATP_URL_TEMPLATE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
WTA_URL_TEMPLATE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"

# The final columns our model needs, mapped from each source's native column
# names. Keys are our standardized names; values are the source column.
COLUMN_MAP = {
    "tourney_id": "tourney_id",
    "tourney_name": "tourney_name",
    "surface": "surface",
    "tourney_level": "tourney_level",
    "tourney_date": "tourney_date",
    "match_num": "match_num",
    "round": "round",
    "best_of": "best_of",
    "score": "score",
    "minutes": "minutes",
    "winner_id": "winner_id",
    "winner_name": "winner_name",
    "winner_hand": "winner_hand",
    "winner_ht": "winner_ht",
    "winner_ioc": "winner_ioc",
    "winner_age": "winner_age",
    "winner_rank": "winner_rank",
    "winner_rank_points": "winner_rank_points",
    "loser_id": "loser_id",
    "loser_name": "loser_name",
    "loser_hand": "loser_hand",
    "loser_ht": "loser_ht",
    "loser_ioc": "loser_ioc",
    "loser_age": "loser_age",
    "loser_rank": "loser_rank",
    "loser_rank_points": "loser_rank_points",
    "w_ace": "w_ace",
    "w_df": "w_df",
    "w_svpt": "w_svpt",
    "w_1stIn": "w_1stIn",
    "w_1stWon": "w_1stWon",
    "w_2ndWon": "w_2ndWon",
    "w_SvGms": "w_SvGms",
    "w_bpSaved": "w_bpSaved",
    "w_bpFaced": "w_bpFaced",
    "l_ace": "l_ace",
    "l_df": "l_df",
    "l_svpt": "l_svpt",
    "l_1stIn": "l_1stIn",
    "l_1stWon": "l_1stWon",
    "l_2ndWon": "l_2ndWon",
    "l_SvGms": "l_SvGms",
    "l_bpSaved": "l_bpSaved",
    "l_bpFaced": "l_bpFaced",
}


def _download_csv(url, timeout=30, retries=3):
    """Download a CSV file over HTTP and return it as a DataFrame."""
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), low_memory=False)
        except Exception as exc:  # noqa: BLE001 - want to retry on any network error
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def fetch_atp_matches(years=YEARS, verbose=True):
    """Pull ATP match history for the given years from Tennismylife/TML-Database."""
    frames = []
    for year in years:
        url = ATP_URL_TEMPLATE.format(year=year)
        if verbose:
            print(f"  ATP {year}: {url}")
        df = _download_csv(url)
        df["tour"] = "ATP"
        df["source_year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return _standardize(combined, tour="ATP")


def fetch_wta_matches(years=YEARS, verbose=True):
    """
    Pull WTA match history from JeffSackmann/tennis_wta.

    This currently fails in this environment because raw.githubusercontent.com
    is returning 404 for every file in that repo (verified independently via
    raw.githubusercontent.com, jsdelivr, and statically.io - all agree the
    repo's raw content is unreachable right now, most likely GitHub-side
    throttling of a very heavily-scraped repo rather than a real removal).
    This function is left fully implemented so it starts working again the
    moment that access is restored - no code changes needed, just re-run it.
    """
    frames = []
    errors = []
    for year in years:
        url = WTA_URL_TEMPLATE.format(year=year)
        if verbose:
            print(f"  WTA {year}: {url}")
        try:
            df = _download_csv(url, retries=1)
            df["tour"] = "WTA"
            df["source_year"] = year
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            errors.append((year, str(exc)))
    if not frames:
        raise RuntimeError(
            "Could not fetch ANY WTA data from JeffSackmann/tennis_wta - "
            f"all {len(errors)} year requests failed. First error: "
            f"{errors[0][1] if errors else 'unknown'}"
        )
    combined = pd.concat(frames, ignore_index=True)
    return _standardize(combined, tour="WTA")


def _standardize(df, tour):
    """Rename source columns to our standard schema and add derived flags."""
    out = pd.DataFrame()
    for our_name, source_name in COLUMN_MAP.items():
        out[our_name] = df[source_name] if source_name in df.columns else pd.NA

    # Indoor/outdoor: Tennismylife (ATP) has a dedicated column; Sackmann's
    # WTA files don't, so we fall back to guessing from the tournament name
    # for WTA until a better source is found.
    if "indoor" in df.columns:
        out["indoor"] = df["indoor"].map({"I": True, "O": False})
    else:
        out["indoor"] = df["tourney_name"].str.contains(
            "indoor", case=False, na=False
        )

    out["tour"] = tour

    # Retirement / walkover flag, detected from the score string, per the
    # user's instruction to exclude these matches from model training.
    score_str = out["score"].astype(str)
    out["retirement_or_walkover"] = (
        score_str.str.contains("RET", case=False, na=False)
        | score_str.str.contains("W/O", case=False, na=False)
        | score_str.str.contains("WALKOVER", case=False, na=False)
        | score_str.str.contains("DEF", case=False, na=False)
    )

    return out


def fetch_all_historical_matches(verbose=True, include_wta=False):
    """
    Fetch historical matches for the last 10 years. ATP-only by default -
    WTA (JeffSackmann/tennis_wta) is currently blocked at the GitHub CDN
    level for this environment, and the decision (2026-07-26) was to proceed
    on ATP data alone rather than block on it. Pass include_wta=True to try
    it again once that access is restored.
    """
    if verbose:
        print(f"Fetching ATP matches for {YEARS[0]}-{YEARS[-1]}...")
    atp = fetch_atp_matches(verbose=verbose)

    if not include_wta:
        return atp, None

    wta = None
    wta_error = None
    if verbose:
        print(f"Fetching WTA matches for {YEARS[0]}-{YEARS[-1]}...")
    try:
        wta = fetch_wta_matches(verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        wta_error = str(exc)
        if verbose:
            print(f"  WTA fetch failed: {wta_error}")

    combined = pd.concat([atp, wta], ignore_index=True) if wta is not None else atp
    return combined, wta_error


if __name__ == "__main__":
    matches, wta_err = fetch_all_historical_matches()
    print(f"\nTotal matches pulled: {len(matches):,}")
    print(matches["tour"].value_counts())
    if wta_err:
        print(f"\nWTA WARNING: {wta_err}")
    out_path = "data/raw/historical_matches.csv"
    matches.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    sys.exit(0)
