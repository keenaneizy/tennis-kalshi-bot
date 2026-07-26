"""
STEP 6 - Trade tracker (data/trades.csv).

One row per recommended trade, created when the 10pm evening message goes
out, then filled in over time: the 8am morning price (Message 2), the
match result once it resolves (Message 4/Step 7), and the running
bankroll/ROI/Brier trend those depend on.

"Actual bet placed" is manual by design - the user decides what they
actually put money on, this system only tracks recommendations and lets
them fill in reality alongside it.
"""

import os

import numpy as np
import pandas as pd

TRADES_CSV_PATH = "data/trades.csv"
STARTING_BANKROLL = 7200.0

COLUMNS = [
    "date", "tournament", "round", "player_1", "player_2", "surface",
    "event_ticker", "match_start_time_ct", "early_morning_flag",
    "recommended_player", "model_predicted_probability",
    "evening_kalshi_price", "morning_kalshi_price", "edge_pct", "confidence_tier",
    "recommended_bet_size", "actual_bet_placed",
    "match_result_winner", "prediction_correct", "profit_or_loss",
    "running_bankroll", "running_roi", "brier_score_that_day",
    "pre_match_alert_sent",
]


def _load_or_init():
    if os.path.exists(TRADES_CSV_PATH):
        df = pd.read_csv(TRADES_CSV_PATH)
        if "pre_match_alert_sent" not in df.columns:
            df["pre_match_alert_sent"] = False
        return df
    # dtype=object so an all-empty column (e.g. match_result_winner before any
    # match resolves) doesn't get inferred as float64 and then reject a
    # string being written into it later.
    return pd.DataFrame(columns=COLUMNS, dtype=object)


def append_evening_recommendations(recommendations, tomorrow_date):
    """Called right after the 10pm message is sent - logs every recommendation as a new row."""
    df = _load_or_init()
    new_rows = []
    for r in recommendations:
        new_rows.append({
            "date": tomorrow_date.isoformat(),
            "tournament": r["tournament"], "round": r["round"],
            "player_1": r["recommended_side"], "player_2": r["opponent"],
            "surface": r.get("surface", ""), "event_ticker": r.get("event_ticker", ""),
            "match_start_time_ct": r["start_time_ct"], "early_morning_flag": r["start_hour_ct"] is not None and r["start_hour_ct"] < 8,
            "recommended_player": r["recommended_side"], "model_predicted_probability": r["model_prob"],
            "evening_kalshi_price": r["kalshi_price"], "morning_kalshi_price": np.nan,
            "edge_pct": r["edge_pp"], "confidence_tier": r["tier"],
            "recommended_bet_size": r["bet_size"], "actual_bet_placed": np.nan,
            "match_result_winner": np.nan, "prediction_correct": np.nan, "profit_or_loss": np.nan,
            "running_bankroll": np.nan, "running_roi": np.nan, "brier_score_that_day": np.nan,
            "pre_match_alert_sent": False,
        })
    if not new_rows:
        return df
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(TRADES_CSV_PATH, index=False)
    return df


def update_morning_price(match_start_time_ct, recommended_player, new_price):
    df = _load_or_init()
    mask = (df["match_start_time_ct"] == match_start_time_ct) & (df["recommended_player"] == recommended_player)
    df.loc[mask, "morning_kalshi_price"] = new_price
    df.to_csv(TRADES_CSV_PATH, index=False)
    return df


def record_result(match_start_time_ct, recommended_player, actual_winner, brier_score_that_day=None):
    """Fills in the outcome for a trade once its match resolves, and rolls bankroll/ROI forward."""
    df = _load_or_init()
    mask = (df["match_start_time_ct"] == match_start_time_ct) & (df["recommended_player"] == recommended_player)
    if not mask.any():
        raise ValueError(f"No trade row found for {recommended_player} at {match_start_time_ct}")

    idx = df[mask].index[0]
    row = df.loc[idx]
    correct = bool(actual_winner == row["recommended_player"])
    bet_size = row["actual_bet_placed"] if pd.notna(row["actual_bet_placed"]) else row["recommended_bet_size"]
    price = row["evening_kalshi_price"]

    if correct:
        pnl = bet_size * (1 - price) / price
    else:
        pnl = -bet_size

    # An all-empty column round-tripped through CSV gets inferred as
    # float64, which then rejects a string/bool being written into it.
    for col in ["match_result_winner", "prediction_correct"]:
        if df[col].dtype != object:
            df[col] = df[col].astype(object)

    df.loc[idx, "match_result_winner"] = actual_winner
    df.loc[idx, "prediction_correct"] = correct
    df.loc[idx, "profit_or_loss"] = round(pnl, 2)
    if brier_score_that_day is not None:
        df.loc[idx, "brier_score_that_day"] = brier_score_that_day

    settled = df[df["profit_or_loss"].notna()].sort_values("date")
    running_bankroll = STARTING_BANKROLL + settled["profit_or_loss"].cumsum()
    total_staked = settled["recommended_bet_size"].where(
        settled["actual_bet_placed"].isna(), settled["actual_bet_placed"]
    ).cumsum()
    running_roi = settled["profit_or_loss"].cumsum() / total_staked.replace(0, np.nan)
    df.loc[settled.index, "running_bankroll"] = running_bankroll.values
    df.loc[settled.index, "running_roi"] = running_roi.values

    df.to_csv(TRADES_CSV_PATH, index=False)
    return df


def compute_model_record(df=None):
    """W-L, accuracy, bankroll, and ROI across every settled trade - used in Telegram messages."""
    if df is None:
        df = _load_or_init()
    settled = df[df["prediction_correct"].notna()]
    if settled.empty:
        return None
    wins = int(settled["prediction_correct"].sum())
    losses = len(settled) - wins
    return {
        "wins": wins,
        "losses": losses,
        "accuracy": wins / len(settled),
        "total_pnl": float(settled["profit_or_loss"].sum()),
        "bankroll": float(settled["running_bankroll"].iloc[-1]) if settled["running_bankroll"].notna().any() else STARTING_BANKROLL,
        "roi": float(settled["running_roi"].iloc[-1]) if settled["running_roi"].notna().any() else np.nan,
    }


if __name__ == "__main__":
    df = _load_or_init()
    print(f"{TRADES_CSV_PATH}: {len(df)} rows")
    record = compute_model_record(df)
    print("Settled record:", record if record else "no settled trades yet")
