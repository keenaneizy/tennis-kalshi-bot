"""
STEP 7 - Continuous learning analytics.

Everything here reads from data/trades.csv - since every recommendation is
already logged there with surface, tier, round, early-morning flag, edge,
confidence tier, and (once resolved) the actual result, all the "rolling
accuracy metrics" the spec asks for are just group-bys over that log rather
than a separate tracking store. Called by the weekly retrain message and
the monthly report.
"""

import numpy as np
import pandas as pd

from trade_tracker import TRADES_CSV_PATH


def _settled(df=None):
    if df is None:
        df = pd.read_csv(TRADES_CSV_PATH)
    return df[df["prediction_correct"].notna()].copy()


def accuracy_by(df, group_col):
    settled = _settled(df)
    if settled.empty:
        return {}
    grouped = settled.groupby(group_col)["prediction_correct"].agg(["mean", "count"])
    return {idx: {"accuracy": row["mean"], "n": int(row["count"])} for idx, row in grouped.iterrows()}


def accuracy_by_surface(df=None):
    return accuracy_by(_settled(df), "surface")


def accuracy_by_tier(df=None):
    return accuracy_by(_settled(df), "tournament")


def accuracy_by_confidence_tier(df=None):
    return accuracy_by(_settled(df), "confidence_tier")


def accuracy_early_morning_vs_normal(df=None):
    settled = _settled(df)
    if settled.empty:
        return {}
    return accuracy_by(settled, "early_morning_flag")


def accuracy_favorites_vs_underdogs(df=None):
    settled = _settled(df)
    if settled.empty:
        return {}
    settled["is_favorite"] = settled["evening_kalshi_price"] > 0.5
    return accuracy_by(settled, "is_favorite")


def roi_by_confidence_tier(df=None):
    settled = _settled(df)
    if settled.empty:
        return {}
    stake_col = settled["actual_bet_placed"].where(settled["actual_bet_placed"].notna(), settled["recommended_bet_size"])
    settled = settled.assign(stake=stake_col)
    out = {}
    for tier, group in settled.groupby("confidence_tier"):
        total_staked = group["stake"].sum()
        total_pnl = group["profit_or_loss"].sum()
        out[tier] = {"roi": total_pnl / total_staked if total_staked else np.nan, "n": len(group)}
    return out


def avg_edge_winning_vs_losing(df=None):
    settled = _settled(df)
    if settled.empty:
        return {}
    return {
        "winning": settled.loc[settled["prediction_correct"].astype(bool), "edge_pct"].mean(),
        "losing": settled.loc[~settled["prediction_correct"].astype(bool), "edge_pct"].mean(),
    }


def flag_systematic_errors(df=None, min_sample=8, threshold_pp=15):
    """
    Simple systematic-error check: any surface/tier bucket with a decent
    sample size whose accuracy trails the overall accuracy by more than
    `threshold_pp` percentage points gets flagged for a human look.
    """
    settled = _settled(df)
    if len(settled) < min_sample:
        return []
    overall_acc = settled["prediction_correct"].mean()
    flags = []
    for group_col in ["surface", "confidence_tier"]:
        for key, stats in accuracy_by(settled, group_col).items():
            if stats["n"] >= min_sample and (overall_acc - stats["accuracy"]) * 100 >= threshold_pp:
                flags.append(
                    f"{group_col}={key}: {stats['accuracy']:.1%} accuracy over {stats['n']} picks "
                    f"vs {overall_acc:.1%} overall"
                )
    return flags


def last_n_days_accuracy(df=None, days=7):
    settled = _settled(df)
    if settled.empty:
        return np.nan
    settled["date"] = pd.to_datetime(settled["date"])
    cutoff = settled["date"].max() - pd.Timedelta(days=days)
    recent = settled[settled["date"] > cutoff]
    return recent["prediction_correct"].mean() if not recent.empty else np.nan


if __name__ == "__main__":
    df = _settled()
    print(f"Settled trades: {len(df)}")
    print("By surface:", accuracy_by_surface(df))
    print("By confidence tier:", accuracy_by_confidence_tier(df))
    print("Early morning vs normal:", accuracy_early_morning_vs_normal(df))
    print("Favorites vs underdogs:", accuracy_favorites_vs_underdogs(df))
    print("ROI by confidence tier:", roi_by_confidence_tier(df))
    print("Avg edge winning vs losing:", avg_edge_winning_vs_losing(df))
    print("Systematic error flags:", flag_systematic_errors(df))
