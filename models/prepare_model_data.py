"""
Turns the Step 2 feature table (data/processed_features.csv) into arrays
the four models can actually train on: categorical columns encoded to
numbers, metadata columns (names/ids/dates) separated out, and a
chronological train/validation/test split.

Split logic (time-based, never shuffled):
  - Oldest 68% of matches  -> TRAIN      (fit the four models)
  - Next 12%               -> VALIDATION (compute ensemble weights + fit
                                           isotonic calibration)
  - Most recent 20%        -> TEST       (final, never-touched-until-the-end
                                           evaluation: accuracy/Brier/ROI/etc)
This is a stricter version of the "80% train / 20% validation" spec: the
20% held-out slice is kept completely separate from anything that touches
model weights (including the ensemble weights and calibration curve), so
the reported test metrics aren't optimistic.
"""

import numpy as np
import pandas as pd

ROUND_ORDER = {
    "R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5,
    "F": 6, "RR": 3.5, "3rd/4th": 5.5, "BR": 5.5,
}
TREND_ORDER = {"declining": -1, "stable": 0, "improving": 1, "unknown": np.nan}

META_COLS = [
    "match_date", "tourney_id", "tourney_name",
    "player_1_id", "player_1_name", "player_2_id", "player_2_name",
]
TARGET_COL = "player_1_won"


def load_features(path="data/processed_features.csv"):
    df = pd.read_csv(path, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def encode_features(df):
    """
    Return (X, y, meta) with all-numeric X, ready for modeling. Works for
    both training (target column present) and live inference (no target
    yet) - y is None in the latter case.
    """
    df = df.copy()

    df["round_ordinal"] = df["round"].map(ROUND_ORDER)
    df["indoor_num"] = df["indoor"].map({True: 1.0, False: 0.0})
    df["p1_x_surf_trend_num"] = df["p1_x_surf_trend_label"].map(TREND_ORDER)
    df["p2_x_surf_trend_num"] = df["p2_x_surf_trend_label"].map(TREND_ORDER)
    df["p1_x_home_country_advantage"] = df["p1_x_home_country_advantage"].astype(float)
    df["p2_x_home_country_advantage"] = df["p2_x_home_country_advantage"].astype(float)
    df["p1_x_high_altitude"] = df["p1_x_high_altitude"].astype(float)
    df["p2_x_high_altitude"] = df["p2_x_high_altitude"].astype(float)

    surface_dummies = pd.get_dummies(df["surface"], prefix="surface", dtype=float)
    level_dummies = pd.get_dummies(df["tourney_level"], prefix="level", dtype=float)

    drop_cols = META_COLS + [
        "surface", "tourney_level", "round", "indoor",
        "p1_x_surf_trend_label", "p2_x_surf_trend_label", TARGET_COL,
    ]
    numeric = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = pd.concat([numeric, surface_dummies, level_dummies], axis=1)
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    meta = df[[c for c in META_COLS if c in df.columns]]
    return X, y, meta


def align_to_training_columns(X, feature_names):
    """
    Reindex a live-inference feature row to the exact column set/order the
    models were trained on. Only affects one-hot surface/level columns that
    a single upcoming match won't produce on its own (e.g. a Clay match has
    no 'surface_Grass' column until reindexed in as 0) - genuinely-missing
    numeric features (e.g. no head-to-head history yet) stay NaN.
    """
    return X.reindex(columns=feature_names, fill_value=0.0)


def time_based_split(X, y, meta, train_frac=0.68, val_frac=0.12):
    """Chronological split - meta['match_date'] is already sorted ascending."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {}
    for name, (start, end) in {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }.items():
        splits[name] = {
            "X": X.iloc[start:end].reset_index(drop=True),
            "y": y.iloc[start:end].reset_index(drop=True),
            "meta": meta.iloc[start:end].reset_index(drop=True),
        }
    return splits


if __name__ == "__main__":
    df = load_features()
    X, y, meta = encode_features(df)
    splits = time_based_split(X, y, meta)
    for name, s in splits.items():
        print(f"{name}: {len(s['X']):,} rows, {meta.iloc[0]['match_date'].date() if name=='train' else ''} "
              f"date range {s['meta']['match_date'].min().date()} to {s['meta']['match_date'].max().date()}")
    print(f"\nFeature count: {X.shape[1]}")
