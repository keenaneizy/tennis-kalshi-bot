"""
STEP 3 - Four-model ensemble for match-win probability.

Trains all four models on the same chronological TRAIN split, evaluates
each on the VALIDATION split to get a Brier score (used both to weight the
ensemble and to fit isotonic calibration), then reports final numbers on
the TEST split - which nothing in this file touches until the very end.

Models:
  1. Logistic Regression   - simple, interpretable baseline
  2. Random Forest (500)   - non-linear, tolerates missing data natively
  3. XGBoost                - tuned via 5-fold TimeSeriesSplit inside TRAIN
  4. Neural net (64/32)     - dropout-regularized, torch

Run with: python3 models/train_ensemble.py
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from prepare_model_data import load_features, encode_features, time_based_split

ARTIFACT_DIR = "models/artifacts"


# ---------------------------------------------------------------------------
# Model 4: small torch MLP with dropout
# ---------------------------------------------------------------------------

class TennisNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits


def train_neural_net(X_train, y_train, X_val, y_val, epochs=100, patience=10, lr=1e-3):
    torch.manual_seed(42)
    model = TennisNet(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train.values, dtype=torch.float32)
    Xval = torch.tensor(X_val, dtype=torch.float32)
    yval = torch.tensor(y_val.values, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def nn_predict_proba(model, X):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


# ---------------------------------------------------------------------------
# Training + evaluation orchestration
# ---------------------------------------------------------------------------

def train_all_models(splits, verbose=True):
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]

    # Imputed + scaled versions, for LR and the neural net. Fit on TRAIN only.
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)

    scaler = StandardScaler().fit(X_train_imp)
    X_train_scaled = scaler.transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    models = {}
    val_preds = {}

    if verbose:
        print("Training Model 1: Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X_train_scaled, y_train)
    models["logistic_regression"] = lr
    val_preds["logistic_regression"] = lr.predict_proba(X_val_scaled)[:, 1]

    if verbose:
        print("Training Model 2: Random Forest (500 trees)...")
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, min_samples_leaf=5)
    rf.fit(X_train, y_train)  # raw features, NaN handled natively
    models["random_forest"] = rf
    val_preds["random_forest"] = rf.predict_proba(X_val)[:, 1]

    if verbose:
        print("Training Model 3: XGBoost (5-fold time-series CV tuning)...")
    t0 = time.time()
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [200, 400],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    search = GridSearchCV(
        xgb_base, param_grid, cv=tscv, scoring="neg_brier_score", n_jobs=1,
    )
    search.fit(X_train, y_train)
    xgb_model = search.best_estimator_
    models["xgboost"] = xgb_model
    val_preds["xgboost"] = xgb_model.predict_proba(X_val)[:, 1]
    if verbose:
        print(f"  best params: {search.best_params_} ({time.time()-t0:.0f}s)")

    if verbose:
        print("Training Model 4: Neural network (64/32, dropout)...")
    nn_model = train_neural_net(X_train_scaled, y_train, X_val_scaled, y_val)
    models["neural_net"] = nn_model
    val_preds["neural_net"] = nn_predict_proba(nn_model, X_val_scaled)

    preprocessing = {"imputer": imputer, "scaler": scaler}
    return models, preprocessing, val_preds


def compute_ensemble_weights(val_preds, y_val):
    """Weight = inverse Brier score (lower Brier = better = higher weight), normalized to sum to 1."""
    briers = {name: brier_score_loss(y_val, p) for name, p in val_preds.items()}
    inv = {name: 1.0 / b for name, b in briers.items()}
    total = sum(inv.values())
    weights = {name: v / total for name, v in inv.items()}
    return weights, briers


def ensemble_predict(preds_dict, weights):
    combined = np.zeros_like(next(iter(preds_dict.values())))
    for name, p in preds_dict.items():
        combined += weights[name] * p
    return combined


def predict_all(models, preprocessing, X):
    X_imp = preprocessing["imputer"].transform(X)
    X_scaled = preprocessing["scaler"].transform(X_imp)
    return {
        "logistic_regression": models["logistic_regression"].predict_proba(X_scaled)[:, 1],
        "random_forest": models["random_forest"].predict_proba(X)[:, 1],
        "xgboost": models["xgboost"].predict_proba(X)[:, 1],
        "neural_net": nn_predict_proba(models["neural_net"], X_scaled),
    }


def roi_simulation(y_true, calibrated_prob, thresholds=(0.55, 0.60, 0.65)):
    """
    Flat, even-money ($1 stake, win +$1 / lose -$1) backtest of betting on
    whichever side the model favors, for matches where its confidence
    clears each threshold. This is a model-quality sanity check, NOT a
    real-money simulation - it assumes even odds because we don't have
    historical Kalshi prices merged into training data. Step 4 replaces this
    with real edge-vs-market-price calculations for live matches.
    """
    results = {}
    confidence = np.maximum(calibrated_prob, 1 - calibrated_prob)
    predicted_p1_wins = calibrated_prob > 0.5
    correct = predicted_p1_wins.astype(int) == y_true.values

    for t in thresholds:
        mask = confidence > t
        n_bets = int(mask.sum())
        if n_bets == 0:
            results[t] = {"n_bets": 0, "roi": None}
            continue
        profit = np.where(correct[mask], 1.0, -1.0).sum()
        results[t] = {
            "n_bets": n_bets,
            "win_rate": float(correct[mask].mean()),
            "roi": float(profit / n_bets),
        }
    return results


def calibration_curve_table(y_true, prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(prob, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_range": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(mask.sum()),
            "avg_predicted": float(prob[mask].mean()),
            "actual_win_rate": float(y_true.values[mask].mean()),
        })
    return rows


def top_features_logistic(model, feature_names, n=10):
    coefs = model.coef_[0]
    order = np.argsort(-np.abs(coefs))[:n]
    return [(feature_names[i], float(coefs[i])) for i in order]


def top_features_tree(model, feature_names, n=10):
    importances = model.feature_importances_
    order = np.argsort(-importances)[:n]
    return [(feature_names[i], float(importances[i])) for i in order]


def main():
    print("=" * 70)
    print("STEP 3 - TRAINING THE 4-MODEL ENSEMBLE")
    print("=" * 70)

    df = load_features()
    X, y, meta = encode_features(df)
    splits = time_based_split(X, y, meta)
    print(f"\nTrain: {len(splits['train']['X']):,} | Validation: {len(splits['val']['X']):,} "
          f"| Test: {len(splits['test']['X']):,}")

    models, preprocessing, val_preds = train_all_models(splits)

    y_val = splits["val"]["y"]
    weights, val_briers = compute_ensemble_weights(val_preds, y_val)

    print("\n--- Validation Brier scores + ensemble weights ---")
    for name in models:
        print(f"  {name:20s} Brier={val_briers[name]:.4f}  weight={weights[name]:.3f}")

    ensemble_val_pred = ensemble_predict(val_preds, weights)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(ensemble_val_pred, y_val)

    print("\n--- Feature importance ---")
    feature_names = list(splits["train"]["X"].columns)
    print("Top 10 features (Logistic Regression coefficients):")
    for name, coef in top_features_logistic(models["logistic_regression"], feature_names):
        print(f"    {name:40s} {coef:+.3f}")
    print("Top 10 features (XGBoost importances):")
    for name, imp in top_features_tree(models["xgboost"], feature_names):
        print(f"    {name:40s} {imp:.4f}")

    # ---- final evaluation on the never-touched TEST split ----
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 70)
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    test_preds = predict_all(models, preprocessing, X_test)
    ensemble_test_pred = ensemble_predict(test_preds, weights)
    calibrated_test_pred = calibrator.predict(ensemble_test_pred)

    acc = accuracy_score(y_test, calibrated_test_pred > 0.5)
    brier = brier_score_loss(y_test, calibrated_test_pred)
    ll = log_loss(y_test, calibrated_test_pred)
    auc = roc_auc_score(y_test, calibrated_test_pred)

    print(f"\nAccuracy:   {acc:.4f}")
    print(f"Brier score: {brier:.4f}  (lower is better; 0.25 = coin flip)")
    print(f"Log loss:   {ll:.4f}")
    print(f"ROC AUC:    {auc:.4f}")

    print("\nCalibration curve (predicted vs. actual win rate by bucket):")
    for row in calibration_curve_table(y_test, calibrated_test_pred):
        print(f"  {row['bin_range']}  n={row['n']:4d}  predicted={row['avg_predicted']:.3f}  actual={row['actual_win_rate']:.3f}")

    print("\nROI simulation (flat $1, even-money, model-quality check - see docstring):")
    roi = roi_simulation(y_test, calibrated_test_pred)
    for t, r in roi.items():
        if r["n_bets"] == 0:
            print(f"  confidence > {t}: no bets met threshold")
        else:
            print(f"  confidence > {t}: {r['n_bets']} bets, win rate {r['win_rate']:.3f}, ROI {r['roi']:+.3f}")

    # ---- persist everything for Step 4 / Step 7 ----
    import os
    import joblib
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(models["logistic_regression"], f"{ARTIFACT_DIR}/logistic_regression.joblib")
    joblib.dump(models["random_forest"], f"{ARTIFACT_DIR}/random_forest.joblib")
    joblib.dump(models["xgboost"], f"{ARTIFACT_DIR}/xgboost.joblib")
    torch.save(models["neural_net"].state_dict(), f"{ARTIFACT_DIR}/neural_net.pt")
    joblib.dump(preprocessing["imputer"], f"{ARTIFACT_DIR}/imputer.joblib")
    joblib.dump(preprocessing["scaler"], f"{ARTIFACT_DIR}/scaler.joblib")
    joblib.dump(calibrator, f"{ARTIFACT_DIR}/isotonic_calibrator.joblib")
    with open(f"{ARTIFACT_DIR}/ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    with open(f"{ARTIFACT_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(f"{ARTIFACT_DIR}/eval_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "brier": brier, "log_loss": ll, "auc": auc,
                    "val_briers": val_briers, "weights": weights}, f, indent=2)

    print(f"\nSaved all models/artifacts to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
