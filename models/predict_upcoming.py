"""
STEP 4 - Kalshi edge identification.

For every ATP match on tomorrow's Kalshi schedule:
  1. Match the two Kalshi player names to a player in our historical data
  2. Work out the surface/tournament-level/indoor context (Kalshi doesn't
     give us these directly - we infer them from the tournament name)
  3. Build the same pre-match features Step 2 uses, from each player's
     CURRENT state (i.e. history built from every match up to today)
  4. Run the Step 3 ensemble + isotonic calibration to get a win probability
  5. Compare to the Kalshi-implied probability and compute edge / EV
  6. Apply the HIGH CONVICTION / STANDARD / EARLY MORNING PRIORITY / skip
     rules and size the recommended bet with 25% Kelly (capped at $144)

Known gap, flagged rather than faked: there's no connected live
injury/retirement-news feed, so the "retirement risk" skip rule currently
only checks a manual watchlist (data/injury_watchlist.csv, empty by
default). Add a player name to that file to force a skip.

Run with: python3 models/predict_upcoming.py
"""

import difflib
import os
import re
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))

from fetch_historical_matches import fetch_all_historical_matches  # noqa: E402
from fetch_kalshi import fetch_all_tennis_matches, matches_for_date, CENTRAL  # noqa: E402
from feature_engineering import (  # noqa: E402
    build_training_features, surface_performance_features, recent_form_features,
    h2h_features, ranking_features, tournament_context_features,
    serve_return_features, physical_scheduling_features, ELO_START,
)
from tournament_metadata import get_tournament_metadata  # noqa: E402
from prepare_model_data import encode_features, align_to_training_columns  # noqa: E402
from train_ensemble import TennisNet, nn_predict_proba, ensemble_predict  # noqa: E402

from datetime import datetime, timedelta

ARTIFACT_DIR = "models/artifacts"
BANKROLL = 7200.0
KELLY_FRACTION = 0.25
MAX_BET = 144.0  # 2% of bankroll, hard cap
MAX_TOTAL_EXPOSURE = 500.0
INJURY_WATCHLIST_PATH = "data/injury_watchlist.csv"

# Kalshi tournament-name -> our historical tourney_name, for the events where
# the naming differs (Kalshi says "ATP Indian Wells", our data says "Indian
# Wells Masters"). Anything not listed here is looked up directly after
# stripping the ATP/WTA prefix.
TOURNAMENT_ALIASES = {
    "Indian Wells": "Indian Wells Masters",
    "Miami": "Miami Masters",
    "Monte Carlo": "Monte Carlo Masters",
    "Rome": "Rome Masters",
    "Madrid": "Madrid Masters",
    "Canada": "Canada Masters",
    "Cincinnati": "Cincinnati Masters",
    "Shanghai": "Shanghai Masters",
    "Paris": "Paris Masters",
}

ROUND_PATTERNS = [
    ("Round Of 128", "R128"), ("Round Of 64", "R64"), ("Round Of 32", "R32"),
    ("Round Of 16", "R16"), ("Quarterfinals", "QF"), ("Quarterfinal", "QF"),
    ("Semifinals", "SF"), ("Semifinal", "SF"), ("Final", "F"),
    ("Round Robin", "RR"),
]


# ---------------------------------------------------------------------------
# Name / tournament resolution helpers
# ---------------------------------------------------------------------------

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def build_name_lookup(players):
    """Map normalized player name -> player_id, preferring the most recently active player on collisions."""
    lookup = {}
    for pid, state in players.items():
        if not state.name:
            continue
        key = normalize_name(state.name)
        if key not in lookup or (state.matches and players[lookup[key]].matches
                                  and state.matches[-1]["date"] > players[lookup[key]].matches[-1]["date"]):
            lookup[key] = pid
    return lookup


def match_player(name, name_lookup):
    key = normalize_name(name)
    if key in name_lookup:
        return name_lookup[key], "exact"
    close = difflib.get_close_matches(key, name_lookup.keys(), n=1, cutoff=0.82)
    if close:
        return name_lookup[close[0]], "fuzzy"
    return None, None


def resolve_tournament(competition_name, historical_tourney_names):
    """competition_name e.g. 'ATP Washington' -> ('Washington', surface, level, indoor) or None."""
    base = re.sub(r"^(ATP|WTA)\s+", "", competition_name).strip()
    candidate = TOURNAMENT_ALIASES.get(base, base)
    if candidate not in historical_tourney_names:
        close = difflib.get_close_matches(candidate, historical_tourney_names, n=1, cutoff=0.7)
        if not close:
            return None
        candidate = close[0]
    return candidate


def parse_round_and_competition(kalshi_title):
    """kalshi_title e.g. 'Will X win the X vs Y: Round Of 32 match?' -> (competition_guess, round_code)."""
    m = re.search(r": (.+) match\?$", kalshi_title)
    round_text = m.group(1) if m else ""
    for pattern, code in ROUND_PATTERNS:
        if round_text.strip().lower() == pattern.lower():
            return code
    return None


def parse_competition_from_rules(rules_primary):
    """Pull '2026 ATP Washington Round Of 32' style text out of the rules_primary sentence."""
    m = re.search(r"in the (\d{4}) (.+?) after a ball has been played", rules_primary or "")
    if not m:
        return None, None
    text = m.group(2)
    for pattern, code in sorted(ROUND_PATTERNS, key=lambda x: -len(x[0])):
        if text.endswith(pattern):
            return text[: -len(pattern)].strip(), code
    return text.strip(), None


# ---------------------------------------------------------------------------
# Feature building for one upcoming match
# ---------------------------------------------------------------------------

def build_live_feature_row(p1_state, p2_state, surface, tourney_level, tourney_name, indoor, current_date):
    p1_feats, p2_feats = {}, {}
    for state, feats in ((p1_state, p1_feats), (p2_state, p2_feats)):
        feats.update(surface_performance_features(state, surface, current_date, "x"))
        feats.update(recent_form_features(state, surface, current_date, "x"))
        feats.update(ranking_features(state, current_date, "x"))
        feats.update(tournament_context_features(state, tourney_name, tourney_level, indoor, current_date, "x"))
        feats.update(serve_return_features(state, surface, "x"))
        feats.update(physical_scheduling_features(state, np.nan, current_date, tourney_name, "x"))
        feats["x_elo_surf"] = state.elo_surface.get(surface, ELO_START)

    # head-to-head, symmetric
    h2h_key = tuple(sorted((p1_state.player_id, p2_state.player_id)))
    return p1_feats, p2_feats, h2h_key


def load_trained_ensemble():
    models = {
        "logistic_regression": joblib.load(f"{ARTIFACT_DIR}/logistic_regression.joblib"),
        "random_forest": joblib.load(f"{ARTIFACT_DIR}/random_forest.joblib"),
        "xgboost": joblib.load(f"{ARTIFACT_DIR}/xgboost.joblib"),
    }
    feature_names = pd.read_json(f"{ARTIFACT_DIR}/feature_names.json", typ="series").tolist()
    nn = TennisNet(input_dim=len(feature_names))
    nn.load_state_dict(torch.load(f"{ARTIFACT_DIR}/neural_net.pt"))
    nn.eval()
    models["neural_net"] = nn
    imputer = joblib.load(f"{ARTIFACT_DIR}/imputer.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    calibrator = joblib.load(f"{ARTIFACT_DIR}/isotonic_calibrator.joblib")
    import json
    with open(f"{ARTIFACT_DIR}/ensemble_weights.json") as f:
        weights = json.load(f)
    return models, {"imputer": imputer, "scaler": scaler}, calibrator, weights, feature_names


def predict_win_probability(models, preprocessing, calibrator, weights, X_row):
    X_imp = preprocessing["imputer"].transform(X_row)
    X_scaled = preprocessing["scaler"].transform(X_imp)
    preds = {
        "logistic_regression": models["logistic_regression"].predict_proba(X_scaled)[:, 1],
        "random_forest": models["random_forest"].predict_proba(X_row)[:, 1],
        "xgboost": models["xgboost"].predict_proba(X_row)[:, 1],
        "neural_net": nn_predict_proba(models["neural_net"], X_scaled),
    }
    raw_ensemble = ensemble_predict(preds, weights)
    return float(calibrator.predict(raw_ensemble)[0]), {k: float(v[0]) for k, v in preds.items()}


def load_injury_watchlist():
    if not os.path.exists(INJURY_WATCHLIST_PATH):
        return set()
    df = pd.read_csv(INJURY_WATCHLIST_PATH)
    return set(normalize_name(n) for n in df.get("player_name", []))


def kelly_bet_size(model_prob, price_paid):
    """25% Kelly on a binary contract costing price_paid dollars per $1 payout, capped at MAX_BET."""
    if price_paid <= 0 or price_paid >= 1:
        return 0.0
    b = (1 - price_paid) / price_paid  # net odds
    kelly_full = (model_prob * (b + 1) - 1) / b
    kelly_full = max(kelly_full, 0.0)
    bet = KELLY_FRACTION * kelly_full * BANKROLL
    return round(min(bet, MAX_BET), 2)


def classify_opportunity(edge_pp, model_prob, volume, start_hour_ct):
    if start_hour_ct is not None and start_hour_ct < 8 and edge_pp >= 5:
        return "EARLY_MORNING_PRIORITY"
    if edge_pp >= 8 and model_prob >= 0.60 and volume >= 1000:
        return "HIGH_CONVICTION"
    if edge_pp >= 5 and model_prob >= 0.55 and volume >= 500:
        return "STANDARD"
    return None


def apply_exposure_cap(recommendations, max_total=MAX_TOTAL_EXPOSURE):
    """
    Actually enforce the $500 total-exposure cap (not just warn about it):
    process highest-edge opportunities first, shrink or drop whatever would
    push the running total over the cap.
    """
    ordered = sorted(recommendations, key=lambda x: -x["edge_pp"])
    total = 0.0
    for r in ordered:
        remaining = round(max_total - total, 2)
        if remaining <= 0:
            r["original_bet_size"] = r["bet_size"]
            r["bet_size"] = 0.0
            r["exposure_note"] = "skipped - $500 total exposure cap reached"
        elif r["bet_size"] > remaining:
            r["original_bet_size"] = r["bet_size"]
            r["bet_size"] = remaining
            r["exposure_note"] = "reduced to fit $500 total exposure cap"
        else:
            r["exposure_note"] = None
        total += r["bet_size"]
    return ordered


def generate_recommendations(verbose=True):
    """
    Runs the full Step 4 pipeline and returns (recommendations, skipped,
    tomorrow_matches, all_analysis) without printing - the reusable entry
    point for Telegram alerts (Step 5) and the trade tracker (Step 6).
    all_analysis has one entry per Kalshi match (analyzed or not, flagged
    or not) with model probability/price/edge for both players - used by
    the 9pm full-board preview message.
    """
    if verbose:
        print("Loading historical ATP matches and rebuilding current player state...")
    matches, _ = fetch_all_historical_matches(verbose=False)
    _features_unused, players, h2h = build_training_features(matches, verbose=False)
    historical_tourney_names = set(matches["tourney_name"].unique())
    name_lookup = build_name_lookup(players)
    injury_watchlist = load_injury_watchlist()
    if verbose:
        print(f"  {len(players):,} known players, {len(historical_tourney_names)} known tournament names")
        print("Loading trained ensemble...")
    models, preprocessing, calibrator, weights, feature_names = load_trained_ensemble()

    if verbose:
        print("Fetching live Kalshi tennis markets...")
    kalshi_matches = fetch_all_tennis_matches()
    atp_matches = [m for m in kalshi_matches if m["tour"] == "ATP"]

    today_ct = datetime.now(CENTRAL).date()
    tomorrow_ct = today_ct + timedelta(days=1)
    tomorrow_matches = matches_for_date(atp_matches, tomorrow_ct)
    if verbose:
        print(f"  {len(tomorrow_matches)} ATP matches on Kalshi tomorrow ({tomorrow_ct})")
        print("  (WTA markets are live on Kalshi but skipped here - no trained WTA model yet)")

    recommendations = []
    skipped = []
    all_analysis = []  # one entry per Kalshi match, whether or not it was analyzable/flagged

    for m in tomorrow_matches:
        p1_name, p2_name = m["player_1"], m["player_2"]
        p1_id, p1_match_type = match_player(p1_name, name_lookup)
        p2_id, p2_match_type = match_player(p2_name, name_lookup)
        if p1_id is None or p2_id is None:
            reason = "could not match player name(s) to historical data"
            skipped.append((m["matchup"], reason))
            all_analysis.append({
                "matchup": m["matchup"], "start_time_ct": m["start_time_ct"],
                "start_hour_ct": m["start_hour_ct"], "analyzed": False, "reason": reason,
            })
            continue

        competition, round_code = parse_competition_from_rules(m.get("rules_primary"))
        if competition is None:
            reason = "could not parse tournament name from Kalshi market"
            skipped.append((m["matchup"], reason))
            all_analysis.append({
                "matchup": m["matchup"], "start_time_ct": m["start_time_ct"],
                "start_hour_ct": m["start_hour_ct"], "analyzed": False, "reason": reason,
            })
            continue
        tourney_name = resolve_tournament(competition, historical_tourney_names)
        if tourney_name is None:
            reason = f"unknown tournament '{competition}' - no surface/level data"
            skipped.append((m["matchup"], reason))
            all_analysis.append({
                "matchup": m["matchup"], "start_time_ct": m["start_time_ct"],
                "start_hour_ct": m["start_hour_ct"], "analyzed": False, "reason": reason,
            })
            continue

        recent_rows = matches[matches["tourney_name"] == tourney_name].sort_values("tourney_date")
        last_row = recent_rows.iloc[-1]
        surface, tourney_level, indoor = last_row["surface"], last_row["tourney_level"], last_row["indoor"]

        p1_state, p2_state = players[p1_id], players[p2_id]
        current_date = pd.Timestamp(m["start_datetime_ct"].date())

        p1_feats, p2_feats, h2h_key = build_live_feature_row(
            p1_state, p2_state, surface, tourney_level, tourney_name, indoor, current_date
        )
        h2h_list = h2h.get(h2h_key, [])
        p1_h2h_view = [{**e, "won": e["winner_id"] == p1_id} for e in h2h_list]
        p2_h2h_view = [{**e, "won": e["winner_id"] == p2_id} for e in h2h_list]
        p1_feats.update(h2h_features(p1_h2h_view, surface, tourney_level, indoor, "x"))
        p2_feats.update(h2h_features(p2_h2h_view, surface, tourney_level, indoor, "x"))

        rank_diff = (p1_feats["x_rank"] - p2_feats["x_rank"]
                     if pd.notna(p1_feats["x_rank"]) and pd.notna(p2_feats["x_rank"]) else np.nan)

        row = {
            "match_date": current_date, "tourney_id": tourney_name, "tourney_name": tourney_name,
            "surface": surface, "tourney_level": tourney_level, "round": round_code, "indoor": indoor,
            "player_1_id": p1_id, "player_1_name": p1_name, "player_2_id": p2_id, "player_2_name": p2_name,
            "rank_diff_p1_minus_p2": rank_diff,
        }
        for k, v in p1_feats.items():
            row[f"p1_{k}"] = v
        for k, v in p2_feats.items():
            row[f"p2_{k}"] = v

        X_row, _, _ = encode_features(pd.DataFrame([row]))
        X_row = align_to_training_columns(X_row, feature_names)
        model_prob_p1, per_model = predict_win_probability(models, preprocessing, calibrator, weights, X_row)

        # Full-board entry for the 9pm preview message - every match that got
        # this far has a real model probability, regardless of whether it
        # ends up qualifying for a bet recommendation below.
        p1_price = float(m["player_1_yes_ask"]) if m["player_1_yes_ask"] is not None else None
        p2_price = float(m["player_2_yes_ask"]) if m["player_2_yes_ask"] is not None else None
        all_analysis.append({
            "matchup": m["matchup"], "tournament": tourney_name, "round": round_code,
            "start_time_ct": m["start_time_ct"], "start_hour_ct": m["start_hour_ct"],
            "analyzed": True,
            "player_1": p1_name, "player_1_model_prob": round(model_prob_p1, 4),
            "player_1_kalshi_price": p1_price,
            "player_1_edge_pp": round((model_prob_p1 - p1_price) * 100, 2) if p1_price is not None else None,
            "player_2": p2_name, "player_2_model_prob": round(1 - model_prob_p1, 4),
            "player_2_kalshi_price": p2_price,
            "player_2_edge_pp": round((1 - model_prob_p1 - p2_price) * 100, 2) if p2_price is not None else None,
            "volume": float(m["player_1_volume"] or 0),
        })

        # ranking-difference / thin-h2h skip rule
        rank_diff_abs = abs(rank_diff) if pd.notna(rank_diff) else 0
        h2h_meetings = p1_feats.get("x_h2h_meetings", 0)
        if h2h_meetings < 3 and rank_diff_abs < 50:
            skipped.append((m["matchup"], "thin H2H sample (<3) and ranking gap <50 - too uncertain"))
            continue
        if rank_diff_abs > 500:
            skipped.append((m["matchup"], f"ranking gap {rank_diff_abs:.0f} exceeds 500 - flagged, not auto-skipped"))
            # per spec this is a FLAG not a hard skip; fall through

        if normalize_name(p1_name) in injury_watchlist or normalize_name(p2_name) in injury_watchlist:
            skipped.append((m["matchup"], "player on injury/retirement watchlist"))
            continue

        # Evaluate both sides, but per the "never bet both players in the same
        # match" rule, only the single highest-edge qualifying side survives.
        candidates = []
        for side, player_name, model_p, yes_ask in (
            ("player_1", p1_name, model_prob_p1, m["player_1_yes_ask"]),
            ("player_2", p2_name, 1 - model_prob_p1, m["player_2_yes_ask"]),
        ):
            if yes_ask is None:
                continue
            price = float(yes_ask)
            implied_prob = price
            edge_pp = (model_p - implied_prob) * 100
            volume = float(m["player_1_volume"] or 0)
            tier = classify_opportunity(edge_pp, model_p, volume, m["start_hour_ct"])
            if tier is None:
                continue
            bet_size = kelly_bet_size(model_p, price)
            expected_profit = bet_size * (1 - price) / price * model_p - bet_size * (1 - model_p)
            candidates.append({
                "matchup": m["matchup"], "tour": m["tour"], "start_time_ct": m["start_time_ct"],
                "event_ticker": m["event_ticker"],
                "start_hour_ct": m["start_hour_ct"], "tournament": tourney_name, "round": round_code,
                "surface": surface,
                "recommended_side": player_name, "opponent": p2_name if side == "player_1" else p1_name,
                "model_prob": round(model_p, 4), "kalshi_price": price, "implied_prob": round(implied_prob, 4),
                "edge_pp": round(edge_pp, 2), "volume": volume, "tier": tier, "bet_size": bet_size,
                "expected_profit": round(expected_profit, 2), "breakeven_win_rate": round(price, 4),
                "per_model_probs": per_model,
            })
        if candidates:
            recommendations.append(max(candidates, key=lambda c: c["edge_pp"]))

    recommendations = apply_exposure_cap(recommendations)
    return recommendations, skipped, tomorrow_matches, all_analysis


def print_report(recommendations, skipped, tomorrow_matches):
    print(f"\n{'='*70}\nSTEP 4 - EDGE IDENTIFICATION RESULTS\n{'='*70}")
    print(f"\nMatches evaluated: {len(tomorrow_matches)}  |  Recommendations: {len(recommendations)}  |  Skipped: {len(skipped)}")

    if skipped:
        print("\nSkipped matches:")
        for matchup, reason in skipped:
            print(f"  {matchup}: {reason}")

    total_exposure = sum(r["bet_size"] for r in recommendations)
    for r in recommendations:
        print(f"\n[{r['tier']}] {r['matchup']} ({r['tournament']}, {r['round']}) - {r['start_time_ct']}")
        print(f"  Recommended: BUY YES on {r['recommended_side']}")
        print(f"  Model prob: {r['model_prob']:.1%}  |  Kalshi price: {r['kalshi_price']:.2f} "
              f"(implied {r['implied_prob']:.1%})  |  Edge: {r['edge_pp']:+.1f}pp")
        print(f"  Bet size: ${r['bet_size']:.2f}  |  Expected profit: ${r['expected_profit']:.2f}  "
              f"|  Breakeven win rate: {r['breakeven_win_rate']:.1%}  |  Volume: ${r['volume']:.0f}")
        if r.get("exposure_note"):
            print(f"  *** {r['exposure_note']} (was ${r['original_bet_size']:.2f}) ***")

    print(f"\nTotal recommended exposure: ${total_exposure:.2f} (cap: ${MAX_TOTAL_EXPOSURE:.0f})")


def main():
    recommendations, skipped, tomorrow_matches, all_analysis = generate_recommendations()
    print_report(recommendations, skipped, tomorrow_matches)


if __name__ == "__main__":
    main()
