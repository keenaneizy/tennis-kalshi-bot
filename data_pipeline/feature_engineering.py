"""
STEP 2 - Feature engineering.

Turns the raw match log (one row per completed match, winner vs loser) into
one training row per match with pre-match features for two players plus a
target (did "player_1" win). Everything is computed in strict chronological
order, using only information that existed BEFORE that match was played -
this is what prevents data leakage. Elo, rolling form, surface stats, and
head-to-head records are all built up incrementally as we walk through
history, the same way they'd be maintained live in production.

Retirement/walkover matches are dropped entirely (per the Step 1 spec) -
they're not included as training targets, and they're also excluded from
the rolling state so a walkover doesn't pollute serve-stat averages.

Run with: python3 data_pipeline/feature_engineering.py
"""

import math
import random
import re

import numpy as np
import pandas as pd

from tournament_metadata import get_tournament_metadata, HIGH_ALTITUDE_THRESHOLD_M

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ELO_K = 32
ELO_START = 1500.0

SET_RE = re.compile(r"(\d+)-(\d+)(?:\((\d+)\))?")


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_score(score, best_of):
    """
    Parse a Sackmann-style score string, e.g. "6-3 4-6 7-6(4)", into a list
    of sets from the MATCH WINNER's point of view: [{w_games, l_games,
    is_tiebreak}, ...]. Returns [] for unparseable/incomplete scores.
    """
    if not isinstance(score, str):
        return []
    sets = []
    for token in score.split():
        m = SET_RE.match(token)
        if not m:
            continue
        w_games, l_games = int(m.group(1)), int(m.group(2))
        is_tiebreak = m.group(3) is not None or {w_games, l_games} == {6, 7}
        sets.append({"w_games": w_games, "l_games": l_games, "is_tiebreak": is_tiebreak})
    return sets


def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return np.nan
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def safe_pct(numerator, denominator):
    if not denominator:
        return np.nan
    return numerator / denominator


def to_num(v):
    """pd.NA (from the live-results supplement, which has no ranking/age data) -> a real np.nan."""
    return np.nan if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# Per-player chronological state
# ---------------------------------------------------------------------------

class PlayerState:
    """Everything we know about a player, updated match-by-match in order."""

    def __init__(self, player_id):
        self.player_id = player_id
        self.matches = []  # chronological list of match-perspective dicts
        self.elo = ELO_START
        self.elo_surface = {s: ELO_START for s in SURFACES}
        self.rank_history = []  # (date, rank)
        self.peak_rank = None
        self.name = None
        self.ioc = None
        self.hand = None
        self.ht = None

    def recent(self, n=None, surface=None, since_date=None, tier=None, indoor=None):
        """Filter this player's match history, most-recent-first by default."""
        ms = self.matches
        if surface is not None:
            ms = [m for m in ms if m["surface"] == surface]
        if tier is not None:
            ms = [m for m in ms if m["tourney_level"] == tier]
        if indoor is not None:
            ms = [m for m in ms if m["indoor"] == indoor]
        if since_date is not None:
            ms = [m for m in ms if m["date"] >= since_date]
        if n is not None:
            ms = ms[-n:]
        return ms

    def update_after_match(self, match_entry, opponent_elo, opponent_elo_surface):
        """Apply Elo updates and append this match to history. Call AFTER features are computed."""
        actual = 1.0 if match_entry["won"] else 0.0
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400.0))
        self.elo = self.elo + ELO_K * (actual - expected)

        surface = match_entry["surface"]
        if surface in self.elo_surface:
            exp_surface = 1.0 / (1.0 + 10 ** ((opponent_elo_surface - self.elo_surface[surface]) / 400.0))
            self.elo_surface[surface] = self.elo_surface[surface] + ELO_K * (actual - exp_surface)

        match_rank = match_entry["rank"]
        match_rank = np.nan if pd.isna(match_rank) else float(match_rank)
        self.rank_history.append((match_entry["date"], match_rank))
        if pd.notna(match_rank) and (self.peak_rank is None or match_rank < self.peak_rank):
            self.peak_rank = match_rank

        self.name = match_entry["name"]
        self.ioc = match_entry["ioc"]
        self.hand = match_entry["hand"]
        self.ht = match_entry["ht"]

        self.matches.append(match_entry)

    def rank_as_of(self):
        return self.rank_history[-1][1] if self.rank_history else np.nan

    def rank_days_ago(self, current_date, days):
        """Most recent known rank at least `days` before current_date."""
        cutoff = current_date - pd.Timedelta(days=days)
        candidates = [r for d, r in self.rank_history if d <= cutoff]
        return candidates[-1] if candidates else np.nan

    def last_tournament_coords(self):
        for m in reversed(self.matches):
            if m["tourney_lat"] is not None:
                return m["tourney_lat"], m["tourney_lon"]
        return None, None


# ---------------------------------------------------------------------------
# Feature computation for one player, as of (but not including) a match
# ---------------------------------------------------------------------------

def _win_pct(ms):
    if not ms:
        return np.nan
    return sum(1 for m in ms if m["won"]) / len(ms)


def surface_performance_features(state, surface, current_date, prefix):
    feats = {}
    since_12m = current_date - pd.Timedelta(days=365)
    since_24m = current_date - pd.Timedelta(days=730)

    feats[f"{prefix}_surf_win_pct_12m"] = _win_pct(state.recent(surface=surface, since_date=since_12m))
    feats[f"{prefix}_surf_win_pct_24m"] = _win_pct(state.recent(surface=surface, since_date=since_24m))

    # NaN-safe sums: matches from the live-results supplement (no serve
    # stats) simply don't contribute to these aggregates instead of being
    # miscounted as "0 aces" etc.
    last20 = state.recent(n=20, surface=surface)
    if last20:
        feats[f"{prefix}_surf_ace_rate_l20"] = safe_pct(
            np.nansum([m["own"]["ace"] for m in last20]), np.nansum([m["own"]["svpt"] for m in last20])
        )
        feats[f"{prefix}_surf_1stIn_pct_l20"] = safe_pct(
            np.nansum([m["own"]["firstIn"] for m in last20]), np.nansum([m["own"]["svpt"] for m in last20])
        )
        bp_conv_opp = np.nansum([m["opp"]["bpFaced"] - m["opp"]["bpSaved"] for m in last20])
        bp_opp_total = np.nansum([m["opp"]["bpFaced"] for m in last20])
        feats[f"{prefix}_surf_bp_converted_l20"] = safe_pct(bp_conv_opp, bp_opp_total)
    else:
        feats[f"{prefix}_surf_ace_rate_l20"] = np.nan
        feats[f"{prefix}_surf_1stIn_pct_l20"] = np.nan
        feats[f"{prefix}_surf_bp_converted_l20"] = np.nan

    # trend: win% in most recent 10 on surface vs the 10 before that
    surf_all = state.recent(surface=surface)
    recent10 = surf_all[-10:]
    prior10 = surf_all[-20:-10]
    wp_recent = _win_pct(recent10)
    wp_prior = _win_pct(prior10)
    if not np.isnan(wp_recent) and not np.isnan(wp_prior):
        trend = wp_recent - wp_prior
    else:
        trend = np.nan
    feats[f"{prefix}_surf_trend"] = trend
    feats[f"{prefix}_surf_trend_label"] = (
        "improving" if trend > 0.1 else "declining" if trend < -0.1 else "stable"
        if not np.isnan(trend) else "unknown"
    )
    return feats


def recent_form_features(state, surface, current_date, prefix):
    feats = {}
    last5 = state.recent(n=5)
    last10 = state.recent(n=10)
    last5_surf = state.recent(n=5, surface=surface)
    last20 = state.recent(n=20)

    feats[f"{prefix}_win_pct_l5"] = _win_pct(last5)
    feats[f"{prefix}_win_pct_l10"] = _win_pct(last10)
    feats[f"{prefix}_win_pct_l5_surf"] = _win_pct(last5_surf)

    if last10:
        feats[f"{prefix}_sets_won_pct_l10"] = safe_pct(
            sum(m["sets_won"] for m in last10), sum(m["sets_played"] for m in last10)
        )
        feats[f"{prefix}_games_won_pct_l10"] = safe_pct(
            sum(m["games_won"] for m in last10), sum(m["games_played"] for m in last10)
        )
    else:
        feats[f"{prefix}_sets_won_pct_l10"] = np.nan
        feats[f"{prefix}_games_won_pct_l10"] = np.nan

    tb_total = sum(m["tiebreaks_played"] for m in last20)
    feats[f"{prefix}_tiebreak_win_pct_l20"] = safe_pct(
        sum(m["tiebreaks_won"] for m in last20), tb_total
    )

    decided = [m for m in last20 if m["decided_set_reached"]]
    feats[f"{prefix}_decided_set_win_pct_l20"] = _win_pct(decided)

    since_14d = current_date - pd.Timedelta(days=14)
    since_7d = current_date - pd.Timedelta(days=7)
    matches_l14 = state.recent(since_date=since_14d)
    matches_l7 = state.recent(since_date=since_7d)
    feats[f"{prefix}_matches_l14d"] = len(matches_l14)
    feats[f"{prefix}_sets_played_l7d"] = sum(m["sets_played"] for m in matches_l7)
    feats[f"{prefix}_days_since_last_match"] = (
        (current_date - state.matches[-1]["date"]).days if state.matches else np.nan
    )

    last10_wins = [m for m in last10 if m["won"]]
    straight_sets_wins = sum(1 for m in last10_wins if m["sets_played"] == m["min_sets_to_win"])
    feats[f"{prefix}_straight_set_win_pct_l10"] = safe_pct(straight_sets_wins, len(last10_wins))

    return feats


def h2h_features(h2h_matches, surface, tier, indoor, prefix):
    feats = {}
    feats[f"{prefix}_h2h_meetings"] = len(h2h_matches)
    feats[f"{prefix}_h2h_win_pct"] = _win_pct(h2h_matches)
    feats[f"{prefix}_h2h_win_pct_surface"] = _win_pct([m for m in h2h_matches if m["surface"] == surface])
    feats[f"{prefix}_h2h_win_pct_tier"] = _win_pct([m for m in h2h_matches if m["tourney_level"] == tier])
    feats[f"{prefix}_h2h_win_pct_indoor_outdoor"] = _win_pct([m for m in h2h_matches if m["indoor"] == indoor])
    last3 = h2h_matches[-3:]
    feats[f"{prefix}_h2h_last3_wins"] = sum(1 for m in last3 if m["won"])
    feats[f"{prefix}_h2h_avg_sets"] = (
        sum(m["sets_played"] for m in h2h_matches) / len(h2h_matches) if h2h_matches else np.nan
    )
    return feats


def ranking_features(state, current_date, prefix):
    feats = {}
    feats[f"{prefix}_rank"] = state.rank_as_of()
    feats[f"{prefix}_rank_points"] = state.matches[-1]["rank_points"] if state.matches else np.nan
    feats[f"{prefix}_peak_rank"] = state.peak_rank if state.peak_rank is not None else np.nan
    rank_90d_ago = state.rank_days_ago(current_date, 90)
    current_rank = state.rank_as_of()
    if pd.notna(rank_90d_ago) and pd.notna(current_rank):
        feats[f"{prefix}_rank_trend_90d"] = rank_90d_ago - current_rank  # positive = improving (rank number dropped)
    else:
        feats[f"{prefix}_rank_trend_90d"] = np.nan
    feats[f"{prefix}_elo"] = state.elo
    return feats


def tournament_context_features(state, tourney_name, tourney_level, indoor, current_date, prefix):
    feats = {}
    since_5y = current_date - pd.Timedelta(days=5 * 365)
    at_tourney = [m for m in state.matches if m["tourney_name"] == tourney_name and m["date"] >= since_5y]
    feats[f"{prefix}_tourney_history_win_pct_5y"] = _win_pct(at_tourney)
    at_tier = [m for m in state.matches if m["tourney_level"] == tourney_level]
    feats[f"{prefix}_tier_win_pct"] = _win_pct(at_tier)

    indoor_matches = [m for m in state.matches if m["indoor"] is True]
    outdoor_matches = [m for m in state.matches if m["indoor"] is False]
    feats[f"{prefix}_indoor_win_pct"] = _win_pct(indoor_matches)
    feats[f"{prefix}_outdoor_win_pct"] = _win_pct(outdoor_matches)

    country, lat, lon, altitude = get_tournament_metadata(tourney_name)
    feats[f"{prefix}_home_country_advantage"] = (
        (state.ioc == country) if country is not None and state.ioc is not None else False
    )
    feats[f"{prefix}_venue_altitude_m"] = altitude if altitude is not None else np.nan
    feats[f"{prefix}_high_altitude"] = bool(altitude is not None and altitude >= HIGH_ALTITUDE_THRESHOLD_M)

    return feats


def serve_return_features(state, surface, prefix):
    feats = {}
    last20 = state.recent(n=20, surface=surface)
    if not last20:
        for name in [
            "1st_serve_pct", "1st_serve_won_pct", "2nd_serve_won_pct", "aces_per_match",
            "dfs_per_match", "return_pts_won_1st_pct", "return_pts_won_2nd_pct",
            "bp_converted_pct", "bp_saved_pct",
        ]:
            feats[f"{prefix}_{name}_l20surf"] = np.nan
        return feats

    # NaN-safe: matches from the live-results supplement have no serve
    # stats and simply don't contribute here (rather than counting as 0).
    svpt = np.nansum([m["own"]["svpt"] for m in last20])
    firstIn = np.nansum([m["own"]["firstIn"] for m in last20])
    firstWon = np.nansum([m["own"]["firstWon"] for m in last20])
    secondWon = np.nansum([m["own"]["secondWon"] for m in last20])
    ace = np.nansum([m["own"]["ace"] for m in last20])
    df = np.nansum([m["own"]["df"] for m in last20])
    bpSaved = np.nansum([m["own"]["bpSaved"] for m in last20])
    bpFaced = np.nansum([m["own"]["bpFaced"] for m in last20])
    matches_with_stats = sum(1 for m in last20 if pd.notna(m["own"]["svpt"]))

    opp_svpt = np.nansum([m["opp"]["svpt"] for m in last20])
    opp_firstIn = np.nansum([m["opp"]["firstIn"] for m in last20])
    opp_firstWon = np.nansum([m["opp"]["firstWon"] for m in last20])
    opp_secondWon = np.nansum([m["opp"]["secondWon"] for m in last20])
    opp_bpFaced = np.nansum([m["opp"]["bpFaced"] for m in last20])
    opp_bpSaved = np.nansum([m["opp"]["bpSaved"] for m in last20])

    feats[f"{prefix}_1st_serve_pct_l20surf"] = safe_pct(firstIn, svpt)
    feats[f"{prefix}_1st_serve_won_pct_l20surf"] = safe_pct(firstWon, firstIn)
    feats[f"{prefix}_2nd_serve_won_pct_l20surf"] = safe_pct(secondWon, svpt - firstIn)
    feats[f"{prefix}_aces_per_match_l20surf"] = safe_pct(ace, matches_with_stats)
    feats[f"{prefix}_dfs_per_match_l20surf"] = safe_pct(df, matches_with_stats)
    feats[f"{prefix}_bp_saved_pct_l20surf"] = safe_pct(bpSaved, bpFaced)

    opp_second_serve_pts = opp_svpt - opp_firstIn
    feats[f"{prefix}_return_pts_won_1st_pct_l20surf"] = safe_pct(opp_firstIn - opp_firstWon, opp_firstIn)
    feats[f"{prefix}_return_pts_won_2nd_pct_l20surf"] = safe_pct(
        opp_second_serve_pts - opp_secondWon, opp_second_serve_pts
    )
    feats[f"{prefix}_bp_converted_pct_l20surf"] = safe_pct(opp_bpFaced - opp_bpSaved, opp_bpFaced)

    return feats


def physical_scheduling_features(state, age, current_date, tourney_name, prefix):
    feats = {}
    feats[f"{prefix}_age"] = age
    year_start = current_date.replace(month=1, day=1)
    feats[f"{prefix}_matches_this_season"] = len(state.recent(since_date=year_start))

    last_lat, last_lon = state.last_tournament_coords()
    _, cur_lat, cur_lon, _ = get_tournament_metadata(tourney_name)
    feats[f"{prefix}_travel_distance_km"] = haversine_km(last_lat, last_lon, cur_lat, cur_lon)

    return feats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_training_features(matches_df, verbose=True, seed=42):
    """
    Walk through all matches in chronological order, computing pre-match
    features for both players, then updating each player's rolling state.
    Returns a DataFrame with one row per match.
    """
    rng = random.Random(seed)

    df = matches_df.copy()
    df = df[~df["retirement_or_walkover"]].copy()
    df["tourney_date_parsed"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(["tourney_date_parsed", "match_num"]).reset_index(drop=True)

    players = {}
    h2h = {}

    def get_player(pid):
        if pid not in players:
            players[pid] = PlayerState(pid)
        return players[pid]

    def get_h2h_key(id1, id2):
        return tuple(sorted((id1, id2)))

    rows = []
    total = len(df)
    for i, row in df.iterrows():
        if verbose and i % 5000 == 0:
            print(f"  processed {i:,}/{total:,} matches...")

        current_date = row["tourney_date_parsed"]
        surface = row["surface"]
        tourney_level = row["tourney_level"]
        tourney_name = row["tourney_name"]
        indoor = row["indoor"] if pd.notna(row["indoor"]) else None
        best_of = int(row["best_of"]) if pd.notna(row["best_of"]) else 3
        min_sets_to_win = (best_of // 2) + 1

        sets = parse_score(row["score"], best_of)
        if not sets:
            continue  # can't build reliable sets/games features without a parseable score

        if pd.isna(row["winner_id"]) or pd.isna(row["loser_id"]):
            continue  # can't track player state without a stable identity

        sets_played = len(sets)
        w_sets_won = sum(1 for s in sets if s["w_games"] > s["l_games"])
        l_sets_won = sets_played - w_sets_won
        w_games = sum(s["w_games"] for s in sets)
        l_games = sum(s["l_games"] for s in sets)
        tb_total = sum(1 for s in sets if s["is_tiebreak"])
        tb_won_by_winner = sum(1 for s in sets if s["is_tiebreak"] and s["w_games"] > s["l_games"])
        decided_set_reached = sets_played == best_of

        country_lookup_name = tourney_name
        _, tourney_lat, tourney_lon, _ = get_tournament_metadata(country_lookup_name)

        winner_id, loser_id = row["winner_id"], row["loser_id"]
        w_state, l_state = get_player(winner_id), get_player(loser_id)

        w_stats = {
            "ace": row["w_ace"], "df": row["w_df"], "svpt": row["w_svpt"],
            "firstIn": row["w_1stIn"], "firstWon": row["w_1stWon"], "secondWon": row["w_2ndWon"],
            "bpSaved": row["w_bpSaved"], "bpFaced": row["w_bpFaced"],
        }
        l_stats = {
            "ace": row["l_ace"], "df": row["l_df"], "svpt": row["l_svpt"],
            "firstIn": row["l_1stIn"], "firstWon": row["l_1stWon"], "secondWon": row["l_2ndWon"],
            "bpSaved": row["l_bpSaved"], "bpFaced": row["l_bpFaced"],
        }
        # Keep missing serve stats as real NaN (not 0) - matches from the
        # live-results supplement have none, and coercing to 0 would make a
        # "no data" match look like a match with zero aces / zero first
        # serves in, silently dragging down rolling serve-stat averages.
        # The aggregation functions above use np.nansum specifically so
        # these matches just don't contribute rather than count as zeros.
        w_stats = {k: (np.nan if pd.isna(v) else v) for k, v in w_stats.items()}
        l_stats = {k: (np.nan if pd.isna(v) else v) for k, v in l_stats.items()}

        # ---- compute features for BOTH players using state as of BEFORE this match ----
        winner_feats = {}
        winner_feats.update(surface_performance_features(w_state, surface, current_date, "x"))
        winner_feats.update(recent_form_features(w_state, surface, current_date, "x"))
        winner_feats.update(ranking_features(w_state, current_date, "x"))
        winner_feats.update(tournament_context_features(w_state, tourney_name, tourney_level, indoor, current_date, "x"))
        winner_feats.update(serve_return_features(w_state, surface, "x"))
        winner_feats.update(physical_scheduling_features(w_state, to_num(row["winner_age"]), current_date, tourney_name, "x"))

        loser_feats = {}
        loser_feats.update(surface_performance_features(l_state, surface, current_date, "x"))
        loser_feats.update(recent_form_features(l_state, surface, current_date, "x"))
        loser_feats.update(ranking_features(l_state, current_date, "x"))
        loser_feats.update(tournament_context_features(l_state, tourney_name, tourney_level, indoor, current_date, "x"))
        loser_feats.update(serve_return_features(l_state, surface, "x"))
        loser_feats.update(physical_scheduling_features(l_state, to_num(row["loser_age"]), current_date, tourney_name, "x"))

        h2h_key = get_h2h_key(winner_id, loser_id)
        h2h_list = h2h.get(h2h_key, [])
        # h2h_list entries record who actually won each past meeting (winner_id);
        # convert to a per-player "won" view at query time for whoever we're computing for.
        winner_h2h_view = [{**m, "won": m["winner_id"] == winner_id} for m in h2h_list]
        loser_h2h_view = [{**m, "won": m["winner_id"] == loser_id} for m in h2h_list]
        winner_feats.update(h2h_features(winner_h2h_view, surface, tourney_level, indoor, "x"))
        loser_feats.update(h2h_features(loser_h2h_view, surface, tourney_level, indoor, "x"))

        winner_feats["x_elo_surf"] = w_state.elo_surface.get(surface, ELO_START)
        loser_feats["x_elo_surf"] = l_state.elo_surface.get(surface, ELO_START)

        # randomize which side is "player_1" so the model can't cheat off row order
        if rng.random() < 0.5:
            p1_feats, p2_feats, target = winner_feats, loser_feats, 1
            p1_id, p2_id, p1_name, p2_name = winner_id, loser_id, row["winner_name"], row["loser_name"]
        else:
            p1_feats, p2_feats, target = loser_feats, winner_feats, 0
            p1_id, p2_id, p1_name, p2_name = loser_id, winner_id, row["loser_name"], row["winner_name"]

        rank_diff = p1_feats["x_rank"] - p2_feats["x_rank"] if pd.notna(p1_feats["x_rank"]) and pd.notna(p2_feats["x_rank"]) else np.nan

        out_row = {
            "match_date": current_date, "tourney_id": row["tourney_id"], "tourney_name": tourney_name,
            "surface": surface, "tourney_level": tourney_level, "round": row["round"], "indoor": indoor,
            "player_1_id": p1_id, "player_1_name": p1_name,
            "player_2_id": p2_id, "player_2_name": p2_name,
            "player_1_won": target,
            "rank_diff_p1_minus_p2": rank_diff,
        }
        for k, v in p1_feats.items():
            out_row[f"p1_{k}"] = v
        for k, v in p2_feats.items():
            out_row[f"p2_{k}"] = v
        rows.append(out_row)

        # ---- now update state with the actual result (AFTER features were computed) ----
        w_entry = {
            "date": current_date, "surface": surface, "tourney_level": tourney_level,
            "tourney_name": tourney_name, "tourney_id": row["tourney_id"], "indoor": indoor,
            "round": row["round"], "won": True, "opponent_id": loser_id,
            "rank": to_num(row["winner_rank"]), "rank_points": to_num(row["winner_rank_points"]),
            "own": w_stats, "opp": l_stats,
            "sets_won": w_sets_won, "sets_played": sets_played, "games_won": w_games, "games_played": w_games + l_games,
            "tiebreaks_played": tb_total, "tiebreaks_won": tb_won_by_winner,
            "decided_set_reached": decided_set_reached, "min_sets_to_win": min_sets_to_win,
            "name": row["winner_name"], "ioc": row["winner_ioc"], "hand": row["winner_hand"], "ht": row["winner_ht"],
            "tourney_lat": tourney_lat, "tourney_lon": tourney_lon,
        }
        l_entry = {
            "date": current_date, "surface": surface, "tourney_level": tourney_level,
            "tourney_name": tourney_name, "tourney_id": row["tourney_id"], "indoor": indoor,
            "round": row["round"], "won": False, "opponent_id": winner_id,
            "rank": to_num(row["loser_rank"]), "rank_points": to_num(row["loser_rank_points"]),
            "own": l_stats, "opp": w_stats,
            "sets_won": l_sets_won, "sets_played": sets_played, "games_won": l_games, "games_played": w_games + l_games,
            "tiebreaks_played": tb_total, "tiebreaks_won": tb_total - tb_won_by_winner,
            "decided_set_reached": decided_set_reached, "min_sets_to_win": min_sets_to_win,
            "name": row["loser_name"], "ioc": row["loser_ioc"], "hand": row["loser_hand"], "ht": row["loser_ht"],
            "tourney_lat": tourney_lat, "tourney_lon": tourney_lon,
        }

        w_state.update_after_match(w_entry, l_state.elo, l_state.elo_surface.get(surface, ELO_START))
        l_state.update_after_match(l_entry, w_state.elo, w_state.elo_surface.get(surface, ELO_START))

        h2h_list.append({
            "date": current_date, "surface": surface, "tourney_level": tourney_level,
            "indoor": indoor, "sets_played": sets_played, "winner_id": winner_id,
        })
        h2h[h2h_key] = h2h_list

    result = pd.DataFrame(rows)
    return result, players, h2h


if __name__ == "__main__":
    print("Loading historical matches...")
    matches = pd.read_csv("data/raw/historical_matches.csv", low_memory=False)
    print(f"  {len(matches):,} matches loaded")

    print("Building training features (chronological, no leakage)...")
    features, _players, _h2h = build_training_features(matches)
    print(f"\nBuilt {len(features):,} training rows with {features.shape[1]} columns")

    out_path = "data/processed_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
