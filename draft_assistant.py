import pandas as pd
import numpy as np
from typing import Optional

# =========================================================
# CONFIG
# =========================================================

LEAGUE_SIZE = 9
ROSTER_SIZE = 15

MAX_OVERSEAS = 4
MIN_BATTERS = 4
MIN_ALLROUNDERS = 2
MIN_BOWLERS = 3

W_PROJECTION = 0.45
W_VOR = 0.25
W_SCARCITY = 0.10
W_CEILING = 0.12
W_FLOOR = 0.08

USE_WEEKLY_MATCH_MULTIPLIER = False

CSV_PATH = "player_stats_updated.csv"
MIN_MATCHES_FILTER = 5  # bump to 3 or 5 if you want to kill tiny-sample weirdness


# =========================================================
# HELPERS
# =========================================================

def normalize_role(role: str) -> str:
    r = str(role).strip().lower()
    if r == "wicketkeeper":
        return "batter"
    if r == "batter":
        return "batter"
    if r == "all_rounder":
        return "allrounder"
    if r == "bowler":
        return "bowler"
    return "batter"


def to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def safe_div(a, b):
    return a / b if b not in (0, None) else 0.0


def batting_sr_bonus(sr: float, avg_runs_per_match: float, avg_balls_per_match: float) -> float:
    if avg_balls_per_match < 10:
        return 0.0

    if sr < 100:
        return -30
    elif sr <= 110:
        return -20
    elif sr <= 120:
        return -10
    elif sr <= 140:
        return 0
    elif sr <= 160:
        return 10 if avg_runs_per_match >= 20 else 0
    elif sr <= 180:
        return 15 if avg_runs_per_match >= 20 else 0
    elif sr <= 200:
        return 20 if avg_runs_per_match >= 20 else 0
    else:
        return 30 if avg_runs_per_match >= 20 else 0


def bowling_econ_bonus(econ: float, avg_overs_per_match: float) -> float:
    if avg_overs_per_match < 2:
        return 0.0

    if econ < 4:
        return 40
    elif econ < 6:
        return 30
    elif econ < 7:
        return 20
    elif econ < 8:
        return 10
    elif econ < 9:
        return 0
    elif econ < 10:
        return -5
    elif econ < 12:
        return -15
    elif econ < 15:
        return -20
    else:
        return -30


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def compute_player_metrics(row: pd.Series) -> pd.Series:
    matches = max(int(row["matches"]), 1)

    batting_total = (
        row["runs"]
        + 1 * row["fours"]
        + 2 * row["sixes"]
        + 10 * row["score_25_plus"]
        + 15 * row["score_40_plus"]
        + 20 * row["score_60_plus"]
        + 25 * row["score_80_plus"]
        + 40 * row["score_100_plus"]
        - 10 * row["ducks"]
        - 5 * row["low_score_le_5"]
    )
    batting_ppm = batting_total / matches

    strike_rate = 100 * safe_div(row["runs"], row["balls_faced"])
    avg_runs_per_match = safe_div(row["runs"], matches)
    avg_balls_per_match = safe_div(row["balls_faced"], matches)
    sr_bonus = batting_sr_bonus(strike_rate, avg_runs_per_match, avg_balls_per_match)

    bowling_total = (
        30 * row["wickets"]
        + row["dot_balls"]
        + 40 * row["maidens"]
        - row["wides"]
        - row["no_balls"]
        + 5 * row["bowled_lbw_wickets"]
        + 10 * row["wickets_2_plus"]
        + 20 * row["wickets_3_plus"]
        + 30 * row["wickets_4_plus"]
        + 40 * row["wickets_5_plus"]
        + 60 * row["wickets_6_plus"]
    )
    bowling_ppm = bowling_total / matches

    total_overs = row["balls_bowled"] / 6 if row["balls_bowled"] > 0 else 0
    economy_rate = safe_div(row["runs_conceded"], total_overs) if total_overs > 0 else np.nan
    avg_overs_per_match = safe_div(total_overs, matches)
    econ_bonus = bowling_econ_bonus(economy_rate, avg_overs_per_match) if total_overs > 0 else 0.0

    fielding_total = (
        10 * row["catches"]
        + 20 * row["stumpings"]
        + 10 * row["runouts"]
        + 10 * row["two_catch_games"]
    )
    fielding_ppm = fielding_total / matches

    participation_ppm = 5
    win_bonus_proxy = 2

    projected_ppm = (
        batting_ppm
        + sr_bonus
        + bowling_ppm
        + econ_bonus
        + fielding_ppm
        + participation_ppm
        + win_bonus_proxy
    )

    weekly_multiplier = (
        row["projected_matches_this_week"]
        if USE_WEEKLY_MATCH_MULTIPLIER and "projected_matches_this_week" in row
        else 1
    )
    weekly_projection = projected_ppm * weekly_multiplier

    ceiling_proxy = (
        2.5 * safe_div(row["score_60_plus"], matches)
        + 4.0 * safe_div(row["score_100_plus"], matches)
        + 2.0 * safe_div(row["wickets_3_plus"], matches)
        + 3.0 * safe_div(row["wickets_4_plus"], matches)
        + 1.5 * safe_div(row["wickets_5_plus"], matches)
        + 0.35 * safe_div(row["sixes"], matches)
        + 1.0 * safe_div(row["maidens"], matches)
        + 0.75 * safe_div(row["two_catch_games"], matches)
    )

    floor_proxy = (
        0.30 * avg_runs_per_match
        + 6.0 * safe_div(row["innings_batted"], matches)
        + 5.0 * avg_overs_per_match
        + 1.5 * safe_div(row["dot_balls"], matches)
        + 1.0 * safe_div(row["catches"], matches)
        - 4.0 * safe_div(row["ducks"], matches)
        - 2.0 * safe_div(row["low_score_le_5"], matches)
    )

    consistency_proxy = (
        1.0 * safe_div(row["score_25_plus"], matches)
        + 1.0 * safe_div(row["wickets_2_plus"], matches)
        + 0.5 * safe_div(row["innings_batted"], matches)
        + 0.5 * avg_overs_per_match
        - 0.8 * safe_div(row["ducks"], matches)
    )

    return pd.Series({
        "normalized_role": normalize_role(row["role"]),
        "strike_rate": round(strike_rate, 2) if row["balls_faced"] > 0 else 0.0,
        "economy_rate": round(economy_rate, 2) if total_overs > 0 else np.nan,
        "batting_ppm": round(batting_ppm, 2),
        "sr_bonus_ppm": round(sr_bonus, 2),
        "bowling_ppm": round(bowling_ppm, 2),
        "econ_bonus_ppm": round(econ_bonus, 2),
        "fielding_ppm": round(fielding_ppm, 2),
        "projected_points_per_match": round(projected_ppm, 2),
        "weekly_projection": round(weekly_projection, 2),
        "ceiling_proxy": round(ceiling_proxy, 4),
        "floor_proxy": round(floor_proxy, 4),
        "consistency_proxy": round(consistency_proxy, 4),
    })


# =========================================================
# DRAFT ENGINE
# =========================================================

class SnakeDraftAssistant:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.available = self.df.copy()
        self.my_roster = []
        self._build_rankings()

    def _build_rankings(self):
        self.available["role_rank"] = (
            self.available.groupby("normalized_role")["weekly_projection"]
            .rank(ascending=False, method="dense")
        )

        replacement_index = {
            "batter": max(1, LEAGUE_SIZE * 5),
            "allrounder": max(1, LEAGUE_SIZE * 2),
            "bowler": max(1, LEAGUE_SIZE * 4),
        }

        replacement_map = {}
        for role in ["batter", "allrounder", "bowler"]:
            sub = (
                self.available[self.available["normalized_role"] == role]
                .sort_values("weekly_projection", ascending=False)
                .reset_index(drop=True)
            )
            if len(sub) == 0:
                replacement_map[role] = 0.0
            else:
                idx = min(len(sub) - 1, replacement_index[role] - 1)
                replacement_map[role] = sub.iloc[idx]["weekly_projection"]

        self.available["replacement_level"] = self.available["normalized_role"].map(replacement_map)
        self.available["vor"] = self.available["weekly_projection"] - self.available["replacement_level"]

        scarcity_frames = []
        for role in ["batter", "allrounder", "bowler"]:
            sub = (
                self.available[self.available["normalized_role"] == role]
                .sort_values("weekly_projection", ascending=False)
                .reset_index(drop=True)
            )
            vals = sub["weekly_projection"].tolist()
            drops = []
            for i, val in enumerate(vals):
                nxt = vals[i + 1] if i + 1 < len(vals) else vals[-1]
                drops.append(val - nxt)
            sub = sub.copy()
            sub["scarcity_score_raw"] = drops
            scarcity_frames.append(sub[["player", "scarcity_score_raw"]])

        scarcity_df = pd.concat(scarcity_frames, ignore_index=True) if scarcity_frames else pd.DataFrame()
        self.available = self.available.drop(columns=["scarcity_score_raw"], errors="ignore")
        self.available = self.available.merge(scarcity_df, on="player", how="left")

        for col in ["weekly_projection", "vor", "scarcity_score_raw", "ceiling_proxy", "floor_proxy"]:
            min_v = self.available[col].min()
            max_v = self.available[col].max()
            if pd.isna(min_v) or pd.isna(max_v) or max_v - min_v == 0:
                self.available[col + "_norm"] = 0.0
            else:
                self.available[col + "_norm"] = (self.available[col] - min_v) / (max_v - min_v)

        self.available["captain_upside"] = (
            0.6 * self.available["ceiling_proxy_norm"] +
            0.4 * self.available["weekly_projection_norm"]
        )

        self.available["draft_score_base"] = (
            W_PROJECTION * self.available["weekly_projection_norm"] +
            W_VOR * self.available["vor_norm"] +
            W_SCARCITY * self.available["scarcity_score_raw_norm"] +
            W_CEILING * self.available["ceiling_proxy_norm"] +
            W_FLOOR * self.available["floor_proxy_norm"]
        )

    def get_my_counts(self):
        roster = pd.DataFrame(self.my_roster)
        if roster.empty:
            return {"size": 0, "overseas": 0, "batter": 0, "allrounder": 0, "bowler": 0}
        return {
            "size": len(roster),
            "overseas": int(roster["is_overseas"].sum()),
            "batter": int((roster["normalized_role"] == "batter").sum()),
            "allrounder": int((roster["normalized_role"] == "allrounder").sum()),
            "bowler": int((roster["normalized_role"] == "bowler").sum()),
        }

    def roster_need_multiplier(self, role: str, is_overseas: bool) -> float:
        counts = self.get_my_counts()
        picks_left = ROSTER_SIZE - counts["size"]

        need_bat = max(0, MIN_BATTERS - counts["batter"])
        need_ar = max(0, MIN_ALLROUNDERS - counts["allrounder"])
        need_bowl = max(0, MIN_BOWLERS - counts["bowler"])
        need_overseas_room = MAX_OVERSEAS - counts["overseas"]

        mult = 1.0

        if role == "batter" and need_bat > 0:
            mult += 0.10 + 0.08 * need_bat
        if role == "allrounder" and need_ar > 0:
            mult += 0.18 + 0.10 * need_ar
        if role == "bowler" and need_bowl > 0:
            mult += 0.12 + 0.08 * need_bowl

        total_required_remaining = need_bat + need_ar + need_bowl
        if picks_left > 0 and total_required_remaining >= picks_left - 1:
            if (role == "batter" and need_bat > 0) or (role == "allrounder" and need_ar > 0) or (role == "bowler" and need_bowl > 0):
                mult += 0.20

        if is_overseas:
            if need_overseas_room <= 0:
                return 0.0
            if need_overseas_room == 1:
                mult -= 0.20
            elif need_overseas_room == 2:
                mult -= 0.08

        return max(mult, 0.0)

    def get_dynamic_board(self) -> pd.DataFrame:
        board = self.available.copy()
        board["draft_score_dynamic"] = [
            row["draft_score_base"] * self.roster_need_multiplier(row["normalized_role"], row["is_overseas"])
            for _, row in board.iterrows()
        ]

        ranks = board["scarcity_score_raw"].rank(method="first")
        unique_count = ranks.nunique()
        if unique_count >= 3:
            board["urgency_label"] = pd.qcut(ranks, q=3, labels=["can wait", "watchlist", "pick now"])
        else:
            board["urgency_label"] = "watchlist"

        return board.sort_values(
            by=["draft_score_dynamic", "weekly_projection"],
            ascending=False
        ).reset_index(drop=True)

    def draft_player(self, player_name: str):
        mask = self.available["player"].str.lower() == player_name.strip().lower()
        if not mask.any():
            print(f"Player '{player_name}' not found.")
            return
        row = self.available[mask].iloc[0].to_dict()
        self.my_roster.append(row)
        self.available = self.available[~mask].reset_index(drop=True)
        self._build_rankings()
        print(f"Drafted: {row['player']} ({row['normalized_role']})")

    def remove_player_taken_by_others(self, player_name: str):
        mask = self.available["player"].str.lower() == player_name.strip().lower()
        if not mask.any():
            print(f"Player '{player_name}' not found.")
            return
        row = self.available[mask].iloc[0]
        self.available = self.available[~mask].reset_index(drop=True)
        self._build_rankings()
        print(f"Removed: {row['player']}")

    def best_available(self, n=15) -> pd.DataFrame:
        board = self.get_dynamic_board()
        cols = [
            "player", "team", "normalized_role", "is_overseas",
            "weekly_projection", "vor", "ceiling_proxy", "floor_proxy",
            "captain_upside", "draft_score_dynamic", "urgency_label"
        ]
        return board[cols].head(n)

    def best_by_role(self, role: str, n=10) -> pd.DataFrame:
        role = "allrounder" if role == "all_rounder" else role
        board = self.get_dynamic_board()
        board = board[board["normalized_role"] == role].copy()
        cols = [
            "player", "team", "is_overseas", "weekly_projection",
            "vor", "ceiling_proxy", "floor_proxy",
            "captain_upside", "draft_score_dynamic", "urgency_label"
        ]
        return board[cols].head(n)

    def suggest_pick(self) -> Optional[pd.Series]:
        board = self.get_dynamic_board()
        if board.empty:
            return None
        return board.iloc[0]

    def captain_candidates(self, n=8) -> pd.DataFrame:
        board = self.get_dynamic_board().copy()
        board["captain_score"] = (
            0.55 * board["weekly_projection_norm"] +
            0.45 * board["ceiling_proxy_norm"]
        )
        cols = [
            "player", "team", "normalized_role", "is_overseas",
            "weekly_projection", "ceiling_proxy", "captain_score"
        ]
        return board.sort_values("captain_score", ascending=False)[cols].head(n)

    def vice_captain_candidates(self, n=8) -> pd.DataFrame:
        board = self.get_dynamic_board().copy()
        board["vice_captain_score"] = (
            0.55 * board["weekly_projection_norm"] +
            0.25 * board["ceiling_proxy_norm"] +
            0.20 * board["floor_proxy_norm"]
        )
        cols = [
            "player", "team", "normalized_role", "is_overseas",
            "weekly_projection", "floor_proxy", "ceiling_proxy", "vice_captain_score"
        ]
        return board.sort_values("vice_captain_score", ascending=False)[cols].head(n)

    def show_roster(self):
        if not self.my_roster:
            print("Your roster is empty.")
            return
        roster = pd.DataFrame(self.my_roster)
        cols = [
            "player", "team", "normalized_role", "is_overseas",
            "weekly_projection", "ceiling_proxy", "floor_proxy"
        ]
        print(roster[cols].sort_values("weekly_projection", ascending=False).to_string(index=False))
        print("\nRoster Counts:")
        print(self.get_my_counts())

    def role_scarcity_report(self):
        rep = self.available.groupby("normalized_role").agg(
            players_left=("player", "count"),
            best_projection=("weekly_projection", "max"),
            median_projection=("weekly_projection", "median"),
            avg_vor=("vor", "mean")
        ).reset_index()
        print(rep.to_string(index=False))

    def next_turn_view(self, picks_until_next_turn=15, n_per_role=8):
        board = self.get_dynamic_board().copy()
        cols = [
            "player", "team", "normalized_role", "is_overseas",
            "weekly_projection", "vor", "draft_score_dynamic", "urgency_label"
        ]

        print("\n=== BEST AVAILABLE RIGHT NOW ===")
        print(board[cols].head(20).to_string(index=False))

        future_board = board.iloc[picks_until_next_turn:].copy()
        print(f"\n=== LIKELY LEFT BY NEXT TURN (~{picks_until_next_turn} PICKS LATER) ===")
        if future_board.empty:
            print("No future board available.")
        else:
            print(future_board[cols].head(20).to_string(index=False))

        for role in ["allrounder", "bowler", "batter"]:
            print(f"\n=== {role.upper()} NOW ===")
            print(board[board["normalized_role"] == role][cols].head(n_per_role).to_string(index=False))

            print(f"\n=== {role.upper()} LIKELY LEFT ===")
            sub_future = future_board[future_board["normalized_role"] == role][cols].head(n_per_role)
            if sub_future.empty:
                print("None likely left.")
            else:
                print(sub_future.to_string(index=False))

    def best_pick_for_turn_one(self, picks_until_next_turn=15):
        board = self.get_dynamic_board().copy()
        future_board = board.iloc[picks_until_next_turn:].copy()

        recommendations = []
        for role in ["allrounder", "bowler", "batter"]:
            now_role = board[board["normalized_role"] == role].head(1)
            future_role = future_board[future_board["normalized_role"] == role].head(1)

            if now_role.empty:
                continue

            now_player = now_role.iloc[0]
            future_score = future_role.iloc[0]["draft_score_dynamic"] if not future_role.empty else 0
            dropoff = now_player["draft_score_dynamic"] - future_score

            recommendations.append({
                "role": role,
                "pick_now": now_player["player"],
                "pick_now_score": round(now_player["draft_score_dynamic"], 4),
                "best_likely_left_next_turn_score": round(future_score, 4),
                "dropoff_to_next_turn": round(dropoff, 4),
            })

        rec_df = pd.DataFrame(recommendations).sort_values("dropoff_to_next_turn", ascending=False).reset_index(drop=True)
        print("\n=== TURN 1 ROLE DROP-OFF ANALYSIS ===")
        print(rec_df.to_string(index=False))
        if not rec_df.empty:
            best = rec_df.iloc[0]
            print(f"\nRecommended first-pick role: {best['role']}")
            print(f"Recommended first pick: {best['pick_now']}")


# =========================================================
# CLI
# =========================================================

def print_help():
    print("""
Commands:
  board [n]               -> overall best available
  role batter [n]         -> top batters
  role all_rounder [n]    -> top all-rounders
  role bowler [n]         -> top bowlers
  suggest                 -> single best suggested pick now
  captain [n]             -> captain candidates
  vice [n]                -> vice-captain candidates
  scarcity                -> role scarcity report
  roster                  -> show your roster
  draft <player name>     -> add your pick
  remove <player name>    -> remove player taken by someone else
  turn1 [picks]           -> 1.01 / next turn view, default 15
  help                    -> show commands
  quit                    -> exit
""")


def main():
    df = pd.read_csv(CSV_PATH)
    df["is_overseas"] = df["is_overseas"].apply(to_bool)

    numeric_cols = [
        "matches","runs","balls_faced","fours","sixes","innings_batted","dismissals","ducks",
        "low_score_le_5","score_25_plus","score_40_plus","score_60_plus","score_80_plus",
        "score_100_plus","balls_bowled","runs_conceded","wickets","dot_balls","maidens",
        "wides","no_balls","bowled_lbw_wickets","wickets_2_plus","wickets_3_plus",
        "wickets_4_plus","wickets_5_plus","wickets_6_plus","catches","stumpings","runouts",
        "two_catch_games"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if MIN_MATCHES_FILTER > 1:
        df = df[df["matches"] >= MIN_MATCHES_FILTER].copy()

    metrics = df.apply(compute_player_metrics, axis=1)
    df = pd.concat([df, metrics], axis=1)

    assistant = SnakeDraftAssistant(df)

    print("\n=== IPL DRAFT ASSISTANT READY ===")
    print(assistant.best_available(15).to_string(index=False))
    print_help()

    while True:
        try:
            raw = input("\ndraft> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"quit", "exit"}:
            break

        elif cmd == "help":
            print_help()

        elif cmd == "board":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
            print(assistant.best_available(n).to_string(index=False))

        elif cmd == "role":
            if len(parts) < 2:
                print("Usage: role batter|all_rounder|bowler [n]")
                continue
            role = parts[1].lower()
            n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 12
            print(assistant.best_by_role(role, n).to_string(index=False))

        elif cmd == "suggest":
            pick = assistant.suggest_pick()
            if pick is None:
                print("No players available.")
            else:
                print(pick[[
                    "player", "team", "normalized_role", "is_overseas",
                    "weekly_projection", "vor", "draft_score_dynamic", "urgency_label"
                ]].to_string())

        elif cmd == "captain":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            print(assistant.captain_candidates(n).to_string(index=False))

        elif cmd == "vice":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            print(assistant.vice_captain_candidates(n).to_string(index=False))

        elif cmd == "scarcity":
            assistant.role_scarcity_report()

        elif cmd == "roster":
            assistant.show_roster()

        elif cmd == "draft":
            name = raw[len("draft"):].strip()
            if not name:
                print("Usage: draft <player name>")
                continue
            assistant.draft_player(name)

        elif cmd == "remove":
            name = raw[len("remove"):].strip()
            if not name:
                print("Usage: remove <player name>")
                continue
            assistant.remove_player_taken_by_others(name)

        elif cmd == "turn1":
            picks = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
            assistant.next_turn_view(picks_until_next_turn=picks)
            assistant.best_pick_for_turn_one(picks_until_next_turn=picks)

        else:
            print("Unknown command. Type 'help'.")

    print("Done.")


if __name__ == "__main__":
    main()