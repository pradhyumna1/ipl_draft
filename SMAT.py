import pandas as pd
import re

BASE_CSV = "/Users/pradhyumnaadusumilli/ipl_draft/player_stats_aggregated.csv"
SMAT_CSV = "/Users/pradhyumnaadusumilli/ipl_draft/smat_player_stats_only.csv"

MATCHED_ONLY_OUT = "/Users/pradhyumnaadusumilli/ipl_draft/ipl_players_with_smat_only.csv"
FULL_UPDATED_OUT = "/Users/pradhyumnaadusumilli/ipl_draft/player_stats_updated.csv"

# Conservative recent-form weighting
DEFAULT_SMAT_WEIGHT = 0.30

STAT_COLS = [
    "matches",
    "runs",
    "balls_faced",
    "fours",
    "sixes",
    "innings_batted",
    "dismissals",
    "ducks",
    "low_score_le_5",
    "score_25_plus",
    "score_40_plus",
    "score_60_plus",
    "score_80_plus",
    "score_100_plus",
    "balls_bowled",
    "runs_conceded",
    "wickets",
    "dot_balls",
    "maidens",
    "wides",
    "no_balls",
    "bowled_lbw_wickets",
    "wickets_2_plus",
    "wickets_3_plus",
    "wickets_4_plus",
    "wickets_5_plus",
    "wickets_6_plus",
    "catches",
    "stumpings",
    "runouts",
    "two_catch_games",
]

# Slightly lower on fielding/matches, otherwise 0.30
STAT_WEIGHTS = {
    "matches": 0.20,
    "runs": 0.30,
    "balls_faced": 0.30,
    "fours": 0.30,
    "sixes": 0.30,
    "innings_batted": 0.25,
    "dismissals": 0.30,
    "ducks": 0.30,
    "low_score_le_5": 0.30,
    "score_25_plus": 0.30,
    "score_40_plus": 0.30,
    "score_60_plus": 0.30,
    "score_80_plus": 0.30,
    "score_100_plus": 0.30,
    "balls_bowled": 0.30,
    "runs_conceded": 0.30,
    "wickets": 0.30,
    "dot_balls": 0.30,
    "maidens": 0.30,
    "wides": 0.30,
    "no_balls": 0.30,
    "bowled_lbw_wickets": 0.30,
    "wickets_2_plus": 0.30,
    "wickets_3_plus": 0.30,
    "wickets_4_plus": 0.30,
    "wickets_5_plus": 0.30,
    "wickets_6_plus": 0.30,
    "catches": 0.20,
    "stumpings": 0.20,
    "runouts": 0.20,
    "two_catch_games": 0.20,
}


def name_key(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    base = pd.read_csv(BASE_CSV)
    smat = pd.read_csv(SMAT_CSV)

    # Normalize names
    base["name_key"] = base["player"].apply(name_key)
    smat["name_key"] = smat["player"].apply(name_key)

    # Ensure numeric types
    for col in STAT_COLS:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)
        else:
            base[col] = 0.0

        if col in smat.columns:
            smat[col] = pd.to_numeric(smat[col], errors="coerce").fillna(0.0)
        else:
            smat[col] = 0.0

    # Only IPL/base players who also exist in SMAT
    matched = base.merge(
        smat[["name_key"] + STAT_COLS],
        on="name_key",
        how="inner",
        suffixes=("_base", "_smat"),
    )

    print(f"Matched IPL/base players with SMAT stats: {len(matched)}")

    # Build matched-only updated dataset
    matched_updated = matched.copy()

    for col in STAT_COLS:
        w = STAT_WEIGHTS.get(col, DEFAULT_SMAT_WEIGHT)
        matched_updated[col] = (
            matched_updated[f"{col}_base"].fillna(0.0)
            + matched_updated[f"{col}_smat"].fillna(0.0) * w
        )

    # Keep base metadata
    meta_cols = ["team", "player", "is_overseas", "role"]
    matched_updated = matched_updated[meta_cols + STAT_COLS].copy()

    # Optional: round weighted outputs for readability
    for col in STAT_COLS:
        matched_updated[col] = matched_updated[col].round(2)

    matched_updated = matched_updated.sort_values(
        by=["team", "player"]
    ).reset_index(drop=True)

    matched_updated.to_csv(MATCHED_ONLY_OUT, index=False)
    print(f"Saved matched-only weighted update -> {MATCHED_ONLY_OUT}")

    # Merge weighted combined stats back into full base file
    combined_lookup = matched_updated.copy()
    combined_lookup["name_key"] = combined_lookup["player"].apply(name_key)

    final_df = base.copy()

    final_df = final_df.merge(
        combined_lookup[["name_key"] + STAT_COLS],
        on="name_key",
        how="left",
        suffixes=("", "_weighted"),
    )

    for col in STAT_COLS:
        weighted_col = f"{col}_weighted"
        final_df[col] = final_df[weighted_col].fillna(final_df[col])

    drop_cols = [f"{c}_weighted" for c in STAT_COLS if f"{c}_weighted" in final_df.columns]
    final_df = final_df.drop(columns=drop_cols + ["name_key"])

    # Round stat columns for readability
    for col in STAT_COLS:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce").fillna(0).round(2)

    final_df.to_csv(FULL_UPDATED_OUT, index=False)
    print(f"Saved full weighted updated file -> {FULL_UPDATED_OUT}")

    # Quick sanity print
    print("\nSample updated matched players:")
    print(matched_updated[["team", "player", "matches", "runs", "wickets"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()