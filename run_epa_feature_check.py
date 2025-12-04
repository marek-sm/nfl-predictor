"""
Quick sanity check for EPA / PBP features.

Usage:
    poetry run python run_epa_feature_check.py
"""

from __future__ import annotations

import pandas as pd

from nfl_predictor.data.feature_engineering.team_stats_pipeline import (
    TeamStatsConfig,
    build_team_features,
    build_game_level_features,
)


def main() -> None:
    # Configure seasons + enable PBP/EPA features
    cfg = TeamStatsConfig(
        seasons=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        use_pbp_features=True,
    )

    print("Building team-level features with EPA / PBP metrics enabled...")
    team_df = build_team_features(config=cfg)
    print(f"Team features shape: {team_df.shape}")
    print("Sample team columns:")
    print(sorted([c for c in team_df.columns if "epa" in c or "success_rate" in c])[:40])

    print("\nBuilding game-level features with EPA matchup diffs...")
    game_df = build_game_level_features(team_df=team_df, config=cfg)
    print(f"Game features shape: {game_df.shape}")
    print("Sample game-level EPA diff columns:")
    epa_diff_cols = [c for c in game_df.columns if "diff_" in c and "epa" in c]
    print(sorted(epa_diff_cols)[:40])

    # Show a couple of rows for eyeballing
    with pd.option_context("display.max_columns", 50):
        print("\n=== Sample team_df head (EPA-related columns) ===")
        print(
            team_df[
                [
                    "season",
                    "team",
                    "game_id",
                    "off_epa_per_play",
                    "def_epa_per_play_allowed",
                ]
            ]
            .head()
            .to_string(index=False)
        )

        epa_cols = [
            c
            for c in game_df.columns
            if "diff_off_epa_per_play_rolling_mean" in c
            or "diff_def_epa_per_play_allowed_rolling_mean" in c
        ]
        print("\n=== Sample game_df head (EPA diff-related columns) ===")
        print(
            game_df[
                ["season", "week", "home_team", "away_team"] + epa_cols[:4]
            ]
            .head()
            .to_string(index=False)
        )

        


if __name__ == "__main__":
    main()