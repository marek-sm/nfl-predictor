import pandas as pd
from nfl_predictor.data.feature_engineering.team_stats_pipeline import (
    TeamStatsConfig,
    build_team_features,
    build_game_level_features,
)

cfg = TeamStatsConfig(
    seasons=[2015, 2016, 2017],
    use_pbp_features=True,
    normalize_epa_by_season=True,
)

team_df = build_team_features(config=cfg)
game_df = build_game_level_features(team_df=team_df, config=cfg)

with pd.option_context("display.max_columns", 50):
    print(team_df[["off_epa_per_play", "def_epa_per_play_allowed"]].describe())

with pd.option_context("display.max_columns", 50):
    print(
        team_df[["season", "off_epa_per_play", "off_epa_per_play_z"]]
        .groupby("season")
        .describe()
    )
