from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.data.feature_engineering.feature_builder import (
    FeatureBuilder,
    FeatureBuilderConfig,
)
from nfl_predictor.data.feature_engineering.rolling_features import (
    RollingSpec,
    add_rolling_features,
)
from nfl_predictor.data.feature_engineering.team_stats_pipeline import (
    TeamStatsConfig,
    build_game_level_features,
    build_team_features,
)
from nfl_predictor.data.preprocessing.base_dataset import (
    BaseDatasetConfig,
    build_base_dataset,
)


def test_feature_timing_basic_rolling() -> None:
    # Simple single-team, single-season series to verify shift+rolling.
    df = pd.DataFrame(
        {
            "team": ["A"] * 5,
            "season": [2021] * 5,
            "game_index": [0, 1, 2, 3, 4],
            "x": [10, 20, 30, 40, 50],
        }
    )

    specs = [RollingSpec(col="x", windows=(2,), stats=("mean",), min_periods=1)]
    out = add_rolling_features(
        df=df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=specs,
    )

    # rolling_mean_2 with a 1-row shift:
    # index 0: no prior -> NaN
    # index 1: prior = [10]        -> 10
    # index 2: prior = [10,20]     -> 15
    # index 3: prior = [20,30]     -> 25
    # index 4: prior = [30,40]     -> 35
    expected = [np.nan, 10.0, 15.0, 25.0, 35.0]
    col = "x_rolling_mean_2"
    assert col in out.columns
    np.testing.assert_allclose(out[col].values, expected, equal_nan=True)


def test_team_and_game_shapes() -> None:
    # 3 games, 4 teams (two matchups).
    base = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2021, 2021, 2021],
            "week": [1, 1, 2],
            "gameday": pd.to_datetime(["2021-09-10", "2021-09-10", "2021-09-17"]),
            "game_index": [0, 1, 2],
            "home_team": ["A", "C", "A"],
            "away_team": ["B", "D", "C"],
            "home_score": [20, 17, 24],
            "away_score": [10, 21, 17],
            "home_win": [1, 0, 1],
            "total_points": [30, 38, 41],
            "spread_line": [-7.0, 3.0, -3.5],
            "total_line": [44.5, 42.0, 45.0],
            "home_moneyline": [-200, 120, -150],
            "away_moneyline": [170, -140, 130],
            "is_regular_season": [True, True, True],
            "is_postseason": [False, False, False],
        }
    )

    cfg = TeamStatsConfig(seasons=[2021], include_postseason=True, rolling_windows=(2,))
    team_df = build_team_features(config=cfg, base_games=base)
    game_df = build_game_level_features(team_df=team_df, base_games=base, config=cfg)

    # 3 games -> 6 team rows
    assert len(team_df) == 2 * len(base)
    # game-level must match base row count
    assert len(game_df) == len(base)

    # Check some home_/away_ rolling cols exist
    assert any(c.startswith("home_points_for_rolling_mean") for c in game_df.columns)
    assert any(c.startswith("away_points_for_rolling_mean") for c in game_df.columns)


def test_no_future_information_for_team_points() -> None:
    # Two teams, same season, alternating games.
    df = pd.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "season": [2021, 2021, 2021, 2021],
            "game_index": [0, 1, 2, 3],
            "points_for": [10, 20, 30, 40],
        }
    )

    spec = RollingSpec(col="points_for", windows=(2,), stats=("mean",), min_periods=1)
    out = add_rolling_features(
        df=df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=[spec],
    )

    col = "points_for_rolling_mean_2"
    # For team A: game_index 0 -> NaN, 2 -> mean of [10] = 10
    a_rows = out[out["team"] == "A"].sort_values("game_index")
    np.testing.assert_allclose(
        a_rows[col].values,
        [np.nan, 10.0],
        equal_nan=True,
    )
    # For team B: game_index 1 -> NaN, 3 -> mean of [20] = 20
    b_rows = out[out["team"] == "B"].sort_values("game_index")
    np.testing.assert_allclose(
        b_rows[col].values,
        [np.nan, 20.0],
        equal_nan=True,
    )


def test_home_away_alignment_correct() -> None:
    # Simple scenario with two games, alternating home team.
    base = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": pd.to_datetime(["2021-09-10", "2021-09-17"]),
            "game_index": [0, 1],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_score": [10, 14],
            "away_score": [7, 21],
            "home_win": [1, 0],
            "total_points": [17, 35],
            "spread_line": [-3.0, 3.0],
            "total_line": [42.0, 44.0],
            "home_moneyline": [-150, 120],
            "away_moneyline": [130, -140],
            "is_regular_season": [True, True],
            "is_postseason": [False, False],
        }
    )

    cfg = TeamStatsConfig(seasons=[2021], include_postseason=True, rolling_windows=(2,))
    team_df = build_team_features(config=cfg, base_games=base)
    game_df = build_game_level_features(team_df=team_df, base_games=base, config=cfg)

    # For game 2 (game_index=1), home_team=B should have home_points_for_rolling_mean_2
    # based only on B's prior game (as away_team in game 1 with 7 points).
    g2 = game_df.loc[game_df["game_id"] == "g2"].iloc[0]
    assert g2["home_team"] == "B"
    # B's prior points_for is 7, so rolling mean_2 should be 7
    assert np.isclose(g2["home_points_for_rolling_mean_2"], 7.0)


def test_schedule_and_season_agg_leak_free_and_nan() -> None:
    # One team, three games with known spacing and outcomes.
    base = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2021, 2021, 2021],
            "week": [1, 2, 3],
            "gameday": pd.to_datetime(["2021-09-10", "2021-09-17", "2021-10-01"]),
            "game_index": [0, 1, 2],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_score": [7, 14, 21],
            "away_score": [10, 7, 3],
            "home_win": [0, 1, 1],
            "total_points": [17, 21, 24],
            "spread_line": [-1.0, -3.0, -7.0],
            "total_line": [40.0, 42.0, 44.0],
            "home_moneyline": [110, -140, -200],
            "away_moneyline": [-130, 120, 170],
            "is_regular_season": [True, True, True],
            "is_postseason": [False, False, False],
        }
    )

    cfg = TeamStatsConfig(seasons=[2021], include_postseason=True, rolling_windows=(2,))
    team_df = build_team_features(config=cfg, base_games=base)

    a_team = team_df[team_df["team"] == "A"].sort_values("game_index")

    # First game: no prior game -> NaNs
    assert np.isnan(a_team.iloc[0]["days_since_last_game"])
    assert np.isnan(a_team.iloc[0]["season_win_pct_to_date"])

    # Second game: 7 days since last, prior wins = [0] -> mean = 0
    assert a_team.iloc[1]["days_since_last_game"] == 7
    assert np.isclose(a_team.iloc[1]["season_win_pct_to_date"], 0.0)

    # Third game: 14 days since last, prior wins = [0,1] -> mean = 0.5
    assert a_team.iloc[2]["days_since_last_game"] == 14
    assert np.isclose(a_team.iloc[2]["season_win_pct_to_date"], 0.5)


def test_schedule_and_implied_prob_features_exposed_and_diffs_correct() -> None:
    # Two teams with different rest and moneylines.
    base = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2021, 2021],
            "week": [1, 2],
            "gameday": pd.to_datetime(["2021-09-10", "2021-09-24"]),
            "game_index": [0, 1],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [21, 17],
            "away_score": [14, 10],
            "home_win": [1, 1],
            "total_points": [35, 27],
            "spread_line": [-3.0, -7.0],
            "total_line": [45.0, 42.0],
            "home_moneyline": [-150, -200],
            "away_moneyline": [130, 170],
            "is_regular_season": [True, True],
            "is_postseason": [False, False],
        }
    )

    cfg = TeamStatsConfig(seasons=[2021], include_postseason=True, rolling_windows=(2,))
    team_df = build_team_features(config=cfg, base_games=base)
    game_df = build_game_level_features(team_df=team_df, base_games=base, config=cfg)

    # Columns should exist
    for col in [
        "home_days_since_last_game",
        "away_days_since_last_game",
        "home_games_played_season_to_date",
        "away_games_played_season_to_date",
        "home_season_win_pct_to_date",
        "away_season_win_pct_to_date",
        "home_implied_prob_ml",
        "away_implied_prob_ml",
        "diff_days_since_last_game",
        "diff_games_played_season_to_date",
        "diff_season_win_pct_to_date",
        "diff_implied_prob_ml",
        # schedule flags should also be propagated
        "home_is_short_week",
        "away_is_short_week",
        "home_is_long_rest",
        "away_is_long_rest",
        "home_coming_off_bye",
        "away_coming_off_bye",
    ]:
        assert col in game_df.columns, f"Missing expected column {col}"

    # For second game: both teams played game1 and game2 is 14 days after game1.
    g2 = game_df.loc[game_df["game_id"] == "g2"].iloc[0]

    # A is home both times: days_since_last_game for home team should be 14
    assert g2["home_days_since_last_game"] == 14

    # B is away both times: days_since_last_game for away team should also be 14
    assert g2["away_days_since_last_game"] == 14

    # games_played_season_to_date: each team has played 1 prior game by game2
    assert g2["home_games_played_season_to_date"] == 1
    assert g2["away_games_played_season_to_date"] == 1
    assert g2["diff_games_played_season_to_date"] == 0

    # implied probs should be consistent with moneylines and diffs should match
    # Compute implied probs for game2:
    def implied(odds: float) -> float:
        if odds < 0:
            return -odds / (-odds + 100.0)
        return 100.0 / (odds + 100.0)

    home_p = implied(-200)
    away_p = implied(170)
    assert np.isclose(g2["home_implied_prob_ml"], home_p, atol=1e-6)
    assert np.isclose(g2["away_implied_prob_ml"], away_p, atol=1e-6)
    assert np.isclose(
        g2["diff_implied_prob_ml"],
        home_p - away_p,
        atol=1e-6,
    )


@pytest.mark.integration
def test_feature_builder_end_to_end_integration() -> None:
    # Smoke test that FeatureBuilder can run over a real season subset.
    base_cfg = BaseDatasetConfig(
        seasons=[2021],
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )
    base_games = build_base_dataset(base_cfg)

    fb_cfg = FeatureBuilderConfig(
        seasons=[2021],
        include_postseason=True,
        include_markets=True,
        drop_preseason=True,
        save_intermediate=False,
    )
    fb = FeatureBuilder(fb_cfg)

    features = fb.build_features(base_games=base_games)

    # shape should match base dataset
    assert len(features) == len(base_games)

    # key targets and feature prefixes should exist
    for col in [
        "target_home_win",
        "target_total_points",
        "target_total_over",
    ]:
        assert col in features.columns

    assert any(c.startswith("home_points_for_rolling_mean") for c in features.columns)
    assert any(c.startswith("away_points_for_rolling_mean") for c in features.columns)
    assert "diff_season_win_pct_to_date" in features.columns