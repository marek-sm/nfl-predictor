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


# ---------------------------------------------------------------------------
# Helpers for synthetic base datasets
# ---------------------------------------------------------------------------


def _make_tiny_base_df_single_matchup() -> pd.DataFrame:
    """
    Tiny base dataset with one matchup repeated to reason about timing.

    Game setup:
        Game 1: A (home) vs B       (2023-09-10)
        Game 2: B (home) vs A       (2023-09-17)
        Game 3: A (home) vs B       (2023-09-24)
    """
    data = {
        "game_id": ["g1", "g2", "g3"],
        "season": [2023, 2023, 2023],
        "week": [1, 2, 3],
        "gameday": pd.to_datetime(["2023-09-10", "2023-09-17", "2023-09-24"]),
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "A", "B"],
        "home_score": [10, 20, 30],
        "away_score": [7, 14, 21],
        "spread_line": [-3.5, -4.5, -6.5],  # home spreads
        "total_line": [42.5, 43.5, 44.5],
        "home_moneyline": [-160, -180, -220],
        "away_moneyline": [140, 155, 190],
    }
    df = pd.DataFrame(data)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]
    df["game_index"] = np.arange(len(df))
    df["is_regular_season"] = True
    df["is_postseason"] = False
    return df


def _make_small_base_df_multi_teams() -> pd.DataFrame:
    """
    Small base dataset with 3 games and 4 distinct teams.

    Games:
        g1: A vs B
        g2: C vs D
        g3: B vs A
    """
    data = {
        "game_id": ["g1", "g2", "g3"],
        "season": [2023, 2023, 2023],
        "week": [1, 1, 2],
        "gameday": pd.to_datetime(["2023-09-10", "2023-09-10", "2023-09-17"]),
        "home_team": ["A", "C", "B"],
        "away_team": ["B", "D", "A"],
        "home_score": [17, 24, 21],
        "away_score": [10, 20, 14],
        "spread_line": [-3.0, -4.0, -2.5],
        "total_line": [45.5, 47.5, 43.5],
        "home_moneyline": [-150, -180, -130],
        "away_moneyline": [130, 160, 110],
    }
    df = pd.DataFrame(data)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]
    df["game_index"] = np.arange(len(df))
    df["is_regular_season"] = True
    df["is_postseason"] = False
    return df


def _make_schedule_asymmetric_df() -> pd.DataFrame:
    """
    Base dataset with asymmetric rest between home and away before a given game.

    Games:
        g1: A vs C on 2023-09-10
        g2: B vs D on 2023-09-14
        g3: A vs B on 2023-09-24  (A had 14 days rest, B had 10 days)
    """
    data = {
        "game_id": ["g1", "g2", "g3"],
        "season": [2023, 2023, 2023],
        "week": [1, 1, 3],
        "gameday": pd.to_datetime(["2023-09-10", "2023-09-14", "2023-09-24"]),
        "home_team": ["A", "B", "A"],
        "away_team": ["C", "D", "B"],
        "home_score": [21, 17, 24],
        "away_score": [10, 14, 20],
        "spread_line": [-3.0, -2.5, -4.0],
        "total_line": [45.5, 42.5, 44.5],
        "home_moneyline": [-150, -130, -170],
        "away_moneyline": [130, 110, 150],
    }
    df = pd.DataFrame(data)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_points"] = df["home_score"] + df["away_score"]
    df["game_index"] = np.arange(len(df))
    df["is_regular_season"] = True
    df["is_postseason"] = False
    return df


# ---------------------------------------------------------------------------
# 1) Rolling logic / timing (pure rolling_features)
# ---------------------------------------------------------------------------


def test_feature_timing_basic_rolling():
    """
    add_rolling_features should use only strictly prior rows in the window.

    Construct a single team with linearly increasing values and verify
    that the rolling mean at each step uses only previous games.
    """
    df = pd.DataFrame(
        {
            "team": ["X"] * 5,
            "season": [2023] * 5,
            "game_index": [0, 1, 2, 3, 4],
            "value": [10, 20, 30, 40, 50],
        }
    )

    spec = RollingSpec(
        col="value",
        windows=[3],
        stats=("mean",),
        min_periods=1,
        prefix="value",
    )

    out = add_rolling_features(
        df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=[spec],
    )

    col = "value_rolling_mean_3"
    assert col in out.columns

    # Expected rolling means with strict "past only" windows:
    # game 0: NaN (no prior games)
    # game 1: mean([10]) = 10
    # game 2: mean([10, 20]) = 15
    # game 3: mean([10, 20, 30]) = 20
    # game 4: mean([20, 30, 40]) = 30
    expected = [np.nan, 10.0, 15.0, 20.0, 30.0]
    actual = out[col].tolist()

    assert np.isnan(actual[0])
    assert actual[1:] == pytest.approx(expected[1:], rel=1e-6)


# ---------------------------------------------------------------------------
# 2) Team and game shapes
# ---------------------------------------------------------------------------


def test_team_and_game_shapes():
    """
    Team-level and game-level DataFrames should have consistent shapes and
    home_/away_ rolling columns should exist.
    """
    base_df = _make_small_base_df_multi_teams()
    cfg = TeamStatsConfig(
        seasons=[2023],
        rolling_windows=(2,),
        min_games_for_rolling=1,
        include_postseason=True,
        save_team_level=False,
        save_game_level=False,
    )

    team_df = build_team_features(config=cfg, base_games=base_df)
    # Two rows per game
    assert len(team_df) == len(base_df) * 2

    game_df = build_game_level_features(
        team_df=team_df,
        base_games=base_df,
        config=cfg,
    )

    # Back to one row per game
    assert len(game_df) == len(base_df)

    # Should have some obvious home/away rolling columns
    assert any(
        col.startswith("home_points_for_rolling_mean_")
        for col in game_df.columns
    )
    assert any(
        col.startswith("away_points_for_rolling_mean_")
        for col in game_df.columns
    )


# ---------------------------------------------------------------------------
# 3) No future information for team points
# ---------------------------------------------------------------------------


def test_no_future_information_for_team_points():
    """
    In the toy dataset, verify that team rolling stats never use future games.

    For Team A (using a 2-game rolling window on team_points_for):
        - Game 1: no history -> NaN
        - Game 2: history = [game1]
        - Game 3: history = [game1, game2]
    """
    base_df = _make_tiny_base_df_single_matchup()
    cfg = TeamStatsConfig(
        seasons=[2023],
        rolling_windows=(2,),
        min_games_for_rolling=1,
        include_postseason=True,
    )

    team_df = build_team_features(config=cfg, base_games=base_df)

    # Filter to team A, sorted by game_index
    team_a = team_df[team_df["team"] == "A"].sort_values("game_index")

    col = "points_for_rolling_mean_2"
    assert col in team_a.columns

    # True points for A in each appearance:
    # game 1 as home: 10
    # game 2 as away: 14
    # game 3 as home: 30
    points_for = team_a["team_points_for"].tolist()
    assert points_for == [10, 14, 30]

    # Rolling mean with past-only 2-game window:
    # idx0: NaN
    # idx1: mean([10]) = 10
    # idx2: mean([10, 14]) = 12
    vals = team_a[col].tolist()
    assert np.isnan(vals[0])
    assert vals[1:] == pytest.approx([10.0, 12.0], rel=1e-6)


# ---------------------------------------------------------------------------
# 4) Home/away alignment correctness
# ---------------------------------------------------------------------------


def test_home_away_alignment_correct():
    """
    Verify that home_* features use the correct team's history and
    away_* features use the away team's history.
    """
    base_df = _make_tiny_base_df_single_matchup()
    cfg = TeamStatsConfig(
        seasons=[2023],
        rolling_windows=(2,),
        min_games_for_rolling=1,
        include_postseason=True,
    )

    team_df = build_team_features(config=cfg, base_games=base_df)
    game_df = build_game_level_features(
        team_df=team_df,
        base_games=base_df,
        config=cfg,
    )

    # Focus on game 2 (g2), where B is home and A is away
    row_g2 = game_df[game_df["game_id"] == "g2"].iloc[0]

    # For team B (home in g2), previous game was g1 as away with 7 points
    home_col = "home_points_for_rolling_mean_2"
    assert home_col in game_df.columns
    assert row_g2[home_col] == pytest.approx(7.0)

    # For team A (away in g2), previous game was g1 as home with 10 points
    away_col = "away_points_for_rolling_mean_2"
    assert away_col in game_df.columns
    assert row_g2[away_col] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 5) Schedule / season aggregates leak-free + NaN semantics
# ---------------------------------------------------------------------------


def test_schedule_and_season_agg_leak_free_and_nan():
    """
    Explicitly test that:
      - days_since_last_game only uses prior games;
      - season_win_pct_to_date uses only prior results;
      - first game entries have NaNs for these aggregates.
    """
    base_df = _make_tiny_base_df_single_matchup()
    cfg = TeamStatsConfig(
        seasons=[2023],
        rolling_windows=(2,),
        min_games_for_rolling=1,
        include_postseason=True,
    )

    team_df = build_team_features(config=cfg, base_games=base_df)

    # Team A
    team_a = team_df[team_df["team"] == "A"].sort_values("game_index")
    ds = team_a["days_since_last_game"].tolist()
    win_pct = team_a["season_win_pct_to_date"].tolist()
    wins = team_a["team_win"].tolist()
    # A's team_win values:
    # g1: A home win -> 1
    # g2: A away loss -> 0
    # g3: A home win -> 1
    assert wins == [1, 0, 1]

    # days_since_last_game: first should be NaN; subsequent equal to 7 days
    assert np.isnan(ds[0])
    assert ds[1:] == [7.0, 7.0]

    # season_win_pct_to_date (shifted expanding mean):
    # shifted team_win: [NaN, 1, 0]
    # expanding mean:   [NaN, 1.0, 0.5]
    assert np.isnan(win_pct[0])
    assert win_pct[1:] == pytest.approx([1.0, 0.5], rel=1e-6)

    # Team B should show analogous NaN behavior for its first game
    team_b = team_df[team_df["team"] == "B"].sort_values("game_index")
    assert np.isnan(team_b["days_since_last_game"].iloc[0])
    assert np.isnan(team_b["season_win_pct_to_date"].iloc[0])


# ---------------------------------------------------------------------------
# 6) Schedule, implied probs, and diff features exposed & correct
# ---------------------------------------------------------------------------


def test_schedule_and_implied_prob_features_exposed_and_diffs_correct():
    """
    Ensure schedule features, implied probabilities, and new diff columns
    make it through to the game-level features and are numerically sensible.
    """
    base_df = _make_schedule_asymmetric_df()
    cfg = TeamStatsConfig(
        seasons=[2023],
        rolling_windows=(2,),
        min_games_for_rolling=1,
        include_postseason=True,
    )

    team_df = build_team_features(config=cfg, base_games=base_df)
    game_df = build_game_level_features(
        team_df=team_df,
        base_games=base_df,
        config=cfg,
    )

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
    ]:
        assert col in game_df.columns, f"Missing expected column {col}"

    # Check implied prob numerics for game 1:
    # g1: home_moneyline = -150, away_moneyline = +130
    row_g1 = game_df[game_df["game_id"] == "g1"].iloc[0]

    home_p = row_g1["home_implied_prob_ml"]
    away_p = row_g1["away_implied_prob_ml"]

    expected_home_p = 150.0 / (150.0 + 100.0)  # -150
    expected_away_p = 100.0 / (130.0 + 100.0)  # +130

    assert home_p == pytest.approx(expected_home_p, rel=1e-6)
    assert away_p == pytest.approx(expected_away_p, rel=1e-6)

    # Now check diffs for game 3 (g3: A vs B, A home with 14 days rest, B with 10)
    row_g3 = game_df[game_df["game_id"] == "g3"].iloc[0]

    home_rest = row_g3["home_days_since_last_game"]
    away_rest = row_g3["away_days_since_last_game"]
    diff_rest = row_g3["diff_days_since_last_game"]
    assert home_rest == pytest.approx(14.0, rel=1e-6)
    assert away_rest == pytest.approx(10.0, rel=1e-6)
    assert diff_rest == pytest.approx(home_rest - away_rest, rel=1e-6)

    # games_played_season_to_date: both A and B have played 1 prior game
    home_games = row_g3["home_games_played_season_to_date"]
    away_games = row_g3["away_games_played_season_to_date"]
    diff_games = row_g3["diff_games_played_season_to_date"]
    assert home_games == 1
    assert away_games == 1
    assert diff_games == 0

    # diff_implied_prob_ml should equal home - away
    diff_prob = row_g3["diff_implied_prob_ml"]
    home_prob_g3 = row_g3["home_implied_prob_ml"]
    away_prob_g3 = row_g3["away_implied_prob_ml"]
    assert diff_prob == pytest.approx(home_prob_g3 - away_prob_g3, rel=1e-6)


# ---------------------------------------------------------------------------
# 7) Optional integration test (real data via nfl_data_py)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_feature_builder_end_to_end_integration():
    """
    Build features over a modest real season range to ensure the
    pipeline runs without error and produces sensible columns.

    This test relies on nfl_data_py via build_base_dataset and may be
    skipped in CI if the dependency or network is unavailable.
    """
    seasons = [2021, 2022]
    fb_config = FeatureBuilderConfig(
        seasons=seasons,
        include_postseason=True,
        include_markets=True,
        drop_preseason=True,
        save_intermediate=False,
    )
    builder = FeatureBuilder(fb_config)

    # Ensure base dataset builds
    base_df = build_base_dataset(
        BaseDatasetConfig(
            seasons=seasons,
            include_markets=True,
            drop_preseason=True,
            save_parquet=False,
        )
    )
    assert len(base_df) > 0

    # Build full features
    features_df = builder.build_features(base_games=base_df)
    assert len(features_df) == len(base_df)

    # Check for some expected columns
    assert "target_home_win" in features_df.columns
    assert "target_total_points" in features_df.columns
    assert "target_total_over" in features_df.columns
    assert any(
        col.startswith("home_points_for_rolling_mean_")
        for col in features_df.columns
    )
    assert any(
        col.startswith("away_points_for_rolling_mean_")
        for col in features_df.columns
    )
    assert any(
        col.startswith("diff_points_for_rolling_mean_")
        for col in features_df.columns
    )