import pandas as pd
import pytest

from nfl_predictor.data.loaders.games import GameDataLoader, GameDataLoaderConfig


@pytest.fixture(scope="module")
def real_games_df() -> pd.DataFrame:
    """
    Integration fixture: load real game data from nfl_data_py.

    We keep the season range modest so tests run reasonably fast.
    Adjust seasons as needed (e.g., range(2010, 2024)) once you're happy.
    """
    seasons = list(range(2020, 2024))  # 2020–2023
    loader = GameDataLoader(
        GameDataLoaderConfig(
            seasons=seasons,
            include_markets=True,
            save_parquet=False,
        )
    )
    df = loader.load()
    return df


@pytest.mark.integration
def test_game_id_unique(real_games_df: pd.DataFrame):
    """Each game_id should be unique (one row per game)."""
    df = real_games_df
    assert df["game_id"].is_unique, "Duplicate game_id values found."


@pytest.mark.integration
def test_no_nulls_in_critical_columns(real_games_df: pd.DataFrame):
    """Critical identifier columns should have no nulls."""
    df = real_games_df
    critical_cols = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
    missing_cols = [c for c in critical_cols if c not in df.columns]
    assert not missing_cols, f"Missing critical columns: {missing_cols}"

    null_counts = df[critical_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Nulls found in critical columns:\n{null_counts}"


@pytest.mark.integration
def test_week_in_reasonable_range(real_games_df: pd.DataFrame):
    """
    NFL weeks should be within a reasonable range.
    We allow up to ~22 to safely cover preseason/postseason labels.
    """
    df = real_games_df
    assert df["week"].ge(0).all(), "Negative week indices found."
    assert df["week"].le(22).all(), "Week index > 22 found, which is suspicious."


@pytest.mark.integration
def test_scores_non_negative_and_integerish(real_games_df: pd.DataFrame):
    """Scores should be non-negative and integer-valued (even if stored as float)."""
    df = real_games_df

    for col in ["home_score", "away_score"]:
        assert col in df.columns, f"Missing score column: {col}"
        assert (df[col] >= 0).all(), f"Negative values in {col}"

        # Check that scores are effectively integers (e.g., 21.0, 17.0)
        frac_part = df[col] % 1
        max_frac = frac_part.max()
        assert max_frac < 1e-6, f"Non-integer scores detected in {col}"


@pytest.mark.integration
def test_targets_consistent_with_scores(real_games_df: pd.DataFrame):
    """home_win and total_points should match the score columns."""
    df = real_games_df
    required = ["home_score", "away_score", "home_win", "total_points"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns for target consistency check: {missing}"

    expected_home_win = (df["home_score"] > df["away_score"]).astype(int)
    assert (df["home_win"] == expected_home_win).all(), "home_win does not match scores."

    expected_total = (df["home_score"] + df["away_score"]).astype(int)
    assert (df["total_points"] == expected_total).all(), "total_points != home_score + away_score."


@pytest.mark.integration
def test_market_columns_reasonable_ranges(real_games_df: pd.DataFrame):
    """
    Sanity-check market columns so we catch obviously corrupted data.
    These are soft bounds based on realistic NFL markets.
    """
    df = real_games_df

    if "spread_line" in df.columns:
        # spreads usually in [-25, 25] range
        assert df["spread_line"].abs().max() < 35, "spread_line values look unrealistic."

    if "total_line" in df.columns:
        # totals usually between 25 and 70
        valid_totals = df["total_line"].between(20, 80) | df["total_line"].isnull()
        assert valid_totals.all(), "total_line values look unrealistic."

    for col in ["home_moneyline", "away_moneyline"]:
        if col in df.columns:
            # typical moneylines in [-2000, 2000]
            valid_ml = df[col].abs() <= 3000
            assert valid_ml.all(), f"{col} values look unrealistic."


@pytest.mark.integration
def test_one_row_per_game_per_season_week(real_games_df: pd.DataFrame):
    """
    Check that the maximum number of games per (season, week) is plausible.
    Regular season weeks have up to 16 games; with postseason, it never explodes.
    """
    df = real_games_df
    games_per_week = df.groupby(["season", "week"]).size()
    max_games = games_per_week.max()
    assert max_games <= 20, f"Unrealistic number of games in a single week: {max_games}"


@pytest.mark.integration
def test_gameday_chronological(real_games_df: pd.DataFrame):
    """Ensure gameday can be sorted chronologically without weird types."""
    df = real_games_df
    assert pd.api.types.is_datetime64_any_dtype(df["gameday"]), "gameday is not datetime."

    df_sorted = df.sort_values("gameday")
    # This may not strictly be monotonic if there are ties or data quirks,
    # but it should not explode or change dtypes.
    assert len(df_sorted) == len(df), "Sorting by gameday changed row count."


@pytest.mark.integration
def test_no_future_games_with_completed_scores(real_games_df: pd.DataFrame):
    """
    As a basic leakage sanity check:
    - For any game with non-null scores, gameday should not be in the future.
    Note: depends on when you run this; it's mainly to catch obviously bad timestamps.
    """
    df = real_games_df

    completed = df[df["home_score"].notnull() & df["away_score"].notnull()]
    if not completed.empty:
        now = pd.Timestamp.now(tz=completed["gameday"].dt.tz) if completed["gameday"].dt.tz is not None else pd.Timestamp.now()
        assert (completed["gameday"] <= now).all(), "Completed games have gameday in the future."