import pandas as pd
import pytest

from nfl_predictor.data.loaders.games import GameDataLoader, GameDataLoaderConfig


def test_games_loader_with_mock(monkeypatch, mock_games_data):
    """Unit test using mock schedules (no external dependency)."""

    def mock_import_schedules(seasons):
        return mock_games_data

    import nfl_data_py as nfl
    monkeypatch.setattr(nfl, "import_schedules", mock_import_schedules)

    config = GameDataLoaderConfig(
        seasons=[2023],
        save_parquet=False,
        include_markets=True,
    )
    loader = GameDataLoader(config=config)
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    # Core columns
    for col in [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "total_points",
    ]:
        assert col in df.columns

    # Market columns should be preserved
    for col in ["spread_line", "total_line", "home_moneyline", "away_moneyline"]:
        assert col in df.columns

    # Targets correct
    expected_home_win = (df["home_score"] > df["away_score"]).astype(int)
    assert (df["home_win"] == expected_home_win).all()

    expected_total = df["home_score"] + df["away_score"]
    assert (df["total_points"] == expected_total).all()


@pytest.mark.integration
def test_games_loader_real_smoke():
    """Optional integration test that hits nfl_data_py for real."""
    config = GameDataLoaderConfig(
        seasons=[2023],
        save_parquet=False,
        include_markets=True,
    )
    loader = GameDataLoader(config=config)
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "game_id" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["gameday"])
