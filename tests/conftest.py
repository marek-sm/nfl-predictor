import pytest
import pandas as pd


@pytest.fixture
def mock_games_data() -> pd.DataFrame:
    """Mock schedules data for unit tests (no live API calls)."""
    return pd.DataFrame(
        {
            "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
            "season": [2023, 2023],
            "week": [1, 1],
            "gameday": pd.to_datetime(["2023-09-07", "2023-09-10"]),
            "home_team": ["KC", "NYJ"],
            "away_team": ["DET", "BUF"],
            "home_score": [20, 22],
            "away_score": [21, 16],
            "result": [-1, 6],
            # market-like columns
            "spread_line": [-6.5, -2.5],
            "total_line": [54.5, 45.5],
            "home_moneyline": [-300, -140],
            "away_moneyline": [250, 120],
        }
    )
