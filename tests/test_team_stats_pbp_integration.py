from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.data.feature_engineering import team_stats_pipeline as tsp


@pytest.fixture
def stub_epa_df() -> pd.DataFrame:
    # minimal team_epa frame that looks like build_team_epa_features output
    return pd.DataFrame(
        {
            "season": [2020, 2020],
            "game_id": ["2020_01_BUF_NE", "2020_01_BUF_NE"],
            "team": ["BUF", "NE"],
            "plays_off": [60, 55],
            "plays_def": [55, 60],
            "off_epa_per_play": [0.2, -0.1],
            "def_epa_per_play_allowed": [-0.1, 0.2],
            "off_success_rate": [0.5, 0.45],
            "def_success_rate_allowed": [0.45, 0.5],
            "off_explosive_play_rate": [0.12, 0.08],
            "def_explosive_play_rate_allowed": [0.08, 0.12],
        }
    )


def test_team_features_without_pbp_does_not_include_epa_columns():
    cfg = tsp.TeamStatsConfig(
        seasons=[2020],
        use_pbp_features=False,
        rolling_windows=(3,),  # keep small for speed
    )

    team_df = tsp.build_team_features(config=cfg)

    # We should not have any EPA-related columns when PBP features are off
    epa_cols = [c for c in team_df.columns if "epa" in c or "success_rate" in c]
    # it is okay if other "success_rate" exist, but we expect no off_/def_ EPA columns
    assert not any(c.startswith("off_epa_per_play") for c in epa_cols)
    assert not any(c.startswith("def_epa_per_play_allowed") for c in epa_cols)


def test_team_features_with_pbp_includes_epa_columns(monkeypatch, stub_epa_df):
    # monkeypatch build_team_epa_features to avoid network / nflreadpy usage
    import nfl_predictor.data.feature_engineering.epa_features as epa_mod

    def fake_build_team_epa_features(base_games, config=None, client=None):
        # Ignore inputs, just return stub
        return stub_epa_df

    monkeypatch.setattr(epa_mod, "build_team_epa_features", fake_build_team_epa_features)

    cfg = tsp.TeamStatsConfig(
        seasons=[2020],
        use_pbp_features=True,
        rolling_windows=(3,),
    )

    team_df = tsp.build_team_features(config=cfg)

    # Base EPA columns should be present
    assert "off_epa_per_play" in team_df.columns
    assert "def_epa_per_play_allowed" in team_df.columns

    # Rolling EPA columns should be present as well
    assert "off_epa_per_play_rolling_mean_3" in team_df.columns
    assert "def_epa_per_play_allowed_rolling_mean_3" in team_df.columns


def test_game_level_epa_matchup_diffs(monkeypatch, stub_epa_df):
    import nfl_predictor.data.feature_engineering.epa_features as epa_mod

    def fake_build_team_epa_features(base_games, config=None, client=None):
        return stub_epa_df

    monkeypatch.setattr(epa_mod, "build_team_epa_features", fake_build_team_epa_features)

    cfg = tsp.TeamStatsConfig(
        seasons=[2020],
        use_pbp_features=True,
        rolling_windows=(3,),
    )

    team_df = tsp.build_team_features(config=cfg)
    game_df = tsp.build_game_level_features(team_df=team_df, config=cfg)

    # Ensure diff_off_epa_per_play_rolling_mean_3 exists when pbp features are on
    diff_cols = [c for c in game_df.columns if "diff_off_epa_per_play_rolling_mean" in c]
    assert diff_cols, "Expected diff_off_epa_per_play_rolling_mean_* columns in game_df"

    # Also check defensive EPA diff exists
    diff_def_cols = [
        c for c in game_df.columns if "diff_def_epa_per_play_allowed_rolling_mean" in c
    ]
    assert diff_def_cols, "Expected diff_def_epa_per_play_allowed_rolling_mean_* columns in game_df"