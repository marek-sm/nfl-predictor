from __future__ import annotations

import pandas as pd

from nfl_predictor.data.feature_engineering.epa_features import (
    EPAFeaturesConfig,
    build_team_epa_features,
)


class StubPbpClient:
    """
    Minimal stub client that mimics NflreadpyClient for unit tests.
    """

    def __init__(self, pbp: pd.DataFrame) -> None:
        self._pbp = pbp

    def load_pbp(self, seasons=None) -> pd.DataFrame:  # signature matches real client
        return self._pbp


def _make_base_games() -> pd.DataFrame:
    # Only fields we actually need here: season, game_id
    return pd.DataFrame(
        {
            "season": [2020, 2020],
            "game_id": ["2020_01_BUF_NE", "2020_01_KC_DEN"],
        }
    )


def _make_stub_pbp() -> pd.DataFrame:
    # Two games, a few plays each, mix of pass/rush + EPA
    rows = [
        # BUF offense vs NE
        dict(
            season=2020,
            game_id="2020_01_BUF_NE",
            posteam="BUF",
            defteam="NE",
            epa=0.5,
            yards_gained=12,
            play_type="pass",
            success=True,
        ),
        dict(
            season=2020,
            game_id="2020_01_BUF_NE",
            posteam="BUF",
            defteam="NE",
            epa=-0.1,
            yards_gained=3,
            play_type="run",
            success=False,
        ),
        # NE offense vs BUF
        dict(
            season=2020,
            game_id="2020_01_BUF_NE",
            posteam="NE",
            defteam="BUF",
            epa=-0.3,
            yards_gained=5,
            play_type="run",
            success=False,
        ),
        # Second game KC vs DEN
        dict(
            season=2020,
            game_id="2020_01_KC_DEN",
            posteam="KC",
            defteam="DEN",
            epa=0.8,
            yards_gained=25,
            play_type="pass",
            success=True,
        ),
    ]
    return pd.DataFrame(rows)


def test_build_team_epa_features_offense_defense_basic():
    base_games = _make_base_games()
    pbp = _make_stub_pbp()
    client = StubPbpClient(pbp)

    epa_cfg = EPAFeaturesConfig(seasons=[2020])
    team_epa = build_team_epa_features(
        base_games=base_games,
        config=epa_cfg,
        client=client,
    )

    # We should have rows for BUF, NE, KC, DEN in 2020
    teams = set(zip(team_epa["season"], team_epa["team"]))
    assert (2020, "BUF") in teams
    assert (2020, "NE") in teams
    assert (2020, "KC") in teams
    assert (2020, "DEN") in teams

    required_cols = [
        "plays_off",
        "plays_def",
        "off_epa_per_play",
        "off_success_rate",
        "off_explosive_play_rate",
        "off_pass_rate",
        "off_rush_rate",
        "off_pass_epa_per_play",
        "off_rush_epa_per_play",
        "def_epa_per_play_allowed",
        "def_success_rate_allowed",
        "def_explosive_play_rate_allowed",
        "def_pass_epa_per_play_allowed",
        "def_rush_epa_per_play_allowed",
    ]
    for col in required_cols:
        assert col in team_epa.columns, f"Missing EPA column: {col}"

    # Simple sanity: plays_off and plays_def are non-negative ints
    assert (team_epa["plays_off"] >= 0).all()
    assert (team_epa["plays_def"] >= 0).all()
    assert team_epa["plays_off"].dtype == int
    assert team_epa["plays_def"].dtype == int

    # EPA per play should be bounded by min/max play EPA in stub data,
    # for teams that actually have offensive plays (non-null EPA).
    min_epa = pbp["epa"].min()
    max_epa = pbp["epa"].max()

    off_epa = team_epa["off_epa_per_play"].dropna()
    def_epa = team_epa["def_epa_per_play_allowed"].dropna()

    assert (off_epa >= min_epa).all()
    assert (off_epa <= max_epa).all()
    assert (def_epa >= min_epa).all()
    assert (def_epa <= max_epa).all()