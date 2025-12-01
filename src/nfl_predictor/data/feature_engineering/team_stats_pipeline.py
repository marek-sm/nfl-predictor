from __future__ import annotations

"""
Team-level and game-level feature engineering (Step 3).

This module assumes that:
- The input `base_games` DataFrame is produced by
  `nfl_predictor.data.preprocessing.base_dataset.build_base_dataset`.
- `game_index` is a global, monotonically increasing index that reflects
  chronological order across the dataset.
- Duplicates and data quality issues have already been handled by Step 2.

Step 3 never mutates or redefines those base semantics; it only adds
team-level and game-level features on top.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_CONFIG
from nfl_predictor.data.preprocessing.base_dataset import BaseDatasetConfig, build_base_dataset

from .rolling_features import RollingSpec, add_rolling_features


@dataclass
class TeamStatsConfig:
    """
    Configuration for building team-level and game-level features (Step 3).

    Attributes
    ----------
    seasons:
        Seasons to include. If None, defaults to DATA_CONFIG.default_seasons.
        NOTE: When used via FeatureBuilder, the seasons in FeatureBuilderConfig
        are the canonical source of truth and are passed through here.
    include_postseason:
        If False, postseason games will be dropped before feature building.
    rolling_windows:
        Rolling window sizes (in number of games) used for team stats.
    min_games_for_rolling:
        Minimum number of prior games needed to compute rolling stats.
    save_team_level:
        If True, save team-game features to DATA_CONFIG.features_dir.
    save_game_level:
        If True, save game-level features to DATA_CONFIG.features_dir.

    Usage notes
    -----------
    - In typical end-to-end usage, you should not construct this
      manually; instead use FeatureBuilder, which creates a consistent
      TeamStatsConfig based on its own FeatureBuilderConfig.
    - `make_base_config()` is primarily a convenience for standalone
      / testing usage where you want to call build_team_features
      directly.
    """

    seasons: list[int] | None = None
    include_postseason: bool = True
    rolling_windows: tuple[int, ...] = (3, 5, 8)
    min_games_for_rolling: int = 1
    save_team_level: bool = False
    save_game_level: bool = False

    def make_base_config(self) -> BaseDatasetConfig:
        """
        Construct the BaseDatasetConfig for Step 2.

        NOTE:
        - If seasons is None, DATA_CONFIG.default_seasons is used.
        - This helper is intended for standalone usage of Step 3
          (e.g., in tests or ad-hoc scripts). When you use the
          FeatureBuilder end-to-end, its FeatureBuilderConfig
          determines the BaseDatasetConfig instead.
        """
        seasons = self.seasons
        if seasons is None:
            seasons = DATA_CONFIG.default_seasons
        return BaseDatasetConfig(
            seasons=seasons,
            include_markets=True,
            drop_preseason=True,
            save_parquet=False,
        )


def _american_odds_to_prob(odds: float | int | None) -> float | None:
    """
    Convert American moneyline odds to implied probability (including vig).

    Returns None for NaN / missing odds.
    """
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _build_team_long_from_base(base_games: pd.DataFrame) -> pd.DataFrame:
    """
    Expand one-row-per-game base dataset into two-row-per-game team-long format.

    Adds:
    - team, opponent, is_home
    - points_for, points_against, point_diff
    - team_win (1/0)
    - ats_margin, covered_spread
    - total_vs_line
    - team_moneyline, implied_prob_ml
    """
    # home rows
    home = base_games.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = True
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["team_win"] = home["home_win"]
    home["team_moneyline"] = home["home_moneyline"]

    # away rows
    away = base_games.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = False
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["team_win"] = 1 - away["home_win"]
    away["team_moneyline"] = away["away_moneyline"]

    team_df = pd.concat([home, away], ignore_index=True)

    # basic derived stats
    team_df["point_diff"] = team_df["points_for"] - team_df["points_against"]

    # spread from the perspective of the team row
    # positive spread_line means home is favored by that many points.
    # For away teams, the effective spread is -spread_line.
    team_df["spread_from_team"] = np.where(
        team_df["is_home"],
        team_df["spread_line"],
        -team_df["spread_line"],
    )
    team_df["ats_margin"] = team_df["point_diff"] + team_df["spread_from_team"]
    team_df["covered_spread"] = (team_df["ats_margin"] > 0).astype(float)

    # totals
    team_df["total_vs_line"] = team_df["total_points"] - team_df["total_line"]

    # implied probability from team_moneyline
    team_df["implied_prob_ml"] = team_df["team_moneyline"].map(_american_odds_to_prob)

    return team_df


def _add_schedule_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule and rest features grouped by [team, season].

    Adds:
    - days_since_last_game
    - games_played_season_to_date
    - is_short_week (<= 4 days)
    - is_long_rest (>= 10 days)
    - coming_off_bye (>= 13 days)
    - season_win_pct_to_date (shifted expanding mean of team_win)
    """
    df = team_df.sort_values(["team", "season", "gameday"]).copy()

    # days since last game (within same team/season)
    df["days_since_last_game"] = (
        df.groupby(["team", "season"])["gameday"]
        .diff()
        .dt.days
    )

    # games played so far this season (0-based, then add 1 if you want game number)
    df["games_played_season_to_date"] = (
        df.groupby(["team", "season"]).cumcount()
    )

    # schedule flags
    df["is_short_week"] = (df["days_since_last_game"] <= 4).astype(float)
    df["is_long_rest"] = (df["days_since_last_game"] >= 10).astype(float)
    df["coming_off_bye"] = (df["days_since_last_game"] >= 13).astype(float)

    # season win pct to date (expanding mean of prior team_win)
    df["season_win_pct_to_date"] = (
        df.groupby(["team", "season"])["team_win"]
        .shift(1)
        .expanding()
        .mean()
    )

    return df


def _add_rolling_stats(team_df: pd.DataFrame, config: TeamStatsConfig) -> pd.DataFrame:
    """
    Add leak-free rolling stats using RollingSpec + add_rolling_features.
    """
    windows = config.rolling_windows
    min_periods = config.min_games_for_rolling

    specs: list[RollingSpec] = [
        RollingSpec(
            col="points_for",
            windows=windows,
            stats=("mean", "sum"),
            min_periods=min_periods,
        ),
        RollingSpec(
            col="points_against",
            windows=windows,
            stats=("mean", "sum"),
            min_periods=min_periods,
        ),
        RollingSpec(
            col="point_diff",
            windows=windows,
            stats=("mean",),
            min_periods=min_periods,
        ),
        RollingSpec(
            col="ats_margin",
            windows=windows,
            stats=("mean",),
            min_periods=min_periods,
        ),
        RollingSpec(
            col="total_vs_line",
            windows=windows,
            stats=("mean",),
            min_periods=min_periods,
        ),
        RollingSpec(
            col="team_win",
            windows=windows,
            stats=("mean",),
            min_periods=min_periods,
            prefix="team_win_rate",
        ),
        RollingSpec(
            col="covered_spread",
            windows=windows,
            stats=("mean",),
            min_periods=min_periods,
            prefix="covered_spread_rate",
        ),
    ]

    rolled = add_rolling_features(
        df=team_df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=specs,
    )
    return rolled


def build_team_features(
    config: TeamStatsConfig | None = None,
    base_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build leak-free team-game level features from the base dataset.

    Parameters
    ----------
    config:
        TeamStatsConfig controlling rolling windows, seasons, and
        saving behavior.
    base_games:
        Optional pre-built base dataset from Step 2. If None, this
        function will call `build_base_dataset(config.make_base_config())`.
        In both cases, it is assumed that `base_games` has already:
            - been filtered to completed games only,
            - had duplicates removed,
            - been assigned a chronological `game_index`.
    """
    if config is None:
        config = TeamStatsConfig()

    if base_games is None:
        base_cfg = config.make_base_config()
        base_games = build_base_dataset(base_cfg)

    df = base_games.copy()

    if not config.include_postseason:
        df = df[df["is_regular_season"]].copy()

    team_df = _build_team_long_from_base(df)
    team_df = _add_schedule_features(team_df)
    team_df = _add_rolling_stats(team_df, config)

    if config.save_team_level:
        seasons = config.seasons or DATA_CONFIG.default_seasons
        out_path = DATA_CONFIG.features_dir / f"team_features_{min(seasons)}_{max(seasons)}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        team_df.to_parquet(out_path)

    return team_df


def build_game_level_features(
    team_df: pd.DataFrame,
    base_games: pd.DataFrame | None = None,
    config: TeamStatsConfig | None = None,
) -> pd.DataFrame:
    """
    Merge team-game features back into a single-row-per-game modeling matrix.

    Parameters
    ----------
    team_df:
        Output of `build_team_features`, containing two rows per game
        (home + away) with rolling, schedule, and market-aware features.
    base_games:
        The base dataset from Step 2. If provided, it should be the
        same DataFrame (or a filtered view of it) that was used to
        build `team_df`. If None, it will be rebuilt using
        `config.make_base_config()`.
    config:
        TeamStatsConfig instance; must match whatever was used to
        create `team_df` if `team_df` was provided externally.

    Returns
    -------
    pd.DataFrame
        One row per game with:
            - original base dataset columns,
            - home_* and away_* features,
            - diff_* matchup features,
            - modeling targets.
    """
    if config is None:
        config = TeamStatsConfig()

    if base_games is None:
        base_cfg = config.make_base_config()
        base_games = build_base_dataset(base_cfg)

    # identity columns that should not get home_/away_ prefixes
    identity_cols = {
        "game_id",
        "team",
        "opponent",
        "is_home",
        "season",
        "week",
        "gameday",
        "game_index",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "total_points",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "is_regular_season",
        "is_postseason",
    } & set(team_df.columns)

    feature_cols = sorted(col for col in team_df.columns if col not in identity_cols)

    home_rows = team_df[team_df["is_home"]].copy()
    away_rows = team_df[~team_df["is_home"]].copy()

    home_renamed = home_rows[["game_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away_renamed = away_rows[["game_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )

    game_df = base_games.merge(home_renamed, on="game_id", how="left")
    game_df = game_df.merge(away_renamed, on="game_id", how="left")

    # matchup diff features for key stats
    def _maybe_add_diff(col_home: str, col_away: str, col_out: str) -> None:
        if col_home in game_df.columns and col_away in game_df.columns:
            game_df[col_out] = game_df[col_home] - game_df[col_away]

    # points_for rolling means
    _maybe_add_diff(
        "home_points_for_rolling_mean_3",
        "away_points_for_rolling_mean_3",
        "diff_points_for_rolling_mean_3",
    )
    _maybe_add_diff(
        "home_points_for_rolling_mean_5",
        "away_points_for_rolling_mean_5",
        "diff_points_for_rolling_mean_5",
    )

    # point_diff rolling means
    _maybe_add_diff(
        "home_point_diff_rolling_mean_3",
        "away_point_diff_rolling_mean_3",
        "diff_point_diff_rolling_mean_3",
    )
    _maybe_add_diff(
        "home_point_diff_rolling_mean_5",
        "away_point_diff_rolling_mean_5",
        "diff_point_diff_rolling_mean_5",
    )

    # season win pct to date
    _maybe_add_diff(
        "home_season_win_pct_to_date",
        "away_season_win_pct_to_date",
        "diff_season_win_pct_to_date",
    )

    # ATS margin rolling means
    _maybe_add_diff(
        "home_ats_margin_rolling_mean_3",
        "away_ats_margin_rolling_mean_3",
        "diff_ats_margin_rolling_mean_3",
    )
    _maybe_add_diff(
        "home_ats_margin_rolling_mean_5",
        "away_ats_margin_rolling_mean_5",
        "diff_ats_margin_rolling_mean_5",
    )

    # covered spread rate rolling means (new but consistent)
    _maybe_add_diff(
        "home_covered_spread_rate_rolling_mean_3",
        "away_covered_spread_rate_rolling_mean_3",
        "diff_covered_spread_rate_rolling_mean_3",
    )
    _maybe_add_diff(
        "home_covered_spread_rate_rolling_mean_5",
        "away_covered_spread_rate_rolling_mean_5",
        "diff_covered_spread_rate_rolling_mean_5",
    )

    # schedule & market context
    _maybe_add_diff(
        "home_days_since_last_game",
        "away_days_since_last_game",
        "diff_days_since_last_game",
    )
    _maybe_add_diff(
        "home_games_played_season_to_date",
        "away_games_played_season_to_date",
        "diff_games_played_season_to_date",
    )
    _maybe_add_diff(
        "home_implied_prob_ml",
        "away_implied_prob_ml",
        "diff_implied_prob_ml",
    )

    # targets
    game_df["target_home_win"] = game_df["home_win"]
    game_df["target_total_points"] = game_df["total_points"]
    game_df["target_total_over"] = (game_df["total_points"] > game_df["total_line"]).astype(int)

    if config.save_game_level:
        seasons = config.seasons or DATA_CONFIG.default_seasons
        out_path = DATA_CONFIG.features_dir / f"game_features_{min(seasons)}_{max(seasons)}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        game_df.to_parquet(out_path)

    return game_df