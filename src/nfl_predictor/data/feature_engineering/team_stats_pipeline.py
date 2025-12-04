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

from nfl_predictor.data.feature_engineering.epa_features import (
    build_team_epa_features,
    EPAFeaturesConfig,
)

from .rolling_features import RollingSpec, add_rolling_features
from .epa_features import build_team_epa_features, EPAFeaturesConfig


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
    use_elo:
        Whether to compute and include Elo-style team ratings.
    elo_base:
        Initial Elo rating for each team at the start of a season.
    elo_k:
        K-factor controlling the magnitude of Elo updates per game.
    elo_home_field_advantage:
        Home-field advantage in Elo points added to the home team's rating
        when computing expected results.
    """

    seasons: list[int] | None = None
    include_postseason: bool = True
    rolling_windows: tuple[int, ...] = (3, 5, 8)
    min_games_for_rolling: int = 1
    save_team_level: bool = False
    save_game_level: bool = False

    # Elo configuration
    use_elo: bool = True
    elo_base: float = 1500.0
    elo_k: float = 20.0
    elo_home_field_advantage: float = 55.0

    # nflreadpy / pbp features toggle
    use_pbp_features: bool = False
    min_games_for_rolling_epa: int = 3

    normalize_epa_by_season: bool = False

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


def _add_elo_ratings(base_games: pd.DataFrame, config: TeamStatsConfig) -> pd.DataFrame:
    """
    Add leak-free pre-game Elo ratings for home and away teams.

    Elo is computed separately for each season, iterating games in
    chronological order using `game_index`. For each game:

    - home_elo_pre / away_elo_pre are the Elo ratings BEFORE the game.
    - Ratings are then updated based on the game result.

    This guarantees that for game i, Elo only depends on games < i
    within the same season (no future information).
    """
    df = base_games.copy()
    df["home_elo_pre"] = np.nan
    df["away_elo_pre"] = np.nan

    base_rating = float(config.elo_base)
    k = float(config.elo_k)
    hfa = float(config.elo_home_field_advantage)

    for season in sorted(df["season"].unique()):
        mask = df["season"] == season
        season_games = df.loc[mask].sort_values("game_index")

        ratings: dict[str, float] = {}

        for idx, row in season_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            r_home = ratings.get(home, base_rating)
            r_away = ratings.get(away, base_rating)

            # store pre-game Elo
            df.at[idx, "home_elo_pre"] = r_home
            df.at[idx, "away_elo_pre"] = r_away

            # expected home win probability using logistic Elo
            diff = (r_home + hfa) - r_away
            exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            # actual result
            s_home = float(row["home_win"])  # 1.0 or 0.0

            change = k * (s_home - exp_home)
            ratings[home] = r_home + change
            ratings[away] = r_away - change

    return df


def _build_team_long_from_base(base_games: pd.DataFrame) -> pd.DataFrame:
    """
    Expand one-row-per-game base dataset into two-row-per-game team-long format.

    Adds:
    - team, opponent, is_home
    - points_for, points_against, point_diff
    - team_win (1/0)
    - ats_margin, covered_spread
    - total_vs_line
    - implied_prob_ml from team_moneyline
    - (if available) Elo-based features:
        * elo  (team pre-game Elo)
        * opponent_elo
        * elo_diff (team elo - opponent elo)
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

    # If Elo has been computed at game level (home_elo_pre / away_elo_pre),
    # propagate to team perspective.
    if "home_elo_pre" in base_games.columns and "away_elo_pre" in base_games.columns:
        home["elo"] = home["home_elo_pre"]
        home["opponent_elo"] = home["away_elo_pre"]

        away["elo"] = away["away_elo_pre"]
        away["opponent_elo"] = away["home_elo_pre"]

    team_df = pd.concat([home, away], ignore_index=True)

    # basic derived stats
    team_df["point_diff"] = team_df["points_for"] - team_df["points_against"]

    # spread from the perspective of the team row
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

    # Elo differentials at team level (if available)
    if "elo" in team_df.columns and "opponent_elo" in team_df.columns:
        team_df["elo_diff"] = team_df["elo"] - team_df["opponent_elo"]

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
    # Ensure correct chronological order within each team/season
    df = team_df.sort_values(["team", "season", "gameday", "game_index"]).copy()

    group = df.groupby(["team", "season"], group_keys=False)

    # Days since last game (within same team/season)
    df["days_since_last_game"] = group["gameday"].diff().dt.days

    # Games played so far this season (0-based count of prior games)
    df["games_played_season_to_date"] = group.cumcount()

    # Schedule flags
    ds = df["days_since_last_game"]
    df["is_short_week"] = (ds <= 4).astype(float)
    df["is_long_rest"] = (ds >= 10).astype(float)
    df["coming_off_bye"] = (ds >= 13).astype(float)

    # Season win% to date (expanding mean of prior team_win only, per team/season)
    def _season_win_pct(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)  # drop current game from its own aggregate
        return shifted.expanding(min_periods=1).mean()

    df["season_win_pct_to_date"] = group["team_win"].apply(_season_win_pct)

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

    # Optional EPA / pbp-derived rolling features
    if config.use_pbp_features:
        epa_cols = [
            "off_epa_per_play",
            "off_pass_epa_per_play",
            "off_rush_epa_per_play",
            "off_success_rate",
            "off_explosive_play_rate",
            "def_epa_per_play_allowed",
            "def_pass_epa_per_play_allowed",
            "def_rush_epa_per_play_allowed",
            "def_success_rate_allowed",
            "def_explosive_play_rate_allowed",
        ]
        for col in epa_cols:
            if col in team_df.columns:
                specs.append(
                    RollingSpec(
                        col=col,
                        windows=windows,
                        stats=("mean",),
                        min_periods=min_periods,
                    )
                )

            # OPTIONAL: rolling stats for season-normalized EPA (z-scores)
        if config.use_pbp_features and config.normalize_epa_by_season:
            epa_z_cols = [
                "off_epa_per_play_z",
                "off_pass_epa_per_play_z",
                "off_rush_epa_per_play_z",
                "def_epa_per_play_allowed_z",
                "def_pass_epa_per_play_allowed_z",
                "def_rush_epa_per_play_allowed_z",
            ]
            for col in epa_z_cols:
                if col in team_df.columns:
                    specs.append(
                        RollingSpec(
                            col=col,
                            windows=windows,
                            stats=("mean",),
                            min_periods=min_periods,
                        )
                    )

    rolled = add_rolling_features(
        df=team_df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=specs,
    )

    if config.use_pbp_features:
        min_n = config.min_games_for_rolling_epa

    for col in rolled.columns:
        if "epa" in col and "rolling" in col:
            # compute number of previous games per team-season
            prev_games = (
                rolled.groupby(["team", "season"])["game_index"]
                .rank(method="first") - 1
            )
            rolled.loc[prev_games < min_n, col] = pd.NA

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
        TeamStatsConfig controlling seasons, rolling windows, and saving behavior.
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

    # Optional Elo ratings at game level (become team features downstream).
    if config.use_elo:
        df = _add_elo_ratings(df, config)

    # Base team-long frame (one row per team per game)
    team_df = _build_team_long_from_base(df)

    # Optional PBP / EPA features (team-game level) merged in BEFORE rolling.
    if config.use_pbp_features:
        epa_cfg = EPAFeaturesConfig(seasons=config.seasons)
        epa_df = build_team_epa_features(base_games=df, config=epa_cfg)

        team_df = team_df.merge(
            epa_df,
            on=["season", "team", "game_id"],
            how="left",
            validate="one_to_one",
        )

        # Optional: per-season normalization of EPA (z-scores)
    if config.use_pbp_features and config.normalize_epa_by_season:
        epa_raw_cols = [
            "off_epa_per_play",
            "off_pass_epa_per_play",
            "off_rush_epa_per_play",
            "def_epa_per_play_allowed",
            "def_pass_epa_per_play_allowed",
            "def_rush_epa_per_play_allowed",
        ]

        for col in epa_raw_cols:
            if col in team_df.columns:
                z_col = f"{col}_z"
                team_df[z_col] = (
                    team_df.groupby("season")[col]
                    .transform(
                        lambda x: (x - x.mean())
                        / (x.std(ddof=0) + 1e-8)
                    )
                )

    if "pbp_coverage_flag" not in team_df.columns:
        team_df["pbp_coverage_flag"] = False

    # Schedule / rest features
    team_df = _add_schedule_features(team_df)

    # Rolling stats (including EPA if enabled)
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
        # game-level Elo pre columns (if present) are not used directly
        "home_elo_pre",
        "away_elo_pre",
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

    if config.use_pbp_features:
        if "home_pbp_coverage_flag" in game_df.columns and "away_pbp_coverage_flag" in game_df.columns:
            game_df["home_pbp_coverage"] = (
                game_df["home_pbp_coverage_flag"].fillna(False).astype(bool)
            )
            game_df["away_pbp_coverage"] = (
                game_df["away_pbp_coverage_flag"].fillna(False).astype(bool)
            )
            game_df["both_teams_have_pbp"] = (
                game_df["home_pbp_coverage"] & game_df["away_pbp_coverage"]
            )

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

        # EPA-based matchup differentials (offense & defense)
    if config.use_pbp_features:
        for window in (3, 5):  # mirrors existing pattern of diff windows
            _maybe_add_diff(
                f"home_off_epa_per_play_rolling_mean_{window}",
                f"away_off_epa_per_play_rolling_mean_{window}",
                f"diff_off_epa_per_play_rolling_mean_{window}",
            )
            _maybe_add_diff(
                f"home_def_epa_per_play_allowed_rolling_mean_{window}",
                f"away_def_epa_per_play_allowed_rolling_mean_{window}",
                f"diff_def_epa_per_play_allowed_rolling_mean_{window}",
            )
            _maybe_add_diff(
                f"home_off_success_rate_rolling_mean_{window}",
                f"away_off_success_rate_rolling_mean_{window}",
                f"diff_off_success_rate_rolling_mean_{window}",
            )
            _maybe_add_diff(
                f"home_def_success_rate_allowed_rolling_mean_{window}",
                f"away_def_success_rate_allowed_rolling_mean_{window}",
                f"diff_def_success_rate_allowed_rolling_mean_{window}",
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

    # Elo differential at game level
    _maybe_add_diff("home_elo", "away_elo", "diff_elo")


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