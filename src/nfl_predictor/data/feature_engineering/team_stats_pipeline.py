from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_CONFIG
from nfl_predictor.data.preprocessing.base_dataset import (
    BaseDatasetConfig,
    build_base_dataset,
)
from .rolling_features import RollingSpec, add_rolling_features


@dataclass
class TeamStatsConfig:
    """
    Configuration for building team-level and game-level features (Step 3).

    Attributes
    ----------
    seasons:
        Seasons to include. If None, defaults to DATA_CONFIG.default_seasons.
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
        - Markets and preseason handling are fixed for Step 3 here; the
          FeatureBuilder orchestrator can also construct a BaseDatasetConfig
          and pass base_games explicitly, which then becomes the single
          source of truth.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_season_range_from_df(df: pd.DataFrame) -> tuple[int, int]:
    if "season" not in df.columns or df["season"].empty:
        raise ValueError("Cannot infer season range: 'season' column missing or empty.")
    seasons = df["season"].unique()
    return int(seasons.min()), int(seasons.max())


def _american_odds_to_prob(odds: pd.Series) -> pd.Series:
    """
    Convert American moneyline odds to implied probabilities.

    Formula
    -------
    For positive odds (e.g., +150):
        p = 100 / (odds + 100)
    For negative odds (e.g., -200):
        p = -odds / (-odds + 100)
    """
    s = odds.astype("float64")
    prob = pd.Series(index=s.index, dtype="float64")

    pos = s > 0
    neg = s < 0

    prob[pos] = 100.0 / (s[pos] + 100.0)
    prob[neg] = -s[neg] / (-s[neg] + 100.0)
    # Zero or NaN odds -> NaN probability
    prob[~(pos | neg)] = np.nan
    return prob


def _build_team_long(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert game-level base dataset to team-game long format.

    Output: two rows per game (home + away), including:

        Identifiers
        -----------
        game_id, season, week, gameday, game_index

        Team context
        ------------
        team, opponent, is_home (1/0)

        Game outcomes
        -------------
        team_points_for, team_points_against, point_diff (team perspective),
        team_win (1 if this team won, else 0),
        total_points, spread_line, total_line

        Market-aware primitives
        ------------------------
        spread_from_team_perspective  (home: spread_line, away: -spread_line)
        ats_margin                     (point_diff + spread_from_team_perspective)
        total_vs_line                  (total_points - total_line)
        team_moneyline                 (home or away moneyline for this team)
        implied_prob_ml                (implied probability from moneyline)
        covered_spread                 (1 if ats_margin > 0 else 0)
    """
    required = [
        "game_id",
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
    ]
    missing = [c for c in required if c not in base_df.columns]
    if missing:
        raise KeyError(f"Base dataset missing required columns: {missing}")

    base = base_df.copy()

    common_cols = [
        "game_id",
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
    ]
    if "is_regular_season" in base.columns:
        common_cols.append("is_regular_season")
    if "is_postseason" in base.columns:
        common_cols.append("is_postseason")

    base = base[common_cols].copy()

    # Home rows
    home = base.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["team_points_for"] = home["home_score"]
    home["team_points_against"] = home["away_score"]
    home["spread_from_team_perspective"] = home["spread_line"]
    home["team_moneyline"] = home["home_moneyline"]

    # Away rows
    away = base.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away["team_points_for"] = away["away_score"]
    away["team_points_against"] = away["home_score"]
    away["spread_from_team_perspective"] = -away["spread_line"]
    away["team_moneyline"] = away["away_moneyline"]

    # Concatenate and compute derived fields
    long_df = pd.concat([home, away], axis=0, ignore_index=True)

    long_df["point_diff"] = (
        long_df["team_points_for"] - long_df["team_points_against"]
    )

    # Team-centric win target: 1 if this team won, 0 otherwise
    long_df["team_win"] = long_df.apply(
        lambda row: row["home_win"] if row["is_home"] == 1 else 1 - row["home_win"],
        axis=1,
    )

    # ATS margin: from team perspective
    long_df["ats_margin"] = (
        long_df["point_diff"] + long_df["spread_from_team_perspective"]
    )

    # Total vs closing line (same for both teams)
    long_df["total_vs_line"] = long_df["total_points"] - long_df["total_line"]

    # Market implied probability from moneyline
    long_df["implied_prob_ml"] = _american_odds_to_prob(long_df["team_moneyline"])

    # Covered spread indicator
    long_df["covered_spread"] = (long_df["ats_margin"] > 0).astype(int)

    return long_df


def _add_schedule_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add schedule/rest features to team_df.

    Adds:
        - days_since_last_game
        - is_short_week (<= 4 days rest)
        - is_long_rest (>= 10 days rest)
        - coming_off_bye (>= 13 days rest)
        - games_played_season_to_date (count of prior games in season)
    """
    if "gameday" not in team_df.columns:
        raise KeyError("team_df must contain 'gameday' for schedule features.")

    df = team_df.copy()
    df = df.sort_values(["team", "season", "gameday", "game_index"])

    group = df.groupby(["team", "season"], group_keys=False)

    df["days_since_last_game"] = group["gameday"].diff().dt.days

    # Number of games played BEFORE this one in the season
    df["games_played_season_to_date"] = group.cumcount()

    ds = df["days_since_last_game"]
    df["is_short_week"] = (ds <= 4).fillna(False).astype(int)
    df["is_long_rest"] = (ds >= 10).fillna(False).astype(int)
    df["coming_off_bye"] = (ds >= 13).fillna(False).astype(int)

    return df


def _add_season_agg_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season-to-date aggregation features such as win% to date.

    Adds:
        - season_win_pct_to_date  (expanding mean of team_win using only prior games)
    """
    df = team_df.copy()
    df = df.sort_values(["team", "season", "gameday", "game_index"])

    def _season_win_pct(group: pd.Series) -> pd.Series:
        # Shift first so current game is NOT included in its own aggregate.
        shifted = group.shift(1)
        return shifted.expanding(min_periods=1).mean()

    df["season_win_pct_to_date"] = (
        df.groupby(["team", "season"], group_keys=False)["team_win"].apply(
            _season_win_pct
        )
    )

    return df


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------


def build_team_features(
    config: TeamStatsConfig | None = None,
    base_games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build leak-free team-game level features from the base dataset.

    Steps
    -----
    1. Build or use provided base_games (Step 2 output).
    2. Convert to team-game long format (2 rows per game).
    3. Optionally drop postseason games.
    4. Add schedule/rest features.
    5. Add season-to-date aggregation features.
    6. Add rolling stats (3/5/8-game means & sums, ATS, totals, win rates).
    7. Optionally save to data/features/.
    """
    if config is None:
        config = TeamStatsConfig()

    # Build / load base dataset if not provided
    if base_games is None:
        base_games = build_base_dataset(config.make_base_config())

    df = base_games.copy()

    if not config.include_postseason and "is_postseason" in df.columns:
        df = df[~df["is_postseason"]].copy()

    team_df = _build_team_long(df)

    # Schedule & season aggregates
    team_df = _add_schedule_features(team_df)
    team_df = _add_season_agg_features(team_df)

    # Rolling specs for core team stats
    windows = config.rolling_windows
    min_p = config.min_games_for_rolling

    specs: list[RollingSpec] = [
        RollingSpec(
            col="team_points_for",
            windows=windows,
            stats=("mean", "sum"),
            min_periods=min_p,
            prefix="points_for",
        ),
        RollingSpec(
            col="team_points_against",
            windows=windows,
            stats=("mean", "sum"),
            min_periods=min_p,
            prefix="points_against",
        ),
        RollingSpec(
            col="point_diff",
            windows=windows,
            stats=("mean",),
            min_periods=min_p,
            prefix="point_diff",
        ),
        RollingSpec(
            col="ats_margin",
            windows=windows,
            stats=("mean",),
            min_periods=min_p,
            prefix="ats_margin",
        ),
        RollingSpec(
            col="total_vs_line",
            windows=windows,
            stats=("mean",),
            min_periods=min_p,
            prefix="total_vs_line",
        ),
        RollingSpec(
            col="team_win",
            windows=windows,
            stats=("mean",),
            min_periods=min_p,
            prefix="team_win_rate",
        ),
        RollingSpec(
            col="covered_spread",
            windows=windows,
            stats=("mean",),
            min_periods=min_p,
            prefix="covered_spread_rate",
        ),
    ]

    team_df = add_rolling_features(
        team_df,
        group_cols=["team", "season"],
        time_col="game_index",
        specs=specs,
    )

    # Optional saving
    if config.save_team_level:
        start_season, end_season = _infer_season_range_from_df(team_df)
        DATA_CONFIG.features_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            DATA_CONFIG.features_dir
            / f"team_features_{start_season}_{end_season}.parquet"
        )
        team_df.to_parquet(out_path, index=False)

    return team_df


def build_game_level_features(
    team_df: pd.DataFrame,
    base_games: pd.DataFrame | None = None,
    config: TeamStatsConfig | None = None,
) -> pd.DataFrame:
    """
    Merge team-game features back into a single-row-per-game modeling matrix.

    For each game, we attach:
        - home_* features (home team's team-level stats)
        - away_* features (away team's team-level stats)
        - matchup diff features for key metrics
        - modeling targets for moneyline and totals
    """
    if config is None:
        config = TeamStatsConfig()

    if base_games is None:
        base_games = build_base_dataset(config.make_base_config())

    df_team = team_df.copy()

    required_cols = ["game_id", "team", "is_home"]
    missing = [c for c in required_cols if c not in df_team.columns]
    if missing:
        raise KeyError(f"team_df missing required columns: {missing}")

    # Identify columns we do NOT want to duplicate as features
    # (these will be preserved from base_games or are raw outcomes/labels).
    # NOTE: schedule/season/market context (e.g. implied_prob_ml, days_since_last_game,
    # season_win_pct_to_date) are intentionally treated as FEATURES and thus
    # excluded from identity_cols so they become home_* / away_* columns.
    identity_cols = {
        "game_id",
        "team",
        "opponent",
        "is_home",
        "season",
        "week",
        "gameday",
        "game_index",
        "home_win",
        "total_points",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "team_points_for",
        "team_points_against",
        "point_diff",
        "team_win",
        "ats_margin",
        "total_vs_line",
        "team_moneyline",
        "is_regular_season",
        "is_postseason",
    } & set(df_team.columns)

    feature_cols = [c for c in df_team.columns if c not in identity_cols]

    home_df = df_team[df_team["is_home"] == 1]
    away_df = df_team[df_team["is_home"] == 0]

    if home_df["game_id"].duplicated().any():
        raise ValueError("Multiple home rows per game_id in team_df.")
    if away_df["game_id"].duplicated().any():
        raise ValueError("Multiple away rows per game_id in team_df.")

    home_features = home_df[["game_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    )
    away_features = away_df[["game_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    )

    merged = base_games.merge(home_features, on="game_id", how="left").merge(
        away_features, on="game_id", how="left"
    )

    # ------------------------------------------------------------------
    # Matchup diff features (home minus away) for core metrics
    # ------------------------------------------------------------------

    def _maybe_add_diff(
        home_col: str,
        away_col: str,
        diff_name: str,
    ) -> None:
        if home_col in merged.columns and away_col in merged.columns:
            merged[diff_name] = merged[home_col] - merged[away_col]

    # Points for rolling means
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

    # Point diff rolling means
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

    # Season win% to date
    _maybe_add_diff(
        "home_season_win_pct_to_date",
        "away_season_win_pct_to_date",
        "diff_season_win_pct_to_date",
    )

    # Rest / schedule diffs
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

    # Market-implied probability diffs
    _maybe_add_diff(
        "home_implied_prob_ml",
        "away_implied_prob_ml",
        "diff_implied_prob_ml",
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

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    merged["target_home_win"] = merged["home_win"].astype(int)
    merged["target_total_points"] = merged["total_points"]
    merged["target_total_over"] = (
        merged["total_points"] > merged["total_line"]
    ).astype(int)

    # Optional saving
    if config.save_game_level:
        start_season, end_season = _infer_season_range_from_df(merged)
        DATA_CONFIG.features_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            DATA_CONFIG.features_dir
            / f"game_features_{start_season}_{end_season}.parquet"
        )
        merged.to_parquet(out_path, index=False)

    return merged