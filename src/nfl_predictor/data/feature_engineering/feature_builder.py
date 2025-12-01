from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nfl_predictor.config import DATA_CONFIG
from nfl_predictor.data.preprocessing.base_dataset import (
    BaseDatasetConfig,
    build_base_dataset,
)
from nfl_predictor.data.feature_engineering.team_stats_pipeline import (
    TeamStatsConfig,
    build_game_level_features,
    build_team_features,
)


@dataclass
class FeatureBuilderConfig:
    """
    Configuration for the high-level feature builder (Step 3 orchestrator).

    Attributes
    ----------
    seasons:
        Seasons to include. If None, uses DATA_CONFIG.default_seasons.
    include_postseason:
        Whether to include postseason games in the modeling matrix.
    include_markets:
        Whether the base dataset should retain market columns.
    drop_preseason:
        Whether to drop preseason games when building the base dataset.
    save_intermediate:
        If True, save team-level and game-level feature parquet files.
    """

    seasons: list[int] | None = None
    include_postseason: bool = True
    include_markets: bool = True
    drop_preseason: bool = True
    save_intermediate: bool = True


class FeatureBuilder:
    """
    End-to-end orchestration for building modeling features (Step 3).

    Typical usage
    -------------
        config = FeatureBuilderConfig(seasons=list(range(2015, 2024)))
        builder = FeatureBuilder(config)
        features_df = builder.build_features()

    This will:
        - Build the base dataset (Step 2).
        - Build team-game long features with rolling & schedule stats.
        - Merge into a game-level modeling matrix with home_/away_ features.
        - Attach matchup differentials and modeling targets.
    """

    def __init__(self, config: FeatureBuilderConfig | None = None) -> None:
        if config is None:
            config = FeatureBuilderConfig()
        self.config = config

    # ------------------------------------------------------------------
    # Helper configuration builders
    # ------------------------------------------------------------------
    def _make_base_config(self) -> BaseDatasetConfig:
        seasons = self.config.seasons
        if seasons is None:
            seasons = DATA_CONFIG.default_seasons

        return BaseDatasetConfig(
            seasons=seasons,
            include_markets=self.config.include_markets,
            drop_preseason=self.config.drop_preseason,
            save_parquet=False,
        )

    def _make_team_config(self) -> TeamStatsConfig:
        # FeatureBuilder is the "single source of truth" when used end-to-end:
        # seasons/postseason choices live here and are passed into TeamStatsConfig.
        return TeamStatsConfig(
            seasons=self.config.seasons,
            include_postseason=self.config.include_postseason,
            save_team_level=self.config.save_intermediate,
            save_game_level=self.config.save_intermediate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_base(self) -> pd.DataFrame:
        """Build (or rebuild) the base dataset (Step 2)."""
        return build_base_dataset(self._make_base_config())

    def build_team_features(
        self,
        base_games: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build team-game level features (two rows per game).

        Parameters
        ----------
        base_games:
            Optional pre-built base dataset. If None, Step 2 will be run.
        """
        team_config = self._make_team_config()
        return build_team_features(config=team_config, base_games=base_games)

    def build_features(
        self,
        base_games: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the full game-level modeling feature matrix.

        Returns
        -------
        pd.DataFrame
            One row per game with:
                - original base dataset columns
                - home_* team rolling/schedule/market features
                - away_* team rolling/schedule/market features
                - diff_* matchup features
                - target_home_win, target_total_points, target_total_over
        """
        if base_games is None:
            base_games = self.build_base()

        team_config = self._make_team_config()
        team_df = build_team_features(config=team_config, base_games=base_games)
        game_features = build_game_level_features(
            team_df=team_df,
            base_games=base_games,
            config=team_config,
        )
        return game_features