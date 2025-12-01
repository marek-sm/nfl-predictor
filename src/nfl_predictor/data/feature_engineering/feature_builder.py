from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nfl_predictor.config import DATA_CONFIG
from nfl_predictor.data.preprocessing.base_dataset import BaseDatasetConfig, build_base_dataset

from .team_stats_pipeline import (
    TeamStatsConfig,
    build_game_level_features,
    build_team_features,
)


@dataclass
class FeatureBuilderConfig:
    """
    Configuration for the high-level feature builder (Step 3 orchestrator).

    This config acts as the single source of truth for:
        - seasons included in the dataset,
        - postseason inclusion,
        - whether market columns are loaded,
        - whether preseason is dropped.

    When you call `FeatureBuilder.build_features()`, a compatible
    BaseDatasetConfig and TeamStatsConfig are created under the hood
    so that Step 2 (base dataset) and Step 3 (feature engineering)
    stay in sync.
    """

    seasons: list[int] | None = None
    include_postseason: bool = True
    include_markets: bool = True
    drop_preseason: bool = True
    save_intermediate: bool = True


class FeatureBuilder:
    """
    Orchestrates Step 2 (base dataset) and Step 3 (feature engineering)
    into a single call that returns a one-row-per-game feature matrix.

    Typical usage
    -------------
    fb_config = FeatureBuilderConfig(seasons=[2021, 2022])
    fb = FeatureBuilder(fb_config)
    features_df = fb.build_features()

    Notes
    -----
    - When using this class, FeatureBuilderConfig is the canonical
      configuration for the base dataset. Internally, compatible
      BaseDatasetConfig and TeamStatsConfig instances are derived
      so that Step 2 and Step 3 remain consistent.
    """

    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config

    def _make_base_config(self) -> BaseDatasetConfig:
        """
        Construct the BaseDatasetConfig used by Step 2 when running
        FeatureBuilder end-to-end.

        This method defines the canonical configuration for the base
        dataset whenever FeatureBuilder is used; TeamStatsConfig
        is derived from this to ensure consistency.
        """
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
        """
        Construct a TeamStatsConfig that is consistent with this
        FeatureBuilder's BaseDatasetConfig.

        NOTE: When using FeatureBuilder, you should not construct
        TeamStatsConfig manually; this method ensures that the
        seasons and postseason settings line up with the base dataset.
        """
        return TeamStatsConfig(
            seasons=self.config.seasons,
            include_postseason=self.config.include_postseason,
            save_team_level=self.config.save_intermediate,
            save_game_level=self.config.save_intermediate,
        )

    def build_features(self, base_games: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Build game-level features (one row per game) for modeling.

        Parameters
        ----------
        base_games:
            Optional pre-built base dataset from Step 2. If None, this
            method will call `build_base_dataset` using the canonical
            BaseDatasetConfig derived from this FeatureBuilderConfig.

        Returns
        -------
        pd.DataFrame
            One row per game with:
                - base dataset columns,
                - home_* and away_* features,
                - diff_* matchup features,
                - modeling targets.
        """
        base_cfg = self._make_base_config()
        team_cfg = self._make_team_config()

        if base_games is None:
            base_games = build_base_dataset(base_cfg)

        team_df = build_team_features(config=team_cfg, base_games=base_games)
        game_df = build_game_level_features(
            team_df=team_df,
            base_games=base_games,
            config=team_cfg,
        )
        return game_df