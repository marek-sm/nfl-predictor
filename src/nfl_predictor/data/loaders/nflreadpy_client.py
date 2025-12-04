from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from nfl_predictor.config import DATA_CONFIG

try:
    import nflreadpy as nflr
except ImportError as e:
    raise ImportError(
        "nflreadpy is required for nflreadpy-based loaders.\n"
        "Install with `pip install nflreadpy` or add it to pyproject.toml."
    ) from e


@dataclass
class NflreadpyConfig:
    """
    Configuration for nflreadpy-based data loading.

    Attributes
    ----------
    seasons:
        Seasons to load. If None, defaults to DATA_CONFIG.default_seasons.
    """

    seasons: Sequence[int] | None = None

    def resolved_seasons(self) -> list[int]:
        if self.seasons is None:
            return list(DATA_CONFIG.default_seasons or [])
        return list(self.seasons)


class NflreadpyClient:
    """
    Thin wrapper around nflreadpy that returns pandas DataFrames.

    - Uses Polars under the hood (from nflreadpy) and converts via `.to_pandas()`.
    - Central place to handle any future schema quirks or caching.
    """

    def __init__(self, config: NflreadpyConfig | None = None) -> None:
        if config is None:
            config = NflreadpyConfig()
        self.config = config

    # ------------- Core loaders -------------

    def load_pbp(self, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        """
        Load play-by-play data for given seasons via nflreadpy.load_pbp.

        Returns a pandas DataFrame with at least:
        - season, game_id
        - posteam, defteam
        - epa, success, yards_gained
        - pass, rush, play_type, etc.
        """
        seasons = list(seasons) if seasons is not None else self.config.resolved_seasons()
        pl_df = nflr.load_pbp(seasons=seasons)
        try:
            return pl_df.to_pandas()
        except AttributeError as e:
            raise TypeError(
                "nflreadpy.load_pbp did not return a Polars DataFrame as expected. "
                "Check nflreadpy version and docs."
            ) from e

    def load_team_stats(self, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        """
        Load team-level stats (weekly) via nflreadpy.load_team_stats.

        This can be a future alternative to computing everything by hand
        from play-by-play, for things like success rate, EPA, etc.
        """
        seasons = list(seasons) if seasons is not None else self.config.resolved_seasons()
        pl_df = nflr.load_team_stats(seasons=seasons, summary_level="week")
        return pl_df.to_pandas()

    def load_schedules(self, seasons: Iterable[int] | None = None) -> pd.DataFrame:
        """
        Load schedule data via nflreadpy.load_schedules.

        NOTE: Your existing GameDataLoader currently uses nfl_data_py.import_schedules.
        This helper is here so we can migrate cleanly later if desired.
        """
        seasons = list(seasons) if seasons is not None else self.config.resolved_seasons()
        pl_df = nflr.load_schedules(seasons=seasons)
        return pl_df.to_pandas()