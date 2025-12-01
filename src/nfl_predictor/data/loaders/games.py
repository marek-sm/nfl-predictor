from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from nfl_predictor.config import DATA_CONFIG

try:
    import nfl_data_py as nfl
except ImportError as e:
    raise ImportError(
        "nfl-data-py is required for GameDataLoader.\n"
        "Install with `pip install nfl-data-py` or add it to pyproject.toml."
    ) from e


@dataclass
class GameDataLoaderConfig:
    """
    Configuration for the game-level data loader.

    Attributes:
        seasons: List of NFL seasons to load.
        save_parquet: If True, saves the merged DataFrame to data/raw/.
        include_markets: If True, keep market-related columns (spread, total, moneyline).
    """

    seasons: List[int]
    save_parquet: bool = True
    include_markets: bool = True


class GameDataLoader:
    """
    Load raw NFL game-level data (and associated betting markets)
    from nfl_data_py.

    This is the foundation of the anti-leakage dataset:
    - One row per game
    - Clear 'game_id', 'season', 'week', 'gameday' columns
    - Outcome targets: home_win, total_points
    - Optional market columns (spread, total, moneylines, etc.)
    """

    # Candidate columns to keep from nfl_data_py.import_schedules
    BASE_COLS = [
        "game_id",
        "season",
        "week",
        "gameday",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "result",
    ]

    MARKET_COLS = [
        # spread / total
        "spread_line",
        "total_line",
        # moneylines
        "home_moneyline",
        "away_moneyline",
        # sometimes books/odds are included
        "over_odds",
        "under_odds",
        "home_spread_odds",
        "away_spread_odds",
    ]

    EXTRA_TIME_COLS = [
        "game_time_et",
    ]

    def __init__(self, config: Optional[GameDataLoaderConfig] = None):
        if config is None:
            config = GameDataLoaderConfig(seasons=DATA_CONFIG.default_seasons)
        self.config = config

    def load(self) -> pd.DataFrame:
        """
        Load game + (optional) market data into a canonical DataFrame.

        Returns:
            A DataFrame with:
            - Core identifiers: game_id, season, week
            - Time: gameday (datetime64)
            - Teams: home_team, away_team
            - Scores: home_score, away_score
            - Targets: home_win (int), total_points (int)
            - Optional markets: spread_line, total_line, moneylines, etc.
        """
        schedules = self._load_raw_schedules()
        games = self._build_games_table(schedules)

        if self.config.save_parquet:
            DATA_CONFIG.raw_data_dir.mkdir(parents=True, exist_ok=True)
            start_season = min(self.config.seasons)
            end_season = max(self.config.seasons)
            path = DATA_CONFIG.raw_data_dir / f"games_{start_season}_{end_season}.parquet"
            games.to_parquet(path, index=False)

        return games

    def _load_raw_schedules(self) -> pd.DataFrame:
        """
        Load raw schedules from nfl_data_py.

        Notes:
            This is written for nfl_data_py >= 0.3.x where
            `import_schedules(seasons)` is available.
        """
        seasons = self.config.seasons
        if not seasons:
            raise ValueError("At least one season must be provided to GameDataLoader.")

        try:
            schedules = nfl.import_schedules(seasons)
        except AttributeError as e:
            raise RuntimeError(
                "nfl_data_py.import_schedules is not available. "
                "Check your nfl-data-py version and update this loader accordingly."
            ) from e

        if not isinstance(schedules, pd.DataFrame):
            raise TypeError("nfl.import_schedules did not return a pandas DataFrame.")

        return schedules

    def _build_games_table(self, schedules: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and enrich the schedules DataFrame to create
        the canonical game-level table.
        """
        df = schedules.copy()

        # Ensure we have a gameday column as datetime
        if "gameday" in df.columns:
            df["gameday"] = pd.to_datetime(df["gameday"])
        elif "game_date" in df.columns:
            df["gameday"] = pd.to_datetime(df["game_date"])
        else:
            raise KeyError(
                "Could not find a 'gameday' or 'game_date' column in schedules."
            )

        # Build column list: base + markets (optional)
        keep_cols = list(dict.fromkeys(self.BASE_COLS))  # unique preserve order
        if self.config.include_markets:
            keep_cols.extend(self.MARKET_COLS)
        keep_cols.extend(self.EXTRA_TIME_COLS)

        # Filter to existing columns only
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()

        # Rename game_date -> gameday if present
        if "game_date" in df.columns and "gameday" not in df.columns:
            df = df.rename(columns={"game_date": "gameday"})

        # Targets
        self._add_outcome_targets_inplace(df)

        # Final column order: identifiers, time, teams, scores, targets, markets...
        col_order = [
            c
            for c in [
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
            ]
            if c in df.columns
        ]
        # append everything else after
        other_cols = [c for c in df.columns if c not in col_order]
        df = df[col_order + other_cols]

        return df

    @staticmethod
    def _add_outcome_targets_inplace(df: pd.DataFrame) -> None:
        """Add home_win and total_points columns to the DataFrame in-place."""
        required = ["home_score", "away_score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing score columns required for targets: {missing}")

        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["total_points"] = (df["home_score"] + df["away_score"]).astype(int)
