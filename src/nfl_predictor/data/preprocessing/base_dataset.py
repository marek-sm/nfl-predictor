from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from nfl_predictor.config import DATA_CONFIG
from nfl_predictor.data.loaders.games import GameDataLoader, GameDataLoaderConfig


@dataclass
class BaseDatasetConfig:
    """
    Configuration for building the base modeling dataset.

    This dataset is the canonical, leak-free table that model training
    and feature engineering will build on.

    Attributes:
        seasons: List of seasons to include in the dataset.
        include_markets: Whether to keep spread/total/moneyline columns.
        drop_preseason: If True, drop obvious preseason games.
        save_parquet: If True, save the resulting dataset to data/processed/.
        filename: Optional custom filename for the saved dataset.
    """

    seasons: List[int]
    include_markets: bool = True
    drop_preseason: bool = True
    save_parquet: bool = True
    filename: Optional[str] = None


def _validate_raw_games(df: pd.DataFrame, seasons: List[int]) -> None:
    """Basic validation for raw games DataFrame."""
    if df is None or len(df) == 0:
        raise ValueError(f"No games loaded for seasons {seasons}")

    required_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw games data: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["gameday"]):
        raise TypeError(
            f"Column 'gameday' must be datetime64; got dtype {df['gameday'].dtype}"
        )


def _add_season_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_regular_season and is_postseason flags.

    Prefer game_type if available; otherwise fall back on week/season heuristics.
    """
    df = df.copy()

    if "game_type" in df.columns:
        # nfl_data_py typically uses: 'REG', 'POST', 'PRE'
        gt = df["game_type"].astype(str).str.upper()
        df["is_regular_season"] = gt == "REG"
        df["is_postseason"] = gt.isin(["POST", "WC", "DIV", "CON", "SB"])
    elif "week" in df.columns and "season" in df.columns:
        # Season-aware heuristic:
        # For seasons >= 2021: 18-week regular season, postseason starts at week 19
        # For seasons < 2021: 17-week regular season, postseason starts at week 18
        def is_postseason_row(row: pd.Series) -> bool:
            week = row["week"]
            season = row["season"]
            if pd.isna(week):
                return False
            if season >= 2021:
                return week >= 19
            else:
                return week >= 18

        df["is_postseason"] = df.apply(is_postseason_row, axis=1)
        df["is_regular_season"] = (~df["is_postseason"]) & (df["week"] >= 1)
    else:
        # Fallback: assume all are regular season
        df["is_regular_season"] = True
        df["is_postseason"] = False

    return df


def build_base_dataset(config: Optional[BaseDatasetConfig] = None) -> pd.DataFrame:
    """
    Build the base modeling dataset from raw game-level data.

    Steps:
        1. Load raw games via GameDataLoader.
        2. Validate structure (columns, dtypes).
        3. Filter to completed games (both scores non-null).
        4. Add season type flags (regular/postseason).
        5. Optionally drop preseason games.
        6. Sort chronologically by gameday/season/week/game_id.
        7. Assign game_index = 0..N-1 in time order.
        8. Optionally save to data/processed.

    Returns:
        A pandas DataFrame with one row per completed game, containing:
        - identifiers: game_id, season, week
        - time: gameday
        - teams: home_team, away_team
        - scores/targets: home_score, away_score, home_win, total_points
        - optional markets: spread_line, total_line, moneylines, etc.
        - flags: is_regular_season, is_postseason
        - index: game_index (0..N-1 in chronological order)
    """
    if config is None:
        config = BaseDatasetConfig(seasons=DATA_CONFIG.default_seasons)

    # 1. Load raw games (do NOT re-save raw parquet here)
    loader = GameDataLoader(
        GameDataLoaderConfig(
            seasons=config.seasons,
            include_markets=config.include_markets,
            save_parquet=False,
        )
    )
    df = loader.load()

    # 2. Validate structure
    _validate_raw_games(df, config.seasons)

    # 3. Filter to completed games: both scores present
    # NOTE: For prediction mode, we will later add a separate function that
    # builds a "prediction dataset" for upcoming games (no scores yet).
    completed = df[df["home_score"].notnull() & df["away_score"].notnull()].copy()
    if completed.empty:
        raise ValueError(f"No completed games found for seasons {config.seasons}")

    # Duplicate game detection (safety)
    duplicates = completed[completed.duplicated(subset=["game_id"], keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"Found {len(duplicates)} duplicate game_ids in base dataset. "
            "Check data source for errors."
        )

    # 4. Add season type flags
    completed = _add_season_type_flags(completed)

    # 5. Optionally drop preseason games (if any)
    if config.drop_preseason:
        # Preseason games are those that are neither regular nor postseason
        mask_keep = completed["is_regular_season"] | completed["is_postseason"]
        completed = completed[mask_keep].copy()
        if completed.empty:
            raise ValueError(
                "After dropping preseason, no games remain. "
                "Check season range or flags."
            )

    # 6. Sort chronologically and assign a stable index
    sort_cols = [c for c in ["gameday", "season", "week", "game_id"] if c in completed.columns]
    completed = completed.sort_values(sort_cols).reset_index(drop=True)
    completed["game_index"] = completed.index  # 0..N-1 in time order

    # 7. Optionally save to processed/ as a parquet file
    if config.save_parquet:
        DATA_CONFIG.processed_data_dir.mkdir(parents=True, exist_ok=True)
        start_season = min(config.seasons)
        end_season = max(config.seasons)
        filename = config.filename or f"base_games_{start_season}_{end_season}.parquet"
        out_path = DATA_CONFIG.processed_data_dir / filename
        completed.to_parquet(out_path, index=False)

    return completed


def load_base_dataset(
    start_season: int,
    end_season: int,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a previously built base dataset from parquet.

    Args:
        start_season: First season in dataset
        end_season: Last season in dataset
        processed_dir: Override default processed data directory

    Returns:
        Base dataset DataFrame

    Raises:
        FileNotFoundError: If dataset hasn't been built yet
        ValueError: If required columns are missing
    """
    if processed_dir is None:
        processed_dir = DATA_CONFIG.processed_data_dir

    filename = f"base_games_{start_season}_{end_season}.parquet"
    filepath = processed_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Base dataset not found: {filepath}\n"
            f"Run build_base_dataset() first with seasons {start_season}-{end_season}."
        )

    df = pd.read_parquet(filepath)

    # Quick validation
    required = ["game_id", "gameday", "home_win", "total_points", "game_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Loaded dataset missing required columns: {missing}")

    return df