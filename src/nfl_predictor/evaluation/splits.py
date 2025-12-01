from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pandas as pd


def _to_int_list(seasons: Optional[Iterable[int]]) -> List[int]:
    if seasons is None:
        return []
    return sorted(int(s) for s in seasons)


def split_by_season(
    df: pd.DataFrame,
    train_seasons: Iterable[int],
    val_seasons: Optional[Iterable[int]] = None,
    test_seasons: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split a DataFrame into train/val/test sets by season.

    This is a simple, transparent helper for:
        - Walk-forward evaluation
        - Clear anti-leakage boundaries

    Args:
        df: Base dataset with a 'season' column.
        train_seasons: Seasons to include in the training set.
        val_seasons: Seasons for validation set (or None).
        test_seasons: Seasons for test set (or None).

    Returns:
        (train_df, val_df, test_df) where val_df/test_df may be None.

    Raises:
        ValueError if seasons overlap between splits, or season column missing.
    """
    if "season" not in df.columns:
        raise ValueError("DataFrame must contain a 'season' column for split_by_season.")

    train = _to_int_list(train_seasons)
    val = _to_int_list(val_seasons)
    test = _to_int_list(test_seasons)

    # Check for overlaps
    train_set, val_set, test_set = set(train), set(val), set(test)
    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        raise ValueError(
            f"Season sets must not overlap.\n"
            f"train={train_set}, val={val_set}, test={test_set}"
        )

    def _subset_for(season_list: List[int]) -> Optional[pd.DataFrame]:
        if not season_list:
            return None
        return df[df["season"].isin(season_list)].copy()

    train_df = _subset_for(train)
    val_df = _subset_for(val)
    test_df = _subset_for(test)

    return train_df, val_df, test_df