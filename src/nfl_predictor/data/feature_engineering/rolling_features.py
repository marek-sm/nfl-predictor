from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class RollingSpec:
    """
    Specification for leak-free rolling features built from a single column.

    Attributes
    ----------
    col:
        Source column in the DataFrame to roll over.
    windows:
        Rolling window sizes, in number of games.
    stats:
        Statistics to compute. Supported: "mean", "sum", "std", "min", "max".
    min_periods:
        Minimum number of prior observations required to compute a value.
        If fewer than min_periods, result will be NaN.
    prefix:
        Prefix used for generated feature names. If None, uses `col`.
        Output columns follow the pattern: "{prefix}_rolling_{stat}_{window}".
    """

    col: str
    windows: Sequence[int]
    stats: Sequence[str] = ("mean",)
    min_periods: int = 1
    prefix: str | None = None


_SUPPORTED_STATS = {"mean", "sum", "std", "min", "max"}


def _validate_specs(df: pd.DataFrame, specs: Iterable[RollingSpec]) -> list[RollingSpec]:
    specs = list(specs)
    if not specs:
        raise ValueError("At least one RollingSpec must be provided.")

    for spec in specs:
        if spec.col not in df.columns:
            raise KeyError(f"RollingSpec refers to missing column: {spec.col}")
        for stat in spec.stats:
            if stat not in _SUPPORTED_STATS:
                raise ValueError(
                    f"Unsupported stat '{stat}' in RollingSpec for col '{spec.col}'. "
                    f"Supported: {_SUPPORTED_STATS}"
                )
    return specs


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    time_col: str,
    specs: Sequence[RollingSpec],
) -> pd.DataFrame:
    """
    Add leak-free rolling features to a DataFrame.

    Anti-leakage guarantee
    ----------------------
    For each row, rolling statistics are computed ONLY from rows with strictly
    smaller `time_col` within the same group (group_cols). This is enforced by
    applying a 1-row shift before rolling.

    Implementation note
    -------------------
    All specs are applied within a single groupby/apply pass per group to avoid
    repeated grouping work, which is more efficient than one groupby per spec.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain group_cols + time_col + all spec.col values.
    group_cols:
        Columns defining an independent time series (e.g., ["team", "season"]).
    time_col:
        Column defining chronological order within each group (e.g., "game_index").
        Must be sortable and consistent with time semantics.
    specs:
        RollingSpec definitions describing what to compute.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with new rolling feature columns appended. Original index
        order is preserved.
    """
    if not group_cols:
        raise ValueError("group_cols must not be empty.")

    for col in group_cols:
        if col not in df.columns:
            raise KeyError(f"group_col '{col}' not found in DataFrame.")
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in DataFrame.")

    specs = _validate_specs(df, specs)

    # Preserve original ordering by index
    original_index = df.index
    result = df.sort_values(list(group_cols) + [time_col]).copy()
    group_cols_list = list(group_cols)

    def _apply_all_specs(group: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all RollingSpecs to a single group.

        IMPORTANT: Uses shifted values so each game's rolling stats are based
        only on prior games (no current-row information).
        """
        g = group.copy()

        for spec in specs:
            values = g[spec.col].shift(1)  # drop current row from window
            prefix = spec.prefix or spec.col

            for window in spec.windows:
                roll = values.rolling(window=window, min_periods=spec.min_periods)

                for stat in spec.stats:
                    if stat == "mean":
                        series = roll.mean()
                    elif stat == "sum":
                        series = roll.sum()
                    elif stat == "std":
                        series = roll.std()
                    elif stat == "min":
                        series = roll.min()
                    elif stat == "max":
                        series = roll.max()
                    else:  # pragma: no cover (validated earlier)
                        raise RuntimeError(f"Unexpected stat: {stat}")

                    col_name = f"{prefix}_rolling_{stat}_{window}"
                    g[col_name] = series

        return g

    result = (
        result.groupby(group_cols_list, group_keys=False, sort=False)
        # NOTE: We intentionally do not pass `include_groups=False` here to
        # keep compatibility with a broad range of pandas versions. The
        # default behavior is compatible with our usage, even though newer
        # pandas versions may emit a FutureWarning about group columns.
        .apply(_apply_all_specs)
    )

    # Restore original row order
    result = result.loc[original_index]

    return result