from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from nfl_predictor.data.loaders.nflreadpy_client import (
    NflreadpyClient,
    NflreadpyConfig,
)


@dataclass
class EPAFeaturesConfig:
    """
    Configuration for EPA / success / explosive / rate features derived from play-by-play.

    Attributes
    ----------
    seasons:
        Seasons to load from play-by-play. If None, infer from the base_games DataFrame.
    explosive_pass_yards:
        Yardage threshold for an explosive pass play.
    explosive_rush_yards:
        Yardage threshold for an explosive rush play.
    """

    seasons: list[int] | None = None
    explosive_pass_yards: int = 15
    explosive_rush_yards: int = 10


def _infer_seasons_from_games(base_games: pd.DataFrame) -> list[int]:
    seasons = sorted(base_games["season"].dropna().unique().tolist())
    return [int(s) for s in seasons]


def _prepare_pbp(df: pd.DataFrame, cfg: EPAFeaturesConfig) -> pd.DataFrame:
    """
    Clean and enrich raw pbp with flags we need for aggregation.
    """
    pbp = df.copy()

    # Keep only valid offensive plays with EPA and teams
    pbp = pbp[
        pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    # Helper to coerce a column to a boolean Series safely
    def _bool_col(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            s = frame[col]
            # handle numeric / float 0/1 or already bool
            if s.dtype == bool:
                return s.fillna(False)
            # treat nonzero as True
            s = s.fillna(0)
            try:
                s = s.astype(int)
            except (TypeError, ValueError):
                # fallback: cast directly to bool
                s = s.astype(bool)
            return s.astype(bool)
        # if column missing, return all-False Series
        return pd.Series(False, index=frame.index)

    pass_flag = _bool_col(pbp, "pass")
    rush_flag = _bool_col(pbp, "rush")
    play_type = pbp.get("play_type")

    # Play type flags: a play is a pass if either the pass flag is true
    # or the play_type is explicitly "pass", similarly for rush.
    pbp["is_pass"] = pass_flag | (play_type == "pass")
    pbp["is_rush"] = rush_flag | (play_type == "run")

    pbp["yards_gained"] = pbp["yards_gained"].fillna(0)

    # Explosive plays
    pbp["is_explosive"] = (
        (pbp["is_pass"] & (pbp["yards_gained"] >= cfg.explosive_pass_yards))
        | (pbp["is_rush"] & (pbp["yards_gained"] >= cfg.explosive_rush_yards))
    )

    # Success flag (nflfastR schema has `success`; otherwise fallback)
    if "success" not in pbp.columns:
        pbp["success"] = pbp["epa"] > 0

    return pbp


def _aggregate_offense(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate offensive EPA / success / explosive / pass & rush rates per team-game.
    """
    group_cols = ["season", "game_id", "posteam"]
    grouped = pbp.groupby(group_cols, dropna=False)

    # We use lambdas referencing the global pbp to filter pass/rush subsets.
    agg = grouped.agg(
        plays_off=("epa", "size"),
        off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"),
        off_explosive_play_rate=("is_explosive", "mean"),
        off_pass_rate=("is_pass", "mean"),
        off_rush_rate=("is_rush", "mean"),
    ).reset_index()

    # Pass / rush EPA separately
    # (do these in a second pass to avoid MultiIndex magic)
    def _per_subset(group: pd.DataFrame, mask_col: str) -> float:
        sub = group[group[mask_col]]
        if sub.empty:
            return 0.0
        return float(sub["epa"].mean())

    per_rows = []
    for (season, game_id, posteam), group in pbp.groupby(group_cols, dropna=False):
        per_rows.append(
            {
                "season": season,
                "game_id": game_id,
                "posteam": posteam,
                "off_pass_epa_per_play": _per_subset(group, "is_pass"),
                "off_rush_epa_per_play": _per_subset(group, "is_rush"),
            }
        )
    per_df = pd.DataFrame(per_rows)

    agg = agg.merge(
        per_df,
        on=["season", "game_id", "posteam"],
        how="left",
        validate="one_to_one",
    )

    agg = agg.rename(columns={"posteam": "team"})

    return agg


def _aggregate_defense(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate defensive EPA allowed per team-game.

    NOTE: We treat offensive EPA as-is; from the defense perspective,
    lower (more negative) EPA allowed is better.
    """
    group_cols = ["season", "game_id", "defteam"]
    grouped = pbp.groupby(group_cols, dropna=False)

    agg = grouped.agg(
        plays_def=("epa", "size"),
        def_epa_per_play_allowed=("epa", "mean"),
        def_success_rate_allowed=("success", "mean"),
        def_explosive_play_rate_allowed=("is_explosive", "mean"),
    ).reset_index()

    # Pass / rush EPA allowed
    def _per_subset(group: pd.DataFrame, mask_col: str) -> float:
        sub = group[group[mask_col]]
        if sub.empty:
            return 0.0
        return float(sub["epa"].mean())

    per_rows = []
    for (season, game_id, defteam), group in pbp.groupby(group_cols, dropna=False):
        per_rows.append(
            {
                "season": season,
                "game_id": game_id,
                "defteam": defteam,
                "def_pass_epa_per_play_allowed": _per_subset(group, "is_pass"),
                "def_rush_epa_per_play_allowed": _per_subset(group, "is_rush"),
            }
        )
    per_df = pd.DataFrame(per_rows)

    agg = agg.merge(
        per_df,
        on=["season", "game_id", "defteam"],
        how="left",
        validate="one_to_one",
    )

    agg = agg.rename(columns={"defteam": "team"})

    return agg


def build_team_epa_features(
    base_games: pd.DataFrame,
    config: EPAFeaturesConfig | None = None,
    client: NflreadpyClient | None = None,
) -> pd.DataFrame:
    """
    Build team-game-level EPA / success / explosive / rate features from play-by-play.

    Returns a DataFrame with columns:
    - season, team, game_id
    - plays_off, plays_def
    - off_epa_per_play, off_pass_epa_per_play, off_rush_epa_per_play
    - off_success_rate, off_explosive_play_rate
    - off_pass_rate, off_rush_rate
    - def_epa_per_play_allowed
    - def_pass_epa_per_play_allowed, def_rush_epa_per_play_allowed
    - def_success_rate_allowed, def_explosive_play_rate_allowed
    """
    if config is None:
        config = EPAFeaturesConfig()

    if config.seasons is not None:
        seasons = config.seasons
    else:
        seasons = _infer_seasons_from_games(base_games)

    if client is None:
        client = NflreadpyClient(NflreadpyConfig(seasons=seasons))

    pbp = client.load_pbp(seasons=seasons)
    pbp = _prepare_pbp(pbp, config)

    off_agg = _aggregate_offense(pbp)
    def_agg = _aggregate_defense(pbp)

    merged = off_agg.merge(
        def_agg,
        on=["season", "game_id", "team"],
        how="outer",
        validate="one_to_one",
    )

    # Fill missing counts with 0
    merged["plays_off"] = merged["plays_off"].fillna(0).astype(int)
    merged["plays_def"] = merged["plays_def"].fillna(0).astype(int)

    # Flag: did we actually have pbp for this team-game?
    merged["pbp_coverage_flag"] = ~(merged["plays_off"].isna() & merged["plays_def"].isna())

    return merged