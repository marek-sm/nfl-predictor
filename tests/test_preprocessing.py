import pandas as pd

from nfl_predictor.data.preprocessing.base_dataset import (
    BaseDatasetConfig,
    build_base_dataset,
    load_base_dataset,
)
from nfl_predictor.evaluation.splits import split_by_season


def test_build_base_dataset_basic():
    """Base dataset should load, filter, and sort correctly."""
    config = BaseDatasetConfig(
        seasons=list(range(2020, 2024)),
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )
    df = build_base_dataset(config)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    required_cols = [
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
        "game_index",
        "is_regular_season",
        "is_postseason",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column in base dataset: {col}"

    # Completed games only
    assert df["home_score"].notnull().all()
    assert df["away_score"].notnull().all()

    # game_index should be 0..N-1
    assert df["game_index"].min() == 0
    assert df["game_index"].max() == len(df) - 1

    # No duplicate game_ids
    assert df["game_id"].is_unique

    # gameday should be non-decreasing (ties allowed)
    gamedates = df["gameday"].values
    for i in range(len(gamedates) - 1):
        assert gamedates[i] <= gamedates[i + 1], (
            f"Games out of chronological order: {gamedates[i]} > {gamedates[i + 1]}"
        )


def test_base_dataset_chronological_index():
    """Verify game_index matches chronological order."""
    config = BaseDatasetConfig(
        seasons=[2023],
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )
    df = build_base_dataset(config)

    df_time_sorted = df.sort_values(
        ["gameday", "season", "week", "game_id"]
    ).reset_index(drop=True)
    expected_indices = df_time_sorted.index.values
    actual_indices = df_time_sorted["game_index"].values

    assert (actual_indices == expected_indices).all(), (
        "game_index does not match chronological order."
    )


def test_split_by_season_no_overlap():
    """split_by_season should create non-overlapping splits."""
    config = BaseDatasetConfig(
        seasons=list(range(2020, 2024)),
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )
    df = build_base_dataset(config)

    train_df, val_df, test_df = split_by_season(
        df,
        train_seasons=[2020, 2021],
        val_seasons=[2022],
        test_seasons=[2023],
    )

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    train_ids = set(train_df["game_id"])
    val_ids = set(val_df["game_id"])
    test_ids = set(test_df["game_id"])

    assert not (train_ids & val_ids), "Train/Val splits share game_ids."
    assert not (train_ids & test_ids), "Train/Test splits share game_ids."
    assert not (val_ids & test_ids), "Val/Test splits share game_ids."

    assert set(train_df["season"]) == {2020, 2021}
    assert set(val_df["season"]) == {2022}
    assert set(test_df["season"]) == {2023}


def test_splits_no_time_leakage():
    """Ensure test set games come AFTER train/val in time."""
    config = BaseDatasetConfig(
        seasons=range(2020, 2024),
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )
    df = build_base_dataset(config)

    train_df, val_df, test_df = split_by_season(
        df,
        train_seasons=[2020, 2021],
        val_seasons=[2022],
        test_seasons=[2023],
    )

    train_max = train_df["gameday"].max()
    val_min = val_df["gameday"].min()
    val_max = val_df["gameday"].max()
    test_min = test_df["gameday"].min()

    assert train_max <= val_min, (
        f"Time leakage: train ends {train_max}, val starts {val_min}"
    )
    assert val_max <= test_min, (
        f"Time leakage: val ends {val_max}, test starts {test_min}"
    )


def test_load_base_dataset_roundtrip():
    """build_base_dataset + load_base_dataset roundtrip should work."""
    seasons = [2023]
    config = BaseDatasetConfig(
        seasons=seasons,
        include_markets=True,
        drop_preseason=True,
        save_parquet=True,  # will save to DATA_CONFIG.processed_data_dir via build_base_dataset
    )

    df_built = build_base_dataset(config)

    loaded = load_base_dataset(
        start_season=2023,
        end_season=2023,
        # use default processed_dir, same as build_base_dataset
    )

    # Basic consistency check
    assert len(df_built) == len(loaded)
    assert set(df_built["game_id"]) == set(loaded["game_id"])