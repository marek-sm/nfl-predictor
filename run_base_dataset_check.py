"""
Quick Base Dataset Check

Run this to verify the base modeling dataset builds correctly.

Example:
    python run_base_dataset_check.py
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd  # noqa: E402
from nfl_predictor.data.preprocessing.base_dataset import (  # noqa: E402
    BaseDatasetConfig,
    build_base_dataset,
)


def main():
    print("=== NFL Predictor: Base Dataset Check ===")

    seasons = list(range(2020, 2024))
    print(f"Building base dataset for seasons: {seasons}")

    config = BaseDatasetConfig(
        seasons=seasons,
        include_markets=True,
        drop_preseason=True,
        save_parquet=False,
    )

    try:
        df = build_base_dataset(config)

        print("\n--- Base Dataset Summary ---")
        print(f"Total games: {len(df)}")
        print("Columns:", list(df.columns))

        print("\n--- Head (first 10 rows) ---")
        print(df.head(10).to_string())

        print("\n--- Value counts: is_regular_season ---")
        print(df["is_regular_season"].value_counts(dropna=False))

        print("\n--- Value counts: is_postseason ---")
        print(df["is_postseason"].value_counts(dropna=False))

        print("\nSuccess! Base dataset built correctly.")
    except Exception as e:
        print("\nERROR while building base dataset:\n")
        print(type(e).__name__, ":", str(e))


if __name__ == "__main__":
    main()