"""
Quick Data Load Check

Run this script to verify that the data loader works correctly.

Example:
    python run_data_check.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nfl_predictor.data.loaders.games import GameDataLoader, GameDataLoaderConfig
import pandas as pd

def main():
    print("=== NFL Predictor: Data Load Check ===")

    # Choose seasons here
    seasons = list(range(2020, 2024))
    print(f"Loading seasons: {seasons}")

    try:
        loader = GameDataLoader(
            GameDataLoaderConfig(
                seasons=seasons,
                include_markets=True,
                save_parquet=False,
            )
        )

        df = loader.load()

        print("\n--- Loaded Data Summary ---")
        print(f"Total games loaded: {len(df)}")
        print(f"Columns: {list(df.columns)}\n")

        print("--- Head (first 10 rows) ---")
        print(df.head(10).to_string())

        print("\n--- Data Types ---")
        print(df.dtypes)

        print("\nSuccess! Data loaded correctly.")

    except Exception as e:
        print("\nERROR: Something went wrong while loading data.\n")
        print(type(e).__name__, ":", str(e))


if __name__ == "__main__":
    main()