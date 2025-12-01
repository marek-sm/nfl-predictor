"""
Quick Feature Engineering Check (Step 3)

Run this script to verify that feature engineering builds correctly on top of
the existing Step 1 (loader) and Step 2 (base dataset).

Example:
    python run_feature_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nfl_predictor.config import DATA_CONFIG  # noqa: E402
from nfl_predictor.data.feature_engineering.feature_builder import (  # noqa: E402
    FeatureBuilder,
    FeatureBuilderConfig,
)


def main() -> None:
    print("▶ Running feature engineering check...\n")

    # Use a small set of recent seasons
    seasons = DATA_CONFIG.default_seasons[-5:]

    cfg = FeatureBuilderConfig(
        seasons=seasons,
        include_postseason=True,
        include_markets=True,
        drop_preseason=True,
        save_intermediate=False,
    )

    builder = FeatureBuilder(cfg)

    try:
        features_df = builder.build_features()
    except Exception as exc:  # pragma: no cover - manual inspection script
        print("\n❌ ERROR while building features:\n")
        print(type(exc).__name__, ":", str(exc))
        return

    print("✅ Features built successfully.")
    print(f"Seasons: {seasons[0]}–{seasons[-1]}")
    print(f"Shape: {features_df.shape[0]} rows x {features_df.shape[1]} columns\n")

    # Basic sanity checks
    assert features_df["game_id"].is_unique, "game_id should be unique per row."

    # We expect at least some rolling and diff features
    has_home_pf = any(
        col.startswith("home_points_for_rolling_mean_")
        for col in features_df.columns
    )
    has_away_pf = any(
        col.startswith("away_points_for_rolling_mean_")
        for col in features_df.columns
    )
    has_diff = any(
        col.startswith("diff_points_for_rolling_mean_")
        for col in features_df.columns
    )
    assert has_home_pf and has_away_pf and has_diff, (
        "Expected home/away rolling points_for and diff features to exist."
    )

    print("--- First 5 rows ---")
    print(features_df.head())

    # Show a subset of key feature columns
    interesting_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
    ]

    interesting_cols += [
        c
        for c in features_df.columns
        if c.startswith("home_points_for_rolling_mean_")
    ][:2]
    interesting_cols += [
        c
        for c in features_df.columns
        if c.startswith("away_points_for_rolling_mean_")
    ][:2]
    interesting_cols += [
        c
        for c in features_df.columns
        if c.startswith("diff_points_for_rolling_mean_")
    ][:2]
    interesting_cols += [
        "home_days_since_last_game",
        "away_days_since_last_game",
        "diff_days_since_last_game",
        "home_implied_prob_ml",
        "away_implied_prob_ml",
        "diff_implied_prob_ml",
    ]

    interesting_cols = [c for c in interesting_cols if c in features_df.columns]

    print("\n--- Sample of key feature columns ---")
    print(features_df[interesting_cols].head())

    print("\nDone.")


if __name__ == "__main__":
    main()