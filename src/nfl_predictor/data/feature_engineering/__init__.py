"""
Feature engineering pipelines for the NFL prediction system (Step 3).

Responsibilities
----------------
- Take the Step 2 base dataset (one row per game).
- Build team-game long format (two rows per game: one per team).
- Add schedule/rest features (days since last game, short week, bye, etc.).
- Add market-aware team-level features (ATS margin, total vs line, implied probs).
- Add leak-free rolling & expanding features grouped by [team, season].
- Re-pivot back to game-level (home_* / away_* features).
- Add matchup differential features and modeling targets.

Usage example
-------------
    from nfl_predictor.data.feature_engineering.feature_builder import (
        FeatureBuilder,
        FeatureBuilderConfig,
    )

    fb_config = FeatureBuilderConfig(seasons=[2021, 2022])
    builder = FeatureBuilder(fb_config)
    features_df = builder.build_features()
"""

# Intentionally keep this file light to avoid circular imports.
# Import concrete modules where you need them, e.g.:
#
#   from nfl_predictor.data.feature_engineering.feature_builder import FeatureBuilder
#
# rather than relying on this package to re-export everything.