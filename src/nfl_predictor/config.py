from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Base directory for the project (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    """Data storage paths and defaults."""

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    default_seasons: Optional[List[int]] = None

    def __post_init__(self):
        if self.default_seasons is None:
            # Multi-season default; training code can override
            object.__setattr__(self, "default_seasons", list(range(2010, 2024)))


@dataclass(frozen=True)
class ModelConfig:
    """Model artifact storage and experiment tracking."""

    models_dir: Path = PROJECT_ROOT / "models"
    moneyline_dir: Path = models_dir / "moneyline"
    totals_dir: Path = models_dir / "totals"

    # Experiment tracking (local MLflow)
    mlflow_tracking_uri: str = f"file://{PROJECT_ROOT / 'mlruns'}"


@dataclass(frozen=True)
class LogConfig:
    """Logging and results paths."""

    logs_dir: Path = PROJECT_ROOT / "logs"
    predictions_dir: Path = PROJECT_ROOT / "results" / "predictions"
    accuracy_dir: Path = PROJECT_ROOT / "results" / "accuracy"


# Global config instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
LOG_CONFIG = LogConfig()
