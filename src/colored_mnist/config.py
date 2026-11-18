"""Central configuration for Step 1 Colored MNIST dataset construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration controlling splits, palette, and assignments."""

    raw_data_dir: Path = Path("data/raw")
    output_dir: Path = Path("artifacts/step_1_dataset")

    # Split configuration
    val_fraction: float = 0.10
    split_seed: int = 2026

    # CIELAB palette configuration (10 evenly spaced hues at constant L* and C*)
    lab_lightness: float = 45.0
    lab_chroma: float = 28.0
    lab_hue_start_deg: float = 18.0
    lab_hue_step_deg: float = 36.0
    preferred_color_by_digit: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # Antipodal derangement on the 10-hue circle: digit d receives preferred color of (d + 5) % 10
    swapped_target_by_digit: tuple[int, ...] = (5, 6, 7, 8, 9, 0, 1, 2, 3, 4)
    min_contrast_ratio: float = 4.5

    # Training association sweep configuration
    p_values: tuple[float, ...] = (0.10, 0.50, 0.70, 0.85, 0.95, 0.99, 1.00)
    dataset_seeds: tuple[int, ...] = (101, 102, 103)

    # Fixed held-out evaluation seed (used for Independent view on val & test)
    evaluation_seed: int = 999

    # Source metadata
    source_name: str = "MNIST (via torchvision / CVDF mirror)"
    source_url: str = "https://github.com/cvdfoundation/mnist"

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        data = asdict(self)
        data["raw_data_dir"] = str(self.raw_data_dir)
        data["output_dir"] = str(self.output_dir)
        data["preferred_color_by_digit"] = list(self.preferred_color_by_digit)
        data["swapped_target_by_digit"] = list(self.swapped_target_by_digit)
        data["p_values"] = list(self.p_values)
        data["dataset_seeds"] = list(self.dataset_seeds)
        return data


DEFAULT_CONFIG = DatasetConfig()

