"""Experimental matrix and data ingestion pipeline for Step 2 training sweeps."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.dataset import ColoredMNISTDataset


@dataclass(frozen=True)
class RunSpec:
    """Specification of a single training run in the experimental matrix."""

    run_id: str
    p: float
    dataset_seed: int
    train_seed: int
    num_train_samples: int
    expected_preferred_count: float
    expected_counterexamples: float
    realized_preferred_count: int
    realized_counterexamples: int
    realized_preferred_rate: float
    condition_type: str  # "unbiased_baseline", "biased", or "fully_confounded"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_condition_type(p: float) -> str:
    """Classify the correlation level into its conceptual role."""
    if abs(p - 0.10) < 1e-6:
        return "unbiased_baseline"
    if abs(p - 1.00) < 1e-6:
        return "fully_confounded"
    return "biased"


def build_run_id(p: float, dataset_seed: int, train_seed: int) -> str:
    """Generate standard deterministic identifier for a sweep run."""
    return f"p_{p:.2f}_dseed_{dataset_seed}_tseed_{train_seed}"


def get_paired_train_seed(dataset_seed: int) -> int:
    """Map dataset_seed (101, 102, 103) to its matched train_seed (201, 202, 203)."""
    return dataset_seed + 100


def build_experimental_matrix(
    artifacts_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> list[RunSpec]:
    """Construct the 21-run experimental grid and audit realized counterexamples against Step 1 manifests."""
    art_dir = artifacts_dir or config.output_dir
    raw_dir = raw_data_dir or config.raw_data_dir

    run_specs: list[RunSpec] = []
    for d_seed in config.dataset_seeds:
        t_seed = get_paired_train_seed(d_seed)
        for p in config.p_values:
            ds = ColoredMNISTDataset.from_artifacts(
                split="train",
                p=p,
                dataset_seed=d_seed,
                artifacts_dir=art_dir,
                raw_data_dir=raw_dir,
                config=config,
                return_metadata=True,
            )
            n_total = len(ds)
            # Count how many samples have their assigned color equal to the class's preferred color
            n_pref = sum(
                1 for entry in ds.entries if entry.color_id == entry.preferred_color_id
            )
            n_counter = n_total - n_pref
            exp_pref = n_total * p
            exp_counter = n_total * (1.0 - p)

            spec = RunSpec(
                run_id=build_run_id(p, d_seed, t_seed),
                p=p,
                dataset_seed=d_seed,
                train_seed=t_seed,
                num_train_samples=n_total,
                expected_preferred_count=exp_pref,
                expected_counterexamples=exp_counter,
                realized_preferred_count=n_pref,
                realized_counterexamples=n_counter,
                realized_preferred_rate=n_pref / n_total,
                condition_type=get_condition_type(p),
            )
            run_specs.append(spec)

    return run_specs


def save_experimental_matrix_csv(run_specs: list[RunSpec], path: Path) -> None:
    """Save the experimental matrix specifications to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "p",
        "dataset_seed",
        "train_seed",
        "condition_type",
        "num_train_samples",
        "expected_preferred_count",
        "expected_counterexamples",
        "realized_preferred_count",
        "realized_counterexamples",
        "realized_preferred_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in run_specs:
            writer.writerow(spec.to_dict())


def get_training_dataset(
    run_spec: RunSpec,
    artifacts_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    config: DatasetConfig = DEFAULT_CONFIG,
    return_metadata: bool = False,
) -> ColoredMNISTDataset:
    """Instantiate the verified training dataset for a specific matrix run."""
    return ColoredMNISTDataset.from_artifacts(
        split="train",
        p=run_spec.p,
        dataset_seed=run_spec.dataset_seed,
        artifacts_dir=artifacts_dir,
        raw_data_dir=raw_data_dir,
        config=config,
        return_metadata=return_metadata,
    )


def get_validation_datasets(
    artifacts_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    config: DatasetConfig = DEFAULT_CONFIG,
    return_metadata: bool = False,
) -> dict[str, ColoredMNISTDataset]:
    """Instantiate the two fixed validation views (val_aligned and val_independent)."""
    val_aligned = ColoredMNISTDataset.from_artifacts(
        split="val",
        view="aligned",
        artifacts_dir=artifacts_dir,
        raw_data_dir=raw_data_dir,
        config=config,
        return_metadata=return_metadata,
    )
    val_independent = ColoredMNISTDataset.from_artifacts(
        split="val",
        view="independent",
        artifacts_dir=artifacts_dir,
        raw_data_dir=raw_data_dir,
        config=config,
        return_metadata=return_metadata,
    )
    return {
        "val_aligned": val_aligned,
        "val_independent": val_independent,
    }


def create_dataloader(
    dataset: ColoredMNISTDataset,
    batch_size: int = 256,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
) -> DataLoader[Any]:
    """Create a standardized PyTorch DataLoader without augmentations or normalizations."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

