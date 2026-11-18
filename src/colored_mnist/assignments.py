"""Deterministic background assignments for training p-sweeps and held-out views."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from colored_mnist.config import DatasetConfig
from colored_mnist.palette import Palette
from colored_mnist.splits import SplitRecord


@dataclass(frozen=True)
class TrainDrawRecord:
    """Base random draws for one training digit under a single dataset_seed."""

    original_id: str
    source_split: str
    source_index: int
    label: int
    dataset_seed: int
    preferred_color_id: int
    alt_color_id: int
    uniform_draw: float

    def color_for_p(self, p: float) -> int:
        """Select preferred color if uniform_draw < p, else the fixed alternative color."""
        return self.preferred_color_id if self.uniform_draw < p else self.alt_color_id


@dataclass(frozen=True)
class HeldoutViewRecord:
    """Background assignments across all three fixed views for one val/test original."""

    original_id: str
    source_split: str
    source_index: int
    split: str  # "val" or "test"
    label: int
    aligned_color_id: int
    independent_color_id: int
    swapped_color_id: int
    swapped_target_class: int

    def color_for_view(self, view: str) -> int:
        if view == "aligned":
            return self.aligned_color_id
        if view == "independent":
            return self.independent_color_id
        if view == "swapped":
            return self.swapped_color_id
        raise ValueError(f"Unknown held-out view: {view!r}")


def p_column_name(p: float) -> str:
    """Format canonical manifest column name for a correlation probability p."""
    return f"color_p_{p:.2f}"


def generate_training_draws(
    splits: list[SplitRecord],
    palette: Palette,
    dataset_seed: int,
) -> list[TrainDrawRecord]:
    """Generate reusable uniform draws and nonpreferred alternatives for training digits."""
    train_records = [r for r in splits if r.split == "train"]
    rng = np.random.default_rng(dataset_seed)

    # Draw both u ~ U(0, 1) and alternative offset in {0..8} per sample in deterministic order
    u_draws = rng.random(len(train_records))
    alt_choices = rng.integers(0, 9, size=len(train_records))

    draws: list[TrainDrawRecord] = []
    for rec, u, alt_idx in zip(train_records, u_draws, alt_choices):
        pref = palette.preferred_color_by_digit[rec.label]
        nonpreferred = [c for c in range(10) if c != pref]
        alt = nonpreferred[int(alt_idx)]
        draws.append(
            TrainDrawRecord(
                original_id=rec.original_id,
                source_split=rec.source_split,
                source_index=rec.source_index,
                label=rec.label,
                dataset_seed=dataset_seed,
                preferred_color_id=pref,
                alt_color_id=alt,
                uniform_draw=float(u),
            )
        )
    return draws


def generate_heldout_views(
    splits: list[SplitRecord],
    palette: Palette,
    evaluation_seed: int,
) -> list[HeldoutViewRecord]:
    """Generate fixed Aligned, Independent, and Swapped views for val and test digits."""
    heldout = [r for r in splits if r.split in ("val", "test")]
    rng = np.random.default_rng(evaluation_seed)
    indep_colors = rng.integers(0, 10, size=len(heldout))

    records: list[HeldoutViewRecord] = []
    for rec, indep in zip(heldout, indep_colors):
        aligned = palette.preferred_color_by_digit[rec.label]
        swapped_target = palette.swapped_target_by_digit[rec.label]
        swapped_color = palette.preferred_color_by_digit[swapped_target]
        records.append(
            HeldoutViewRecord(
                original_id=rec.original_id,
                source_split=rec.source_split,
                source_index=rec.source_index,
                split=rec.split,
                label=rec.label,
                aligned_color_id=aligned,
                independent_color_id=int(indep),
                swapped_color_id=swapped_color,
                swapped_target_class=swapped_target,
            )
        )
    return records


def save_training_manifest(
    draws: list[TrainDrawRecord],
    p_values: tuple[float, ...],
    path: Path,
) -> None:
    """Save training base draws and resolved color IDs for each p in the sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p_cols = [p_column_name(p) for p in p_values]
    fieldnames = [
        "original_id",
        "source_split",
        "source_index",
        "label",
        "dataset_seed",
        "preferred_color_id",
        "alt_color_id",
        "uniform_draw",
        *p_cols,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in draws:
            row: dict[str, object] = {
                "original_id": d.original_id,
                "source_split": d.source_split,
                "source_index": d.source_index,
                "label": d.label,
                "dataset_seed": d.dataset_seed,
                "preferred_color_id": d.preferred_color_id,
                "alt_color_id": d.alt_color_id,
                "uniform_draw": repr(d.uniform_draw),
            }
            for p, col in zip(p_values, p_cols):
                row[col] = d.color_for_p(p)
            writer.writerow(row)


def load_training_manifest(path: Path) -> list[TrainDrawRecord]:
    """Load TrainDrawRecord list from CSV manifest."""
    records: list[TrainDrawRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                TrainDrawRecord(
                    original_id=row["original_id"],
                    source_split=row["source_split"],
                    source_index=int(row["source_index"]),
                    label=int(row["label"]),
                    dataset_seed=int(row["dataset_seed"]),
                    preferred_color_id=int(row["preferred_color_id"]),
                    alt_color_id=int(row["alt_color_id"]),
                    uniform_draw=float(row["uniform_draw"]),
                )
            )
    return records


def save_heldout_manifest(records: list[HeldoutViewRecord], path: Path) -> None:
    """Save held-out (val + test) paired view assignments to CSV manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "original_id",
        "source_split",
        "source_index",
        "split",
        "label",
        "aligned_color_id",
        "independent_color_id",
        "swapped_color_id",
        "swapped_target_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "original_id": r.original_id,
                    "source_split": r.source_split,
                    "source_index": r.source_index,
                    "split": r.split,
                    "label": r.label,
                    "aligned_color_id": r.aligned_color_id,
                    "independent_color_id": r.independent_color_id,
                    "swapped_color_id": r.swapped_color_id,
                    "swapped_target_class": r.swapped_target_class,
                }
            )


def load_heldout_manifest(
    path: Path, *, split: str | None = None,
) -> list[HeldoutViewRecord]:
    """Load records for the requested split, or all records when unspecified."""
    records: list[HeldoutViewRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(
                HeldoutViewRecord(
                    original_id=row["original_id"],
                    source_split=row["source_split"],
                    source_index=int(row["source_index"]),
                    split=row["split"],
                    label=int(row["label"]),
                    aligned_color_id=int(row["aligned_color_id"]),
                    independent_color_id=int(row["independent_color_id"]),
                    swapped_color_id=int(row["swapped_color_id"]),
                    swapped_target_class=int(row["swapped_target_class"]),
                )
            )
    return records


def build_all_assignments(
    splits: list[SplitRecord],
    palette: Palette,
    config: DatasetConfig,
) -> tuple[dict[int, list[TrainDrawRecord]], list[HeldoutViewRecord]]:
    """Build training draws for all configured dataset_seeds and held-out views."""
    train_by_seed = {
        seed: generate_training_draws(splits, palette, seed)
        for seed in config.dataset_seeds
    }
    heldout = generate_heldout_views(splits, palette, config.evaluation_seed)
    return train_by_seed, heldout
