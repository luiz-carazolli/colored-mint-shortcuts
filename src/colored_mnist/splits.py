"""Deterministic, stratified MNIST split generation with stable original IDs."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

from colored_mnist.config import DatasetConfig


@dataclass(frozen=True)
class SplitRecord:
    """Single original MNIST sample assigned to train, val, or test."""

    original_id: str
    source_split: str  # "train" or "test" in official MNIST
    source_index: int
    split: str  # "train", "val", or "test"
    label: int


@dataclass(frozen=True)
class RawMNISTData:
    """In-memory raw MNIST uint8 images and labels indexed by official split."""

    train_images: np.ndarray  # (60000, 28, 28) uint8
    train_labels: np.ndarray  # (60000,) int64
    test_images: np.ndarray | None
    test_labels: np.ndarray | None
    raw_file_hashes: dict[str, str]

    def get_image_and_label(self, source_split: str, source_index: int) -> tuple[np.ndarray, int]:
        if source_split == "train":
            return self.train_images[source_index], int(self.train_labels[source_index])
        if source_split == "test":
            if self.test_images is None or self.test_labels is None:
                raise ValueError("The official MNIST test split was not loaded")
            return self.test_images[source_index], int(self.test_labels[source_index])
        raise ValueError(f"Unknown source_split: {source_split!r}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_mnist(
    raw_data_dir: Path, download: bool = True, *, include_test: bool = True
) -> RawMNISTData:
    """Load and hash MNIST data, optionally reading only official training files."""
    if download and not include_test:
        raise ValueError("Train-only MNIST loading requires existing data and download=False")
    train_ds = MNIST(root=str(raw_data_dir), train=True, download=download)
    test_ds = MNIST(root=str(raw_data_dir), train=False, download=download) if include_test else None

    raw_folder = raw_data_dir / "MNIST" / "raw"
    hashes: dict[str, str] = {}
    if raw_folder.exists():
        for file_path in sorted(raw_folder.iterdir()):
            if not include_test and not file_path.name.startswith("train-"):
                continue
            if file_path.is_file() and not file_path.name.endswith(".gz"):
                hashes[file_path.name] = _sha256_file(file_path)

    return RawMNISTData(
        train_images=train_ds.data.numpy().astype(np.uint8, copy=False),
        train_labels=train_ds.targets.numpy().astype(np.int64, copy=False),
        test_images=test_ds.data.numpy().astype(np.uint8, copy=False) if test_ds is not None else None,
        test_labels=test_ds.targets.numpy().astype(np.int64, copy=False) if test_ds is not None else None,
        raw_file_hashes=hashes,
    )


def make_original_id(source_split: str, source_index: int) -> str:
    """Build stable identifier from official MNIST split and zero-based index."""
    return f"mnist_{source_split}_{source_index:05d}"


def build_stratified_splits(raw: RawMNISTData, config: DatasetConfig) -> list[SplitRecord]:
    """Create disjoint train/val/test SplitRecords stratified by label on official train."""
    if not (0.0 < config.val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {config.val_fraction}")

    rng = np.random.default_rng(config.split_seed)
    class_indices_by_digit = [np.flatnonzero(raw.train_labels == d) for d in range(10)]
    target_total_val = int(round(len(raw.train_labels) * config.val_fraction))

    # Largest-remainder stratified allocation so total validation size is exact (e.g. 6000)
    exact_quotas = [len(idx) * config.val_fraction for idx in class_indices_by_digit]
    val_counts_per_digit = [int(np.floor(q)) for q in exact_quotas]
    remainder = target_total_val - sum(val_counts_per_digit)
    if remainder > 0:
        order = sorted(
            range(10),
            key=lambda d: (exact_quotas[d] - val_counts_per_digit[d], -d),
            reverse=True,
        )
        for d in order[:remainder]:
            val_counts_per_digit[d] += 1

    val_indices: set[int] = set()
    for digit, class_indices in enumerate(class_indices_by_digit):
        chosen = rng.choice(class_indices, size=val_counts_per_digit[digit], replace=False)
        val_indices.update(int(i) for i in chosen)

    records: list[SplitRecord] = []
    for idx in range(len(raw.train_labels)):
        split = "val" if idx in val_indices else "train"
        records.append(
            SplitRecord(
                original_id=make_original_id("train", idx),
                source_split="train",
                source_index=idx,
                split=split,
                label=int(raw.train_labels[idx]),
            )
        )

    for idx in range(len(raw.test_labels)):
        records.append(
            SplitRecord(
                original_id=make_original_id("test", idx),
                source_split="test",
                source_index=idx,
                split="test",
                label=int(raw.test_labels[idx]),
            )
        )

    return records


def save_split_manifest(records: list[SplitRecord], path: Path) -> None:
    """Write SplitRecord list to CSV manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["original_id", "source_split", "source_index", "split", "label"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "original_id": r.original_id,
                    "source_split": r.source_split,
                    "source_index": r.source_index,
                    "split": r.split,
                    "label": r.label,
                }
            )


def load_split_manifest(path: Path) -> list[SplitRecord]:
    """Read SplitRecord list from CSV manifest."""
    records: list[SplitRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SplitRecord(
                    original_id=row["original_id"],
                    source_split=row["source_split"],
                    source_index=int(row["source_index"]),
                    split=row["split"],
                    label=int(row["label"]),
                )
            )
    return records
