"""PyTorch Dataset interface for Colored MNIST training sweeps and held-out views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from colored_mnist.assignments import (
    HeldoutViewRecord,
    TrainDrawRecord,
    load_heldout_manifest,
    load_training_manifest,
)
from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.palette import Palette, composite_digit_srgb, load_palette
from colored_mnist.splits import RawMNISTData, load_raw_mnist


@dataclass(frozen=True)
class SampleEntry:
    """Resolved sample entry pairing an original MNIST image with one background color."""

    original_id: str
    source_split: str
    source_index: int
    label: int
    color_id: int
    preferred_color_id: int
    swapped_target_class: int | None
    view: str


class ColoredMNISTDataset(Dataset[Any]):
    """PyTorch Dataset serving deterministic Colored MNIST splits and counterfactual views."""

    def __init__(
        self,
        raw: RawMNISTData,
        palette: Palette,
        entries: list[SampleEntry],
        return_metadata: bool = False,
    ) -> None:
        self.raw = raw
        self.palette = palette
        self.entries = entries
        self.return_metadata = return_metadata
        self._bg_linear_table = palette.bg_linear_table

    @classmethod
    def for_training(
        cls,
        raw: RawMNISTData,
        palette: Palette,
        train_draws: list[TrainDrawRecord],
        p: float,
        return_metadata: bool = False,
    ) -> ColoredMNISTDataset:
        """Build training dataset for a fixed correlation strength p."""
        entries = [
            SampleEntry(
                original_id=d.original_id,
                source_split=d.source_split,
                source_index=d.source_index,
                label=d.label,
                color_id=d.color_for_p(p),
                preferred_color_id=d.preferred_color_id,
                swapped_target_class=None,
                view=f"train_p_{p:.2f}_seed_{d.dataset_seed}",
            )
            for d in train_draws
        ]
        return cls(raw=raw, palette=palette, entries=entries, return_metadata=return_metadata)

    @classmethod
    def for_heldout(
        cls,
        raw: RawMNISTData,
        palette: Palette,
        heldout_records: list[HeldoutViewRecord],
        split: str,
        view: str,
        return_metadata: bool = False,
    ) -> ColoredMNISTDataset:
        """Build validation or test dataset for one of: aligned, independent, swapped."""
        if split not in ("val", "test"):
            raise ValueError(f"Held-out split must be 'val' or 'test', got {split!r}")
        if view not in ("aligned", "independent", "swapped"):
            raise ValueError(f"Unknown held-out view: {view!r}")

        entries = [
            SampleEntry(
                original_id=r.original_id,
                source_split=r.source_split,
                source_index=r.source_index,
                label=r.label,
                color_id=r.color_for_view(view),
                preferred_color_id=r.aligned_color_id,
                swapped_target_class=r.swapped_target_class if view == "swapped" else None,
                view=view,
            )
            for r in heldout_records
            if r.split == split
        ]
        return cls(raw=raw, palette=palette, entries=entries, return_metadata=return_metadata)

    @classmethod
    def from_artifacts(
        cls,
        split: str,
        p: float | None = None,
        dataset_seed: int | None = None,
        view: str | None = None,
        artifacts_dir: Path | None = None,
        raw_data_dir: Path | None = None,
        config: DatasetConfig = DEFAULT_CONFIG,
        return_metadata: bool = False,
    ) -> ColoredMNISTDataset:
        """Load dataset directly from saved Step 1 manifest files."""
        art_dir = artifacts_dir or config.output_dir
        raw_dir = raw_data_dir or config.raw_data_dir
        raw = load_raw_mnist(raw_dir, download=False, include_test=(split == "test"))
        palette = load_palette(art_dir / "palette.json")

        if split == "train":
            if p is None or dataset_seed is None:
                raise ValueError("split='train' requires both p and dataset_seed")
            manifest_path = art_dir / "manifests" / f"train_seed_{dataset_seed}.csv"
            draws = load_training_manifest(manifest_path)
            return cls.for_training(
                raw=raw,
                palette=palette,
                train_draws=draws,
                p=p,
                return_metadata=return_metadata,
            )

        if split in ("val", "test"):
            if view is None:
                raise ValueError("Held-out split requires view ('aligned', 'independent', 'swapped')")
            manifest_path = art_dir / "manifests" / "heldout_views.csv"
            heldout = load_heldout_manifest(manifest_path, split=split)
            return cls.for_heldout(
                raw=raw,
                palette=palette,
                heldout_records=heldout,
                split=split,
                view=view,
                return_metadata=return_metadata,
            )

        raise ValueError(f"Unknown split: {split!r}")

    def __len__(self) -> int:
        return len(self.entries)

    def render_uint8(self, index: int) -> np.ndarray:
        """Render the sRGB uint8 image of shape (28, 28, 3) for item index."""
        entry = self.entries[index]
        mask_u8, _ = self.raw.get_image_and_label(entry.source_split, entry.source_index)
        bg_lin = self._bg_linear_table[entry.color_id]
        return composite_digit_srgb(mask_u8, bg_lin)

    def __getitem__(self, index: int) -> Any:
        """Return the sample, with target class -1 in metadata for unswapped views."""
        entry = self.entries[index]
        img_u8 = self.render_uint8(index)
        tensor = torch.from_numpy(img_u8).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

        if not self.return_metadata:
            return tensor, entry.label

        metadata = {
            "original_id": entry.original_id,
            "source_split": entry.source_split,
            "source_index": entry.source_index,
            "label": entry.label,
            "color_id": entry.color_id,
            "preferred_color_id": entry.preferred_color_id,
            "swapped_target_class": (
                entry.swapped_target_class if entry.swapped_target_class is not None else -1
            ),
            "view": entry.view,
        }
        return tensor, entry.label, metadata

