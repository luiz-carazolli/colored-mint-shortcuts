"""Verification test suite for Step 1 Colored MNIST dataset deliverables."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from colored_mnist.assignments import (
    load_heldout_manifest,
    load_training_manifest,
)
from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.palette import build_palette, composite_digit_srgb
from colored_mnist.splits import load_raw_mnist, load_split_manifest
from colored_mnist.verify import run_all_verifications

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
RAW_DIR = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir


def test_palette_gamut_and_contrast() -> None:
    palette = build_palette(DEFAULT_CONFIG)
    assert len(palette.colors) == 10
    for c in palette.colors:
        assert all(0.0 <= v <= 1.0 for v in c.linear_rgb_exact)
        assert all(0 <= v <= 255 for v in c.srgb_u8)
        assert c.contrast_vs_white >= DEFAULT_CONFIG.min_contrast_ratio

    # Verify tight isoluminance across all 10 classes (spread < 0.1 in CR)
    crs = [c.contrast_vs_white for c in palette.colors]
    assert max(crs) - min(crs) < 0.1


def test_compositing_endpoints_and_dtype() -> None:
    palette = build_palette(DEFAULT_CONFIG)
    mask = np.array([[0, 128, 255]], dtype=np.uint8)
    for c in palette.colors:
        rendered = composite_digit_srgb(mask, c.linear_rgb_quantized)
        assert rendered.shape == (1, 3, 3)
        assert rendered.dtype == np.uint8
        # Pure background pixel matches quantized palette RGB exactly
        assert tuple(int(v) for v in rendered[0, 0]) == c.srgb_u8
        # Pure foreground pixel is pure white
        assert tuple(int(v) for v in rendered[0, 2]) == (255, 255, 255)
        # Blended edge pixel sits strictly between background and white
        assert all(c.srgb_u8[ch] < rendered[0, 1, ch] < 255 for ch in range(3))


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "manifests" / "splits.csv").exists(),
    reason="Step 1 artifacts must be generated first via scripts/build_step1_dataset.py",
)
def test_saved_manifests_and_verifications() -> None:
    raw = load_raw_mnist(RAW_DIR, download=False)
    palette = build_palette(DEFAULT_CONFIG)
    splits = load_split_manifest(ARTIFACTS_DIR / "manifests" / "splits.csv")
    train_by_seed = {
        seed: load_training_manifest(ARTIFACTS_DIR / "manifests" / f"train_seed_{seed}.csv")
        for seed in DEFAULT_CONFIG.dataset_seeds
    }
    heldout = load_heldout_manifest(ARTIFACTS_DIR / "manifests" / "heldout_views.csv")

    report, _ = run_all_verifications(
        raw=raw,
        splits=splits,
        train_by_seed=train_by_seed,
        heldout=heldout,
        palette=palette,
        config=DEFAULT_CONFIG,
    )
    assert report["status"] == "PASSED"
    assert report["splits"]["train_count"] == 54000
    assert report["splits"]["val_count"] == 6000
    assert report["splits"]["test_count"] == 10000

    # Check required deliverable files exist
    for rel_path in (
        "config.json",
        "palette.json",
        "dataset_metadata.json",
        "verification_report.json",
        "digit_color_counts.csv",
        "validation_sample_grid.png",
        "README.md",
        "manifests/splits.csv",
        "manifests/heldout_views.csv",
    ):
        assert (ARTIFACTS_DIR / rel_path).is_file(), f"Missing artifact: {rel_path}"

    saved_report = json.loads((ARTIFACTS_DIR / "verification_report.json").read_text(encoding="utf-8"))
    assert saved_report["status"] == "PASSED"


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "manifests" / "heldout_views.csv").exists(),
    reason="Step 1 artifacts must be generated first via scripts/build_step1_dataset.py",
)
def test_pytorch_dataset_interface_and_pairing() -> None:
    seed = DEFAULT_CONFIG.dataset_seeds[0]
    ds_p01 = ColoredMNISTDataset.from_artifacts(
        split="train",
        p=0.10,
        dataset_seed=seed,
        artifacts_dir=ARTIFACTS_DIR,
        raw_data_dir=RAW_DIR,
        return_metadata=True,
    )
    ds_p10 = ColoredMNISTDataset.from_artifacts(
        split="train",
        p=1.00,
        dataset_seed=seed,
        artifacts_dir=ARTIFACTS_DIR,
        raw_data_dir=RAW_DIR,
        return_metadata=True,
    )
    assert len(ds_p01) == 54000
    assert len(ds_p10) == 54000

    x0, y0, m0 = ds_p01[0]
    x1, y1, m1 = ds_p10[0]
    assert x0.shape == (3, 28, 28)
    assert x0.dtype == torch.float32
    assert 0.0 <= float(x0.min()) <= float(x0.max()) <= 1.0
    assert m0["original_id"] == m1["original_id"]
    assert y0 == y1
    assert m1["color_id"] == m1["preferred_color_id"]

    # Held-out views share identical original_id and label one-to-one
    ds_aligned = ColoredMNISTDataset.from_artifacts(
        split="test", view="aligned", artifacts_dir=ARTIFACTS_DIR, raw_data_dir=RAW_DIR, return_metadata=True
    )
    ds_indep = ColoredMNISTDataset.from_artifacts(
        split="test", view="independent", artifacts_dir=ARTIFACTS_DIR, raw_data_dir=RAW_DIR, return_metadata=True
    )
    ds_swapped = ColoredMNISTDataset.from_artifacts(
        split="test", view="swapped", artifacts_dir=ARTIFACTS_DIR, raw_data_dir=RAW_DIR, return_metadata=True
    )
    assert len(ds_aligned) == len(ds_indep) == len(ds_swapped) == 10000

    for dataset in (ds_p01, ds_aligned, ds_indep, ds_swapped):
        images, _, metadata = next(iter(DataLoader(dataset, batch_size=4)))
        assert images.shape == (4, 3, 28, 28)
        assert list(metadata["original_id"]) == [entry.original_id for entry in dataset.entries[:4]]
        expected_targets = [
            entry.swapped_target_class if entry.swapped_target_class is not None else -1
            for entry in dataset.entries[:4]
        ]
        assert metadata["swapped_target_class"].tolist() == expected_targets

    for idx in (0, 17, 500, 9999):
        xa, ya, ma = ds_aligned[idx]
        xi, yi, mi = ds_indep[idx]
        xs, ys, ms = ds_swapped[idx]
        assert ma["original_id"] == mi["original_id"] == ms["original_id"]
        assert ya == yi == ys
        assert ms["swapped_target_class"] is not None
        assert ms["swapped_target_class"] != ys
        assert xa.shape == xi.shape == xs.shape == (3, 28, 28)


@pytest.mark.skipif(
    not (ARTIFACTS_DIR / "manifests" / "heldout_views.csv").exists(),
    reason="Step 1 artifacts must be generated first via scripts/build_step1_dataset.py",
)
def test_artifact_loader_uses_saved_palette(tmp_path: Path) -> None:
    custom_palette = build_palette(replace(DEFAULT_CONFIG, lab_chroma=20.0))
    (tmp_path / "palette.json").write_text(json.dumps(custom_palette.to_dict()), encoding="utf-8")
    (tmp_path / "manifests").mkdir()
    shutil.copyfile(
        ARTIFACTS_DIR / "manifests" / "heldout_views.csv",
        tmp_path / "manifests" / "heldout_views.csv",
    )

    dataset = ColoredMNISTDataset.from_artifacts(
        split="val", view="aligned", artifacts_dir=tmp_path, raw_data_dir=RAW_DIR
    )
    entry = dataset.entries[0]
    mask, _ = dataset.raw.get_image_and_label(entry.source_split, entry.source_index)
    color = custom_palette.colors[entry.color_id]
    assert np.array_equal(dataset.palette.bg_linear_table, custom_palette.bg_linear_table)
    assert np.array_equal(
        dataset.render_uint8(0), composite_digit_srgb(mask, color.linear_rgb_quantized)
    )

