"""Tests for Step 2 Section 1: Experimental matrix and data ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.matrix import (
    build_experimental_matrix,
    create_dataloader,
    get_condition_type,
    get_paired_train_seed,
    get_training_dataset,
    get_validation_datasets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
RAW_DIR = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir


@pytest.mark.skipif(
    not (STEP1_DIR / "manifests" / "splits.csv").exists(),
    reason="Step 1 artifacts must be present to test Step 2 data ingestion",
)
def test_experimental_matrix_structure() -> None:
    specs = build_experimental_matrix(
        artifacts_dir=STEP1_DIR,
        raw_data_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
    )
    assert len(specs) == 21, f"Expected 21 runs (7 p * 3 seeds), got {len(specs)}"

    # Check that each seed pair and p is represented exactly once
    run_ids = {s.run_id for s in specs}
    assert len(run_ids) == 21

    for spec in specs:
        assert spec.dataset_seed in DEFAULT_CONFIG.dataset_seeds
        assert spec.train_seed == get_paired_train_seed(spec.dataset_seed)
        assert spec.p in DEFAULT_CONFIG.p_values
        assert spec.num_train_samples == 54000
        assert spec.realized_preferred_count + spec.realized_counterexamples == 54000
        assert 0.0 <= spec.realized_preferred_rate <= 1.0

        if spec.p == 0.10:
            assert spec.condition_type == "unbiased_baseline"
            # In unbiased baseline, ~90% are counterexamples
            assert 48000 <= spec.realized_counterexamples <= 49000
        elif spec.p == 1.00:
            assert spec.condition_type == "fully_confounded"
            assert spec.realized_counterexamples == 0
            assert spec.realized_preferred_count == 54000
        else:
            assert spec.condition_type == "biased"


@pytest.mark.skipif(
    not (STEP1_DIR / "manifests" / "splits.csv").exists(),
    reason="Step 1 artifacts must be present to test Step 2 data ingestion",
)
def test_training_dataloader_tensor_contract() -> None:
    specs = build_experimental_matrix(
        artifacts_dir=STEP1_DIR,
        raw_data_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
    )
    # Test first run (p=0.10, seed 101) and last run (p=1.00, seed 103)
    for test_spec in (specs[0], specs[-1]):
        ds = get_training_dataset(
            test_spec,
            artifacts_dir=STEP1_DIR,
            raw_data_dir=RAW_DIR,
            config=DEFAULT_CONFIG,
            return_metadata=False,
        )
        assert len(ds) == 54000
        loader = create_dataloader(ds, batch_size=256, shuffle=False)

        # Ingest first batch
        batch_images, batch_labels = next(iter(loader))
        assert batch_images.shape == (256, 3, 28, 28)
        assert batch_images.dtype == torch.float32
        assert torch.isfinite(batch_images).all()
        assert 0.0 <= float(batch_images.min()) <= float(batch_images.max()) <= 1.0

        assert batch_labels.shape == (256,)
        assert batch_labels.dtype == torch.int64
        assert (batch_labels >= 0).all() and (batch_labels <= 9).all()


@pytest.mark.skipif(
    not (STEP1_DIR / "manifests" / "splits.csv").exists(),
    reason="Step 1 artifacts must be present to test Step 2 data ingestion",
)
def test_dual_validation_ingestion() -> None:
    val_map = get_validation_datasets(
        artifacts_dir=STEP1_DIR,
        raw_data_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
        return_metadata=True,
    )
    assert set(val_map.keys()) == {"val_aligned", "val_independent"}
    assert len(val_map["val_aligned"]) == 6000
    assert len(val_map["val_independent"]) == 6000

    # Test aligned view has 100% preferred colors
    val_aligned_pref = sum(
        1 for e in val_map["val_aligned"].entries if e.color_id == e.preferred_color_id
    )
    assert val_aligned_pref == 6000

    # Test independent view has ~10% preferred colors
    val_indep_pref = sum(
        1 for e in val_map["val_independent"].entries if e.color_id == e.preferred_color_id
    )
    assert 550 <= val_indep_pref <= 650

    loader = create_dataloader(val_map["val_aligned"], batch_size=256, shuffle=False)
    v_imgs, v_lbls, _ = next(iter(loader))
    assert v_imgs.shape == (256, 3, 28, 28)
    assert v_imgs.dtype == torch.float32
    assert 0.0 <= float(v_imgs.min()) <= float(v_imgs.max()) <= 1.0


@pytest.mark.skipif(
    not (STEP1_DIR / "manifests" / "splits.csv").exists(),
    reason="Step 1 artifacts must be present to test Step 2 data ingestion",
)
def test_paired_runs_share_identical_digits_and_masks() -> None:
    specs = build_experimental_matrix(
        artifacts_dir=STEP1_DIR,
        raw_data_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
    )
    # Check seed 101 across p=0.10 and p=0.99
    spec_01 = [s for s in specs if s.dataset_seed == 101 and s.p == 0.10][0]
    spec_99 = [s for s in specs if s.dataset_seed == 101 and s.p == 0.99][0]

    ds_01 = get_training_dataset(
        spec_01, artifacts_dir=STEP1_DIR, raw_data_dir=RAW_DIR, config=DEFAULT_CONFIG, return_metadata=True
    )
    ds_99 = get_training_dataset(
        spec_99, artifacts_dir=STEP1_DIR, raw_data_dir=RAW_DIR, config=DEFAULT_CONFIG, return_metadata=True
    )

    for i in (0, 100, 1000, 53999):
        e01 = ds_01.entries[i]
        e99 = ds_99.entries[i]
        assert e01.original_id == e99.original_id
        assert e01.label == e99.label
        assert e01.source_index == e99.source_index

