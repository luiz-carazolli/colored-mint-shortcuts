"""Tests for Step 3 Substep 1: Freezing inputs and verifying paired test sets."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.evaluate import (
    freeze_and_verify_inputs,
    verify_paired_test_datasets,
    verify_step1_and_raw_provenance,
    verify_sweep_registry,
)
from colored_mnist.train import read_checkpoint, write_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / "artifacts" / "step_1_dataset"
STEP2_DIR = PROJECT_ROOT / "artifacts" / "step_2_training"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

pytestmark = pytest.mark.usefixtures("trained_sweep")


def test_real_frozen_inputs_and_test_pairs_pass() -> None:
    """Verify the real Step 1, Step 2, raw IDX, and paired test contracts pass strictly."""
    frozen = freeze_and_verify_inputs(
        step1_dir=STEP1_DIR,
        step2_dir=STEP2_DIR,
        raw_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
    )
    assert frozen.status == "PASSED"
    assert len(frozen.sweep_runs) == 21
    assert len(frozen.excluded_runs) == 1
    assert frozen.excluded_runs[0]["role"] == "pilot"

    for run in frozen.sweep_runs:
        assert run.completed_epoch == 10
        assert run.global_step == 2110
        assert len(run.checkpoint_sha256) == 64

    summary = frozen.test_pairs_summary
    assert summary["test_count_per_view"] == 10_000
    assert summary["views"] == ["aligned", "independent", "swapped"]
    assert summary["identical_ids_labels_sources_verified"] is True
    assert summary["disjoint_from_train_and_val_verified"] is True
    assert summary["swapped_targets_distinct_verified"] is True
    assert summary["swapped_color_belongs_to_target_verified"] is True
    assert set(summary["independent_color_counts"]) == {str(i) for i in range(10)}


def test_mismatch_failures_on_copied_fixtures(tmp_path: Path) -> None:
    """Ensure negative checks abort loudly on corrupted provenance, registry, or test pairs."""
    # Lightweight copy of Step 1 metadata/manifests (without duplicating giant raw IDX files)
    s1 = tmp_path / "step1"
    shutil.copytree(STEP1_DIR / "manifests", s1 / "manifests")
    shutil.copy2(STEP1_DIR / "palette.json", s1 / "palette.json")
    shutil.copy2(STEP1_DIR / "dataset_metadata.json", s1 / "dataset_metadata.json")

    palette, s1_hashes, raw_hashes = verify_step1_and_raw_provenance(
        step1_dir=s1,
        raw_dir=RAW_DIR,
        config=DEFAULT_CONFIG,
    )

    # 1. Registry with incomplete sweep status or missing matrix combination must fail
    s2 = tmp_path / "step2"
    shutil.copytree(STEP2_DIR / "checkpoints", s2 / "checkpoints")
    entries = json.loads((STEP2_DIR / "runs_manifest.json").read_text(encoding="utf-8"))

    bad_entries = json.loads(json.dumps(entries))
    for entry in bad_entries:
        if entry["role"] == "sweep":
            entry["status"] = "running"
            break
    (s2 / "runs_manifest.json").write_text(json.dumps(bad_entries), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        verify_sweep_registry(s2, s1_hashes, raw_hashes, DEFAULT_CONFIG)

    # 2. Checkpoint config_sha256 / epoch mismatch must fail
    (s2 / "runs_manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    first_sweep = next(e for e in entries if e["role"] == "sweep")
    ckpt_file = s2 / first_sweep["checkpoint_path"]
    saved = read_checkpoint(ckpt_file)
    saved["global_step"] = 2000
    write_atomic(ckpt_file, saved, checkpoint=True)
    with pytest.raises(ValueError, match="global_step"):
        verify_sweep_registry(s2, s1_hashes, raw_hashes, DEFAULT_CONFIG)

    # 3. Corrupted swapped target in heldout_views.csv must stop before scoring
    heldout_path = s1 / "manifests" / "heldout_views.csv"
    with heldout_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    for row in rows:
        if row["split"] == "test":
            row["swapped_target_class"] = row["label"]
            break
    with heldout_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Swapped target"):
        verify_paired_test_datasets(s1, RAW_DIR, palette, DEFAULT_CONFIG)

    # All views share the same manifest, so cross-view agreement alone misses this.
    for row in rows:
        if row["split"] == "test":
            row["swapped_target_class"] = str(palette.swapped_target_by_digit[int(row["label"])])
            row["source_index"] = str((int(row["source_index"]) + 1) % 10_000)
            break
    with heldout_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Invalid official test source reference"):
        verify_paired_test_datasets(s1, RAW_DIR, palette, DEFAULT_CONFIG)
