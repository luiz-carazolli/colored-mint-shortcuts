"""Audit and document Step 2 Section 1: Experimental matrix and data ingestion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.matrix import (
    RunSpec,
    build_experimental_matrix,
    create_dataloader,
    get_training_dataset,
    get_validation_datasets,
    save_experimental_matrix_csv,
)


def audit_data_ingestion(
    config: DatasetConfig = DEFAULT_CONFIG,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit all 21 experimental matrix runs and dual validation views for data ingestion compliance."""
    step1_dir = PROJECT_ROOT / config.output_dir
    raw_dir = PROJECT_ROOT / config.raw_data_dir
    step2_dir = output_dir or (PROJECT_ROOT / "artifacts" / "step_2_training")
    step2_dir.mkdir(parents=True, exist_ok=True)

    print("Building experimental matrix for all 21 runs...")
    run_specs = build_experimental_matrix(
        artifacts_dir=step1_dir,
        raw_data_dir=raw_dir,
        config=config,
    )
    save_experimental_matrix_csv(run_specs, step2_dir / "experimental_matrix.csv")

    matrix_audit: list[dict[str, Any]] = []
    pairing_verification: dict[int, dict[str, Any]] = {}

    # 1. Audit training runs
    print("Auditing 21 training data loaders and tensor batches...")
    for spec in run_specs:
        ds = get_training_dataset(
            spec,
            artifacts_dir=step1_dir,
            raw_data_dir=raw_dir,
            config=config,
            return_metadata=True,
        )
        assert len(ds) == 54000, f"Expected 54,000 samples, got {len(ds)} for {spec.run_id}"
        assert ds.raw.test_images is None and ds.raw.test_labels is None
        assert all(entry.source_split == "train" for entry in ds.entries)

        # Test DataLoader batching
        loader = create_dataloader(ds, batch_size=256, shuffle=False)
        first_batch = next(iter(loader))
        images, labels, metadata = first_batch

        # Verify tensor properties
        assert images.shape == (256, 3, 28, 28), f"Unexpected shape {images.shape}"
        assert images.dtype == torch.float32, f"Unexpected dtype {images.dtype}"
        assert torch.isfinite(images).all(), "Images contain NaN/Inf"
        min_val = float(images.min())
        max_val = float(images.max())
        assert 0.0 <= min_val <= max_val <= 1.0, f"Values out of bounds: [{min_val}, {max_val}]"

        assert labels.shape == (256,)
        assert labels.dtype == torch.int64
        unique_labels = torch.unique(labels).tolist()
        assert all(0 <= lbl <= 9 for lbl in unique_labels)

        # Record audit details
        matrix_audit.append(
            {
                "run_id": spec.run_id,
                "p": spec.p,
                "dataset_seed": spec.dataset_seed,
                "train_seed": spec.train_seed,
                "condition_type": spec.condition_type,
                "num_samples": spec.num_train_samples,
                "expected_counterexamples": spec.expected_counterexamples,
                "realized_counterexamples": spec.realized_counterexamples,
                "realized_preferred_count": spec.realized_preferred_count,
                "realized_preferred_rate": round(spec.realized_preferred_rate, 6),
                "first_batch_shape": list(images.shape),
                "first_batch_min": round(min_val, 4),
                "first_batch_max": round(max_val, 4),
            }
        )

    # 2. Verify pairing invariant within each dataset seed:
    # Digit shapes and original IDs must be strictly identical across all 7 p values!
    print("Verifying pairing invariants across correlation levels within each seed...")
    for d_seed in config.dataset_seeds:
        specs_for_seed = [s for s in run_specs if s.dataset_seed == d_seed]
        base_spec = specs_for_seed[0]
        base_ds = get_training_dataset(
            base_spec,
            artifacts_dir=step1_dir,
            raw_data_dir=raw_dir,
            config=config,
            return_metadata=True,
        )
        base_ids = [e.original_id for e in base_ds.entries]
        base_labels = [e.label for e in base_ds.entries]
        base_sources = [(e.source_split, e.source_index) for e in base_ds.entries]

        for other_spec in specs_for_seed[1:]:
            other_ds = get_training_dataset(
                other_spec,
                artifacts_dir=step1_dir,
                raw_data_dir=raw_dir,
                config=config,
                return_metadata=True,
            )
            other_ids = [e.original_id for e in other_ds.entries]
            other_labels = [e.label for e in other_ds.entries]
            assert base_ids == other_ids, (
                f"Original ID sequence mismatch between {base_spec.run_id} and {other_spec.run_id}"
            )
            assert base_labels == other_labels, (
                f"Label sequence mismatch between {base_spec.run_id} and {other_spec.run_id}"
            )
            assert base_sources == [(e.source_split, e.source_index) for e in other_ds.entries], (
                f"Source image sequence mismatch between {base_spec.run_id} and {other_spec.run_id}"
            )

        pairing_verification[d_seed] = {
            "p_values_verified": [s.p for s in specs_for_seed],
            "sample_count": len(base_ids),
            "original_ids_identical": True,
            "labels_identical": True,
            "source_images_identical": True,
            "only_background_color_varies": True,
        }

    # 3. Audit validation datasets
    print("Auditing validation views (val_aligned and val_independent)...")
    val_datasets = get_validation_datasets(
        artifacts_dir=step1_dir,
        raw_data_dir=raw_dir,
        config=config,
        return_metadata=True,
    )
    val_audit: dict[str, Any] = {}
    for v_name, v_ds in val_datasets.items():
        assert len(v_ds) == 6000, f"Expected 6,000 val samples, got {len(v_ds)} for {v_name}"
        assert v_ds.raw.test_images is None and v_ds.raw.test_labels is None
        assert all(entry.source_split == "train" for entry in v_ds.entries)
        loader = create_dataloader(v_ds, batch_size=256, shuffle=False)
        v_images, v_labels, _ = next(iter(loader))
        assert v_images.shape == (256, 3, 28, 28)
        assert v_images.dtype == torch.float32
        assert torch.isfinite(v_images).all()
        assert 0.0 <= float(v_images.min()) <= float(v_images.max()) <= 1.0

        pref_count = sum(
            1 for e in v_ds.entries if e.color_id == e.preferred_color_id
        )
        val_audit[v_name] = {
            "num_samples": len(v_ds),
            "preferred_count": pref_count,
            "counterexamples": len(v_ds) - pref_count,
            "preferred_rate": round(pref_count / len(v_ds), 4),
            "batch_shape": list(v_images.shape),
        }

    print("Verified that audited datasets contain only official training data.")

    full_report: dict[str, Any] = {
        "status": "PASSED",
        "section": "Step 2 - Section 1: Experimental matrix and data ingestion",
        "num_matrix_runs": len(run_specs),
        "dataset_seeds": list(config.dataset_seeds),
        "p_values": list(config.p_values),
        "seed_pairs": [
            {"dataset_seed": s, "train_seed": s + 100} for s in config.dataset_seeds
        ],
        "training_split_size": 54000,
        "validation_split_size": 6000,
        "test_embargo_quarantined": True,
        "tensor_contract": {
            "verification_scope": "First batch of each training run and validation view",
            "shape": "(B, 3, 28, 28)",
            "dtype": "torch.float32",
            "range": "[0.0, 1.0]",
            "labels": "int64 in {0, ..., 9}",
            "augmentations": "NONE (pure signal preservation)",
            "normalizations": "NONE (absolute sRGB scale preserved)",
        },
        "validation_views": val_audit,
        "pairing_verification": pairing_verification,
        "runs": matrix_audit,
    }

    report_path = step2_dir / "data_ingestion_audit.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        f.write("\n")

    print(f"Data ingestion audit report saved to: {report_path}")
    print(f"Experimental matrix CSV saved to: {step2_dir / 'experimental_matrix.csv'}")
    return full_report


if __name__ == "__main__":
    audit_data_ingestion()

