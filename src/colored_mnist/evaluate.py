"""Frozen input verification and paired test-pair checks for Step 3 evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from colored_mnist.assignments import HeldoutViewRecord, load_heldout_manifest
from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.matrix import build_run_id, get_paired_train_seed
from colored_mnist.models import build_model
from colored_mnist.palette import Palette, load_palette
from colored_mnist.splits import load_split_manifest, make_original_id
from colored_mnist.train import (
    BATCH_SIZE,
    CHECKPOINT_VERSION,
    EPOCHS,
    TrainingSetup,
    configure_runtime,
    file_digest,
    json_digest,
    read_checkpoint,
    write_csv,
    write_json,
)

EXPECTED_TEST_COUNT = 10_000
EXPECTED_UPDATES = 2_110
EVALUATOR_VERSION = 1
TEST_VIEWS = ("aligned", "independent", "swapped")

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "aligned_accuracy": {
        "numerator": "Count of test items where aligned_pred == label",
        "denominator": "All 10,000 aligned test digits",
    },
    "independent_accuracy": {
        "numerator": "Count of test items where independent_pred == label",
        "denominator": "All 10,000 independent test digits",
    },
    "swapped_accuracy": {
        "numerator": "Count of test items where swapped_pred == label",
        "denominator": "All 10,000 swapped test digits",
    },
    "overall_flip_rate": {
        "numerator": "Count of aligned-swapped pairs where aligned_pred != swapped_pred",
        "denominator": "All 10,000 aligned-swapped test pairs",
    },
    "targeted_flip_rate": {
        "numerator": "Count of pairs where aligned_pred == label and swapped_pred == swapped_target_class",
        "denominator": "Count of aligned-correct pairs where aligned_pred == label (missing if denominator == 0)",
    },
}


@dataclass(frozen=True)
class FrozenRunRecord:
    """Verified Step 2 sweep checkpoint bound to the Step 3 evaluation protocol."""

    run_id: str
    p: float
    dataset_seed: int
    train_seed: int
    condition_type: str
    completed_epoch: int
    global_step: int
    config_sha256: str
    checkpoint_relpath: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class FrozenEvaluationInputs:
    """Complete frozen input bundle verified prior to test-split inference."""

    status: str
    step1_dir: str
    step2_dir: str
    raw_dir: str
    registry_sha256: str
    step1_hashes: dict[str, str]
    raw_idx_hashes: dict[str, str]
    excluded_runs: list[dict[str, Any]]
    sweep_runs: list[FrozenRunRecord]
    test_pairs_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _require_sha256(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file missing ({label}): {path}")
    return file_digest(path)


def verify_step1_and_raw_provenance(
    step1_dir: Path,
    raw_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[Palette, dict[str, str], dict[str, str]]:
    """Verify Step 1 manifests/palette and raw MNIST IDX hashes against dataset_metadata.json."""
    palette_path = step1_dir / "palette.json"
    metadata_path = step1_dir / "dataset_metadata.json"
    splits_path = step1_dir / "manifests" / "splits.csv"
    heldout_path = step1_dir / "manifests" / "heldout_views.csv"

    palette = load_palette(palette_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_raw_hashes: dict[str, str] = metadata.get("raw_idx_sha256", {})

    required_idx = (
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    )
    raw_idx_hashes: dict[str, str] = {}
    raw_mnist_dir = raw_dir / "MNIST" / "raw"
    for filename in required_idx:
        if filename not in expected_raw_hashes:
            raise ValueError(f"dataset_metadata.json missing expected SHA256 for {filename}")
        actual = _require_sha256(raw_mnist_dir / filename, f"raw IDX {filename}")
        if actual != expected_raw_hashes[filename]:
            raise ValueError(
                f"Raw IDX SHA256 mismatch for {filename}: "
                f"expected {expected_raw_hashes[filename]}, got {actual}"
            )
        raw_idx_hashes[filename] = actual

    step1_hashes: dict[str, str] = {
        "palette.json": _require_sha256(palette_path, "palette"),
        "dataset_metadata.json": _require_sha256(metadata_path, "dataset_metadata"),
        "manifests/splits.csv": _require_sha256(splits_path, "splits manifest"),
        "manifests/heldout_views.csv": _require_sha256(heldout_path, "heldout_views manifest"),
    }
    for d_seed in config.dataset_seeds:
        rel = f"manifests/train_seed_{d_seed}.csv"
        step1_hashes[rel] = _require_sha256(step1_dir / rel, f"train manifest seed {d_seed}")

    return palette, step1_hashes, raw_idx_hashes


def verify_sweep_registry(
    step2_dir: Path,
    step1_hashes: dict[str, str],
    raw_idx_hashes: dict[str, str],
    config: DatasetConfig = DEFAULT_CONFIG,
    *,
    expected_epoch: int = EPOCHS,
    expected_updates: int = EXPECTED_UPDATES,
) -> tuple[str, list[FrozenRunRecord], list[dict[str, Any]]]:
    """Audit runs_manifest.json and verify all 21 sweep checkpoints against provenance & matrix."""
    manifest_path = step2_dir / "runs_manifest.json"
    registry_sha256 = _require_sha256(manifest_path, "runs_manifest")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("runs_manifest.json must contain a JSON list")

    expected_keys = {
        (round(float(p), 6), int(d_seed), get_paired_train_seed(int(d_seed))): build_run_id(
            float(p), int(d_seed), get_paired_train_seed(int(d_seed))
        )
        for d_seed in config.dataset_seeds
        for p in config.p_values
    }

    sweep_entries: dict[str, dict[str, Any]] = {}
    excluded_runs: list[dict[str, Any]] = []
    seen_combo_keys: set[tuple[float, int, int]] = set()

    for entry in entries:
        role = entry.get("role")
        run_id = entry.get("run_id")
        if role == "pilot":
            excluded_runs.append(
                {
                    "run_id": run_id,
                    "role": role,
                    "status": entry.get("status"),
                    "reason": "Pilot run excluded from Step 3 test evaluation summary.",
                }
            )
            continue
        if role != "sweep":
            raise ValueError(f"Unexpected run role {role!r} for run_id={run_id!r}")

        if run_id in sweep_entries:
            raise ValueError(f"Duplicate sweep run_id in registry: {run_id!r}")

        combo = (round(float(entry["p"]), 6), int(entry["dataset_seed"]), int(entry["train_seed"]))
        if combo not in expected_keys:
            raise ValueError(f"Unexpected experimental matrix combination in registry: {combo}")
        if combo in seen_combo_keys:
            raise ValueError(f"Duplicate experimental matrix combination in registry: {combo}")
        if run_id != expected_keys[combo]:
            raise ValueError(
                f"Registry run_id mismatch for {combo}: expected {expected_keys[combo]!r}, got {run_id!r}"
            )
        seen_combo_keys.add(combo)
        sweep_entries[run_id] = entry

    missing_combos = set(expected_keys) - seen_combo_keys
    if missing_combos:
        raise ValueError(f"Missing experimental matrix entries in registry: {sorted(missing_combos)}")

    frozen_runs: list[FrozenRunRecord] = []
    # Sort deterministically by (dataset_seed, p)
    ordered_entries = sorted(
        sweep_entries.values(),
        key=lambda e: (int(e["dataset_seed"]), float(e["p"])),
    )
    checkpoints_root = (step2_dir / "checkpoints").resolve()

    for entry in ordered_entries:
        run_id = entry["run_id"]
        if entry.get("status") != "complete":
            raise ValueError(f"Sweep run {run_id} is not complete: status={entry.get('status')!r}")
        if int(entry.get("completed_epoch", -1)) != expected_epoch:
            raise ValueError(
                f"Sweep run {run_id} expected completed_epoch={expected_epoch}, "
                f"got {entry.get('completed_epoch')}"
            )
        if int(entry.get("global_step", -1)) != expected_updates:
            raise ValueError(
                f"Sweep run {run_id} expected global_step={expected_updates}, "
                f"got {entry.get('global_step')}"
            )

        relpath = str(entry["checkpoint_path"])
        ckpt_path = (step2_dir / relpath).resolve()
        if not ckpt_path.is_relative_to(checkpoints_root):
            raise ValueError(f"Checkpoint path escapes checkpoints directory: {ckpt_path}")
        ckpt_sha256 = _require_sha256(ckpt_path, f"checkpoint {run_id}")

        saved = read_checkpoint(ckpt_path)
        _verify_checkpoint_contents(
            saved=saved,
            entry=entry,
            step1_hashes=step1_hashes,
            raw_idx_hashes=raw_idx_hashes,
            expected_epoch=expected_epoch,
            expected_updates=expected_updates,
        )

        frozen_runs.append(
            FrozenRunRecord(
                run_id=run_id,
                p=float(entry["p"]),
                dataset_seed=int(entry["dataset_seed"]),
                train_seed=int(entry["train_seed"]),
                condition_type=str(entry["condition_type"]),
                completed_epoch=int(saved["completed_epoch"]),
                global_step=int(saved["global_step"]),
                config_sha256=str(saved["config_sha256"]),
                checkpoint_relpath=relpath,
                checkpoint_sha256=ckpt_sha256,
            )
        )

    return registry_sha256, frozen_runs, excluded_runs


def _verify_checkpoint_contents(
    *,
    saved: dict[str, Any],
    entry: dict[str, Any],
    step1_hashes: dict[str, str],
    raw_idx_hashes: dict[str, str],
    expected_epoch: int,
    expected_updates: int,
) -> None:
    run_id = entry["run_id"]
    if saved.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{run_id}: unexpected checkpoint version {saved.get('version')!r}")
    if int(saved.get("completed_epoch", -1)) != expected_epoch:
        raise ValueError(
            f"{run_id}: checkpoint completed_epoch={saved.get('completed_epoch')} != {expected_epoch}"
        )
    if int(saved.get("global_step", -1)) != expected_updates:
        raise ValueError(
            f"{run_id}: checkpoint global_step={saved.get('global_step')} != {expected_updates}"
        )

    cfg = saved.get("config", {})
    cfg_sha = saved.get("config_sha256")
    if cfg_sha != entry.get("config_sha256"):
        raise ValueError(
            f"{run_id}: config_sha256 mismatch between checkpoint ({cfg_sha}) and registry ({entry.get('config_sha256')})"
        )
    if json_digest(cfg) != cfg_sha:
        raise ValueError(f"{run_id}: checkpoint config JSON does not match checkpoint config_sha256")
    if cfg.get("run_id") != run_id:
        raise ValueError(f"{run_id}: checkpoint config run_id={cfg.get('run_id')!r} mismatch")

    cond = cfg.get("condition", {})
    if (
        round(float(cond.get("p", -1.0)), 6) != round(float(entry["p"]), 6)
        or int(cond.get("dataset_seed", -1)) != int(entry["dataset_seed"])
        or int(cond.get("train_seed", -1)) != int(entry["train_seed"])
    ):
        raise ValueError(f"{run_id}: checkpoint condition metadata mismatch against registry: {cond}")

    data_prov = saved.get("data_sha256", {})
    expected_prov_keys = {
        "palette.json": step1_hashes["palette.json"],
        "manifests/splits.csv": step1_hashes["manifests/splits.csv"],
        f"manifests/train_seed_{int(entry['dataset_seed'])}.csv": step1_hashes[
            f"manifests/train_seed_{int(entry['dataset_seed'])}.csv"
        ],
        "MNIST/raw/train-images-idx3-ubyte": raw_idx_hashes["train-images-idx3-ubyte"],
        "MNIST/raw/train-labels-idx1-ubyte": raw_idx_hashes["train-labels-idx1-ubyte"],
    }
    for key, expected_digest in expected_prov_keys.items():
        if data_prov.get(key) != expected_digest:
            raise ValueError(
                f"{run_id}: provenance hash mismatch for {key}: "
                f"checkpoint has {data_prov.get(key)!r}, expected {expected_digest!r}"
            )


def verify_paired_test_datasets(
    step1_dir: Path,
    raw_dir: Path,
    palette: Palette,
    config: DatasetConfig = DEFAULT_CONFIG,
    *,
    expected_test_count: int = EXPECTED_TEST_COUNT,
) -> dict[str, Any]:
    """Load all three official test views and verify one-to-one pairing & label/color invariants."""
    splits = load_split_manifest(step1_dir / "manifests" / "splits.csv")
    train_ids = {r.original_id for r in splits if r.split == "train"}
    val_ids = {r.original_id for r in splits if r.split == "val"}
    test_records = [r for r in splits if r.split == "test"]
    test_by_id = {r.original_id: r for r in test_records}
    test_ids = set(test_by_id)

    if len(test_records) != len(test_ids) or len(test_ids) != expected_test_count:
        raise ValueError(
            f"Expected {expected_test_count} unique test IDs in splits.csv, "
            f"found {len(test_ids)} unique across {len(test_records)} rows"
        )
    if (test_ids & train_ids) or (test_ids & val_ids) or (train_ids & val_ids):
        raise ValueError("Split IDs are not strictly disjoint across train, val, and test")
    expected_ids = {make_original_id("test", idx) for idx in range(expected_test_count)}
    if test_ids != expected_ids:
        raise ValueError("splits.csv does not contain the canonical official test IDs")
    if any(r.source_split == "test" for r in splits if r.split != "test"):
        raise ValueError("Official test source referenced outside the test split")

    heldout_test: list[HeldoutViewRecord] = load_heldout_manifest(
        step1_dir / "manifests" / "heldout_views.csv",
        split="test",
    )
    heldout_by_id = {r.original_id: r for r in heldout_test}
    if len(heldout_test) != len(heldout_by_id) or set(heldout_by_id) != test_ids:
        raise ValueError("heldout_views.csv test rows do not match splits.csv test IDs one-to-one")

    # Load all three views via the canonical artifact loader with metadata enabled
    datasets: dict[str, ColoredMNISTDataset] = {
        view: ColoredMNISTDataset.from_artifacts(
            split="test",
            view=view,
            artifacts_dir=step1_dir,
            raw_data_dir=raw_dir,
            config=config,
            return_metadata=True,
        )
        for view in TEST_VIEWS
    }

    records_by_view: dict[str, dict[str, dict[str, Any]]] = {}
    for view, ds in datasets.items():
        if len(ds) != expected_test_count:
            raise ValueError(
                f"Test view {view!r} expected {expected_test_count} items, got {len(ds)}"
            )
        view_map: dict[str, dict[str, Any]] = {}
        for idx in range(len(ds)):
            tensor, label, meta = ds[idx]
            if (tuple(tensor.shape) != (3, 28, 28) or tensor.dtype != torch.float32
                    or not ((tensor >= 0) & (tensor <= 1)).all()):
                raise ValueError(f"Unexpected tensor contract in {view!r} idx={idx}: {tensor.shape}, {tensor.dtype}")
            oid = str(meta["original_id"])
            if oid in view_map:
                raise ValueError(f"Duplicate original_id {oid!r} in test view {view!r}")
            if oid in train_ids or oid in val_ids:
                raise ValueError(f"Test original_id {oid!r} leaks into train/val splits")
            if int(meta["label"]) != int(label):
                raise ValueError(f"Metadata label mismatch for {oid!r} in view {view!r}")
            view_map[oid] = dict(meta)

        if set(view_map) != test_ids:
            missing = len(test_ids - set(view_map))
            extra = len(set(view_map) - test_ids)
            raise ValueError(
                f"Test view {view!r} ID set mismatch vs official test split "
                f"(missing={missing}, extra={extra})"
            )
        records_by_view[view] = view_map

    indep_histogram = {c: 0 for c in range(10)}
    indep_match_count = 0

    raw_labels = datasets["aligned"].raw.test_labels
    if raw_labels is None or len(raw_labels) != expected_test_count:
        raise ValueError("Unexpected number of official test labels")

    for oid in sorted(test_ids):
        aligned_meta = records_by_view["aligned"][oid]
        indep_meta = records_by_view["independent"][oid]
        swapped_meta = records_by_view["swapped"][oid]
        manifest_rec = heldout_by_id[oid]

        y = int(aligned_meta["label"])
        src_split = aligned_meta["source_split"]
        src_idx = int(aligned_meta["source_index"])

        if (src_split != "test" or not 0 <= src_idx < expected_test_count
                or oid != make_original_id("test", src_idx)):
            raise ValueError(f"Invalid official test source reference for {oid!r}")
        split_rec = test_by_id[oid]
        if (split_rec.source_split, split_rec.source_index, split_rec.label) != (
            src_split, src_idx, y
        ) or y != int(raw_labels[src_idx]):
            raise ValueError(f"Official test source or label mismatch for {oid!r}")
        for v_meta in (indep_meta, swapped_meta):
            if (
                int(v_meta["label"]) != y
                or v_meta["source_split"] != src_split
                or int(v_meta["source_index"]) != src_idx
            ):
                raise ValueError(f"Cross-view identity/label/source mismatch for {oid!r}")

        if (
            int(manifest_rec.label) != y
            or manifest_rec.source_split != src_split
            or int(manifest_rec.source_index) != src_idx
        ):
            raise ValueError(f"Manifest mismatch vs dataset for {oid!r}")

        # Verify view-specific color & target contracts
        pref_color = palette.preferred_color_by_digit[y]
        if int(aligned_meta["color_id"]) != pref_color or int(aligned_meta["swapped_target_class"]) != -1:
            raise ValueError(f"Aligned view contract violated for {oid!r}")

        indep_color = int(indep_meta["color_id"])
        if indep_color != manifest_rec.independent_color_id or int(indep_meta["swapped_target_class"]) != -1:
            raise ValueError(f"Independent view contract violated for {oid!r}")
        indep_histogram[indep_color] += 1
        if indep_color == pref_color:
            indep_match_count += 1

        swapped_target = int(swapped_meta["swapped_target_class"])
        expected_target = palette.swapped_target_by_digit[y]
        if swapped_target != expected_target or swapped_target == y:
            raise ValueError(
                f"Swapped target contract violated for {oid!r}: "
                f"target={swapped_target}, expected={expected_target}, label={y}"
            )
        expected_swapped_color = palette.preferred_color_by_digit[swapped_target]
        if int(swapped_meta["color_id"]) != expected_swapped_color:
            raise ValueError(
                f"Swapped color contract violated for {oid!r}: "
                f"color={swapped_meta['color_id']}, expected={expected_swapped_color}"
            )

    return {
        "views": list(TEST_VIEWS),
        "test_count_per_view": len(test_ids),
        "identical_ids_labels_sources_verified": True,
        "disjoint_from_train_and_val_verified": True,
        "aligned_preferred_match_verified": True,
        "independent_manifest_match_verified": True,
        "swapped_target_by_digit": list(palette.swapped_target_by_digit),
        "official_source_indices_and_labels_verified": True,
        "swapped_targets_distinct_verified": True,
        "swapped_color_belongs_to_target_verified": True,
        "independent_color_counts": {str(k): v for k, v in indep_histogram.items()},
        "independent_preferred_match_count": indep_match_count,
        "independent_preferred_match_rate": indep_match_count / len(test_ids),
        "sample_source_reference_first_id": sorted(test_ids)[0],
    }


def freeze_and_verify_inputs(
    *,
    step1_dir: Path,
    step2_dir: Path,
    raw_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
    output_path: Path | None = None,
    expected_epoch: int = EPOCHS,
    expected_updates: int = EXPECTED_UPDATES,
    expected_test_count: int = EXPECTED_TEST_COUNT,
) -> FrozenEvaluationInputs:
    """Verify all Step 1, Step 2, and paired test-view inputs before any test inference."""
    palette, step1_hashes, raw_idx_hashes = verify_step1_and_raw_provenance(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        config=config,
    )
    registry_sha256, frozen_runs, excluded_runs = verify_sweep_registry(
        step2_dir=step2_dir,
        step1_hashes=step1_hashes,
        raw_idx_hashes=raw_idx_hashes,
        config=config,
        expected_epoch=expected_epoch,
        expected_updates=expected_updates,
    )
    test_pairs_summary = verify_paired_test_datasets(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        palette=palette,
        config=config,
        expected_test_count=expected_test_count,
    )

    frozen = FrozenEvaluationInputs(
        status="PASSED",
        step1_dir=str(step1_dir),
        step2_dir=str(step2_dir),
        raw_dir=str(raw_dir),
        registry_sha256=registry_sha256,
        step1_hashes=step1_hashes,
        raw_idx_hashes=raw_idx_hashes,
        excluded_runs=excluded_runs,
        sweep_runs=frozen_runs,
        test_pairs_summary=test_pairs_summary,
    )
    if output_path is not None:
        write_json(output_path, frozen.to_dict())
    return frozen


def build_view_batches(
    dataset: ColoredMNISTDataset,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[torch.Tensor, list[dict[str, Any]]]]:
    """Materialize fixed-order evaluation batches (shuffle=False, last batch retained)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batches: list[tuple[torch.Tensor, list[dict[str, Any]]]] = []
    for images, _targets, metadata in loader:
        batch_size_actual = int(images.shape[0])
        records = [
            {
                "original_id": str(metadata["original_id"][idx]),
                "source_split": str(metadata["source_split"][idx]),
                "source_index": int(metadata["source_index"][idx]),
                "label": int(metadata["label"][idx]),
                "color_id": int(metadata["color_id"][idx]),
                "preferred_color_id": int(metadata["preferred_color_id"][idx]),
                "swapped_target_class": int(metadata["swapped_target_class"][idx]),
                "view": str(metadata["view"][idx]),
            }
            for idx in range(batch_size_actual)
        ]
        batches.append((images.to(dtype=torch.float32).contiguous(), records))
    if not batches:
        raise ValueError("Cannot run evaluation on an empty dataset")
    return batches


def predict_view_batches(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, list[dict[str, Any]]]],
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, Any]]:
    """Run deterministic float32 argmax inference over fixed-order batches."""
    model.eval()
    predictions: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for images, records in batches:
            logits = model(images.to(device=device, dtype=torch.float32))
            if logits.shape != (len(records), 10) or not torch.isfinite(logits).all():
                raise RuntimeError(f"Invalid or non-finite logits during inference: {logits.shape}")
            preds = logits.argmax(dim=1).tolist()
            for rec, pred in zip(records, preds, strict=True):
                oid = rec["original_id"]
                if oid in predictions:
                    raise ValueError(f"Duplicate original_id encountered during prediction: {oid!r}")
                predictions[oid] = {**rec, "pred": int(pred)}
    return predictions


def _make_metric_row(
    run: FrozenRunRecord,
    metric: str,
    numerator: int,
    denominator: int,
) -> dict[str, Any]:
    rate: float | None = None if denominator == 0 else numerator / denominator
    return {
        "run_id": run.run_id,
        "p": f"{run.p:.2f}",
        "dataset_seed": run.dataset_seed,
        "train_seed": run.train_seed,
        "condition_type": run.condition_type,
        "metric": metric,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": "" if rate is None else rate,
    }


def join_and_compute_metrics(
    run: FrozenRunRecord,
    view_predictions: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Perform strict one-to-one join on original_id and compute all five paired metrics."""
    for required_view in TEST_VIEWS:
        if required_view not in view_predictions:
            raise ValueError(f"{run.run_id}: missing predictions for view {required_view!r}")

    aligned = view_predictions["aligned"]
    independent = view_predictions["independent"]
    swapped = view_predictions["swapped"]

    aligned_ids = set(aligned)
    indep_ids = set(independent)
    swapped_ids = set(swapped)
    if not (aligned_ids == indep_ids == swapped_ids):
        raise ValueError(
            f"{run.run_id}: one-to-one join mismatch across views "
            f"(aligned={len(aligned_ids)}, independent={len(indep_ids)}, swapped={len(swapped_ids)})"
        )
    if not aligned_ids:
        raise ValueError(f"{run.run_id}: zero paired predictions to score")

    prediction_rows: list[dict[str, Any]] = []
    aligned_correct = 0
    independent_correct = 0
    swapped_correct = 0
    overall_flips = 0
    targeted_flips = 0

    for oid in aligned:
        a_rec = aligned[oid]
        i_rec = independent[oid]
        w_rec = swapped[oid]

        y = int(a_rec["label"])
        if int(i_rec["label"]) != y or int(w_rec["label"]) != y:
            raise ValueError(
                f"{run.run_id}: inconsistent true labels across views for {oid!r}: "
                f"{y}, {i_rec['label']}, {w_rec['label']}"
            )
        if (
            a_rec["source_split"] != i_rec["source_split"]
            or a_rec["source_split"] != w_rec["source_split"]
            or int(a_rec["source_index"]) != int(i_rec["source_index"])
            or int(a_rec["source_index"]) != int(w_rec["source_index"])
        ):
            raise ValueError(f"{run.run_id}: inconsistent source image reference for {oid!r}")

        s = int(w_rec["swapped_target_class"])
        if s == y or not (0 <= s < 10):
            raise ValueError(f"{run.run_id}: invalid swapped_target_class={s} for label={y} ({oid!r})")

        a_pred = int(a_rec["pred"])
        i_pred = int(i_rec["pred"])
        w_pred = int(w_rec["pred"])

        is_aligned_correct = a_pred == y
        aligned_correct += int(is_aligned_correct)
        independent_correct += int(i_pred == y)
        swapped_correct += int(w_pred == y)
        overall_flips += int(a_pred != w_pred)
        targeted_flips += int(is_aligned_correct and w_pred == s)

        prediction_rows.append(
            {
                "run_id": run.run_id,
                "p": f"{run.p:.2f}",
                "dataset_seed": run.dataset_seed,
                "train_seed": run.train_seed,
                "original_id": oid,
                "label": y,
                "swapped_target_class": s,
                "aligned_color_id": int(a_rec["color_id"]),
                "independent_color_id": int(i_rec["color_id"]),
                "swapped_color_id": int(w_rec["color_id"]),
                "aligned_pred": a_pred,
                "independent_pred": i_pred,
                "swapped_pred": w_pred,
            }
        )

    n_total = len(prediction_rows)
    if targeted_flips > overall_flips:
        raise RuntimeError(
            f"{run.run_id}: targeted_flips ({targeted_flips}) exceeded overall_flips ({overall_flips})"
        )

    metric_rows = [
        _make_metric_row(run, "aligned_accuracy", aligned_correct, n_total),
        _make_metric_row(run, "independent_accuracy", independent_correct, n_total),
        _make_metric_row(run, "swapped_accuracy", swapped_correct, n_total),
        _make_metric_row(run, "overall_flip_rate", overall_flips, n_total),
        _make_metric_row(run, "targeted_flip_rate", targeted_flips, aligned_correct),
    ]
    return prediction_rows, metric_rows


def load_frozen_model_for_inference(
    checkpoint_path: Path,
    train_seed: int,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Load only weights into the fixed Step 2 CNN with strict state-dict verification."""
    saved = read_checkpoint(checkpoint_path)
    model = build_model(train_seed).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    return model


def evaluate_sweep(
    *,
    step1_dir: Path,
    step2_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate the approved frozen sweep inputs and save paired predictions and metrics."""
    setup = TrainingSetup(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        device="cpu",
        threads=1,
    )
    environment = configure_runtime(setup)
    device = torch.device("cpu")

    freeze_path = output_dir / "input_freeze_audit.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Run the input freeze audit before evaluation: {freeze_path}")
    approved = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen = freeze_and_verify_inputs(
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        raw_dir=raw_dir,
        config=config,
    )
    # Normalize tuples to JSON arrays before comparing the saved audit record.
    if approved != json.loads(json.dumps(frozen.to_dict())):
        raise ValueError("Evaluation inputs differ from the saved input freeze audit")

    view_batches: dict[str, list[tuple[torch.Tensor, list[dict[str, Any]]]]] = {}
    for view in TEST_VIEWS:
        ds = ColoredMNISTDataset.from_artifacts(
            split="test",
            view=view,
            artifacts_dir=step1_dir,
            raw_data_dir=raw_dir,
            config=config,
            return_metadata=True,
        )
        view_batches[view] = build_view_batches(ds, batch_size=batch_size)

    all_prediction_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []

    for run in frozen.sweep_runs:
        ckpt_path = step2_dir / run.checkpoint_relpath
        if file_digest(ckpt_path) != run.checkpoint_sha256:
            raise ValueError(f"Checkpoint changed after input verification: {run.run_id}")
        model = load_frozen_model_for_inference(ckpt_path, run.train_seed, device=device)
        view_preds = {
            view: predict_view_batches(model, view_batches[view], device=device)
            for view in TEST_VIEWS
        }
        pred_rows, metric_rows = join_and_compute_metrics(run, view_preds)
        if len(pred_rows) != EXPECTED_TEST_COUNT:
            raise ValueError(f"{run.run_id}: expected {EXPECTED_TEST_COUNT} paired predictions")
        all_prediction_rows.extend(pred_rows)
        all_metric_rows.extend(metric_rows)

        summary_line = ", ".join(
            f"{m['metric']}={m['numerator']}/{m['denominator']}"
            + ("" if m["rate"] == "" else f" ({float(m['rate']):.4f})")
            for m in metric_rows
        )
        print(f"Evaluated {run.run_id}: {summary_line}", flush=True)

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    write_csv(predictions_path, all_prediction_rows)
    write_csv(metrics_path, all_metric_rows)

    return {
        "status": "COMPLETE",
        "num_runs": len(frozen.sweep_runs),
        "num_prediction_rows": len(all_prediction_rows),
        "num_metric_rows": len(all_metric_rows),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "environment": environment,
        "metrics": all_metric_rows,
    }


def _recompute_counts_from_predictions(
    predictions_path: Path,
    expected_runs: dict[str, FrozenRunRecord],
    expected_examples: dict[str, HeldoutViewRecord],
    expected_rows_per_run: int,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Read saved predictions.csv and independently recompute all five metric fractions per run."""
    required_cols = {
        "run_id", "p", "dataset_seed", "train_seed", "original_id",
        "label", "swapped_target_class",
        "aligned_color_id", "independent_color_id", "swapped_color_id",
        "aligned_pred", "independent_pred", "swapped_pred",
    }
    seen_ids_by_run: dict[str, set[str]] = {}
    tallies: dict[str, dict[str, int]] = {}

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(f"predictions.csv missing required columns: {reader.fieldnames}")
        for row in reader:
            rid = str(row["run_id"])
            oid = str(row["original_id"])
            run = expected_runs.get(rid)
            example = expected_examples.get(oid)
            if run is None or example is None:
                raise ValueError(f"Unexpected run or official test ID: {rid}/{oid}")
            if (float(row["p"]), int(row["dataset_seed"]), int(row["train_seed"])) != (
                run.p, run.dataset_seed, run.train_seed
            ):
                raise ValueError(f"Prediction run metadata mismatch for {rid}/{oid}")
            for field in ("label", "swapped_target_class", "aligned_color_id",
                          "independent_color_id", "swapped_color_id"):
                if int(row[field]) != getattr(example, field):
                    raise ValueError(f"Prediction metadata mismatch for {rid}/{oid}: {field}")
            seen = seen_ids_by_run.setdefault(rid, set())
            if oid in seen:
                raise ValueError(f"Duplicate original_id {oid!r} for run {rid!r} in predictions.csv")
            seen.add(oid)

            y = int(row["label"])
            s = int(row["swapped_target_class"])
            a = int(row["aligned_pred"])
            i = int(row["independent_pred"])
            w = int(row["swapped_pred"])
            for pred_col, pred_val in (("aligned_pred", a), ("independent_pred", i), ("swapped_pred", w)):
                if not (0 <= pred_val < 10):
                    raise ValueError(f"Invalid prediction {pred_val} in {pred_col} for {rid}/{oid}")
            if s == y or not (0 <= s < 10):
                raise ValueError(f"Invalid swapped_target_class={s} for label={y} in {rid}/{oid}")

            counts = tallies.setdefault(
                rid,
                {"n": 0, "a_ok": 0, "i_ok": 0, "w_ok": 0, "flip": 0, "tflip": 0},
            )
            counts["n"] += 1
            a_ok = a == y
            counts["a_ok"] += int(a_ok)
            counts["i_ok"] += int(i == y)
            counts["w_ok"] += int(w == y)
            counts["flip"] += int(a != w)
            counts["tflip"] += int(a_ok and w == s)

    if set(tallies) != set(expected_runs):
        raise ValueError(
            f"predictions.csv run set mismatch: expected {len(expected_runs)} runs, got {len(tallies)}"
        )

    fractions: dict[str, dict[str, tuple[int, int]]] = {}
    for rid in expected_runs:
        c = tallies[rid]
        if c["n"] != expected_rows_per_run:
            raise ValueError(
                f"Run {rid} expected {expected_rows_per_run} rows in predictions.csv, found {c['n']}"
            )
        if c["tflip"] > c["flip"]:
            raise ValueError(
                f"Run {rid}: targeted flip count ({c['tflip']}) exceeds overall flip count ({c['flip']})"
            )
        fractions[rid] = {
            "aligned_accuracy": (c["a_ok"], c["n"]),
            "independent_accuracy": (c["i_ok"], c["n"]),
            "swapped_accuracy": (c["w_ok"], c["n"]),
            "overall_flip_rate": (c["flip"], c["n"]),
            "targeted_flip_rate": (c["tflip"], c["a_ok"]),
        }
    return fractions


def _verify_metrics_csv_matches_predictions(
    metrics_path: Path,
    recomputed: dict[str, dict[str, tuple[int, int]]],
    expected_runs: dict[str, FrozenRunRecord],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare every row in metrics.csv against independently recomputed prediction counts."""
    expected_metrics = set(METRIC_DEFINITIONS)
    verified_summary: dict[str, dict[str, dict[str, Any]]] = {}

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    expected_count = len(recomputed) * len(expected_metrics)
    if len(rows) != expected_count:
        raise ValueError(f"metrics.csv expected {expected_count} rows, found {len(rows)}")

    for row in rows:
        rid = str(row["run_id"])
        metric = str(row["metric"])
        if rid not in recomputed or metric not in expected_metrics:
            raise ValueError(f"Unexpected run/metric in metrics.csv: {rid!r}, {metric!r}")
        run = expected_runs[rid]
        if (float(row["p"]), int(row["dataset_seed"]), int(row["train_seed"]), row["condition_type"]) != (
            run.p, run.dataset_seed, run.train_seed, run.condition_type
        ):
            raise ValueError(f"Metric run metadata mismatch for {rid}/{metric}")
        if metric in verified_summary.get(rid, {}):
            raise ValueError(f"Duplicate metric for {rid}/{metric}")
        num = int(row["numerator"])
        den = int(row["denominator"])
        exp_num, exp_den = recomputed[rid][metric]
        if (num, den) != (exp_num, exp_den):
            raise ValueError(
                f"Metric count mismatch for {rid}/{metric}: "
                f"metrics.csv has {num}/{den}, predictions.csv gives {exp_num}/{exp_den}"
            )
        rate_str = str(row["rate"]).strip()
        if den == 0:
            if rate_str != "":
                raise ValueError(f"Expected blank missing rate for zero denominator in {rid}/{metric}")
            rate_val: float | None = None
        else:
            if rate_str == "":
                raise ValueError(f"Unexpected blank rate for nonzero denominator in {rid}/{metric}")
            rate_val = float(rate_str)
            if rate_val != num / den:
                raise ValueError(f"Metric rate mismatch for {rid}/{metric}: {rate_val} vs {num / den}")
        verified_summary.setdefault(rid, {})[metric] = {
            "numerator": num,
            "denominator": den,
            "rate": rate_val,
        }

    for rid in recomputed:
        if set(verified_summary.get(rid, {})) != expected_metrics:
            raise ValueError(f"Run {rid} missing metrics in metrics.csv")

    return verified_summary


def verify_saved_evaluation_outputs(
    *,
    step1_dir: Path,
    step2_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
    manifest_output_path: Path | None = None,
    expected_rows_per_run: int = EXPECTED_TEST_COUNT,
) -> dict[str, Any]:
    """Independently verify Step 3 CSV outputs, confirm Step 1/2 stayed read-only, and write manifest."""
    freeze_path = output_dir / "input_freeze_audit.json"
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    for req in (freeze_path, predictions_path, metrics_path):
        if not req.is_file():
            raise FileNotFoundError(f"Required Step 3 artifact missing: {req}")

    # Compare current input contents with the saved freeze record.
    approved_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _, current_step1_hashes, current_raw_hashes = verify_step1_and_raw_provenance(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        config=config,
    )
    current_registry_sha, current_runs, excluded_runs = verify_sweep_registry(
        step2_dir=step2_dir,
        step1_hashes=current_step1_hashes,
        raw_idx_hashes=current_raw_hashes,
        config=config,
    )
    if (
        approved_freeze.get("status") != "PASSED"
        or approved_freeze.get("step1_hashes") != current_step1_hashes
        or approved_freeze.get("raw_idx_hashes") != current_raw_hashes
        or approved_freeze.get("registry_sha256") != current_registry_sha
        or approved_freeze.get("sweep_runs") != json.loads(json.dumps([asdict(r) for r in current_runs]))
    ):
        raise ValueError("Step 1 or Step 2 artifacts differ from frozen input_freeze_audit.json")

    expected_runs = {run.run_id: run for run in current_runs}
    examples = load_heldout_manifest(step1_dir / "manifests/heldout_views.csv", split="test")
    expected_examples = {example.original_id: example for example in examples}
    if len(examples) != expected_rows_per_run or len(expected_examples) != len(examples):
        raise ValueError("Frozen test manifest has missing or duplicate examples")
    recomputed = _recompute_counts_from_predictions(
        predictions_path=predictions_path,
        expected_runs=expected_runs,
        expected_examples=expected_examples,
        expected_rows_per_run=expected_rows_per_run,
    )
    verified_summary = _verify_metrics_csv_matches_predictions(metrics_path, recomputed, expected_runs)

    setup = TrainingSetup(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        device="cpu",
        threads=1,
    )
    environment = configure_runtime(setup)

    project_root = Path(__file__).resolve().parents[2]
    evaluator_sources: dict[str, str] = {}
    for rel in (
        "src/colored_mnist/evaluate.py",
        "src/colored_mnist/models.py",
        "scripts/evaluate_sweep.py",
        "scripts/verify_step3_outputs.py",
        "scripts/audit_step3_freeze_inputs.py",
    ):
        src_file = project_root / rel
        if src_file.is_file():
            evaluator_sources[rel] = file_digest(src_file)

    manifest: dict[str, Any] = {
        "status": "PASSED",
        "evaluator_version": EVALUATOR_VERSION,
        "source_sha256_at_verification": evaluator_sources,
        "verification_runtime": environment,
        "evaluation_protocol": {
            "dtype": "float32",
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "drop_last": False,
        },
        "commands": {
            "freeze": "uv run python scripts/audit_step3_freeze_inputs.py",
            "evaluate": "uv run python scripts/evaluate_sweep.py",
            "verify": "uv run python scripts/verify_step3_outputs.py",
        },
        "input_paths_and_hashes": {
            "input_freeze_audit_json": {
                "path": str(freeze_path),
                "sha256": file_digest(freeze_path),
            },
            "runs_manifest_json": {
                "path": str(step2_dir / "runs_manifest.json"),
                "sha256": current_registry_sha,
            },
            "step1_hashes": current_step1_hashes,
            "raw_idx_hashes": current_raw_hashes,
            "excluded_runs": excluded_runs,
            "sweep_checkpoints": [
                {
                    "run_id": r.run_id,
                    "p": r.p,
                    "dataset_seed": r.dataset_seed,
                    "train_seed": r.train_seed,
                    "completed_epoch": r.completed_epoch,
                    "global_step": r.global_step,
                    "config_sha256": r.config_sha256,
                    "path": str(step2_dir / r.checkpoint_relpath),
                    "sha256": r.checkpoint_sha256,
                }
                for r in current_runs
            ],
        },
        "output_files": {
            "predictions_csv": {
                "path": str(predictions_path),
                "sha256": file_digest(predictions_path),
                "rows": len(expected_runs) * expected_rows_per_run,
            },
            "metrics_csv": {
                "path": str(metrics_path),
                "sha256": file_digest(metrics_path),
                "rows": len(expected_runs) * len(METRIC_DEFINITIONS),
            },
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "verification_results": {
            "num_runs_verified": len(expected_runs),
            "paired_rows_per_run": expected_rows_per_run,
            "total_prediction_rows": len(expected_runs) * expected_rows_per_run,
            "total_metric_rows": len(expected_runs) * len(METRIC_DEFINITIONS),
            "all_views_non_missing_predictions": True,
            "recomputed_metrics_match_csv": True,
            "targeted_never_exceeds_overall_flips": True,
            "input_hashes_match_frozen_record": True,
            "prediction_metadata_matches_frozen_manifest": True,
        },
        "run_metrics_summary": verified_summary,
    }
    target_manifest_path = manifest_output_path or (output_dir / "evaluation_manifest.json")
    write_json(target_manifest_path, manifest)
    return manifest

