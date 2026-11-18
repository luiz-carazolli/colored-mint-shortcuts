"""Tests for Step 3 Substeps 2 & 3: paired metrics, failure cases, val smoke test, and output verification."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.evaluate import (
    FrozenRunRecord,
    build_view_batches,
    evaluate_sweep,
    join_and_compute_metrics,
    load_frozen_model_for_inference,
    predict_view_batches,
    verify_saved_evaluation_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / "artifacts" / "step_1_dataset"
STEP2_DIR = PROJECT_ROOT / "artifacts" / "step_2_training"
STEP3_DIR = PROJECT_ROOT / "artifacts" / "step_3_evaluation"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DUMMY_RUN = FrozenRunRecord(
    run_id="synthetic_run",
    p=0.95,
    dataset_seed=101,
    train_seed=201,
    condition_type="biased",
    completed_epoch=10,
    global_step=2110,
    config_sha256="0" * 64,
    checkpoint_relpath="checkpoints/p_0.10_dseed_101_tseed_201.pt",
    checkpoint_sha256="1" * 64,
)


def _rec(
    oid: str,
    label: int,
    color_id: int,
    pred: int,
    view: str,
    *,
    src_idx: int = 0,
    swapped_target: int = -1,
) -> dict[str, Any]:
    return {
        "original_id": oid,
        "source_split": "test",
        "source_index": src_idx,
        "label": label,
        "color_id": color_id,
        "preferred_color_id": label,
        "swapped_target_class": swapped_target,
        "view": view,
        "pred": pred,
    }


def _metric_lookup(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in metric_rows}


def test_hand_computed_paired_metric_formulas() -> None:
    """Verify exact numerators/denominators across unchanged, wrong-flip, targeted-flip, and already-wrong pairs."""
    # Six hand-crafted items (y, s = (y + 5) % 10):
    # id0: y=0, s=5 | a=0, i=0, w=0 -> unchanged aligned-correct
    # id1: y=1, s=6 | a=1, i=2, w=6 -> aligned-correct, targeted flip (w == s)
    # id2: y=2, s=7 | a=2, i=2, w=3 -> aligned-correct, non-targeted flip (w != y, w != s)
    # id3: y=3, s=8 | a=4, i=3, w=4 -> aligned-wrong, unchanged wrong
    # id4: y=4, s=9 | a=0, i=9, w=9 -> aligned-wrong flips to swapped target (counts overall flip only, NOT targeted)
    # id5: y=5, s=0 | a=1, i=5, w=5 -> aligned-wrong flips back to true label (counts overall flip & swapped_acc, NOT targeted)
    specs = [
        ("id0", 0, 5, 0, 0, 0),
        ("id1", 1, 6, 1, 2, 6),
        ("id2", 2, 7, 2, 2, 3),
        ("id3", 3, 8, 4, 3, 4),
        ("id4", 4, 9, 0, 9, 9),
        ("id5", 5, 0, 1, 5, 5),
    ]
    view_preds: dict[str, dict[str, dict[str, Any]]] = {
        "aligned": {oid: _rec(oid, y, y, a, "aligned", src_idx=idx) for idx, (oid, y, _, a, _, _) in enumerate(specs)},
        "independent": {oid: _rec(oid, y, (y + 1) % 10, i, "independent", src_idx=idx) for idx, (oid, y, _, _, i, _) in enumerate(specs)},
        "swapped": {oid: _rec(oid, y, s, w, "swapped", src_idx=idx, swapped_target=s) for idx, (oid, y, s, _, _, w) in enumerate(specs)},
    }

    pred_rows, metric_rows = join_and_compute_metrics(DUMMY_RUN, view_preds)
    assert len(pred_rows) == 6
    m = _metric_lookup(metric_rows)

    # aligned_correct: id0, id1, id2 -> 3/6
    assert (m["aligned_accuracy"]["numerator"], m["aligned_accuracy"]["denominator"]) == (3, 6)
    assert m["aligned_accuracy"]["rate"] == pytest.approx(3 / 6)
    # independent_correct: id0, id2, id3, id5 -> 4/6
    assert (m["independent_accuracy"]["numerator"], m["independent_accuracy"]["denominator"]) == (4, 6)
    # swapped_correct: id0, id5 -> 2/6
    assert (m["swapped_accuracy"]["numerator"], m["swapped_accuracy"]["denominator"]) == (2, 6)
    # overall_flip: id1 (1->6), id2 (2->3), id4 (0->9), id5 (1->5) -> 4/6
    assert (m["overall_flip_rate"]["numerator"], m["overall_flip_rate"]["denominator"]) == (4, 6)
    # targeted_flip: only id1 (a == y and w == s) out of 3 aligned-correct items -> 1/3
    assert (m["targeted_flip_rate"]["numerator"], m["targeted_flip_rate"]["denominator"]) == (1, 3)
    assert m["targeted_flip_rate"]["rate"] == pytest.approx(1 / 3)
    assert m["targeted_flip_rate"]["numerator"] <= m["overall_flip_rate"]["numerator"]


def test_zero_eligible_targeted_flip_rate_retained_as_missing() -> None:
    """When aligned_correct == 0, targeted numerator/denominator stay 0 and rate is missing ('')."""
    view_preds = {
        "aligned": {"id0": _rec("id0", 0, 0, 1, "aligned"), "id1": _rec("id1", 2, 2, 3, "aligned")},
        "independent": {"id0": _rec("id0", 0, 4, 0, "independent"), "id1": _rec("id1", 2, 5, 2, "independent")},
        "swapped": {"id0": _rec("id0", 0, 5, 5, "swapped", swapped_target=5), "id1": _rec("id1", 2, 7, 7, "swapped", swapped_target=7)},
    }
    _, metric_rows = join_and_compute_metrics(DUMMY_RUN, view_preds)
    m = _metric_lookup(metric_rows)

    assert (m["aligned_accuracy"]["numerator"], m["aligned_accuracy"]["denominator"]) == (0, 2)
    assert (m["overall_flip_rate"]["numerator"], m["overall_flip_rate"]["denominator"]) == (2, 2)
    assert m["targeted_flip_rate"]["numerator"] == 0
    assert m["targeted_flip_rate"]["denominator"] == 0
    assert m["targeted_flip_rate"]["rate"] == ""


def test_join_and_prediction_failure_cases() -> None:
    """Require missing IDs, duplicate IDs, and inconsistent labels to fail."""
    # Missing ID across views
    with pytest.raises(ValueError, match="one-to-one join mismatch"):
        join_and_compute_metrics(
            DUMMY_RUN,
            {
                "aligned": {"id0": _rec("id0", 0, 0, 0, "aligned"), "id1": _rec("id1", 1, 1, 1, "aligned")},
                "independent": {"id0": _rec("id0", 0, 0, 0, "independent")},
                "swapped": {"id0": _rec("id0", 0, 5, 5, "swapped", swapped_target=5), "id1": _rec("id1", 1, 6, 6, "swapped", swapped_target=6)},
            },
        )

    # Inconsistent labels across views for same ID
    with pytest.raises(ValueError, match="inconsistent true labels"):
        join_and_compute_metrics(
            DUMMY_RUN,
            {
                "aligned": {"id0": _rec("id0", 0, 0, 0, "aligned")},
                "independent": {"id0": _rec("id0", 9, 0, 0, "independent")},
                "swapped": {"id0": _rec("id0", 0, 5, 5, "swapped", swapped_target=5)},
            },
        )

    # Duplicate original_id inside prediction batches
    dummy_model = torch.nn.Linear(3 * 28 * 28, 10)
    flat_wrapper = torch.nn.Sequential(torch.nn.Flatten(), dummy_model)
    dup_batches = [(torch.zeros((2, 3, 28, 28), dtype=torch.float32), [
        {"original_id": "dup_0", "source_split": "test", "source_index": 0, "label": 0, "color_id": 0, "preferred_color_id": 0, "swapped_target_class": -1, "view": "aligned"},
        {"original_id": "dup_0", "source_split": "test", "source_index": 0, "label": 0, "color_id": 0, "preferred_color_id": 0, "swapped_target_class": -1, "view": "aligned"},
    ])]
    with pytest.raises(ValueError, match="Duplicate original_id"):
        predict_view_batches(flat_wrapper, dup_batches)


@pytest.mark.usefixtures("trained_sweep", "evaluated_sweep")
def test_stale_freeze_audit_stops_evaluation(tmp_path: Path) -> None:
    fake_out = tmp_path / "step3"
    fake_out.mkdir(parents=True)
    stale = json.loads((STEP3_DIR / "input_freeze_audit.json").read_text(encoding="utf-8"))
    stale["registry_sha256"] = "0" * 64
    (fake_out / "input_freeze_audit.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the saved input freeze audit"):
        evaluate_sweep(
            step1_dir=STEP1_DIR,
            step2_dir=STEP2_DIR,
            raw_dir=RAW_DIR,
            output_dir=fake_out,
            config=DEFAULT_CONFIG,
        )


@pytest.mark.usefixtures("trained_sweep")
def test_checkpoint_smoke_on_validation_data() -> None:
    """Smoke-test weight-only evaluator on official validation split and compare against runs_manifest.json."""
    entries = json.loads((STEP2_DIR / "runs_manifest.json").read_text(encoding="utf-8"))
    target = next(e for e in entries if e["run_id"] == "p_0.10_dseed_101_tseed_201")
    run = FrozenRunRecord(
        run_id=target["run_id"],
        p=float(target["p"]),
        dataset_seed=int(target["dataset_seed"]),
        train_seed=int(target["train_seed"]),
        condition_type=str(target["condition_type"]),
        completed_epoch=int(target["completed_epoch"]),
        global_step=int(target["global_step"]),
        config_sha256=str(target["config_sha256"]),
        checkpoint_relpath=str(target["checkpoint_path"]),
        checkpoint_sha256="verified",
    )

    model = load_frozen_model_for_inference(STEP2_DIR / target["checkpoint_path"], run.train_seed)
    view_preds = {
        view: predict_view_batches(
            model,
            build_view_batches(
                ColoredMNISTDataset.from_artifacts(
                    split="val",
                    view=view,
                    artifacts_dir=STEP1_DIR,
                    raw_data_dir=RAW_DIR,
                    config=DEFAULT_CONFIG,
                    return_metadata=True,
                )
            ),
        )
        for view in ("aligned", "independent", "swapped")
    }
    pred_rows, metric_rows = join_and_compute_metrics(run, view_preds)
    assert len(pred_rows) == 6000
    m = _metric_lookup(metric_rows)
    assert float(m["aligned_accuracy"]["rate"]) == pytest.approx(target["val_aligned_accuracy"], abs=1e-6)
    assert float(m["independent_accuracy"]["rate"]) == pytest.approx(target["val_independent_accuracy"], abs=1e-6)
    assert m["targeted_flip_rate"]["numerator"] <= m["overall_flip_rate"]["numerator"]


@pytest.mark.usefixtures("trained_sweep", "evaluated_sweep")
def test_verify_saved_step3_outputs_and_detect_tampering(tmp_path: Path) -> None:
    """Verify real Step 3 CSV outputs recompute cleanly and catch tampered metric rows."""
    manifest = verify_saved_evaluation_outputs(
        step1_dir=STEP1_DIR,
        step2_dir=STEP2_DIR,
        raw_dir=RAW_DIR,
        output_dir=STEP3_DIR,
        config=DEFAULT_CONFIG,
        manifest_output_path=tmp_path / "evaluation_manifest.json",
    )
    assert manifest["status"] == "PASSED"
    vr = manifest["verification_results"]
    assert vr["num_runs_verified"] == 21
    assert vr["paired_rows_per_run"] == 10_000
    assert vr["total_prediction_rows"] == 210_000
    assert vr["total_metric_rows"] == 105
    assert vr["recomputed_metrics_match_csv"] is True
    assert vr["targeted_never_exceeds_overall_flips"] is True
    assert vr["input_hashes_match_frozen_record"] is True

    # Tamper a single row in copied metrics.csv and ensure verification rejects it
    tampered_dir = tmp_path / "step3_tampered"
    tampered_dir.mkdir(parents=True)
    shutil.copy2(STEP3_DIR / "input_freeze_audit.json", tampered_dir / "input_freeze_audit.json")
    shutil.copy2(STEP3_DIR / "predictions.csv", tampered_dir / "predictions.csv")
    with (STEP3_DIR / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    rows[0]["numerator"] = str(int(rows[0]["numerator"]) + 1)
    with (tampered_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Metric count mismatch"):
        verify_saved_evaluation_outputs(
            step1_dir=STEP1_DIR,
            step2_dir=STEP2_DIR,
            raw_dir=RAW_DIR,
            output_dir=tampered_dir,
            config=DEFAULT_CONFIG,
            manifest_output_path=tmp_path / "tampered_manifest.json",
        )
