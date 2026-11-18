"""Tests for Step 4: Verified sweep summary, error inspection, confounds, and plots."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from colored_mnist.analyze import (
    build_run_level_and_paired_tables,
    build_sweep_metric_summary,
    load_and_verify_step3_inputs,
    run_step4_analysis,
    run_step4_substep1,
    summarize_rates,
)
from colored_mnist.config import DEFAULT_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
STEP3_DIR = PROJECT_ROOT / "artifacts" / "step_3_evaluation"
RAW_DIR = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir


def test_summarize_rates_handles_missing_values() -> None:
    n_def, mean_v, min_v, max_v = summarize_rates([0.20, None, 0.50])
    assert n_def == 2
    assert mean_v == pytest.approx(0.35)
    assert min_v == pytest.approx(0.20)
    assert max_v == pytest.approx(0.50)

    assert summarize_rates([None, None, None]) == (0, None, None, None)


@pytest.mark.usefixtures("evaluated_sweep")
def test_missing_rate_handling_and_paired_deltas() -> None:
    verified = load_and_verify_step3_inputs(STEP3_DIR, DEFAULT_CONFIG)
    # Inject a zero-eligible targeted rate on one run in memory and check summary counts
    rows = json.loads(json.dumps(verified.rows_by_run))
    rows["p_1.00_dseed_102_tseed_202"]["targeted_flip_rate"]["numerator"] = 0
    rows["p_1.00_dseed_102_tseed_202"]["targeted_flip_rate"]["denominator"] = 0
    rows["p_1.00_dseed_102_tseed_202"]["targeted_flip_rate"]["rate"] = None

    summary = build_sweep_metric_summary(rows, DEFAULT_CONFIG)
    p100_tflip = next(r for r in summary if r["p"] == "1.00" and r["metric"] == "targeted_flip_rate")
    assert p100_tflip["num_defined_runs"] == 2
    assert p100_tflip["seed_102_numerator"] == 0
    assert p100_tflip["seed_102_denominator"] == 0
    assert p100_tflip["seed_102_rate"] == ""

    run_rows, paired_summary = build_run_level_and_paired_tables(rows, DEFAULT_CONFIG)
    assert len(run_rows) == 21
    p100_delta_tflip = next(
        r for r in paired_summary if r["p"] == "1.00" and r["paired_metric"] == "delta_targeted_flip_rate_vs_p010_pp"
    )
    assert p100_delta_tflip["num_defined_runs"] == 2
    assert p100_delta_tflip["seed_102_pp"] == ""


@pytest.mark.usefixtures("evaluated_sweep")
def test_step3_manifest_and_hash_enforcement(tmp_path: Path) -> None:
    """Require passing status, matching CSV SHA-256 digests, and full 21-run coverage."""
    copy_s3 = tmp_path / "step3"
    copy_s3.mkdir()
    for name in ("evaluation_manifest.json", "metrics.csv", "predictions.csv"):
        shutil.copy2(STEP3_DIR / name, copy_s3 / name)

    # Non-PASSED manifest must fail
    m_path = copy_s3 / "evaluation_manifest.json"
    bad_manifest = json.loads(m_path.read_text(encoding="utf-8"))
    bad_manifest["status"] = "FAILED"
    m_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must be PASSED"):
        load_and_verify_step3_inputs(copy_s3, DEFAULT_CONFIG)

    # Restore PASSED status, then tamper metrics.csv -> hash check must fail
    shutil.copy2(STEP3_DIR / "evaluation_manifest.json", m_path)
    with (copy_s3 / "metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_and_verify_step3_inputs(copy_s3, DEFAULT_CONFIG)


@pytest.mark.usefixtures("evaluated_sweep")
def test_run_step4_substep1_creates_tables_and_figures(tmp_path: Path) -> None:
    """Execute Substep 1 into temporary directory and verify output schemas and file sizes."""
    out = tmp_path / "step4"
    record = run_step4_substep1(
        step3_dir=STEP3_DIR,
        output_dir=out,
        config=DEFAULT_CONFIG,
    )
    assert record["status"] == "PASSED"
    assert record["num_runs"] == 21

    with (out / "sweep_metric_summary.csv").open(newline="", encoding="utf-8") as f:
        sweep_rows = list(csv.DictReader(f))
    with (out / "run_level_metrics.csv").open(newline="", encoding="utf-8") as f:
        run_rows = list(csv.DictReader(f))
    with (out / "paired_differences_summary.csv").open(newline="", encoding="utf-8") as f:
        paired_rows = list(csv.DictReader(f))

    assert len(sweep_rows) == 35  # 7 p values * 5 metrics
    assert len(run_rows) == 21
    assert len(paired_rows) == 28  # 7 p values * 4 paired differences
    assert all(int(r["num_defined_runs"]) == 3 for r in sweep_rows)
    assert all(int(r["num_defined_runs"]) == 3 for r in paired_rows)

    png = out / "sweep_accuracy_and_flips.png"
    pdf = out / "sweep_accuracy_and_flips.pdf"
    assert png.is_file() and png.stat().st_size > 10_000
    assert pdf.is_file() and pdf.stat().st_size > 10_000


@pytest.mark.usefixtures("evaluated_sweep")
def test_end_to_end_step4_substep2_outputs(tmp_path: Path) -> None:
    """Verify full Step 4 analysis produces valid Substep 2 error breakdowns, confounds, examples, and figures."""
    out = tmp_path / "step_4_analysis_full"
    manifest = run_step4_analysis(
        step1_dir=STEP1_DIR,
        step3_dir=STEP3_DIR,
        raw_dir=RAW_DIR,
        output_dir=out,
        config=DEFAULT_CONFIG,
    )
    assert manifest["status"] == "PASSED"
    assert (out / "step4_analysis_manifest.json").is_file()

    with (out / "by_digit_swapped_metrics.csv").open(newline="", encoding="utf-8") as f:
        digit_rows = list(csv.DictReader(f))
    assert len(digit_rows) == 70  # 7 p values * 10 digits
    p100_rows = [r for r in digit_rows if abs(float(r["p"]) - 1.0) < 1e-9]
    assert len(p100_rows) == 10
    assert all(float(r["swapped_accuracy"]) == pytest.approx(0.0) for r in p100_rows)

    for fname, expect_diag_zero in [
        ("confusion_swapped_p010.csv", False),
        ("confusion_swapped_p100.csv", True),
    ]:
        with (out / fname).open(newline="", encoding="utf-8") as f:
            conf_rows = list(csv.DictReader(f))
        assert len(conf_rows) == 10
        assert sum(int(r["row_total"]) for r in conf_rows) == 30_000
        if expect_diag_zero:
            for i, row in enumerate(conf_rows):
                assert int(row[f"pred_{i}"]) == 0

    with (out / "baseline_independent_by_color.csv").open(newline="", encoding="utf-8") as f:
        color_rows = list(csv.DictReader(f))
    assert len(color_rows) == 10
    assert sum(int(r["n_test_images"]) for r in color_rows) == 10_000

    with (out / "paired_examples.csv").open(newline="", encoding="utf-8") as f:
        ex_rows = list(csv.DictReader(f))
    assert len(ex_rows) == 4
    with (out / "paired_examples.json").open(encoding="utf-8") as f:
        ex_json = json.load(f)
    assert len(ex_json["examples"]) == 4

    for fig_name in [
        "swapped_confusion_matrices.png",
        "swapped_confusion_matrices.pdf",
        "paired_examples_grid.png",
        "paired_examples_grid.pdf",
        "dist_p_vs_acc_aligned.png",
        "dist_p_vs_acc_aligned.pdf",
        "dist_p_vs_acc_independent.png",
        "dist_p_vs_acc_independent.pdf",
        "dist_p_vs_acc_swapped.png",
        "dist_p_vs_acc_swapped.pdf",
        "dist_p_vs_acc_combined.png",
        "dist_p_vs_acc_combined.pdf",
    ]:
        fig_path = out / fig_name
        assert fig_path.is_file() and fig_path.stat().st_size > 10_000
