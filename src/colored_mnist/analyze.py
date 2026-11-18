"""Step 4 Substep 1 analysis: verified sweep summaries, paired comparisons, and plots."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Use non-interactive backend and writable Matplotlib cache before importing pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.matrix import build_run_id, get_paired_train_seed
from colored_mnist.palette import load_palette
from colored_mnist.train import file_digest, write_csv, write_json

EXPECTED_METRICS = (
    "aligned_accuracy",
    "independent_accuracy",
    "swapped_accuracy",
    "overall_flip_rate",
    "targeted_flip_rate",
)
BASELINE_P = 0.10


@dataclass(frozen=True)
class Step3VerifiedInputs:
    """Verified Step 3 evaluation manifest and loaded run-level metric records."""

    manifest: dict[str, Any]
    metrics_sha256: str
    predictions_sha256: str
    expected_run_ids: list[str]
    rows_by_run: dict[str, dict[str, dict[str, Any]]]


def _parse_rate(numerator: int, denominator: int, rate_str: str) -> float | None:
    stripped = rate_str.strip()
    if denominator == 0:
        if stripped != "":
            raise ValueError(f"Expected blank missing rate for zero denominator, got {rate_str!r}")
        return None
    if stripped == "":
        raise ValueError("Unexpected blank rate for non-zero denominator")
    val = float(stripped)
    expected = numerator / denominator
    if abs(val - expected) > 1e-12:
        raise ValueError(f"Rate mismatch in metrics.csv: {val} vs {expected}")
    return val


def load_and_verify_step3_inputs(
    step3_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> Step3VerifiedInputs:
    """Load Step 3 evaluation outputs after verifying manifest status, CSV hashes, and coverage."""
    manifest_path = step3_dir / "evaluation_manifest.json"
    metrics_path = step3_dir / "metrics.csv"
    predictions_path = step3_dir / "predictions.csv"

    for path in (manifest_path, metrics_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Step 3 artifact missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASSED":
        raise ValueError(f"evaluation_manifest.json status must be PASSED, got {manifest.get('status')!r}")

    output_files = manifest.get("output_files", {})
    exp_metrics_sha = output_files.get("metrics_csv", {}).get("sha256")
    exp_preds_sha = output_files.get("predictions_csv", {}).get("sha256")
    actual_metrics_sha = file_digest(metrics_path)
    actual_preds_sha = file_digest(predictions_path)

    if actual_metrics_sha != exp_metrics_sha:
        raise ValueError(
            f"metrics.csv SHA256 mismatch: expected {exp_metrics_sha}, got {actual_metrics_sha}"
        )
    if actual_preds_sha != exp_preds_sha:
        raise ValueError(
            f"predictions.csv SHA256 mismatch: expected {exp_preds_sha}, got {actual_preds_sha}"
        )

    expected_run_ids = [
        build_run_id(float(p), int(d_seed), get_paired_train_seed(int(d_seed)))
        for d_seed in config.dataset_seeds
        for p in config.p_values
    ]
    manifest_checkpoints = manifest.get("input_paths_and_hashes", {}).get("sweep_checkpoints", [])
    manifest_run_ids = [str(c["run_id"]) for c in manifest_checkpoints]
    if set(manifest_run_ids) != set(expected_run_ids) or len(manifest_run_ids) != len(expected_run_ids):
        raise ValueError("Manifest sweep_checkpoints do not match the 21 planned experimental runs")

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    expected_row_count = len(expected_run_ids) * len(EXPECTED_METRICS)
    if len(raw_rows) != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} metric rows, found {len(raw_rows)}")

    rows_by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in raw_rows:
        rid = str(row["run_id"])
        metric = str(row["metric"])
        if rid not in expected_run_ids:
            raise ValueError(f"Unexpected run_id in metrics.csv: {rid!r}")
        if metric not in EXPECTED_METRICS:
            raise ValueError(f"Unexpected metric in metrics.csv: {metric!r}")
        if metric in rows_by_run[rid]:
            raise ValueError(f"Duplicate metric {metric!r} for run {rid!r}")

        num = int(row["numerator"])
        den = int(row["denominator"])
        rate = _parse_rate(num, den, str(row["rate"]))
        rows_by_run[rid][metric] = {
            "run_id": rid,
            "p": float(row["p"]),
            "dataset_seed": int(row["dataset_seed"]),
            "train_seed": int(row["train_seed"]),
            "condition_type": str(row["condition_type"]),
            "numerator": num,
            "denominator": den,
            "rate": rate,
        }

    for rid in expected_run_ids:
        if set(rows_by_run[rid]) != set(EXPECTED_METRICS):
            raise ValueError(f"Missing metrics for run {rid!r}: {set(EXPECTED_METRICS) - set(rows_by_run[rid])}")

    return Step3VerifiedInputs(
        manifest=manifest,
        metrics_sha256=actual_metrics_sha,
        predictions_sha256=actual_preds_sha,
        expected_run_ids=expected_run_ids,
        rows_by_run=dict(rows_by_run),
    )


def summarize_rates(values: list[float | None]) -> tuple[int, float | None, float | None, float | None]:
    """Compute unweighted mean, min, and max over non-missing values and report n_defined."""
    defined = [v for v in values if v is not None]
    if not defined:
        return 0, None, None, None
    return len(defined), sum(defined) / len(defined), min(defined), max(defined)


def build_sweep_metric_summary(
    rows_by_run: dict[str, dict[str, dict[str, Any]]],
    config: DatasetConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    """Tabulate per-run values, unweighted mean, min, max, and targeted counts across p."""
    summary_rows: list[dict[str, Any]] = []
    for p in config.p_values:
        for metric in EXPECTED_METRICS:
            run_records: dict[int, dict[str, Any]] = {}
            rates: list[float | None] = []
            for d_seed in config.dataset_seeds:
                rid = build_run_id(float(p), int(d_seed), get_paired_train_seed(int(d_seed)))
                rec = rows_by_run[rid][metric]
                run_records[int(d_seed)] = rec
                rates.append(rec["rate"])

            n_defined, mean_val, min_val, max_val = summarize_rates(rates)
            row: dict[str, Any] = {
                "p": f"{p:.2f}",
                "metric": metric,
                "num_defined_runs": n_defined,
                "mean_rate": "" if mean_val is None else mean_val,
                "min_rate": "" if min_val is None else min_val,
                "max_rate": "" if max_val is None else max_val,
            }
            for d_seed in config.dataset_seeds:
                rec = run_records[int(d_seed)]
                row[f"seed_{d_seed}_numerator"] = rec["numerator"]
                row[f"seed_{d_seed}_denominator"] = rec["denominator"]
                row[f"seed_{d_seed}_rate"] = "" if rec["rate"] is None else rec["rate"]
            summary_rows.append(row)
    return summary_rows


def _subtract_defined(val: float | None, base: float | None) -> float | None:
    if val is None or base is None:
        return None
    return (val - base) * 100.0


def build_run_level_and_paired_tables(
    rows_by_run: dict[str, dict[str, dict[str, Any]]],
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute run-level metrics, aligned-swapped gap, and within-seed deltas vs p=0.10 (pp)."""
    baselines: dict[int, dict[str, float | None]] = {}
    for d_seed in config.dataset_seeds:
        base_rid = build_run_id(BASELINE_P, int(d_seed), get_paired_train_seed(int(d_seed)))
        base_rec = rows_by_run[base_rid]
        baselines[int(d_seed)] = {
            "swapped_accuracy": base_rec["swapped_accuracy"]["rate"],
            "overall_flip_rate": base_rec["overall_flip_rate"]["rate"],
            "targeted_flip_rate": base_rec["targeted_flip_rate"]["rate"],
        }

    run_level_rows: list[dict[str, Any]] = []
    paired_by_p_metric: dict[tuple[float, str], dict[int, float | None]] = defaultdict(dict)

    for d_seed in config.dataset_seeds:
        t_seed = get_paired_train_seed(int(d_seed))
        base = baselines[int(d_seed)]
        for p in config.p_values:
            rid = build_run_id(float(p), int(d_seed), t_seed)
            m = rows_by_run[rid]
            aligned = m["aligned_accuracy"]["rate"]
            indep = m["independent_accuracy"]["rate"]
            swapped = m["swapped_accuracy"]["rate"]
            oflip = m["overall_flip_rate"]["rate"]
            tflip_rec = m["targeted_flip_rate"]
            tflip = tflip_rec["rate"]

            gap_pp = _subtract_defined(aligned, swapped)
            d_swapped_pp = _subtract_defined(swapped, base["swapped_accuracy"])
            d_oflip_pp = _subtract_defined(oflip, base["overall_flip_rate"])
            d_tflip_pp = _subtract_defined(tflip, base["targeted_flip_rate"])

            paired_values = {
                "aligned_minus_swapped_gap_pp": gap_pp,
                "delta_swapped_accuracy_vs_p010_pp": d_swapped_pp,
                "delta_overall_flip_rate_vs_p010_pp": d_oflip_pp,
                "delta_targeted_flip_rate_vs_p010_pp": d_tflip_pp,
            }
            for pmetric, val in paired_values.items():
                paired_by_p_metric[(float(p), pmetric)][int(d_seed)] = val

            run_level_rows.append(
                {
                    "run_id": rid,
                    "p": f"{p:.2f}",
                    "dataset_seed": int(d_seed),
                    "train_seed": t_seed,
                    "condition_type": m["aligned_accuracy"]["condition_type"],
                    "aligned_accuracy": "" if aligned is None else aligned,
                    "independent_accuracy": "" if indep is None else indep,
                    "swapped_accuracy": "" if swapped is None else swapped,
                    "overall_flip_rate": "" if oflip is None else oflip,
                    "targeted_flip_rate": "" if tflip is None else tflip,
                    "targeted_flip_numerator": tflip_rec["numerator"],
                    "aligned_correct_denominator": tflip_rec["denominator"],
                    "aligned_minus_swapped_gap_pp": "" if gap_pp is None else gap_pp,
                    "delta_swapped_accuracy_vs_p010_pp": "" if d_swapped_pp is None else d_swapped_pp,
                    "delta_overall_flip_rate_vs_p010_pp": "" if d_oflip_pp is None else d_oflip_pp,
                    "delta_targeted_flip_rate_vs_p010_pp": "" if d_tflip_pp is None else d_tflip_pp,
                }
            )

    paired_metrics_order = (
        "aligned_minus_swapped_gap_pp",
        "delta_swapped_accuracy_vs_p010_pp",
        "delta_overall_flip_rate_vs_p010_pp",
        "delta_targeted_flip_rate_vs_p010_pp",
    )
    paired_summary_rows: list[dict[str, Any]] = []
    for p in config.p_values:
        for pmetric in paired_metrics_order:
            per_seed = paired_by_p_metric[(float(p), pmetric)]
            n_def, mean_pp, min_pp, max_pp = summarize_rates([per_seed[int(s)] for s in config.dataset_seeds])
            row: dict[str, Any] = {
                "p": f"{p:.2f}",
                "paired_metric": pmetric,
                "num_defined_runs": n_def,
                "mean_pp": "" if mean_pp is None else mean_pp,
                "min_pp": "" if min_pp is None else min_pp,
                "max_pp": "" if max_pp is None else max_pp,
            }
            for s in config.dataset_seeds:
                v = per_seed[int(s)]
                row[f"seed_{s}_pp"] = "" if v is None else v
            paired_summary_rows.append(row)

    return run_level_rows, paired_summary_rows


def _plot_series(
    ax: Any,
    p_vals: list[float],
    series_data: list[tuple[str, str, str, list[list[float]], list[float]]],
    *,
    show_seed_markers: bool = True,
) -> None:
    """Plot seed points (percentage) and unweighted mean curve on linear numeric p."""
    seed_offsets = (-0.004, 0.0, 0.004)
    seed_shapes = ("^", "s", "D")
    for label, color, linestyle, per_seed_lists, mean_list in series_data:
        ax.plot(
            p_vals,
            mean_list,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            label=label,
            zorder=4,
        )
        if show_seed_markers:
            for s_idx, seed_vals in enumerate(per_seed_lists):
                xs = [p + seed_offsets[s_idx] for p in p_vals]
                ax.scatter(
                    xs,
                    seed_vals,
                    color=color,
                    marker=seed_shapes[s_idx],
                    s=18,
                    alpha=0.55,
                    edgecolors="none",
                    zorder=3,
                )


def plot_sweep_curves(
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[Path, Path]:
    """Generate two-panel accuracy & flip-rate figure vs numeric p with high-p zoom insets."""
    lookup = {(row["p"], row["metric"]): row for row in summary_rows}
    p_vals = [float(p) for p in config.p_values]
    p_strs = [f"{p:.2f}" for p in p_vals]

    def extract(metric: str) -> tuple[list[list[float]], list[float]]:
        per_seed = [
            [float(lookup[(ps, metric)][f"seed_{s}_rate"]) * 100.0 for ps in p_strs]
            for s in config.dataset_seeds
        ]
        means = [float(lookup[(ps, metric)]["mean_rate"]) * 100.0 for ps in p_strs]
        return per_seed, means

    acc_series = [
        ("Aligned test", "#1b6ca8", "-", *extract("aligned_accuracy")),
        ("Independent test", "#d97706", "--", *extract("independent_accuracy")),
        ("Swapped test", "#b91c1c", "-", *extract("swapped_accuracy")),
    ]
    flip_series = [
        ("Overall flip (aligned $\\neq$ swapped)", "#4b5563", "--", *extract("overall_flip_rate")),
        ("Targeted flip (to swapped target)", "#7c2d12", "-", *extract("targeted_flip_rate")),
    ]

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax_acc, ax_flip) = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=300, constrained_layout=True)

    # Panel A: Accuracy vs numeric p
    _plot_series(ax_acc, p_vals, acc_series)
    ax_acc.set_title("(a) Test Accuracy by Background View", fontweight="bold", pad=8)
    ax_acc.set_xlabel("Training Background–Class Correlation $p$")
    ax_acc.set_ylabel("Test Accuracy (%)")
    ax_acc.set_xlim(0.06, 1.03)
    ax_acc.set_ylim(-3, 105)
    ax_acc.set_xticks([0.10, 0.30, 0.50, 0.70, 0.85, 1.00])
    ax_acc.grid(True, linestyle=":", alpha=0.45)
    ax_acc.legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.87),
        frameon=True,
        framealpha=0.92,
        fontsize=8.5,
    )

    # Zoom inset for the steep high-p regime in bottom-left open space
    inset_acc = inset_axes(ax_acc, width="40%", height="40%", loc="lower left", borderpad=2.1)
    _plot_series(inset_acc, p_vals, acc_series, show_seed_markers=True)
    inset_acc.set_xlim(0.945, 1.007)
    inset_acc.set_ylim(-4, 105)
    inset_acc.set_xticks([0.95, 0.99, 1.00])
    inset_acc.tick_params(labelsize=7.5)
    inset_acc.set_title("High-$p$ regime ($p \\geq 0.95$)", fontsize=8)
    inset_acc.grid(True, linestyle=":", alpha=0.4)

    # Panel B: Flip rates vs numeric p
    _plot_series(ax_flip, p_vals, flip_series)
    ax_flip.set_title("(b) Paired Aligned-to-Swapped Flip Rates", fontweight="bold", pad=8)
    ax_flip.set_xlabel("Training Background–Class Correlation $p$")
    ax_flip.set_ylabel("Flip Rate (%)")
    ax_flip.set_xlim(0.06, 1.03)
    ax_flip.set_ylim(-3, 105)
    ax_flip.set_xticks([0.10, 0.30, 0.50, 0.70, 0.85, 1.00])
    ax_flip.grid(True, linestyle=":", alpha=0.45)
    # Seed marker legend note included cleanly
    ax_flip.scatter([], [], color="#6b7280", marker="^", s=22, label="Seed 101/201")
    ax_flip.scatter([], [], color="#6b7280", marker="s", s=22, label="Seed 102/202")
    ax_flip.scatter([], [], color="#6b7280", marker="D", s=22, label="Seed 103/203")
    ax_flip.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.0, ncol=1)

    inset_flip = inset_axes(ax_flip, width="40%", height="42%", loc="center left", borderpad=2.1)
    _plot_series(inset_flip, p_vals, flip_series, show_seed_markers=True)
    inset_flip.set_xlim(0.945, 1.007)
    inset_flip.set_ylim(-4, 105)
    inset_flip.set_xticks([0.95, 0.99, 1.00])
    inset_flip.tick_params(labelsize=7.5)
    inset_flip.set_title("High-$p$ regime ($p \\geq 0.95$)", fontsize=8)
    inset_flip.grid(True, linestyle=":", alpha=0.4)

    png_path = output_dir / "sweep_accuracy_and_flips.png"
    pdf_path = output_dir / "sweep_accuracy_and_flips.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def run_step4_substep1(
    *,
    step3_dir: Path,
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Verify Step 3 inputs, generate sweep & paired difference tables and plots, and save outputs."""
    verified = load_and_verify_step3_inputs(step3_dir=step3_dir, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = build_sweep_metric_summary(verified.rows_by_run, config=config)
    run_level_rows, paired_summary_rows = build_run_level_and_paired_tables(
        verified.rows_by_run, config=config
    )

    sweep_summary_path = output_dir / "sweep_metric_summary.csv"
    run_level_path = output_dir / "run_level_metrics.csv"
    paired_summary_path = output_dir / "paired_differences_summary.csv"

    write_csv(sweep_summary_path, summary_rows)
    write_csv(run_level_path, run_level_rows)
    write_csv(paired_summary_path, paired_summary_rows)

    png_path, pdf_path = plot_sweep_curves(summary_rows, output_dir, config=config)

    record = {
        "status": "PASSED",
        "command": "uv run python scripts/analyze_step4_sweep.py",
        "input_hashes": {
            "evaluation_manifest_json": file_digest(step3_dir / "evaluation_manifest.json"),
            "metrics_csv": verified.metrics_sha256,
            "predictions_csv": verified.predictions_sha256,
        },
        "num_runs": len(verified.expected_run_ids),
        "p_values": list(config.p_values),
        "dataset_seeds": list(config.dataset_seeds),
        "outputs": {
            "sweep_metric_summary_csv": str(sweep_summary_path),
            "run_level_metrics_csv": str(run_level_path),
            "paired_differences_summary_csv": str(paired_summary_path),
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
        },
        "sweep_metric_summary": summary_rows,
        "paired_differences_summary": paired_summary_rows,
    }
    write_json(output_dir / "step4_substep1_summary.json", record)
    return record


def _collect_prediction_breakdowns(
    predictions_path: Path,
    palette_preferred_by_digit: tuple[int, ...],
    config: DatasetConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Stream predictions.csv once to accumulate by-digit, confusion, color, and pair stats."""
    # Per (p_str, label, d_seed) counts for exact pooled + per-seed by-digit tables and p x acc distributions
    digit_seed_counts: dict[tuple[str, int, int], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "a_ok": 0, "i_ok": 0, "w_ok": 0, "flip": 0, "tflip": 0}
    )
    conf_p010 = np.zeros((10, 10), dtype=np.int64)
    conf_p100 = np.zeros((10, 10), dtype=np.int64)

    # Independent view at p=0.10: per-color and global per-digit baseline accuracy
    color_counts: dict[int, dict[str, Any]] = {
        c: {"n_eval": 0, "correct": 0, "digit_eval_counts": [0] * 10}
        for c in range(10)
    }
    indep_digit_totals = [[0, 0] for _ in range(10)]  # [correct, total] at p=0.10
    unique_test_ids_by_color: dict[int, set[str]] = {c: set() for c in range(10)}

    # Keep seed 101 predictions at p in (0.10, 0.99, 1.00) for documented example selection
    focus_ps = {"0.10", "0.99", "1.00"}
    seed101_focus: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            p_str = str(row["p"])
            d_seed = int(row["dataset_seed"])
            oid = str(row["original_id"])
            y = int(row["label"])
            s = int(row["swapped_target_class"])
            a_col = int(row["aligned_color_id"])
            i_col = int(row["independent_color_id"])
            w_col = int(row["swapped_color_id"])
            a = int(row["aligned_pred"])
            i = int(row["independent_pred"])
            w = int(row["swapped_pred"])

            dc = digit_seed_counts[(p_str, y, d_seed)]
            dc["n"] += 1
            a_ok = a == y
            dc["a_ok"] += int(a_ok)
            dc["i_ok"] += int(i == y)
            dc["w_ok"] += int(w == y)
            dc["flip"] += int(a != w)
            dc["tflip"] += int(a_ok and w == s)

            if p_str == "0.10":
                conf_p010[y, w] += 1
                cc = color_counts[i_col]
                cc["n_eval"] += 1
                cc["correct"] += int(i == y)
                cc["digit_eval_counts"][y] += 1
                unique_test_ids_by_color[i_col].add(oid)
                indep_digit_totals[y][0] += int(i == y)
                indep_digit_totals[y][1] += 1
            elif p_str == "1.00":
                conf_p100[y, w] += 1

            if d_seed == config.dataset_seeds[0] and p_str in focus_ps:
                seed101_focus[oid][p_str] = {
                    "label": y,
                    "swapped_target_class": s,
                    "aligned_color_id": a_col,
                    "independent_color_id": i_col,
                    "swapped_color_id": w_col,
                    "aligned_pred": a,
                    "independent_pred": i,
                    "swapped_pred": w,
                }

    by_digit_rows: list[dict[str, Any]] = []
    for p in config.p_values:
        p_str = f"{p:.2f}"
        for y in range(10):
            s = int(config.swapped_target_by_digit[y])
            s_col = int(palette_preferred_by_digit[s])
            seed_recs = [digit_seed_counts[(p_str, y, int(ds))] for ds in config.dataset_seeds]
            n_pool = sum(r["n"] for r in seed_recs)
            w_ok_pool = sum(r["w_ok"] for r in seed_recs)
            flip_pool = sum(r["flip"] for r in seed_recs)
            a_ok_pool = sum(r["a_ok"] for r in seed_recs)
            tflip_pool = sum(r["tflip"] for r in seed_recs)
            by_digit_rows.append(
                {
                    "p": p_str,
                    "label": y,
                    "swapped_target_class": s,
                    "swapped_color_id": s_col,
                    "n_pairs_pooled": n_pool,
                    "n_pairs_per_seed": seed_recs[0]["n"],
                    "swapped_correct_num": w_ok_pool,
                    "swapped_accuracy": w_ok_pool / n_pool,
                    "overall_flip_num": flip_pool,
                    "overall_flip_rate": flip_pool / n_pool,
                    "targeted_flip_num": tflip_pool,
                    "aligned_correct_den": a_ok_pool,
                    "targeted_flip_rate": "" if a_ok_pool == 0 else tflip_pool / a_ok_pool,
                    "seed_101_swapped_acc": seed_recs[0]["w_ok"] / seed_recs[0]["n"],
                    "seed_102_swapped_acc": seed_recs[1]["w_ok"] / seed_recs[1]["n"],
                    "seed_103_swapped_acc": seed_recs[2]["w_ok"] / seed_recs[2]["n"],
                }
            )

    view_ok_keys = {
        "aligned": "a_ok",
        "independent": "i_ok",
        "swapped": "w_ok",
    }
    view_digit_seed_acc: dict[str, dict[str, list[float]]] = {v: {} for v in view_ok_keys}
    view_run_seed_acc: dict[str, dict[str, list[float]]] = {v: {} for v in view_ok_keys}
    for p in config.p_values:
        p_str = f"{p:.2f}"
        for view, ok_key in view_ok_keys.items():
            digit_rates: list[float] = []
            run_rates: list[float] = []
            for ds in config.dataset_seeds:
                seed_ok = 0
                seed_n = 0
                for y in range(10):
                    rec = digit_seed_counts[(p_str, y, int(ds))]
                    digit_rates.append(100.0 * rec[ok_key] / rec["n"])
                    seed_ok += rec[ok_key]
                    seed_n += rec["n"]
                run_rates.append(100.0 * seed_ok / seed_n)
            view_digit_seed_acc[view][p_str] = digit_rates
            view_run_seed_acc[view][p_str] = run_rates

    return {
        "by_digit_rows": by_digit_rows,
        "conf_p010": conf_p010,
        "conf_p100": conf_p100,
        "color_counts": color_counts,
        "unique_test_ids_by_color": unique_test_ids_by_color,
        "indep_digit_totals": indep_digit_totals,
        "seed101_focus": dict(seed101_focus),
        "view_digit_seed_acc": view_digit_seed_acc,
        "view_run_seed_acc": view_run_seed_acc,
    }


def _export_confusion_csv(matrix: np.ndarray, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for y in range(10):
        row: dict[str, Any] = {
            "true_digit": y,
            "swapped_target_class": (y + 5) % 10,
            "row_total": int(matrix[y].sum()),
        }
        for pred in range(10):
            row[f"pred_{pred}"] = int(matrix[y, pred])
        rows.append(row)
    write_csv(path, rows)


def plot_swapped_confusion_matrices(
    conf_p010: np.ndarray,
    conf_p100: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot baseline (p=0.10) vs confounded (p=1.00) swapped confusion heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=300, constrained_layout=True)
    panels = [
        (axes[0], conf_p010, "(a) Baseline $p = 0.10$ on Swapped Test"),
        (axes[1], conf_p100, "(b) Confounded $p = 1.00$ on Swapped Test"),
    ]

    for ax, mat, title in panels:
        row_sums = mat.sum(axis=1, keepdims=True)
        pct = (mat / np.maximum(row_sums, 1)) * 100.0
        im = ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=100.0, aspect="equal")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Class on Swapped Background")
        ax.set_ylabel("True Digit Class $y$")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{y} (swap->{(y + 5) % 10})" for y in range(10)], fontsize=8)

        for y in range(10):
            # Highlight true diagonal (dashed navy) and swapped target cell (solid crimson)
            s = (y + 5) % 10
            ax.add_patch(
                patches.Rectangle((y - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="#1d4ed8", linewidth=1.1, linestyle="--")
            )
            ax.add_patch(
                patches.Rectangle((s - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="#b91c1c", linewidth=1.6)
            )
            for pred in range(10):
                val = pct[y, pred]
                if val >= 1.0 or pred == y or pred == s:
                    txt_color = "white" if val >= 52.0 else "#111827"
                    ax.text(
                        pred,
                        y,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color=txt_color,
                        fontweight="bold" if pred in (y, s) else "normal",
                    )

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Row Percentage (%)")

    png_path = output_dir / "swapped_confusion_matrices.png"
    pdf_path = output_dir / "swapped_confusion_matrices.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def build_baseline_color_table(
    step1_dir: Path,
    color_counts: dict[int, dict[str, Any]],
    unique_ids_by_color: dict[int, set[str]],
    indep_digit_totals: list[list[int]],
) -> list[dict[str, Any]]:
    """Build p=0.10 independent-view accuracy by background color with digit-mix controls."""
    palette = load_palette(step1_dir / "palette.json")
    global_digit_acc = [c / t for c, t in indep_digit_totals]

    rows: list[dict[str, Any]] = []
    for color_rec in palette.colors:
        cid = int(color_rec.color_id)
        cc = color_counts[cid]
        n_eval = int(cc["n_eval"])
        correct = int(cc["correct"])
        digit_eval_counts = cc["digit_eval_counts"]
        expected_correct_from_mix = sum(
            digit_eval_counts[d] * global_digit_acc[d] for d in range(10)
        )
        expected_acc = expected_correct_from_mix / n_eval
        actual_acc = correct / n_eval
        row: dict[str, Any] = {
            "color_id": cid,
            "hue_deg": color_rec.hue_deg,
            "hex_code": color_rec.hex_code,
            "contrast_vs_white": color_rec.contrast_vs_white,
            "n_test_images": len(unique_ids_by_color[cid]),
            "n_eval_3seeds": n_eval,
            "correct_count": correct,
            "error_count": n_eval - correct,
            "accuracy": actual_acc,
            "digit_mix_expected_accuracy": expected_acc,
            "accuracy_minus_mix_expected_pp": (actual_acc - expected_acc) * 100.0,
        }
        for d in range(10):
            # Divide by 3 to show unique image count per digit within this color
            row[f"digit_{d}_images"] = digit_eval_counts[d] // 3
        rows.append(row)
    return rows


def select_and_render_paired_examples(
    *,
    step1_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    seed101_focus: dict[str, dict[str, dict[str, int]]],
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Select documented paired test examples (targeted flips & stable at p=0.99) and render grid."""
    ordered_oids = sorted(seed101_focus)
    targeted_examples: list[dict[str, Any]] = []
    stable_examples: list[dict[str, Any]] = []
    used_digits_targeted: set[int] = set()
    used_digits_stable: set[int] = set()

    # Deterministic rule on seed 101:
    # 1) Targeted flip at p=0.99: p=0.10 gets all 3 views right; p=0.99 & p=1.00 get aligned right & flip to swapped target.
    # 2) Stable prediction at p=0.99: p=0.10 & p=0.99 get all 3 views right; p=1.00 gets aligned right & flips to swapped target.
    for oid in ordered_oids:
        r010 = seed101_focus[oid]["0.10"]
        r099 = seed101_focus[oid]["0.99"]
        r100 = seed101_focus[oid]["1.00"]
        y = r010["label"]
        s = r010["swapped_target_class"]

        all_ok_010 = (r010["aligned_pred"], r010["independent_pred"], r010["swapped_pred"]) == (y, y, y)
        tflip_100 = r100["aligned_pred"] == y and r100["swapped_pred"] == s
        if not (all_ok_010 and tflip_100):
            continue

        tflip_099 = r099["aligned_pred"] == y and r099["swapped_pred"] == s and r099["independent_pred"] != y
        all_ok_099 = (r099["aligned_pred"], r099["independent_pred"], r099["swapped_pred"]) == (y, y, y)

        if tflip_099 and len(targeted_examples) < 2 and y not in used_digits_targeted:
            used_digits_targeted.add(y)
            targeted_examples.append({
                "category": "Targeted flip by p=0.99",
                "original_id": oid,
                "label": y,
                "swapped_target_class": s,
                "p_0.10": r010,
                "p_0.99": r099,
                "p_1.00": r100,
            })
        elif all_ok_099 and len(stable_examples) < 2 and y not in used_digits_stable:
            used_digits_stable.add(y)
            stable_examples.append({
                "category": "Stable at p=0.99; flip at p=1.00",
                "original_id": oid,
                "label": y,
                "swapped_target_class": s,
                "p_0.10": r010,
                "p_0.99": r099,
                "p_1.00": r100,
            })
        if len(targeted_examples) == 2 and len(stable_examples) == 2:
            break

    selected = targeted_examples + stable_examples
    if len(selected) != 4:
        raise RuntimeError(f"Expected 4 paired examples, selected {len(selected)}")

    views = ("aligned", "independent", "swapped")
    datasets = {
        v: ColoredMNISTDataset.from_artifacts(
            split="test",
            view=v,
            artifacts_dir=step1_dir,
            raw_data_dir=raw_dir,
            config=config,
            return_metadata=True,
        )
        for v in views
    }
    index_by_id = {entry.original_id: idx for idx, entry in enumerate(datasets["aligned"].entries)}

    fig, axes = plt.subplots(4, 3, figsize=(10.5, 9.2), dpi=300, constrained_layout=True)
    col_titles = ("Aligned Background", "Independent Background", "Swapped Background")
    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(col_title, fontweight="bold", pad=10, fontsize=10)

    flat_rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(selected):
        oid = item["original_id"]
        y = item["label"]
        s = item["swapped_target_class"]
        ds_idx = index_by_id[oid]
        r010, r099, r100 = item["p_0.10"], item["p_0.99"], item["p_1.00"]

        flat_rows.append({
            "category": item["category"],
            "original_id": oid,
            "label": y,
            "swapped_target_class": s,
            "aligned_color_id": r010["aligned_color_id"],
            "independent_color_id": r010["independent_color_id"],
            "swapped_color_id": r010["swapped_color_id"],
            "p010_aligned_pred": r010["aligned_pred"],
            "p010_indep_pred": r010["independent_pred"],
            "p010_swapped_pred": r010["swapped_pred"],
            "p099_aligned_pred": r099["aligned_pred"],
            "p099_indep_pred": r099["independent_pred"],
            "p099_swapped_pred": r099["swapped_pred"],
            "p100_aligned_pred": r100["aligned_pred"],
            "p100_indep_pred": r100["independent_pred"],
            "p100_swapped_pred": r100["swapped_pred"],
        })

        for col_idx, view in enumerate(views):
            ax = axes[row_idx, col_idx]
            img_u8 = datasets[view].render_uint8(ds_idx)
            ax.imshow(img_u8, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            pred_key = f"{view}_pred"
            cid = r010[f"{view}_color_id"]
            lbl = (
                f"color={cid} | preds (0.10 / 0.99 / 1.00): "
                f"{r010[pred_key]} / {r099[pred_key]} / {r100[pred_key]}"
            )
            ax.set_xlabel(lbl, fontsize=7.6, labelpad=4)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{oid}\ny={y}, swap={s}\n({item['category']})",
                    fontsize=7.8,
                    rotation=0,
                    labelpad=65,
                    va="center",
                    ha="right",
                )

    png_path = output_dir / "paired_examples_grid.png"
    pdf_path = output_dir / "paired_examples_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    write_csv(output_dir / "paired_examples.csv", flat_rows)
    write_json(
        output_dir / "paired_examples.json",
        {
            "selection_rule": (
                "First two distinct digits in sorted official test original_id order on seed pair 101/201 "
                "for each regime: (A) Targeted flip by p=0.99 (p=0.10 gets all 3 views right; p=0.99 and p=1.00 "
                "predict true label on aligned and swap target on swapped), and (B) Stable through p=0.99 "
                "(both p=0.10 and p=0.99 get all 3 views right while p=1.00 flips aligned-correct to swap target)."
            ),
            "examples": flat_rows,
        },
    )
    return flat_rows, png_path, pdf_path


def plot_view_accuracy_distributions(
    view_digit_seed_acc: dict[str, dict[str, list[float]]],
    view_run_seed_acc: dict[str, dict[str, list[float]]],
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> dict[str, tuple[Path, Path]]:
    """Plot three p x Accuracy distribution figures (aligned, independent, swapped)."""
    p_vals = [float(p) for p in config.p_values]
    p_strs = [f"{p:.2f}" for p in p_vals]
    xs = np.arange(len(p_vals), dtype=float)
    rng = np.random.default_rng(7)
    jitter_offsets = rng.uniform(-0.14, 0.14, size=30)

    view_specs = [
        (
            "aligned",
            r"Aligned Test Variation: $p \times \mathrm{Accuracy}$ Distribution",
            "#1f77b4",
            "#cfe2f3",
            True,
        ),
        (
            "independent",
            r"Independent Test Variation: $p \times \mathrm{Accuracy}$ Distribution",
            "#ff7f0e",
            "#ffe0b2",
            False,
        ),
        (
            "swapped",
            r"Swapped Test Variation: $p \times \mathrm{Accuracy}$ Distribution",
            "#d62728",
            "#f8c8c8",
            False,
        ),
    ]

    seed_shapes = ("^", "s", "D")
    seed_offsets = (-0.08, 0.0, 0.08)
    outputs: dict[str, tuple[Path, Path]] = {}

    for view_key, title, color, fill_color, is_aligned in view_specs:
        fig, ax = plt.subplots(figsize=(8.8, 3.6), dpi=300, constrained_layout=True)

        digit_groups = [view_digit_seed_acc[view_key][ps] for ps in p_strs]
        run_lists = [view_run_seed_acc[view_key][ps] for ps in p_strs]
        run_means = [float(np.mean(r)) for r in run_lists]
        run_mins = [float(np.min(r)) for r in run_lists]
        run_maxs = [float(np.max(r)) for r in run_lists]

        bp = ax.boxplot(
            digit_groups,
            positions=xs,
            widths=0.44,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for box in bp["boxes"]:
            box.set(facecolor=fill_color, edgecolor=color, linewidth=1.2, alpha=0.65)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.1, linestyle="-", alpha=0.75)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.2, alpha=0.85)
        for median in bp["medians"]:
            median.set(color="#222222", linewidth=1.4, linestyle=":")

        for idx, vals in enumerate(digit_groups):
            n_pts = len(vals)
            ax.scatter(
                xs[idx] + jitter_offsets[:n_pts],
                vals,
                s=14,
                color=color,
                alpha=0.35,
                edgecolors="none",
                zorder=3,
                label="Per-(digit, seed) accuracy ($n=30$)" if idx == 0 else None,
            )

        ax.fill_between(
            xs,
            run_mins,
            run_maxs,
            color=color,
            alpha=0.14,
            zorder=1,
            label="Run-level 3-seed [min, max] band",
        )
        ax.plot(
            xs,
            run_means,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5.5,
            zorder=5,
            label="Unweighted 3-seed mean ($N=10{,}000$/seed)",
        )
        for s_idx, d_seed in enumerate(config.dataset_seeds):
            t_seed = get_paired_train_seed(int(d_seed))
            ax.scatter(
                xs + seed_offsets[s_idx],
                [r[s_idx] for r in run_lists],
                marker=seed_shapes[s_idx],
                s=28,
                color="#111111",
                alpha=0.75,
                zorder=6,
                label=f"Seed {d_seed}/{t_seed}",
            )

        baseline_mean = run_means[0]
        ax.axhline(
            baseline_mean,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            zorder=1,
        )

        if is_aligned:
            min_pt = min(min(g) for g in digit_groups)
            y_low = np.floor((min_pt - 0.6) * 2.0) / 2.0
            ax.set_ylim(y_low, 100.85)
            for idx, m_val in enumerate(run_means):
                ax.text(
                    xs[idx],
                    100.28,
                    f"{m_val:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.9, "lw": 0.6},
                )
            ax.set_title(f"{title} (zoomed y-axis)", fontsize=10.5, fontweight="bold")
        else:
            ax.axhline(
                10.0,
                color="#666666",
                linestyle=":",
                linewidth=1.1,
                alpha=0.85,
                zorder=1,
                label="Uniform chance (10.0%)",
            )
            ax.set_ylim(-12.5, 112.0)
            for idx, (m_val, g_vals) in enumerate(zip(run_means, digit_groups)):
                top_val = max(max(g_vals), run_maxs[idx])
                if top_val < 12.0:
                    y_text = min(min(g_vals), run_mins[idx]) - 3.0
                    va_align = "top"
                else:
                    y_text = min(103.5, top_val + 4.0) if top_val < 90.0 else 103.2
                    va_align = "bottom"
                ax.text(
                    xs[idx],
                    y_text,
                    f"{m_val:.2f}%",
                    ha="center",
                    va=va_align,
                    fontsize=8.2,
                    fontweight="bold",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.9, "lw": 0.6},
                )
            ax.set_title(title, fontsize=10.5, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(p_strs, fontsize=9.2)
        ax.set_xlabel("Training color-digit correlation strength $p$ (counterexample rate $1 - p$)", fontsize=9.2)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9.2)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")

        handles, labels = ax.get_legend_handles_labels()
        # Keep legend compact with 2 columns in an unobtrusive spot
        leg_loc = "lower right" if is_aligned else "lower left"
        ax.legend(
            handles,
            labels,
            loc=leg_loc,
            ncol=2,
            fontsize=7.3,
            framealpha=0.92,
            handletextpad=0.5,
            columnspacing=0.9,
        )

        png_path = output_dir / f"dist_p_vs_acc_{view_key}.png"
        pdf_path = output_dir / f"dist_p_vs_acc_{view_key}.pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs[view_key] = (png_path, pdf_path)

    return outputs


def plot_combined_view_accuracy_distributions(
    view_digit_seed_acc: dict[str, dict[str, list[float]]],
    view_run_seed_acc: dict[str, dict[str, list[float]]],
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> tuple[Path, Path]:
    """Plot all three p x Accuracy distributions together (aligned blue, independent orange, swapped red)."""
    p_vals = [float(p) for p in config.p_values]
    p_strs = [f"{p:.2f}" for p in p_vals]
    xs = np.arange(len(p_vals), dtype=float)
    rng = np.random.default_rng(11)
    jitter_offsets = rng.uniform(-0.055, 0.055, size=30)

    specs = [
        ("aligned", "Aligned Test Variation", "#1f77b4", "#cfe2f3", -0.25),
        ("independent", "Independent Test Variation", "#ff7f0e", "#ffe0b2", 0.00),
        ("swapped", "Swapped Test Variation", "#d62728", "#f8c8c8", 0.25),
    ]

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=300, constrained_layout=True)

    for sep_x in xs[:-1] + 0.5:
        ax.axvline(sep_x, color="#dddddd", linestyle="-", linewidth=0.9, zorder=0)

    ax.axhline(
        10.0,
        color="#666666",
        linestyle=":",
        linewidth=1.1,
        alpha=0.85,
        zorder=1,
        label="Uniform random chance (10.0%)",
    )

    for view_key, label, color, fill_color, x_off in specs:
        positions = xs + x_off
        digit_groups = [view_digit_seed_acc[view_key][ps] for ps in p_strs]
        run_lists = [view_run_seed_acc[view_key][ps] for ps in p_strs]
        run_means = [float(np.mean(r)) for r in run_lists]
        run_mins = [float(np.min(r)) for r in run_lists]
        run_maxs = [float(np.max(r)) for r in run_lists]

        bp = ax.boxplot(
            digit_groups,
            positions=positions,
            widths=0.20,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        for box in bp["boxes"]:
            box.set(facecolor=fill_color, edgecolor=color, linewidth=1.1, alpha=0.68)
        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=1.0, alpha=0.75)
        for cap in bp["caps"]:
            cap.set(color=color, linewidth=1.1, alpha=0.85)
        for median in bp["medians"]:
            median.set(color="#222222", linewidth=1.2, linestyle=":")

        for idx, vals in enumerate(digit_groups):
            n_pts = len(vals)
            ax.scatter(
                positions[idx] + jitter_offsets[:n_pts],
                vals,
                s=10,
                color=color,
                alpha=0.30,
                edgecolors="none",
                zorder=3,
            )

        ax.fill_between(
            positions,
            run_mins,
            run_maxs,
            color=color,
            alpha=0.14,
            zorder=1,
        )
        ax.plot(
            positions,
            run_means,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5.2,
            zorder=5,
            label=f"{label} (3-seed mean & digit-seed dist.)",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(p_strs, fontsize=9.8)
    ax.set_xlim(-0.52, len(p_vals) - 1 + 0.52)
    ax.set_ylim(-4.5, 106.5)
    ax.set_xlabel("Training color-digit correlation strength $p$ (counterexample rate $1 - p$)", fontsize=10.0)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10.0)
    ax.set_title(
        r"Combined $p \times \mathrm{Accuracy}$ Distributions Across All Three Test Variations",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="lower left", fontsize=8.4, framealpha=0.94)

    png_path = output_dir / "dist_p_vs_acc_combined.png"
    pdf_path = output_dir / "dist_p_vs_acc_combined.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def run_step4_analysis(
    *,
    step1_dir: Path,
    step3_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: DatasetConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Execute complete Step 4 analysis (Substeps 1 and 2) from verified Step 3 outputs."""
    substep1 = run_step4_substep1(step3_dir=step3_dir, output_dir=output_dir, config=config)
    palette = load_palette(step1_dir / "palette.json")
    breakdowns = _collect_prediction_breakdowns(
        step3_dir / "predictions.csv",
        palette.preferred_color_by_digit,
        config=config,
    )

    by_digit_path = output_dir / "by_digit_swapped_metrics.csv"
    write_csv(by_digit_path, breakdowns["by_digit_rows"])

    conf_p010_csv = output_dir / "confusion_swapped_p010.csv"
    conf_p100_csv = output_dir / "confusion_swapped_p100.csv"
    _export_confusion_csv(breakdowns["conf_p010"], conf_p010_csv)
    _export_confusion_csv(breakdowns["conf_p100"], conf_p100_csv)
    conf_png, conf_pdf = plot_swapped_confusion_matrices(
        breakdowns["conf_p010"],
        breakdowns["conf_p100"],
        output_dir,
    )

    baseline_color_rows = build_baseline_color_table(
        step1_dir=step1_dir,
        color_counts=breakdowns["color_counts"],
        unique_ids_by_color=breakdowns["unique_test_ids_by_color"],
        indep_digit_totals=breakdowns["indep_digit_totals"],
    )
    baseline_color_csv = output_dir / "baseline_independent_by_color.csv"
    write_csv(baseline_color_csv, baseline_color_rows)

    examples, ex_png, ex_pdf = select_and_render_paired_examples(
        step1_dir=step1_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        seed101_focus=breakdowns["seed101_focus"],
        config=config,
    )
    view_dist_figs = plot_view_accuracy_distributions(
        breakdowns["view_digit_seed_acc"],
        breakdowns["view_run_seed_acc"],
        output_dir,
        config=config,
    )
    combined_dist_png, combined_dist_pdf = plot_combined_view_accuracy_distributions(
        breakdowns["view_digit_seed_acc"],
        breakdowns["view_run_seed_acc"],
        output_dir,
        config=config,
    )

    manifest = {
        "status": "PASSED",
        "command": "uv run python scripts/analyze_step4_sweep.py",
        "input_hashes": {
            **substep1["input_hashes"],
            "palette_json": file_digest(step1_dir / "palette.json"),
        },
        "outputs": {
            **substep1["outputs"],
            "by_digit_swapped_metrics_csv": str(by_digit_path),
            "confusion_swapped_p010_csv": str(conf_p010_csv),
            "confusion_swapped_p100_csv": str(conf_p100_csv),
            "confusion_figure_png": str(conf_png),
            "confusion_figure_pdf": str(conf_pdf),
            "baseline_independent_by_color_csv": str(baseline_color_csv),
            "paired_examples_csv": str(output_dir / "paired_examples.csv"),
            "paired_examples_json": str(output_dir / "paired_examples.json"),
            "paired_examples_png": str(ex_png),
            "paired_examples_pdf": str(ex_pdf),
            "dist_p_vs_acc_aligned_png": str(view_dist_figs["aligned"][0]),
            "dist_p_vs_acc_aligned_pdf": str(view_dist_figs["aligned"][1]),
            "dist_p_vs_acc_independent_png": str(view_dist_figs["independent"][0]),
            "dist_p_vs_acc_independent_pdf": str(view_dist_figs["independent"][1]),
            "dist_p_vs_acc_swapped_png": str(view_dist_figs["swapped"][0]),
            "dist_p_vs_acc_swapped_pdf": str(view_dist_figs["swapped"][1]),
            "dist_p_vs_acc_combined_png": str(combined_dist_png),
            "dist_p_vs_acc_combined_pdf": str(combined_dist_pdf),
        },
        "baseline_independent_by_color": baseline_color_rows,
        "paired_examples": examples,
    }
    write_json(output_dir / "step4_analysis_manifest.json", manifest)
    return manifest

