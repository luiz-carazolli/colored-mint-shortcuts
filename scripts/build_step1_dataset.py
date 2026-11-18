"""Build, verify, and document all Step 1 Colored MNIST deliverables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision

# Allow running directly as `python scripts/build_step1_dataset.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colored_mnist.assignments import (
    build_all_assignments,
    save_heldout_manifest,
    save_training_manifest,
)
from colored_mnist.config import DEFAULT_CONFIG, DatasetConfig
from colored_mnist.palette import build_palette
from colored_mnist.splits import (
    build_stratified_splits,
    load_raw_mnist,
    save_split_manifest,
)
from colored_mnist.verify import (
    render_validation_sample_grid,
    run_all_verifications,
    save_digit_color_counts_csv,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_dataset_readme(output_dir: Path, config: DatasetConfig) -> None:
    readme = f"""# Step 1 — Controlled Colored MNIST Dataset Artifacts

This directory contains the reproducible specifications, manifests, and verification artifacts generated from `scripts/build_step1_dataset.py`.

## Source dataset & regeneration procedure

- **Source:** {config.source_name} ({config.source_url})
- **Raw data directory:** `{config.raw_data_dir}` (official 60,000 train / 10,000 test IDX files; SHA256 hashes recorded in `dataset_metadata.json`).
- **Splitting:** Official test split (`10,000`) is preserved untouched. `{int(config.val_fraction * 100)}%` (`6,000`) of the official training split is set aside for validation using stratified sampling with `split_seed={config.split_seed}`; the remaining `54,000` form the training split. Every sample receives a stable `original_id` (`mnist_train_XXXXX` / `mnist_test_XXXXX`) prior to recoloring.
- **Palette & rendering:** Ten evenly spaced hues in CIELAB (`L*={config.lab_lightness}`, `C*={config.lab_chroma}`, `h0={config.lab_hue_start_deg} deg`, step `{config.lab_hue_step_deg} deg`) converted via D65 XYZ to unclipped sRGB (`palette.json`). Each MNIST grayscale image (`uint8`) is normalized to an opacity mask `alpha in [0, 1]`, composited with a white foreground `(1, 1, 1)` over the quantized background's linear RGB value, and encoded back to 8-bit sRGB.
- **Assignments:**
  - **Training:** For each `dataset_seed` in `{list(config.dataset_seeds)}`, a single uniform draw `u ~ U(0, 1)` and alternative nonpreferred color are sampled per training `original_id` and reused across `p in {list(config.p_values)}`.
  - **Validation & Test:** Three fixed paired views (`aligned`, `independent` with `evaluation_seed={config.evaluation_seed}`, and `swapped` via antipodal class permutation `{list(config.swapped_target_by_digit)}`) share identical foreground masks and labels per `original_id`.

## Files in this directory

- `config.json` — Full dataset configuration, seeds, and `p` sweep schedule.
- `palette.json` — CIELAB coordinates, quantized sRGB values, hex codes, luminance, contrast ratios vs. white, and class/swapped mappings.
- `dataset_metadata.json` — Source IDX SHA256 hashes, environment versions, and regeneration command.
- `manifests/splits.csv` — Stable `original_id`, official source index, assigned split, and label for all 70,000 digits.
- `manifests/train_seed_<seed>.csv` — Training base draws and resolved color IDs across all `p` values.
- `manifests/heldout_views.csv` — Paired `aligned`, `independent`, and `swapped` assignments (+ `swapped_target_class`) for all 16,000 validation and test digits.
- `digit_color_counts.csv` — Realized 10x10 digit-by-color counts across every training condition and held-out view.
- `verification_report.json` — Automated verification results and binomial consistency z-scores.
- `validation_sample_grid.png` — Visual inspection grid showing one validation sample per digit `0..9` rendered across all 10 palette colors (black border = aligned preferred color, amber border = swapped target color).
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main(config: DatasetConfig = DEFAULT_CONFIG) -> None:
    output_dir = PROJECT_ROOT / config.output_dir
    raw_dir = PROJECT_ROOT / config.raw_data_dir
    manifests_dir = output_dir / "manifests"

    raw = load_raw_mnist(raw_dir, download=True)
    palette = build_palette(config)
    splits = build_stratified_splits(raw, config)
    train_by_seed, heldout = build_all_assignments(splits, palette, config)

    report, freq_summaries = run_all_verifications(
        raw=raw,
        splits=splits,
        train_by_seed=train_by_seed,
        heldout=heldout,
        palette=palette,
        config=config,
    )

    # Save manifests
    save_split_manifest(splits, manifests_dir / "splits.csv")
    for seed, draws in train_by_seed.items():
        save_training_manifest(draws, config.p_values, manifests_dir / f"train_seed_{seed}.csv")
    save_heldout_manifest(heldout, manifests_dir / "heldout_views.csv")

    # Save specs, metadata, verification report, count tables, and visual grid
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "palette.json", palette.to_dict())
    _write_json(
        output_dir / "dataset_metadata.json",
        {
            "source_name": config.source_name,
            "source_url": config.source_url,
            "raw_idx_sha256": raw.raw_file_hashes,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "regeneration_command": "uv run python scripts/build_step1_dataset.py",
        },
    )
    _write_json(output_dir / "verification_report.json", report)
    save_digit_color_counts_csv(freq_summaries, output_dir / "digit_color_counts.csv")
    grid_path = render_validation_sample_grid(
        raw=raw,
        splits=splits,
        palette=palette,
        output_path=output_dir / "validation_sample_grid.png",
    )
    _write_dataset_readme(output_dir, config)

    print(f"Step 1 build & verification: {report['status']}")
    print(
        f"Splits: train={report['splits']['train_count']}, "
        f"val={report['splits']['val_count']}, test={report['splits']['test_count']}"
    )
    print(
        f"Palette contrast vs white: "
        f"{report['palette_summary']['min_contrast_vs_white']} .. "
        f"{report['palette_summary']['max_contrast_vs_white']}"
    )
    print(f"Artifacts saved to: {output_dir}")
    print(f"Validation grid saved to: {grid_path}")


if __name__ == "__main__":
    main()

