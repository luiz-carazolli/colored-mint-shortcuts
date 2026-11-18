"""Run Step 3 Substep 2: Standalone paired-test inference and metric computation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.evaluate import evaluate_sweep


def main() -> None:
    step1_dir = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
    step2_dir = PROJECT_ROOT / "artifacts" / "step_2_training"
    raw_dir = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir
    output_dir = PROJECT_ROOT / "artifacts" / "step_3_evaluation"

    result = evaluate_sweep(
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        config=DEFAULT_CONFIG,
    )
    print(
        f"Completed evaluation of {result['num_runs']} runs "
        f"({result['num_prediction_rows']} prediction rows, "
        f"{result['num_metric_rows']} metric rows)."
    )
    print(f"Saved predictions: {result['predictions_path']}")
    print(f"Saved metrics:     {result['metrics_path']}")


if __name__ == "__main__":
    main()
