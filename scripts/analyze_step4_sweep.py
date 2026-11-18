"""Run Step 4 (Substeps 1 & 2): Verified sweep analysis, error breakdowns, and figures."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colored_mnist.analyze import run_step4_analysis
from colored_mnist.config import DEFAULT_CONFIG


def main() -> None:
    step1_dir = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
    step3_dir = PROJECT_ROOT / "artifacts" / "step_3_evaluation"
    raw_dir = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir
    output_dir = PROJECT_ROOT / "artifacts" / "step_4_analysis"

    manifest = run_step4_analysis(
        step1_dir=step1_dir,
        step3_dir=step3_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        config=DEFAULT_CONFIG,
    )
    print(f"Step 4 analysis status: {manifest['status']}")
    for key, path in manifest["outputs"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
