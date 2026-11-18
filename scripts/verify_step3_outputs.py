"""Execute Step 3 Substep 3: Verify saved outputs, recompute metrics, and write manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.evaluate import verify_saved_evaluation_outputs


def main() -> None:
    step1_dir = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
    step2_dir = PROJECT_ROOT / "artifacts" / "step_2_training"
    raw_dir = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir
    step3_dir = PROJECT_ROOT / "artifacts" / "step_3_evaluation"
    manifest_path = step3_dir / "evaluation_manifest.json"

    manifest = verify_saved_evaluation_outputs(
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        raw_dir=raw_dir,
        output_dir=step3_dir,
        config=DEFAULT_CONFIG,
        manifest_output_path=manifest_path,
    )

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "evaluator_version": manifest["evaluator_version"],
                "verification_results": manifest["verification_results"],
                "output_files": manifest["output_files"],
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
