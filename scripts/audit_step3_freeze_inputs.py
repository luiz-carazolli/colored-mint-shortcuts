"""Execute Step 3 Substep 1: Freeze inputs and verify paired official test views."""

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
from colored_mnist.evaluate import freeze_and_verify_inputs


def main() -> None:
    step1_dir = PROJECT_ROOT / DEFAULT_CONFIG.output_dir
    step2_dir = PROJECT_ROOT / "artifacts" / "step_2_training"
    raw_dir = PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir
    step3_dir = PROJECT_ROOT / "artifacts" / "step_3_evaluation"
    step3_dir.mkdir(parents=True, exist_ok=True)
    audit_path = step3_dir / "input_freeze_audit.json"

    frozen = freeze_and_verify_inputs(
        step1_dir=step1_dir,
        step2_dir=step2_dir,
        raw_dir=raw_dir,
        config=DEFAULT_CONFIG,
        output_path=audit_path,
    )

    payload = frozen.to_dict()
    print(json.dumps(
        {
            "status": payload["status"],
            "registry_sha256": payload["registry_sha256"],
            "num_sweep_runs": len(payload["sweep_runs"]),
            "excluded_runs": payload["excluded_runs"],
            "step1_hashes": payload["step1_hashes"],
            "raw_idx_hashes": payload["raw_idx_hashes"],
            "test_pairs_summary": payload["test_pairs_summary"],
            "audit_path": str(audit_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
