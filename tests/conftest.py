from pathlib import Path

import pytest


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture
def trained_sweep() -> None:
    training_dir = ARTIFACTS_DIR / "step_2_training"
    if not (training_dir / "checkpoints").is_dir():
        pytest.skip("Run the training sweep first")


@pytest.fixture
def evaluated_sweep() -> None:
    evaluation_dir = ARTIFACTS_DIR / "step_3_evaluation"
    if not (evaluation_dir / "evaluation_manifest.json").is_file():
        pytest.skip("Evaluate and verify the sweep first")
