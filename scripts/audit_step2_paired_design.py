"""Audit paired initialization and data presentation without optimizing the model."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.matrix import RunSpec, build_experimental_matrix, create_dataloader, get_training_dataset
from colored_mnist.models import build_model
from colored_mnist.train import (
    BATCH_SIZE, EPOCHS, TrainingSetup, configure_runtime, make_order_generator,
    read_checkpoint, write_json,
)


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(param.detach().numpy().tobytes())
    return digest.hexdigest()


def source_keys(dataset: ColoredMNISTDataset) -> list[tuple[str, str, int, int]]:
    return [(entry.original_id, entry.source_split, entry.source_index, entry.label)
            for entry in dataset.entries]


def audit_order(spec: RunSpec, sample_count: int) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Use the training loader's sampling path with indices as lightweight payloads."""
    generator = make_order_generator(spec.train_seed)
    loader = create_dataloader(range(sample_count), BATCH_SIZE, shuffle=True, generator=generator)
    epoch_hashes = []
    first_batch = None
    for _ in range(EPOCHS):
        batches = list(loader)
        assert len(batches) == (sample_count + BATCH_SIZE - 1) // BATCH_SIZE
        assert all(len(batch) == BATCH_SIZE for batch in batches[:-1])
        assert len(batches[-1]) == (sample_count % BATCH_SIZE or BATCH_SIZE)
        indices = torch.cat(batches)
        assert torch.equal(torch.sort(indices).values, torch.arange(sample_count)), spec.run_id
        epoch_hashes.append(hashlib.sha256(indices.numpy().astype("<i8", copy=False).tobytes()).hexdigest())
        if first_batch is None:
            first_batch = batches[0]
    return epoch_hashes, first_batch, generator.get_state()


def audit_rendered_batch(
    spec: RunSpec, dataset: ColoredMNISTDataset, expected_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    loader = create_dataloader(dataset, BATCH_SIZE, shuffle=True,
                               generator=make_order_generator(spec.train_seed))
    images, labels, meta = next(iter(loader))
    entries = [dataset.entries[index] for index in expected_indices.tolist()]
    assert meta["original_id"] == [entry.original_id for entry in entries]
    assert torch.equal(labels, torch.tensor([entry.label for entry in entries]))
    assert torch.equal(meta["color_id"], torch.tensor([entry.color_id for entry in entries]))
    assert images.shape == (BATCH_SIZE, 3, 28, 28) and images.dtype == torch.float32

    masks = dataset.raw.train_images[[entry.source_index for entry in entries]]
    pixels = images.permute(0, 2, 3, 1).numpy()
    assert (pixels[masks == 255] == 1.0).all()
    for rendered, mask, entry in zip(pixels, masks, entries, strict=True):
        color = np.asarray(dataset.palette.colors[entry.color_id].srgb_u8, dtype=np.float32) / 255
        assert (rendered[mask == 0] == color).all(), entry.original_id
    return images, meta["color_id"], masks


def run_paired_design_audit() -> dict:
    if not __debug__:
        raise RuntimeError("Run the audit with Python assertions enabled")
    setup = TrainingSetup(
        step1_dir=PROJECT_ROOT / DEFAULT_CONFIG.output_dir,
        raw_dir=PROJECT_ROOT / DEFAULT_CONFIG.raw_data_dir,
        output_dir=PROJECT_ROOT / "artifacts" / "step_2_training",
    )
    environment = configure_runtime(setup)
    runtime_settings = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_threads": torch.get_num_threads(),
    }
    assert runtime_settings == {
        "deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "cudnn_allow_tf32": False,
        "matmul_precision": "highest", "cublas_workspace_config": ":4096:8",
        "torch_threads": setup.threads,
    }
    specs = build_experimental_matrix(setup.step1_dir, setup.raw_dir)
    assert len(specs) == len(DEFAULT_CONFIG.dataset_seeds) * len(DEFAULT_CONFIG.p_values)

    pairs = []
    pilot_path = setup.output_dir / "checkpoints" / "pilot_p_0.10_dseed_101_tseed_201.pt"
    pilot = read_checkpoint(pilot_path) if pilot_path.exists() else None
    pilot_order_checked = False
    for dataset_seed in DEFAULT_CONFIG.dataset_seeds:
        conditions = [spec for spec in specs if spec.dataset_seed == dataset_seed]
        assert tuple(spec.p for spec in conditions) == DEFAULT_CONFIG.p_values
        train_seed = conditions[0].train_seed
        assert all(spec.train_seed == train_seed == dataset_seed + 100 for spec in conditions)
        runs = []
        for spec in conditions:
            rng_before = torch.get_rng_state()
            model = build_model(spec.train_seed)
            assert torch.equal(rng_before, torch.get_rng_state()), spec.run_id
            assert all(param.dtype == torch.float32 for param in model.parameters())
            assert sum(param.numel() for param in model.parameters()) == 106154
            digest = parameter_hash(model)

            dataset = get_training_dataset(spec, setup.step1_dir, setup.raw_dir, return_metadata=True)
            assert len(dataset) == spec.num_train_samples == 54000
            keys = source_keys(dataset)
            assert all(key[1] == "train" for key in keys)
            assert dataset.raw.test_images is None and dataset.raw.test_labels is None
            epoch_hashes, first_batch, order_state = audit_order(spec, len(dataset))
            images, colors, masks = audit_rendered_batch(spec, dataset, first_batch)
            if not runs:
                reference = (digest, keys, dataset.raw.raw_file_hashes, dataset.palette,
                             epoch_hashes, images, colors, masks)
            ref_digest, ref_keys, ref_raw, ref_palette, ref_epochs, ref_images, ref_colors, ref_masks = reference
            assert digest == ref_digest and epoch_hashes == ref_epochs, spec.run_id
            assert keys == ref_keys and dataset.raw.raw_file_hashes == ref_raw, spec.run_id
            assert dataset.palette == ref_palette and np.array_equal(masks, ref_masks), spec.run_id
            changed_colors = colors != ref_colors
            changed_images = (images != ref_images).flatten(1).any(dim=1)
            assert torch.equal(changed_colors, changed_images), spec.run_id

            if pilot is not None and spec.to_dict() == pilot["config"]["condition"]:
                assert pilot["completed_epoch"] == EPOCHS and pilot["environment"] == environment
                assert torch.equal(order_state, pilot["rng"]["data_order"])
                pilot_order_checked = True
            runs.append({"run_id": spec.run_id, "p": spec.p,
                         "changed_colors_vs_baseline_in_first_batch": int(changed_colors.sum())})
        pairs.append({
            "dataset_seed": dataset_seed,
            "train_seed": train_seed,
            "parameter_sha256": ref_digest,
            "epoch_index_order_sha256": ref_epochs,
            "conditions": runs,
        })

    assert len({pair["parameter_sha256"] for pair in pairs}) == len(pairs)
    for epoch in range(EPOCHS):
        assert len({pair["epoch_index_order_sha256"][epoch] for pair in pairs}) == len(pairs)
    assert pilot is None or pilot_order_checked
    report = {
        "status": "PASSED",
        "scope": {
            "ordered_source_identity_and_labels": "All training rows in all 21 conditions",
            "initialization_and_cpu_rng_isolation": "All 21 condition seeds",
            "batch_order": "Production loader over indices mapped to the verified source rows",
            "epochs_per_condition": EPOCHS,
            "batches_per_epoch": (54000 + BATCH_SIZE - 1) // BATCH_SIZE,
            "final_batch_size": 54000 % BATCH_SIZE or BATCH_SIZE,
            "rendering": "First actual training batch per condition. Checks original masks, fully "
                         "opaque white strokes, zero-alpha background colors, and image changes "
                         "against the p=0.10 batch. Partially transparent stroke edges blend with the background.",
            "rendered_samples_per_condition": BATCH_SIZE,
            "runtime": "CPU execution. CUDA and cuDNN flags were inspected on CPU.",
        },
        "pilot_final_data_order_rng_matches": pilot_order_checked if pilot is not None else None,
        "pairs": pairs,
        "runtime_environment": environment,
        "runtime_settings": runtime_settings,
    }
    path = setup.output_dir / "paired_design_audit.json"
    write_json(path, report)
    print(f"Audited {len(specs)} conditions across {EPOCHS} epochs. Report saved to {path}")
    return report


if __name__ == "__main__":
    run_paired_design_audit()
