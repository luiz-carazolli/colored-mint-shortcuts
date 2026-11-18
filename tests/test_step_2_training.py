"""Training preparation checks that never update model parameters."""

import random

import numpy as np
import torch

from colored_mnist.config import DEFAULT_CONFIG
from colored_mnist.matrix import create_dataloader, get_paired_train_seed
from colored_mnist.models import build_model
from colored_mnist.train import (
    BATCH_SIZE,
    capture_rng,
    make_order_generator,
    read_checkpoint,
    restore_rng,
    write_atomic,
)


def test_classifier_shape_and_paired_initialization():
    for dataset_seed in DEFAULT_CONFIG.dataset_seeds:
        train_seed = get_paired_train_seed(dataset_seed)
        rng_before = torch.get_rng_state()
        model = build_model(train_seed)
        assert torch.equal(torch.get_rng_state(), rng_before)
        for _ in DEFAULT_CONFIG.p_values:
            paired_model = build_model(train_seed)
            assert all(
                torch.equal(left, right)
                for left, right in zip(model.parameters(), paired_model.parameters(), strict=True)
            )
    assert sum(parameter.numel() for parameter in model.parameters()) == 106154
    with torch.no_grad():
        logits = model(torch.zeros(4, 3, 28, 28))
    assert logits.shape == (4, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_order_and_rng_resume_without_optimization(tmp_path):
    seed = 201
    device = torch.device("cpu")
    order = make_order_generator(seed)
    paired = make_order_generator(seed)
    # Indices stand in for the same ordered source images in each condition.
    indices = range(54000)
    loader = create_dataloader(indices, BATCH_SIZE, shuffle=True, generator=order)
    paired_loader = create_dataloader(indices, BATCH_SIZE, shuffle=True, generator=paired)

    first_epoch = list(loader)
    assert len(first_epoch) == 211 and len(first_epoch[-1]) == 240
    assert torch.equal(torch.cat(first_epoch), torch.cat(list(paired_loader)))

    path = tmp_path / "epoch.pt"
    model = build_model(seed)
    saved = {"rng": capture_rng(order, device), "model_state_dict": model.state_dict()}
    write_atomic(path, saved, checkpoint=True)
    expected_random = (random.random(), np.random.random(), torch.rand(1).item())
    uninterrupted_epoch = torch.cat(list(loader))

    resumed = make_order_generator(0)
    restored = read_checkpoint(path)
    resumed_model = build_model(0)
    resumed_model.load_state_dict(restored["model_state_dict"])
    assert all(torch.equal(left, right) for left, right in
               zip(model.parameters(), resumed_model.parameters(), strict=True))
    restore_rng(restored["rng"], resumed, device)
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected_random
    resumed_loader = create_dataloader(indices, BATCH_SIZE, shuffle=True, generator=resumed)
    assert torch.equal(torch.cat(list(resumed_loader)), uninterrupted_epoch)
    assert torch.equal(torch.cat(list(paired_loader)), uninterrupted_epoch)
