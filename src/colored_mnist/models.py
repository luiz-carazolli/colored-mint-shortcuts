"""Fixed convolutional classifier for the background correlation sweep."""

import torch
from torch import nn


def build_model(train_seed: int) -> nn.Sequential:
    """Initialize on CPU without advancing the caller's random stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(train_seed)
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).float()
