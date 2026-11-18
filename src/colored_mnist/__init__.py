"""Controlled Colored MNIST package for shortcut-learning experiments."""

from colored_mnist.config import DatasetConfig, DEFAULT_CONFIG
from colored_mnist.dataset import ColoredMNISTDataset
from colored_mnist.palette import Palette, build_palette, composite_digit_srgb

__all__ = [
    "ColoredMNISTDataset",
    "DEFAULT_CONFIG",
    "DatasetConfig",
    "Palette",
    "build_palette",
    "composite_digit_srgb",
]

