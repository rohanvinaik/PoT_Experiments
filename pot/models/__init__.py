"""Utility helpers for loading models with optional CPU-only execution."""

from __future__ import annotations

import torch


def load_model(path: str, cpu_only: bool = False):
    """Load a PyTorch model from ``path``.

    Args:
        path: Path to the model checkpoint.
        cpu_only: If True, force loading on CPU and avoid CUDA initialization.

    Returns:
        The loaded model.
    """
    map_location = "cpu" if cpu_only else None
    return torch.load(path, weights_only=False, map_location=map_location)
