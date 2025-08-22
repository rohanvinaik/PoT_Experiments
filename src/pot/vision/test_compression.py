import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pot.vision.models import apply_pruning, apply_quantization, CompressionConfig


def _train_simple_model():
    torch.manual_seed(0)
    X = torch.randn(512, 2)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model, dataset


def _accuracy(model, X, y):
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def count_nonzero(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())


def test_pruning_and_quantization_reduce_params_and_retain_accuracy():
    model, dataset = _train_simple_model()
    X, y = dataset.tensors
    base_acc = _accuracy(model, X, y)
    orig_params = count_nonzero(model)

    config = CompressionConfig(backend="fbgemm", prune_amount=0.5)

    pruned_model = apply_pruning(model, amount=config.prune_amount)
    pruned_params = count_nonzero(pruned_model)
    assert pruned_params < orig_params

    calib_loader = DataLoader(dataset, batch_size=32)
    try:
        quant_model = apply_quantization(
            pruned_model, backend=config.backend, calibration_data=calib_loader
        )
        quant_acc = _accuracy(quant_model, X, y)
    except NotImplementedError:
        pytest.skip("Quantization not supported on this backend")
    assert quant_acc >= base_acc - 0.1
