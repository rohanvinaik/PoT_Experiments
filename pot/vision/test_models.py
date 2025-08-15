import torch
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pot.vision import models


def test_mock_model_features_deterministic_shape():
    model = models.MockVisionModel(feature_dim=16)
    x = torch.randn(2, 3, 8, 8)
    f1 = model.get_features(x)
    f2 = model.get_features(x)
    assert f1.shape == (2, 16)
    assert torch.allclose(f1, f2)
    norms = f1.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))
    assert f1.device.type == model.device


@pytest.mark.parametrize("variant, dim", [("resnet18", 512), ("resnet50", 2048)])
def test_resnet_features_deterministic_shape(variant, dim):
    if not models.TORCHVISION_AVAILABLE:
        pytest.skip("torchvision not available")
    vm = models.VisionModel(model_name=variant)
    vm.model = models.load_resnet(variant, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    f1 = vm.get_features(x)
    f2 = vm.get_features(x)
    assert f1.shape == (2, dim)
    assert torch.allclose(f1, f2)
    norms = f1.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))
    assert f1.device.type == vm.device
