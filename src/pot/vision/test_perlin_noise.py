import torch
import numpy as np

from pot.vision.verifier import VisionVerifier
from pot.vision.models import MockVisionModel

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def _get_verifier():
    return VisionVerifier(MockVisionModel(), use_sequential=False, detect_wrappers=False)


def test_perlin_noise_reproducible():
    verifier = _get_verifier()
    img1 = verifier._generate_perlin_noise((64, 64), octaves=2, scale=0.05, seed=123)
    img2 = verifier._generate_perlin_noise((64, 64), octaves=2, scale=0.05, seed=123)
    img3 = verifier._generate_perlin_noise((64, 64), octaves=2, scale=0.05, seed=124)
    assert torch.allclose(img1, img2)
    assert not torch.allclose(img1, img3)


def test_perlin_noise_low_frequency():
    verifier = _get_verifier()
    img = verifier._generate_perlin_noise((128, 128), octaves=4, scale=0.05, seed=42)
    arr = img.numpy()[0]
    arr = arr - arr.mean()
    power = np.abs(np.fft.rfft2(arr)) ** 2
    low = power[:10, :10].sum()
    high = power[10:, 10:].sum()
    assert low > high
