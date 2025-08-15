import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pot.vision.probes import render_sine_grating, render_texture


def test_render_sine_grating_properties():
    h, w = 32, 32
    params = dict(freq=5, theta=np.pi / 4, phase=0.0, contrast=1.0, seed=123)
    img1 = render_sine_grating(h, w, **params)
    img2 = render_sine_grating(h, w, **params)

    assert img1.shape == (h, w)
    assert np.all(img1 >= 0.0) and np.all(img1 <= 1.0)
    assert np.allclose(img1, img2)


def test_render_texture_noise():
    h, w = 32, 32
    img1 = render_texture(h, w, texture_type="noise", seed=42)
    img2 = render_texture(h, w, texture_type="noise", seed=42)

    assert img1.shape == (h, w)
    assert np.all(img1 >= 0.0) and np.all(img1 <= 1.0)
    assert np.allclose(img1, img2)


def test_render_texture_checkerboard():
    h, w = 32, 32
    img1 = render_texture(h, w, texture_type="checkerboard", freq=4, theta=np.pi / 4)
    img2 = render_texture(h, w, texture_type="checkerboard", freq=4, theta=np.pi / 4)

    assert img1.shape == (h, w)
    assert np.all(img1 >= 0.0) and np.all(img1 <= 1.0)
    assert np.allclose(img1, img2)


def test_render_texture_perlin():
    h, w = 32, 32
    img1 = render_texture(h, w, octaves=2, scale=8.0, texture_type="perlin", seed=7)
    img2 = render_texture(h, w, octaves=2, scale=8.0, texture_type="perlin", seed=7)

    assert img1.shape == (h, w)
    assert np.all(img1 >= 0.0) and np.all(img1 <= 1.0)
    assert np.allclose(img1, img2)

