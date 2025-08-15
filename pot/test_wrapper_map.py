import numpy as np

from pot.core.attacks import WrapperConfig
from pot.lm.attacks import wrapper_map as lm_wrapper_map, inverse_wrapper_map as lm_inverse_wrapper
from pot.vision.attacks import wrapper_map as vis_wrapper_map, inverse_wrapper_map as vis_inverse_wrapper


def check_transform(wrapper, inverse, cfg):
    logits = np.array([1.0, -2.0, 3.0], dtype=float)
    transformed = wrapper(logits, cfg)
    expected = logits / cfg.temperature + cfg.bias
    assert np.allclose(transformed, expected)
    restored = inverse(transformed, cfg)
    assert np.allclose(restored, logits)


def test_lm_wrapper_transform_and_inverse():
    cfg = WrapperConfig(temperature=2.0, bias=0.5)
    check_transform(lm_wrapper_map, lm_inverse_wrapper, cfg)


def test_vision_wrapper_transform_and_inverse():
    cfg = WrapperConfig(temperature=0.5, bias=-1.0)
    check_transform(vis_wrapper_map, vis_inverse_wrapper, cfg)
