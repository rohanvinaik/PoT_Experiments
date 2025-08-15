import numpy as np


import numpy as np


class BaseModel:
    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def __call__(self, challenge):
        # challenge is a dict with key 'freq'
        freq = challenge.get("freq", 0.0)
        return np.array([self.scale * freq + self.bias])


def ReferenceModel():
    return BaseModel(scale=1.0)


def IdenticalModel():
    return BaseModel(scale=1.0)


def VariantModel():
    # Slightly different scaling to produce non-zero distances
    return BaseModel(scale=1.2)
