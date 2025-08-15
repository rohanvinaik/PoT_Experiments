import torch
import numpy as np
import random
import time
from contextlib import contextmanager

def set_reproducibility(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)

@contextmanager
def timer(name: str = "Operation"):
    start = time.time()
    yield
    print(f"{name} took {time.time() - start:.4f} seconds")