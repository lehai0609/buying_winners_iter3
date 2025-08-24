# Repro utilities for M0
from __future__ import annotations
import random
from contextlib import contextmanager
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Seed Python and NumPy global RNGs."""
    random.seed(seed)
    np.random.seed(seed)

def make_rng(seed: int = 42) -> np.random.Generator:
    """Prefer dependency-injected RNGs over globals."""
    return np.random.default_rng(seed)

@contextmanager
def fixed_seed(seed: int = 42):
    """Temporarily fix RNGs, then restore previous states."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
