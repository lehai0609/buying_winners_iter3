import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import set_seed, make_rng, fixed_seed

def test_global_seed_determinism():
    set_seed(123)
    a1 = np.random.rand(3)
    set_seed(123)
    a2 = np.random.rand(3)
    assert np.allclose(a1, a2)

def test_generator_determinism():
    r1 = make_rng(7).normal(size=5)
    r2 = make_rng(7).normal(size=5)
    assert np.allclose(r1, r2)

def test_fixed_seed_context_manager_restores_state():
    # capture outer stream by drawing once
    _ = np.random.rand()
    before = np.random.rand()
    with fixed_seed(9):
        inner1 = np.random.rand()
        with fixed_seed(9):
            inner2 = np.random.rand()
        assert np.allclose(inner1, inner2)
    after = np.random.rand()
    # outer stream unchanged by the context
    assert not np.allclose(before, after)
