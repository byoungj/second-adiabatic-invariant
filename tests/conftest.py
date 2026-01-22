"""Shared fixtures for particle_analysis tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_boozer():
    """Load example Boozer data for integration tests."""
    from particle_analysis import load_boozmn, get_example_boozmn
    return load_boozmn(get_example_boozmn())


@pytest.fixture
def simple_sinusoidal_B():
    """Create simple sinusoidal B field for unit tests.

    Returns (zeta, B) where B oscillates between 2.0 and 3.0.
    """
    zeta = np.linspace(0, 4 * np.pi, 1000)
    B = 2.5 + 0.5 * np.sin(zeta)
    return zeta, B


@pytest.fixture
def parabolic_well():
    """Create parabolic well with known J solution.

    Returns (zeta, B, B_min, a) for B = B_min + a * zeta^2.
    """
    zeta = np.linspace(-1, 1, 1000)
    B_min = 2.0
    a = 0.5
    B = B_min + a * zeta**2
    return zeta, B, B_min, a


@pytest.fixture
def mock_boozer_dict():
    """Create minimal mock Boozer coordinate dictionary for unit tests."""
    ns = 10
    nmodes = 5
    return {
        'nfp': 5,
        'ns': ns,
        'bmnc_b': np.random.rand(nmodes, ns) * 0.1 + 2.5,
        'xm_b': np.array([0, 1, 1, 2, 2]),
        'xn_b': np.array([0, 0, 5, 0, 5]),
        'iota': np.linspace(0.8, 1.0, ns),
        'Boozer_G': np.ones(ns) * 20.0,
        'Boozer_I': np.ones(ns) * 0.1,
        'booz': None,
    }
