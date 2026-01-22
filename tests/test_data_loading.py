"""Tests for data loading functions."""

import pytest
import numpy as np
from pathlib import Path

from particle_analysis import (
    get_example_wout,
    get_example_boozmn,
    load_boozmn,
)


class TestDataLoading:
    """Tests for data loading functions."""

    @pytest.mark.unit
    def test_get_example_wout_returns_valid_path(self):
        """Test that get_example_wout returns existing file path."""
        path = get_example_wout()
        assert Path(path).exists(), f"Example wout file not found at {path}"
        assert str(path).endswith('.nc'), "Expected NetCDF file extension"

    @pytest.mark.unit
    def test_get_example_boozmn_returns_valid_path(self):
        """Test that get_example_boozmn returns existing file path."""
        path = get_example_boozmn()
        assert Path(path).exists(), f"Example boozmn file not found at {path}"
        assert str(path).endswith('.nc'), "Expected NetCDF file extension"

    @pytest.mark.unit
    def test_load_boozmn_returns_expected_keys(self):
        """Test that load_boozmn returns dict with required keys."""
        boozer = load_boozmn(get_example_boozmn())

        required_keys = ['nfp', 'ns', 'bmnc_b', 'xm_b', 'xn_b', 'iota', 'Boozer_G', 'Boozer_I']
        for key in required_keys:
            assert key in boozer, f"Missing required key: {key}"

    @pytest.mark.unit
    def test_load_boozmn_array_shapes(self):
        """Test that loaded arrays have consistent shapes."""
        boozer = load_boozmn(get_example_boozmn())

        ns = boozer['ns']

        assert boozer['iota'].shape == (ns,), f"iota shape mismatch: {boozer['iota'].shape}"
        assert boozer['Boozer_G'].shape == (ns,), f"Boozer_G shape mismatch"
        assert boozer['Boozer_I'].shape == (ns,), f"Boozer_I shape mismatch"

        assert boozer['bmnc_b'].shape[1] == ns, f"bmnc_b radial dimension mismatch"

        nmodes = boozer['bmnc_b'].shape[0]
        assert boozer['xm_b'].shape == (nmodes,), f"xm_b shape mismatch"
        assert boozer['xn_b'].shape == (nmodes,), f"xn_b shape mismatch"

    @pytest.mark.unit
    def test_load_boozmn_values_reasonable(self):
        """Test that iota, Boozer_G, Boozer_I have physically reasonable values."""
        boozer = load_boozmn(get_example_boozmn())

        assert np.all(np.abs(boozer['iota']) < 10), "iota values unreasonably large"
        assert np.all(np.isfinite(boozer['iota'])), "iota contains non-finite values"

        assert np.all(boozer['Boozer_G'] > 0), "Boozer_G should be positive"

        assert np.all(np.isfinite(boozer['Boozer_G'])), "Boozer_G contains non-finite values"
        assert np.all(np.isfinite(boozer['Boozer_I'])), "Boozer_I contains non-finite values"

    @pytest.mark.unit
    def test_load_boozmn_nfp_positive_integer(self):
        """Test that nfp (number of field periods) is a positive integer."""
        boozer = load_boozmn(get_example_boozmn())

        assert isinstance(boozer['nfp'], (int, np.integer)), "nfp should be an integer"
        assert boozer['nfp'] > 0, "nfp should be positive"
        assert boozer['nfp'] <= 10, "nfp unexpectedly large (sanity check)"
