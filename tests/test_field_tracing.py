"""Tests for field line tracing functions."""

import pytest
import numpy as np

from particle_analysis import (
    trace_field_line,
    get_global_B_range,
    load_boozmn,
    get_example_boozmn,
)


class TestFieldTracing:
    """Tests for field line tracing."""

    @pytest.mark.unit
    def test_trace_field_line_output_shape(self, sample_boozer):
        """Test that trace_field_line returns arrays of expected length."""
        n_zeta = 1000
        zeta, B = trace_field_line(sample_boozer, s_idx=50, alpha=0.0, n_zeta=n_zeta)

        assert zeta.shape == (n_zeta,), f"zeta shape mismatch: {zeta.shape}"
        assert B.shape == (n_zeta,), f"B shape mismatch: {B.shape}"

    @pytest.mark.unit
    def test_trace_field_line_zeta_range(self, sample_boozer):
        """Test that zeta spans expected range (n_periods * 2pi/nfp)."""
        n_periods = 10
        nfp = sample_boozer['nfp']

        zeta, B = trace_field_line(sample_boozer, s_idx=50, alpha=0.0, n_periods=n_periods)

        expected_range = n_periods * 2 * np.pi / nfp
        actual_range = zeta[-1] - zeta[0]

        np.testing.assert_allclose(actual_range, expected_range, rtol=0.01,
            err_msg=f"zeta range {actual_range} != expected {expected_range}")

    @pytest.mark.unit
    def test_trace_field_line_B_positive(self, sample_boozer):
        """Test that B values are positive."""
        zeta, B = trace_field_line(sample_boozer, s_idx=50, alpha=0.0)

        assert np.all(B > 0), "B field should be positive everywhere"

    @pytest.mark.unit
    def test_trace_field_line_B_reasonable_range(self, sample_boozer):
        """Test that B values are in physically reasonable range."""
        zeta, B = trace_field_line(sample_boozer, s_idx=50, alpha=0.0)

        assert np.all(B > 1.0), "B unexpectedly small"
        assert np.all(B < 10.0), "B unexpectedly large"

    @pytest.mark.unit
    def test_trace_field_line_different_alpha(self, sample_boozer):
        """Test that different alpha values give different B profiles."""
        zeta1, B1 = trace_field_line(sample_boozer, s_idx=50, alpha=0.0)
        zeta2, B2 = trace_field_line(sample_boozer, s_idx=50, alpha=np.pi/2)

        np.testing.assert_allclose(zeta1, zeta2, rtol=1e-10)

        assert not np.allclose(B1, B2), "B profiles should differ for different alpha"

    @pytest.mark.unit
    def test_trace_field_line_different_s_idx(self, sample_boozer):
        """Test that different radial positions give different B profiles."""
        zeta1, B1 = trace_field_line(sample_boozer, s_idx=20, alpha=0.0)
        zeta2, B2 = trace_field_line(sample_boozer, s_idx=100, alpha=0.0)

        assert not np.allclose(B1, B2), "B profiles should differ at different radii"


class TestGlobalBRange:
    """Tests for get_global_B_range function."""

    @pytest.mark.unit
    def test_get_global_B_range_single_array(self):
        """Test get_global_B_range with single B array."""
        B = np.array([2.0, 2.5, 3.0, 2.8, 2.2])
        B_min, B_max = get_global_B_range([B])

        np.testing.assert_allclose(B_min, 2.0)
        np.testing.assert_allclose(B_max, 3.0)

    @pytest.mark.unit
    def test_get_global_B_range_multiple_arrays(self):
        """Test get_global_B_range aggregates across multiple arrays."""
        B1 = np.array([2.0, 2.5, 3.0])
        B2 = np.array([1.5, 2.0, 2.5])
        B3 = np.array([2.5, 3.0, 3.5])

        B_min, B_max = get_global_B_range([B1, B2, B3])

        np.testing.assert_allclose(B_min, 1.5)
        np.testing.assert_allclose(B_max, 3.5)

    @pytest.mark.unit
    def test_get_global_B_range_identical_arrays(self):
        """Test with identical arrays gives same result as single array."""
        B = np.array([2.0, 2.5, 3.0])

        B_min_single, B_max_single = get_global_B_range([B])
        B_min_multi, B_max_multi = get_global_B_range([B, B, B])

        np.testing.assert_allclose(B_min_single, B_min_multi)
        np.testing.assert_allclose(B_max_single, B_max_multi)
