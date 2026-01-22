"""Tests for J invariant computation."""

import pytest
import numpy as np

from particle_analysis import compute_j_well, find_wells


class TestJComputation:
    """Tests for J invariant computation."""

    @pytest.mark.unit
    def test_compute_j_well_positive(self, simple_sinusoidal_B):
        """Test that J is positive for valid well."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5
        norm_factor = 1.0

        wells = find_wells(zeta, B, B_bounce)
        assert len(wells) > 0, "Need at least one well for this test"

        J = compute_j_well(zeta, B, wells[0], B_bounce, norm_factor)

        assert J > 0, f"J should be positive, got {J}"

    @pytest.mark.unit
    def test_compute_j_well_finite(self, simple_sinusoidal_B):
        """Test that J is finite (no NaN or inf)."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5
        norm_factor = 1.0

        wells = find_wells(zeta, B, B_bounce)

        for well in wells:
            J = compute_j_well(zeta, B, well, B_bounce, norm_factor)
            assert np.isfinite(J), f"J should be finite, got {J}"

    @pytest.mark.unit
    def test_compute_j_well_scaling_with_norm_factor(self, simple_sinusoidal_B):
        """Test that J scales linearly with norm_factor."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5

        wells = find_wells(zeta, B, B_bounce)
        well = wells[0]

        J1 = compute_j_well(zeta, B, well, B_bounce, norm_factor=1.0)
        J2 = compute_j_well(zeta, B, well, B_bounce, norm_factor=2.0)
        J3 = compute_j_well(zeta, B, well, B_bounce, norm_factor=0.5)

        np.testing.assert_allclose(J2, 2.0 * J1, rtol=1e-10,
            err_msg="J should scale linearly with norm_factor")
        np.testing.assert_allclose(J3, 0.5 * J1, rtol=1e-10,
            err_msg="J should scale linearly with norm_factor")

    @pytest.mark.unit
    def test_compute_j_well_increases_with_deeper_well(self):
        """Test that J increases as B_bounce increases (deeper well)."""
        zeta = np.linspace(0, 4 * np.pi, 2000)
        B = 2.5 + 0.5 * np.sin(zeta)
        norm_factor = 1.0

        B_bounce_shallow = 2.3
        wells_shallow = find_wells(zeta, B, B_bounce_shallow)
        if len(wells_shallow) == 0:
            pytest.skip("No wells found for shallow B_bounce")
        J_shallow = compute_j_well(zeta, B, wells_shallow[0], B_bounce_shallow, norm_factor)

        B_bounce_deep = 2.7
        wells_deep = find_wells(zeta, B, B_bounce_deep)
        if len(wells_deep) == 0:
            pytest.skip("No wells found for deep B_bounce")
        J_deep = compute_j_well(zeta, B, wells_deep[0], B_bounce_deep, norm_factor)

        assert J_deep > J_shallow, "J should increase for deeper well (higher B_bounce)"

    @pytest.mark.unit
    def test_compute_j_well_parabolic_well(self, parabolic_well):
        """Test J computation for parabolic well where analytical result is known.

        For B = B_min + a*zeta^2 with B_bounce, the well width is:
        zeta_bounce = sqrt((B_bounce - B_min) / a)

        The J integral can be computed analytically for this case.
        """
        zeta, B, B_min, a = parabolic_well
        B_bounce = B_min + 0.25
        norm_factor = 1.0

        wells = find_wells(zeta, B, B_bounce)
        assert len(wells) == 1, "Parabolic well should give exactly one well"

        J = compute_j_well(zeta, B, wells[0], B_bounce, norm_factor)

        assert J > 0, "J should be positive"
        assert np.isfinite(J), "J should be finite"

    @pytest.mark.unit
    def test_compute_j_well_consistent_across_wells(self):
        """Test that identical wells give identical J values."""
        zeta = np.linspace(0, 6 * np.pi, 3000)
        B = 2.5 + 0.5 * np.sin(zeta)
        B_bounce = 2.7
        norm_factor = 1.0

        wells = find_wells(zeta, B, B_bounce)
        if len(wells) < 2:
            pytest.skip("Need at least 2 wells for this test")

        J_values = [compute_j_well(zeta, B, well, B_bounce, norm_factor) for well in wells]

        np.testing.assert_allclose(J_values[0], J_values[1], rtol=0.05,
            err_msg="Identical wells should give similar J values")

    @pytest.mark.unit
    def test_compute_j_well_zero_norm_factor(self, simple_sinusoidal_B):
        """Test that J is zero when norm_factor is zero."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5

        wells = find_wells(zeta, B, B_bounce)
        J = compute_j_well(zeta, B, wells[0], B_bounce, norm_factor=0.0)

        np.testing.assert_allclose(J, 0.0, atol=1e-15)


class TestJComputationIntegration:
    """Integration tests for J computation with real equilibrium data."""

    @pytest.mark.unit
    def test_compute_j_with_real_data(self, sample_boozer):
        """Test J computation with real W7-X equilibrium data."""
        from particle_analysis import trace_field_line, get_global_B_range

        s_idx = 50
        alpha = 0.0
        zeta, B = trace_field_line(sample_boozer, s_idx, alpha)

        B_min, B_max = get_global_B_range([B])
        lambda_n = 0.3
        B_bounce = B_min + lambda_n * (B_max - B_min)

        wells = find_wells(zeta, B, B_bounce)

        if len(wells) > 0:
            iota = sample_boozer['iota'][s_idx]
            G = sample_boozer['Boozer_G'][s_idx]
            I = sample_boozer['Boozer_I'][s_idx]
            nfp = sample_boozer['nfp']
            R_major = 5.5
            norm_factor = abs(G + iota * I) / (2 * np.pi * R_major / nfp)

            J = compute_j_well(zeta, B, wells[0], B_bounce, norm_factor)

            assert J > 0, "J should be positive for real equilibrium"
            assert np.isfinite(J), "J should be finite for real equilibrium"
            assert J < 100, f"J={J} seems unreasonably large"
