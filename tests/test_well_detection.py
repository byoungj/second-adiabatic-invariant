"""Tests for magnetic well detection."""

import pytest
import numpy as np

from particle_analysis import find_wells


class TestWellDetection:
    """Tests for magnetic well detection."""

    @pytest.mark.unit
    def test_find_wells_simple_sinusoid(self, simple_sinusoidal_B):
        """Test well detection on simple sinusoidal B field."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.7

        wells = find_wells(zeta, B, B_bounce)

        assert len(wells) >= 1, f"Expected at least 1 well, found {len(wells)}"

    @pytest.mark.unit
    def test_find_wells_no_wells(self, simple_sinusoidal_B):
        """Test when B_bounce < B_min (no wells)."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 1.5

        wells = find_wells(zeta, B, B_bounce)

        assert len(wells) == 0, f"Expected no wells, found {len(wells)}"

    @pytest.mark.unit
    def test_find_wells_returns_zeta_tuples(self, simple_sinusoidal_B):
        """Test that wells are returned as (zeta_start, zeta_end) tuples."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5

        wells = find_wells(zeta, B, B_bounce)

        for well in wells:
            assert isinstance(well, tuple), "Well should be a tuple"
            assert len(well) == 2, "Well tuple should have 2 elements"
            zeta_start, zeta_end = well
            assert isinstance(zeta_start, (float, np.floating)), "Well start should be float (zeta value)"
            assert isinstance(zeta_end, (float, np.floating)), "Well end should be float (zeta value)"
            assert zeta_start < zeta_end, "Well start should be before end"

    @pytest.mark.unit
    def test_find_wells_zeta_within_bounds(self, simple_sinusoidal_B):
        """Test that well boundaries are within the zeta array bounds."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5

        wells = find_wells(zeta, B, B_bounce)

        for zeta_start, zeta_end in wells:
            assert zeta_start >= zeta[0], "Well start should be >= zeta[0]"
            assert zeta_end <= zeta[-1], "Well end should be <= zeta[-1]"

    @pytest.mark.unit
    def test_find_wells_non_overlapping(self, simple_sinusoidal_B):
        """Test that detected wells do not overlap."""
        zeta, B = simple_sinusoidal_B
        B_bounce = 2.5

        wells = find_wells(zeta, B, B_bounce)

        if len(wells) > 1:
            for i in range(len(wells) - 1):
                _, end_i = wells[i]
                start_next, _ = wells[i + 1]
                assert end_i < start_next, f"Wells {i} and {i+1} overlap"

    @pytest.mark.unit
    def test_find_wells_ordering(self):
        """Test that wells are returned in order of increasing zeta."""
        zeta = np.linspace(0, 6 * np.pi, 3000)
        B = 2.5 + 0.5 * np.sin(zeta)
        B_bounce = 2.7

        wells = find_wells(zeta, B, B_bounce)

        if len(wells) > 1:
            for i in range(len(wells) - 1):
                assert wells[i][0] < wells[i + 1][0], "Wells should be ordered by zeta"

    @pytest.mark.unit
    def test_find_wells_empty_array(self):
        """Test behavior with empty or minimal arrays."""
        zeta = np.array([0.0])
        B = np.array([2.5])
        B_bounce = 2.0

        wells = find_wells(zeta, B, B_bounce)
        assert isinstance(wells, list)

    @pytest.mark.unit
    def test_find_wells_B_bounce_at_B_min(self, simple_sinusoidal_B):
        """Test when B_bounce equals B_min (edge case)."""
        zeta, B = simple_sinusoidal_B
        B_bounce = B.min()

        wells = find_wells(zeta, B, B_bounce)

        assert isinstance(wells, list)

    @pytest.mark.unit
    def test_find_wells_B_bounce_at_B_max(self, simple_sinusoidal_B):
        """Test when B_bounce equals B_max (edge case)."""
        zeta, B = simple_sinusoidal_B
        B_bounce = B.max()

        wells = find_wells(zeta, B, B_bounce)

        assert isinstance(wells, list)
