"""Tests for frankenz.priors module."""

import numpy as np
import pytest

from frankenz.priors import pmag, bpz_pt_m, bpz_pz_tm, _bpz_prior


class TestPmag:
    """Tests for P(mag) function."""

    def test_positive_everywhere(self):
        """P(mag) should be non-negative."""
        mags = np.linspace(15, 27, 100)
        pm = pmag(mags, maglim=24.0)
        assert np.all(pm >= 0)

    def test_faint_end_dominated(self):
        """P(mag) should increase toward faint magnitudes (galaxy counts)."""
        mags = np.linspace(10, 28, 1000)
        pm = pmag(mags, maglim=24.0)
        # Faint-end probability should be higher than bright-end
        bright = np.interp(18.0, mags, pm)
        faint = np.interp(26.0, mags, pm)
        assert faint > bright

    def test_normalizes(self):
        """P(mag) integral should be approximately 1."""
        mags = np.linspace(10, 28, 10000)
        pm = pmag(mags, maglim=24.0)
        integral = np.trapz(pm, mags)
        assert integral == pytest.approx(1.0, abs=0.02)


class TestBpzPtm:
    """Tests for BPZ P(t | m) prior."""

    def test_three_types_sum_to_one(self):
        """P(t | m) summed over all 3 types should be ~1."""
        m = 22.0
        probs = [bpz_pt_m(t, m) for t in range(3)]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_invalid_type_raises(self):
        """Types outside [0, 2] should raise ValueError."""
        with pytest.raises(ValueError):
            bpz_pt_m(3, 22.0)
        with pytest.raises(ValueError):
            bpz_pt_m(-1, 22.0)

    def test_fractions_at_m20(self):
        """At m=20, fractions should be ~35% E/S0, ~50% Spiral, ~15% Irr."""
        p0 = bpz_pt_m(0, 20.0)  # E/S0
        p1 = bpz_pt_m(1, 20.0)  # Spiral
        p2 = bpz_pt_m(2, 20.0)  # Irr
        assert p0 == pytest.approx(0.35, abs=0.05)
        assert p1 == pytest.approx(0.50, abs=0.05)
        assert p2 == pytest.approx(0.15, abs=0.05)


class TestBpzPztm:
    """Tests for BPZ P(z | t, m) prior."""

    def test_nonnegative(self):
        """P(z | t, m) should be non-negative."""
        zgrid = np.linspace(0.01, 5, 100)
        for t in range(3):
            probs = np.array([bpz_pz_tm(z, t, 22.0) for z in zgrid])
            assert np.all(probs >= 0)

    def test_invalid_type_raises(self):
        """Types outside [0, 2] should raise ValueError."""
        with pytest.raises(ValueError):
            bpz_pz_tm(1.0, 3, 22.0)


class TestBpzPriorInternal:
    """Tests for _bpz_prior internal function."""

    def test_output_shapes(self):
        """_bpz_prior should return correct shapes."""
        zgrid = np.linspace(0.01, 5, 100)
        p_i, f_t = _bpz_prior(22.0, zgrid)
        assert p_i.shape == (100, 3)
        assert f_t.shape == (3,)

    def test_type_fractions_sum_to_one(self):
        """Type fractions should sum to 1."""
        zgrid = np.linspace(0.01, 5, 100)
        _, f_t = _bpz_prior(22.0, zgrid)
        assert sum(f_t) == pytest.approx(1.0, abs=1e-10)
