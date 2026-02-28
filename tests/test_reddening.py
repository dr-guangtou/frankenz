"""Tests for frankenz.reddening module."""

import numpy as np
import pytest

from frankenz.reddening import madau_teff, _madau_tau1, _madau_tau2


class TestMadauTeff:
    """Tests for the Madau IGM effective transmission."""

    def test_no_attenuation_above_lya(self):
        """At wavelengths >> Ly-alpha and z=0, teff should be 1."""
        wave = np.linspace(2000, 10000, 100)
        teff = madau_teff(wave, z=0.0)
        np.testing.assert_allclose(teff, 1.0, atol=1e-10)

    def test_bounded_01(self):
        """Transmission should be between 0 and 1."""
        wave = np.linspace(800, 5000, 500)
        for z in [0.5, 1.0, 2.0, 3.0]:
            teff = madau_teff(wave, z)
            assert np.all(teff >= 0.0)
            assert np.all(teff <= 1.0 + 1e-10)

    def test_higher_z_more_attenuation(self):
        """Higher redshift should produce more attenuation at UV wavelengths."""
        wave = np.linspace(800, 1200, 100)
        teff_low = madau_teff(wave, z=1.0)
        teff_high = madau_teff(wave, z=3.0)
        # Mean transmission should be lower at higher z
        assert np.mean(teff_high) < np.mean(teff_low)

    def test_long_wavelength_no_attenuation(self):
        """Wavelengths >> 1216*(1+z) should have no attenuation."""
        wave = np.array([10000.0, 15000.0, 20000.0])
        teff = madau_teff(wave, z=2.0)
        np.testing.assert_allclose(teff, 1.0, atol=1e-10)


class TestMadauComponents:
    """Tests for individual Madau optical depth components."""

    def test_tau1_nonnegative(self):
        """Line absorption optical depth should be non-negative."""
        wave = np.linspace(800, 2000, 200)
        tau1 = _madau_tau1(wave, z=2.0)
        assert np.all(tau1 >= 0)

    def test_tau2_nonnegative(self):
        """Continuum absorption optical depth should be non-negative."""
        wave = np.linspace(500, 1500, 200)
        tau2 = _madau_tau2(wave, z=2.0)
        assert np.all(tau2 >= 0)
