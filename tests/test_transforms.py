"""Tests for frankenz.transforms — extracted transforms + factory."""

import numpy as np
import pytest

from frankenz.transforms import (
    identity, magnitude, inv_magnitude, luptitude, inv_luptitude,
    get_transform,
)
from frankenz.config import FrankenzConfig, TransformConfig


class TestIdentity:
    """Identity transform passes data through unchanged."""

    def test_passthrough(self):
        phot = np.array([[1.0, 2.0, 3.0]])
        err = np.array([[0.1, 0.2, 0.3]])
        out_phot, out_err = identity(phot, err)
        np.testing.assert_array_equal(out_phot, phot)
        np.testing.assert_array_equal(out_err, err)


class TestMagnitudeRoundtrip:
    """magnitude <-> inv_magnitude roundtrip."""

    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        phot = rng.uniform(0.1, 100.0, size=(10, 5))
        err = rng.uniform(0.01, 1.0, size=(10, 5))
        mag, mag_err = magnitude(phot, err)
        phot2, err2 = inv_magnitude(mag, mag_err)
        np.testing.assert_allclose(phot2, phot, rtol=1e-10)

    def test_roundtrip_with_zeropoints(self):
        phot = np.array([[100.0, 200.0]])
        err = np.array([[1.0, 2.0]])
        zp = np.array([3631.0, 3631.0])
        mag, mag_err = magnitude(phot, err, zeropoints=zp)
        phot2, _ = inv_magnitude(mag, mag_err, zeropoints=zp)
        np.testing.assert_allclose(phot2, phot, rtol=1e-10)


class TestLuptitudeRoundtrip:
    """luptitude <-> inv_luptitude roundtrip."""

    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        phot = rng.uniform(0.1, 100.0, size=(10, 5))
        err = rng.uniform(0.01, 1.0, size=(10, 5))
        mag, mag_err = luptitude(phot, err)
        phot2, err2 = inv_luptitude(mag, mag_err)
        np.testing.assert_allclose(phot2, phot, rtol=1e-10)

    def test_handles_negative_flux(self):
        """Luptitudes should work for negative/zero flux (unlike magnitudes)."""
        phot = np.array([[-1.0, 0.0, 1.0]])
        err = np.array([[0.1, 0.1, 0.1]])
        mag, mag_err = luptitude(phot, err)
        assert np.all(np.isfinite(mag))
        phot2, _ = inv_luptitude(mag, mag_err)
        np.testing.assert_allclose(phot2, phot, rtol=1e-10)


class TestGetTransform:
    """Factory function get_transform."""

    def test_identity(self):
        cfg = TransformConfig(type="identity")
        func = get_transform(cfg)
        phot = np.array([[1.0, 2.0]])
        err = np.array([[0.1, 0.2]])
        out, out_err = func(phot, err)
        np.testing.assert_array_equal(out, phot)

    def test_magnitude(self):
        cfg = TransformConfig(type="magnitude", zeropoints=3631.0)
        func = get_transform(cfg)
        phot = np.array([[3631.0]])
        err = np.array([[1.0]])
        mag, _ = func(phot, err)
        np.testing.assert_allclose(mag, 0.0, atol=1e-10)

    def test_luptitude(self):
        cfg = TransformConfig(type="luptitude", skynoise=[1.0])
        func = get_transform(cfg)
        phot = np.array([[10.0]])
        err = np.array([[0.1]])
        mag, mag_err = func(phot, err)
        assert np.all(np.isfinite(mag))

    def test_accepts_frankenz_config(self):
        cfg = FrankenzConfig(transform=TransformConfig(type="identity"))
        func = get_transform(cfg)
        out, _ = func(np.array([[1.0]]), np.array([[0.1]]))
        np.testing.assert_array_equal(out, [[1.0]])

    def test_invalid_type_raises(self):
        cfg = TransformConfig(type="bogus")
        with pytest.raises(ValueError, match="Unknown transform type"):
            get_transform(cfg)


class TestBackwardCompat:
    """Transforms are still importable from pdf.py."""

    def test_import_from_pdf(self):
        from frankenz.pdf import magnitude as mag_from_pdf
        from frankenz.transforms import magnitude as mag_from_transforms
        assert mag_from_pdf is mag_from_transforms

    def test_import_inv_from_pdf(self):
        from frankenz.pdf import inv_luptitude as inv_from_pdf
        from frankenz.transforms import inv_luptitude as inv_from_transforms
        assert inv_from_pdf is inv_from_transforms
