"""Tests for PDF utility functions in frankenz.pdf."""

import numpy as np
import pytest

from frankenz.pdf import (magnitude, inv_magnitude, luptitude, inv_luptitude,
                           PDFDict, pdfs_resample, pdfs_summarize, gauss_kde)


class TestMagnitudeRoundtrip:
    """Tests for magnitude/luptitude conversion roundtrips."""

    def test_magnitude_roundtrip(self):
        """phot -> mag -> phot should be identity."""
        phot = np.array([[100.0, 200.0, 50.0]])
        err = np.array([[10.0, 20.0, 5.0]])
        mag, mag_err = magnitude(phot, err)
        phot2, err2 = inv_magnitude(mag, mag_err)
        np.testing.assert_allclose(phot, phot2, rtol=1e-10)
        np.testing.assert_allclose(err, err2, rtol=1e-10)

    def test_luptitude_roundtrip(self):
        """phot -> lupt -> phot should be identity."""
        phot = np.array([[100.0, 200.0, 50.0]])
        err = np.array([[10.0, 20.0, 5.0]])
        sky = np.array([1.0, 1.0, 1.0])
        lupt, lupt_err = luptitude(phot, err, skynoise=sky)
        phot2, err2 = inv_luptitude(lupt, lupt_err, skynoise=sky)
        np.testing.assert_allclose(phot, phot2, rtol=1e-6)

    def test_magnitude_brighter_is_smaller(self):
        """Brighter (larger flux) should give smaller magnitude."""
        phot = np.array([[100.0, 1000.0]])
        err = np.array([[10.0, 10.0]])
        mag, _ = magnitude(phot, err)
        assert mag[0, 1] < mag[0, 0]


class TestPDFDict:
    """Tests for the PDFDict kernel dictionary."""

    def test_initialization(self):
        """PDFDict should initialize with correct sizes."""
        pdf_grid = np.linspace(0, 3, 300)
        sigma_grid = np.linspace(0.01, 0.5, 50)
        pdict = PDFDict(pdf_grid, sigma_grid)
        assert pdict.Ngrid == 300
        assert pdict.Ndict == 50

    def test_fit_returns_indices(self):
        """fit() should return valid indices."""
        pdf_grid = np.linspace(0, 3, 300)
        sigma_grid = np.linspace(0.01, 0.5, 50)
        pdict = PDFDict(pdf_grid, sigma_grid)
        X = np.array([0.5, 1.0, 2.0])
        Xe = np.array([0.1, 0.2, 0.3])
        x_idx, xe_idx = pdict.fit(X, Xe)
        assert len(x_idx) == 3
        assert len(xe_idx) == 3
        assert np.all(xe_idx >= 0)
        assert np.all(xe_idx < 50)


class TestPdfsResample:
    """Tests for PDF resampling."""

    def test_preserves_normalization(self):
        """Resampled PDFs should still sum to ~1."""
        old_grid = np.linspace(0, 3, 100)
        new_grid = np.linspace(0, 3, 200)
        pdfs = np.zeros((5, 100))
        pdfs[:, 50] = 1.0  # delta functions
        resampled = pdfs_resample(pdfs, old_grid, new_grid)
        sums = resampled.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=0.1)


class TestPdfsSummarize:
    """Tests for PDF summary statistics."""

    def test_output_structure(self):
        """pdfs_summarize should return correct structure."""
        rstate = np.random.RandomState(42)
        pgrid = np.linspace(0, 3, 100)
        # Create peaked PDFs near z=1
        pdfs = np.exp(-0.5 * ((pgrid - 1.0) / 0.1) ** 2)
        pdfs = np.tile(pdfs, (5, 1))
        pdfs = pdfs / pdfs.sum(axis=1, keepdims=True)

        result = pdfs_summarize(pdfs, pgrid, rstate=rstate)
        # Should return: (mean_tuple, med_tuple, mode_tuple, best_tuple,
        #                  quantile_tuple, pmc)
        assert len(result) == 6
        mean_tuple = result[0]
        assert len(mean_tuple) == 4  # (val, std, conf, risk)
        assert mean_tuple[0].shape == (5,)

    def test_mean_near_peak(self):
        """Mean should be near the peak for symmetric PDFs."""
        rstate = np.random.RandomState(42)
        pgrid = np.linspace(0, 3, 300)
        pdfs = np.exp(-0.5 * ((pgrid - 1.5) / 0.1) ** 2)
        pdfs = pdfs.reshape(1, -1)
        pdfs = pdfs / pdfs.sum(axis=1, keepdims=True)

        result = pdfs_summarize(pdfs, pgrid, rstate=rstate)
        pmean = result[0][0][0]
        assert pmean == pytest.approx(1.5, abs=0.05)

    def test_mode_at_peak(self):
        """Mode should be at the grid point with highest PDF value."""
        rstate = np.random.RandomState(42)
        pgrid = np.linspace(0, 3, 300)
        pdfs = np.exp(-0.5 * ((pgrid - 2.0) / 0.05) ** 2)
        pdfs = pdfs.reshape(1, -1)
        pdfs = pdfs / pdfs.sum(axis=1, keepdims=True)

        result = pdfs_summarize(pdfs, pgrid, rstate=rstate)
        pmode = result[2][0][0]
        assert pmode == pytest.approx(2.0, abs=0.02)


class TestGaussKDE:
    """Tests for gauss_kde function."""

    def test_single_point_produces_gaussian(self):
        """KDE with one point should produce a Gaussian-like PDF."""
        x = np.linspace(0, 3, 300)
        y = np.array([1.5])
        y_std = np.array([0.1])
        pdf = gauss_kde(y, y_std, x)
        assert pdf.sum() > 0
        assert x[np.argmax(pdf)] == pytest.approx(1.5, abs=0.02)

    def test_pdf_nonnegative(self):
        """KDE output should be non-negative."""
        x = np.linspace(0, 3, 300)
        y = np.array([0.5, 1.0, 1.5, 2.0])
        y_std = np.array([0.1, 0.2, 0.1, 0.15])
        pdf = gauss_kde(y, y_std, x)
        assert np.all(pdf >= 0)
