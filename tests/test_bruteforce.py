"""Tests for frankenz.bruteforce module."""

import numpy as np
import pytest

from frankenz.bruteforce import BruteForce
from frankenz.pdf import PDFDict


@pytest.mark.slow
class TestBruteForce:
    """Integration tests for BruteForce fit/predict pipeline."""

    def test_fit_stores_results(self, mock_survey, train_test_split, zgrid):
        """BruteForce.fit() should store log-posteriors."""
        train, test = train_test_split
        n_test = 5  # small subset for speed

        phot_train = train['phot']
        phot_err_train = train['phot_err']
        mask_train = np.isfinite(phot_train).astype(float)

        phot_test = test['phot'][:n_test]
        phot_err_test = test['phot_err'][:n_test]
        mask_test = np.isfinite(phot_test).astype(float)

        bf = BruteForce(phot_train, phot_err_train, mask_train)
        bf.fit(phot_test, phot_err_test, mask_test, verbose=False)

        assert bf.fit_lnprob is not None
        assert bf.fit_lnprob.shape == (n_test, len(phot_train))
        assert bf.NDATA == n_test

    def test_fit_predict_produces_pdfs(self, mock_survey, train_test_split,
                                       zgrid):
        """BruteForce.fit_predict() should produce normalized PDFs."""
        train, test = train_test_split
        n_test = 5

        phot_train = train['phot']
        phot_err_train = train['phot_err']
        mask_train = np.isfinite(phot_train).astype(float)

        phot_test = test['phot'][:n_test]
        phot_err_test = test['phot_err'][:n_test]
        mask_test = np.isfinite(phot_test).astype(float)

        z_train = train['redshifts']
        z_err_train = np.full_like(z_train, 0.01)

        bf = BruteForce(phot_train, phot_err_train, mask_train)
        pdfs = bf.fit_predict(phot_test, phot_err_test, mask_test,
                              z_train, z_err_train,
                              label_grid=zgrid, verbose=False)

        assert pdfs.shape == (n_test, len(zgrid))
        # PDFs should be normalized (sum to ~1)
        sums = pdfs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.01)
        # PDFs should be non-negative
        assert np.all(pdfs >= 0)

    def test_gof_metrics(self, mock_survey, train_test_split, zgrid):
        """return_gof=True should produce valid GoF metrics."""
        train, test = train_test_split
        n_test = 3

        phot_train = train['phot']
        phot_err_train = train['phot_err']
        mask_train = np.isfinite(phot_train).astype(float)

        phot_test = test['phot'][:n_test]
        phot_err_test = test['phot_err'][:n_test]
        mask_test = np.isfinite(phot_test).astype(float)

        z_train = train['redshifts']
        z_err_train = np.full_like(z_train, 0.01)

        bf = BruteForce(phot_train, phot_err_train, mask_train)
        pdfs, (lmap, levid) = bf.fit_predict(
            phot_test, phot_err_test, mask_test,
            z_train, z_err_train,
            label_grid=zgrid, verbose=False, return_gof=True)

        assert lmap.shape == (n_test,)
        assert levid.shape == (n_test,)
        assert np.all(np.isfinite(levid))
