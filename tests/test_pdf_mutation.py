"""Tests verifying that mutation bugs in pdf.py are fixed.

These tests ensure that loglike() and pdfs_summarize() do not
modify their input arrays in place (P01_002, P01_005).
"""

import numpy as np
import pytest

from frankenz.pdf import loglike, pdfs_summarize


@pytest.mark.mutation
class TestLoglikeMutation:
    """Verify loglike() does not mutate input arrays (P01_002)."""

    def test_data_not_mutated(self):
        """Input data array should not be modified."""
        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        data_orig = data.copy()
        data_err = np.array([0.1, 0.2, 0.1, 0.1, 0.1])
        data_mask = np.array([True, True, True, True, True])
        models = np.array([[1.1, 2.1, 3.1, 4.1, 5.1]])
        models_err = np.array([[0.05, 0.05, 0.05, 0.05, 0.05]])
        models_mask = np.ones_like(models)

        loglike(data, data_err, data_mask, models, models_err, models_mask)
        np.testing.assert_array_equal(data, data_orig)

    def test_data_err_not_mutated(self):
        """Input data_err array should not be modified."""
        data = np.array([1.0, 2.0, 3.0])
        data_err = np.array([0.1, -0.2, 0.1])  # negative error to trigger clean
        data_err_orig = data_err.copy()
        data_mask = np.array([True, True, True])
        models = np.array([[1.1, 2.1, 3.1]])
        models_err = np.array([[0.05, 0.05, 0.05]])
        models_mask = np.ones_like(models)

        loglike(data, data_err, data_mask, models, models_err, models_mask)
        np.testing.assert_array_equal(data_err, data_err_orig)

    def test_data_mask_not_mutated(self):
        """Input data_mask array should not be modified."""
        data = np.array([1.0, np.inf, 3.0])
        data_err = np.array([0.1, 0.2, 0.1])
        data_mask = np.array([True, True, True])
        data_mask_orig = data_mask.copy()
        models = np.array([[1.1, 2.1, 3.1]])
        models_err = np.array([[0.05, 0.05, 0.05]])
        models_mask = np.ones_like(models)

        loglike(data, data_err, data_mask, models, models_err, models_mask)
        np.testing.assert_array_equal(data_mask, data_mask_orig)


@pytest.mark.mutation
class TestPdfsSummarizeMutation:
    """Verify pdfs_summarize() does not mutate input PDFs (P01_005)."""

    def test_pdfs_not_mutated(self):
        """Input pdfs array should not be modified when renormalize=True."""
        rstate = np.random.RandomState(42)
        pgrid = np.linspace(0, 3, 100)
        pdfs = rstate.dirichlet(np.ones(100), size=10) * 2.0  # not normalized
        pdfs_orig = pdfs.copy()

        pdfs_summarize(pdfs, pgrid, renormalize=True, rstate=rstate)
        np.testing.assert_array_equal(pdfs, pdfs_orig)

    def test_pdfs_not_mutated_when_no_renormalize(self):
        """Input pdfs should not be modified when renormalize=False either."""
        rstate = np.random.RandomState(42)
        pgrid = np.linspace(0, 3, 100)
        pdfs = rstate.dirichlet(np.ones(100), size=10)
        pdfs_orig = pdfs.copy()

        pdfs_summarize(pdfs, pgrid, renormalize=False, rstate=rstate)
        np.testing.assert_array_equal(pdfs, pdfs_orig)
