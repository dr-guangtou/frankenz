"""Tests for likelihood math functions in frankenz.pdf."""

import numpy as np
import pytest

from frankenz.pdf import _loglike, _loglike_s, loglike, gaussian, gaussian_bin


class TestLoglike:
    """Tests for _loglike (fixed-scale likelihood)."""

    def test_perfect_match_gives_best_chi2(self):
        """When data == model, chi2 should be 0."""
        data = np.array([1.0, 2.0, 3.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.ones(3)
        models = np.array([[1.0, 2.0, 3.0]])
        models_err = np.zeros((1, 3))
        models_mask = np.ones((1, 3))

        lnl, ndim, chi2 = _loglike(data, data_err, data_mask,
                                    models, models_err, models_mask,
                                    ignore_model_err=True)
        assert chi2[0] == pytest.approx(0.0, abs=1e-10)
        assert ndim[0] == 3

    def test_worse_match_gives_lower_loglike(self):
        """A worse match should have lower log-likelihood."""
        data = np.array([1.0, 2.0, 3.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.ones(3)
        models = np.array([[1.01, 2.01, 3.01],  # near-perfect match
                           [2.0, 3.0, 4.0]])     # offset by 1
        models_err = np.zeros((2, 3))
        models_mask = np.ones((2, 3))

        lnl, _, _ = _loglike(data, data_err, data_mask,
                             models, models_err, models_mask,
                             ignore_model_err=True)
        assert lnl[0] > lnl[1]

    def test_mask_excludes_filters(self):
        """Masked filters should be excluded from chi2."""
        data = np.array([1.0, 2.0, 999.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.array([1.0, 1.0, 0.0])  # mask out third filter
        models = np.array([[1.0, 2.0, 0.0]])
        models_err = np.zeros((1, 3))
        models_mask = np.ones((1, 3))

        lnl, ndim, chi2 = _loglike(data, data_err, data_mask,
                                    models, models_err, models_mask,
                                    ignore_model_err=True)
        assert ndim[0] == 2
        assert chi2[0] == pytest.approx(0.0, abs=1e-10)

    def test_model_errors_increase_variance(self):
        """Including model errors should generally increase tolerance."""
        data = np.array([1.0, 2.0, 3.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.ones(3)
        models = np.array([[1.05, 2.05, 3.05]])
        models_err = np.array([[0.1, 0.1, 0.1]])
        models_mask = np.ones((1, 3))

        _, _, chi2_noerr = _loglike(data, data_err, data_mask,
                                    models, models_err, models_mask,
                                    ignore_model_err=True)
        _, _, chi2_err = _loglike(data, data_err, data_mask,
                                  models, models_err, models_mask,
                                  ignore_model_err=False)
        assert chi2_err[0] < chi2_noerr[0]


class TestLoglikeS:
    """Tests for _loglike_s (free-scale likelihood)."""

    def test_scale_recovery(self):
        """Should recover the correct scale factor."""
        data = np.array([2.0, 4.0, 6.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.ones(3)
        models = np.array([[1.0, 2.0, 3.0]])
        models_err = np.zeros((1, 3))
        models_mask = np.ones((1, 3))

        _, _, _, scale, _ = _loglike_s(data, data_err, data_mask,
                                       models, models_err, models_mask,
                                       ignore_model_err=True,
                                       return_scale=True)
        assert scale[0] == pytest.approx(2.0, rel=1e-5)


class TestLoglikeWrapper:
    """Tests for the loglike() wrapper function."""

    def test_handles_nan_data(self):
        """loglike should handle NaN values gracefully."""
        data = np.array([1.0, np.nan, 3.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.array([True, True, True])
        models = np.array([[1.0, 2.0, 3.0]])
        models_err = np.array([[0.05, 0.05, 0.05]])
        models_mask = np.ones((1, 3))

        lnl, ndim, chi2 = loglike(data, data_err, data_mask,
                                   models, models_err, models_mask)
        assert np.all(np.isfinite(lnl))

    def test_free_scale_flag(self):
        """free_scale=True should return scale factor."""
        data = np.array([2.0, 4.0, 6.0])
        data_err = np.array([0.1, 0.1, 0.1])
        data_mask = np.ones(3)
        models = np.array([[1.0, 2.0, 3.0]])
        models_err = np.zeros((1, 3))
        models_mask = np.ones((1, 3))

        result = loglike(data, data_err, data_mask,
                         models, models_err, models_mask,
                         free_scale=True, return_scale=True,
                         ignore_model_err=True)
        assert len(result) == 5  # lnl, ndim, chi2, scale, scale_err


class TestGaussian:
    """Tests for gaussian and gaussian_bin helper functions."""

    def test_gaussian_integrates_to_one(self):
        """Gaussian PDF should integrate to approximately 1."""
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        pdf = gaussian(0.0, 1.0, x)
        integral = np.sum(pdf) * dx
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_gaussian_peak_at_mu(self):
        """Peak should be at mu."""
        x = np.linspace(-5, 5, 1000)
        pdf = gaussian(1.0, 0.5, x)
        assert x[np.argmax(pdf)] == pytest.approx(1.0, abs=0.02)

    def test_gaussian_bin_sums_correctly(self):
        """Binned Gaussian should sum to approximately 1."""
        bins = np.linspace(-10, 10, 1001)
        pdf = gaussian_bin(0.0, 1.0, bins)
        assert np.sum(pdf) == pytest.approx(1.0, abs=0.01)
