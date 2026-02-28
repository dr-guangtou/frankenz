"""Tests for frankenz.samplers module."""

import numpy as np
import pytest

from frankenz.samplers import (loglike_nz, population_sampler,
                                hierarchical_sampler)


class TestLoglikeNz:
    """Tests for population log-likelihood."""

    def test_uniform_nz_with_uniform_pdfs(self):
        """Uniform N(z) with uniform PDFs should give finite loglike."""
        Nbins = 50
        nz = np.ones(Nbins) / Nbins
        pdfs = np.ones((10, Nbins)) / Nbins
        lnl = loglike_nz(nz, pdfs)
        assert np.isfinite(lnl)

    def test_negative_nz_gives_neg_inf(self):
        """Negative N(z) values should give -inf loglike."""
        nz = np.array([0.5, -0.1, 0.6])
        pdfs = np.ones((5, 3)) / 3.0
        lnl = loglike_nz(nz, pdfs)
        assert lnl == -np.inf

    def test_return_overlap(self):
        """return_overlap=True should return overlap array."""
        nz = np.ones(20) / 20
        pdfs = np.ones((5, 20)) / 20
        lnl, overlap = loglike_nz(nz, pdfs, return_overlap=True)
        assert overlap.shape == (5,)


class TestHierarchicalSamplerReset:
    """Tests for hierarchical_sampler.reset() (P01_004 regression)."""

    def test_reset_clears_samples(self):
        """reset() should clear self.samples."""
        pdfs = np.ones((10, 20)) / 20
        sampler = hierarchical_sampler(pdfs)
        sampler.samples.append(np.ones(20) / 20)
        sampler.samples_lnp.append(-10.0)

        sampler.reset()
        assert len(sampler.samples) == 0
        assert len(sampler.samples_lnp) == 0

    def test_reset_does_not_create_spurious_attributes(self):
        """reset() should NOT create attributes that don't exist in __init__."""
        pdfs = np.ones((10, 20)) / 20
        sampler = hierarchical_sampler(pdfs)
        sampler.reset()
        assert not hasattr(sampler, 'samples_prior')
        assert not hasattr(sampler, 'samples_counts')


class TestPopulationSampler:
    """Tests for population_sampler."""

    def test_reset_clears_samples(self):
        """reset() should clear samples and samples_lnp."""
        pdfs = np.ones((10, 20)) / 20
        sampler = population_sampler(pdfs)
        sampler.samples.append(np.ones(20) / 20)
        sampler.samples_lnp.append(-10.0)

        sampler.reset()
        assert len(sampler.samples) == 0
        assert len(sampler.samples_lnp) == 0

    def test_results_property(self):
        """results property should return numpy arrays."""
        pdfs = np.ones((10, 20)) / 20
        sampler = population_sampler(pdfs)
        samples, lnp = sampler.results
        assert isinstance(samples, np.ndarray)
        assert isinstance(lnp, np.ndarray)

    @pytest.mark.slow
    def test_short_mcmc_run(self):
        """A short MCMC run should produce valid samples."""
        rstate = np.random.RandomState(42)
        pdfs = rstate.dirichlet(np.ones(20), size=10)
        sampler = population_sampler(pdfs)
        sampler.run_mcmc(5, thin=10, rstate=rstate, verbose=False)
        samples, lnp = sampler.results
        assert samples.shape == (5, 20)
        assert np.all(np.isfinite(lnp))
        # Samples should be valid probability distributions
        np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-10)
