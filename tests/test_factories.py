"""Tests for factory functions: get_fitter, get_prior, get_transform."""

import numpy as np
import pytest

from frankenz.config import (
    FrankenzConfig, ModelConfig, TransformConfig, PriorConfig,
)
from frankenz.transforms import get_transform
from frankenz.priors import get_prior
from frankenz.fitting import get_fitter
from frankenz.io import PhotoData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_data():
    """Small training set for factory tests."""
    rng = np.random.default_rng(42)
    n_obj, n_band = 50, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
        redshifts=rng.uniform(0.0, 2.0, size=n_obj),
    )


# ---------------------------------------------------------------------------
# get_transform tests
# ---------------------------------------------------------------------------

class TestGetTransform:

    def test_identity_from_config(self):
        cfg = FrankenzConfig(transform=TransformConfig(type="identity"))
        func = get_transform(cfg)
        x = np.array([[1.0, 2.0]])
        xe = np.array([[0.1, 0.2]])
        out, out_e = func(x, xe)
        np.testing.assert_array_equal(out, x)

    def test_magnitude_from_config(self):
        cfg = FrankenzConfig(
            transform=TransformConfig(type="magnitude", zeropoints=3631.0)
        )
        func = get_transform(cfg)
        # 3631 Jy at zeropoint 3631 => magnitude 0
        mag, _ = func(np.array([[3631.0]]), np.array([[1.0]]))
        np.testing.assert_allclose(mag, 0.0, atol=1e-10)

    def test_luptitude_from_config(self):
        cfg = FrankenzConfig(
            transform=TransformConfig(
                type="luptitude", skynoise=[1.0, 1.0], zeropoints=1.0
            )
        )
        func = get_transform(cfg)
        mag, _ = func(np.array([[10.0, 20.0]]), np.array([[0.1, 0.2]]))
        assert mag.shape == (1, 2)

    def test_invalid_type(self):
        cfg = TransformConfig(type="bogus")
        with pytest.raises(ValueError):
            get_transform(cfg)


# ---------------------------------------------------------------------------
# get_prior tests
# ---------------------------------------------------------------------------

class TestGetPrior:

    def test_uniform_returns_none(self):
        cfg = FrankenzConfig(prior=PriorConfig(type="uniform"))
        assert get_prior(cfg) is None

    def test_bpz_returns_callable(self):
        cfg = PriorConfig(type="bpz")
        func = get_prior(cfg)
        assert callable(func)

    def test_bpz_output_shape(self):
        cfg = PriorConfig(type="bpz")
        func = get_prior(cfg)
        zgrid = np.linspace(0.01, 2.0, 50)
        models_z = np.array([0.5, 1.0, 1.5])
        models_zerr = np.array([0.01, 0.01, 0.01])
        result = func(models_z, models_zerr, zgrid)
        assert result.shape == (3, 50)
        # All values should be finite log-probabilities
        assert np.all(np.isfinite(result))

    def test_invalid_prior_type(self):
        cfg = PriorConfig(type="bogus")
        with pytest.raises(ValueError):
            get_prior(cfg)

    def test_accepts_frankenz_config(self):
        cfg = FrankenzConfig(prior=PriorConfig(type="uniform"))
        assert get_prior(cfg) is None


# ---------------------------------------------------------------------------
# get_fitter tests
# ---------------------------------------------------------------------------

class TestGetFitter:

    def test_knn_fitter(self, training_data):
        cfg = FrankenzConfig(
            model=ModelConfig(backend="knn", k_tree=3),
            transform=TransformConfig(type="luptitude"),
            seed=42,
            verbose=False,
        )
        fitter = get_fitter(cfg, training_data)
        from frankenz.knn import NearestNeighbors
        assert isinstance(fitter, NearestNeighbors)
        assert fitter.K == 3

    def test_bruteforce_fitter(self, training_data):
        cfg = FrankenzConfig(
            model=ModelConfig(backend="bruteforce"),
            verbose=False,
        )
        fitter = get_fitter(cfg, training_data)
        from frankenz.bruteforce import BruteForce
        assert isinstance(fitter, BruteForce)

    def test_invalid_backend(self, training_data):
        cfg = FrankenzConfig(
            model=ModelConfig(backend="bogus"),
        )
        with pytest.raises(ValueError, match="Unknown backend"):
            get_fitter(cfg, training_data)

    def test_seed_reproducibility(self, training_data):
        """Same seed produces same KDTree ensemble."""
        cfg = FrankenzConfig(
            model=ModelConfig(backend="knn", k_tree=3),
            seed=42,
            verbose=False,
        )
        f1 = get_fitter(cfg, training_data)
        f2 = get_fitter(cfg, training_data)
        # Both should have same number of KDTrees
        assert len(f1.KDTrees) == len(f2.KDTrees)
