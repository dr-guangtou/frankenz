"""Tests for frankenz.knn module."""

import numpy as np
import pytest

from frankenz.knn import NearestNeighbors
from frankenz.pdf import luptitude


class TestFeatureMapValidation:
    """Tests for feature_map validation bug (P01_003 regression)."""

    def test_custom_feature_map_accepted(self):
        """A valid custom feature_map function should be accepted."""
        rstate = np.random.RandomState(42)
        models = rstate.uniform(10, 100, size=(50, 5))
        models_err = rstate.uniform(1, 10, size=(50, 5))
        models_mask = np.ones((50, 5))

        def custom_map(x, xe, *args, **kwargs):
            return np.log(np.abs(x) + 1), xe / (np.abs(x) + 1)

        nn = NearestNeighbors(models, models_err, models_mask,
                              feature_map=custom_map, K=3,
                              rstate=rstate, verbose=False)
        assert nn.feature_map is custom_map

    def test_invalid_feature_map_raises(self):
        """An invalid feature_map should raise ValueError."""
        rstate = np.random.RandomState(42)
        models = rstate.uniform(10, 100, size=(50, 5))
        models_err = rstate.uniform(1, 10, size=(50, 5))
        models_mask = np.ones((50, 5))

        with pytest.raises(ValueError):
            NearestNeighbors(models, models_err, models_mask,
                             feature_map="not_a_valid_option", K=3,
                             rstate=rstate, verbose=False)

    def test_builtin_luptitude(self):
        """Default 'luptitude' feature_map should work."""
        rstate = np.random.RandomState(42)
        models = rstate.uniform(10, 100, size=(50, 5))
        models_err = rstate.uniform(1, 10, size=(50, 5))
        models_mask = np.ones((50, 5))

        nn = NearestNeighbors(models, models_err, models_mask,
                              feature_map='luptitude', K=3,
                              rstate=rstate, verbose=False)
        assert nn.KDTrees is not None
        assert len(nn.KDTrees) == 3


@pytest.mark.slow
class TestNearestNeighborsPipeline:
    """Integration tests for KMCkNN pipeline."""

    def test_fit_predict(self, mock_survey, train_test_split, zgrid):
        """KNN fit_predict should produce valid PDFs."""
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

        rstate = np.random.RandomState(42)
        nn = NearestNeighbors(phot_train, phot_err_train, mask_train,
                              K=5, rstate=rstate, verbose=False)

        pdfs = nn.fit_predict(phot_test, phot_err_test, mask_test,
                              z_train, z_err_train,
                              label_grid=zgrid, k=10,
                              rstate=rstate, verbose=False)

        assert pdfs.shape == (n_test, len(zgrid))
        sums = pdfs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.01)
        assert np.all(pdfs >= 0)

    def test_separate_fit_predict(self, mock_survey, train_test_split, zgrid):
        """Separate fit() then predict() should also work."""
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

        rstate = np.random.RandomState(42)
        nn = NearestNeighbors(phot_train, phot_err_train, mask_train,
                              K=5, rstate=rstate, verbose=False)

        nn.fit(phot_test, phot_err_test, mask_test,
               k=10, rstate=rstate, verbose=False)
        pdfs = nn.predict(z_train, z_err_train,
                          label_grid=zgrid, verbose=False)

        assert pdfs.shape == (n_test, len(zgrid))
        assert np.all(pdfs >= 0)
