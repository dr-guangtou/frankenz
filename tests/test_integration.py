"""End-to-end integration tests for frankenz pipeline.

These tests exercise full workflows: MockSurvey -> fitting -> PDF -> summary.
"""

import numpy as np
import pytest

from frankenz.bruteforce import BruteForce
from frankenz.pdf import pdfs_summarize


@pytest.mark.slow
class TestFullPipeline:
    """Full pipeline smoke tests."""

    def test_bruteforce_photoz_correlation(self, mock_survey, train_test_split,
                                           zgrid):
        """BruteForce photo-z should correlate with true-z."""
        train, test = train_test_split
        n_test = 20

        phot_train = train['phot']
        phot_err_train = train['phot_err']
        mask_train = np.isfinite(phot_train).astype(float)

        phot_test = test['phot'][:n_test]
        phot_err_test = test['phot_err'][:n_test]
        mask_test = np.isfinite(phot_test).astype(float)

        z_train = train['redshifts']
        z_err_train = np.full_like(z_train, 0.01)
        z_true = test['redshifts'][:n_test]

        bf = BruteForce(phot_train, phot_err_train, mask_train)
        pdfs = bf.fit_predict(phot_test, phot_err_test, mask_test,
                              z_train, z_err_train,
                              label_grid=zgrid, verbose=False)

        # Extract point estimates (mean)
        z_photo = np.dot(pdfs, zgrid)

        # Photo-z should correlate with true-z
        valid = np.isfinite(z_photo) & np.isfinite(z_true) & (z_true > 0)
        if valid.sum() > 5:
            correlation = np.corrcoef(z_true[valid], z_photo[valid])[0, 1]
            assert correlation > 0.3

    def test_pdfs_summarize_pipeline(self, mock_survey, train_test_split,
                                     zgrid):
        """Full pipeline through pdfs_summarize should produce valid stats."""
        train, test = train_test_split
        n_test = 10

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

        rstate = np.random.RandomState(42)
        result = pdfs_summarize(pdfs, zgrid, rstate=rstate)

        # Check structure
        mean_tuple, med_tuple, mode_tuple, best_tuple, quantiles, pmc = result

        # Point estimates should be within zgrid range
        pmean = mean_tuple[0]
        assert np.all(pmean >= zgrid[0])
        assert np.all(pmean <= zgrid[-1])

        # Standard deviations should be positive
        pmean_std = mean_tuple[1]
        assert np.all(pmean_std > 0)

        # Confidence should be between 0 and 1
        pmean_conf = mean_tuple[2]
        assert np.all(pmean_conf >= 0)
        assert np.all(pmean_conf <= 1)

        # Quantiles should be ordered
        plow95, plow68, phigh68, phigh95 = quantiles
        assert np.all(plow95 <= plow68)
        assert np.all(plow68 <= phigh68)
        assert np.all(phigh68 <= phigh95)

        # MC samples should be in range
        assert np.all(pmc >= zgrid[0])
        assert np.all(pmc <= zgrid[-1])

    def test_mock_survey_deterministic(self):
        """MockSurvey with same seed should produce identical results."""
        from frankenz.simulate import MockSurvey

        rstate1 = np.random.RandomState(42)
        ms1 = MockSurvey(survey='sdss', templates='cww+', prior='bpz',
                         rstate=rstate1)
        ms1.make_mock(50, rstate=rstate1, verbose=False)

        rstate2 = np.random.RandomState(42)
        ms2 = MockSurvey(survey='sdss', templates='cww+', prior='bpz',
                         rstate=rstate2)
        ms2.make_mock(50, rstate=rstate2, verbose=False)

        np.testing.assert_array_equal(ms1.data['redshifts'],
                                      ms2.data['redshifts'])
        np.testing.assert_array_equal(ms1.data['phot_obs'],
                                      ms2.data['phot_obs'])
