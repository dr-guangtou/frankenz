"""Tests for frankenz.simulate module."""

import numpy as np
import pytest

from frankenz.simulate import mag_err, MockSurvey


class TestMagErr:
    """Tests for the mag_err() function (P01_001 regression)."""

    def test_runs_without_name_error(self):
        """mag_err should not raise NameError for undefined variables."""
        result = mag_err(22.0, 24.0)
        assert np.isfinite(result)

    def test_brighter_has_smaller_error(self):
        """Brighter objects should have smaller magnitude errors."""
        err_bright = mag_err(20.0, 24.0)
        err_faint = mag_err(23.0, 24.0)
        assert err_bright < err_faint

    def test_at_limit_error_reasonable(self):
        """At the detection limit, error should be around 1/sigdet ~ 0.2."""
        err = mag_err(24.0, 24.0, sigdet=5.0)
        assert 0.05 < err < 1.0

    def test_array_input(self):
        """Should handle array input."""
        mags = np.array([20.0, 22.0, 24.0])
        errs = mag_err(mags, 24.0)
        assert errs.shape == (3,)
        assert np.all(np.isfinite(errs))


class TestMockSurvey:
    """Tests for MockSurvey initialization."""

    def test_sdss_survey_loads(self):
        """SDSS survey preset should load 5 filters."""
        ms = MockSurvey(survey='sdss')
        assert ms.NFILTER == 5

    def test_cww_templates_load(self):
        """CWW+ template preset should load templates."""
        ms = MockSurvey(survey='sdss', templates='cww+')
        assert ms.NTEMPLATE > 0

    def test_bpz_prior_loads(self):
        """BPZ prior preset should set pm, ptm, pztm."""
        ms = MockSurvey(survey='sdss', templates='cww+', prior='bpz')
        assert ms.pm is not None
        assert ms.ptm is not None
        assert ms.pztm is not None

    def test_invalid_survey_raises(self):
        """Invalid survey name should raise ValueError."""
        with pytest.raises(ValueError):
            MockSurvey(survey='nonexistent')

    @pytest.mark.slow
    def test_make_mock_shapes(self):
        """make_mock should produce correct array shapes."""
        rstate = np.random.RandomState(42)
        ms = MockSurvey(survey='sdss', templates='cww+', prior='bpz',
                        rstate=rstate)
        ms.make_mock(50, rstate=rstate, verbose=False)
        assert ms.data['phot_obs'].shape == (50, 5)
        assert ms.data['phot_err'].shape == (50, 5)
        assert ms.data['redshifts'].shape == (50,)
        assert ms.data['templates'].shape == (50,)
