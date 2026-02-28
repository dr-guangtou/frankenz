"""
Session-scoped fixtures for frankenz test suite.

Generates a deterministic mock survey dataset once per pytest session
using MockSurvey with seed=42. Provides train/test splits and a
redshift grid for integration tests.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def mock_survey():
    """Generate a deterministic MockSurvey with SDSS filters + CWW+ templates."""
    from frankenz.simulate import MockSurvey

    rstate = np.random.RandomState(42)
    ms = MockSurvey(survey='sdss', templates='cww+', prior='bpz', rstate=rstate)
    ms.make_mock(250, rstate=rstate, verbose=False)
    return ms


@pytest.fixture(scope="session")
def train_test_split(mock_survey):
    """Split mock survey data into 200 train / 50 test objects."""
    data = mock_survey.data
    n_train = 200

    train = {
        'phot': data['phot_obs'][:n_train],
        'phot_err': data['phot_err'][:n_train],
        'redshifts': data['redshifts'][:n_train],
        'templates': data['templates'][:n_train],
    }
    test = {
        'phot': data['phot_obs'][n_train:],
        'phot_err': data['phot_err'][n_train:],
        'redshifts': data['redshifts'][n_train:],
        'templates': data['templates'][n_train:],
    }
    return train, test


@pytest.fixture(scope="session")
def zgrid():
    """Standard redshift grid for PDF evaluation."""
    return np.linspace(0, 3, 300)
