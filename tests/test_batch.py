"""Tests for frankenz.batch — chunked pipeline runner."""

import numpy as np
import pytest

from frankenz.config import FrankenzConfig, ModelConfig, TransformConfig, ZGridConfig
from frankenz.batch import run_pipeline, PipelineResult
from frankenz.io import PhotoData


@pytest.fixture
def training_data():
    """Small training set."""
    rng = np.random.default_rng(42)
    n_obj, n_band = 100, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
        redshifts=rng.uniform(0.0, 2.0, size=n_obj),
        redshift_errs=np.full(n_obj, 0.01),
    )


@pytest.fixture
def test_data():
    """Small test set."""
    rng = np.random.default_rng(123)
    n_obj, n_band = 25, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
    )


@pytest.fixture
def config():
    """Minimal config for fast testing."""
    return FrankenzConfig(
        model=ModelConfig(backend="knn", k_tree=3, k_point=5),
        transform=TransformConfig(type="luptitude"),
        zgrid=ZGridConfig(z_start=0.0, z_end=2.0, z_delta=0.1),
        seed=42,
        verbose=False,
    )


class TestRunPipeline:

    @pytest.mark.slow
    def test_basic_pipeline(self, config, training_data, test_data):
        result = run_pipeline(config, training_data, test_data, chunk_size=10)
        assert isinstance(result, PipelineResult)
        assert result.pdfs.shape[0] == test_data.n_objects
        assert result.pdfs.shape[1] == len(result.zgrid)
        assert result.config is config

    @pytest.mark.slow
    def test_single_chunk(self, config, training_data, test_data):
        """All data in one chunk."""
        result = run_pipeline(config, training_data, test_data,
                              chunk_size=10000)
        assert result.pdfs.shape[0] == test_data.n_objects

    @pytest.mark.slow
    def test_chunk_consistency(self, config, training_data, test_data):
        """Chunked and non-chunked produce same PDFs."""
        result_1 = run_pipeline(config, training_data, test_data,
                                chunk_size=5)
        result_2 = run_pipeline(config, training_data, test_data,
                                chunk_size=10000)
        np.testing.assert_allclose(result_1.pdfs, result_2.pdfs, rtol=1e-10)

    @pytest.mark.slow
    def test_zgrid_from_config(self, config, training_data, test_data):
        result = run_pipeline(config, training_data, test_data)
        expected = np.arange(0.0, 2.0 + 0.05, 0.1)
        np.testing.assert_allclose(result.zgrid, expected, atol=1e-10)


class TestPipelineResult:

    def test_dataclass_fields(self):
        r = PipelineResult()
        assert r.pdfs is None
        assert r.zgrid is None
        assert r.summary == {}
        assert r.config is None
