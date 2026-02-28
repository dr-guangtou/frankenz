"""End-to-end pipeline test: config -> data -> fitter -> PDFs."""

import numpy as np
import pytest

from frankenz.config import FrankenzConfig, TransformConfig, ModelConfig, ZGridConfig
from frankenz.io import PhotoData, write_numpy, read_numpy
from frankenz.batch import run_pipeline
from frankenz.transforms import get_transform
from frankenz.fitting import get_fitter
from frankenz.priors import get_prior


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n_obj, n_band = 100, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
        redshifts=rng.uniform(0.0, 2.0, size=n_obj),
        redshift_errs=np.full(n_obj, 0.01),
        band_names=["g", "r", "i", "zband", "y"],
    )


@pytest.fixture
def test_data():
    rng = np.random.default_rng(123)
    n_obj, n_band = 10, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
        band_names=["g", "r", "i", "zband", "y"],
    )


@pytest.mark.slow
class TestEndToEnd:

    def test_config_to_pdfs(self, training_data, test_data):
        """Full pipeline: config -> fitter -> fit_predict -> PDFs."""
        cfg = FrankenzConfig(
            model=ModelConfig(backend="knn", k_tree=3, k_point=5),
            transform=TransformConfig(type="luptitude"),
            zgrid=ZGridConfig(z_start=0.0, z_end=2.0, z_delta=0.1),
            seed=42,
            verbose=False,
        )
        result = run_pipeline(cfg, training_data, test_data)

        assert result.pdfs.shape == (10, len(result.zgrid))
        assert result.zgrid[0] == 0.0
        assert result.zgrid[-1] <= 2.01

    def test_yaml_roundtrip_pipeline(self, training_data, test_data, tmp_path):
        """Config -> YAML -> load -> run pipeline."""
        cfg = FrankenzConfig(
            model=ModelConfig(backend="knn", k_tree=3, k_point=5),
            transform=TransformConfig(type="luptitude"),
            zgrid=ZGridConfig(z_start=0.0, z_end=2.0, z_delta=0.1),
            seed=42,
            verbose=False,
        )
        yaml_path = tmp_path / "config.yaml"
        cfg.to_yaml(yaml_path)
        cfg2 = FrankenzConfig.from_yaml(yaml_path)

        result = run_pipeline(cfg2, training_data, test_data)
        assert result.pdfs.shape[0] == 10

    def test_data_io_roundtrip_pipeline(self, training_data, test_data,
                                        tmp_path):
        """Save data -> load -> run pipeline."""
        train_path = tmp_path / "train.npz"
        test_path = tmp_path / "test.npz"
        write_numpy(training_data, train_path)
        write_numpy(test_data, test_path)

        train_loaded = read_numpy(train_path)
        test_loaded = read_numpy(test_path)

        cfg = FrankenzConfig(
            model=ModelConfig(backend="knn", k_tree=3, k_point=5),
            transform=TransformConfig(type="luptitude"),
            zgrid=ZGridConfig(z_start=0.0, z_end=2.0, z_delta=0.1),
            seed=42,
            verbose=False,
        )
        result = run_pipeline(cfg, train_loaded, test_loaded)
        assert result.pdfs.shape[0] == test_data.n_objects

    def test_bruteforce_pipeline(self, training_data, test_data):
        """BruteForce backend end-to-end."""
        cfg = FrankenzConfig(
            model=ModelConfig(backend="bruteforce"),
            zgrid=ZGridConfig(z_start=0.0, z_end=2.0, z_delta=0.1),
            seed=42,
            verbose=False,
        )
        result = run_pipeline(cfg, training_data, test_data)
        assert result.pdfs.shape[0] == 10
