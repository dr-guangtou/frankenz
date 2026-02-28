"""Tests for frankenz.io — PhotoData container + format readers/writers."""

import numpy as np
import pytest

from frankenz.io import (
    PhotoData, load_data, save_data,
    read_csv, write_csv, read_numpy, write_numpy,
)


# ---------------------------------------------------------------------------
# Fixture: sample PhotoData
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create a small PhotoData for testing."""
    rng = np.random.default_rng(42)
    n_obj, n_band = 20, 5
    return PhotoData(
        flux=rng.uniform(1.0, 100.0, size=(n_obj, n_band)),
        flux_err=rng.uniform(0.1, 5.0, size=(n_obj, n_band)),
        mask=np.ones((n_obj, n_band), dtype=int),
        redshifts=rng.uniform(0.0, 2.0, size=n_obj),
        redshift_errs=rng.uniform(0.001, 0.01, size=n_obj),
        object_ids=np.arange(n_obj),
        band_names=["g", "r", "i", "zband", "y"],
    )


# ---------------------------------------------------------------------------
# PhotoData container
# ---------------------------------------------------------------------------

class TestPhotoData:

    def test_properties(self, sample_data):
        assert sample_data.n_objects == 20
        assert sample_data.n_bands == 5

    def test_empty_properties(self):
        d = PhotoData()
        assert d.n_objects == 0
        assert d.n_bands == 0

    def test_validate_ok(self, sample_data):
        sample_data.validate()

    def test_validate_missing_flux(self):
        d = PhotoData(flux_err=np.ones((5, 3)))
        with pytest.raises(ValueError, match="flux is required"):
            d.validate()

    def test_validate_missing_flux_err(self):
        d = PhotoData(flux=np.ones((5, 3)))
        with pytest.raises(ValueError, match="flux_err is required"):
            d.validate()

    def test_validate_shape_mismatch(self):
        d = PhotoData(
            flux=np.ones((5, 3)),
            flux_err=np.ones((5, 4)),
        )
        with pytest.raises(ValueError, match="shape"):
            d.validate()

    def test_validate_1d_flux(self):
        d = PhotoData(flux=np.ones(5), flux_err=np.ones(5))
        with pytest.raises(ValueError, match="2D"):
            d.validate()

    def test_subset(self, sample_data):
        sub = sample_data.subset([0, 5, 10])
        assert sub.n_objects == 3
        assert sub.n_bands == sample_data.n_bands
        np.testing.assert_array_equal(sub.flux, sample_data.flux[[0, 5, 10]])
        np.testing.assert_array_equal(sub.redshifts,
                                      sample_data.redshifts[[0, 5, 10]])
        assert sub.band_names == sample_data.band_names

    def test_subset_without_optional(self):
        d = PhotoData(
            flux=np.ones((10, 3)),
            flux_err=np.ones((10, 3)),
        )
        sub = d.subset([0, 1])
        assert sub.n_objects == 2
        assert sub.redshifts is None


# ---------------------------------------------------------------------------
# CSV roundtrip
# ---------------------------------------------------------------------------

class TestCSV:

    def test_roundtrip(self, sample_data, tmp_path):
        path = tmp_path / "test.csv"
        write_csv(sample_data, path)
        column_map = {
            "flux_columns": sample_data.band_names,
            "flux_err_columns": [f"{b}_err" for b in sample_data.band_names],
            "redshift_column": "z",
            "redshift_err_column": "zerr",
            "object_id_column": "object_id",
        }
        loaded = read_csv(path, column_map=column_map)
        np.testing.assert_allclose(loaded.flux, sample_data.flux)
        np.testing.assert_allclose(loaded.redshifts, sample_data.redshifts)

    def test_missing_column_map_raises(self, tmp_path):
        path = tmp_path / "test.csv"
        import pandas as pd
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="column_map"):
            read_csv(path)


# ---------------------------------------------------------------------------
# NumPy roundtrip
# ---------------------------------------------------------------------------

class TestNumPy:

    def test_roundtrip(self, sample_data, tmp_path):
        path = tmp_path / "test.npz"
        write_numpy(sample_data, path)
        loaded = read_numpy(path)
        np.testing.assert_allclose(loaded.flux, sample_data.flux)
        np.testing.assert_allclose(loaded.flux_err, sample_data.flux_err)
        np.testing.assert_allclose(loaded.redshifts, sample_data.redshifts)
        assert loaded.band_names == sample_data.band_names

    def test_minimal_data(self, tmp_path):
        """Only flux and flux_err, no optional fields."""
        path = tmp_path / "minimal.npz"
        d = PhotoData(
            flux=np.ones((5, 3)),
            flux_err=np.ones((5, 3)) * 0.1,
        )
        write_numpy(d, path)
        loaded = read_numpy(path)
        assert loaded.n_objects == 5
        assert loaded.redshifts is None


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------

class TestLoadSave:

    def test_auto_detect_csv(self, sample_data, tmp_path):
        path = tmp_path / "test.csv"
        save_data(sample_data, path)
        column_map = {
            "flux_columns": sample_data.band_names,
            "flux_err_columns": [f"{b}_err" for b in sample_data.band_names],
        }
        loaded = load_data(path, column_map=column_map)
        np.testing.assert_allclose(loaded.flux, sample_data.flux)

    def test_auto_detect_npz(self, sample_data, tmp_path):
        path = tmp_path / "test.npz"
        save_data(sample_data, path)
        loaded = load_data(path)
        np.testing.assert_allclose(loaded.flux, sample_data.flux)

    def test_unknown_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot detect format"):
            load_data(tmp_path / "test.xyz")

    def test_explicit_format(self, sample_data, tmp_path):
        path = tmp_path / "data_file"  # no extension
        write_numpy(sample_data, path)
        loaded = load_data(path, format="numpy")
        np.testing.assert_allclose(loaded.flux, sample_data.flux)


# ---------------------------------------------------------------------------
# FITS (skip if astropy not available)
# ---------------------------------------------------------------------------

class TestFITS:

    @pytest.fixture(autouse=True)
    def _skip_if_no_astropy(self):
        pytest.importorskip("astropy")

    def test_roundtrip(self, sample_data, tmp_path):
        from frankenz.io import read_fits, write_fits
        path = tmp_path / "test.fits"
        write_fits(sample_data, path)
        column_map = {
            "flux_columns": sample_data.band_names,
            "flux_err_columns": [f"{b}_err" for b in sample_data.band_names],
            "redshift_column": "z",
            "object_id_column": "object_id",
        }
        loaded = read_fits(path, column_map=column_map)
        np.testing.assert_allclose(loaded.flux, sample_data.flux, rtol=1e-6)
        np.testing.assert_allclose(loaded.redshifts, sample_data.redshifts,
                                   rtol=1e-6)


# ---------------------------------------------------------------------------
# HDF5 (skip if h5py not available)
# ---------------------------------------------------------------------------

class TestHDF5:

    @pytest.fixture(autouse=True)
    def _skip_if_no_h5py(self):
        pytest.importorskip("h5py")

    def test_roundtrip(self, sample_data, tmp_path):
        from frankenz.io import read_hdf5, write_hdf5
        path = tmp_path / "test.hdf5"
        write_hdf5(sample_data, path)
        loaded = read_hdf5(path)
        np.testing.assert_allclose(loaded.flux, sample_data.flux)
        np.testing.assert_allclose(loaded.redshifts, sample_data.redshifts)
        assert loaded.band_names == sample_data.band_names
