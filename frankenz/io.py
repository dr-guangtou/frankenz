"""
Data I/O for frankenz: PhotoData container + multi-format readers/writers.

Supports CSV (always available), FITS (requires astropy), HDF5 (requires h5py),
and NumPy (.npz) formats. Format auto-detected from file extension.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "PhotoData", "load_data", "save_data",
    "read_csv", "write_csv", "read_numpy", "write_numpy",
    "read_fits", "write_fits", "read_hdf5", "write_hdf5",
]


# ---------------------------------------------------------------------------
# PhotoData container
# ---------------------------------------------------------------------------

@dataclass
class PhotoData:
    """Container for multi-band photometric data.

    Parameters
    ----------
    flux : ndarray of shape (n_objects, n_bands)
        Photometric flux densities.
    flux_err : ndarray of shape (n_objects, n_bands)
        Flux density errors.
    mask : ndarray of shape (n_objects, n_bands)
        Binary mask (1 = valid, 0 = masked).
    redshifts : ndarray of shape (n_objects,) or None
        Spectroscopic redshifts (training data).
    redshift_errs : ndarray of shape (n_objects,) or None
        Redshift errors.
    object_ids : ndarray of shape (n_objects,) or None
        Object identifiers.
    band_names : list of str or None
        Names for each photometric band.
    metadata : dict
        Arbitrary metadata.
    """
    flux: np.ndarray = None
    flux_err: np.ndarray = None
    mask: np.ndarray = None
    redshifts: np.ndarray = None
    redshift_errs: np.ndarray = None
    object_ids: np.ndarray = None
    band_names: list = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_objects(self):
        if self.flux is None:
            return 0
        return self.flux.shape[0]

    @property
    def n_bands(self):
        if self.flux is None:
            return 0
        return self.flux.shape[1]

    def validate(self):
        """Check internal consistency. Raises ValueError on problems."""
        if self.flux is None:
            raise ValueError("flux is required")
        if self.flux_err is None:
            raise ValueError("flux_err is required")
        if self.flux.shape != self.flux_err.shape:
            raise ValueError(
                f"flux shape {self.flux.shape} != flux_err shape "
                f"{self.flux_err.shape}"
            )
        if self.flux.ndim != 2:
            raise ValueError(f"flux must be 2D, got {self.flux.ndim}D")
        if self.mask is not None and self.mask.shape != self.flux.shape:
            raise ValueError(
                f"mask shape {self.mask.shape} != flux shape {self.flux.shape}"
            )
        if self.redshifts is not None:
            if self.redshifts.shape[0] != self.n_objects:
                raise ValueError(
                    f"redshifts length {self.redshifts.shape[0]} != "
                    f"n_objects {self.n_objects}"
                )

    def subset(self, indices):
        """Return a new PhotoData with only the selected objects."""
        indices = np.asarray(indices)
        return PhotoData(
            flux=self.flux[indices],
            flux_err=self.flux_err[indices],
            mask=self.mask[indices] if self.mask is not None else None,
            redshifts=(self.redshifts[indices]
                       if self.redshifts is not None else None),
            redshift_errs=(self.redshift_errs[indices]
                           if self.redshift_errs is not None else None),
            object_ids=(self.object_ids[indices]
                        if self.object_ids is not None else None),
            band_names=self.band_names,
            metadata=self.metadata.copy(),
        )


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------

_EXTENSION_MAP = {
    ".csv": "csv",
    ".fits": "fits",
    ".fit": "fits",
    ".hdf5": "hdf5",
    ".h5": "hdf5",
    ".npz": "numpy",
}


def _detect_format(path):
    """Detect format from file extension."""
    suffix = Path(path).suffix.lower()
    fmt = _EXTENSION_MAP.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Cannot detect format from extension {suffix!r}. "
            f"Supported: {list(_EXTENSION_MAP.keys())}"
        )
    return fmt


def load_data(path, format=None, column_map=None, **kwargs):
    """Load photometric data from file.

    Parameters
    ----------
    path : str or Path
        File path to read.
    format : str or None
        One of 'csv', 'fits', 'hdf5', 'numpy'. Auto-detected from extension
        if None.
    column_map : dict or None
        Maps file column names to standard names. Keys are standard names
        (flux, flux_err, redshift, etc.), values are lists of file columns
        or single column names.
    **kwargs
        Passed to the format-specific reader.

    Returns
    -------
    PhotoData
    """
    if format is None:
        format = _detect_format(path)

    readers = {
        "csv": read_csv,
        "fits": read_fits,
        "hdf5": read_hdf5,
        "numpy": read_numpy,
    }
    reader = readers.get(format)
    if reader is None:
        raise ValueError(f"Unknown format: {format!r}")
    return reader(path, column_map=column_map, **kwargs)


def save_data(data, path, format=None, **kwargs):
    """Save PhotoData to file.

    Parameters
    ----------
    data : PhotoData
        Data to save.
    path : str or Path
        Output file path.
    format : str or None
        Auto-detected from extension if None.
    **kwargs
        Passed to the format-specific writer.
    """
    if format is None:
        format = _detect_format(path)

    writers = {
        "csv": write_csv,
        "fits": write_fits,
        "hdf5": write_hdf5,
        "numpy": write_numpy,
    }
    writer = writers.get(format)
    if writer is None:
        raise ValueError(f"Unknown format: {format!r}")
    writer(data, path, **kwargs)


# ---------------------------------------------------------------------------
# Column map helper
# ---------------------------------------------------------------------------

def _apply_column_map(column_map):
    """Normalize a column_map into standard field names.

    Returns a dict with keys: flux_columns, flux_err_columns,
    redshift_column, redshift_err_column, object_id_column.
    """
    if column_map is None:
        return {}
    result = {}
    for key in ("flux_columns", "flux_err_columns"):
        if key in column_map:
            result[key] = column_map[key]
    for key in ("redshift_column", "redshift_err_column", "object_id_column"):
        if key in column_map:
            result[key] = column_map[key]
    return result


# ---------------------------------------------------------------------------
# CSV reader/writer
# ---------------------------------------------------------------------------

def read_csv(path, column_map=None, **kwargs):
    """Read PhotoData from a CSV file.

    The CSV must have columns for flux and flux errors. Use column_map to
    specify which columns correspond to which fields.

    Parameters
    ----------
    path : str or Path
        CSV file path.
    column_map : dict or None
        Maps standard field names to CSV column names. Required keys:
        'flux_columns' (list of str), 'flux_err_columns' (list of str).
        Optional: 'redshift_column', 'redshift_err_column', 'object_id_column'.

    Returns
    -------
    PhotoData
    """
    import pandas as pd

    df = pd.read_csv(path)
    cm = _apply_column_map(column_map)

    flux_cols = cm.get("flux_columns")
    flux_err_cols = cm.get("flux_err_columns")

    if flux_cols is None or flux_err_cols is None:
        raise ValueError(
            "column_map must specify 'flux_columns' and 'flux_err_columns' "
            "for CSV format"
        )

    flux = df[flux_cols].values.astype(float)
    flux_err = df[flux_err_cols].values.astype(float)
    mask = np.ones_like(flux, dtype=int)

    redshifts = None
    z_col = cm.get("redshift_column")
    if z_col and z_col in df.columns:
        redshifts = df[z_col].values.astype(float)

    redshift_errs = None
    zerr_col = cm.get("redshift_err_column")
    if zerr_col and zerr_col in df.columns:
        redshift_errs = df[zerr_col].values.astype(float)

    object_ids = None
    id_col = cm.get("object_id_column")
    if id_col and id_col in df.columns:
        object_ids = df[id_col].values

    return PhotoData(
        flux=flux, flux_err=flux_err, mask=mask,
        redshifts=redshifts, redshift_errs=redshift_errs,
        object_ids=object_ids, band_names=list(flux_cols),
    )


def write_csv(data, path, **kwargs):
    """Write PhotoData to CSV."""
    import pandas as pd

    d = {}
    band_names = data.band_names or [f"band_{i}" for i in range(data.n_bands)]
    for i, name in enumerate(band_names):
        d[name] = data.flux[:, i]
        d[f"{name}_err"] = data.flux_err[:, i]
    if data.redshifts is not None:
        d["z"] = data.redshifts
    if data.redshift_errs is not None:
        d["zerr"] = data.redshift_errs
    if data.object_ids is not None:
        d["object_id"] = data.object_ids

    df = pd.DataFrame(d)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# NumPy reader/writer
# ---------------------------------------------------------------------------

def read_numpy(path, column_map=None, **kwargs):
    """Read PhotoData from a .npz file.

    Expected arrays: 'flux', 'flux_err'. Optional: 'mask', 'redshifts',
    'redshift_errs', 'object_ids'.
    """
    path = Path(path)
    # np.savez appends .npz — try with extension if file not found
    if not path.exists() and path.with_suffix(".npz").exists():
        path = path.with_suffix(".npz")
    data = np.load(path, allow_pickle=True)
    mask = data["mask"] if "mask" in data else np.ones_like(data["flux"],
                                                            dtype=int)
    redshifts = data["redshifts"] if "redshifts" in data else None
    redshift_errs = data["redshift_errs"] if "redshift_errs" in data else None
    object_ids = data["object_ids"] if "object_ids" in data else None
    band_names = (list(data["band_names"])
                  if "band_names" in data else None)

    return PhotoData(
        flux=data["flux"].astype(float),
        flux_err=data["flux_err"].astype(float),
        mask=mask.astype(int),
        redshifts=redshifts,
        redshift_errs=redshift_errs,
        object_ids=object_ids,
        band_names=band_names,
    )


def write_numpy(data, path, **kwargs):
    """Write PhotoData to a .npz file."""
    path = Path(path)
    # np.savez appends .npz if missing — ensure consistent path
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    arrays = {
        "flux": data.flux,
        "flux_err": data.flux_err,
    }
    if data.mask is not None:
        arrays["mask"] = data.mask
    if data.redshifts is not None:
        arrays["redshifts"] = data.redshifts
    if data.redshift_errs is not None:
        arrays["redshift_errs"] = data.redshift_errs
    if data.object_ids is not None:
        arrays["object_ids"] = data.object_ids
    if data.band_names is not None:
        arrays["band_names"] = np.array(data.band_names)

    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# FITS reader/writer (optional: astropy)
# ---------------------------------------------------------------------------

def _require_astropy():
    try:
        from astropy.io import fits
        from astropy.table import Table
        return fits, Table
    except ImportError:
        raise ImportError(
            "FITS support requires astropy. Install with: "
            "uv pip install 'frankenz[fits]'"
        )


def read_fits(path, column_map=None, **kwargs):
    """Read PhotoData from a FITS table."""
    _, Table = _require_astropy()
    table = Table.read(path)
    cm = _apply_column_map(column_map)

    flux_cols = cm.get("flux_columns")
    flux_err_cols = cm.get("flux_err_columns")

    if flux_cols is None or flux_err_cols is None:
        raise ValueError(
            "column_map must specify 'flux_columns' and 'flux_err_columns' "
            "for FITS format"
        )

    flux = np.column_stack([table[c].data.astype(float) for c in flux_cols])
    flux_err = np.column_stack(
        [table[c].data.astype(float) for c in flux_err_cols]
    )
    mask = np.ones_like(flux, dtype=int)

    redshifts = None
    z_col = cm.get("redshift_column")
    if z_col and z_col in table.colnames:
        redshifts = table[z_col].data.astype(float)

    redshift_errs = None
    zerr_col = cm.get("redshift_err_column")
    if zerr_col and zerr_col in table.colnames:
        redshift_errs = table[zerr_col].data.astype(float)

    object_ids = None
    id_col = cm.get("object_id_column")
    if id_col and id_col in table.colnames:
        object_ids = np.array(table[id_col].data)

    return PhotoData(
        flux=flux, flux_err=flux_err, mask=mask,
        redshifts=redshifts, redshift_errs=redshift_errs,
        object_ids=object_ids, band_names=list(flux_cols),
    )


def write_fits(data, path, **kwargs):
    """Write PhotoData to a FITS table."""
    _, Table = _require_astropy()

    d = {}
    band_names = data.band_names or [f"band_{i}" for i in range(data.n_bands)]
    for i, name in enumerate(band_names):
        d[name] = data.flux[:, i]
        d[f"{name}_err"] = data.flux_err[:, i]
    if data.redshifts is not None:
        d["z"] = data.redshifts
    if data.redshift_errs is not None:
        d["zerr"] = data.redshift_errs
    if data.object_ids is not None:
        d["object_id"] = data.object_ids

    table = Table(d)
    table.write(path, overwrite=True)


# ---------------------------------------------------------------------------
# HDF5 reader/writer (optional: h5py)
# ---------------------------------------------------------------------------

def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "HDF5 support requires h5py. Install with: "
            "uv pip install 'frankenz[hdf5]'"
        )


def read_hdf5(path, column_map=None, group="data", **kwargs):
    """Read PhotoData from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    column_map : dict or None
        Not used for HDF5 (expects standard dataset names).
    group : str
        HDF5 group containing the datasets. Default: "data".
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        g = f[group] if group in f else f
        flux = np.array(g["flux"])
        flux_err = np.array(g["flux_err"])
        mask = np.array(g["mask"]) if "mask" in g else np.ones_like(
            flux, dtype=int
        )
        redshifts = np.array(g["redshifts"]) if "redshifts" in g else None
        redshift_errs = (np.array(g["redshift_errs"])
                         if "redshift_errs" in g else None)
        object_ids = np.array(g["object_ids"]) if "object_ids" in g else None
        band_names = None
        if "band_names" in g:
            band_names = [s.decode() if isinstance(s, bytes) else s
                          for s in g["band_names"][:]]

    return PhotoData(
        flux=flux.astype(float), flux_err=flux_err.astype(float),
        mask=mask.astype(int),
        redshifts=redshifts, redshift_errs=redshift_errs,
        object_ids=object_ids, band_names=band_names,
    )


def write_hdf5(data, path, group="data", **kwargs):
    """Write PhotoData to an HDF5 file."""
    h5py = _require_h5py()
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.create_dataset("flux", data=data.flux)
        g.create_dataset("flux_err", data=data.flux_err)
        if data.mask is not None:
            g.create_dataset("mask", data=data.mask)
        if data.redshifts is not None:
            g.create_dataset("redshifts", data=data.redshifts)
        if data.redshift_errs is not None:
            g.create_dataset("redshift_errs", data=data.redshift_errs)
        if data.object_ids is not None:
            g.create_dataset("object_ids", data=data.object_ids)
        if data.band_names is not None:
            g.create_dataset("band_names",
                             data=np.array(data.band_names, dtype="S"))
