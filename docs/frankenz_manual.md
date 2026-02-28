---
title: "Frankenz User Manual"
date: 2026-02-28
version: 0.4.0
author: Song Huang (documentation), Joshua S. Speagle (code)
---

# Frankenz User Manual

## 1. What Is Frankenz?

Frankenz is a **supervised Bayesian photometric redshift** library for Python. Given a training set of galaxies with known (spectroscopic) redshifts and multi-band photometry, frankenz estimates redshift probability distributions P(z) for new galaxies by comparing their photometry against the training set.

**Frankenz is NOT template fitting.** In template-fitting codes (e.g., EAZY, LePhare), you compare observed photometry against synthetic SED models at each redshift. In frankenz, the "models" are real galaxies (or SED-template grids) with known redshifts, and the method is fundamentally a supervised learning approach.

### How It Works

```
Input: Training set with (flux, flux_err, spec-z) for each galaxy
       Target set with (flux, flux_err) for each galaxy

Step 1: Feature transform
        Convert fluxes -> luptitudes (or magnitudes) for neighbor search

Step 2: Neighbor search (KMCkNN)
        Build K Monte Carlo KDTree ensembles from noisy training data
        For each target: query K trees, take k neighbors from each, union results

Step 3: Likelihood
        Compute chi2 likelihood between target and each neighbor
        Using combined data + model flux errors

Step 4: KDE smoothing
        Weight neighbor redshifts by posterior probability
        Smooth with Gaussian kernel (bandwidth = neighbor's redshift error)

Output: P(z) PDF for each target galaxy on a redshift grid
```

### Fitting Backends

| Backend | Class | Strategy | When to Use |
|---------|-------|----------|-------------|
| **KMCkNN** | `NearestNeighbors` | K Monte Carlo KDTree ensembles, k nearest neighbors each | Production runs with large training sets (> 10k) |
| **Brute Force** | `BruteForce` | Compare against every training galaxy | Small training sets (< 10k), reference comparisons |
| **SOM** | `SelfOrganizingMap` | Self-organizing map compression | Dimensionality reduction, visualization |
| **GNG** | `GrowingNeuralGas` | Adaptive topology growth | Exploratory analysis |

For most production work, use **KMCkNN** (`NearestNeighbors`).

---

## 2. Installation

### Requirements

- Python >= 3.9
- Core dependencies: numpy, scipy, matplotlib, pandas, networkx, pyyaml

### Install from Source

```bash
# Clone the repository
git clone https://github.com/joshspeagle/frankenz.git
cd frankenz

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Optional Extras

```bash
# FITS file support (requires astropy)
uv pip install -e ".[fits]"

# HDF5 file support (requires h5py)
uv pip install -e ".[hdf5]"

# Progress bars (requires tqdm)
uv pip install -e ".[progress]"

# All optional dependencies
uv pip install -e ".[all]"
```

### Verify Installation

```python
import frankenz
print(frankenz.__version__)  # Should print "0.4.0"
```

---

## 3. Input Data

Frankenz requires two datasets:

1. **Training set**: galaxies with known redshifts + multi-band photometry
2. **Target set**: galaxies whose redshifts you want to estimate

Both must be in the **same photometric system** — same filters, same flux units, same zeropoints.

### Data Format

All photometric data enters frankenz as three arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `flux` | `(N_objects, N_bands)` | Flux densities |
| `flux_err` | `(N_objects, N_bands)` | 1-sigma Gaussian flux errors |
| `mask` | `(N_objects, N_bands)` | Binary mask: 1 = valid, 0 = missing |

Training data additionally requires:

| Array | Shape | Description |
|-------|-------|-------------|
| `redshifts` | `(N_objects,)` | Known (spectroscopic) redshifts |
| `redshift_errs` | `(N_objects,)` | Redshift uncertainties |

### 3.1 Generating Mock Data (for Testing)

Frankenz includes a `MockSurvey` simulator for generating synthetic photometric data. This is useful for testing the pipeline before applying it to real data.

```python
import numpy as np
from frankenz.simulate import MockSurvey

# Initialize a mock survey with preset filters + templates + prior
rstate = np.random.RandomState(42)
survey = MockSurvey(
    survey='sdss',       # Filter set: 'sdss', 'hsc', 'lsst', 'cosmos', 'euclid'
    templates='cww+',    # SED library: 'cww+' (8), 'brown' (129), 'polletta+' (31)
    prior='bpz',         # Prior: 'bpz' (Benitez 2000)
    rstate=rstate,
)

# Generate 500 mock galaxies with noisy photometry
survey.make_mock(500, rstate=rstate, verbose=True)
# This draws (z, type, mag) from the BPZ prior,
# then generates noisy multi-band photometry from the SED templates.

# Access the generated data
data = survey.data
flux = data['phot_obs']       # (500, N_bands) — noisy observed fluxes
flux_err = data['phot_err']   # (500, N_bands) — flux errors
redshifts = data['redshifts'] # (500,) — true redshifts
templates = data['templates'] # (500,) — template indices

# Split into training (400) and target (100) sets
n_train = 400
train_flux = flux[:n_train]
train_flux_err = flux_err[:n_train]
train_z = redshifts[:n_train]
train_zerr = np.full(n_train, 0.001)  # typical spec-z errors

target_flux = flux[n_train:]
target_flux_err = flux_err[n_train:]
target_z_true = redshifts[n_train:]   # hold-out for validation
```

You can also generate a model grid (useful for template-fitting mode):

```python
# Generate a template photometry grid at fixed redshifts
zgrid_model = np.arange(0, 3.01, 0.01)
survey.make_model_grid(redshifts=zgrid_model)

# Access the grid: shape (N_z, N_templates, N_filters)
model_flux = survey.models
```

### 3.2 Preparing Real Observational Data

To use frankenz with real data, you need two catalogs in the same photometric system.

**Step 1: Prepare arrays from your catalogs**

```python
import numpy as np

# --- Example: you have N galaxies with M photometric bands ---
# Training set: galaxies with spectroscopic redshifts
#   train_flux:     (N_train, M) — flux in each band
#   train_flux_err: (N_train, M) — flux error in each band
#   train_z:        (N_train,)   — spectroscopic redshifts
#   train_zerr:     (N_train,)   — spec-z errors (typically 0.0001-0.001)

# Target set: galaxies to estimate photo-z
#   target_flux:     (N_target, M) — flux in each band
#   target_flux_err: (N_target, M) — flux error in each band
```

**Step 2: Quality cuts**

```python
# Remove objects with invalid photometry
def clean_photometry(flux, flux_err):
    """Remove objects where any band has NaN or non-positive error."""
    valid = np.all(np.isfinite(flux), axis=1) & np.all(flux_err > 0, axis=1)
    return flux[valid], flux_err[valid], valid

train_flux, train_flux_err, good_train = clean_photometry(
    train_flux, train_flux_err
)
train_z = train_z[good_train]
train_zerr = train_zerr[good_train]
```

**Step 3: Build masks**

```python
# Binary masks: 1 = valid measurement, 0 = missing/bad
# After quality cuts, all bands are valid:
train_mask = np.ones_like(train_flux, dtype=int)

# For targets, some bands may be missing:
target_mask = np.isfinite(target_flux) & (target_flux_err > 0)
target_mask = target_mask.astype(int)
```

**Example: loading from a FITS catalog**

```python
from astropy.io import fits

# Load training catalog
hdu = fits.open('spectroscopic_catalog.fits')
cat = hdu[1].data

# Extract M-band photometry (e.g., grizy for HSC)
bands = ['g', 'r', 'i', 'z', 'y']
train_flux = np.column_stack([cat[f'flux_{b}'] for b in bands])
train_flux_err = np.column_stack([cat[f'flux_err_{b}'] for b in bands])
train_z = cat['z_spec']
train_zerr = cat['z_spec_err']
train_mask = np.ones_like(train_flux, dtype=int)
```

### 3.3 Using the PhotoData Container

Frankenz v0.4.0 provides a `PhotoData` dataclass for bundling photometric data with metadata. This is used by the config-driven pipeline (`run_pipeline`) and the I/O functions.

```python
from frankenz.io import PhotoData

# Create a PhotoData object
training_data = PhotoData(
    flux=train_flux,               # (N_train, M) required
    flux_err=train_flux_err,       # (N_train, M) required
    mask=train_mask,               # (N_train, M) optional; defaults to all-valid
    redshifts=train_z,             # (N_train,) required for training data
    redshift_errs=train_zerr,      # (N_train,) optional
    band_names=['g', 'r', 'i', 'z', 'y'],  # optional
)

test_data = PhotoData(
    flux=target_flux,
    flux_err=target_flux_err,
    mask=target_mask,
)

# Validate internal consistency
training_data.validate()  # raises ValueError if shapes mismatch

# Properties
print(training_data.n_objects)  # number of objects
print(training_data.n_bands)    # number of bands

# Subsetting
subset = training_data.subset([0, 1, 2, 3, 4])  # first 5 objects
```

### 3.4 Multi-Format I/O

Frankenz supports reading and writing `PhotoData` in multiple formats.

**CSV**

```python
from frankenz.io import load_data, save_data

# Reading: you must specify which CSV columns map to flux/error
column_map = {
    'flux_columns': ['flux_g', 'flux_r', 'flux_i', 'flux_z', 'flux_y'],
    'flux_err_columns': ['fluxerr_g', 'fluxerr_r', 'fluxerr_i',
                         'fluxerr_z', 'fluxerr_y'],
    'redshift_column': 'z_spec',
    'redshift_err_column': 'z_spec_err',
    'object_id_column': 'object_id',
}
data = load_data('catalog.csv', column_map=column_map)

# Writing
save_data(data, 'output.csv')
```

**FITS** (requires `astropy`)

```python
# Same column_map interface
data = load_data('catalog.fits', column_map=column_map)
save_data(data, 'output.fits')
```

**HDF5** (requires `h5py`)

```python
# HDF5 uses standard dataset names: flux, flux_err, mask, redshifts, etc.
data = load_data('catalog.hdf5')  # no column_map needed
save_data(data, 'output.hdf5')
```

**NumPy (.npz)**

```python
# Uses standard array names
data = load_data('catalog.npz')
save_data(data, 'output.npz')
```

The format is auto-detected from the file extension (`.csv`, `.fits`/`.fit`, `.hdf5`/`.h5`, `.npz`), or you can specify it explicitly with `format='csv'`.

---

## 4. Running the Pipeline

Frankenz offers two ways to run the photo-z estimation pipeline:

- **Direct API**: maximum control, step by step
- **Config-driven pipeline**: YAML config + `run_pipeline()` for reproducible runs

### 4.1 Direct API (Step-by-Step)

This gives you full control over each stage of the pipeline.

#### Step 1: Initialize a Fitter

```python
from frankenz.fitting import NearestNeighbors, BruteForce

# --- KMCkNN (recommended for production) ---
fitter = NearestNeighbors(
    models=train_flux,            # (N_train, M)
    models_err=train_flux_err,    # (N_train, M)
    models_mask=train_mask,       # (N_train, M)
    K=25,                         # Number of Monte Carlo KDTrees
    feature_map='luptitude',      # 'luptitude', 'magnitude', or 'identity'
    leafsize=50,                  # KDTree leaf size
    verbose=True,                 # Print progress as trees are built
)
# Output: "1/25 KDTrees constructed", ..., "25/25 KDTrees constructed"

# --- BruteForce (small training sets only) ---
fitter = BruteForce(
    models=train_flux,
    models_err=train_flux_err,
    models_mask=train_mask,
)
```

#### Step 2: Define the Redshift Grid

```python
import numpy as np

# The output P(z) will be evaluated on this grid
zgrid = np.linspace(0, 3, 301)  # 0 to 3 in steps of 0.01
```

#### Step 3: Fit and Predict

```python
# --- One-step fit_predict (recommended) ---
pdfs = fitter.fit_predict(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_grid=zgrid,
    lprob_kwargs={
        'free_scale': False,     # Use magnitude matching (see Section 6)
        'dim_prior': True,       # Correct for varying number of observed bands
    },
    return_gof=False,
    verbose=True,
)
# pdfs shape: (N_target, len(zgrid))
# Each row is a normalized P(z) on the zgrid

# --- With goodness-of-fit metrics ---
pdfs, (lmap, levid) = fitter.fit_predict(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_grid=zgrid,
    lprob_kwargs={'free_scale': False, 'dim_prior': True},
    return_gof=True,
    verbose=True,
)
# lmap:  (N_target,) — log(MAP) = max log-posterior per object
# levid: (N_target,) — log(evidence) = marginal likelihood
```

**KNN-specific parameters** for `fit_predict`:

```python
pdfs = fitter.fit_predict(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_grid=zgrid,
    k=20,                   # Neighbors per tree (total pool ~ K * k unique)
    eps=1e-3,               # Approximate NN tolerance
    lp_norm=2,              # Distance metric (2 = Euclidean)
    rstate=np.random.RandomState(42),  # For reproducibility
    lprob_kwargs={'free_scale': False, 'dim_prior': True},
    verbose=True,
)
```

#### Step 4: Extract Point Estimates

```python
from frankenz.pdf import pdfs_summarize

results = pdfs_summarize(pdfs, zgrid)

# Results structure:
# results[0] = (mean, mean_std, mean_conf, mean_risk)    — L2-optimal
# results[1] = (median, median_std, median_conf, median_risk) — L1-optimal
# results[2] = (mode, mode_std, mode_conf, mode_risk)    — MAP estimate
# results[3] = (best, best_std, best_conf, best_risk)    — min Lorentz risk
# results[4] = (low95, low68, high68, high95)             — credible intervals
# results[5] = mc_realization                              — MC sample from PDF

# Extract commonly used quantities
z_mean = results[0][0]        # (N_target,) mean redshift
z_median = results[1][0]      # (N_target,) median redshift
z_mode = results[2][0]        # (N_target,) MAP redshift
z_std = results[0][1]         # (N_target,) uncertainty (std around mean)
z_low68 = results[4][1]       # (N_target,) lower 68% credible bound
z_high68 = results[4][2]      # (N_target,) upper 68% credible bound

# Quality metrics per estimator:
#   *_std:  standard deviation computed around the estimator
#   *_conf: fraction of PDF within +/- window around estimator
#           (default window: 0.03 * (1+z), designed for photo-z)
#   *_risk: risk under the Lorentz loss kernel
```

#### Step 5: Save Results

```python
np.savez(
    'photoz_results.npz',
    zgrid=zgrid,
    pdfs=pdfs,
    z_mean=z_mean,
    z_median=z_median,
    z_mode=z_mode,
    z_std=z_std,
)
```

### 4.2 Using a Pre-Computed KDE Dictionary (Performance)

For large runs, pre-computing the KDE kernel dictionary avoids recomputing Gaussian kernels per object:

```python
from frankenz.pdf import PDFDict

# Pre-compute dictionary of Gaussian kernels
sigma_grid = np.arange(0.001, 0.5, 0.001)  # range of possible bandwidths
pdict = PDFDict(zgrid, sigma_grid)

# Use the dictionary in fit_predict
pdfs = fitter.fit_predict(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_dict=pdict,   # <-- use dictionary instead of label_grid
    lprob_kwargs={'free_scale': False},
    verbose=True,
)
```

### 4.3 Two-Step Fit then Predict

If you want to inspect the intermediate fit results or reuse the fit for different redshift grids:

```python
# Step A: Fit — compute likelihoods between targets and training set
fitter.fit(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    lprob_kwargs={'free_scale': False, 'dim_prior': True},
)

# Step B: Predict — convert likelihoods to P(z) via KDE
pdfs = fitter.predict(
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_grid=zgrid,
)
```

### 4.4 Config-Driven Pipeline

For reproducible runs with YAML-based configuration:

```python
from frankenz.config import FrankenzConfig
from frankenz.io import PhotoData
from frankenz.batch import run_pipeline

# --- Option A: Use all defaults ---
config = FrankenzConfig()

# --- Option B: Create from a dictionary ---
config = FrankenzConfig.from_dict({
    'model': {
        'backend': 'knn',
        'k_tree': 25,
        'k_point': 20,
        'free_scale': False,
        'dim_prior': True,
    },
    'transform': {
        'type': 'luptitude',
        'zeropoints': 1.0,
        'skynoise': [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    'prior': {'type': 'uniform'},
    'zgrid': {
        'z_start': 0.0,
        'z_end': 3.0,
        'z_delta': 0.01,
    },
    'verbose': True,
    'seed': 42,
})

# --- Option C: Load from YAML ---
config = FrankenzConfig.from_yaml('my_config.yaml')

# --- Option D: Start from defaults, apply overrides ---
config = FrankenzConfig()
config = config.override({
    'model': {'k_tree': 30, 'k_point': 25},
    'zgrid': {'z_end': 4.0},
})

# Package data as PhotoData
training_data = PhotoData(
    flux=train_flux,
    flux_err=train_flux_err,
    mask=train_mask,
    redshifts=train_z,
    redshift_errs=train_zerr,
)
test_data = PhotoData(
    flux=target_flux,
    flux_err=target_flux_err,
    mask=target_mask,
)

# Run the full pipeline with chunked processing
result = run_pipeline(
    config=config,
    training_data=training_data,
    test_data=test_data,
    chunk_size=1000,   # objects per chunk (controls memory)
)

# Access results
pdfs = result.pdfs       # (N_target, N_zgrid)
zgrid = result.zgrid     # the redshift grid used
summary = result.summary # pdfs_summarize output (if verbose=True)

# Save config for reproducibility
config.to_yaml('my_run_config.yaml')
```

---

## 5. Configuration Reference

The configuration system uses Python dataclasses organized in a hierarchy. All fields have sensible defaults.

### 5.1 Top-Level: `FrankenzConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `ModelConfig` | see below | Fitting backend and parameters |
| `transform` | `TransformConfig` | see below | Feature transform for NN search |
| `prior` | `PriorConfig` | see below | Bayesian prior configuration |
| `zgrid` | `ZGridConfig` | see below | Redshift grid specification |
| `pdf` | `PDFConfig` | see below | PDF construction parameters |
| `data` | `DataConfig` | see below | Data I/O configuration |
| `verbose` | `bool` | `True` | Print progress |
| `seed` | `int` | `None` | Random seed for reproducibility |

### 5.2 `ModelConfig`

Controls the fitting backend and likelihood parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str` | `"knn"` | `"knn"` (KMCkNN) or `"bruteforce"` |
| `k_tree` | `int` | `25` | Number of Monte Carlo KDTrees (K). More trees = more robust, slower init. |
| `k_point` | `int` | `20` | Neighbors per tree (k). Total candidate pool ~= K * k unique galaxies. |
| `free_scale` | `bool` | `False` | Allow free amplitude scaling. `True` = color-only. `False` = magnitude matching. |
| `ignore_model_err` | `bool` | `False` | If True, ignore training data flux errors in likelihood. |
| `dim_prior` | `bool` | `True` | Apply chi2 DOF correction for varying number of observed bands. |
| `track_scale` | `bool` | `False` | Return fitted scale factors. |
| `kdtree` | `KDTreeConfig` | see below | KDTree construction parameters. |

### 5.3 `KDTreeConfig`

Controls the KDTree used in KMCkNN neighbor search.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `leafsize` | `int` | `50` | KDTree leaf size. Smaller = faster query, slower build. |
| `eps` | `float` | `1e-3` | Approximate NN tolerance. k-th neighbor within (1+eps) of true distance. |
| `lp_norm` | `int` | `2` | Distance metric: 1 = Manhattan, 2 = Euclidean. |
| `distance_upper_bound` | `float` | `inf` | Maximum neighbor distance. |

### 5.4 `TransformConfig`

Controls the photometric feature transform applied before KDTree construction. Only affects `NearestNeighbors` (not `BruteForce`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"luptitude"` | `"luptitude"`, `"magnitude"`, or `"identity"`. |
| `zeropoints` | `float` | `1.0` | Flux zeropoint for the magnitude/luptitude transform. |
| `skynoise` | `list` | `[1.0]` | Per-band sky noise for the luptitude softening parameter. |

**Transform types explained:**

- **`luptitude`** (default, recommended): Asinh magnitudes (Lupton et al. 1999). Behaves like magnitudes for bright sources and transitions smoothly to linear flux for faint sources. Avoids the log-divergence of standard magnitudes at zero flux. Best for photometric data with low signal-to-noise bands.
- **`magnitude`**: Standard AB magnitudes. Produces `NaN` for non-positive fluxes. Only use if all bands have high S/N.
- **`identity`**: No transform; uses raw fluxes. Useful when data is already in a suitable feature space.

### 5.5 `PriorConfig`

Controls the Bayesian prior on redshift.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"uniform"` | `"uniform"` or `"bpz"`. |
| `k_tree` | `int` | `25` | Reserved for future KNN-based data-driven prior. |
| `k_point` | `int` | `20` | Reserved for future KNN-based data-driven prior. |

**Prior types:**

- **`uniform`** (default): Flat prior `P(z) = const`. When using real spectroscopic training data with `free_scale=False`, the training set density acts as an implicit prior, so a flat explicit prior is often sufficient.
- **`bpz`**: The BPZ prior from Benitez (2000), parameterized by galaxy type and magnitude. Uses a fixed reference magnitude of 25.0. Useful when working with SED template grids and `free_scale=True`.

### 5.6 `ZGridConfig`

Defines the redshift grid on which P(z) is evaluated.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `z_start` | `float` | `0.0` | Start of redshift grid. |
| `z_end` | `float` | `7.0` | End of redshift grid. |
| `z_delta` | `float` | `0.01` | Grid spacing. |

The grid is generated as `np.arange(z_start, z_end + z_delta/2, z_delta)`.

**Guidance**: Grid spacing should be smaller than the expected redshift errors. For spec-z training data, `z_delta=0.01` is typically fine. A coarser grid (e.g., 0.05) speeds up computation at the cost of PDF resolution.

### 5.7 `PDFConfig`

Controls the KDE step that builds P(z) from weighted neighbor redshifts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wt_thresh` | `float` | `1e-3` | Weight threshold: ignore neighbors with weight < thresh * max_weight. Speeds up KDE. |
| `cdf_thresh` | `float` | `2e-4` | CDF threshold (used when `wt_thresh=None`). |

### 5.8 `DataConfig`

Specifies column mappings for data I/O.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | `str` | `"csv"` | File format: `"csv"`, `"fits"`, `"hdf5"`, `"numpy"`. |
| `flux_columns` | `list` | `[]` | Column names for flux values. |
| `flux_err_columns` | `list` | `[]` | Column names for flux errors. |
| `redshift_column` | `str` | `"z"` | Column name for redshifts. |
| `redshift_err_column` | `str` | `"zerr"` | Column name for redshift errors. |
| `object_id_column` | `str` | `"object_id"` | Column name for object IDs. |

### 5.9 Example YAML Config

```yaml
model:
  backend: knn
  k_tree: 25
  k_point: 20
  free_scale: false
  ignore_model_err: false
  dim_prior: true
  track_scale: false
  kdtree:
    leafsize: 50
    eps: 0.001
    lp_norm: 2
    distance_upper_bound: .inf

transform:
  type: luptitude
  zeropoints: 1.0
  skynoise: [1.0, 1.0, 1.0, 1.0, 1.0]

prior:
  type: uniform

zgrid:
  z_start: 0.0
  z_end: 3.0
  z_delta: 0.01

pdf:
  wt_thresh: 0.001
  cdf_thresh: 0.0002

data:
  format: csv
  flux_columns: [flux_g, flux_r, flux_i, flux_z, flux_y]
  flux_err_columns: [fluxerr_g, fluxerr_r, fluxerr_i, fluxerr_z, fluxerr_y]
  redshift_column: z_spec
  redshift_err_column: z_spec_err
  object_id_column: object_id

verbose: true
seed: 42
```

### 5.10 Working with Configs Programmatically

```python
from frankenz.config import FrankenzConfig

# Create with all defaults
cfg = FrankenzConfig()

# Serialize to/from YAML
cfg.to_yaml('config.yaml')
cfg2 = FrankenzConfig.from_yaml('config.yaml')

# Serialize to/from dict
d = cfg.to_dict()
cfg3 = FrankenzConfig.from_dict(d)

# Apply overrides (returns a new config, does not mutate)
cfg4 = cfg.override({'model': {'k_tree': 50}, 'seed': 123})
# cfg.model.k_tree is still 25; cfg4.model.k_tree is 50

# Access nested fields
print(cfg.model.backend)        # 'knn'
print(cfg.model.kdtree.eps)     # 0.001
print(cfg.transform.type)       # 'luptitude'
```

---

## 6. Complete Worked Example

Here is a full end-to-end example using mock data:

```python
import numpy as np
from frankenz.simulate import MockSurvey
from frankenz.fitting import NearestNeighbors
from frankenz.pdf import pdfs_summarize

# ----------------------------------------------------------------
# 1. Generate mock data
# ----------------------------------------------------------------
rstate = np.random.RandomState(42)
survey = MockSurvey(survey='sdss', templates='cww+', prior='bpz', rstate=rstate)
survey.make_mock(500, rstate=rstate, verbose=False)

data = survey.data
n_train = 400
train_flux = data['phot_obs'][:n_train]
train_flux_err = data['phot_err'][:n_train]
train_mask = np.ones_like(train_flux, dtype=int)
train_z = data['redshifts'][:n_train]
train_zerr = np.full(n_train, 0.001)

target_flux = data['phot_obs'][n_train:]
target_flux_err = data['phot_err'][n_train:]
target_mask = np.ones_like(target_flux, dtype=int)
target_z_true = data['redshifts'][n_train:]

# ----------------------------------------------------------------
# 2. Build the KMCkNN fitter
# ----------------------------------------------------------------
fitter = NearestNeighbors(
    models=train_flux,
    models_err=train_flux_err,
    models_mask=train_mask,
    K=25,
    feature_map='luptitude',
    leafsize=50,
    rstate=rstate,
    verbose=True,
)

# ----------------------------------------------------------------
# 3. Estimate P(z) for all targets
# ----------------------------------------------------------------
zgrid = np.linspace(0, 3, 301)

pdfs, (lmap, levid) = fitter.fit_predict(
    data=target_flux,
    data_err=target_flux_err,
    data_mask=target_mask,
    model_labels=train_z,
    model_label_errs=train_zerr,
    label_grid=zgrid,
    lprob_kwargs={'free_scale': False, 'dim_prior': True},
    return_gof=True,
    verbose=True,
)

# ----------------------------------------------------------------
# 4. Extract point estimates
# ----------------------------------------------------------------
results = pdfs_summarize(pdfs, zgrid)
z_mean = results[0][0]
z_median = results[1][0]
z_mode = results[2][0]
z_std = results[0][1]

# ----------------------------------------------------------------
# 5. Evaluate performance
# ----------------------------------------------------------------
dz = (z_mean - target_z_true) / (1 + target_z_true)
bias = np.median(dz)
scatter = 1.4826 * np.median(np.abs(dz - bias))  # normalized MAD
outlier_frac = np.mean(np.abs(dz) > 0.15)

print(f"Bias:     {bias:.4f}")
print(f"Scatter:  {scatter:.4f}")
print(f"Outliers: {outlier_frac:.1%} (|dz/(1+z)| > 0.15)")
```

---

## 7. Advanced: Population and Hierarchical Inference

After computing individual P(z) PDFs, frankenz can infer the population redshift distribution N(z) via MCMC.

### Population N(z) Sampling

Given individual PDFs, sample the population N(z):

```python
from frankenz.samplers import population_sampler

pop = population_sampler(pdfs)  # pdfs: (N_obj, N_zbins)
pop.run_mcmc(
    Niter=1000,    # Number of saved samples
    thin=400,      # Gibbs steps between saves
    mh_steps=3,    # Metropolis-Hastings proposals per Gibbs step
)
nz_samples, lnpost = pop.results
# nz_samples: (1000, N_zbins) — posterior N(z) samples
# lnpost:     (1000,) — log-posterior values
```

### Hierarchical Inference

Jointly infer N(z) and individual redshifts:

```python
from frankenz.samplers import hierarchical_sampler

# IMPORTANT: pdfs must be LIKELIHOODS (not posteriors) for hierarchical inference.
# The prior is modeled hierarchically via a Dirichlet distribution.
hier = hierarchical_sampler(pdfs)
hier.run_mcmc(
    Niter=1000,
    thin=5,
    alpha=None,       # Dirichlet concentration; None = flat (alpha=1)
    ref_sample=None,  # Optional spectroscopic reference counts
)
nz_samples, lnpost = hier.results
```

---

## 8. Common Issues and Pitfalls

### 8.1 The `free_scale` Choice (Most Important)

This is the single most common source of poor photo-z performance.

| Scenario | `free_scale` | Why |
|----------|-------------|-----|
| **Spectroscopic training data** | `False` (default) | Magnitude-based likelihoods implicitly encode the training set's redshift-magnitude prior. This is usually more informative than any analytic prior. |
| **SED template grids** | `True` | Template amplitudes are arbitrary — you need color-only matching plus an explicit prior (e.g., BPZ). |

If you use `free_scale=True` with real training data, you lose the implicit magnitude prior and need to supply an explicit one. If you use `free_scale=False` with template grids, the arbitrary template normalization will dominate and produce garbage.

### 8.2 Mismatched Photometric Systems

Target and training data **must** share:
- The same filter set (same bandpasses in the same order)
- The same flux units (e.g., both in microJansky, or both in nanomaggies)
- The same zeropoints

Frankenz performs **no** internal calibration, unit conversion, or filter matching. If your training set is in AB magnitudes and your targets are in Jansky, results will be meaningless.

### 8.3 Memory Limits

- **BruteForce** allocates `(N_target, N_train)` arrays internally. For 100k targets x 100k training, that is 80 GB. Use `NearestNeighbors` for production.
- **NearestNeighbors** is memory-efficient: it only computes likelihoods for ~K*k neighbors per target. Use `chunk_size` in `run_pipeline()` to process targets in batches.

### 8.4 Redshift Error Arrays for KDE

The `model_label_errs` (redshift errors) array directly sets the KDE kernel bandwidth. These should be:
- **Spectroscopic training data**: actual spec-z errors, typically 0.0001 to 0.001.
- **Template grid**: use a small fixed value comparable to the grid spacing (e.g., 0.005 to 0.01).

If you set redshift errors too large, the PDFs will be over-smoothed. If too small (or zero), the PDFs will be a sum of delta functions and numerically unstable.

### 8.5 Feature Map for KMCkNN

The `feature_map` parameter only affects the KDTree neighbor search in `NearestNeighbors`. The likelihood computation always uses raw fluxes.

- **`luptitude`** (default): Best for surveys with faint/noisy bands. Gracefully handles zero and negative fluxes.
- **`magnitude`**: Faster, but produces `NaN` for non-positive fluxes. Only use if all training and target fluxes are positive.
- **`identity`**: Uses raw fluxes. May be appropriate if your data is already in a well-conditioned feature space.

### 8.6 Spectroscopic Selection Bias

The primary systematic for supervised photo-z: if the training set under-represents faint or high-redshift galaxies (common for spec-z samples), the resulting PDFs will be biased toward the populations that are well-represented.

**Mitigations:**
- Use a training set that spans the target's full magnitude-redshift range.
- Use `hierarchical_sampler` to partially correct for incompleteness.
- Consider weighting schemes (not built into frankenz; implement as a custom `lprob_func`).

### 8.7 Band Naming in CSV/FITS I/O

When using column-map-based readers, avoid band names that conflict with metadata columns. For example, if your survey has a `z` band and you also have a `z` redshift column, use distinct column names like `flux_z` and `z_spec`.

### 8.8 Reproducibility

For reproducible results with `NearestNeighbors`, always set `seed` in the config or pass an explicit `rstate`:

```python
rstate = np.random.RandomState(42)
fitter = NearestNeighbors(..., rstate=rstate)
pdfs = fitter.fit_predict(..., rstate=rstate)
```

The Monte Carlo KDTree construction uses random noise realizations, so different random states produce different (but statistically equivalent) neighbor sets.

---

## 9. Module Reference

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `frankenz.config` | Configuration system | `FrankenzConfig`, `ModelConfig`, `TransformConfig`, `PriorConfig`, `ZGridConfig`, `PDFConfig`, `DataConfig`, `KDTreeConfig` |
| `frankenz.transforms` | Feature transforms | `identity`, `magnitude`, `luptitude`, `inv_magnitude`, `inv_luptitude`, `get_transform` |
| `frankenz.io` | Data I/O | `PhotoData`, `load_data`, `save_data`, `read_csv`, `read_fits`, `read_hdf5`, `read_numpy`, `write_csv`, `write_fits`, `write_hdf5`, `write_numpy` |
| `frankenz.batch` | Pipeline runner | `run_pipeline`, `PipelineResult` |
| `frankenz.fitting` | Fitter classes + factory | `BruteForce`, `NearestNeighbors`, `SelfOrganizingMap`, `GrowingNeuralGas`, `get_fitter` |
| `frankenz.pdf` | Likelihoods, KDE, PDF tools | `loglike`, `logprob`, `gauss_kde`, `gauss_kde_dict`, `PDFDict`, `pdfs_summarize`, `pdfs_resample` |
| `frankenz.priors` | Bayesian priors + factory | `pmag`, `bpz_pt_m`, `bpz_pz_tm`, `get_prior` |
| `frankenz.simulate` | Mock data generation | `MockSurvey`, `mag_err`, `draw_mag`, `draw_ztm` |
| `frankenz.samplers` | Population/hierarchical MCMC | `population_sampler`, `hierarchical_sampler` |
| `frankenz.reddening` | IGM attenuation | `madau_teff` |
| `frankenz.plotting` | Visualization | `input_vs_pdf`, `cdf_vs_epdf`, `plot2d_network` |
