---
title: "Frankenz Usage Guide"
date: 2026-02-28
version: 0.3.5
author: Song Huang (documentation), Joshua S. Speagle (code)
tags:
  - frankenz
  - photo-z
  - photometric-redshift
  - supervised-learning
  - bayesian
status: draft
---

# Frankenz Usage Guide

## Overview

`frankenz` is a **supervised Bayesian photometric redshift** library. It estimates galaxy redshifts by comparing observed photometry against a **labeled training set** — galaxies with known redshifts and multi-band photometry — using heteroscedastic Gaussian likelihoods and kernel density estimation (KDE) to construct posterior P(z) distributions.

The training set can come from two sources:
- **Spectroscopic catalogs** (real galaxies with spec-z): preferred for production, implicitly encodes realistic priors from the training set's redshift-magnitude distribution.
- **SED template grids** (synthetic photometry at known redshifts): useful for testing, simulation studies, or when spectroscopic training data is unavailable.

Both sources produce the same input format — `(N_train, N_filter)` photometry arrays with associated redshifts — and frankenz treats them identically.

### Core Pipeline

```
Labeled Training Data (photometry + redshifts)
  |  [from spec-z catalogs OR SED template grids]
  v
Training Photometry (N_train, N_filter) + redshifts (N_train,)
  |
  v
Compare vs. Target Data (N_obj, N_filter)
  |  [chi2 likelihood with combined data+model errors]
  v
Log-posterior weights per training galaxy
  |  [Gaussian KDE, bandwidth = neighbor z_err]
  v
Redshift PDF per object (N_obj, N_zgrid)
  |
  v
Point estimates + credible intervals + GoF metrics
```

### Fitting Methods

| Method | Class | Strategy | When to Use |
|--------|-------|----------|-------------|
| **Brute Force** | `BruteForce` | Evaluate ALL training galaxies | Small training sets (< 10k), or gold-standard reference |
| **KMCkNN** | `NearestNeighbors` | K Monte Carlo KDTrees, k nearest neighbors each | Large training sets (> 10k), production runs |
| **SOM** | `SelfOrganizingMap` | Self-organizing map compression | Dimensionality reduction, visualization |
| **GNG** | `GrowingNeuralGas` | Adaptive node growth | Auto-topology, exploratory |

All four produce the same output format: `(N_obj, N_zgrid)` PDF arrays.

---

## Installation

```bash
# From source (using uv)
git clone <repo_url>
cd frankenz
uv pip install -e .

# Dependencies: numpy, scipy, matplotlib, pandas, networkx
# Requires Python >= 3.9
```

---

## Step-by-Step Usage

### Step 1: Prepare the Training Set

The training set is an array of multi-band photometry for galaxies with known redshifts. Each row is one training galaxy. The training data can come from spectroscopic catalogs (preferred for production) or from SED template grids (useful for testing and simulations).

**Option A: Real spectroscopic training data (recommended for production)**

```python
import numpy as np

# Load your spectroscopic catalog
# Format: (N_train, N_filter) flux arrays + (N_train,) redshift arrays
# Data format can be HDF5, FITS, CSV, etc.
train_flux = np.load('spec_catalog_fluxes.npy')        # (N_train, N_filter)
train_flux_err = np.load('spec_catalog_flux_errs.npy')  # (N_train, N_filter)
train_mask = np.isfinite(train_flux) & (train_flux_err > 0)  # valid measurements

train_redshifts = np.load('spec_catalog_redshifts.npy')      # (N_train,)
train_redshift_errs = np.load('spec_catalog_redshift_errs.npy')  # (N_train,)
# Typical spec-z errors: 0.0001-0.001 for high-quality spectra
```

> **Why real training data?** When the training set is representative of the target population, magnitude-based likelihoods (`free_scale=False`) implicitly encode the prior from the training set's redshift-magnitude distribution. This is usually more informative than an analytic prior, and eliminates the need for `free_scale=True`.

**Option B: Synthetic model grid (SED templates)**

```python
from frankenz import simulate

survey = simulate.MockSurvey()

# Load a pre-defined survey (filter set + depth)
# Available: 'cosmos', 'euclid', 'hsc', 'lsst', 'sdss'
survey.load_survey('hsc')

# Load SED template library
# Available: 'brown' (129 galaxies), 'cww+' (8 templates), 'polletta+' (31 templates)
survey.load_templates('cww+')

# Optional: load BPZ prior
survey.load_prior('bpz')

# Generate model grid at specified redshifts
zgrid_model = np.arange(0, 6.01, 0.01)
survey.make_model_grid(zgrid=zgrid_model)

# Extract arrays — same format as real training data
train_flux = survey.models              # (N_model, N_filter)
train_flux_err = survey.models_err      # (N_model, N_filter)
train_mask = survey.models_mask         # (N_model, N_filter), binary

train_redshifts = survey.model_grid['redshifts']          # (N_model,)
train_redshift_errs = survey.model_grid['redshifts_err']  # (N_model,)
```

> **Template grids vs. real data**: With template grids, model amplitudes are arbitrary — you typically need `free_scale=True` to match colors only, plus an explicit prior (e.g., BPZ) to constrain magnitudes. With real spec-z catalogs, magnitudes carry physical information and `free_scale=False` is usually correct.

### Step 2: Prepare Target Data

```python
# Shape: (N_objects, N_filters)
data = np.array(...)       # Target galaxy fluxes
data_err = np.array(...)   # 1-sigma Gaussian errors (must be > 0 for valid bands)
data_mask = np.array(...)  # Binary: 1 = observed, 0 = missing/invalid
```

> **Critical**: Target and training data MUST be in the same flux system with consistent zeropoints and filter set. Frankenz performs no internal calibration or unit conversion.

### Step 3: Initialize a Fitter

**Brute Force** (simple, exhaustive — compares target against every training galaxy):
```python
from frankenz.fitting import BruteForce

fitter = BruteForce(train_flux, train_flux_err, train_mask)
```

**KMCkNN** (fast, recommended for production — uses KDTree ensembles for neighbor search):
```python
from frankenz.fitting import NearestNeighbors

fitter = NearestNeighbors(
    train_flux, train_flux_err, train_mask,
    K=25,                    # Number of MC KDTrees (more = robust, slower)
    feature_map='luptitude', # 'luptitude' (default), 'magnitude', or 'identity'
    leafsize=50,             # KDTree leaf size
)
# This prints progress as KDTrees are built: "1/25 KDTrees constructed"
```

> **Feature maps**: `luptitude` (asinh magnitudes) is recommended for photometric data with low S/N — it behaves like magnitudes for bright sources and transitions smoothly to linear flux for faint sources, avoiding the log-divergence of standard magnitudes at zero flux.

### Step 4: Fit and Generate PDFs

Define the output redshift grid:
```python
zgrid = np.linspace(0, 6, 601)  # PDF evaluation grid
```

**One-step fit+predict** (recommended, more memory-efficient):
```python
# With real training data: free_scale=False (default) is usually correct
pdfs = fitter.fit_predict(
    data, data_err, data_mask,
    train_redshifts, train_redshift_errs,
    label_grid=zgrid,
    lprob_kwargs={'free_scale': False, 'dim_prior': True},
    return_gof=False,
    verbose=True,
)
# pdfs shape: (N_objects, len(zgrid))

# With template grids: free_scale=True for color-only matching
pdfs = fitter.fit_predict(
    data, data_err, data_mask,
    train_redshifts, train_redshift_errs,
    label_grid=zgrid,
    lprob_kwargs={'free_scale': True, 'dim_prior': True},
    verbose=True,
)
```

**Two-step fit then predict** (if you want to inspect fits or reuse):
```python
fitter.fit(data, data_err, data_mask,
           lprob_kwargs={'free_scale': False})
pdfs = fitter.predict(train_redshifts, train_redshift_errs, label_grid=zgrid)
```

**Using a pre-computed KDE dictionary** (faster for many objects):
```python
from frankenz.pdf import PDFDict

sigma_grid = np.arange(0.001, 0.5, 0.001)
pdict = PDFDict(zgrid, sigma_grid)

pdfs = fitter.fit_predict(
    data, data_err, data_mask,
    train_redshifts, train_redshift_errs,
    label_dict=pdict,  # uses dictionary instead of label_grid
)
```

### Step 5: Extract Point Estimates and Diagnostics

```python
from frankenz.pdf import pdfs_summarize

results = pdfs_summarize(pdfs, zgrid)

# Unpack results
(pmean, pmean_std, pmean_conf, pmean_risk) = results[0]   # Mean (L2 optimal)
(pmed, pmed_std, pmed_conf, pmed_risk) = results[1]       # Median (L1 optimal)
(pmode, pmode_std, pmode_conf, pmode_risk) = results[2]    # Mode (MAP)
(pbest, pbest_std, pbest_conf, pbest_risk) = results[3]    # Best (min Lorentz risk)
(plow95, plow68, phigh68, phigh95) = results[4]            # Credible intervals
pmc = results[5]                                            # MC realization

# Each estimator comes with:
#   *_std  : standard deviation around the estimator
#   *_conf : PDF fraction within +/- width window (default: 0.03*(1+z))
#   *_risk : risk under the chosen loss kernel (default: Lorentz)
```

**Goodness-of-fit metrics** (optional):
```python
pdfs, (lmap, levid) = fitter.fit_predict(
    data, data_err, data_mask,
    train_redshifts, train_redshift_errs,
    label_grid=zgrid,
    return_gof=True,
)
# lmap  : ln(MAP) = max log-posterior per object
# levid : ln(evidence) = log of the marginal likelihood
```

### Step 6 (Optional): Population and Hierarchical Inference

**Population N(z) sampling** (given individual PDFs):
```python
from frankenz.samplers import population_sampler

pop = population_sampler(pdfs)  # pdfs: (N_obj, N_zbins)
pop.run_mcmc(
    Niter=1000,      # Number of saved samples
    thin=400,        # Gibbs steps between saves
    mh_steps=3,      # MH proposals per Gibbs step
)
nz_samples, lnpost = pop.results
# nz_samples shape: (1000, N_zbins) - posterior N(z) samples
```

**Hierarchical inference** (jointly infers N(z) and individual redshifts):
```python
from frankenz.samplers import hierarchical_sampler

# IMPORTANT: pdfs must be LIKELIHOODS (not posteriors)
# The prior is modeled hierarchically via a Dirichlet distribution
hier = hierarchical_sampler(pdfs)
hier.run_mcmc(
    Niter=1000,
    thin=5,
    alpha=None,       # Dirichlet concentration (default: flat, alpha=1)
    ref_sample=None,  # Optional spectroscopic reference counts
)
nz_samples, lnpost = hier.results
```

---

## Key Configuration Parameters

### Likelihood Function Parameters

These are passed via `lprob_kwargs` to `fit()` or `fit_predict()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `free_scale` | bool | `False` | Allow free amplitude scaling. `True` = color-only matching (use with template grids + explicit prior). `False` = magnitude matching (use with real training data, encodes implicit prior). |
| `dim_prior` | bool | `True` | Apply chi2-distribution correction for varying number of observed filters per object. Important for heterogeneous surveys. |
| `ignore_model_err` | bool | `False` | If True, only use data errors (ignore model uncertainties). |
| `ltol` | float | `1e-4` | Convergence tolerance for iterative scale-factor optimization (only used when `free_scale=True` and `ignore_model_err=False`). |
| `return_scale` | bool | `False` | Return the fitted scale factor and its error. |

### NearestNeighbors Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K` | int | 25 | Number of Monte Carlo KDTrees. More trees = more robust neighbor selection, slower initialization. |
| `k` | int | 20 | Neighbors per tree. Total candidate pool ~= K * k unique models. |
| `feature_map` | str/func | `'luptitude'` | Feature transformation for NN search. `'luptitude'` recommended for photometric data with low S/N. |
| `eps` | float | `1e-3` | Approximate NN tolerance. k-th neighbor within (1+eps) of true distance. |
| `leafsize` | int | 50 | KDTree leaf size (trade-off between build time and query time). |

### KDE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wt_thresh` | float | `1e-3` | Ignore models with weight < thresh * max_weight. Speeds up KDE. |
| `cdf_thresh` | float | `2e-4` | Alternative threshold using CDF (used when `wt_thresh=None`). |
| `sig_thresh` | float | `5.0` | Gaussian kernel truncation in units of sigma. |

---

## Common Pitfalls

1. **`free_scale` choice matters fundamentally**:
   - **Real training data** (spec-z catalogs): Use `free_scale=False` (default). Magnitude-based likelihoods implicitly encode the training set's redshift-magnitude prior. This is usually more informative than any analytic prior.
   - **Template grids**: Use `free_scale=True`. Template amplitudes are arbitrary, so you need color-only matching plus an explicit prior (e.g., BPZ) to constrain magnitudes.
   - Using the wrong setting is the single most common source of poor photo-z performance.

2. **Mismatched flux systems**: Target and training data must use identical units and zeropoints. No internal calibration is performed.

3. **Memory for large runs**: `BruteForce` allocates `(N_data, N_train)` arrays. For 1M targets x 100k training galaxies, this is ~800 GB. Use `NearestNeighbors` (KMCkNN) for production-scale runs.

4. **Flat priors by default**: The `logprob` wrapper applies `lnprior = 0` (flat prior). For informative priors (e.g., BPZ), write a custom `lprob_func` and pass it via `lprob_func=` parameter. With representative training data and `free_scale=False`, the flat prior is often sufficient because the training density encodes an implicit prior.

5. **KDE bandwidth from redshift errors**: The quality of P(z) PDFs depends on the `redshift_errs` array used for KDE kernel widths. For spec-z training data, use actual spec-z uncertainties (typically 0.0001–0.001). For template grids, use a small fixed value like the grid spacing (e.g., 0.005–0.01).

6. **Feature map choice for KMCkNN**: `luptitude` (default) is recommended for photometric data. It handles low-S/N bands gracefully, unlike standard `magnitude` which diverges at zero flux.

---

## Real-Data Workflow Example

This example shows a complete workflow using spectroscopic training data, based on patterns from the frankenz4DESI production pipeline (Zechang Sun). This is the recommended approach for survey-scale photo-z estimation.

```python
import numpy as np
from astropy.io import fits
from frankenz.fitting import NearestNeighbors
from frankenz.pdf import pdfs_summarize, luptitude

# --- 1. Load spectroscopic training catalog ---
train_cat = fits.getdata('spec_training_catalog.fits')
train_flux = train_cat['flux']          # (N_train, N_filter), e.g. grizy
train_flux_err = train_cat['flux_err']  # (N_train, N_filter)
train_z = train_cat['z_spec']           # (N_train,)
train_zerr = train_cat['z_spec_err']    # (N_train,)

# Quality cuts: keep clean detections
good = np.all(np.isfinite(train_flux), axis=1) & np.all(train_flux_err > 0, axis=1)
train_flux = train_flux[good]
train_flux_err = train_flux_err[good]
train_z = train_z[good]
train_zerr = train_zerr[good]

# Binary mask: all bands valid after quality cut
train_mask = np.ones_like(train_flux, dtype=bool)

# --- 2. Load target photometry ---
target_cat = fits.getdata('photometric_targets.fits')
target_flux = target_cat['flux']
target_flux_err = target_cat['flux_err']
target_mask = np.isfinite(target_flux) & (target_flux_err > 0)

# --- 3. Build KMCkNN fitter ---
fitter = NearestNeighbors(
    train_flux, train_flux_err, train_mask,
    K=25,                     # 25 Monte Carlo KDTree ensemble
    feature_map='luptitude',  # Robust for faint sources
    leafsize=20,
)

# --- 4. Run inference ---
zgrid = np.linspace(0, 7, 701)

pdfs, (lmap, levid) = fitter.fit_predict(
    target_flux, target_flux_err, target_mask,
    train_z, train_zerr,
    label_grid=zgrid,
    lprob_kwargs={
        'free_scale': False,   # Magnitude matching (real training data)
        'dim_prior': True,     # Correct for varying number of observed bands
    },
    return_gof=True,
    verbose=True,
)

# --- 5. Extract point estimates ---
results = pdfs_summarize(pdfs, zgrid)
z_mean = results[0][0]     # Mean redshift (L2 optimal)
z_median = results[1][0]   # Median redshift (L1 optimal)
z_mode = results[2][0]     # MAP redshift
z_std = results[0][1]      # Uncertainty (std around mean)
z_low68, z_high68 = results[4][1], results[4][2]  # 68% credible interval

# --- 6. Save results ---
np.savez('photoz_results.npz',
         zgrid=zgrid, pdfs=pdfs,
         z_mean=z_mean, z_median=z_median, z_mode=z_mode,
         z_std=z_std, lmap=lmap, levid=levid)
```

### Training Set Selection Guidelines

| Consideration | Recommendation |
|---------------|----------------|
| **Size** | > 10k for KMCkNN; > 1k for BruteForce |
| **Depth** | Must span the magnitude range of target data |
| **Completeness** | Representative in color-redshift space; avoid bright-only spec-z bias |
| **Redshift range** | Training z range should cover expected target z range |
| **Bands** | Same filter set with consistent zeropoints as target data |

> **Spectroscopic selection bias** is the primary systematic for supervised photo-z. If the training set under-represents faint/high-z galaxies, the resulting PDFs will be biased. Weighting schemes or hierarchical inference (`hierarchical_sampler`) can partially mitigate this.

---

## Module Reference

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `frankenz.fitting` | Fitting classes | `BruteForce`, `NearestNeighbors`, `SelfOrganizingMap`, `GrowingNeuralGas` |
| `frankenz.pdf` | Likelihoods, KDE, PDF tools | `loglike`, `logprob`, `gauss_kde`, `gauss_kde_dict`, `PDFDict`, `pdfs_summarize`, `pdfs_resample`, `magnitude`, `luptitude` |
| `frankenz.simulate` | Mock data generation | `MockSurvey`, `mag_err`, `draw_mag`, `draw_ztm` |
| `frankenz.priors` | Prior distributions | `pmag`, `bpz_pt_m`, `bpz_pz_tm` |
| `frankenz.reddening` | IGM attenuation | `madau_teff` |
| `frankenz.samplers` | Population/hierarchical MCMC | `population_sampler`, `hierarchical_sampler`, `loglike_nz` |
| `frankenz.plotting` | Visualization | `input_vs_pdf`, `cdf_vs_epdf`, `plot2d_network` |

---

## Demo Notebooks

The `demos/` directory contains 6 Jupyter notebooks demonstrating the full workflow:

1. **`1 - Mock Data.ipynb`** - Generate synthetic photometric surveys using `MockSurvey`
2. **`2 - Photometric Inference.ipynb`** - Fit mock data with different methods
3. **`3 - Photometric PDFs.ipynb`** - Generate and analyze redshift PDFs
4. **`4 - Posterior Approximations.ipynb`** - Compare fitting methods and their PDF approximations
5. **`5 - Population Inference with Redshifts.ipynb`** - Population N(z) distributions
6. **`6 - Hierarchical Inference with Redshifts.ipynb`** - Hierarchical Bayesian inference
