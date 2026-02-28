# Frankenz Project Instructions

## What Is Frankenz

Frankenz is a **supervised Bayesian photometric redshift** library. It is NOT template fitting. It learns mappings from photometric feature space to redshift via labeled training data (spectroscopic redshifts + multi-band photometry). Core algorithm: KMCkNN (K Monte Carlo k-Nearest Neighbors) with heteroscedastic Gaussian KDE for PDF construction.

### Algorithm Pipeline
```
Training data (flux, flux_err, spec-z)
  -> Feature transform (luptitude/magnitude)
  -> K Monte Carlo KDTree ensembles (noise-aware indexing)
  -> Per-object: query K trees, union k neighbors each
  -> Likelihood: chi2 with combined data+model errors, optional free_scale
  -> KDE smoothing: neighbor redshifts weighted by posterior, bandwidth = neighbor z_err
  -> Output: P(z) PDF per object on redshift grid
```

### Key Distinction from Template Fitting
- Training set = labeled data (real galaxies with spec-z), not just SED templates
- Magnitude likelihoods implicitly encode the training set's prior
- `free_scale=True` -> color-only matching (loses magnitude info, needs explicit prior)
- `free_scale=False` -> magnitude matching (implicit prior from training set density)

## Architecture

| Module | Role |
|--------|------|
| `config.py` | Dataclass config hierarchy + YAML serde |
| `transforms.py` | Feature transforms (identity/magnitude/luptitude) + factory |
| `io.py` | PhotoData container + CSV/FITS/HDF5/NumPy readers/writers |
| `batch.py` | Pipeline runner with chunked processing |
| `pdf.py` | Likelihood math, KDE, PDF utilities |
| `knn.py` | KMCkNN fitting (production backend) |
| `bruteforce.py` | Exhaustive fitting (small grids / reference) |
| `networks.py` | SOM + GNG neural network compression |
| `fitting.py` | Fitter re-exports + `get_fitter()` factory |
| `samplers.py` | Population N(z) + hierarchical MCMC |
| `simulate.py` | Mock survey + synthetic photometry |
| `priors.py` | BPZ-style Bayesian priors + `get_prior()` factory |
| `reddening.py` | Madau IGM attenuation |
| `plotting.py` | Visualization utilities |

All fitters share: `fit()` -> `predict()` -> `fit_predict()` with internal generators for streaming.

### Factory Functions
- `get_transform(config)` — returns configured transform callable
- `get_prior(config)` — returns prior callable (or None for uniform)
- `get_fitter(config, training_data)` — returns configured BruteForce or NearestNeighbors
- `run_pipeline(config, train, test)` — full chunked pipeline with PDFs

## Critical Bugs (All fixed in Phase 01)

All 5 critical/high bugs have been fixed:
1. `simulate.py` — `mag_err()` undefined variable names (fixed)
2. `pdf.py` — `loglike()` mutates input arrays in place (fixed)
3. `knn.py` — custom `feature_map` validation uses undefined `X_train` (fixed)
4. `samplers.py` — `hierarchical_sampler.reset()` clears wrong attributes (fixed)
5. `pdf.py` — `pdfs_summarize()` mutates input via `/=` (fixed)

## Development Rules

### Planning & Tracking
- Plan in `docs/plan/phase_NN.md` (e.g., `phase_01.md`)
- Each task has ID: `PNN_NNN` (e.g., `P01_001` = phase 1, task 1)
- Track in `docs/TODO.md` with per-phase tables
- Record mistakes/instructions in `docs/LESSONS.md`

### Code Style
- `snake_case` everywhere, no camelCase
- English only for code, docs, commits
- No redundant comments restating names
- Review 3-5 similar files before writing new code

### Workflow
- Never work on `main`/`master` directly; use feature branches
- No merge without explicit permission
- Use `uv` for Python dependencies
- Validate at small scale before full execution

## Reference: frankenz4DESI Patterns

The `../frankenz4DESI/` wrapper by Zechang Sun shows production patterns that
have been incorporated into frankenz v0.4.0:
- YAML config system (dataclasses + pyyaml, replacing YACS)
- Multi-format I/O (CSV/FITS/HDF5/NumPy via PhotoData container)
- Factory functions: `get_transform()`, `get_prior()`, `get_fitter()`
- Batch processing with chunking and optional tqdm progress
- HSC 5-band (grizy) photometry pipeline
- KNN-based data-driven prior (deferred to future phase)
