# Frankenz TODO

## Phase 01: Codebase Stabilization and Documentation

**Branch**: `phase-01/stabilize`

### Step 1: Test Infrastructure

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_T01 | Create `tests/conftest.py` with session-scoped MockSurvey fixture | done | 200 train + 50 test, seed=42 |
| P01_T02 | Create `pyproject.toml` with markers (slow, mutation, regression) | done | |

### Step 2: Write "Before" Tests for 5 Bugs

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_T03 | Write `test_pdf_mutation.py` — loglike + pdfs_summarize mutation tests | done | 5 tests |
| P01_T04 | Write `test_simulate.py` — mag_err() NameError test | done | 8 tests |
| P01_T05 | Write `test_knn.py` — custom feature_map acceptance test | done | 5 tests |
| P01_T06 | Write `test_samplers.py` — reset() clears self.samples test | done | 7 tests |

### Step 3: Fix 5 Critical/High Bugs

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_001 | Fix `mag_err()` undefined variables | done | m->mag, mlim->maglim, sigmadet->sigdet |
| P01_002 | Fix `loglike()` in-place mutation | done | np.array() copies at entry |
| P01_003 | Fix `feature_map` validation in KNN | done | X_train->self.models, except Exception |
| P01_004 | Fix `hierarchical_sampler.reset()` | done | Clear self.samples + self.samples_lnp only |
| P01_005 | Fix `pdfs_summarize()` in-place mutation | done | pdfs = pdfs / ... instead of pdfs /= ... |

### Step 4: Write Remaining Unit Tests

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_T07 | Write `test_pdf_math.py` — loglike, gauss_kde, gaussian | done | 11 tests |
| P01_T08 | Write `test_pdf_utilities.py` — PDFDict, mag/lupt roundtrips | done | 10 tests |
| P01_T09 | Write `test_priors.py` — pmag, bpz_pt_m, bpz_pz_tm | done | 9 tests |
| P01_T10 | Write `test_reddening.py` — madau_teff | done | 6 tests |

### Step 5: Fix Medium-Severity Issues

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_006 | Add convergence guard to `_loglike_s()` | done | max_iter=100 |
| P01_007 | Replace bare `except:` clauses | done | All 13 occurrences across all modules |
| P01_008 | Fix duplicate `import warnings` | done | All 10 modules |
| P01_009 | Fix docstring errors in `priors.py` | done | z/t params, mbounds/zbounds |

### Step 6: Code Hygiene

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_010 | Drop Python 2 / `six` compatibility | done | All modules + setup.py, require >=3.9 |
| P01_011 | Replace wildcard imports with explicit | done | bruteforce.py, knn.py, networks.py |

### Extra: Python 3 Compatibility Fix

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_E01 | Fix `Npoints=5e4` float default in `load_survey()` | done | Changed to int `50000` |

### Step 7: Integration Tests + Full Suite

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_T11 | Write `test_bruteforce.py` — fit/predict, GoF metrics | done | 3 tests @slow |
| P01_T12 | Write `test_knn.py` slow tests — KNN fit_predict | done | 2 tests @slow |
| P01_T13 | Write `test_integration.py` — full pipeline smoke tests | done | 3 tests @slow |
| P01_T14 | Run full pytest suite — all tests pass | done | 70 tests, 1.68s |

### Step 8: Documentation Updates

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_012 | Update usage docs (supervised learning framing) | done | Reframed overview, pipeline, steps, pitfalls; added real-data workflow |
| P01_013 | Update review docs (mark fixed bugs) | done | 14/18 issues marked fixed, summary table updated |

### Step 9: Validate with Demo Notebooks

| ID | Task | Status | Notes |
|----|------|--------|-------|
| P01_014 | Run demo notebooks 1-4 as integration test | deferred | Requires manual inspection |

---

## Phase 02: Production-Ready API (Planned)

_Incorporate frankenz4DESI patterns: config system, HDF5 I/O, batch processing, data-driven priors._

## Phase 03: HSC Pipeline (Planned)

_End-to-end pipeline for HSC survey photo-z with training/validation/prediction workflow._
