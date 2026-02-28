# Phase 01: Codebase Stabilization and Documentation

**Goal**: Fix critical bugs, clean up code hygiene, and produce accurate documentation that reflects frankenz as a supervised learning method (not template fitting).

**Branch**: `phase-01/stabilize` (from `master`)

---

## Tasks

### P01_001: Fix critical bug — `mag_err()` undefined variables
- **File**: `frankenz/simulate.py:86-90`
- **Action**: Replace `m` -> `mag`, `mlim` -> `maglim`, `sigmadet` -> `sigdet`
- **Verify**: Write a quick smoke test calling `mag_err(22.0, 25.0)`

### P01_002: Fix critical bug — `loglike()` in-place mutation
- **File**: `frankenz/pdf.py:309-311`
- **Action**: Copy `data`, `data_err`, `data_mask` at function entry before the cleaning step
- **Verify**: Call `loglike()` twice on the same array, confirm values unchanged

### P01_003: Fix high bug — `feature_map` validation in NearestNeighbors
- **File**: `frankenz/knn.py:131-139`
- **Action**: Replace `X_train` -> `self.models`, `Xe_train` -> `self.models_err`. Change bare `except:` to `except Exception:`
- **Verify**: Pass a custom lambda as `feature_map`, confirm no misleading error

### P01_004: Fix high bug — `hierarchical_sampler.reset()` wrong attributes
- **File**: `frankenz/samplers.py:336-341`
- **Action**: Reset `self.samples` and `self.samples_lnp` only. Remove `samples_prior` and `samples_counts`.
- **Verify**: Instantiate sampler, add samples, call `reset()`, confirm `self.samples == []`

### P01_005: Fix high bug — `pdfs_summarize()` in-place mutation
- **File**: `frankenz/pdf.py:984`
- **Action**: Replace `pdfs /= ...` with `pdfs = pdfs / ...`
- **Verify**: Pass array, call `pdfs_summarize()`, confirm original array unchanged

### P01_006: Add convergence guard to `_loglike_s()` iteration
- **File**: `frankenz/pdf.py:198-223`
- **Action**: Add `max_iter=100` loop counter; emit `warnings.warn()` on non-convergence
- **Verify**: Construct degenerate case (all-zero model), confirm no infinite loop

### P01_007: Replace bare `except:` with `except Exception:`
- **Files**: `knn.py:137`, `pdf.py:1022`, `samplers.py:176`
- **Action**: Mechanical replacement
- **Verify**: `grep -rn 'except:' frankenz/` returns zero hits

### P01_008: Fix duplicate `import warnings`
- **Files**: `pdf.py`, `simulate.py`, `knn.py`, `bruteforce.py`, `networks.py`, `samplers.py`
- **Action**: Remove duplicate `import warnings` lines
- **Verify**: Each file has exactly one `import warnings`

### P01_009: Fix docstring errors in `priors.py`
- **File**: `frankenz/priors.py:188-203`
- **Action**: Correct parameter name documentation (z/t swap, mbounds/zbounds duplicate)
- **Verify**: Read docstrings, confirm they match function signature

### P01_010: Drop Python 2 compatibility layer
- **All source files**
- **Action**: Remove `from __future__ import ...`, `import six`, `from six.moves import range`. Update `setup.py` to require Python >= 3.9
- **Verify**: `grep -rn 'six' frankenz/` returns zero hits; `python3 -c "import frankenz"` succeeds

### P01_011: Replace wildcard imports with explicit imports
- **Files**: `bruteforce.py:20`, `knn.py:23`, `networks.py:26`
- **Action**: Replace `from .pdf import *` with explicit names used in each module
- **Verify**: `grep -rn 'import \*' frankenz/` returns zero hits

### P01_012: Update `docs/frankenz_usage.md` — correct framing
- **Action**: Rewrite overview to describe frankenz as supervised learning, not template fitting. Clarify that the "model grid" is typically a labeled training set with spec-z. Keep template-based workflow as one option. Add real-data workflow section based on frankenz4DESI patterns.
- **Verify**: No mention of "template fitting" as the primary description

### P01_013: Update `docs/frankenz_review.md` — mark fixed bugs
- **Action**: After fixing P01_001 through P01_011, mark each resolved bug with status and commit hash
- **Verify**: Summary table reflects remaining vs fixed counts

### P01_014: Run demo notebooks as integration test
- **Action**: Execute notebooks 1-4 in `demos/` to confirm nothing is broken by the changes
- **Verify**: All cells execute without error; outputs are qualitatively sensible

---

## Acceptance Criteria

- [ ] All 5 critical/high bugs fixed and verified
- [ ] No bare `except:`, no `six`, no duplicate imports, no wildcard imports
- [ ] Documentation accurately describes frankenz as supervised learning
- [ ] Demo notebooks 1-4 pass without errors
- [ ] All changes on `phase-01/stabilize` branch, not `master`
