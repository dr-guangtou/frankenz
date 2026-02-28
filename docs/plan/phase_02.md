# Phase 02: Production-Ready API

**Goal**: Add config system, flexible I/O, extracted transforms with factories, and batch processing. Patterns proven in frankenz4DESI, now in the core library.

**Branch**: `phase-02/production-api` (from `main`)

**Breaking changes are fine** — no external users depend on current API.

---

## New Modules

| File | Purpose |
|------|---------|
| `frankenz/config.py` | Dataclass config hierarchy + YAML serde |
| `frankenz/transforms.py` | Transform functions extracted from pdf.py + factory |
| `frankenz/io.py` | PhotoData container + CSV/FITS/HDF5/NumPy readers/writers |
| `frankenz/batch.py` | Pipeline runner with chunked processing |

## Tasks

### Step 0: Planning Documents

| ID | Task | Status |
|----|------|--------|
| P02_000a | Create `docs/plan/phase_02.md` | done |
| P02_000b | Update `docs/TODO.md` with Phase 02 task table | done |

### Step 1: Config System

| ID | Task | Status |
|----|------|--------|
| P02_001 | Create `frankenz/config.py` with dataclass hierarchy | done |
| P02_002 | Implement YAML load/save + `from_dict()`/`to_dict()` | done |
| P02_003 | Write `tests/test_config.py` | done |

### Step 2: Extract Transforms

| ID | Task | Status |
|----|------|--------|
| P02_004 | Extract transforms from `pdf.py` into `frankenz/transforms.py` | done |
| P02_005 | Add `identity()` and `get_transform(config)` factory | done |
| P02_006 | Update imports in `knn.py` and `networks.py` | done |
| P02_007 | Write `tests/test_transforms.py` | done |

### Step 3: I/O Module

| ID | Task | Status |
|----|------|--------|
| P02_008 | Define `PhotoData` dataclass container | done |
| P02_009 | Implement CSV reader/writer | done |
| P02_010 | Implement FITS reader/writer (optional dep: astropy) | done |
| P02_011 | Implement HDF5 reader/writer (optional dep: h5py) | done |
| P02_012 | Implement NumPy reader/writer | done |
| P02_013 | Add `load_data()` / `save_data()` format dispatcher | done |
| P02_014 | Write `tests/test_io.py` | done |

### Step 4: Factory Functions

| ID | Task | Status |
|----|------|--------|
| P02_015 | Add `get_prior(config)` factory to `priors.py` | done |
| P02_016 | Add `get_fitter(config, training_data)` factory to `fitting.py` | done |
| P02_017 | Write `tests/test_factories.py` | done |

### Step 5: Batch Processing

| ID | Task | Status |
|----|------|--------|
| P02_018 | Create `frankenz/batch.py` with `run_pipeline()` | done |
| P02_019 | Add optional tqdm progress tracking | done |
| P02_020 | Write `tests/test_batch.py` | done |

### Step 6: Integration

| ID | Task | Status |
|----|------|--------|
| P02_022 | Update `__init__.py` exports + bump to v0.4.0 | done |
| P02_023 | Update `pyproject.toml` (pyyaml required, optional deps) | done |
| P02_024 | Write `tests/test_pipeline.py` (end-to-end) | done |
| P02_025 | Update docs (TODO, CLAUDE.md) | done |
| P02_026 | Verify all 70+ existing tests still pass | done |

## Implementation Order

```
Step 0 -> Step 1 -> Step 2 -> Step 3 -> Step 4 -> Step 5 -> Step 6
```

Each step produces a testable, committable result.
