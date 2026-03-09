# Session Handover: S23b Parameter Sweep (2026-03-06)

## What Was Done

Completed the full S23b Frankenz parameter sweep (Phases A through E) to optimize photo-z performance.

### Infrastructure Built
1. **`s23b/run_sweep.py`** (new, 795 lines) — Full sweep automation with 4 phases
2. **`s23b/run_photoz_s23b.py`** — Added `--save-intermediates`, `run_single_fold_with_intermediates()`, `resweep_bandwidth()`, vectorized `compute_pit()` and `compute_crps()`
3. **`frankenz/batch.py`** — Fixed critical bug: `lprob_kwargs` (free_scale, ignore_model_err, dim_prior) were not being passed to `fit_predict()`

### Sweep Results

| Phase | What | Key Finding |
|-------|------|-------------|
| A | 16 configs on fold 2 | `snrcap_aggressive` [50]*5 best sNMAD=0.079 |
| C | 2 configs x 8 folds | aggressive sNMAD=0.080 (vs baseline 0.095) |
| D | 9 bandwidth configs | bw(0.10, 0.05) gives cov68=0.709 |
| E | Final 8-fold run | See below |

### Final Optimized Config
```yaml
snr_cap: [50, 50, 50, 50, 50]
bw_frac: 0.10
bw_floor: 0.05
k_point: 20  # unchanged
k_tree: 25   # unchanged
```

### Final Metrics (443k objects, 8-fold CV)

| Estimator | sNMAD | f_out | bias |
|-----------|-------|-------|------|
| z_best | 0.081 | 0.325 | -0.006 |
| z_median | 0.088 | 0.334 | -0.004 |
| z_mode | 0.090 | 0.363 | -0.012 |
| z_mean | 0.130 | 0.393 | +0.005 |

Coverage: 68%=0.709, 95%=0.888

Per-source (z_best): DESI sNMAD=0.027, COSMOSWeb sNMAD=0.250

### vs Baseline

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| sNMAD (z_best) | 0.095 | 0.081 | -15% |
| f_out | 0.36 | 0.325 | -10% |
| coverage_68 | 0.24 | 0.709 | +195% |
| coverage_95 | 0.40 | 0.888 | +122% |

## Key Insights

1. **z_mean is unreliable with wide KDE bandwidth** — broader PDFs pull z_mean toward intermediate redshifts. Use z_best (MAP) or z_median for point estimates.
2. **Bandwidth is orthogonal to point estimates** — z_mode and z_best are identical across all bandwidth configs. Only PDF width changes.
3. **Aggressive S/N cap helps** — capping all bands at S/N=50 (vs 100/100/100/80/50 baseline) improves sNMAD by 15%. More error dilation = better neighbor matching for bright objects.
4. **COSMOSWeb performance is training-sample-limited** — sNMAD=0.250 reflects the faint/high-z regime where the training sample is sparse. This won't improve from parameter tuning alone.

## What's NOT Done

1. **Not committed** — All changes are uncommitted on `main`. Need to create feature branch and commit.
2. **QA figures not regenerated** — The full 11 QA figures haven't been generated with the optimized config yet. Need to run `run_photoz_s23b.py` pointing at `sweep/final/` results.
3. **Config YAML not updated** — `frankenz_s23b_config.yaml` still has baseline values.
4. **Intermediates are large** — ~3.5 GB in `sweep/final/intermediates/` (8 folds x 444 MB). Can be deleted after final figures are generated.

## Output Files

```
s23b/output/undeblended/sweep/
  phase_a_results.csv          # Phase A screening results
  phase_c_results.csv          # Phase C full CV results
  phase_d_results_snrcap_aggressive.csv  # Phase D bandwidth sweep
  baseline/                    # Phase C baseline intermediates
  snrcap_aggressive/           # Phase C aggressive intermediates
  final/                       # Phase E final run
    best_params.json
    final_config.yaml
    prepared/                  # Fold train/test HDF5 + auxiliary.npz
    results/                   # Per-fold NPZ (pdfs, point estimates, CI)
    intermediates/             # Per-fold neighbor data (~444 MB each)
```
