# Frankenz Lessons Learned

## Algorithmic Understanding

- **Frankenz is supervised learning, not template fitting.** The "model grid" is a labeled training set (photometry + spec-z). Template grids are one way to construct training data, but real spec-z catalogs work equally well and are preferred for production.
- **`free_scale=True` vs `False` is a fundamental choice**, not just a parameter. `True` = color-only matching (needs explicit prior). `False` = magnitude matching (implicit prior from training density). For real training data, `False` is usually correct.
- **Magnitude likelihoods implicitly encode the prior** from the training set's redshift-magnitude distribution. This is why explicit priors are less important with representative training data.

## Code Patterns

- **In-place mutation was the most dangerous pattern in this codebase.** Both `loglike()` and `pdfs_summarize()` used to silently modify their inputs via numpy array aliasing. **Fixed in Phase 01**: `loglike()` now copies inputs at entry; `pdfs_summarize()` uses `pdfs = pdfs / ...` instead of `pdfs /= ...`.
- **Python 2 compat removed in Phase 01.** All `from __future__`, `import six`, `from six.moves` eliminated. The `six.iteritems()` calls in networks.py were replaced with `.items()`. Setup.py now requires Python >= 3.9.
- **`Npoints=5e4` float default** caused `TypeError` with Python 3's strict `int` requirement in `np.linspace`. Changed to `50000`.
- **Bare `except:` is never acceptable** — it catches `KeyboardInterrupt` and `SystemExit`. All 13 instances replaced with `except Exception:`.
- **`_loglike_s()` convergence loop had no iteration guard** — could spin forever with pathological inputs. Added `max_iter=100`.
- **Generator-based fitting (`_fit`, `_predict`, `_fit_predict`)** is a deliberate memory optimization. The public methods wrap generators and handle accumulation + progress printing. Do not break this pattern.
- **The `logprob()` wrapper** applies a flat prior by default. For custom priors, pass `lprob_func=` to the fitter, not to `logprob()` directly.

## Phase 02: Production API Design

- **Dataclass config > YACS/argparse.** Using `@dataclass` with `from_dict()`/`to_dict()` is simpler than YACS (which frankenz4DESI used) and avoids the YACS dependency. YAML serde via `pyyaml` is sufficient; no need for schema validation libraries.
- **Re-export for backward compatibility.** When extracting `magnitude()`/`luptitude()` from `pdf.py` to `transforms.py`, re-exporting via `pdf.__all__` preserves all existing import paths. This avoids breaking user code while keeping the canonical location in the new module.
- **PhotoData container simplifies I/O dispatch.** A single dataclass holding `flux, flux_err, mask, redshifts, redshift_errs, band_names, metadata` replaces scattered arrays. The `validate()` method catches shape mismatches early. Format dispatch via file extension (`.csv`, `.fits`, `.hdf5`, `.npz`) keeps the API simple.
- **Factory functions centralize configuration.** `get_transform(config)`, `get_prior(config)`, `get_fitter(config, training_data)` each read from the config hierarchy and return ready-to-use objects. This eliminates the error-prone pattern of manually wiring config fields to constructor arguments.
- **Chunked batch processing avoids OOM.** `run_pipeline()` processes test data in chunks (default 1000 objects), accumulating PDFs. This is the proper fix for ISSUE-08 (memory scaling) rather than just warning about it.
- **Optional dependencies via extras.** `pyproject.toml` extras (`fits`, `hdf5`, `progress`, `all`) with lazy imports at call sites. Never import `astropy`/`h5py`/`tqdm` at module level — import inside the function that needs them and raise `ImportError` with a helpful message.

## Demo Notebook Modernization

- **Keep pickle for backward compatibility.** Notebooks 2-4 load pickle files created by Notebook 1. Replacing `dill.dump` with `pickle.dump` is safe (the objects are plain dicts/MockSurvey), but adding PhotoData I/O as an *additional* section preserves the existing workflow.
- **Show old and new API side by side.** Rather than replacing the direct API examples with config-driven ones, adding the new API as an additional section at the end lets users compare approaches and migrate at their own pace.
- **`scipy.misc.logsumexp` is long gone.** The try/except fallback pattern (`try: from scipy.special import logsumexp; except: from scipy.misc import logsumexp`) is dead code since scipy 1.0. Just import from `scipy.special` directly.

## frankenz4DESI Reference

- Zechang Sun's wrapper shows the production workflow: YAML config -> HDF5 load -> luptitude transform -> KDTree ensemble -> batch predict -> HDF5 output.
- The wrapper has a typo in `model.py:26`: `get_transfrom` instead of `get_transform`.
- KNN-based data-driven prior (using neighbor redshift distribution) is a useful alternative to flat or BPZ priors.
