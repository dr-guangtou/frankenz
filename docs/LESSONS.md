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

## frankenz4DESI Reference

- Zechang Sun's wrapper shows the production workflow: YAML config -> HDF5 load -> luptitude transform -> KDTree ensemble -> batch predict -> HDF5 output.
- The wrapper has a typo in `model.py:26`: `get_transfrom` instead of `get_transform`.
- KNN-based data-driven prior (using neighbor redshift distribution) is a useful alternative to flat or BPZ priors.
