# Improved QA Notebook Plan

**Target**: Rewrite `s23b/qa_training_sample.ipynb` with publication-quality figures
following the `s19a_plot.py` style, per-photometry-type analysis, and spec-z source
overlap diagnostics.

**Reference**: `../hsc_photoz/notebook/s19a_plot.py` (7 plotting functions),
`../hsc_photoz/notebook/s19a_study_training.ipynb` (usage patterns)

## Design Principles

1. **Plotting library**: `s23b/s23b_plot.py` — standalone module with reusable plotting
   functions following `s19a_plot.py` patterns. The notebook calls these functions.
2. **Fine 2D histograms**: `plt.hist2d()` with bins=[120, 100], `LogNorm()`, `cmin=2`,
   scatter background, colorbars on all 2D plots.
3. **Scale factor axis**: `a = 1/(1+z)` as primary x-axis with `secondary_xaxis` for z.
4. **Error representation**: `log10(dz/(1+z))` for redshift error y-axis.
5. **Per-source panels**: Side-by-side DESI vs COSMOSWeb (like WIDE vs DUD in s19a).
6. **Highlight overlay**: Red scatter overlay for flagged/outlier subsets.
7. **Registry-driven**: All flux types use the same code path via `FLUX_TYPE_REGISTRY`.

## Part A: `s23b/s23b_plot.py` — Plotting Module

Reusable functions modeled on `s19a_plot.py`. Each function takes axes, data arrays,
and styling kwargs. Returns the axes object.

### Functions

```
plot_scale_dz(scale, logdz, ax, ...)
    Scale factor vs log10(dz/(1+z)). 2D histogram + scatter + secondary z axis.
    Direct adaptation of s19a_plot.plot_scale_dz.

plot_mag_dz(mag, logdz, ax, ...)
    Magnitude vs log10(dz/(1+z)). 2D histogram + scatter.
    Direct adaptation of s19a_plot.plot_mag_dz.

plot_mag_z(mag, z, ax, ...)
    Magnitude vs redshift. 2D histogram + scatter.
    Direct adaptation of s19a_plot.plot_mag_z.

plot_color_z(color, z, ax, ...)
    Color vs redshift. 2D histogram + scatter.
    Adaptation of s19a_plot.plot_color_z (takes pre-computed color, not two mags).

plot_color_color(color_x, color_y, ax, ...)
    Color-color diagram. 2D histogram + scatter.
    New function (current notebook has this as hexbin).

plot_z_hist_by_source(redshift, sources, ax, ...)
    Stacked/overlaid redshift histograms split by spec-z source.

plot_z_compare(z_1, z_2, ax, ...)
    z_1 vs z_2 comparison (e.g., DESI vs COSMOSWeb for dual objects).
    Adaptation of s19a_plot.plot_z_compare_point.

plot_completeness_heatmap(completeness_matrix, flux_types, bands, ax, ...)
    Annotated heatmap of per-band completeness by flux type.
```

All `hist2d`-based functions share these defaults:
- `bins=[120, 100]`, `cmin=2`, `cmap='viridis'`, `norm=LogNorm()`
- Scatter underlay: `s=2, alpha=0.2`
- Grid: `linestyle='--', linewidth=2, alpha=0.6`
- Optional `highlight` mask for red overlay
- Returns colorbar handle for external layout control

### Style Constants

```python
PLOT_DEFAULTS = {
    "bins": [120, 100],
    "cmin": 2,
    "cmap": "viridis",
    "scatter_s": 2,
    "scatter_alpha": 0.2,
    "grid_ls": "--",
    "grid_lw": 2,
    "grid_alpha": 0.6,
    "label_fontsize": 25,
    "text_fontsize": 30,
}

SOURCE_STYLE = {
    "DESI_DR1":           {"cmap": "viridis", "color": "steelblue", "label": "DESI"},
    "COSMOSWeb2025_v1":   {"cmap": "inferno", "color": "orangered", "label": "COSMOSWeb"},
}
```

## Part B: Notebook Structure

### Section 0: Configuration (cells 0-2)
Same as current: paths, bands, registry, parameters. Add:
- `ALL_FLUX_TYPES` list (all 8 types) for completeness analysis
- `SELECTED_FLUX_TYPES` for detailed analysis (default: cmodel, gaap, convolved)
- Plotting defaults imported from `s23b_plot`

### Section 1: Imports + Helpers (cells 3-4)
Same as current. Add `from s23b_plot import *`.

### Section 2: Data Loading (cells 5-6)
Same as current.

### Section 3: Redshift QA — Flux-Independent (cells 7-14)
Improved from current. Key additions:

**Cell 7**: Extract redshift arrays (same as current cell 9).

**Cell 8**: Redshift distribution by source — same as current but add:
- `ax1`: Linear-scale histograms by source (keep)
- `ax2`: Log-scale y-axis version for tail visibility

**Cell 9**: Redshift error quality table (same as current cell 11).

**Cell 10**: NEW — **Scale factor vs log10(dz/(1+z)) by source** (the key s19a-style plot)
- 2x1 panel: DESI (left) vs COSMOSWeb (right)
- Uses `plot_scale_dz()` with secondary redshift axis
- COSMOSWeb only (DESI has sentinel zerr = -1)
- For DESI, show just redshift distribution (no zerr info)

**Cell 11**: NEW — **Spec-z source overlap analysis**
- Venn-style count: DESI-only, COSMOSWeb-only, Dual
- For 554 dual objects: `plot_z_compare()` — DESI z vs COSMOSWeb z
- Statistics: median |dz|, outlier fraction, agreement rate

**Cell 12**: Object type breakdown (improved from current cell 12).

**Cell 13**: Cross-validation fold balance (same as current cell 13).

**Cell 14**: NEW — **Spatial distribution**
- RA vs Dec colored by spec-z source
- RA vs Dec colored by redshift (heat)

### Section 4: Completeness Overview — All 8 Flux Types (cells 15-17)

**Cell 15**: Completeness heatmap for ALL 8 flux types (expanded from current 2).
Uses `plot_completeness_heatmap()`. 8 rows x 5 bands.

**Cell 16**: Per-band NaN count table for all 8 types.

**Cell 17**: Recommendation text cell — which flux types are suitable for frankenz.

### Section 5: Per-Flux-Type Photometry QA (cells 18-30)
Loop over `SELECTED_FLUX_TYPES`. For each type, produce:

**Cell 18+i*4**: **Magnitude-redshift relations** (s19a-style, per source)
- 2x3 panel figure (13x19 inches, like s19a `visual_training_cat` fig2):
  - Row 1: i-mag vs z (DESI left, COSMOSWeb right) using `plot_mag_z()`
  - Row 2: g-r color vs z (DESI left, COSMOSWeb right) using `plot_color_z()`
  - Row 3: r-y color vs z (DESI left, COSMOSWeb right) using `plot_color_z()`

**Cell 19+i*4**: **Redshift error relations** (s19a-style, COSMOSWeb only)
- 2x2 panel (13x13 inches, like s19a `visual_training_cat` fig1):
  - Top-left: scale factor vs log10(dz/(1+z)) using `plot_scale_dz()`
  - Top-right: i-mag vs log10(dz/(1+z)) using `plot_mag_dz()`
  - Bottom-left: g-r color vs log10(dz/(1+z))
  - Bottom-right: color-color (g-r vs i-z)

**Cell 20+i*4**: **SNR and magnitude distributions** (improved from current)
- 5-band SNR histograms (keep)
- 5-band magnitude histograms (keep)
- Add median lines and source-split overlays

**Cell 21+i*4**: **Redshift in magnitude bins** (DEmP-style magbin)
- Same as current but source-split overlays (DESI vs COSMOSWeb)

### Section 6: Cross-Flux-Type Comparison (cells 31-34)

**Cell 31**: i-band magnitude comparison — 2D histogram (not hexbin)
- Use `plot_z_compare()`-style for mag_a vs mag_b
- With colorbar, residual stats

**Cell 32**: Color comparison — (g-r) CModel vs GaaP, (i-z) CModel vs GaaP

**Cell 33**: NEW — **Flux ratio analysis**
- Per-band: histogram of flux_gaap / flux_cmodel
- Identify systematic offsets

**Cell 34**: NEW — **Magnitude-dependent residual**
- i-mag vs (mag_gaap - mag_cmodel) as 2D histogram

### Section 7: Quality Cuts (cells 35-39)
Same as current but with improved figures:

**Cell 35**: Quality cut function (same).

**Cell 36**: Apply cuts per flux type with attrition table (same).

**Cell 37**: Attrition waterfall (same).

**Cell 38**: Pre/post-cut redshift distribution (same).

**Cell 39**: NEW — **Post-cut mag-z and color-z** (confirmation that cuts are reasonable)

### Section 8: Extinction Correction + KDE Bandwidth (cells 40-42)
Same as current.

### Section 9: Export (cells 43-46)
Same as current.

### Section 10: Summary (cell 47)
Same as current.

## Figure Inventory

Total: ~25-30 figures (vs. 17 in current notebook)

| # | Figure | Style | Source |
|---|--------|-------|--------|
| 1 | Redshift distribution by source | Histogram + CDF | Improved |
| 2 | Scale factor vs log10(dz/(1+z)) by source | 2D hist + scatter | NEW |
| 3 | Spec-z source overlap (dual objects z comparison) | 2D hist | NEW |
| 4 | Object type distribution | Bar chart | Same |
| 5 | Cross-validation fold balance | Bar chart | Same |
| 6 | Spatial distribution (RA/Dec) | Scatter | NEW |
| 7 | Completeness heatmap (all 8 types) | Heatmap | Expanded |
| 8-10 | Mag-z + color-z panels per flux type (x3 types) | 2x3 panel | NEW |
| 11-13 | Zerr relations per flux type (x3 types) | 2x2 panel | NEW |
| 14-16 | SNR distribution per flux type (x3 types) | 5-band hist | Improved |
| 17-19 | Magnitude distribution per flux type (x3 types) | 5-band hist | Same |
| 20-22 | z in mag bins per flux type (x3 types) | Multi-panel | Improved |
| 23 | Cross-flux i-mag comparison | 2D hist | Improved |
| 24 | Cross-flux color comparison | 2D hist | Improved |
| 25 | Flux ratio analysis | Histogram | NEW |
| 26 | Magnitude-dependent residual | 2D hist | NEW |
| 27-29 | Attrition waterfall per flux type | Waterfall | Same |
| 30 | Pre/post-cut redshift distribution | Overlay hist | Same |
| 31 | Post-cut mag-z confirmation | 2D hist | NEW |
| 32 | dN/dz distribution | Histogram | Same |

## Implementation Order

1. Write `s23b/s23b_plot.py` with all 8 plotting functions
2. Rewrite notebook sections 0-4 (config, imports, loading, redshift QA, completeness)
3. Rewrite section 5 (per-flux-type QA with new figure style)
4. Rewrite section 6 (cross-flux comparison)
5. Sections 7-10 (quality cuts, export) — mostly keep, minor improvements
6. Test run notebook end-to-end

## Key Technical Notes

- All `hist2d` plots need `range=` parameter to avoid autoscaling artifacts with outliers
- `LogNorm()` requires `cmin >= 1` (already set to 2 in defaults)
- Scale factor transform: `a = 1/(1+z)`, inverse: `z = (1-a)/a`
- Secondary axis: `ax.secondary_xaxis('top', functions=(forward, inverse))`
- DESI zerr is always -1.0 — skip zerr plots for DESI, or plot separately
- GAaP g-band has 15% NaN — need explicit handling in color computations
- `logmstar` and `sfr` columns need `json.loads()` parsing if used
