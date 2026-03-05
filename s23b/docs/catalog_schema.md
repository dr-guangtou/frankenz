# HSC S23b Deep Training Catalog Schema

**File**: `hsc_s23b_deep_matched_train_SFR_v3.fits`
**Objects**: 453,652 rows, 343 columns
**Bands**: g, r, i, z, y (5-band HSC)
**Flux units**: nJy (AB ZP = 31.4; `m_AB = -2.5 * log10(flux_nJy) + 31.4`)

## Column Layout

Each band block is interleaved with extinction (`a_{b}`) at the start.
The pattern repeats for g, r, i, z, y. Columns are listed once using `{b}` as band placeholder.

### 1. Astrometry & Identification (10 columns)

| Column | FITS Format | Description |
|--------|-------------|-------------|
| `object_id` | K (int64) | HSC pipeline object identifier |
| `tract` | I (int16) | Sky tract index |
| `patch` | I (int16) | Patch within tract |
| `ra` | D (float64) | Right ascension (degrees) |
| `dec` | D (float64) | Declination (degrees) |

Each has a corresponding `*_isnull` column (boolean). All `_isnull` columns are `True` in this catalog (unusable).

### 2. Galactic Extinction (5 columns)

| Column | Format | Description |
|--------|--------|-------------|
| `a_{b}` | E (float32) | Galactic extinction A_b in magnitudes, per band |

Bands: `a_g`, `a_r`, `a_i`, `a_z`, `a_y`. Plus 5 `_isnull` columns.

### 3. Photometric Flux (8 types x 5 bands = 40 flux + 40 fluxerr columns)

All flux and fluxerr columns are float32 (FITS format `E`), units nJy.

| Flux Type | Flux Column | Fluxerr Column | Description |
|-----------|-------------|----------------|-------------|
| **cmodel** | `{b}_cmodel_flux` | `{b}_cmodel_fluxerr` | Composite model (de Vaucouleurs + exponential) |
| **cmodel_exp** | `{b}_cmodel_exp_flux` | `{b}_cmodel_exp_fluxerr` | Exponential component of CModel |
| **cmodel_dev** | `{b}_cmodel_dev_flux` | `{b}_cmodel_dev_fluxerr` | De Vaucouleurs component of CModel |
| **psf** | `{b}_psfflux_flux` | `{b}_psfflux_fluxerr` | PSF model flux (point-source optimal) |
| **gaap_optimal** | `{b}_gaapflux_1_15x_optimal_flux` | `{b}_gaapflux_1_15x_optimal_fluxerr` | GAaP (Gaussian Aperture and PSF) optimal |
| **gaap_psf** | `{b}_gaapflux_1_15x_psfflux_flux` | `{b}_gaapflux_1_15x_psfflux_fluxerr` | GAaP PSF-matched |
| **convolved** | `{b}_convolvedflux_3_15_flux` | `{b}_convolvedflux_3_15_fluxerr` | PSF-convolved aperture (3.15 arcsec) |
| **undeblended** | `{b}_undeblended_convolvedflux_3_15_flux` | `{b}_undeblended_convolvedflux_3_15_fluxerr` | Undeblended convolved aperture |

Each flux and fluxerr column has a corresponding `_isnull` column (80 total).

#### Completeness (% positive and finite per band)

| Flux Type | g | r | i | z | y | Notes |
|-----------|---|---|---|---|---|-------|
| cmodel | 98.3 | 99.0 | 99.2 | 99.1 | 98.5 | NaN: 3k-7k |
| cmodel_exp | 98.4 | 99.0 | 99.3 | 99.1 | 98.5 | Similar to cmodel |
| cmodel_dev | 98.3 | 99.0 | 99.2 | 99.1 | 98.5 | Similar to cmodel |
| psf | 99.1 | 99.7 | 99.9 | 99.8 | 99.4 | Fewest NaN (121-1595) |
| gaap_optimal | 84.6 | 96.0 | 99.2 | 98.0 | 92.0 | g-band worst (66k NaN) |
| gaap_psf | 99.0 | 99.7 | 99.9 | 99.8 | 99.3 | Similar to psf |
| convolved | 99.2 | 99.7 | 99.9 | 99.8 | 99.4 | |
| undeblended | 99.1 | 99.7 | 99.9 | 99.8 | 99.4 | |

### 4. Error-Only Photometry (2 types x 5 bands = 10 columns)

These have fluxerr but **no matching flux column**.

| Column | Description |
|--------|-------------|
| `{b}_apertureflux_15_fluxerr` | 1.5 arcsec aperture flux error |
| `{b}_undeblended_apertureflux_15_fluxerr` | Undeblended 1.5 arcsec aperture flux error |

Plus 10 `_isnull` columns.

### 5. Shape Measurements (9 x 5 bands = 45 columns)

SDSS-style adaptive moments per band. All float32.

| Column | Description |
|--------|-------------|
| `{b}_sdssshape_shape11` | Second moment M_xx |
| `{b}_sdssshape_shape22` | Second moment M_yy |
| `{b}_sdssshape_shape12` | Second moment M_xy |
| `{b}_sdssshape_shape11err` | Error on M_xx |
| `{b}_sdssshape_shape22err` | Error on M_yy |
| `{b}_sdssshape_shape12err` | Error on M_xy |
| `{b}_sdssshape_psf_shape11` | PSF model M_xx |
| `{b}_sdssshape_psf_shape22` | PSF model M_yy |
| `{b}_sdssshape_psf_shape12` | PSF model M_xy |

Plus 45 `_isnull` columns.

### 6. Quality Flags (3 x 5 bands = 15 columns)

All boolean. **All values are True in this catalog** (unusable for quality filtering).

| Column | Description |
|--------|-------------|
| `{b}_psfflux_flag` | PSF flux measurement flag |
| `{b}_cmodel_flag` | CModel flux measurement flag |
| `{b}_convolvedflux_3_15_flag` | Convolved flux measurement flag |

Plus 15 `_isnull` columns.

### 7. Magnitude Offsets (5 columns)

| Column | Format | Description |
|--------|--------|-------------|
| `{b}_mag_offset` | E (float32) | Per-band magnitude offset |

Plus 5 `_isnull` columns.

### 8. Spectroscopic Redshift (7 columns)

| Column | Format | Description |
|--------|--------|-------------|
| `ra_specz` | E (float32) | Spec-z source right ascension |
| `dec_specz` | E (float32) | Spec-z source declination |
| `redshift` | E (float32) | Spectroscopic redshift |
| `redshift_err` | E (float32) | Redshift error (see caveats below) |
| `specz_id` | K (int64) | Spec-z source object ID |
| `specz_sources` | 100A (string) | Source catalog name |
| `object_type` | 20A (string) | Object classification |

#### Spectroscopic Source Breakdown

| Source | Count | z Median | z Max | zerr Behavior |
|--------|-------|----------|-------|---------------|
| COSMOSWeb2025_v1 | 310,010 | 1.26 | 7.0 | 77% good (0 < zerr < 1); 23% invalid (zerr >= 1 or zerr = 0) |
| DESI_DR1 | 143,088 | 0.67 | 6.65 | **Always -1.0** (sentinel, no real errors) |
| Dual (both sources) | 554 | 0.54 | 3.45 | Ultra-precise (median zerr = 0.0007) |

**Warning**: `redshift_err` is NOT uniformly reliable. DESI entries use -1.0 as sentinel. COSMOSWeb has a 23% invalid rate. Do not use raw zerr for KDE bandwidth without filtering.

#### Object Type Distribution

| Type | Count | Fraction | Description |
|------|-------|----------|-------------|
| G | 371,987 | 82.0% | Galaxy |
| Q | 32,994 | 7.3% | QSO/AGN |
| G/G | 26,770 | 5.9% | Galaxy (dual-source, both say galaxy) |
| Q/Q | 9,437 | 2.1% | QSO (dual-source, both say QSO) |
| G/Q | 6,266 | 1.4% | Galaxy/QSO disagreement |
| Q/G | 5,644 | 1.2% | QSO/Galaxy disagreement |
| S | 369 | 0.08% | Star |
| S/G, G/S, S/S, ... | ~185 | <0.05% | Star-related classifications |

### 9. Catalog Metadata (6 columns)

| Column | Format | Description |
|--------|--------|-------------|
| `flag_homogeneous` | L (bool) | Homogeneous sample flag (all True) |
| `objectIndex` | E (float32) | Internal index (always 2.0; unclear purpose) |
| `objectIndexSources` | K (int64) | Source index mapping |
| `logmstar` | 100A (string) | Stellar mass (log10 M_sun); **JSON-string array** (e.g., `'[9.334]'`) |
| `sfr` | 100A (string) | Star formation rate; **JSON-string array** |
| `sample_crossval` | K (int64) | Cross-validation fold assignment (integer) |

## Column Count Summary

| Category | Data Cols | isnull Cols | Total |
|----------|-----------|-------------|-------|
| Astrometry & ID | 5 | 5 | 10 |
| Extinction | 5 | 5 | 10 |
| Flux (8 types x 5 bands) | 80 | 80 | 160 |
| Error-only flux (2 types x 5 bands) | 10 | 10 | 20 |
| Shape (9 params x 5 bands) | 45 | 45 | 90 |
| Flags (3 x 5 bands) | 15 | 15 | 30 |
| Magnitude offsets | 5 | 5 | 10 |
| Spectroscopic redshift | 7 | 0 | 7 |
| Catalog metadata | 6 | 0 | 6 |
| **Total** | **178** | **165** | **343** |

## Known Data Issues

1. **All `_isnull` columns are True**: 165 columns are uniformly True, providing no quality information. Quality must be assessed from data values (NaN, positive, finite).
2. **All flag columns are True**: The 15 per-band quality flags (`psfflux_flag`, `cmodel_flag`, `convolvedflux_3_15_flag`) are all True and cannot be used for filtering.
3. **DESI zerr = -1.0 sentinel**: 143k objects have no usable redshift error.
4. **COSMOSWeb zerr**: ~23% have zerr >= 1.0 or zerr = 0 (invalid).
5. **logmstar/sfr as JSON strings**: Stored as string representations of Python lists (e.g., `'[9.334]'`), require `json.loads()` to parse.
6. **GAaP g-band incompleteness**: gaap_optimal has only 84.6% positive flux in g-band (66k NaN), significantly worse than other types.
7. **objectIndex**: Always 2.0 (float32). Purpose unknown; likely a catalog construction artifact.

## Frankenz Column Mapping

For use with `frankenz.io.read_fits()` or the QA notebook `FLUX_TYPE_REGISTRY`:

```python
FLUX_TYPE_REGISTRY = {
    "cmodel":      ("{b}_cmodel_flux",                           "{b}_cmodel_fluxerr"),
    "cmodel_exp":  ("{b}_cmodel_exp_flux",                       "{b}_cmodel_exp_fluxerr"),
    "cmodel_dev":  ("{b}_cmodel_dev_flux",                       "{b}_cmodel_dev_fluxerr"),
    "psf":         ("{b}_psfflux_flux",                          "{b}_psfflux_fluxerr"),
    "gaap_optimal":("{b}_gaapflux_1_15x_optimal_flux",           "{b}_gaapflux_1_15x_optimal_fluxerr"),
    "gaap_psf":    ("{b}_gaapflux_1_15x_psfflux_flux",           "{b}_gaapflux_1_15x_psfflux_fluxerr"),
    "convolved":   ("{b}_convolvedflux_3_15_flux",               "{b}_convolvedflux_3_15_fluxerr"),
    "undeblended": ("{b}_undeblended_convolvedflux_3_15_flux",   "{b}_undeblended_convolvedflux_3_15_fluxerr"),
}
```

Where `{b}` is one of `g`, `r`, `i`, `z`, `y`.
