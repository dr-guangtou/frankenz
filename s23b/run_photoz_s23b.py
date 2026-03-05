#!/usr/bin/env python
"""
S23b photo-z training rehearsal with frankenz v0.4.0.

Runs KMCkNN photometric redshift inference on HSC S23b training catalog
using undeblended convolved photometry. Performs K-fold cross-validation
over all sample_crossval folds, computes comprehensive evaluation metrics,
and generates publication-quality QA figures.

Usage:
    python s23b/run_photoz_s23b.py
    python s23b/run_photoz_s23b.py --test-fold 2
    python s23b/run_photoz_s23b.py --skip-prep --skip-run
    python s23b/run_photoz_s23b.py --chunk-size 200 --test-fold 2

Requires: frankenz[all] (h5py, astropy, tqdm)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from frankenz.config import FrankenzConfig
from frankenz.io import PhotoData, write_hdf5, read_hdf5
from frankenz.batch import run_pipeline
from frankenz.pdf import pdfs_summarize


# ============================================================================
# A. Constants
# ============================================================================

BANDS = ["g", "r", "i", "z", "y"]
N_BANDS = len(BANDS)
AB_ZP = 31.4

# Column patterns for undeblended convolved photometry
FLUX_PATTERN = "{b}_undeblended_convolvedflux_3_15_flux"
FLUXERR_PATTERN = "{b}_undeblended_convolvedflux_3_15_fluxerr"

# Default catalog path
DEFAULT_CATALOG = (
    "/Users/shuang/work/hsc/photoz/s23b_photoz_calib_v3/"
    "hsc_s23b_deep_matched_train_SFR_v3.fits"
)

# Quality cut parameters
SNR_THRESHOLD = 3.0
Z_MIN = 0.0
Z_MAX = 7.0
ALLOWED_OBJECT_TYPES = {"G", "Q", "G/G", "G/Q", "Q/G", "Q/Q"}

# Skynoise estimation
FAINT_THRESHOLD = 25.0

# Plotting
FIGSIZE = (8, 6)
DPI = 150


# ============================================================================
# B. Data Preparation Helpers
# ============================================================================

def flux_to_ab_mag(flux_njy):
    """Convert nJy flux to AB magnitude."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux_njy) + AB_ZP


def compute_snr(flux, flux_err):
    """Signal-to-noise ratio."""
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = flux / flux_err
    snr[flux_err <= 0] = np.nan
    return snr


def extract_photometry(table):
    """Extract undeblended flux/fluxerr arrays from FITS table."""
    flux_cols = [FLUX_PATTERN.format(b=b) for b in BANDS]
    err_cols = [FLUXERR_PATTERN.format(b=b) for b in BANDS]
    flux = np.column_stack(
        [np.array(table[c], dtype=np.float64) for c in flux_cols]
    )
    flux_err = np.column_stack(
        [np.array(table[c], dtype=np.float64) for c in err_cols]
    )
    return flux, flux_err


def extract_extinction(table):
    """Extract per-band galactic extinction A_b array."""
    return np.column_stack(
        [np.array(table[f"a_{b}"], dtype=np.float64) for b in BANDS]
    )


def apply_extinction_correction(flux_njy, extinction_ab):
    """Correct for galactic extinction. Errors are NOT corrected."""
    raw_mag = flux_to_ab_mag(flux_njy)
    corrected_mag = raw_mag - extinction_ab
    corrected_flux = 10.0 ** (-0.4 * (corrected_mag - AB_ZP))
    return corrected_flux


def apply_quality_cuts(flux, flux_err, redshift, object_type):
    """Apply sequential quality cuts with attrition logging."""
    n_total = len(flux)
    mask = np.ones(n_total, dtype=bool)
    cuts_log = [("Initial", n_total, 0)]

    cut = np.isfinite(flux).all(axis=1) & np.isfinite(flux_err).all(axis=1)
    removed = mask.sum() - (mask & cut).sum()
    mask &= cut
    cuts_log.append(("Finite flux (all bands)", mask.sum(), removed))

    cut = (flux > 0).all(axis=1)
    removed = mask.sum() - (mask & cut).sum()
    mask &= cut
    cuts_log.append(("Positive flux (all bands)", mask.sum(), removed))

    i_snr = compute_snr(flux[:, 2], flux_err[:, 2])
    cut = np.isfinite(i_snr) & (i_snr >= SNR_THRESHOLD)
    removed = mask.sum() - (mask & cut).sum()
    mask &= cut
    cuts_log.append((f"i-band SNR >= {SNR_THRESHOLD}", mask.sum(), removed))

    cut = (redshift >= Z_MIN) & (redshift <= Z_MAX) & np.isfinite(redshift)
    removed = mask.sum() - (mask & cut).sum()
    mask &= cut
    cuts_log.append((f"Redshift in [{Z_MIN}, {Z_MAX}]", mask.sum(), removed))

    cut = np.array([t in ALLOWED_OBJECT_TYPES for t in object_type])
    removed = mask.sum() - (mask & cut).sum()
    mask &= cut
    cuts_log.append(("Object type filter", mask.sum(), removed))

    return mask, cuts_log


def estimate_skynoise(flux_err, i_mag, faint_threshold=FAINT_THRESHOLD):
    """Per-band median flux_err for faint objects (i > threshold).

    Provides physically meaningful luptitude softening parameters
    in nJy units, specific to this photometry type.
    """
    faint = i_mag > faint_threshold
    n_faint = faint.sum()
    if n_faint < 100:
        print(f"  WARNING: Only {n_faint} faint objects for skynoise estimation")
    skynoise = np.nanmedian(flux_err[faint], axis=0)
    return skynoise


def build_photo_data(flux, flux_err, redshifts, kde_bw, object_ids):
    """Construct and validate a PhotoData container."""
    data = PhotoData(
        flux=flux.astype(np.float64),
        flux_err=flux_err.astype(np.float64),
        mask=np.ones_like(flux, dtype=int),
        redshifts=redshifts.astype(np.float64),
        redshift_errs=kde_bw.astype(np.float64),
        object_ids=object_ids,
        band_names=BANDS,
    )
    data.validate()
    return data


# ============================================================================
# C. Metrics Computation
# ============================================================================

def compute_point_metrics(z_spec, z_phot):
    """Standard photo-z point estimate metrics."""
    dz = (z_phot - z_spec) / (1.0 + z_spec)
    return {
        "bias": float(np.median(dz)),
        "sigma_nmad": float(1.4826 * np.median(np.abs(dz - np.median(dz)))),
        "outlier_frac": float(np.mean(np.abs(dz) > 0.15)),
        "catastrophic_frac": float(np.mean(np.abs(dz) > 0.5)),
        "rms": float(np.sqrt(np.mean(dz ** 2))),
        "n_objects": len(z_spec),
    }


def compute_pit(pdfs, zgrid, z_spec):
    """Probability Integral Transform values."""
    cdfs = np.cumsum(pdfs, axis=1)
    cdf_norm = cdfs / cdfs[:, -1:]
    pit = np.array([
        np.interp(z_spec[i], zgrid, cdf_norm[i])
        for i in range(len(z_spec))
    ])
    return pit


def compute_crps(pdfs, zgrid, z_spec):
    """Continuous Ranked Probability Score per object."""
    cdfs = np.cumsum(pdfs, axis=1)
    cdf_norm = cdfs / cdfs[:, -1:]
    crps = np.zeros(len(z_spec))
    for i in range(len(z_spec)):
        heaviside = (zgrid >= z_spec[i]).astype(float)
        crps[i] = _trapezoid((cdf_norm[i] - heaviside) ** 2, zgrid)
    return crps


def compute_coverage(z_spec, intervals):
    """Empirical coverage fractions for 68% and 95% credible intervals."""
    plow95, plow68, phigh68, phigh95 = intervals
    cov_68 = np.mean((z_spec >= plow68) & (z_spec <= phigh68))
    cov_95 = np.mean((z_spec >= plow95) & (z_spec <= phigh95))
    return {"coverage_68": float(cov_68), "coverage_95": float(cov_95)}


def compute_binned_metrics(z_spec, z_phot, bin_values, n_bins=10):
    """Compute bias, sigma_nmad, outlier_frac in bins of bin_values."""
    bin_edges = np.percentile(bin_values, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1

    centers = np.zeros(n_bins)
    bias = np.zeros(n_bins)
    sigma_nmad = np.zeros(n_bins)
    outlier_frac = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (bin_values >= bin_edges[i]) & (bin_values < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= (bin_values == bin_edges[i + 1])
        if mask.sum() < 5:
            centers[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
            bias[i] = sigma_nmad[i] = outlier_frac[i] = np.nan
            counts[i] = mask.sum()
            continue
        m = compute_point_metrics(z_spec[mask], z_phot[mask])
        centers[i] = np.median(bin_values[mask])
        bias[i] = m["bias"]
        sigma_nmad[i] = m["sigma_nmad"]
        outlier_frac[i] = m["outlier_frac"]
        counts[i] = mask.sum()

    return {
        "bin_centers": centers,
        "bias": bias,
        "sigma_nmad": sigma_nmad,
        "outlier_frac": outlier_frac,
        "counts": counts,
    }


def compute_all_metrics(z_spec, pdfs, zgrid, summary, i_mag=None):
    """Compute all evaluation metrics."""
    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary

    estimators = {
        "mean": mean_stats[0],
        "median": median_stats[0],
        "mode": mode_stats[0],
        "best": best_stats[0],
    }
    point_metrics = {}
    for name, z_phot in estimators.items():
        point_metrics[name] = compute_point_metrics(z_spec, z_phot)

    pit = compute_pit(pdfs, zgrid, z_spec)
    crps = compute_crps(pdfs, zgrid, z_spec)
    ks_stat, ks_pvalue = stats.kstest(pit, "uniform")
    coverage = compute_coverage(z_spec, intervals)

    pdf_metrics = {
        "pit": pit,
        "crps": crps,
        "crps_mean": float(np.mean(crps)),
        "crps_median": float(np.median(crps)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        **coverage,
    }

    z_best = best_stats[0]
    binned_vs_z = compute_binned_metrics(z_spec, z_best, z_spec)
    binned = {"vs_zspec": binned_vs_z}
    if i_mag is not None:
        binned["vs_imag"] = compute_binned_metrics(z_spec, z_best, i_mag)

    return {
        "point": point_metrics,
        "pdf": pdf_metrics,
        "binned": binned,
        "estimators": estimators,
        "summary": summary,
    }


def compute_source_metrics(z_spec, z_phot, specz_sources):
    """Compute metrics split by spectroscopic source (DESI vs COSMOSWeb)."""
    results = {}
    for source_key, source_name in [
        ("DESI_DR1", "DESI"),
        ("COSMOSWeb2025_v1", "COSMOSWeb"),
    ]:
        mask = specz_sources == source_key
        if mask.sum() < 10:
            continue
        results[source_name] = compute_point_metrics(z_spec[mask], z_phot[mask])
    return results


# ============================================================================
# D. Visualization (12 figures)
# ============================================================================

def fig_01_scatter_4panel(z_spec, estimators, metrics, output_path):
    """z_spec vs z_phot for mean/median/mode/best, density hexbin."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    names = ["mean", "median", "mode", "best"]

    for ax, name in zip(axes.flat, names):
        z_phot = estimators[name]
        m = metrics["point"][name]

        z_max = min(max(z_spec.max(), z_phot.max()) * 1.05, 4.0)
        ax.hexbin(z_spec, z_phot, gridsize=80, cmap="viridis",
                  mincnt=1, norm=LogNorm(), extent=[0, z_max, 0, z_max])
        ax.plot([0, z_max], [0, z_max], "r-", lw=0.8, alpha=0.7)
        zz = np.linspace(0, z_max, 100)
        ax.plot(zz, zz + 0.15 * (1 + zz), "r--", lw=0.6, alpha=0.5)
        ax.plot(zz, zz - 0.15 * (1 + zz), "r--", lw=0.6, alpha=0.5)

        ax.set_xlim(0, z_max)
        ax.set_ylim(0, z_max)
        ax.set_xlabel("$z_{\\rm spec}$")
        ax.set_ylabel("$z_{\\rm phot}$")
        ax.set_title(f"{name}", fontsize=12)

        text = (f"bias = {m['bias']:.4f}\n"
                f"$\\sigma_{{\\rm NMAD}}$ = {m['sigma_nmad']:.4f}\n"
                f"$f_{{\\rm out}}$ = {m['outlier_frac']:.3f}\n"
                f"N = {m['n_objects']}")
        ax.text(0.03, 0.97, text, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Photo-z Scatter: $z_{\\rm spec}$ vs $z_{\\rm phot}$",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_02_residual_histogram(z_spec, z_best, metrics_best, output_path):
    """Histogram of dz/(1+z) with Gaussian fit overlay."""
    dz = (z_best - z_spec) / (1.0 + z_spec)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(-0.5, 0.5, 101)
    ax.hist(dz, bins=bins, density=True, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Data")

    mu, sigma = metrics_best["bias"], metrics_best["sigma_nmad"]
    x = np.linspace(-0.5, 0.5, 300)
    gauss = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, gauss, "r-", lw=2,
            label=f"Gaussian ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})")

    ax.axvline(-0.15, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0.15, color="gray", ls="--", lw=0.8, alpha=0.6)

    text = (f"bias = {mu:.4f}\n"
            f"$\\sigma_{{\\rm NMAD}}$ = {sigma:.4f}\n"
            f"$f_{{\\rm out}}$ = {metrics_best['outlier_frac']:.3f}")
    ax.text(0.97, 0.97, text, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("$\\Delta z / (1 + z_{\\rm spec})$")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution (best estimator)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_03_residual_vs_zspec(z_spec, z_best, output_path):
    """dz/(1+z) vs z_spec with running median and 68% band."""
    dz = (z_best - z_spec) / (1.0 + z_spec)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hexbin(z_spec, dz, gridsize=80, cmap="viridis", mincnt=1,
              norm=LogNorm(), extent=[0, z_spec.max() * 1.05, -0.5, 0.5])

    z_bins = np.linspace(z_spec.min(), z_spec.max(), 21)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    running_median = np.zeros(len(z_centers))
    running_lo = np.zeros(len(z_centers))
    running_hi = np.zeros(len(z_centers))
    for i in range(len(z_centers)):
        mask = (z_spec >= z_bins[i]) & (z_spec < z_bins[i + 1])
        if mask.sum() < 5:
            running_median[i] = running_lo[i] = running_hi[i] = np.nan
            continue
        running_median[i] = np.median(dz[mask])
        running_lo[i] = np.percentile(dz[mask], 16)
        running_hi[i] = np.percentile(dz[mask], 84)

    ax.plot(z_centers, running_median, "r-", lw=2, label="Running median")
    ax.fill_between(z_centers, running_lo, running_hi,
                    color="red", alpha=0.2, label="68% band")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(0.15, color="gray", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(-0.15, color="gray", ls=":", lw=0.6, alpha=0.5)

    ax.set_xlabel("$z_{\\rm spec}$")
    ax.set_ylabel("$\\Delta z / (1 + z_{\\rm spec})$")
    ax.set_title("Residuals vs Spectroscopic Redshift")
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_04_residual_vs_mag(z_spec, z_best, i_mag, output_path):
    """dz/(1+z) vs i-band magnitude with running median and 68% band."""
    dz = (z_best - z_spec) / (1.0 + z_spec)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    valid = np.isfinite(i_mag) & np.isfinite(dz)
    ax.hexbin(i_mag[valid], dz[valid], gridsize=80, cmap="viridis",
              mincnt=1, norm=LogNorm())

    mag_bins = np.linspace(np.nanpercentile(i_mag, 1),
                           np.nanpercentile(i_mag, 99), 21)
    mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    running_median = np.zeros(len(mag_centers))
    running_lo = np.zeros(len(mag_centers))
    running_hi = np.zeros(len(mag_centers))
    for i in range(len(mag_centers)):
        mask = (i_mag >= mag_bins[i]) & (i_mag < mag_bins[i + 1])
        if mask.sum() < 5:
            running_median[i] = running_lo[i] = running_hi[i] = np.nan
            continue
        running_median[i] = np.median(dz[mask])
        running_lo[i] = np.percentile(dz[mask], 16)
        running_hi[i] = np.percentile(dz[mask], 84)

    ax.plot(mag_centers, running_median, "r-", lw=2, label="Running median")
    ax.fill_between(mag_centers, running_lo, running_hi,
                    color="red", alpha=0.2, label="68% band")
    ax.axhline(0, color="gray", ls="--", lw=0.8)

    ax.set_xlabel("$i$-band magnitude (AB)")
    ax.set_ylabel("$\\Delta z / (1 + z_{\\rm spec})$")
    ax.set_title("Residuals vs $i$-band Magnitude")
    ax.set_ylim(-0.5, 0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_05_pit_qq(pit, ks_stat, ks_pvalue, output_path):
    """PIT histogram + QQ plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(pit, bins=20, density=True, color="steelblue", alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax1.axhline(1.0, color="red", ls="--", lw=1.5, label="Uniform")
    ax1.set_xlabel("PIT value")
    ax1.set_ylabel("Density")
    ax1.set_title("PIT Histogram")
    ax1.set_xlim(0, 1)
    ax1.legend()

    pit_sorted = np.sort(pit)
    n = len(pit_sorted)
    empirical_cdf = np.arange(1, n + 1) / n
    ax2.plot(pit_sorted, empirical_cdf, "b-", lw=1.5, label="PIT ECDF")
    ax2.plot([0, 1], [0, 1], "r--", lw=1, label="1:1 (uniform)")
    ax2.set_xlabel("PIT quantile")
    ax2.set_ylabel("Empirical CDF")
    ax2.set_title("QQ Plot")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")

    text = f"KS stat = {ks_stat:.4f}\n$p$-value = {ks_pvalue:.4g}"
    ax2.text(0.03, 0.97, text, transform=ax2.transAxes,
             va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax2.legend(loc="lower right")

    fig.suptitle("PDF Calibration: Probability Integral Transform", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_06_nz_comparison(z_spec, pdfs, zgrid, output_path):
    """Stacked N(z) from PDFs vs true z_spec histogram."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    z_max = min(z_spec.max() * 1.1, 4.0)
    bins = np.arange(0, z_max, 0.05)
    ax.hist(z_spec, bins=bins, density=True, color="steelblue", alpha=0.5,
            edgecolor="white", linewidth=0.3, label="$z_{\\rm spec}$ (true)")

    nz_pdf = pdfs.sum(axis=0)
    nz_pdf = nz_pdf / _trapezoid(nz_pdf, zgrid)
    mask = zgrid <= z_max
    ax.plot(zgrid[mask], nz_pdf[mask], "r-", lw=2,
            label="Stacked $P(z)$ (predicted)")

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Density")
    ax.set_title("$N(z)$ Comparison: True vs Predicted")
    ax.set_xlim(0, z_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_07_example_pdfs(z_spec, z_best, pdfs, zgrid, output_path):
    """3x3 grid of individual PDFs: good, moderate, outlier."""
    dz = np.abs(z_best - z_spec) / (1.0 + z_spec)

    good_idx = np.where(dz < 0.02)[0]
    moderate_idx = np.where((dz > 0.05) & (dz < 0.15))[0]
    outlier_idx = np.where(dz > 0.15)[0]

    rng = np.random.RandomState(42)
    selections = []
    for pool, label in [(good_idx, "Good"), (moderate_idx, "Moderate"),
                        (outlier_idx, "Outlier")]:
        if len(pool) >= 3:
            chosen = rng.choice(pool, 3, replace=False)
        elif len(pool) > 0:
            chosen = rng.choice(pool, min(3, len(pool)), replace=True)
        else:
            chosen = rng.choice(len(z_spec), 3, replace=False)
        selections.append((chosen, label))

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for row, (chosen, label) in enumerate(selections):
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            z_max_plot = min(max(z_spec[idx], z_best[idx]) * 2.5, 4.0)
            z_max_plot = max(z_max_plot, 0.5)
            pmask = zgrid <= z_max_plot
            ax.plot(zgrid[pmask], pdfs[idx, pmask], "b-", lw=1.5)
            ax.axvline(z_spec[idx], color="green", ls="--", lw=1.2,
                       label=f"$z_{{\\rm spec}}$={z_spec[idx]:.3f}")
            ax.axvline(z_best[idx], color="red", ls=":", lw=1.2,
                       label=f"$z_{{\\rm best}}$={z_best[idx]:.3f}")
            ax.set_xlabel("$z$")
            if col == 0:
                ax.set_ylabel(f"{label}\n$P(z)$")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlim(0, z_max_plot)

    fig.suptitle("Example PDFs: Good / Moderate / Outlier", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_08_metrics_vs_zbin(binned_z, output_path):
    """3-panel: bias, sigma_nmad, outlier_frac vs redshift bin."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    x = binned_z["bin_centers"]

    ax1.plot(x, binned_z["bias"], "ko-", ms=5)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Bias")
    ax1.set_title("Metrics vs Spectroscopic Redshift")

    ax2.plot(x, binned_z["sigma_nmad"], "ko-", ms=5)
    ax2.set_ylabel("$\\sigma_{\\rm NMAD}$")

    ax3.plot(x, binned_z["outlier_frac"], "ko-", ms=5)
    ax3.axhline(0.15, color="red", ls="--", lw=0.8, alpha=0.5,
                label="15% threshold")
    ax3.set_ylabel("Outlier fraction")
    ax3.set_xlabel("$z_{\\rm spec}$")
    ax3.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_09_metrics_vs_magbin(binned_mag, output_path):
    """3-panel: bias, sigma_nmad, outlier_frac vs i-band magnitude."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    x = binned_mag["bin_centers"]

    ax1.plot(x, binned_mag["bias"], "ko-", ms=5)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Bias")
    ax1.set_title("Metrics vs $i$-band Magnitude")

    ax2.plot(x, binned_mag["sigma_nmad"], "ko-", ms=5)
    ax2.set_ylabel("$\\sigma_{\\rm NMAD}$")

    ax3.plot(x, binned_mag["outlier_frac"], "ko-", ms=5)
    ax3.set_ylabel("Outlier fraction")
    ax3.set_xlabel("$i$-band magnitude (AB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_10_credible_coverage(z_spec, pdfs, zgrid, output_path):
    """Actual vs nominal coverage at multiple quantile levels."""
    cdfs = np.cumsum(pdfs, axis=1)
    cdf_norm = cdfs / cdfs[:, -1:]

    nominal_levels = np.array([0.10, 0.20, 0.30, 0.40, 0.50,
                               0.60, 0.68, 0.80, 0.90, 0.95])
    actual_coverage = np.zeros(len(nominal_levels))

    for j, level in enumerate(nominal_levels):
        alpha_lo = (1.0 - level) / 2.0
        alpha_hi = 1.0 - alpha_lo
        covered = 0
        for i in range(len(z_spec)):
            q_lo = np.interp(alpha_lo, cdf_norm[i], zgrid)
            q_hi = np.interp(alpha_hi, cdf_norm[i], zgrid)
            if q_lo <= z_spec[i] <= q_hi:
                covered += 1
        actual_coverage[j] = covered / len(z_spec)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="Perfect calibration")
    ax.plot(nominal_levels, actual_coverage, "bo-", ms=6, lw=2,
            label="Observed")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Actual coverage")
    ax.set_title("Credible Interval Coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_11_nz_by_source(z_spec, pdfs, zgrid, specz_sources, output_path):
    """N(z) by source: DESI vs COSMOSWeb histograms with stacked PDFs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    z_max = min(z_spec.max() * 1.1, 4.0)
    bins = np.arange(0, z_max, 0.05)

    mask_desi = specz_sources == "DESI_DR1"
    mask_cosmos = specz_sources == "COSMOSWeb2025_v1"

    ax.hist(z_spec[mask_desi], bins=bins, density=True, alpha=0.4,
            color="steelblue", label=f"DESI spec-z (N={mask_desi.sum():,})")
    ax.hist(z_spec[mask_cosmos], bins=bins, density=True, alpha=0.4,
            color="orangered", label=f"COSMOSWeb spec-z (N={mask_cosmos.sum():,})")

    # Stacked PDFs per source
    for m, color, label in [
        (mask_desi, "steelblue", "DESI $\\Sigma P(z)$"),
        (mask_cosmos, "orangered", "COSMOSWeb $\\Sigma P(z)$"),
    ]:
        if m.sum() == 0:
            continue
        nz = pdfs[m].sum(axis=0)
        nz = nz / _trapezoid(nz, zgrid)
        zmask = zgrid <= z_max
        ax.plot(zgrid[zmask], nz[zmask], color=color, lw=2,
                ls="--", label=label)

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Density")
    ax.set_title("$N(z)$ by Spectroscopic Source")
    ax.set_xlim(0, z_max)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_12_metrics_by_source(z_spec, z_best, specz_sources, source_metrics,
                             output_path):
    """Side-by-side scatter plots for DESI and COSMOSWeb."""
    mask_desi = specz_sources == "DESI_DR1"
    mask_cosmos = specz_sources == "COSMOSWeb2025_v1"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for ax, m, name, color in [
        (ax1, mask_desi, "DESI", "steelblue"),
        (ax2, mask_cosmos, "COSMOSWeb", "orangered"),
    ]:
        if m.sum() == 0:
            continue
        zs, zp = z_spec[m], z_best[m]
        z_max = min(max(zs.max(), zp.max()) * 1.05, 4.0)
        ax.hexbin(zs, zp, gridsize=60, cmap="viridis",
                  mincnt=1, norm=LogNorm(), extent=[0, z_max, 0, z_max])
        ax.plot([0, z_max], [0, z_max], "r-", lw=0.8, alpha=0.7)
        zz = np.linspace(0, z_max, 100)
        ax.plot(zz, zz + 0.15 * (1 + zz), "r--", lw=0.6, alpha=0.5)
        ax.plot(zz, zz - 0.15 * (1 + zz), "r--", lw=0.6, alpha=0.5)

        ax.set_xlim(0, z_max)
        ax.set_ylim(0, z_max)
        ax.set_xlabel("$z_{\\rm spec}$")
        ax.set_ylabel("$z_{\\rm phot}$")
        ax.set_title(name, fontsize=13)

        if name in source_metrics:
            sm = source_metrics[name]
            text = (f"$\\sigma_{{\\rm NMAD}}$ = {sm['sigma_nmad']:.4f}\n"
                    f"$f_{{\\rm out}}$ = {sm['outlier_frac']:.3f}\n"
                    f"bias = {sm['bias']:.4f}\n"
                    f"N = {sm['n_objects']}")
            ax.text(0.03, 0.97, text, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Photo-z by Spectroscopic Source", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(z_spec, metrics, pdfs, zgrid, i_mag,
                         specz_sources, source_metrics, fig_dir):
    """Generate all 12 QA figures."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = metrics["summary"]
    _, _, _, best_stats, intervals, _ = summary
    z_best = best_stats[0]
    m_best = metrics["point"]["best"]

    print("\nGenerating figures...")

    print("  [01/12] Scatter 4-panel")
    fig_01_scatter_4panel(
        z_spec, metrics["estimators"], metrics,
        fig_dir / "fig_01_scatter_4panel.png")

    print("  [02/12] Residual histogram")
    fig_02_residual_histogram(z_spec, z_best, m_best,
                              fig_dir / "fig_02_residual_histogram.png")

    print("  [03/12] Residual vs z_spec")
    fig_03_residual_vs_zspec(z_spec, z_best,
                             fig_dir / "fig_03_residual_vs_zspec.png")

    print("  [04/12] Residual vs i-mag")
    fig_04_residual_vs_mag(z_spec, z_best, i_mag,
                           fig_dir / "fig_04_residual_vs_mag.png")

    print("  [05/12] PIT + QQ")
    fig_05_pit_qq(
        metrics["pdf"]["pit"],
        metrics["pdf"]["ks_stat"], metrics["pdf"]["ks_pvalue"],
        fig_dir / "fig_05_pit_qq.png")

    print("  [06/12] N(z) comparison")
    fig_06_nz_comparison(z_spec, pdfs, zgrid,
                         fig_dir / "fig_06_nz_comparison.png")

    print("  [07/12] Example PDFs")
    fig_07_example_pdfs(z_spec, z_best, pdfs, zgrid,
                        fig_dir / "fig_07_example_pdfs.png")

    print("  [08/12] Metrics vs z_spec")
    fig_08_metrics_vs_zbin(metrics["binned"]["vs_zspec"],
                           fig_dir / "fig_08_metrics_vs_zbin.png")

    if "vs_imag" in metrics["binned"]:
        print("  [09/12] Metrics vs i-mag")
        fig_09_metrics_vs_magbin(metrics["binned"]["vs_imag"],
                                 fig_dir / "fig_09_metrics_vs_magbin.png")

    print("  [10/12] Credible coverage")
    fig_10_credible_coverage(z_spec, pdfs, zgrid,
                             fig_dir / "fig_10_credible_coverage.png")

    print("  [11/12] N(z) by source")
    fig_11_nz_by_source(z_spec, pdfs, zgrid, specz_sources,
                        fig_dir / "fig_11_nz_by_source.png")

    print("  [12/12] Metrics by source")
    fig_12_metrics_by_source(z_spec, z_best, specz_sources, source_metrics,
                             fig_dir / "fig_12_metrics_by_source.png")

    print(f"  All figures saved to {fig_dir}/")


# ============================================================================
# E. Report Generation
# ============================================================================

def write_metrics_report(metrics, source_metrics, output_path):
    """Write human-readable evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("Frankenz v0.4.0 — S23b Photo-z Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    lines.append("POINT ESTIMATE METRICS")
    lines.append("-" * 70)
    header = f"{'Estimator':<10} {'Bias':>10} {'sigma_NMAD':>12} "
    header += f"{'f_out':>8} {'f_cat':>8} {'RMS':>10} {'N':>8}"
    lines.append(header)
    lines.append("-" * 70)
    for name in ["mean", "median", "mode", "best"]:
        m = metrics["point"][name]
        line = f"{name:<10} {m['bias']:>10.5f} {m['sigma_nmad']:>12.5f} "
        line += f"{m['outlier_frac']:>8.4f} {m['catastrophic_frac']:>8.4f} "
        line += f"{m['rms']:>10.5f} {m['n_objects']:>8d}"
        lines.append(line)
    lines.append("")

    lines.append("PDF QUALITY METRICS")
    lines.append("-" * 70)
    pm = metrics["pdf"]
    lines.append(f"  CRPS (mean)     : {pm['crps_mean']:.5f}")
    lines.append(f"  CRPS (median)   : {pm['crps_median']:.5f}")
    lines.append(f"  KS statistic    : {pm['ks_stat']:.5f}")
    lines.append(f"  KS p-value      : {pm['ks_pvalue']:.4g}")
    lines.append(f"  Coverage (68%)  : {pm['coverage_68']:.4f}"
                 f"  (nominal: 0.6800)")
    lines.append(f"  Coverage (95%)  : {pm['coverage_95']:.4f}"
                 f"  (nominal: 0.9500)")
    lines.append("")

    if source_metrics:
        lines.append("PER-SOURCE METRICS (best estimator)")
        lines.append("-" * 70)
        header = f"{'Source':<12} {'Bias':>10} {'sigma_NMAD':>12} "
        header += f"{'f_out':>8} {'N':>8}"
        lines.append(header)
        lines.append("-" * 70)
        for name, m in source_metrics.items():
            line = f"{name:<12} {m['bias']:>10.5f} {m['sigma_nmad']:>12.5f} "
            line += f"{m['outlier_frac']:>8.4f} {m['n_objects']:>8d}"
            lines.append(line)
        lines.append("")

    for bin_type, label in [("vs_zspec", "z_spec"), ("vs_imag", "i-mag")]:
        if bin_type not in metrics["binned"]:
            continue
        b = metrics["binned"][bin_type]
        lines.append(f"BINNED METRICS vs {label}")
        lines.append("-" * 70)
        header = f"{'Bin center':>12} {'Bias':>10} {'sigma_NMAD':>12} "
        header += f"{'f_out':>8} {'N':>8}"
        lines.append(header)
        lines.append("-" * 70)
        for i in range(len(b["bin_centers"])):
            line = f"{b['bin_centers'][i]:>12.4f} "
            line += f"{b['bias'][i]:>10.5f} {b['sigma_nmad'][i]:>12.5f} "
            line += f"{b['outlier_frac'][i]:>8.4f} {b['counts'][i]:>8d}"
            lines.append(line)
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"\nMetrics report saved to {output_path}")
    return report


def write_metrics_json(metrics, source_metrics, output_path):
    """Write machine-readable metrics JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "point": metrics["point"],
        "pdf": {
            k: v for k, v in metrics["pdf"].items()
            if k not in ("pit", "crps")
        },
        "source": source_metrics,
    }
    output_path.write_text(json.dumps(data, indent=2))


def write_report_markdown(config, prep_metadata, metrics, source_metrics,
                          skynoise, output_path):
    """Write comprehensive markdown report with lessons learned."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# S23b Photo-z Training Rehearsal Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Backend**: {config.model.backend}, "
                 f"K_tree={config.model.k_tree}, K_point={config.model.k_point}")
    lines.append(f"- **Transform**: {config.transform.type}")
    lines.append(f"- **z-grid**: [{config.zgrid.z_start}, {config.zgrid.z_end}] "
                 f"dz={config.zgrid.z_delta}")
    lines.append(f"- **KDE bandwidth**: frac={config.pdf.kde_bandwidth_fraction}, "
                 f"floor={config.pdf.kde_bandwidth_floor}")
    lines.append(f"- **Flux type**: undeblended convolved (3.15 arcsec)")
    lines.append(f"- **Flux units**: nJy (AB ZP = {AB_ZP})")
    lines.append("")

    lines.append("## Data Summary")
    lines.append("")
    lines.append(f"- **Source catalog**: `{prep_metadata.get('catalog', 'N/A')}`")
    lines.append(f"- **Objects after cuts**: {prep_metadata.get('n_clean', 'N/A'):,}")
    lines.append(f"- **Folds**: {prep_metadata.get('n_folds', 'N/A')}")
    lines.append("")

    if "cuts_log" in prep_metadata:
        lines.append("### Quality Cut Attrition")
        lines.append("")
        lines.append("| Cut | Remaining | Removed | Survival |")
        lines.append("|-----|-----------|---------|----------|")
        n_total = prep_metadata["cuts_log"][0][1]
        for name, remaining, removed in prep_metadata["cuts_log"]:
            pct = remaining / n_total * 100
            lines.append(f"| {name} | {remaining:,} | {removed:,} | {pct:.1f}% |")
        lines.append("")

    lines.append("## Skynoise Estimation")
    lines.append("")
    lines.append(f"Estimated from median flux_err of faint objects "
                 f"(i > {FAINT_THRESHOLD} mag):")
    lines.append("")
    for i, b in enumerate(BANDS):
        lines.append(f"- **{b}**: {skynoise[i]:.4f} nJy")
    lines.append("")

    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Estimator | Bias | sigma_NMAD | f_out | f_cat | RMS | N |")
    lines.append("|-----------|------|------------|-------|-------|-----|---|")
    for name in ["mean", "median", "mode", "best"]:
        m = metrics["point"][name]
        lines.append(
            f"| {name} | {m['bias']:.5f} | {m['sigma_nmad']:.5f} | "
            f"{m['outlier_frac']:.4f} | {m['catastrophic_frac']:.4f} | "
            f"{m['rms']:.5f} | {m['n_objects']} |"
        )
    lines.append("")

    lines.append("## PDF Quality")
    lines.append("")
    pm = metrics["pdf"]
    lines.append(f"- **CRPS (mean)**: {pm['crps_mean']:.5f}")
    lines.append(f"- **CRPS (median)**: {pm['crps_median']:.5f}")
    lines.append(f"- **KS statistic**: {pm['ks_stat']:.5f}")
    lines.append(f"- **KS p-value**: {pm['ks_pvalue']:.4g}")
    lines.append(f"- **Coverage (68%)**: {pm['coverage_68']:.4f} (nominal: 0.68)")
    lines.append(f"- **Coverage (95%)**: {pm['coverage_95']:.4f} (nominal: 0.95)")
    lines.append("")

    if source_metrics:
        lines.append("## Per-Source Metrics (best estimator)")
        lines.append("")
        lines.append("| Source | Bias | sigma_NMAD | f_out | N |")
        lines.append("|--------|------|------------|-------|---|")
        for name, m in source_metrics.items():
            lines.append(
                f"| {name} | {m['bias']:.5f} | {m['sigma_nmad']:.5f} | "
                f"{m['outlier_frac']:.4f} | {m['n_objects']} |"
            )
        lines.append("")

    lines.append("## Known Limitations")
    lines.append("")
    lines.append("1. **Skynoise approximation**: Estimated from faint-object "
                 "flux_err rather than sky background measurement. Should be "
                 "recalibrated for production use.")
    lines.append("2. **Incomplete training sample**: Missing bright/low-z "
                 "coverage from SDSS, BOSS, and GAMA spectroscopic surveys.")
    lines.append("3. **DESI spec-z errors**: DESI DR1 redshift errors are "
                 "sentinels (-1.0), so KDE bandwidth uses "
                 "max(frac * z, floor) for all objects.")
    lines.append("4. **Self-training bias**: Cross-validation mitigates but "
                 "doesn't eliminate bias from training on same field.")
    lines.append("")

    lines.append("## Lessons Learned")
    lines.append("")
    lines.append("(To be filled after execution with observations from "
                 "the actual results.)")
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Report saved to {output_path}")


# ============================================================================
# F. Results Saving
# ============================================================================

def save_fold_results(pdfs, zgrid, summary, fold, output_dir):
    """Save per-fold pipeline results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary
    path = output_dir / f"fold_{fold}_results.npz"
    np.savez(
        path,
        pdfs=pdfs,
        zgrid=zgrid,
        z_mean=mean_stats[0], z_mean_std=mean_stats[1],
        z_median=median_stats[0], z_median_std=median_stats[1],
        z_mode=mode_stats[0], z_mode_std=mode_stats[1],
        z_best=best_stats[0], z_best_std=best_stats[1],
        plow95=intervals[0], plow68=intervals[1],
        phigh68=intervals[2], phigh95=intervals[3],
        z_mc=mc,
    )
    print(f"  Fold {fold} results saved to {path}")


def save_combined_results(pdfs, zgrid, metrics, output_path):
    """Save combined results across all folds."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = metrics["summary"]
    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary

    arrays = {
        "pdfs": pdfs,
        "zgrid": zgrid,
        "z_mean": mean_stats[0], "z_mean_std": mean_stats[1],
        "z_median": median_stats[0], "z_median_std": median_stats[1],
        "z_mode": mode_stats[0], "z_mode_std": mode_stats[1],
        "z_best": best_stats[0], "z_best_std": best_stats[1],
        "plow95": intervals[0], "plow68": intervals[1],
        "phigh68": intervals[2], "phigh95": intervals[3],
        "z_mc": mc,
        "pit": metrics["pdf"]["pit"],
        "crps": metrics["pdf"]["crps"],
    }
    np.savez(output_path, **arrays)
    print(f"Combined results saved to {output_path}")


# ============================================================================
# G. Data Preparation Pipeline
# ============================================================================

def load_catalog(catalog_path):
    """Load the FITS catalog and extract common arrays."""
    from astropy.table import Table

    print(f"\nLoading catalog from {catalog_path}")
    cat = Table.read(catalog_path)
    print(f"  {len(cat):,} rows, {len(cat.colnames)} columns")

    redshift = np.array(cat["redshift"], dtype=np.float64)
    redshift_err = np.array(cat["redshift_err"], dtype=np.float64)
    specz_sources = np.array(cat["specz_sources"], dtype=str)
    object_type = np.array(cat["object_type"], dtype=str)
    sample_crossval = np.array(cat["sample_crossval"], dtype=int)
    object_id = np.array(cat["object_id"])

    flux, flux_err = extract_photometry(cat)
    extinction = extract_extinction(cat)

    print(f"  Redshift range: [{redshift.min():.4f}, {redshift.max():.4f}]")
    print(f"  Flux type: undeblended convolved (3.15 arcsec)")

    return {
        "flux": flux,
        "flux_err": flux_err,
        "extinction": extinction,
        "redshift": redshift,
        "redshift_err": redshift_err,
        "specz_sources": specz_sources,
        "object_type": object_type,
        "sample_crossval": sample_crossval,
        "object_id": object_id,
    }


def prepare_all_folds(catalog_data, config, output_dir, force=False):
    """Apply quality cuts, extinction correction, estimate skynoise,
    split by fold, and save train/test HDF5 per fold.

    Returns prep_metadata dict with attrition info and clean arrays.
    """
    import yaml as _yaml

    output_dir = Path(output_dir)
    prepared_dir = output_dir / "prepared"

    metadata_path = prepared_dir / "metadata.yaml"
    if metadata_path.exists() and not force:
        print(f"\nPrepared data already exists at {prepared_dir}/")
        print("  Use --force to re-prepare.")
        with open(metadata_path) as f:
            return _yaml.safe_load(f)

    prepared_dir.mkdir(parents=True, exist_ok=True)

    flux = catalog_data["flux"]
    flux_err = catalog_data["flux_err"]
    extinction = catalog_data["extinction"]
    redshift = catalog_data["redshift"]
    object_type = catalog_data["object_type"]
    specz_sources = catalog_data["specz_sources"]
    sample_crossval = catalog_data["sample_crossval"]
    object_id = catalog_data["object_id"]

    # Quality cuts
    print("\nApplying quality cuts...")
    clean_mask, cuts_log = apply_quality_cuts(
        flux, flux_err, redshift, object_type
    )
    n_clean = clean_mask.sum()
    print(f"  {n_clean:,} / {len(flux):,} objects pass quality cuts")

    for name, remaining, removed in cuts_log:
        pct = remaining / len(flux) * 100
        print(f"    {name:<35} {remaining:>8,} ({pct:>5.1f}%)"
              f"  removed {removed:>6,}")

    # Apply mask
    flux_clean = flux[clean_mask]
    flux_err_clean = flux_err[clean_mask]
    ext_clean = extinction[clean_mask]
    z_clean = redshift[clean_mask]
    src_clean = specz_sources[clean_mask]
    cv_clean = sample_crossval[clean_mask]
    oid_clean = object_id[clean_mask]

    # Extinction correction
    print("\nApplying extinction correction...")
    flux_corrected = apply_extinction_correction(flux_clean, ext_clean)
    print(f"  Median A_i = {np.median(ext_clean[:, 2]):.4f} mag")

    # Estimate skynoise from faint objects
    i_mag_clean = flux_to_ab_mag(flux_corrected[:, 2])
    skynoise = estimate_skynoise(flux_err_clean, i_mag_clean)
    n_faint = (i_mag_clean > FAINT_THRESHOLD).sum()
    print(f"\nSkynoise estimation (from {n_faint:,} faint objects, "
          f"i > {FAINT_THRESHOLD}):")
    for i, b in enumerate(BANDS):
        print(f"  {b}: {skynoise[i]:.4f} nJy")

    # Override config skynoise
    config.transform.skynoise = skynoise.tolist()
    print(f"\n  Config skynoise updated: {config.transform.skynoise}")

    # KDE bandwidth
    bw_frac = config.pdf.kde_bandwidth_fraction
    bw_floor = config.pdf.kde_bandwidth_floor
    kde_bw = np.maximum(bw_frac * z_clean, bw_floor)
    print(f"  KDE bandwidth: max({bw_frac} * z, {bw_floor})")

    # Split by fold and save
    folds = np.unique(cv_clean)
    print(f"\nSplitting into {len(folds)} folds: {folds.tolist()}")

    for fold in folds:
        test_mask = cv_clean == fold
        train_mask = ~test_mask

        train_data = build_photo_data(
            flux_corrected[train_mask], flux_err_clean[train_mask],
            z_clean[train_mask], kde_bw[train_mask], oid_clean[train_mask],
        )
        test_data = build_photo_data(
            flux_corrected[test_mask], flux_err_clean[test_mask],
            z_clean[test_mask], kde_bw[test_mask], oid_clean[test_mask],
        )

        write_hdf5(train_data, prepared_dir / f"fold_{fold}_train.hdf5")
        write_hdf5(test_data, prepared_dir / f"fold_{fold}_test.hdf5")
        print(f"  Fold {fold}: train={train_data.n_objects:,}, "
              f"test={test_data.n_objects:,}")

    # Save auxiliary arrays (for post-processing without reloading FITS)
    aux_path = prepared_dir / "auxiliary.npz"
    np.savez(
        aux_path,
        specz_sources=src_clean,
        object_type=catalog_data["object_type"][clean_mask],
        i_mag=i_mag_clean,
        sample_crossval=cv_clean,
        skynoise=skynoise,
    )
    print(f"  Auxiliary data saved to {aux_path}")

    # Save metadata
    prep_metadata = {
        "catalog": str(catalog_data.get("catalog_path", DEFAULT_CATALOG)),
        "flux_type": "undeblended",
        "flux_pattern": FLUX_PATTERN,
        "n_total": int(len(flux)),
        "n_clean": int(n_clean),
        "n_folds": int(len(folds)),
        "folds": folds.tolist(),
        "cuts_log": [[name, int(rem), int(rmv)] for name, rem, rmv in cuts_log],
        "skynoise": skynoise.tolist(),
        "faint_threshold": FAINT_THRESHOLD,
        "kde_bandwidth_fraction": bw_frac,
        "kde_bandwidth_floor": bw_floor,
    }
    with open(metadata_path, "w") as f:
        _yaml.dump(prep_metadata, f, default_flow_style=False, sort_keys=False)
    print(f"  Metadata saved to {metadata_path}")

    return prep_metadata


# ============================================================================
# H. Pipeline Execution
# ============================================================================

def run_single_fold(fold, config, output_dir, chunk_size):
    """Run frankenz pipeline for a single fold."""
    prepared_dir = Path(output_dir) / "prepared"
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_path = results_dir / f"fold_{fold}_results.npz"
    if result_path.exists():
        print(f"\n  Fold {fold} results already exist, skipping.")
        return

    print(f"\n--- Fold {fold} ---")
    train_data = read_hdf5(prepared_dir / f"fold_{fold}_train.hdf5")
    test_data = read_hdf5(prepared_dir / f"fold_{fold}_test.hdf5")
    print(f"  Train: {train_data.n_objects:,}, Test: {test_data.n_objects:,}")

    t0 = time.time()
    result = run_pipeline(config, train_data, test_data, chunk_size=chunk_size)
    elapsed = time.time() - t0
    rate = test_data.n_objects / elapsed
    print(f"  Pipeline complete in {elapsed:.1f}s ({rate:.0f} obj/s)")

    pdfs = result.pdfs
    zgrid = result.zgrid

    print("  Computing PDF summary statistics...")
    summary = pdfs_summarize(pdfs, zgrid, renormalize=True)

    save_fold_results(pdfs, zgrid, summary, fold, results_dir)


def aggregate_fold_results(folds, output_dir):
    """Load per-fold results and concatenate in original object order."""
    results_dir = Path(output_dir) / "results"
    prepared_dir = Path(output_dir) / "prepared"

    # Load auxiliary to get fold assignments
    aux = np.load(prepared_dir / "auxiliary.npz", allow_pickle=True)
    cv = aux["sample_crossval"]
    n_total = len(cv)

    # Load first fold to get zgrid shape
    first = np.load(results_dir / f"fold_{folds[0]}_results.npz")
    zgrid = first["zgrid"]
    n_z = len(zgrid)

    # Pre-allocate arrays
    all_pdfs = np.zeros((n_total, n_z))
    all_z_mean = np.zeros(n_total)
    all_z_median = np.zeros(n_total)
    all_z_mode = np.zeros(n_total)
    all_z_best = np.zeros(n_total)
    all_z_mean_std = np.zeros(n_total)
    all_z_median_std = np.zeros(n_total)
    all_z_mode_std = np.zeros(n_total)
    all_z_best_std = np.zeros(n_total)
    all_plow95 = np.zeros(n_total)
    all_plow68 = np.zeros(n_total)
    all_phigh68 = np.zeros(n_total)
    all_phigh95 = np.zeros(n_total)
    all_z_mc = np.zeros(n_total)

    for fold in folds:
        fold_data = np.load(results_dir / f"fold_{fold}_results.npz")
        mask = cv == fold
        idx = np.where(mask)[0]

        all_pdfs[idx] = fold_data["pdfs"]
        all_z_mean[idx] = fold_data["z_mean"]
        all_z_median[idx] = fold_data["z_median"]
        all_z_mode[idx] = fold_data["z_mode"]
        all_z_best[idx] = fold_data["z_best"]
        all_z_mean_std[idx] = fold_data["z_mean_std"]
        all_z_median_std[idx] = fold_data["z_median_std"]
        all_z_mode_std[idx] = fold_data["z_mode_std"]
        all_z_best_std[idx] = fold_data["z_best_std"]
        all_plow95[idx] = fold_data["plow95"]
        all_plow68[idx] = fold_data["plow68"]
        all_phigh68[idx] = fold_data["phigh68"]
        all_phigh95[idx] = fold_data["phigh95"]
        all_z_mc[idx] = fold_data["z_mc"]

    # Reconstruct summary tuple for compute_all_metrics
    summary = (
        (all_z_mean, all_z_mean_std),
        (all_z_median, all_z_median_std),
        (all_z_mode, all_z_mode_std),
        (all_z_best, all_z_best_std),
        (all_plow95, all_plow68, all_phigh68, all_phigh95),
        all_z_mc,
    )

    return all_pdfs, zgrid, summary


# ============================================================================
# I. CLI and Main
# ============================================================================

def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="S23b photo-z training rehearsal with frankenz v0.4.0")
    parser.add_argument(
        "--catalog", type=str, default=DEFAULT_CATALOG,
        help="Path to S23b FITS catalog")
    parser.add_argument(
        "--config", type=str,
        default=str(script_dir / "frankenz_s23b_config.yaml"),
        help="Path to frankenz config YAML")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(script_dir / "output" / "undeblended"),
        help="Output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Batch chunk size")
    parser.add_argument(
        "--test-fold", type=int, default=None,
        help="Run single fold only (default: all folds)")
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-preparation of data")
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="Skip data preparation, load existing HDF5")
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip pipeline, load existing results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Frankenz v0.4.0 — S23b Photo-z Training Rehearsal")
    print("  Photometry: undeblended convolved (3.15 arcsec)")
    print("=" * 60)

    # --- Load config ---
    print(f"\nLoading config from {args.config}")
    config = FrankenzConfig.from_yaml(args.config)
    print(f"  Backend: {config.model.backend}, "
          f"K_tree={config.model.k_tree}, K_point={config.model.k_point}")
    print(f"  Transform: {config.transform.type}")
    print(f"  z-grid: [{config.zgrid.z_start}, {config.zgrid.z_end}] "
          f"dz={config.zgrid.z_delta}")

    # --- Prepare data ---
    import yaml as _yaml

    if args.skip_prep:
        print("\nSkipping data preparation (--skip-prep)")
        metadata_path = output_dir / "prepared" / "metadata.yaml"
        with open(metadata_path) as f:
            prep_metadata = _yaml.safe_load(f)
        # Load skynoise and patch config
        skynoise = np.array(prep_metadata["skynoise"])
        config.transform.skynoise = skynoise.tolist()
    else:
        catalog_data = load_catalog(args.catalog)
        catalog_data["catalog_path"] = args.catalog
        prep_metadata = prepare_all_folds(
            catalog_data, config, output_dir, force=args.force
        )
        skynoise = np.array(prep_metadata["skynoise"])

    print(f"\n  Skynoise (nJy): {[f'{s:.4f}' for s in skynoise]}")

    # --- Determine folds ---
    folds = prep_metadata["folds"]
    if args.test_fold is not None:
        if args.test_fold not in folds:
            print(f"ERROR: fold {args.test_fold} not in {folds}")
            sys.exit(1)
        folds = [args.test_fold]
    print(f"  Folds to process: {folds}")

    # --- Run pipeline per fold ---
    if not args.skip_run:
        print("\n" + "=" * 60)
        print("Running pipeline...")
        print("=" * 60)
        t0_total = time.time()
        for fold in folds:
            run_single_fold(fold, config, output_dir, args.chunk_size)
        elapsed_total = time.time() - t0_total
        print(f"\nAll folds complete in {elapsed_total:.1f}s")
    else:
        print("\nSkipping pipeline (--skip-run)")

    # --- Check that results exist ---
    results_dir = output_dir / "results"
    missing = [f for f in folds
               if not (results_dir / f"fold_{f}_results.npz").exists()]
    if missing:
        print(f"\nResults missing for folds {missing}. "
              f"Run without --skip-run first.")
        sys.exit(1)

    # --- Aggregate results ---
    print("\nAggregating results across folds...")
    pdfs, zgrid, summary = aggregate_fold_results(folds, output_dir)

    # Load auxiliary arrays
    aux = np.load(output_dir / "prepared" / "auxiliary.npz", allow_pickle=True)

    # If only processing a subset of folds, restrict to those objects
    cv = aux["sample_crossval"]
    if args.test_fold is not None:
        fold_mask = cv == args.test_fold
    else:
        fold_mask = np.ones(len(cv), dtype=bool)

    specz_sources = aux["specz_sources"][fold_mask]
    i_mag = aux["i_mag"][fold_mask]

    pdfs = pdfs[fold_mask]
    summary_masked = tuple(
        tuple(arr[fold_mask] for arr in part) if isinstance(part, tuple)
        else part[fold_mask] if isinstance(part, np.ndarray)
        else part
        for part in summary
    )

    # Load z_spec from test HDF5
    z_spec_parts = []
    for fold in folds:
        test_data = read_hdf5(output_dir / "prepared" / f"fold_{fold}_test.hdf5")
        z_spec_parts.append(test_data.redshifts)

    if len(folds) == 1:
        z_spec = z_spec_parts[0]
    else:
        # Reconstruct in original order
        z_spec_all = np.zeros(len(cv))
        for fold, zs in zip(folds, z_spec_parts):
            idx = np.where(cv == fold)[0]
            z_spec_all[idx] = zs
        z_spec = z_spec_all[fold_mask]

    # --- Compute metrics ---
    print("Computing evaluation metrics...")
    metrics = compute_all_metrics(
        z_spec, pdfs, zgrid, summary_masked, i_mag=i_mag
    )

    # Per-source metrics
    z_best = summary_masked[3][0]  # best_stats[0]
    source_metrics = compute_source_metrics(z_spec, z_best, specz_sources)

    # Print key metrics
    m_best = metrics["point"]["best"]
    print(f"\n  Best estimator:  bias={m_best['bias']:.5f}  "
          f"sigma_NMAD={m_best['sigma_nmad']:.5f}  "
          f"f_out={m_best['outlier_frac']:.4f}")
    pm = metrics["pdf"]
    print(f"  CRPS={pm['crps_mean']:.5f}  "
          f"Coverage(68%)={pm['coverage_68']:.4f}  "
          f"Coverage(95%)={pm['coverage_95']:.4f}")
    for name, sm in source_metrics.items():
        print(f"  {name}:  sigma_NMAD={sm['sigma_nmad']:.5f}  "
              f"f_out={sm['outlier_frac']:.4f}  N={sm['n_objects']}")

    # --- Generate figures ---
    fig_dir = output_dir / "figures"
    generate_all_figures(
        z_spec, metrics, pdfs, zgrid, i_mag,
        specz_sources, source_metrics, fig_dir
    )

    # --- Write reports ---
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = write_metrics_report(
        metrics, source_metrics, results_dir / "metrics.txt"
    )
    print("\n" + report)

    write_metrics_json(metrics, source_metrics, results_dir / "metrics.json")

    write_report_markdown(
        config, prep_metadata, metrics, source_metrics, skynoise,
        output_dir / "report.md"
    )

    # --- Save combined results ---
    save_combined_results(
        pdfs, zgrid, metrics, results_dir / "combined_results.npz"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
