#!/usr/bin/env python
"""
Production photo-z estimation with frankenz v0.4.0.

Runs KMCkNN photometric redshift inference on HSC S20 training sample data,
computes comprehensive evaluation metrics, and generates publication-quality
QA figures.

Usage:
    python example/run_photoz.py
    python example/run_photoz.py --config path/to/config.yaml
    python example/run_photoz.py --chunk-size 500 --output-dir results/

Requires: frankenz[all] (h5py, astropy, tqdm)
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
from astropy import units as u
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# frankenz imports
from frankenz.config import FrankenzConfig
from frankenz.io import PhotoData
from frankenz.batch import run_pipeline
from frankenz.pdf import pdfs_summarize


# ============================================================================
# A. Data Loading
# ============================================================================

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
BAND_NAMES = ["g", "r", "i", "z", "y"]

# Fixed KDE bandwidth for spectroscopic redshift errors.
# The old frankenz4DESI pipeline uses ENABLE_ZERR=False with ZSMOOTH=0.01,
# ignoring actual zerr values and using a fixed bandwidth.  We replicate
# that here because the HSC HDF5 zerr column contains ~31% sentinel values
# (-9, 0, 99) that would produce over-smoothed PDFs if used as bandwidth.
ZSMOOTH = 0.01


def load_hdf5_with_header(path):
    """Load an HDF5 file that has a YAML header prepended.

    Scans for the HDF5 magic bytes and opens from that offset using
    an in-memory buffer (no temp files).
    """
    import h5py

    path = Path(path)
    raw = path.read_bytes()
    offset = raw.find(HDF5_MAGIC)
    if offset < 0:
        raise ValueError(f"No HDF5 magic bytes found in {path}")
    if offset > 0:
        print(f"  Skipping {offset}-byte YAML header in {path.name}")
    buf = io.BytesIO(raw[offset:])
    return h5py.File(buf, "r")


def load_hsc_data(path):
    """Load HSC photometry from a frankenz4DESI-format HDF5 file.

    Applies flag filtering and aperture correction following the
    frankenz4DESI convention:
        corrected_mag = mag(flux) - a
        corrected_flux = 10^(-0.4 * corrected_mag) * 3631e6  [uJy]

    Error arrays are NOT aperture-corrected (following frankenz4DESI).

    Returns a PhotoData container.
    """
    f = load_hdf5_with_header(path)

    raw_flux = np.array(f["cmodel/flux"])
    raw_err = np.array(f["cmodel/err"])
    raw_flag = np.array(f["cmodel/flag"])
    raw_a = np.array(f["a"])

    # Flag filter: keep objects where all bands have flag == 0
    good = raw_flag.sum(axis=1) == 0
    flux = raw_flux[good]
    err = raw_err[good]
    a_corr = raw_a[good]

    n_total = len(raw_flux)
    n_good = int(good.sum())
    print(f"  Flag filter: {n_good}/{n_total} objects pass "
          f"({n_good / n_total * 100:.1f}%)")

    # Aperture correction: flux -> mag -> subtract a -> flux
    mag = (flux * u.uJy).to(u.ABmag).value
    corrected_mag = mag - a_corr
    corrected_flux = (corrected_mag * u.ABmag).to(u.uJy).value

    # Load metadata
    redshifts = np.array(f["z"])[good] if "z" in f else None
    object_ids = np.array(f["object_id"])[good] if "object_id" in f else None

    # Use fixed KDE bandwidth (ZSMOOTH) instead of raw zerr values.
    # The HSC zerr column has ~31% sentinel values (-9, 0, 99) that would
    # produce over-smoothed PDFs.  The old frankenz4DESI pipeline ignores
    # zerr entirely (ENABLE_ZERR=false) and uses ZSMOOTH=0.01 as a fixed
    # Gaussian KDE bandwidth for all training objects.
    n_good = int(good.sum())
    redshift_errs = np.full(n_good, ZSMOOTH)
    print(f"  Using fixed KDE bandwidth (ZSMOOTH={ZSMOOTH})")

    f.close()

    data = PhotoData(
        flux=corrected_flux.astype(np.float64),
        flux_err=err.astype(np.float64),
        mask=np.ones_like(corrected_flux, dtype=int),
        redshifts=redshifts.astype(np.float64) if redshifts is not None else None,
        redshift_errs=redshift_errs,
        object_ids=object_ids,
        band_names=BAND_NAMES,
    )
    data.validate()
    return data


# ============================================================================
# B. Metrics Computation
# ============================================================================

def compute_point_metrics(z_spec, z_phot):
    """Compute standard photo-z point estimate metrics.

    Returns dict with bias, sigma_nmad, outlier_frac, catastrophic_frac, rms.
    """
    dz = (z_phot - z_spec) / (1.0 + z_spec)
    return {
        "bias": np.median(dz),
        "sigma_nmad": 1.4826 * np.median(np.abs(dz - np.median(dz))),
        "outlier_frac": np.mean(np.abs(dz) > 0.15),
        "catastrophic_frac": np.mean(np.abs(dz) > 0.5),
        "rms": np.sqrt(np.mean(dz ** 2)),
        "n_objects": len(z_spec),
    }


def compute_pit(pdfs, zgrid, z_spec):
    """Compute Probability Integral Transform values.

    PIT = CDF(z_spec) for each object. Should be uniformly distributed
    for well-calibrated PDFs.
    """
    cdfs = np.cumsum(pdfs, axis=1)
    # Normalize CDFs to [0, 1]
    cdf_norm = cdfs / cdfs[:, -1:]
    pit = np.array([
        np.interp(z_spec[i], zgrid, cdf_norm[i])
        for i in range(len(z_spec))
    ])
    return pit


def compute_crps(pdfs, zgrid, z_spec):
    """Compute Continuous Ranked Probability Score per object.

    CRPS = integral of (CDF(z) - H(z - z_spec))^2 dz
    Lower is better.
    """
    cdfs = np.cumsum(pdfs, axis=1)
    cdf_norm = cdfs / cdfs[:, -1:]
    crps = np.zeros(len(z_spec))
    for i in range(len(z_spec)):
        heaviside = (zgrid >= z_spec[i]).astype(float)
        crps[i] = np.trapezoid((cdf_norm[i] - heaviside) ** 2, zgrid)
    return crps


def compute_coverage(z_spec, intervals):
    """Compute empirical coverage fractions for credible intervals.

    Parameters
    ----------
    z_spec : array
    intervals : tuple (plow95, plow68, phigh68, phigh95)

    Returns dict with coverage_68 and coverage_95.
    """
    plow95, plow68, phigh68, phigh95 = intervals
    cov_68 = np.mean((z_spec >= plow68) & (z_spec <= phigh68))
    cov_95 = np.mean((z_spec >= plow95) & (z_spec <= phigh95))
    return {"coverage_68": cov_68, "coverage_95": cov_95}


def compute_binned_metrics(z_spec, z_phot, bin_values, n_bins=10):
    """Compute bias, sigma_nmad, outlier_frac in bins of bin_values.

    Returns dict with bin_centers, bias, sigma_nmad, outlier_frac arrays.
    """
    bin_edges = np.percentile(bin_values, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges
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
    """Compute all evaluation metrics.

    Parameters
    ----------
    z_spec : array of true redshifts
    pdfs : array of PDFs
    zgrid : redshift grid
    summary : output of pdfs_summarize (6-tuple)
    i_mag : optional i-band magnitudes for binned metrics

    Returns nested dict of all metrics.
    """
    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary

    # Point estimate metrics for each estimator
    estimators = {
        "mean": mean_stats[0],
        "median": median_stats[0],
        "mode": mode_stats[0],
        "best": best_stats[0],
    }
    point_metrics = {}
    for name, z_phot in estimators.items():
        point_metrics[name] = compute_point_metrics(z_spec, z_phot)

    # PDF quality metrics
    pit = compute_pit(pdfs, zgrid, z_spec)
    crps = compute_crps(pdfs, zgrid, z_spec)
    ks_stat, ks_pvalue = stats.kstest(pit, "uniform")
    coverage = compute_coverage(z_spec, intervals)

    pdf_metrics = {
        "pit": pit,
        "crps": crps,
        "crps_mean": np.mean(crps),
        "crps_median": np.median(crps),
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        **coverage,
    }

    # Binned metrics (using best estimator)
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


# ============================================================================
# C. Visualization — 10 QA figures
# ============================================================================

FIGSIZE = (8, 6)
DPI = 150


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
        # 1:1 line
        ax.plot([0, z_max], [0, z_max], "r-", lw=0.8, alpha=0.7)
        # ±0.15(1+z) envelope
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

    # Gaussian fit overlay
    mu, sigma = metrics_best["bias"], metrics_best["sigma_nmad"]
    x = np.linspace(-0.5, 0.5, 300)
    gauss = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, gauss, "r-", lw=2,
            label=f"Gaussian ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})")

    # Outlier boundaries
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

    # Running statistics
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

    # Running statistics
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

    # PIT histogram
    ax1.hist(pit, bins=20, density=True, color="steelblue", alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax1.axhline(1.0, color="red", ls="--", lw=1.5, label="Uniform")
    ax1.set_xlabel("PIT value")
    ax1.set_ylabel("Density")
    ax1.set_title("PIT Histogram")
    ax1.set_xlim(0, 1)
    ax1.legend()

    # QQ plot: empirical CDF of PIT vs uniform CDF
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

    # True N(z) histogram
    z_max = min(z_spec.max() * 1.1, 4.0)
    bins = np.arange(0, z_max, 0.05)
    ax.hist(z_spec, bins=bins, density=True, color="steelblue", alpha=0.5,
            edgecolor="white", linewidth=0.3, label="$z_{\\rm spec}$ (true)")

    # Stacked PDFs
    nz_pdf = pdfs.sum(axis=0)
    nz_pdf = nz_pdf / np.trapezoid(nz_pdf, zgrid)  # normalize to density
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

    # Select representative objects
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
            mask = zgrid <= z_max_plot
            ax.plot(zgrid[mask], pdfs[idx, mask], "b-", lw=1.5)
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
    # Compute CDFs
    cdfs = np.cumsum(pdfs, axis=1)
    cdf_norm = cdfs / cdfs[:, -1:]

    # Nominal levels: symmetric around 50%
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


def generate_all_figures(z_spec, metrics, pdfs, zgrid, i_mag, fig_dir):
    """Generate all 10 QA figures."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = metrics["summary"]
    _, _, _, best_stats, intervals, _ = summary
    z_best = best_stats[0]
    m_best = metrics["point"]["best"]

    print("\nGenerating figures...")

    print("  [01/10] Scatter 4-panel")
    fig_01_scatter_4panel(
        z_spec, metrics["estimators"], metrics,
        fig_dir / "fig_01_scatter_4panel.png")

    print("  [02/10] Residual histogram")
    fig_02_residual_histogram(z_spec, z_best, m_best,
                              fig_dir / "fig_02_residual_histogram.png")

    print("  [03/10] Residual vs z_spec")
    fig_03_residual_vs_zspec(z_spec, z_best,
                             fig_dir / "fig_03_residual_vs_zspec.png")

    print("  [04/10] Residual vs i-mag")
    fig_04_residual_vs_mag(z_spec, z_best, i_mag,
                           fig_dir / "fig_04_residual_vs_mag.png")

    print("  [05/10] PIT + QQ")
    fig_05_pit_qq(
        metrics["pdf"]["pit"],
        metrics["pdf"]["ks_stat"], metrics["pdf"]["ks_pvalue"],
        fig_dir / "fig_05_pit_qq.png")

    print("  [06/10] N(z) comparison")
    fig_06_nz_comparison(z_spec, pdfs, zgrid,
                         fig_dir / "fig_06_nz_comparison.png")

    print("  [07/10] Example PDFs")
    fig_07_example_pdfs(z_spec, z_best, pdfs, zgrid,
                        fig_dir / "fig_07_example_pdfs.png")

    print("  [08/10] Metrics vs z_spec")
    fig_08_metrics_vs_zbin(metrics["binned"]["vs_zspec"],
                           fig_dir / "fig_08_metrics_vs_zbin.png")

    if "vs_imag" in metrics["binned"]:
        print("  [09/10] Metrics vs i-mag")
        fig_09_metrics_vs_magbin(metrics["binned"]["vs_imag"],
                                 fig_dir / "fig_09_metrics_vs_magbin.png")

    print("  [10/10] Credible coverage")
    fig_10_credible_coverage(z_spec, pdfs, zgrid,
                             fig_dir / "fig_10_credible_coverage.png")

    print(f"  All figures saved to {fig_dir}/")


# ============================================================================
# D. Report Generation
# ============================================================================

def write_metrics_report(metrics, output_path):
    """Write human-readable evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("Frankenz v0.4.0 — Photo-z Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    # Point estimate metrics
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

    # PDF quality metrics
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

    # Binned metrics
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


# ============================================================================
# E. Results Saving
# ============================================================================

def save_results(pdfs, zgrid, metrics, output_path):
    """Save PDFs, point estimates, and metric arrays to NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = metrics["summary"]
    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary

    arrays = {
        "pdfs": pdfs,
        "zgrid": zgrid,
        # Point estimates
        "z_mean": mean_stats[0],
        "z_median": median_stats[0],
        "z_mode": mode_stats[0],
        "z_best": best_stats[0],
        # Uncertainties (std dev around estimator)
        "z_mean_std": mean_stats[1],
        "z_median_std": median_stats[1],
        "z_mode_std": mode_stats[1],
        "z_best_std": best_stats[1],
        # Credible intervals
        "plow95": intervals[0],
        "plow68": intervals[1],
        "phigh68": intervals[2],
        "phigh95": intervals[3],
        # Monte Carlo realization
        "z_mc": mc,
        # PDF quality
        "pit": metrics["pdf"]["pit"],
        "crps": metrics["pdf"]["crps"],
    }
    np.savez(output_path, **arrays)
    print(f"Results saved to {output_path}")


# ============================================================================
# F. CLI and Main
# ============================================================================

def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Frankenz v0.4.0 photo-z estimation with QA")
    parser.add_argument(
        "--config", type=str,
        default=str(script_dir / "frankenz_config.yaml"),
        help="Path to frankenz config YAML")
    parser.add_argument(
        "--train", type=str,
        default=str(script_dir / "s20train-train.hdf5"),
        help="Path to training data HDF5")
    parser.add_argument(
        "--test", type=str,
        default=str(script_dir / "s20train-test.hdf5"),
        help="Path to test data HDF5")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(script_dir / "output"),
        help="Output directory")
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Chunk size for batch processing")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Frankenz v0.4.0 — Photo-z Estimation Pipeline")
    print("=" * 60)

    # --- Load config ---
    print(f"\nLoading config from {args.config}")
    config = FrankenzConfig.from_yaml(args.config)
    print(f"  Backend: {config.model.backend}, "
          f"K_tree={config.model.k_tree}, K_point={config.model.k_point}")
    print(f"  Transform: {config.transform.type}")
    print(f"  z-grid: [{config.zgrid.z_start}, {config.zgrid.z_end}] "
          f"dz={config.zgrid.z_delta}")

    # --- Load data ---
    print(f"\nLoading training data from {args.train}")
    train_data = load_hsc_data(args.train)
    print(f"  Training: {train_data.n_objects} objects, "
          f"{train_data.n_bands} bands")

    print(f"\nLoading test data from {args.test}")
    test_data = load_hsc_data(args.test)
    print(f"  Test: {test_data.n_objects} objects, "
          f"{test_data.n_bands} bands")

    # Compute i-band magnitudes for QA (band index 2 = i)
    with np.errstate(divide="ignore", invalid="ignore"):
        i_flux = test_data.flux[:, 2]
        i_mag = -2.5 * np.log10(i_flux) + 23.9  # AB mag from uJy

    # --- Run pipeline ---
    print(f"\nRunning pipeline (chunk_size={args.chunk_size})...")
    t0 = time.time()
    result = run_pipeline(
        config, train_data, test_data, chunk_size=args.chunk_size)
    elapsed = time.time() - t0
    print(f"Pipeline complete in {elapsed:.1f}s "
          f"({test_data.n_objects / elapsed:.0f} obj/s)")

    pdfs = result.pdfs
    zgrid = result.zgrid
    z_spec = test_data.redshifts

    # --- Compute summary statistics ---
    print("\nComputing PDF summary statistics...")
    summary = pdfs_summarize(pdfs, zgrid, renormalize=True)

    # --- Compute metrics ---
    print("Computing evaluation metrics...")
    metrics = compute_all_metrics(z_spec, pdfs, zgrid, summary, i_mag=i_mag)

    # Print key metrics
    m_best = metrics["point"]["best"]
    print(f"\n  Best estimator:  bias={m_best['bias']:.5f}  "
          f"sigma_NMAD={m_best['sigma_nmad']:.5f}  "
          f"f_out={m_best['outlier_frac']:.4f}")
    pm = metrics["pdf"]
    print(f"  CRPS={pm['crps_mean']:.5f}  "
          f"Coverage(68%)={pm['coverage_68']:.4f}  "
          f"Coverage(95%)={pm['coverage_95']:.4f}")

    # --- Generate figures ---
    fig_dir = output_dir / "figures"
    generate_all_figures(z_spec, metrics, pdfs, zgrid, i_mag, fig_dir)

    # --- Write report ---
    report = write_metrics_report(metrics, output_dir / "metrics.txt")
    print("\n" + report)

    # --- Save results ---
    save_results(pdfs, zgrid, metrics, output_dir / "results.npz")

    print("\nDone!")


if __name__ == "__main__":
    main()
