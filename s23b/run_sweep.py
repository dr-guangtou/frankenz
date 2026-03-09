#!/usr/bin/env python
"""
S23b Frankenz parameter sweep.

Systematically varies KNN, S/N cap, and KDE bandwidth parameters
to find optimal configuration for photo-z estimation.

Usage:
    python s23b/run_sweep.py --phase A     # KNN + S/N cap screening
    python s23b/run_sweep.py --phase D     # KDE bandwidth sweep
    python s23b/run_sweep.py --phase E     # Final combined run

Requires prepared data from run_photoz_s23b.py (--skip-run OK).
"""

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from frankenz.config import FrankenzConfig
from frankenz.fitting import get_fitter
from frankenz.io import PhotoData, read_hdf5, write_hdf5
from frankenz.batch import run_pipeline
from frankenz.pdf import logprob, gauss_kde, pdfs_summarize

from run_photoz_s23b import (
    BANDS, N_BANDS, AB_ZP, SNR_CAP,
    flux_to_ab_mag, apply_snr_cap, build_photo_data,
    compute_point_metrics, compute_pit, compute_crps, compute_coverage,
    compute_source_metrics, save_fold_results,
    run_single_fold_with_intermediates, resweep_bandwidth,
)

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


# ============================================================================
# Sweep configuration definitions
# ============================================================================

# S/N cap configurations
SNR_CAP_CONFIGS = {
    "baseline":   [100.0, 100.0, 100.0, 80.0, 50.0],
    "s19a":       [100.0, 100.0, 100.0, 80.0, 80.0],
    "aggressive": [50.0,  50.0,  50.0,  50.0, 50.0],
    "relaxed":    [200.0, 200.0, 200.0, 100.0, 100.0],
    "no_cap":     [np.inf, np.inf, np.inf, np.inf, np.inf],
}

# KDE bandwidth configurations
BANDWIDTH_CONFIGS = [
    (0.01, 0.01),
    (0.05, 0.03),
    (0.10, 0.05),
    (0.15, 0.08),
    (0.15, 0.10),
    (0.20, 0.10),
    (0.20, 0.15),
    (0.25, 0.10),
    (0.30, 0.15),
]


# ============================================================================
# Data loading utilities
# ============================================================================

def load_prepared_data(output_dir):
    """Load auxiliary data and metadata from prepared directory.

    If auxiliary.npz lacks flux_corrected/flux_err_clean (older format),
    re-extracts them from the FITS catalog and saves an augmented version.
    """
    import yaml

    prepared_dir = Path(output_dir) / "prepared"
    metadata_path = prepared_dir / "metadata.yaml"

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    aux_path = prepared_dir / "auxiliary.npz"
    aux = np.load(aux_path, allow_pickle=True)

    if "flux_corrected" not in aux:
        print("Augmenting auxiliary.npz with flux arrays "
              "(one-time operation)...")
        from run_photoz_s23b import (
            load_catalog, apply_quality_cuts, apply_extinction_correction,
            extract_photometry, extract_extinction,
            DEFAULT_CATALOG,
        )

        catalog_path = metadata.get("catalog", DEFAULT_CATALOG)
        catalog_data = load_catalog(catalog_path)

        flux = catalog_data["flux"]
        flux_err = catalog_data["flux_err"]
        extinction = catalog_data["extinction"]
        redshift = catalog_data["redshift"]
        object_type = catalog_data["object_type"]
        object_id = catalog_data["object_id"]

        clean_mask, _ = apply_quality_cuts(
            flux, flux_err, redshift, object_type)

        flux_clean = flux[clean_mask]
        flux_err_clean = flux_err[clean_mask]
        ext_clean = extinction[clean_mask]
        z_clean = redshift[clean_mask]
        oid_clean = object_id[clean_mask]

        flux_corrected = apply_extinction_correction(flux_clean, ext_clean)

        # Save augmented auxiliary
        aug_data = {k: aux[k] for k in aux.files}
        aug_data["flux_corrected"] = flux_corrected
        aug_data["flux_err_clean"] = flux_err_clean
        aug_data["redshift"] = z_clean
        aug_data["object_id"] = oid_clean

        np.savez(aux_path, **aug_data)
        print(f"  Saved augmented auxiliary ({aux_path})")

        # Reload
        aux = np.load(aux_path, allow_pickle=True)

    return metadata, aux


def build_fold_data(aux, fold, config, snr_cap=None):
    """Build train/test PhotoData for a fold with optional S/N cap override.

    Loads pre-cap flux and errors from auxiliary, applies the specified S/N
    cap, computes KDE bandwidth, and returns (train_data, test_data, i_mag_test).
    """
    cv = aux["sample_crossval"]
    flux_corrected = aux["flux_corrected"]
    flux_err_clean = aux["flux_err_clean"]
    redshift = aux["redshift"]
    i_mag = aux["i_mag"]

    if snr_cap is None:
        snr_cap = SNR_CAP

    # Apply S/N cap
    flux_err_capped = apply_snr_cap(flux_corrected, flux_err_clean,
                                     snr_cap=snr_cap)

    # KDE bandwidth
    bw_frac = config.pdf.kde_bandwidth_fraction
    bw_floor = config.pdf.kde_bandwidth_floor
    kde_bw = np.maximum(bw_frac * redshift, bw_floor)

    # Split
    test_mask = cv == fold
    train_mask = ~test_mask

    # Use dummy object IDs (simple integers)
    n_total = len(redshift)
    object_ids = np.arange(n_total)

    train_data = build_photo_data(
        flux_corrected[train_mask], flux_err_capped[train_mask],
        redshift[train_mask], kde_bw[train_mask], object_ids[train_mask],
    )
    test_data = build_photo_data(
        flux_corrected[test_mask], flux_err_capped[test_mask],
        redshift[test_mask], kde_bw[test_mask], object_ids[test_mask],
    )

    return train_data, test_data, i_mag[test_mask]


# ============================================================================
# Quick metrics computation (no figures, just numbers)
# ============================================================================

def quick_metrics(z_spec, pdfs, zgrid, summary, i_mag=None,
                  specz_sources=None):
    """Compute key metrics quickly (no figures, no binned metrics)."""
    mean_stats, median_stats, mode_stats, best_stats, intervals, mc = summary
    z_best = best_stats[0]

    point = compute_point_metrics(z_spec, z_best)

    pit = compute_pit(pdfs, zgrid, z_spec)
    crps = compute_crps(pdfs, zgrid, z_spec)
    from scipy import stats as sp_stats
    ks_stat, ks_pvalue = sp_stats.kstest(pit, "uniform")
    coverage = compute_coverage(z_spec, intervals)

    result = {
        "bias": point["bias"],
        "sigma_nmad": point["sigma_nmad"],
        "f_outlier": point["outlier_frac"],
        "f_catastrophic": point["catastrophic_frac"],
        "rms": point["rms"],
        "n_objects": point["n_objects"],
        "crps_mean": float(np.mean(crps)),
        "coverage_68": coverage["coverage_68"],
        "coverage_95": coverage["coverage_95"],
        "ks_stat": float(ks_stat),
    }

    # Per-source metrics
    if specz_sources is not None:
        src = compute_source_metrics(z_spec, z_best, specz_sources)
        for name, sm in src.items():
            result[f"{name}_sigma_nmad"] = sm["sigma_nmad"]
            result[f"{name}_f_outlier"] = sm["outlier_frac"]
            result[f"{name}_n"] = sm["n_objects"]

    return result


# ============================================================================
# Phase A: KNN + S/N cap screening
# ============================================================================

def run_phase_a(config, output_dir, folds, chunk_size):
    """Screen KNN and S/N cap parameters on specified folds."""
    metadata, aux = load_prepared_data(output_dir)
    cv = aux["sample_crossval"]
    specz_sources_all = aux["specz_sources"]

    sweep_dir = Path(output_dir) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ---- Define all sweep configs ----
    sweep_configs = []

    # Baseline
    sweep_configs.append(("baseline", {}))

    # k_point variations
    for kp in [10, 15, 30, 50]:
        sweep_configs.append((f"k_point_{kp}", {"k_point": kp}))

    # k_tree variations
    for kt in [10, 15, 40, 60]:
        sweep_configs.append((f"k_tree_{kt}", {"k_tree": kt}))

    # dim_prior off
    sweep_configs.append(("dim_prior_false", {"dim_prior": False}))

    # ignore_model_err on
    sweep_configs.append(("ignore_model_err", {"ignore_model_err": True}))

    # free_scale on
    sweep_configs.append(("free_scale", {"free_scale": True}))

    # S/N cap variations
    for cap_name, cap_vals in SNR_CAP_CONFIGS.items():
        if cap_name == "baseline":
            continue  # already included
        sweep_configs.append((f"snrcap_{cap_name}", {"snr_cap": cap_vals}))

    total_runs = len(sweep_configs) * len(folds)
    run_idx = 0

    for config_name, overrides in sweep_configs:
        # Build per-run config
        run_config = FrankenzConfig.from_dict(config.to_dict())

        snr_cap = overrides.pop("snr_cap", None)

        for key, val in overrides.items():
            if hasattr(run_config.model, key):
                setattr(run_config.model, key, val)

        for fold in folds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {config_name} fold={fold}")

            t0 = time.time()

            # Build data with potentially different S/N cap
            train_data, test_data, i_mag_test = build_fold_data(
                aux, fold, run_config, snr_cap=snr_cap)

            # Update skynoise in config from prepared metadata
            run_config.transform.skynoise = metadata["skynoise"]

            # Run pipeline
            result = run_pipeline(
                run_config, train_data, test_data, chunk_size=chunk_size)

            elapsed = time.time() - t0
            pdfs = result.pdfs
            zgrid = result.zgrid

            # Get z_spec for test fold
            z_spec = test_data.redshifts
            specz_test = specz_sources_all[cv == fold]

            # Compute summary + metrics
            summary = pdfs_summarize(pdfs, zgrid, renormalize=True)
            metrics = quick_metrics(
                z_spec, pdfs, zgrid, summary, specz_sources=specz_test)

            metrics["config"] = config_name
            metrics["fold"] = fold
            metrics["elapsed_s"] = elapsed
            results.append(metrics)

            print(f"  sNMAD={metrics['sigma_nmad']:.4f}  "
                  f"f_out={metrics['f_outlier']:.3f}  "
                  f"cov68={metrics['coverage_68']:.3f}  "
                  f"t={elapsed:.1f}s")

    # Aggregate results
    df = pd.DataFrame(results)
    csv_path = sweep_dir / "phase_a_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Phase A results saved to {csv_path}")
    print(f"{'='*60}")

    # Print summary table (averaged across folds)
    summary_df = df.groupby("config").agg({
        "sigma_nmad": "mean",
        "f_outlier": "mean",
        "coverage_68": "mean",
        "coverage_95": "mean",
        "elapsed_s": "sum",
    }).round(4)
    summary_df = summary_df.sort_values("sigma_nmad")
    print("\nPhase A Summary (averaged across folds):")
    print(summary_df.to_string())

    # Per-source if available
    for src in ["DESI", "COSMOSWeb"]:
        col = f"{src}_sigma_nmad"
        if col in df.columns:
            src_summary = df.groupby("config")[col].mean().sort_values()
            print(f"\n{src} sigma_NMAD (mean across folds):")
            print(src_summary.to_string())

    return df


# ============================================================================
# Phase C: Full CV confirmation
# ============================================================================

def run_phase_c(config, output_dir, top_configs, chunk_size):
    """Run full 8-fold CV for top configs from Phase A, with intermediates."""
    metadata, aux = load_prepared_data(output_dir)
    folds = metadata["folds"]

    sweep_dir = Path(output_dir) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for config_name, overrides in top_configs:
        run_config = FrankenzConfig.from_dict(config.to_dict())
        run_config.transform.skynoise = metadata["skynoise"]

        snr_cap = overrides.pop("snr_cap", None) if "snr_cap" in overrides else None

        for key, val in overrides.items():
            if hasattr(run_config.model, key):
                setattr(run_config.model, key, val)

        # Create per-config output directory
        config_dir = sweep_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for each fold
        prepared_dir = config_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        results_dir = config_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = config_dir / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Phase C: {config_name} — all {len(folds)} folds")
        print(f"{'='*60}")

        t0_all = time.time()
        cv = aux["sample_crossval"]
        specz_sources_all = aux["specz_sources"]

        for fold in folds:
            # Build data
            train_data, test_data, _ = build_fold_data(
                aux, fold, run_config, snr_cap=snr_cap)

            # Write HDF5 so intermediates function can find them
            write_hdf5(train_data, prepared_dir / f"fold_{fold}_train.hdf5")
            write_hdf5(test_data, prepared_dir / f"fold_{fold}_test.hdf5")

        # Copy auxiliary.npz to config dir (needed by aggregate_fold_results)
        import shutil
        src_aux = Path(output_dir) / "prepared" / "auxiliary.npz"
        shutil.copy2(src_aux, prepared_dir / "auxiliary.npz")

        # Now run with intermediates
        for fold in folds:
            run_single_fold_with_intermediates(
                fold, run_config, config_dir, chunk_size)

        elapsed_all = time.time() - t0_all
        print(f"  All folds complete in {elapsed_all:.1f}s")

        # Aggregate and compute metrics
        from run_photoz_s23b import aggregate_fold_results
        pdfs, zgrid, summary = aggregate_fold_results(folds, config_dir)

        # Reconstruct z_spec in original order
        z_spec_all = np.zeros(len(cv))
        for fold in folds:
            td = read_hdf5(prepared_dir / f"fold_{fold}_test.hdf5")
            idx = np.where(cv == fold)[0]
            z_spec_all[idx] = td.redshifts
        z_spec = z_spec_all

        metrics = quick_metrics(
            z_spec, pdfs, zgrid, summary,
            specz_sources=specz_sources_all)
        metrics["config"] = config_name
        metrics["elapsed_s"] = elapsed_all
        results.append(metrics)

        print(f"  sNMAD={metrics['sigma_nmad']:.4f}  "
              f"f_out={metrics['f_outlier']:.3f}  "
              f"cov68={metrics['coverage_68']:.3f}")

    df = pd.DataFrame(results)
    csv_path = sweep_dir / "phase_c_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPhase C results saved to {csv_path}")
    print(df.to_string())

    return df


# ============================================================================
# Phase D: KDE bandwidth sweep (using saved intermediates)
# ============================================================================

def run_phase_d(config, output_dir, config_name="baseline"):
    """Sweep KDE bandwidth using saved intermediates from Phase C."""
    metadata, aux = load_prepared_data(output_dir)
    folds = metadata["folds"]
    cv = aux["sample_crossval"]
    specz_sources_all = aux["specz_sources"]

    sweep_dir = Path(output_dir) / "sweep"
    config_dir = sweep_dir / config_name

    # Check intermediates exist
    inter_dir = config_dir / "intermediates"
    if not inter_dir.exists():
        # Fall back to main output dir
        inter_dir = Path(output_dir) / "intermediates"
        config_dir = Path(output_dir)
    if not inter_dir.exists():
        print(f"ERROR: No intermediates found at {inter_dir}")
        print("Run Phase C with --save-intermediates first, or run Phase A "
              "with --save-intermediates.")
        sys.exit(1)

    zgrid = np.arange(
        config.zgrid.z_start,
        config.zgrid.z_end + config.zgrid.z_delta * 0.5,
        config.zgrid.z_delta,
    )

    results = []

    for bw_frac, bw_floor in BANDWIDTH_CONFIGS:
        bw_name = f"bw_{bw_frac:.2f}_{bw_floor:.2f}"
        print(f"\n--- {bw_name} ---")
        t0 = time.time()

        # Re-sweep all folds
        all_pdfs = np.zeros((len(cv), len(zgrid)))
        all_z_best = np.zeros(len(cv))
        all_plow68 = np.zeros(len(cv))
        all_phigh68 = np.zeros(len(cv))
        all_plow95 = np.zeros(len(cv))
        all_phigh95 = np.zeros(len(cv))

        for fold in folds:
            pdfs_fold, summary_fold = resweep_bandwidth(
                fold, bw_frac, bw_floor, str(config_dir), zgrid)

            idx = np.where(cv == fold)[0]
            all_pdfs[idx] = pdfs_fold

            mean_s, median_s, mode_s, best_s, intervals_s, mc_s = summary_fold
            all_z_best[idx] = best_s[0]
            all_plow95[idx] = intervals_s[0]
            all_plow68[idx] = intervals_s[1]
            all_phigh68[idx] = intervals_s[2]
            all_phigh95[idx] = intervals_s[3]

        elapsed = time.time() - t0

        # Reconstruct z_spec
        prepared_dir = config_dir / "prepared"
        z_spec = np.zeros(len(cv))
        for fold in folds:
            td = read_hdf5(prepared_dir / f"fold_{fold}_test.hdf5")
            idx = np.where(cv == fold)[0]
            z_spec[idx] = td.redshifts

        # Compute metrics directly from z_best + intervals
        point = compute_point_metrics(z_spec, all_z_best)
        coverage = compute_coverage(
            z_spec, (all_plow95, all_plow68, all_phigh68, all_phigh95))

        pit = compute_pit(all_pdfs, zgrid, z_spec)
        crps = compute_crps(all_pdfs, zgrid, z_spec)
        from scipy import stats as sp_stats
        ks_stat, _ = sp_stats.kstest(pit, "uniform")

        metrics = {
            "bw_frac": bw_frac,
            "bw_floor": bw_floor,
            "bias": point["bias"],
            "sigma_nmad": point["sigma_nmad"],
            "f_outlier": point["outlier_frac"],
            "coverage_68": coverage["coverage_68"],
            "coverage_95": coverage["coverage_95"],
            "crps_mean": float(np.mean(crps)),
            "ks_stat": float(ks_stat),
            "elapsed_s": elapsed,
        }

        # Per-source
        src = compute_source_metrics(z_spec, all_z_best, specz_sources_all)
        for name, sm in src.items():
            metrics[f"{name}_sigma_nmad"] = sm["sigma_nmad"]
            metrics[f"{name}_f_outlier"] = sm["outlier_frac"]

        results.append(metrics)
        print(f"  sNMAD={metrics['sigma_nmad']:.4f}  "
              f"cov68={metrics['coverage_68']:.3f}  "
              f"cov95={metrics['coverage_95']:.3f}  "
              f"t={elapsed:.1f}s")

    df = pd.DataFrame(results)
    csv_path = sweep_dir / f"phase_d_results_{config_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Phase D results saved to {csv_path}")
    print(f"{'='*60}")
    print(df[["bw_frac", "bw_floor", "sigma_nmad", "f_outlier",
              "coverage_68", "coverage_95"]].to_string(index=False))

    return df


# ============================================================================
# Phase E: Final combined run
# ============================================================================

def run_phase_e(config, output_dir, best_overrides, best_snr_cap,
                best_bw_frac, best_bw_floor, chunk_size):
    """Final run with best parameters, full figures and reports."""
    metadata, aux = load_prepared_data(output_dir)
    folds = metadata["folds"]
    cv = aux["sample_crossval"]

    # Build final config
    final_config = FrankenzConfig.from_dict(config.to_dict())
    final_config.transform.skynoise = metadata["skynoise"]
    final_config.pdf.kde_bandwidth_fraction = best_bw_frac
    final_config.pdf.kde_bandwidth_floor = best_bw_floor

    for key, val in best_overrides.items():
        if hasattr(final_config.model, key):
            setattr(final_config.model, key, val)

    # Create final output directory
    final_dir = Path(output_dir) / "sweep" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = final_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Phase E: Final combined run")
    print(f"  k_point={final_config.model.k_point}, "
          f"k_tree={final_config.model.k_tree}")
    print(f"  dim_prior={final_config.model.dim_prior}, "
          f"free_scale={final_config.model.free_scale}")
    print(f"  bw_frac={best_bw_frac}, bw_floor={best_bw_floor}")
    print(f"  snr_cap={best_snr_cap}")
    print(f"{'='*60}")

    # Prepare fold data
    for fold in folds:
        train_data, test_data, _ = build_fold_data(
            aux, fold, final_config, snr_cap=best_snr_cap)
        write_hdf5(train_data, prepared_dir / f"fold_{fold}_train.hdf5")
        write_hdf5(test_data, prepared_dir / f"fold_{fold}_test.hdf5")

    # Copy auxiliary
    import shutil
    src_aux = Path(output_dir) / "prepared" / "auxiliary.npz"
    shutil.copy2(src_aux, prepared_dir / "auxiliary.npz")

    # Copy metadata
    import yaml
    src_meta = Path(output_dir) / "prepared" / "metadata.yaml"
    with open(src_meta) as f:
        meta = yaml.safe_load(f)
    meta["kde_bandwidth_fraction"] = best_bw_frac
    meta["kde_bandwidth_floor"] = best_bw_floor
    with open(prepared_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    # Run with intermediates
    t0 = time.time()
    for fold in folds:
        run_single_fold_with_intermediates(
            fold, final_config, final_dir, chunk_size)
    elapsed = time.time() - t0
    print(f"\nAll folds complete in {elapsed:.1f}s")

    # Save final config
    final_config.to_yaml(final_dir / "final_config.yaml")

    # Save parameter summary
    param_summary = {
        "k_point": final_config.model.k_point,
        "k_tree": final_config.model.k_tree,
        "dim_prior": final_config.model.dim_prior,
        "free_scale": final_config.model.free_scale,
        "ignore_model_err": final_config.model.ignore_model_err,
        "snr_cap": best_snr_cap,
        "bw_frac": best_bw_frac,
        "bw_floor": best_bw_floor,
    }
    with open(final_dir / "best_params.json", "w") as f:
        json.dump(param_summary, f, indent=2)

    print(f"\nFinal results in {final_dir}/")
    print("Run the main script with --output-dir pointing to final/ "
          "for full figures and reports.")

    return final_dir


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="S23b Frankenz parameter sweep")
    parser.add_argument(
        "--config", type=str,
        default=str(script_dir / "frankenz_s23b_config.yaml"),
        help="Path to baseline frankenz config YAML")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(script_dir / "output" / "undeblended"),
        help="Base output directory (must have prepared/ data)")
    parser.add_argument(
        "--phase", type=str, required=True, choices=["A", "C", "D", "E"],
        help="Sweep phase to run")
    parser.add_argument(
        "--folds", type=str, default="2,3",
        help="Comma-separated fold numbers for Phase A screening")
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Batch chunk size")
    parser.add_argument(
        "--config-name", type=str, default="baseline",
        help="Config name for Phase D (intermediates directory)")
    # Phase E parameters
    parser.add_argument(
        "--best-k-point", type=int, default=None,
        help="Best k_point for Phase E")
    parser.add_argument(
        "--best-k-tree", type=int, default=None,
        help="Best k_tree for Phase E")
    parser.add_argument(
        "--best-snr-cap", type=str, default=None,
        help="Best S/N cap name for Phase E (from SNR_CAP_CONFIGS)")
    parser.add_argument(
        "--best-bw-frac", type=float, default=None,
        help="Best bandwidth fraction for Phase E")
    parser.add_argument(
        "--best-bw-floor", type=float, default=None,
        help="Best bandwidth floor for Phase E")
    # Phase C parameters
    parser.add_argument(
        "--top-configs", type=str, default=None,
        help="Comma-separated config names for Phase C "
             "(e.g. 'baseline,k_point_30,snrcap_s19a')")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"Frankenz S23b Parameter Sweep — Phase {args.phase}")
    print("=" * 60)

    # Load baseline config
    config = FrankenzConfig.from_yaml(args.config)

    # Patch skynoise from metadata
    import yaml
    meta_path = output_dir / "prepared" / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        config.transform.skynoise = meta["skynoise"]

    if args.phase == "A":
        folds = [int(f) for f in args.folds.split(",")]
        print(f"Screening folds: {folds}")
        run_phase_a(config, output_dir, folds, args.chunk_size)

    elif args.phase == "C":
        if args.top_configs is None:
            print("ERROR: --top-configs required for Phase C")
            print("Example: --top-configs 'baseline,k_point_30,snrcap_s19a'")
            sys.exit(1)

        # Parse top configs into (name, overrides) pairs
        top_configs = []
        for name in args.top_configs.split(","):
            name = name.strip()
            overrides = {}
            if name.startswith("k_point_"):
                overrides["k_point"] = int(name.split("_")[-1])
            elif name.startswith("k_tree_"):
                overrides["k_tree"] = int(name.split("_")[-1])
            elif name == "dim_prior_false":
                overrides["dim_prior"] = False
            elif name == "ignore_model_err":
                overrides["ignore_model_err"] = True
            elif name == "free_scale":
                overrides["free_scale"] = True
            elif name.startswith("snrcap_"):
                cap_name = name.replace("snrcap_", "")
                if cap_name in SNR_CAP_CONFIGS:
                    overrides["snr_cap"] = SNR_CAP_CONFIGS[cap_name]
            top_configs.append((name, overrides))

        run_phase_c(config, output_dir, top_configs, args.chunk_size)

    elif args.phase == "D":
        run_phase_d(config, output_dir, config_name=args.config_name)

    elif args.phase == "E":
        if args.best_bw_frac is None or args.best_bw_floor is None:
            print("ERROR: --best-bw-frac and --best-bw-floor required "
                  "for Phase E")
            sys.exit(1)

        overrides = {}
        if args.best_k_point is not None:
            overrides["k_point"] = args.best_k_point
        if args.best_k_tree is not None:
            overrides["k_tree"] = args.best_k_tree

        snr_cap = SNR_CAP
        if args.best_snr_cap and args.best_snr_cap in SNR_CAP_CONFIGS:
            snr_cap = SNR_CAP_CONFIGS[args.best_snr_cap]

        run_phase_e(
            config, output_dir, overrides, snr_cap,
            args.best_bw_frac, args.best_bw_floor, args.chunk_size)


if __name__ == "__main__":
    main()
