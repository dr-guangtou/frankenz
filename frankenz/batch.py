"""
Batch pipeline runner for frankenz.

Provides `run_pipeline()` which takes config + data, splits into chunks,
runs fit_predict, and concatenates results. Controls memory via chunk_size
and optionally shows progress via tqdm.
"""

import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

from .config import FrankenzConfig
from .fitting import get_fitter
from .io import PhotoData
from .priors import get_prior

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass
class PipelineResult:
    """Container for pipeline output."""
    pdfs: np.ndarray = None
    zgrid: np.ndarray = None
    summary: dict = field(default_factory=dict)
    config: FrankenzConfig = None


def run_pipeline(config, training_data, test_data, chunk_size=1000):
    """Run the full frankenz pipeline with chunked processing.

    Parameters
    ----------
    config : FrankenzConfig
        Full pipeline configuration.
    training_data : PhotoData
        Labeled training data (must have redshifts).
    test_data : PhotoData
        Test data to compute PDFs for.
    chunk_size : int
        Number of objects per chunk. Controls memory usage.

    Returns
    -------
    PipelineResult
        Contains PDFs, redshift grid, and optional summary statistics.
    """
    # Build redshift grid
    zgrid = np.arange(
        config.zgrid.z_start,
        config.zgrid.z_end + config.zgrid.z_delta * 0.5,
        config.zgrid.z_delta,
    )

    # Create fitter
    fitter = get_fitter(config, training_data)

    # Get prior function
    lprob_func = get_prior(config)

    # Prepare random state
    rstate = None
    if config.seed is not None:
        rstate = np.random.RandomState(config.seed)

    # Prepare fit_predict kwargs — common to all backends
    fit_kwargs = {
        "model_labels": training_data.redshifts,
        "model_label_errs": (training_data.redshift_errs
                             if training_data.redshift_errs is not None
                             else np.zeros(training_data.n_objects)),
        "label_grid": zgrid,
        "lprob_func": lprob_func,
        "return_gof": False,
        "track_scale": config.model.track_scale,
        "verbose": False,
        "save_fits": False,
    }

    # Add KNN-specific params
    if config.model.backend.lower() == "knn":
        fit_kwargs.update({
            "rstate": rstate,
            "k": config.model.k_point,
            "eps": config.model.kdtree.eps,
            "lp_norm": config.model.kdtree.lp_norm,
            "distance_upper_bound": config.model.kdtree.distance_upper_bound,
        })

    # Split test data into chunks and process
    n_test = test_data.n_objects
    n_chunks = max(1, (n_test + chunk_size - 1) // chunk_size)

    # Try to import tqdm for progress
    progress = _get_progress_bar(n_chunks, config.verbose)

    all_pdfs = []
    for chunk_idx in progress:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_test)
        chunk = test_data.subset(range(start, end))

        # Run fit_predict on this chunk
        result = fitter.fit_predict(
            data=chunk.flux,
            data_err=chunk.flux_err,
            data_mask=chunk.mask,
            **fit_kwargs,
        )

        # fit_predict returns (pdfs, ...) — extract PDFs
        if isinstance(result, tuple):
            chunk_pdfs = result[0]
        else:
            chunk_pdfs = result

        all_pdfs.append(chunk_pdfs)

    pdfs = np.vstack(all_pdfs)

    # Optionally compute summary statistics
    summary = {}
    if config.verbose:
        from .pdf import pdfs_summarize
        try:
            summary = pdfs_summarize(pdfs, zgrid)
        except Exception as e:
            warnings.warn(f"Summary statistics failed: {e}")

    return PipelineResult(
        pdfs=pdfs,
        zgrid=zgrid,
        summary=summary,
        config=config,
    )


def _get_progress_bar(n_chunks, verbose):
    """Return an iterable with optional tqdm progress."""
    items = range(n_chunks)
    if not verbose:
        return items
    try:
        from tqdm import tqdm
        return tqdm(items, desc="Processing chunks", unit="chunk")
    except ImportError:
        return items
