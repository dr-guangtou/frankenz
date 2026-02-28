#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Objects used to store data, compute fits, and generate PDFs.

"""

import sys
import os
import warnings
import math
import numpy as np

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from .bruteforce import BruteForce
from .knn import NearestNeighbors
from .networks import SelfOrganizingMap, GrowingNeuralGas

__all__ = ["BruteForce", "NearestNeighbors", "SelfOrganizingMap",
           "GrowingNeuralGas", "get_fitter"]


def get_fitter(config, training_data):
    """Create a configured fitter instance from config and training data.

    Parameters
    ----------
    config : FrankenzConfig
        Full configuration.
    training_data : PhotoData
        Training data with flux, flux_err, mask arrays.

    Returns
    -------
    fitter : BruteForce or NearestNeighbors
        Configured fitter instance, ready for predict().
    """
    from .config import FrankenzConfig
    from .io import PhotoData
    from .transforms import get_transform

    backend = config.model.backend.lower()

    # Prepare random state
    rstate = None
    if config.seed is not None:
        rstate = np.random.RandomState(config.seed)

    if backend == "bruteforce":
        fitter = BruteForce(
            models=training_data.flux,
            models_err=training_data.flux_err,
            models_mask=training_data.mask,
        )
    elif backend == "knn":
        # Build fmap_kwargs from config transform
        transform_func = get_transform(config.transform)
        fitter = NearestNeighbors(
            models=training_data.flux,
            models_err=training_data.flux_err,
            models_mask=training_data.mask,
            leafsize=config.model.kdtree.leafsize,
            K=config.model.k_tree,
            feature_map=transform_func,
            rstate=rstate,
            verbose=config.verbose,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Valid: 'knn', 'bruteforce'."
        )

    return fitter
