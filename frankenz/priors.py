#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P(z, t, m) priors used when simulating observations.

"""

import sys
import os
import warnings
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator as grid_interp

__all__ = ["pmag", "_bpz_prior", "bpz_pt_m", "bpz_pz_tm", "get_prior"]

bpz_ptm = None
bpz_pztm = None


def pmag(mag, maglim, mbounds=(10., 28.), alpha=15., beta=2., gamma=1.,
         Npoints=1000, *args, **kwargs):
    """
    Function that returns P(mag) given an input magnitude limit `maglim` as::

        pm = mag**alpha * np.exp(-(mag / (maglim - gamma))**beta)

    Parameters
    ----------
    mag : float or `~numpy.ndarray`
        (Array of) magnitudes.

    maglim : float
        The 5-sigma limiting magnitude.

    mbounds : tuple of shape (2,), optional
        Lower and upper magnitude bounds, respectively. Default is `(10, 28)`.

    alpha : float, optional
        First of three parameters used to parameterize the P(mag) function.
        Default is `15.`.

    beta : float
        Second of three parameters used to parameterize the P(mag) function.
        Default is `2.`.

    gamma : float
        Third of three parameters used to parameterize the P(mag) function.
        Default is `1.`.

    Npoints : int
        The number of points used to interpolate P(mag). Default is `1000`.

    Returns
    -------
    pm : float or `~numpy.ndarray`
        Corresponding P(mag).

    """

    # Compute probabilities.
    mgrid = np.linspace(mbounds[0], mbounds[1], Npoints)  # mag grid
    pmgrid = mgrid**alpha * np.exp(-(mgrid / (maglim - gamma))**beta)  # P(mag)
    pmgrid /= np.trapz(pmgrid, mgrid)  # normalize integral
    pm = np.interp(mag, mgrid, pmgrid)  # extract P(mag)

    return pm


def _bpz_prior(m, zgrid, mbounds=(20, 32), zbounds=(0, 15), *args, **kwargs):
    """
    Internal function used to generate the BPZ prior P(t | m, z).

    Parameters
    ----------
    m : float
        Magnitude (at ~8140A).

    zgrid : `~numpy.ndarray`
        Redshift grid.

    mbounds : tuple of shape (2,), optional
        Magnitude lower/upper bounds. Default is `(20, 32)`.

    zbounds : tuple of shape (2,), optional
        Redshift lower/upper bounds. Default is `(0, 15)`.

    Returns
    -------
    p_i : `~numpy.ndarray` of shape (Nz, Nt,)
        Probability of type as a function of redshift at fixed magnitude.

    f_t : `~numpy.ndarray` of shape (Nt,)
        Type fraction at fixed magnitude.

    """

    # Formula: zm = zo + km*dm,  p(z | T, m) = z**a * exp(-(z / zm)**a)
    # coefficients from Table 1 of Benitez (2000)
    a = np.array([2.465, 1.806, 0.906])
    zo = np.array([0.431, 0.390, 0.0626])
    km = np.array([0.0913, 0.0636, 0.123])
    k_t = np.array([0.450, 0.147])

    # Fractions expected at m=20: 35% E/S0, 50% Spiral, 15% Irr
    fo_t = np.array([0.35, 0.5, 0.15])

    # Establish magnitude bounds.
    m = np.clip(m, mbounds[0], mbounds[1])
    dm = m - mbounds[0]  # dmag

    # Establish redshift bounds.
    zmt = np.clip(zo + km * dm, zbounds[0], zbounds[1])
    zmt_at_a = zmt**a
    zt_at_a = np.power.outer(zgrid, a)

    # Compute morphological fractions (0=Ell/S0, 1=Spiral, 2=Irr).
    f_t = np.zeros(3)
    f_t[:2] = fo_t[:2] * np.exp(-k_t * dm)
    f_t[2] = 1 - sum(f_t)

    # Compute probability.
    p_i = zt_at_a * np.exp(-np.clip(zt_at_a / zmt_at_a, 0., 700.))
    p_i /= p_i.sum(axis=0)
    p_i *= f_t

    return p_i, f_t


def bpz_pt_m(t, m, mbounds=(20, 32), bpz_ptm_func=None, *args, **kwargs):
    """
    BPZ conditional prior for P(t | m).

    Parameters
    ----------
    t : int
        Type.

    m : float
        Magnitude.

    mbounds : tuple of shape (2,), optional
        Magnitude lower/upper bounds. Default is `(20, 32)`.

    bpz_ptm_func : function, optional
        A function that represents an interpolation over a grid of types and
        magnitudes. If not provided, a pre-interpolated grid will be generated
        and stored.

    Returns
    -------
    prob : float
        Probability of type `t` at fixed magnitude `m`.

    """

    if t < 0 or t > 2:
        raise ValueError("t must be between 0 and 2 (inclusive).")

    if bpz_ptm_func is None:
        global bpz_ptm
        if bpz_ptm is None:

            # Compute results over an (m, z) grid.
            mgrid = np.linspace(20., 32., 1000)
            zgrid = np.linspace(0., 15., 1000)

            # Define the N-D linearly interpolated prior.
            bpz_arr = np.array([_bpz_prior(m, zgrid)[1] for m in mgrid])
            bpz_ptm = grid_interp((mgrid, [0, 1, 2]), bpz_arr)

        bpz_ptm_func = bpz_ptm

    return bpz_ptm_func((np.clip(m, mbounds[0], mbounds[1]), t))


def bpz_pz_tm(z, t, m, mbounds=(20, 32), zbounds=(0, 15), bpz_pztm_func=None,
              *args, **kwargs):
    """
    BPZ conditional prior for P(z | t, m).

    Parameters
    ----------
    z : float
        Redshift.

    t : int
        Type.

    m : float
        Magnitude.

    mbounds : tuple of shape (2,), optional
        Magnitude lower/upper bounds. Default is `(20, 32)`.

    zbounds : tuple of shape (2,), optional
        Redshift lower/upper bounds. Default is `(0, 15)`.

    bpz_pztm_func : function, optional
        A function that represents an interpolation over a grid of types and
        magnitudes. If not provided, a pre-interpolated grid will be generated
        and stored.

    Returns
    -------
    prob : float
        Probability of redshift `z` at fixed type `t` and magnitude `m`.

    """

    if t < 0 or t > 2:
        raise ValueError("t must be between 0 and 2 (inclusive).")

    if bpz_pztm_func is None:
        global bpz_pztm
        if bpz_pztm is None:

            # Compute results over an (m, z) grid.
            mgrid = np.linspace(20., 32., 1000)
            zgrid = np.linspace(0., 15., 1000)

            # Define the N-D linearly interpolated prior.
            bpz_arr = np.array([_bpz_prior(m, zgrid)[0] for m in mgrid])
            bpz_pztm = grid_interp((mgrid, zgrid, [0, 1, 2]), bpz_arr)

        bpz_pztm_func = bpz_pztm

    return bpz_pztm_func((np.clip(m, mbounds[0], mbounds[1]),
                          np.clip(z, zbounds[0], zbounds[1]), t))


def get_prior(config):
    """Factory that returns a prior callable from config.

    Parameters
    ----------
    config : PriorConfig or FrankenzConfig
        Prior configuration. If FrankenzConfig, uses its `.prior` attribute.

    Returns
    -------
    lprob_func : callable or None
        A function compatible with the `lprob_func` parameter of fitters.
        Returns None for uniform prior (fitters use flat prior by default).
    """
    from .config import FrankenzConfig, PriorConfig
    if isinstance(config, FrankenzConfig):
        config = config.prior

    prior_type = config.type.lower()

    if prior_type == "uniform":
        return None
    elif prior_type == "bpz":
        return _bpz_lprob_func
    else:
        raise ValueError(
            f"Unknown prior type: {config.type!r}. "
            f"Valid types: 'uniform', 'bpz'."
        )


def _bpz_lprob_func(models_z, models_zerr, label_grid, *args, **kwargs):
    """BPZ prior wrapped as lprob_func for use with fitters.

    Returns log-prior array of shape (Nmodel, Ngrid) by combining
    P(z|t,m) over types with uniform type mixing.
    """
    Nmodel = len(models_z)
    Ngrid = len(label_grid)
    lnprior = np.zeros((Nmodel, Ngrid))

    for i in range(Nmodel):
        # Sum P(z|t,m) over 3 BPZ types with equal weight,
        # using magnitude 25 as a default reference magnitude.
        pz = np.zeros(Ngrid)
        for t in range(3):
            pz += np.array([bpz_pz_tm(z, t, 25.0) for z in label_grid])
        pz = np.clip(pz, 1e-300, None)
        lnprior[i] = np.log(pz)

    return lnprior
