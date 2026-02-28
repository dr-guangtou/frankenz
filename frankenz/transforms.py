"""
Photometric feature transforms: magnitude, luptitude, and identity.

Extracted from pdf.py to provide a focused module with a config-driven
factory function.
"""

import functools
import warnings

import numpy as np

__all__ = [
    "identity", "magnitude", "inv_magnitude", "luptitude", "inv_luptitude",
    "get_transform",
]


def identity(phot, err, *args, **kwargs):
    """
    Identity transform — returns photometry unchanged.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Input photometry, unchanged.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Input errors, unchanged.

    """
    return phot, err


def magnitude(phot, err, zeropoints=1., *args, **kwargs):
    """
    Convert photometry to AB magnitudes.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes errors corresponding to input `err`.

    """

    # Warn about non-positive fluxes (produce NaN via log10).
    bad = np.asarray(phot) <= 0
    if np.any(bad):
        n_bad = int(np.sum(bad))
        warnings.warn("{} non-positive flux value(s) encountered in "
                      "magnitude(); consider using luptitude() "
                      "instead.".format(n_bad))

    # Compute magnitudes.
    mag = -2.5 * np.log10(phot / zeropoints)

    # Compute errors.
    mag_err = 2.5 / np.log(10.) * err / phot

    return mag, mag_err


def inv_magnitude(mag, err, zeropoints=1., *args, **kwargs):
    """
    Convert AB magnitudes to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitude errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """

    # Compute magnitudes.
    phot = 10**(-0.4 * mag) * zeropoints

    # Compute errors.
    phot_err = err * 0.4 * np.log(10.) * phot

    return phot, phot_err


def luptitude(phot, err, skynoise=1., zeropoints=1., *args, **kwargs):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). See Lupton et
    al. (1999) for more details.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes errors corresponding to input `err`.

    """

    # Compute asinh magnitudes.
    mag = -2.5 / np.log(10.) * (np.arcsinh(phot / (2. * skynoise)) +
                                np.log(skynoise / zeropoints))

    # Compute errors.
    mag_err = np.sqrt(np.square(2.5 * np.log10(np.e) * err) /
                      (np.square(2. * skynoise) + np.square(phot)))

    return mag, mag_err


def inv_luptitude(mag, err, skynoise=1., zeropoints=1., *args, **kwargs):
    """
    Convert asinh magnitudes ("Luptitudes") to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitude errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """

    # Compute photometry.
    phot = (2. * skynoise) * np.sinh(np.log(10.) / -2.5 * mag -
                                     np.log(skynoise / zeropoints))

    # Compute errors.
    phot_err = np.sqrt((np.square(2. * skynoise) + np.square(phot)) *
                       np.square(err)) / (2.5 * np.log10(np.e))

    return phot, phot_err


def get_transform(config):
    """
    Factory that returns a configured transform function.

    Parameters
    ----------
    config : TransformConfig or FrankenzConfig
        Configuration specifying the transform type and parameters.
        If a FrankenzConfig is passed, uses its `.transform` attribute.

    Returns
    -------
    transform : callable
        A function with signature `(phot, err, *args, **kwargs) -> (vals, errs)`
        that has zeropoints/skynoise pre-bound from the config.

    """
    # Accept either TransformConfig or FrankenzConfig
    from .config import FrankenzConfig, TransformConfig
    if isinstance(config, FrankenzConfig):
        config = config.transform

    transform_type = config.type.lower()

    if transform_type == "identity":
        return identity
    elif transform_type == "magnitude":
        return functools.partial(magnitude, zeropoints=config.zeropoints)
    elif transform_type == "luptitude":
        return functools.partial(
            luptitude,
            skynoise=np.array(config.skynoise),
            zeropoints=config.zeropoints,
        )
    else:
        raise ValueError(f"Unknown transform type: {config.type!r}. "
                         f"Valid types: 'identity', 'magnitude', 'luptitude'.")
