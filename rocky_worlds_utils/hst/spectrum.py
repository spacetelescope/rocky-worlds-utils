#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process data from HST spectra.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import astropy.constants as c
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import simpson


# Local scripts
from rocky_worlds_utils.hst.tools import nearest_index

__all__ = [
    "read_hsla_product",
    "calculate_snr_hsla",
    "plot_lines_hsla",
]

_KEY_LINE_IDS = [
    ["Si II", "Si II", "C II", "C II"],
    ["Si III", "Si III", "Si IV", "Si IV"],
    ["C III", "C IV", "C IV", "O IV"],
    ["N V", "N V", "O V", "Ne V"],
]
_KEY_LINE_CENTERS = [
    [1260.4, 1264.7, 1334.5, 1335.708],
    [1294.5, 1206.5, 1393.8, 1402.7],
    [1175, 1548.1, 1550.8, 1401.1],
    [1238.8, 1242.8, 1371.3, 1145.6],
]
_C_SPEED = c.c.to(u.km / u.s).value


# Read data from an HSLA spectrum
def read_hsla_product(filename, prefix=None):
    """
    Read data from an HSLA spectrum data product.

    Parameters
    ----------
    filename : ``str``
        Name of the HSLA product file.

    prefix : ``str`` or ``None``, optional
        Prefix to prepend to filename. If ``None``, the assumed prefix is
        ``./``. Default is ``None``.

    Returns
    -------
    wavelength : ``numpy.ndarray``
        Wavelength array.

    flux : ``numpy.ndarray``
        Flux array.

    flux_err : ``numpy.ndarray``
        Flux uncertainty array.
    """
    if prefix is None:
        prefix = ""
    else:
        pass

    full_file_path = os.path.join(prefix, filename)
    data = fits.getdata(full_file_path, ext=1)
    wavelength = data["wavelength"].ravel()
    flux = data["flux"].ravel()
    error = data["error"].ravel()
    return wavelength, flux, error


# Calculates SNR of an HSLA data product
def calculate_snr_hsla(wavelength_array, flux_array, error_array):
    """
    Calculates the signal-to-noise ratio (SNR) of an HSLA spectrum based on the
    observed fluxes in function of wavelength.

    Parameters
    ----------
    wavelength_array : ``numpy.ndarray``
        Wavelength array.

    flux_array : ``numpy.ndarray``
        Flux array.

    error_array : ``numpy.ndarray``
        Flux uncertainty array.

    Returns
    -------
    snr : ``float``
        Signal-to-noise ratio.
    """
    # Integrate flux
    int_flux = simpson(flux_array, x=wavelength_array)
    n_samples = 10000
    # Draw a sample of spectra and compute the fluxes for each
    samples = np.random.normal(
        loc=flux_array, scale=error_array, size=[n_samples, len(flux_array)]
    )
    fluxes = []
    for i in range(n_samples):
        fluxes.append(simpson(samples[i], x=wavelength_array))
    fluxes = np.array(fluxes)
    uncertainty = np.std(fluxes)
    snr = int_flux / uncertainty
    return snr


# Plot the flux and print SNR of key emission lines in a HSLA spectrum
def plot_lines_hsla(
    wavelength, flux, error, scale=1e-14, velocity_lower=-100.0, velocity_upper=100.0
):
    """
    Plot the HSLA spectrum in key emission lines.

    Parameters
    ----------
    wavelength : ``numpy.ndarray``
        Wavelength array.

    flux : ``numpy.ndarray``
        Flux array.

    error : ``numpy.ndarray``
        Flux uncertainty array.

    scale : ``float``, optional
        Scaling division factor to apply in the plots (this is used to avoid the
        scientific notation in the axes of the plot). Default is ``1E-14``.

    velocity_lower : ``float``, optional
        Lower limit of the Doppler velocity in km/s in the x-axis of the plot.
        Default is -100.

    velocity_upper : ``float``, optional
        Upper limit of the Doppler velocity in km/s in the x-axis of the plot.
        Default is +100.

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        Figure object.

    ax : ``matplotlib.axes.Axes``
        Axes object.
    """
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=False)

    velocity_range = np.array([velocity_lower, velocity_upper])

    for row in range(nrows):
        for col in range(ncols):
            central_wl = _KEY_LINE_CENTERS[row][col]
            velocity = (wavelength - central_wl) / central_wl * _C_SPEED
            i0 = nearest_index(velocity, velocity_range[0])
            i1 = nearest_index(velocity, velocity_range[1])
            v_plot = velocity[i0: i1 + 1]
            wl_plot = wavelength[i0: i1 + 1]
            f_plot = flux[i0: i1 + 1] / scale
            u_plot = error[i0: i1 + 1] / scale
            snr = calculate_snr_hsla(wl_plot, f_plot, u_plot)
            ax[row, col].plot(v_plot, f_plot, label=str(snr))
            ax[row, col].set_title(
                _KEY_LINE_IDS[row][col] + r"@{}$\AA$".format(str(central_wl))
            )
            ax[row, col].annotate(
                "SNR = %.1f" % snr, xy=(velocity_range[0], max(f_plot) * 0.9)
            )
            if row == nrows - 1:
                ax[row, col].set_xlabel(r"Velocity [km s$^{-1}$]")

    return fig, ax
