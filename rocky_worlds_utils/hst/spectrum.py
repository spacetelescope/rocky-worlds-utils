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
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import os
from rocky_worlds_utils.hst import tools
from scipy.integrate import simpson
from scipy.stats import binned_statistic


# Local scripts
from rocky_worlds_utils.hst.tools import nearest_index

__all__ = [
    "read_hsla_product",
    "calculate_snr_hsla",
    "plot_lines_hsla",
    "coadd_first_order"
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


# Co-adds spectra
def coadd_first_order(datasets, prefix='./',
                      acceptable_dq_flags=(0, 64, 128, 1024, 2048),
                      weight='sensitivity'):
    """
    Co-adds HST first-order spectra of the same mode. The basic workflow of this
    function is to interpolate the spectra into a common wavelength table and
    calculate the weighted mean flux for each wavelength bin. If the fluxes are
    inconsistent between each other, the code can use the flux with higher SNR
    instead of the mean. If there are still outlier fluxes (compared to
    neighboring pixels), the code uses the flux from the lower SNR spectrum
    instead. Co-added pixels will have their DQ flag set to `32768` if
    they are the result of combining good pixels (according to the list of
    acceptable flags). Their DQ flag will be set to `65536` if the combined
    pixels do not have an acceptable DQ flag.

    Parameters
    ----------
    datasets : ``list``
        List of dataset names containing the spectra to be co-added. They must
        all have been observed in the same mode.

    prefix : ``str``
        Location of x1d files to be co-added.

    acceptable_dq_flags : array-like, optional
        Data-quality flags that are acceptable when co-adding overlapping
        spectra. The default values are (0, 64, 128, 1024, 2048), which
        correspond to: 0 = regular pixel, 64 = vignetted pixel, 128 = pixel in
        overscan region, 1024 = small blemish, 2048 = more than 30% of
        background pixels rejected by sigma-clipping in the data reduction.

    weight : ``str``, optional
        Defines how to co-add the spectra. The options currently implemented are
        ``'sensitivity'`` and ``'snr'`` (inverse square of the uncertainties).
        Default is ``'sensitivity'``.

    Returns
    -------
    coadded_spectrum : ``dict``:
        Dictionary containing the co-added spectrum.
    """
    n_datasets = len(datasets)
    bitwise_or_acceptable_dq_flags = reduce(np.bitwise_or, acceptable_dq_flags)

    if tools.obs_mode_check(datasets, prefix):
        pass
    else:
        raise ValueError('Observations do not have the same mode.')

    # Check if datasets correspond to the same mode
    instrument = []
    optical_element = []
    central_wavelength = []
    for dataset in datasets:
        header = fits.getheader(prefix + dataset + '_x1d.fits')
        instrument.append(header['INSTRUME'])
        optical_element.append(header['OPT_ELEM'])
        central_wavelength.append(header['CENWAVE'])

    spectra = []
    for dataset in datasets:
        data = fits.getdata(prefix + dataset + '_x1d.fits')
        spectrum = {
            'wavelength' : data['WAVELENGTH'][0],
            'flux' : data['FLUX'][0],
            'uncertainty' : data['ERROR'][0],
            'data_quality' : data['DQ'][0],
            'net' : data['NET'][0],
            'gross' : data['GROSS'][0]
        }
        spectra.append(spectrum)

    # First we need to determine which spectrum has a higher sensitivity
    avg_sensitivity = np.array([np.nanmean(sk['net'] / sk['flux'])
                                for sk in spectra])

    # We interpolate the lower-SNR spectra to the wavelength bins of the higher
    # SNR spectrum.
    max_sens_idx = np.where(avg_sensitivity == np.nanmax(avg_sensitivity))[0][0]
    reference = spectra.pop(max_sens_idx)

    f_interp = []
    err_interp = []
    net_interp = []
    gross_interp = []
    dq_interp = []
    for i in range(n_datasets - 1):
        # Perform the interpolation
        f_interp.append(np.interp(reference['wavelength'],
                                  spectra[i]['wavelength'],
                                  spectra[i]['flux']))
        err_interp.append(np.interp(reference['wavelength'],
                                    spectra[i]['wavelength'],
                                    spectra[i]['uncertainty']))
        net_interp.append(np.interp(reference['wavelength'],
                                    spectra[i]['wavelength'],
                                    spectra[i]['net']))
        gross_interp.append(np.interp(reference['wavelength'],
                                      spectra[i]['wavelength'],
                                      spectra[i]['gross']))

        # Instead of interpolating DQ flags, we take the bitwise or of DQ flag
        # corresponding to the pixels that make up the interpolated information.
        # This is done using scipy.stats.binned_statistic(). This is very hacky,
        # so I apologize for the mess
        wavelength_bin_widths = np.diff(reference['wavelength'])
        wavelength_bin_edges_trim = \
            reference['wavelength'][:-1] - wavelength_bin_widths / 2
        wavelength_bin_edges = \
            np.concatenate((
                wavelength_bin_edges_trim,
                np.array(
                    [wavelength_bin_edges_trim[-1] + wavelength_bin_widths[-1],
                     reference['wavelength'][-1] + wavelength_bin_widths[-1]
                     / 2])
            ))

        def _combine_dq(dq_flags_array):
            return reduce(np.bitwise_or, dq_flags_array).astype(int)

        dq_interp_i, _, _ = binned_statistic(
            spectra[i]['wavelength'],
            spectra[i]['data_quality'],
            statistic=_combine_dq, bins=wavelength_bin_edges
        )
        dq_interp.append(dq_interp_i)

    f_interp = np.array(f_interp)
    err_interp = np.array(err_interp)
    net_interp = np.array(net_interp)
    gross_interp = np.array(gross_interp)
    dq_interp = np.array(dq_interp).astype(int)
    sens_interp = net_interp / f_interp  # This is a good estimate of the
    # sensitivity. If there were NaNs, however, we set the sensitivity to an
    # arbitraty value; these interpolated pixels will be ignored anyway during
    # merging
    sens_interp[np.where(np.isnan(sens_interp))] = 0.0
    sens_ref = reference['net'] / reference['flux']

    # Co-add the spectra. We will take the weighted averaged, with weights equal
    # to the inverse of the uncertainties squared multiplied by a scale factor
    # to avoid numerical overflows.
    if weight == 'sensitivity':
        scale = 1E-10
        weights_interp = sens_interp * scale
        weights_ref = sens_ref * scale
    elif weight == 'snr':
        scale = 1E-20
        weights_interp = (1 / err_interp) ** 2 * scale
        weights_ref = (1 / reference['uncertainty']) ** 2 * scale
    else:
        raise ValueError(
            'The weighting option "{}" is not implemented.'.format(weight))

    # Here we deal with the data-quality flags. We only accept flags that are
    # listed in `acceptable_dq_flags`. Let's initialize the dq flag arrays
    dq_ref = reference['data_quality']
    # We start assuming that all the dq weights are zero
    dq_weights_ref = np.zeros_like(dq_ref)
    dq_weights_interp = np.zeros_like(dq_interp)
    # And then for each acceptable dq, if the element of the dq array is one
    # of the acceptable flags, we set its dq weight to one
    dq_weights_ref[np.where(dq_ref & bitwise_or_acceptable_dq_flags)] = 1
    dq_weights_interp[np.where(dq_interp & bitwise_or_acceptable_dq_flags)] = 1
    # The lines above do not catch a DQ flag of zero, so we have to manually
    # add them in case they are an acceptable DQ flag
    dq_weights_ref_0 = np.zeros_like(dq_weights_ref)
    dq_weights_interp_0 = np.zeros_like(dq_weights_interp)
    if any(0 in acceptable_dq_flags for it in range(len(acceptable_dq_flags))) \
            is True:
        dq_weights_ref_0[np.where(dq_ref == 0)] = 1
        dq_weights_interp_0[np.where(dq_interp == 0)] = 1
    dq_weights_ref += dq_weights_ref_0
    dq_weights_interp += dq_weights_interp_0

    # Calculate the data quality array of the coadded spectrum. This is done by
    # doing a bitwise-or between all components that go into the co-added
    # spectrum
    dq_interp_collapsed = np.bitwise_or.reduce(dq_interp, axis=0)
    dq_coadd = np.bitwise_or.reduce(np.array([dq_interp_collapsed, dq_ref]),
                                    axis=0)

    # Now we need to verify if we are setting the dq weighting to zero in both
    # the reference and the interpolated dqs. If this is the case, we will
    # set their weights to one and then flag these pixels
    sum_dq_weights = np.copy(dq_weights_ref + np.sum(dq_weights_interp, axis=0))
    dq_weights_ref[sum_dq_weights < 1] = 1
    for i in range(n_datasets - 1):
        dq_weights_interp[i][sum_dq_weights < 1] = 1
    dq_coadd[sum_dq_weights < 1] = 65536

    # And then we multiply the original weights by the dq weights
    weights_interp *= dq_weights_interp
    weights_ref *= dq_weights_ref

    # This following array will be important later
    sum_weights = np.sum(weights_interp, axis=0) + weights_ref

    # Finally co-add the spectra
    wl_coadd = np.copy(reference['wavelength'])

    f_coadd = np.zeros_like(reference['flux'])
    err_coadd = np.zeros_like(reference['uncertainty'])
    net_coadd = np.zeros_like(reference['net'])
    gross_coadd = np.zeros_like(reference['gross'])
    for i in range(n_datasets - 1):
        f_coadd += f_interp[i] * weights_interp[i]
        err_coadd += err_interp[i] ** 2 * weights_interp[i] ** 2
        net_coadd += net_interp[i] * weights_interp[i]
        gross_coadd += gross_interp[i] * weights_interp[i]
    f_coadd += reference['flux'] * weights_ref
    err_coadd += reference['uncertainty'] ** 2 * weights_ref ** 2
    net_coadd += reference['net'] * weights_ref
    f_coadd = f_coadd / sum_weights
    err_coadd = err_coadd ** 0.5 / sum_weights
    net_coadd = net_coadd / sum_weights
    gross_coadd = gross_coadd / sum_weights
    coadded_spectrum = {'wavelength': wl_coadd, 'flux': f_coadd,
                      'uncertainty': err_coadd, 'data_quality': dq_coadd,
                      'net': net_coadd, 'gross': gross_coadd}

    return coadded_spectrum


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
