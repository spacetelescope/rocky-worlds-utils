#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

from astropy.io import fits
from astroquery.mast import Observations
from crds import assign_bestrefs
import glob
import os
import numpy as np


__all__ = [
    "obs_mode_check", "nearest_index", "get_observations", "get_stis_lsf"
]


# Observation mode check
def obs_mode_check(datasets, prefix):
    """
    Checks wheter a list of datasets were observed in the same mode.

    Parameters
    ----------
    datasets : ``list``
        List of dataset names containing the spectra to be co-added. They must
        all have been observed in the same mode.

    prefix : ``str``
        Location of x1d files to be co-added.
    """
    # Check if datasets correspond to the same mode
    instrument = []
    optical_element = []
    central_wavelength = []
    for dataset in datasets:
        filename = dataset + "_x1d.fits"
        full_file_path = os.path.join(prefix, filename)
        header = fits.getheader(full_file_path)
        instrument.append(header['INSTRUME'])
        optical_element.append(header['OPT_ELEM'])
        central_wavelength.append(header['CENWAVE'])

    def _all_elements_equal(lst):
        return all(x == lst[0] for x in lst)

    if (not _all_elements_equal(instrument) or
            not _all_elements_equal(optical_element) or
            not _all_elements_equal(central_wavelength)):
        return False
    else:
        return True  # All lists are uniform and equal


def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Parameters
    ----------
    array : ``numpy.array``
        Target array.
    target_value : ``float``
        Target value.

    Returns
    -------
    index : ``int``
        Index of the value in ``array`` that is closest to ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index


# Module to download HST data
def get_observations(dataset, save_path, download_references=False):
    """
    Download HST observations of a specific dataset or a list of datasets from
    the MAST archive.

    Parameters
    ----------
    dataset : ``str`` or ``list``
        Observing ID or list of observing IDs to be downloaded from MAST.

    save_path : ``str``
        Location to save downloaded observations.

    download_references : ``bool``, optional
        Sets whether to download the reference files corresponding to the
        requested observing IDs. Default is ``False``.
    """
    if isinstance(dataset, str):
        obs_id = [dataset, ]
    elif isinstance(dataset, list):
        obs_id = dataset
    else:
        raise TypeError("dataset must be either str or list")

    # Querying MAST for the requested dataset
    data_query = Observations.query_criteria(obs_id=obs_id)
    mast_products = Observations.get_product_list(data_query)

    # Download the data
    Observations.download_products(mast_products, download_dir=save_path,
                                   project=["CALSTIS", "CALCOS"], flat=True)

    if download_references:
        files = glob.glob(dataset + "*")
        assign_bestrefs(files, sync_references=True, verbosity=0)


# Retrieve the STIS LSF from stsci.edu
def get_stis_lsf(grating, aperture, wavelength_region=1200):
    """
    Retrieves the STIS line spread function from the STScI website for a given
    combination of grating, slit aperture and wavelength region.

    Parameters
    ----------
    grating : ``str``
        String that describes the STIS grating. The options are ``'G140L'``,
        ``'G140M'``, ``'E140M'``, ``'E140H'``, `'G230L'``,  ``'G230M'``,
        ``'E230M'``, ``'E230H'`, ``'G430L'``, ``'G430M'``, ``'G750L'`` and
        ``'G750M'``.

    aperture : ``str``
        String that describes the STIS aperture. The options are ``'52x0.1'``,
        ``'52x0.2'``, ``'52x0.5'``, and ``'52x2.0'``.

    wavelength_region : ``int``, optional
        Wavelength region on which the LSF is measured. The options are 1200,
        1500, 1700, 2400, 3200, 5500 and 7000. Default is 1200.

    Returns
    -------
    lsf_wavelength : ``numpy.ndarray``
        Wavelength array, centered on zero, corresponding to the line spread
        function of the requested mode in unit of Angstrom.

    lsf_profile : ``numpy.ndarray``
        LSF profile array, normalized.
    """
    url_str_head = "https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/stis/performance/spectral-resolution/_documents/"

    try:
        filename = 'LSF_{}_{}.txt'.format(grating, str(wavelength_region))
        url_str = url_str_head + filename
        lsf_data = np.loadtxt(url_str, skiprows=2)
    except FileNotFoundError:
        try:
            # Sometimes the URL needs to have an extra "LSF/" in it
            filename = 'LSF/LSF_{}_{}.txt'.format(grating, str(wavelength_region))
            url_str = url_str_head + filename
            lsf_data = np.loadtxt(url_str, skiprows=2)
        except FileNotFoundError:
            raise ValueError('Grating and/or wavelength region not found.')

    if aperture == '52x0.1':
        col = 1
    elif aperture == '52x0.2':
        col = 2
    elif aperture == '52x0.5':
        col = 3
    elif aperture == '52x2.0':
        col = 4
    else:
        raise ValueError('Selected aperture not available.')

    # Figure out the pixel to wavelength conversion
    # These values are very approximate, but should be good enough for LSF
    # convolution given uncertainties of cool star observations
    pixel_to_wavelength = {
        'G140L': 0.6,  # Angstrom / px
        'G140M': 0.05,
        'E140M': 0.0155,
        'E140H': 0.0061,
        'G230L': 1.58,
        'G230M': 0.09,
        'E230M': 0.039,
        'E230H': 0.0105,
        'G430L': 2.73,
        'G430M': 0.28,
        'G750L': 4.92,
        'G750M': 0.56,
    }

    lsf_wavelength = lsf_data[:, 0] * pixel_to_wavelength[grating]
    lsf_profile = lsf_data[:, col]

    return lsf_wavelength, lsf_profile
