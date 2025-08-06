#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to facilitate numerical calculations.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

from astroquery.mast import Observations
from crds import assign_bestrefs
import glob
import numpy as np


__all__ = [
    "nearest_index", "get_observations"
]


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
    Observations.download_products(mast_products, download_dir=str(save_path),
                                   project=["CALSTIS", "CALCOS"], flat=True)

    if download_references:
        files = glob.glob(dataset + "*")
        assign_bestrefs(files, sync_references=True, verbosity=0)
