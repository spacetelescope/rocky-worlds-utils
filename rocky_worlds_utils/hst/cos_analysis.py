#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process HST/COS data.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

import numpy as np
import costools
import os
import glob
import calcos
import shutil
import multiprocessing

from calcos.x1d import concatenateSegments
from astropy.io import fits

__all__ = [
    "timetag_split",
]


__N_PROCESSES = multiprocessing.cpu_count()
print(
    "{} processing units available for parallelized COS data reduction.".format(
        __N_PROCESSES
    )
)


# Divide exposures into sub-exposures for TIME-TAG data and process them
def timetag_split(
    dataset,
    prefix,
    output_dir,
    n_subexposures=10,
    temporal_resolution=None,
    clean_intermediate_steps=True,
    overwrite=False,
    output_file_name=None,
    n_cpus=__N_PROCESSES,
):
    """
    Creates a new time-series of x1d fits files of an HST/COS dataset.

    Parameters
    ----------
    dataset : ``str``
        Dataset name (example: ``ld9m17d3q``).

    prefix : ``str``
        Fixed path to dataset directory.

    output_dir : ``str``
        Path to output directory.

    n_subexposures : ``int``
        Number of subexposures to produce in the time series. This value is
        overridden if the ``temporal_resolution`` is defined. Default is 10.

    temporal_resolution : ``float``, ``int`` or ``None``, optional
        Temporal resolution in unit of seconds for the time series. If ``None``,
        then the temporal resolution will be defined by ``n_subexposures``.
        If not ``None``, then the number of subexposures is overridden by the
        temporal resolution. Default is ``None``.

    clean_intermediate_steps : ``bool``, optional
        Sets whether intermediate steps should be cleaned up after each run.
        Default is ``True``.

    overwrite : ``bool``, optional
        Overwrite the output file if it already exists. Default is ``False``.

    output_file_name : ``str`` or ``None``, optional
        Sets the name of the output file. If set, it must contain the extension
        ``.fits``. If ``None``, then the default output file name is
        ``[dataset]_ts_x1d.fits``. Default is ``None``.

    n_cpus : ``int``, optional
        Number of CPU cores to use in data reduction. Default is the maximum
        number of available units in your system.
    """
    # Initial checks
    if output_dir == prefix:
        raise ValueError("The output directory must be different from the prefix.")

    # If output_dir doesn't exist, create it.
    os.makedirs(output_dir, exist_ok=True)

    if output_file_name is None:
        output_file_name = dataset + "_ts_x1d.fits"
        output_file = os.path.join(output_dir, output_file_name)
    elif not output_file_name.endswith(".fits"):
        raise ValueError("The extension of the output file must be .fits.")
    else:
        output_file = os.path.join(output_dir, output_file_name)

    # Test if output file exists, and if it does, delete it if overwrite is True
    if os.path.isfile(output_file):
        if overwrite is False:
            raise IOError("Time-tag split output file already exists.")
        else:
            os.remove(output_file)
    else:
        pass

    x1d_dataset = dataset + "_x1d.fits"
    x1d_filename = os.path.join(prefix, x1d_dataset)
    x1d_header_0 = fits.getheader(x1d_filename, 0)
    x1d_header_1 = fits.getheader(x1d_filename, 1)

    # Perform some other checks
    if x1d_header_0["OBSTYPE"] != "SPECTROSCOPIC":
        raise ValueError("Observation type must be SPECTROSCOPIC.")
    if x1d_header_0["OBSMODE"] != "TIME-TAG":
        raise ValueError("Observing mode must be TIME-TAG.")

    # Extracting some useful information
    exp_time = x1d_header_1["EXPTIME"]

    # Define the number of sub-exposures
    if temporal_resolution is not None:
        n_subexposures = int(round(exp_time / temporal_resolution))
    else:
        pass

    # The CalCOS pipeline requires time bins to perform the time-tag split
    time_bins = np.linspace(0, exp_time, n_subexposures + 1)

    # We will use splittag to break down the full exposure into subexposures
    tag_filename_a = os.path.join(prefix, dataset + "_corrtag_a.fits")
    tag_filename_b = os.path.join(prefix, dataset + "_corrtag_b.fits")
    time_list = ""
    for time in time_bins:
        time_list += str(time) + ", "

    costools.splittag.splittag(
        infiles=tag_filename_a,
        outroot=os.path.join(output_dir, dataset),
        time_list=time_list,
    )
    costools.splittag.splittag(
        infiles=tag_filename_b,
        outroot=os.path.join(output_dir, dataset),
        time_list=time_list,
    )

    # Run the tag split data in the CalCOS pipeline
    # Some hack necessary to avoid IO error when using x1dcorr
    split_list = glob.glob(os.path.join(output_dir, dataset + "_*_corrtag_*.fits"))
    for split_tag in split_list:
        # This loop will rename datasets in the output directory by swapping corrtag and segment
        # example ld9m17d3q_1_corrtag_b.fits ----> ld9m17d3q_1_b_corrtag.fits
        # split_tag.stem prints rootname_split-tag-number_caltype_segment
        # for example: output_dir/ld9m17d3q_1_corrtag_b.fits, the stem would be ld9m17d3q_1_corrtag_b
        basename, ext = os.path.splitext(os.path.basename(split_tag))
        print(basename.split("_"))
        _, split_tag_number, caltype, segment = basename.split("_")
        new_split_tag_name = (
            "_".join([dataset, split_tag_number, segment, caltype]) + ext
        )
        new_split_tag = os.path.join(output_dir, new_split_tag_name)
        os.rename(split_tag, new_split_tag)

    # Extract the tag-split spectra
    split_list = glob.glob(os.path.join(output_dir, dataset + "_*_*_corrtag.fits"))
    if n_cpus > 1:
        try:
            with multiprocessing.Pool(processes=n_cpus) as pool:
                _ = pool.starmap(
                    calcos.calcos,
                    [
                        (subexposure, os.path.join(output_dir, "temp/"))
                        for subexposure in split_list
                    ],
                )
        except FileExistsError:
            for subexposure in split_list:
                os.remove(subexposure)
            shutil.rmtree(os.path.join(output_dir, "temp/"))
            raise OSError(
                "Error encountered during multiprocessing. Temporary "
                "files were deleted."
            )
    else:
        for subexposure in split_list:
            _ = calcos.calcos(subexposure, os.path.join(output_dir, "temp/"))

    # Move x1ds to output folder
    split_list = glob.glob(os.path.join(output_dir, "temp/", dataset + "*_x1d.fits"))
    for subexposure in split_list:
        shutil.move(subexposure, output_dir)

    # Clean the intermediate steps files
    if clean_intermediate_steps is True:
        shutil.rmtree(os.path.join(output_dir, "temp/"))
    else:
        pass

    # Return the filenames back to normal
    split_list = glob.glob(os.path.join(output_dir, dataset) + "*_corrtag.fits")
    for corrtag in split_list:
        basename, ext = os.path.splitext(os.path.basename(corrtag))
        _, split_tag_number, caltype, segment = basename.split("_")
        new_corrtag_name = "_".join([dataset, split_tag_number, segment, caltype]) + ext
        new_corrtag = os.path.join(output_dir, new_corrtag_name)
        os.rename(corrtag, new_corrtag)

    # Concatenate segments `a` and `b` of the detector
    for i in range(n_subexposures):
        x1d_list = glob.glob(
            os.path.join(output_dir, dataset + "_%i_?_x1d.fits" % (i + 1))
        )

        concatenateSegments(
            x1d_list, os.path.join(output_dir, dataset + "_%i" % (i + 1) + "_x1d.fits")
        )

    # Remove more intermediate steps
    if clean_intermediate_steps is True:
        remove_list = glob.glob(os.path.join(output_dir, dataset + "_*_*_x1d.fits"))
        for item in remove_list:
            os.remove(item)

    # Merge the splits into a single time-series fits file, like it's done in
    # the STIS code
    with fits.open(os.path.join(output_dir, dataset + "_1_x1d.fits")) as hdu:
        primary_header = hdu[0].header
        primary_data = hdu[0].data
        bintable_header = hdu[1].header
        bintable_data = hdu[1].data
        primary_header["FILENAME"] = output_file_name

    new_primary_hdu = fits.PrimaryHDU(header=primary_header, data=primary_data)
    new_bintable_hdu = fits.BinTableHDU(header=bintable_header, data=bintable_data)
    hdu_list = [new_primary_hdu, new_bintable_hdu]

    for i in range(n_subexposures - 1):
        with fits.open(
            os.path.join(output_dir, dataset + "_{}_x1d.fits".format(str(i + 2)))
        ) as hdu:
            next_bintable_header = hdu[1].header
            next_bintable_data = hdu[1].data
        # Correct the EXPSTART value because there is a bug in costools.splittag
        next_bintable_header["EXPSTART"] += time_bins[i + 1] / 24 / 3600
        next_bintable_hdu = fits.BinTableHDU(
            header=next_bintable_header, data=next_bintable_data
        )
        hdu_list.append(next_bintable_hdu)

    # Write time series to a new fits file
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(output_file)

    # Remove last intermediate steps
    if clean_intermediate_steps is True:
        for i in range(n_subexposures):
            os.remove(os.path.join(output_dir, dataset + "_%i_x1d.fits" % (i + 1)))
        remove_list = glob.glob(os.path.join(output_dir, dataset + "_*_corrtag_*.fits"))
        for item in remove_list:
            os.remove(item)
