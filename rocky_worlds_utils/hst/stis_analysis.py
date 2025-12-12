#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to process HST/STIS data.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

import stistools
import os

from astropy.io import fits
from stistools.poisson_err import poisson_err

__all__ = ["timetag_split", "extract",]


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
):
    """
    Creates a new time-series of x1d fits files of an HST/STIS dataset.

    Parameters
    ----------
    dataset : ``str``
        Dataset name (example: ``o4z301040``).

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
    """
    # Initial checks
    if output_dir == prefix:
        raise ValueError("The output directory must be different from the prefix.")

    # If output_dir doesn't exist, create it.
    os.makedirs(output_dir, exist_ok=True)

    if not prefix.endswith(os.path.sep):
        prefix += os.path.sep

    if not output_dir.endswith(os.path.sep):
        output_dir += os.path.sep

    if output_file_name is None:
        output_file = os.path.join(output_dir, dataset + "_ts_x1d.fits")
    elif not output_file_name.endswith(".fits"):
        raise ValueError("The extension of the output file must be .fits.")
    else:
        output_file = os.path.join(output_dir, output_file_name)

    print(output_file)

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
    x1d_data = fits.getdata(x1d_filename)

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

    # We will use inttag to break down the full exposure into subexposures
    # XXX WARNING: Dependending on the number of subexposures, the file can be
    # very large! XXX
    tag_filename = dataset + "_tag.fits"

    stistools.inttag.inttag(
        tagfile=os.path.join(prefix, tag_filename),
        output=os.path.join(output_dir, dataset + "_ts_raw.fits"),
        rcount=n_subexposures,
    )

    # And now we run the raw file through calstis, more specifically basic2d,
    # which does not extract the spectrum yet
    stistools.basic2d.basic2d(
        os.path.join(output_dir, dataset + "_ts_raw.fits"),
        os.path.join(output_dir, dataset + "_ts_flt.fits"),
    )

    stistools.wavecal.wavecal(
        os.path.join(output_dir, dataset + "_ts_flt.fits"),
        wavecal=os.path.join(prefix, dataset + "_wav.fits"),
    )

    # Now we extract the spectrum in the same location as the original full
    # exposure and turn off the automatic trace finding (set by the maxsrch
    # parameter)
    extract_yloc = x1d_data["A2CENTER"][0]
    extract_size = x1d_data["EXTRSIZE"][0]
    extract_bk1_size = x1d_data["BK1SIZE"][0]
    extract_bk2_size = x1d_data["BK2SIZE"][0]
    extract_bk1_offset = x1d_data["BK1OFFST"][0]
    extract_bk2_offset = x1d_data["BK2OFFST"][0]

    # Process the time series
    extract(
        dataset + "_ts",
        output_dir,
        output_dir,
        extract_yloc,
        extract_size,
        extract_bk1_size,
        extract_bk2_size,
        extract_bk1_offset,
        extract_bk2_offset,
        overwrite,
    )

    # Clean intermediate steps if requested
    if clean_intermediate_steps:
        os.remove(os.path.join(output_dir, dataset + "_ts_flt.fits"))
        os.remove(os.path.join(output_dir, dataset + "_ts_raw.fits"))
    else:
        pass


# Extract STIS first-order spectra with user-defined trace positions
def extract(
    dataset,
    prefix,
    output_dir,
    a2center,
    extraction_size=11,
    background1_size=5,
    background2_size=5,
    background1_offset=-300,
    background2_offset=300,
    overwrite=False,
    output_file_name=None,
):
    """
    Sometimes CALSTIS cannot properly find the trace location of faint FUV
    targets in first-order spectroscopy. This function allows the user to
    extract STIS first-order spectra in user-defined positions.

    Parameters
    ----------
    dataset : ``str``
        Dataset name (example: ``o4z301040``).

    prefix : ``str``
        Fixed path to dataset directory.

    output_dir : ``str``
        Path to output directory.

    a2center : ``float``
        Location of the trace where it crosses the center of the detector in the
        dispersion direction.

    extraction_size : ``float``, optional
        Height of the spectral extraction in units of pixels. Default value is
        11.

    background1_size : ``float``, optional
        Height of the lower background extraction in units of pixels. Default
        value is 5.

    background2_size : ``float``, optional
        Height of the upper background extraction in units of pixels. Default
        value is 5.

    background1_offset : ``float``, optional
        Offset of the lower background extraction in units of pixels. Default
        value is -300.

    background2_offset : ``float``, optional
        Offset of the upper background extraction in units of pixels. Default
        value is 300.

    overwrite : ``bool``, optional
        Overwrite the output file if it already exists. Default is ``False``.

    output_file_name : ``str`` or ``None``, optional
        Sets the name of the output file. If set, it must contain the extension
        ``.fits``. If ``None``, then the default output file name is
        ``[dataset]_ts_x1d.fits``. Default is ``None``.
    """
    # Initial checks
    if not prefix.endswith(os.path.sep):
        prefix += os.path.sep

    if not output_dir.endswith(os.path.sep):
        output_dir += os.path.sep

    # I/O
    input_file = os.path.join(prefix, dataset + "_flt.fits")
    if output_file_name is not None:
        output_file = os.path.join(output_dir, output_file_name)
    else:
        output_file = os.path.join(output_dir, dataset + "_x1d.fits")

    # Test if output file exists, and if it does, delete it if overwrite is True
    if os.path.isfile(output_file):
        if overwrite is False:
            raise IOError("Extracted output spectrum already exists.")
        else:
            os.remove(output_file)
    else:
        pass

    # Process the time series
    stistools.x1d.x1d(
        input_file,
        output=output_file,
        maxsrch=0,
        # Setting this to zero ensures that code will not look for
        # trace location
        a2center=a2center,
        extrsize=extraction_size,
        backcorr="perform",
        bk1size=background1_size,
        bk2size=background2_size,
        bk1offst=background1_offset,
        bk2offst=background2_offset,
    )
