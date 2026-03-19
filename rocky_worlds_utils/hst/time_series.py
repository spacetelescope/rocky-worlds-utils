#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to time series from HST STIS and COS spectra.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

import astropy.units as u
from astropy.io import fits
from astropy.stats import poisson_conf_interval
from astropy.time import Time
import numpy as np
import os
from scipy.integrate import simpson

__all__ = ["integrate_flux", "read_fits", "generate_light_curve",
           "generate_lc_hlsp"]


# This function integrates the flux within a wavelength range for given arrays
# for wavelength and flux
def integrate_flux(
        wavelength_range,
        wavelength_list,
        flux_list,
        net_list,
        exposure_time,
        poisson_interval="sherpagehrels",
        mask_list=None
):
    """
    Integrate fluxes from HST STIS and COS spectra within a range of
    wavelengths. This code takes into account fractional pixels and correctly
    estimates uncertainties in the Poisson counting regime.

    Parameters
    ----------
    wavelength_range : array-like
        List, array or tuple of two floats containing the start and end of the
        wavelength range to be integrated.

    wavelength_list : ``numpy.ndarray``
        Array containing the wavelengths of the spectrum.

    flux_list : ``numpy.ndarray``
        Array containing the flux values of the spectrum.

    net_list : ``numpy.ndarray``
        Array containing the net count rates of the spectrum.

    exposure_time : ``float``
        Exposure time in seconds.

    poisson_interval : ``str``, optional
        Poisson confidence interval to use in calculation of errors. The options
        are ``‘root-n’``, ``’root-n-0’``, ``’pearson’``, ``’sherpagehrels’, and
        ``’frequentist-confidence’`` (same as those in
        ``astropy.stats.poisson_conf_interval``). Default value is
        ``'sherpagehrels'``.

    mask_list : ``numpy.ndarray``, optional
        Array containing the mask to be applied to the spectrum. It must have
        the same shape as ``flux_list`` and it must contain multiplicative
        values that represent how much weight should be applied to each pixel
        of the spectrum. Normally, if one wants to have a binary mask, one would
        assign values of zeros for fully-masked pixels and ones for non-masked
        pixels. Default value is ``None`` (no masking).

    Returns
    -------
    integrated_flux : ``float``
        Integrated flux.

    integrated_error : ``float``
        Uncertainty of the integrated flux.

    integrated_net : ``float``
        Integrated net count rate.

    integrated_net_error : ``float``
        Uncertainty of the integrated net count rate.
    """
    # Raise an error if the user-defined wavelength range is outside of the
    # hard boundaries of the wavelength list
    if max(wavelength_range) > max(wavelength_list) or min(
            wavelength_range) < min(
            wavelength_list
    ):
        raise ValueError(
            "Wavelength_range must be within the boundaries of the wavelength_list."
        )

    if mask_list is None:
        mask_list = np.ones_like(net_list)

    # Since the pixels may not range exactly in the interval above,
    # we will need to deal with fractional pixels. But first, let's
    # integrate the pixels that are fully inside the range
    net_count_list = net_list * exposure_time * mask_list

    # At first, we integrate the net counts and later convert them into fluxes
    # using the sensitivity function. We do this because, in order to calculate
    # uncertainties in the Poisson counting regime, we need to work on count
    # space first, and then convert to fluxes.
    full_indexes = np.where(
        (wavelength_list > wavelength_range[0])
        & (wavelength_list < wavelength_range[1])
    )[0]
    full_pixel_net_counts = np.sum(net_count_list[full_indexes])

    # And now we deal with the net counts in the fractional pixels
    index_left = full_indexes[0]
    index_right = full_indexes[-1]
    pixel_width_left = wavelength_list[index_left] - wavelength_list[
        index_left - 1]
    fraction_left = (
            1 - (wavelength_range[0] - wavelength_list[
        index_left - 1]) / pixel_width_left
    )
    pixel_width_right = wavelength_list[index_right + 1] - wavelength_list[
        index_right]
    fraction_right = (
            1 - (wavelength_list[index_right + 1] - wavelength_range[
        1]) / pixel_width_right
    )

    # Get fractional-pixel net counts
    fractional_net_count_left = net_count_list[index_left - 1] * fraction_left
    fractional_net_count_right = (
            net_count_list[index_right + 1] * fraction_right)

    # Calculate the total net counts
    integrated_net_count = (
            full_pixel_net_counts + fractional_net_count_left +
            fractional_net_count_right)

    # Calculate the Poisson uncertainties
    net_count_error = (
            poisson_conf_interval(np.abs(integrated_net_count),
                                  interval=poisson_interval)
            - integrated_net_count
    )
    # Take the average net count error for simplicity
    average_net_count_error = (-net_count_error[0] + net_count_error[1]) / 2

    # Convert net counts to net count rates
    integrated_net = integrated_net_count / exposure_time
    integrated_net_error = average_net_count_error / exposure_time

    sensitivity = flux_list[full_indexes] / net_list[full_indexes]
    mean_sensitivity = np.nanmean(sensitivity)

    # If mean sensitivity is NaN, it's probably because there were no net counts
    # registered, so flux should be zero anyway
    if np.isnan(mean_sensitivity):
        mean_sensitivity = 0

    integrated_error = (
            average_net_count_error / exposure_time * mean_sensitivity)
    integrated_flux = integrated_net * mean_sensitivity

    return (integrated_flux, integrated_error, integrated_net,
            integrated_net_error)


# Read the time-series fits file
def read_fits(dataset, prefix, target_name=None):
    """
    Read data from a time-series file processed with the ``cos_analysis`` and
    ``stis_analysis`` modules.

    Parameters
    ----------
    dataset : ``str``
        Dataset name (example: ``ld9m17d3q`` or ``o4z301040``).

    prefix : ``str``
        Fixed path to dataset directory.

    target_name : ``str``, optional
        Name of observed target. If ``None``, uses the default value retrieved
        from the header information. Default is ``None``.

    Returns
    -------
    time_series_dict : ``dict``
        Dictionary containing the following information about the time series:
        - `proposal_id`
        - `instrument`
        - `detector`
        - `target`
        - `ra` (right ascension)
        - `dec` (declination)
        - `grating`
        - `aperture`
        - `cenwave` (central wavelength)
        - `fppos` (FP-POS number, relevant only for COS)
        - `exp_start` (exposure start in MJD)
        - `exp_end` (exposure end in MJD)
        - `time_stamp` (time stamp in MJD)
        - `exp_time` (exposure time in s)
        - `n_detector_segments` (number of detector segments)
        - `wavelength` (wavelength in Angstrom)
        - `flux` (flux density in erg / s / cm ** 2 / A)
        - `error` (flux density error in  erg / s / cm ** 2 / A)
        - `gross_counts` (gross counts)
        - `net` (net count rate in counts / s)
    """
    x1d_filename = dataset + "_ts_x1d.fits"
    x1d_filepath = os.path.join(prefix, x1d_filename)
    x1d_header_0 = fits.getheader(x1d_filepath, 0)

    instrument = x1d_header_0["INSTRUME"]
    grating = x1d_header_0["OPT_ELEM"]
    cenwave = x1d_header_0["CENWAVE"]
    aperture = x1d_header_0["PROPAPER"]
    declination = x1d_header_0["DEC_TARG"]
    right_ascension = x1d_header_0["RA_TARG"]
    detector = x1d_header_0["DETECTOR"]
    proposal_id = x1d_header_0["PROPOSID"]

    if instrument == "STIS":
        cal_pipeline = "CALSTIS "
    else:
        cal_pipeline = "CALCOS "
    cal_version = cal_pipeline + x1d_header_0["CAL_VER"]

    try:
        fp_pos = x1d_header_0["FPPOS"]  # Present only in headers of COS data
    except KeyError:
        fp_pos = None  # Assign a None if STIS data

    if target_name is None:
        target_name = x1d_header_0["TARGNAME"]

    with fits.open(x1d_filepath) as hdu:
        n_subexposures = len(hdu) - 1

        # Instantiate some important arrays
        exposure_start = np.zeros(n_subexposures)
        exposure_end = np.zeros(n_subexposures)
        time_stamp = np.zeros(n_subexposures)
        exposure_time = np.zeros(n_subexposures)
        data_shape = np.shape(hdu[1].data["WAVELENGTH"])
        ts_data_shape = (n_subexposures,) + data_shape
        wavelength_array = np.zeros(ts_data_shape)
        flux_array = np.zeros(ts_data_shape)
        error_array = np.zeros(ts_data_shape)
        gross_array = np.zeros(ts_data_shape)
        net_array = np.zeros(ts_data_shape)

        # Populate arrays
        for i in range(n_subexposures):
            x1d_header_i = hdu[i + 1].header
            exposure_start[i] = x1d_header_i["EXPSTART"]
            exposure_end[i] = x1d_header_i["EXPEND"]
            exposure_time[i] = x1d_header_i["EXPTIME"]
            time_stamp[i] = (exposure_start[i] + exposure_end[i]) / 2
            data = hdu[i + 1].data
            wavelength_array[i] += data["WAVELENGTH"]
            flux_array[i] += data["FLUX"]
            error_array[i] += data["ERROR"]
            gross_array[i] += data["GROSS"] * x1d_header_i["EXPTIME"]
            net_array[i] += data["NET"]

    time_series_dict = {
        "proposal_id": proposal_id,
        "instrument": instrument,
        "cal_version": cal_version,
        "detector": detector,
        "target": target_name,
        "ra": right_ascension,
        "dec": declination,
        "grating": grating,
        "aperture": aperture,
        "cenwave": cenwave,
        "fppos": fp_pos,
        "exp_start": exposure_start,  # MJD
        "exp_end": exposure_end,  # MJD
        "time_stamp": time_stamp,  # MJD
        "exp_time": exposure_time,  # s
        "n_detector_segments": data_shape[0],
        "wavelength": wavelength_array,  # Angstrom
        "flux": flux_array,  # erg / s / cm ** 2 / A
        "error": error_array,  # erg / s / cm ** 2 / A
        "gross_counts": gross_array,  # counts
        "net": net_array,  # counts / s
    }

    return time_series_dict


# Calculate light curve
def generate_light_curve(
        dataset,
        prefix,
        wavelength_range=None,
        period=None,
        reference_time=None,
        baseline_flux=None,
        mask_ranges=None,
        poisson_interval="sherpagehrels",
):
    """
    Calculate a light curve for a time-series observation.

    Parameters
    ----------
    dataset : ``str`` or ``list``
        Dataset name (example: ``ld9m17d3q`` or ``o4z301040``) or list of
        dataset names.

    prefix : ``str``
        Fixed path to datasets directory.

    wavelength_range : array-like
        List, array or tuple of two floats containing the start and end of the
        wavelength range to be integrated.

    period : ``float``, optional
        Revolution period of the light curve in unit of days. If set, then this
        function returns revolution phases between -0.5 and 0.5. Default is
        ``None`` (no phase folding).

    reference_time : ``float``, optional
        Zero-phase reference time of the light curve in unit of Modified Julian
        Date (MJD). Required if ``period`` is set. Default is ``None``

    baseline_flux : ``float``, optional
        Flux baseline against which to normalize the light curve. If set, then
        the returned fluxes are in units of the baseline flux. Default is
        ``None`` (no normalization).

    poisson_interval : ``str``, optional
        Poisson confidence interval to use in calculation of errors. The options
        are ``‘root-n’``, ``’root-n-0’``, ``’pearson’``, ``’sherpagehrels’, and
        ``’frequentist-confidence’`` (same as those in
        ``astropy.stats.poisson_conf_interval``). Default value is
        ``'sherpagehrels'``.

    Returns
    -------
    time : ``numpy.ndarray``
        Time stamps of the light curve in unit of Modified Julian Date (MJD). If
        values are set to ``period`` and ``reference_time``, then these
        correspond to phases between -0.5 and 0.5.

    flux : ``numpy.ndarray``
        Flux values of the light curve in erg / s / cm ** 2. If
        ``baseline_flux`` is set, flux values are normalized to units of
        ``baseline_flux``.

    flux_error : ``numpy.ndarray``
        Uncertainties of the flux values of the light curve in
         erg / s / cm ** 2. If  `baseline_flux`` is set, flux values are
         normalized to units of ``baseline_flux``.

    net : ``numpy.ndarray``
        Net count rate of the light curve in counts / s.

    net_error : ``numpy.ndarray``
        Uncertainties of the net count rate of the light curve in counts / s.
    """
    if isinstance(dataset, str):
        n_dataset = 1
        time_series_dict = [
            read_fits(dataset, prefix),
        ]
    elif isinstance(dataset, list):
        n_dataset = len(dataset)
        time_series_dict = [read_fits(dataset, prefix) for dataset in dataset]
    else:
        raise TypeError("Dataset must be a string or a list.")

    n_segments = time_series_dict[0]["n_detector_segments"]
    n_subexposures = len(time_series_dict[0]["time_stamp"])

    # We are going to integrate fluxes within the wavelength range for each
    # segment, each subexposure, and each dataset
    time = np.zeros([n_dataset, n_subexposures])
    flux = np.zeros([n_dataset, n_subexposures])
    flux_error = np.zeros([n_dataset, n_subexposures])
    net = np.zeros([n_dataset, n_subexposures])
    net_error = np.zeros([n_dataset, n_subexposures])

    for row in range(n_dataset):
        for col in range(n_subexposures):
            wavelength = time_series_dict[row]["wavelength"][col]
            flux_density = time_series_dict[row]["flux"][col]
            net_rate = time_series_dict[row]["net"][col]
            current_exp_time = time_series_dict[row]["exp_time"][col]

            # Deal with masking wavelength ranges
            mask_array = np.ones_like(flux_density)
            if mask_ranges is None:
                pass
            else:
                # Parse the mask_ranges object
                mask_ranges = np.array(mask_ranges)
                mask_ranges_shape = mask_ranges.shape
                if len(mask_ranges_shape) < 2:
                    mask_ranges = np.array([mask_ranges,])
                else:
                    pass

                # Assign zeros to wavelength ranges that the user chose
                for mask_range in mask_ranges:
                    mask_array[(wavelength > mask_range[0]) &
                               (wavelength < mask_range[1])] = 0

            # Start integrating fluxes
            int_flux = 0.0
            int_error_squared = 0.0
            int_net = 0.0
            int_net_error_squared = 0.0
            time[row, col] = time_series_dict[row]["time_stamp"][col]
            for segment in range(n_segments):
                # Figure out the wavelength range
                if wavelength_range is None:
                    wl_0 = min(wavelength[segment])
                    wl_1 = max(wavelength[segment])
                    current_wavelength_range = np.array([wl_0, wl_1])
                else:
                    current_wavelength_range = wavelength_range
                try:
                    (current_int_flux,
                     current_int_error,
                     current_int_net,
                     current_int_net_error) = integrate_flux(
                        current_wavelength_range,
                        wavelength[segment],
                        flux_density[segment],
                        net_rate[segment],
                        current_exp_time,
                        poisson_interval=poisson_interval,
                        mask_list=mask_array[segment]
                    )
                except ValueError:
                    current_int_flux = 0.0
                    current_int_error = 0.0
                    current_int_net = 0.0
                    current_int_net_error = 0.0
                int_flux += current_int_flux
                int_error_squared += current_int_error ** 2
                int_net += current_int_net
                int_net_error_squared += current_int_net_error ** 2
            int_error = np.sqrt(int_error_squared)
            int_net_error = np.sqrt(int_net_error_squared)
            flux[row, col] = int_flux
            flux_error[row, col] = int_error
            net[row, col] = int_net
            net_error[row, col] = int_net_error

    # Flatten the arrays
    time = time.flatten()
    flux = flux.flatten()
    flux_error = flux_error.flatten()
    net = net.flatten()
    net_error = net_error.flatten()

    if period is not None and reference_time is not None:
        phase = ((np.copy(time) - reference_time) / period) % 1.0
        # Center around 0
        phase[phase > 0.5] -= 1.0
        time = phase

    if baseline_flux is not None:
        flux /= baseline_flux
        flux_error /= baseline_flux

    return time, flux, flux_error, net, net_error


# Create an HLSP file for a time series
def generate_lc_hlsp(
        dataset,
        prefix,
        wavelength_ranges,
        source_doi,
        output_dir="./",
        filename=None,
        feature_names=None,
        version="1.0",
):
    """
    Generate a high-level spectral product for a time-series observation. The
    fits file will contain one data extension per wavelength range passed to
    this function.

    Parameters
    ----------
    dataset : ``str`` or ``list``
        Dataset name (example: ``ld9m17d3q`` or ``o4z301040``) or list of
        dataset names.

    prefix : ``str``
        Fixed path to datasets directory.

    wavelength_ranges : ``numpy.ndarray``
        Array containing the start and end of the wavelength range(s) to be
        integrated. Can be passed as a two-dimensional array containing more
        than one range, in which case the shape must be (2, N).

    source_doi : ``str``
        Source DOI of the observation.

    output_dir : ``str``
        Path to output directory.

    filename : ``str``, optional
        Output filename. If ``None``, then the output filename will be
        ``hlsp_rocky-worlds_hst_[instrument]_[target]_[grating]_v[version]_lc.fits``.
        Default is ``None``.

    feature_names : ``str`` or ``list``, optional
        Name or list of names of the spectroscopic features corresponding to
        each of the wavelength ranges passed in ``wavelength_ranges``. Default
        is ``None``.

    version : ``str``, optional
        Version of this HLSP, must have a {major}.{minor} format and it must be
        a string. Default is ``'1.0'``.
    """
    if isinstance(feature_names, str):
        feature_names = [
            feature_names,
        ]

    if isinstance(dataset, str):
        time_series_dict = [
            read_fits(dataset, prefix),
        ]
    elif isinstance(dataset, list):
        time_series_dict = [read_fits(dataset, prefix) for dataset in dataset]
    else:
        raise TypeError("Dataset must be a string or a list.")

    # First, we deal with the Primary extension material

    # Compile lists of meta data
    exp_start_list = np.array([d["exp_start"] for d in time_series_dict])[0]
    exp_end_list = np.array([d["exp_end"] for d in time_series_dict])[0]
    elapsed_time = ((max(exp_end_list) - min(exp_start_list)) * u.d).to(
        u.s).value
    exposure_time = np.sum(np.array([d["exp_time"] for d in time_series_dict]))

    # Instantiate the list of HDUs that will be included in the fits file
    hdu_list = []

    hdu_0 = fits.PrimaryHDU()

    # Set the common meta data
    hdu_0.header["HLSPTYPE"] = ("Light curve", "HLSP Type")
    hdu_0.header["DATE-BEG"] = (
        Time(min(exp_start_list), format="mjd").iso,
        "ISO-8601 date-time start of the observation",
    )
    hdu_0.header["DATE-END"] = (
        Time(max(exp_end_list), format="mjd").iso,
        "ISO-8601 date-time end of the observation",
    )
    hdu_0.header["DOI"] = ("10.17909/qsyr-ny68", "Digital Object Identifier")
    hdu_0.header["HLSPID"] = ("ROCKY-WORLDS",
                              "Identifier of this HLSP collection")
    hdu_0.header["HLSP_PI"] = (
        "Hannah Diamond-Lowe",
        "Principal Investigator of this HLSP collection",
    )
    hdu_0.header["HLSPLEAD"] = ("Leonardo dos Santos",
                                "Full name of HLSP project lead")
    hdu_0.header["HLSPNAME"] = ("Rocky Worlds DDT",
                                "Title of this HLSP project")
    hdu_0.header["HLSPTARG"] = (
        time_series_dict[0]["target"],
        "Designation of the target",
    )
    hdu_0.header["HLSPVER"] = (version, "Data product version")
    hdu_0.header["INSTRUME"] = (
        time_series_dict[0]["instrument"],
        "Instrument used for this observation",
    )
    hdu_0.header["CAL_VER"] = (
        time_series_dict[0]["cal_version"],
        "HST Calibration Software Version",
    )
    hdu_0.header["PIPELINE"] = (
        "rocky-worlds-utils",
        "Pipeline used to reduce the HLSP data",
    )
    # hdu_0.header["PIPE_VER"] = (pipeline_version, "Pipeline version used to reduce the HLSP data")
    hdu_0.header["LICENSE"] = ("CC BY 4.0", "License for use of these data")
    hdu_0.header["LICENURL"] = (
        "https://creativecommons.org/licenses/by/4.0/",
        "Data license URL",
    )
    hdu_0.header["MJD-BEG"] = (min(exp_start_list),
                               "Start of the observation in MJD")
    hdu_0.header["MJD-END"] = (max(exp_end_list),
                               "End of the observation in MJD")
    hdu_0.header["MJD-MID"] = (
        (max(exp_end_list) + min(exp_start_list)) / 2,
        "Mid-time of the observation in MJD",
    )
    hdu_0.header["OBSERVAT"] = ("HST",
                                "Observatory used to obtain this observation")
    hdu_0.header["PROPOSID"] = (
        time_series_dict[0]["proposal_id"],
        "Observatory program/proposal identifier",
    )
    # hdu_0.header["REFERENC"] = ("TBD", "Bibliographic identifier")
    hdu_0.header["TELAPSE"] = (
        elapsed_time,
        "Time elapsed between start- and end-time of observation in seconds",
    )
    hdu_0.header["TELESCOP"] = ("HST", "Telescope used for this observation")
    hdu_0.header["TIMESYS"] = ("UTC", "Time scale of time-related keywords")
    hdu_0.header["XPOSURE"] = (
        exposure_time,
        "Duration of exposure in seconds, exclusive of dead time",
    )

    # Add the Primary HDU to the list
    hdu_list.append(hdu_0)

    # Now we deal with the light curves
    # Figure out whether there is only one or more wavelength ranges
    array_shape = np.shape(wavelength_ranges)
    n_ranges = len(array_shape)

    # Now calculate the light curve for each wavelength range
    for i in range(n_ranges):
        if n_ranges > 1:
            wavelength_range = wavelength_ranges[i]
        else:
            wavelength_range = wavelength_ranges

        # Calculate light curve
        time_array, flux_array, error_array, gross_array, gross_error_array = (
            generate_light_curve(
                dataset, prefix, wavelength_range, return_integrated_gross=True
            )
        )

        # Set the light curve meta data
        hdu_1 = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="TIME", format="D", array=time_array),
                fits.Column(name="FLUX", format="D", array=flux_array),
                fits.Column(name="FLUXERROR", format="D", array=error_array),
                fits.Column(name="COUNTS", format="D", array=gross_array),
                fits.Column(name="COUNTSERROR", format="D",
                            array=gross_error_array),
            ]
        )
        if feature_names is not None:
            hdu_1.header["DESCRIP"] = (
                feature_names[i] + " light curve",
                "Description of data",
            )
        else:
            hdu_1.header["DESCRIP"] = ("Light curve", "Description of data")

        hdu_1.header["SRC_DOI"] = (
            source_doi,
            "DOI for the source data taken from MAST",
        )
        hdu_1.header["WAVSTART"] = (
            (wavelength_range[0]),
            "Wavelength integration start value",
        )
        hdu_1.header["WAVE_END"] = (
            (wavelength_range[1]),
            "Wavelength integration end value",
        )
        hdu_1.header["APERTURE"] = (
            time_series_dict[0]["aperture"],
            "Aperture used for the exposure",
        )
        hdu_1.header["DEC_TARG"] = (
            time_series_dict[0]["dec"],
            "Declination coordinate of the target in deg",
        )
        hdu_1.header["DETECTOR"] = (
            time_series_dict[0]["detector"],
            "Detector used for exposure",
        )
        hdu_1.header["GRATING"] = (
            time_series_dict[0]["grating"],
            "Grating used for the exposure",
        )
        hdu_1.header["CENWAVE"] = (
            time_series_dict[0]["cenwave"],
            "Central wavelength used for the exposure",
        )
        hdu_1.header["FP-POS"] = (
            time_series_dict[0]["fppos"],
            "FP-POS used for the exposure",
        )
        hdu_1.header["RADESYS"] = ("ICRS",
                                   "Celestial coordinate reference system")
        hdu_1.header["RA_TARG"] = (
            time_series_dict[0]["ra"],
            "Right Ascension coordinate of the target in deg",
        )

        # Add the data HDU to the list
        hdu_list.append(hdu_1)

    # Finally create the corresponding FITS file
    if filename is None:
        filename = "hlsp_rocky-worlds_hst_{}_{}_{}_v{}_lc.fits".format(
            time_series_dict[0]["instrument"].lower(),
            time_series_dict[0]["target"].lower(),
            time_series_dict[0]["grating"].lower(),
            version,
        )
    else:
        pass

    hdul = fits.HDUList(hdu_list)
    hdul.writeto(output_dir + filename)
