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
from scipy.integrate import simpson
import os

__all__ = ["integrate_flux",
           "read_fits",
           "generate_light_curve",
           "generate_lc_hlsp"]


# This function integrates the flux within a wavelength range for given arrays
# for wavelength and flux
def integrate_flux(
    wavelength_range,
    wavelength_list,
    flux_list,
    gross_list,
    net_list,
    exposure_time,
    poisson_interval="sherpagehrels",
    return_integrated_gross=False,
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

    gross_list : ``numpy.ndarray``
        Array containing the gross counts of the spectrum.

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

    return_integrated_gross : ``bool``, optional
        Sets whether the function returns the integrated gross and error (in
        counts) in addition to the flux. Default value is ``False``.

    Returns
    -------
    integrated_flux : ``float``
        Integrated flux.

    integrated_error : ``float``
        Uncertainty of the integrated flux.

    integrated_gross : ``float``
        Integrated gross counts. Returned only if ``return_integrated_gross`` is
        set to ``True``.

    average_gross_error :
        Uncertainty of the integrated gross counts. Returned only if
        ``return_integrated_gross`` is set to ``True``.
    """
    # Raise an error if the user-defined wavelength range is outside of the
    # hard boundaries of the wavelength list
    if max(wavelength_range) > max(wavelength_list) or min(wavelength_range) < min(
        wavelength_list
    ):
        raise ValueError(
            "Wavelength_range must be within the boundaries of the wavelength_list."
        )

    # Since the pixels may not range exactly in the interval above,
    # we will need to deal with fractional pixels. But first, let's
    # integrate the pixels that are fully inside the range
    full_indexes = np.where(
        (wavelength_list > wavelength_range[0])
        & (wavelength_list < wavelength_range[1])
    )[0]
    full_pixel_flux = simpson(
        y=flux_list[full_indexes], x=wavelength_list[full_indexes]
    )

    # And now we deal with the flux in the fractional pixels
    index_left = full_indexes[0]
    index_right = full_indexes[-1]
    pixel_width_left = wavelength_list[index_left] - wavelength_list[index_left - 1]
    fraction_left = (
        1 - (wavelength_range[0] - wavelength_list[index_left - 1]) / pixel_width_left
    )
    pixel_width_right = wavelength_list[index_right + 1] - wavelength_list[index_right]
    fraction_right = (
        1 - (wavelength_list[index_right + 1] - wavelength_range[1]) / pixel_width_right
    )
    fractional_flux_left = flux_list[index_left - 1] * fraction_left
    fractional_flux_right = flux_list[index_right + 1] * fraction_right

    # There is a bug in stistools that overestimates errors of time-tag split
    # subexposures. So we will need to calculate them manually here
    # based on the raw counts in the extracted spectra. Here we go.
    full_pixel_gross = np.sum(gross_list[full_indexes])
    fractional_gross_left = gross_list[index_left - 1] * fraction_left
    fractional_gross_right = gross_list[index_right + 1] * fraction_right
    integrated_gross = full_pixel_gross + fractional_gross_left + fractional_gross_right
    sensitivity = flux_list[full_indexes] / net_list[full_indexes]
    mean_sensitivity = np.nanmean(sensitivity)
    gross_error = (
        poisson_conf_interval(integrated_gross, interval=poisson_interval)
        - integrated_gross
    )
    # Take the average gross error for simplicity
    average_gross_error = (-gross_error[0] + gross_error[1]) / 2
    integrated_error = average_gross_error * mean_sensitivity / exposure_time

    integrated_flux = full_pixel_flux + fractional_flux_left + fractional_flux_right

    if return_integrated_gross is False:
        return integrated_flux, integrated_error
    else:
        return (
            integrated_flux,
            integrated_error,
            integrated_gross,
            average_gross_error,
        )


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

    if instrument == 'STIS':
        cal_pipeline = 'CALSTIS '
    else:
        cal_pipeline = 'CALCOS '
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
            gross_array[i] += data["GROSS"]
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
    dataset, prefix, wavelength_range=None, return_integrated_gross=False
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

    return_integrated_gross : ``bool``, optional
        Sets whether the function returns the integrated gross and error (in
        counts) in addition to the flux. Default value is ``False``.

    Returns
    -------
    time : ``numpy.ndarray``
        Time stamps of the light curve in Modified Julian Date (MJD).

    flux : ``numpy.ndarray``
        Flux values of the light curve in erg / s / cm ** 2.

    flux_error : ``numpy.ndarray``
        Uncertainties of the flux values of the light curve in
         erg / s / cm ** 2.
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
    gross = np.zeros([n_dataset, n_subexposures])
    gross_error = np.zeros([n_dataset, n_subexposures])

    for row in range(n_dataset):
        for col in range(n_subexposures):
            wavelength = time_series_dict[row]["wavelength"][col]
            flux_density = time_series_dict[row]["flux"][col]
            gross_counts = time_series_dict[row]["gross_counts"][col]
            net = time_series_dict[row]["net"][col]
            current_exp_time = time_series_dict[row]["exp_time"][col]
            int_flux = 0.0
            int_error_squared = 0.0
            int_gross = 0.0
            int_gross_err_squared = 0.0
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
                    if return_integrated_gross is False:
                        current_int_flux, current_int_error = integrate_flux(
                            current_wavelength_range,
                            wavelength[segment],
                            flux_density[segment],
                            gross_counts[segment],
                            net[segment],
                            current_exp_time,
                            return_integrated_gross=False,
                        )
                        current_int_gross = 0.0
                        current_gross_err = 0.0
                    else:
                        (
                            current_int_flux,
                            current_int_error,
                            current_int_gross,
                            current_gross_err,
                        ) = integrate_flux(
                            current_wavelength_range,
                            wavelength[segment],
                            flux_density[segment],
                            gross_counts[segment],
                            net[segment],
                            current_exp_time,
                            return_integrated_gross=True,
                        )
                except ValueError:
                    current_int_flux = 0.0
                    current_int_error = 0.0
                    current_int_gross = 0.0
                    current_gross_err = 0.0
                int_flux += current_int_flux
                int_gross += current_int_gross
                int_error_squared += current_int_error**2
                int_gross_err_squared += current_gross_err**2
            int_error = np.sqrt(int_error_squared)
            int_gross_error = np.sqrt(int_gross_err_squared)
            flux[row, col] = int_flux
            flux_error[row, col] = int_error
            gross[row, col] = int_gross
            gross_error[row, col] = int_gross_error

    # Flatten the arrays
    time = time.flatten()
    flux = flux.flatten()
    flux_error = flux_error.flatten()
    gross = gross.flatten()
    gross_error = gross_error.flatten()

    if return_integrated_gross is False:
        return time, flux, flux_error
    else:
        return time, flux, flux_error, gross, gross_error


# Create an HLSP file for a time series
def generate_lc_hlsp(
    dataset, prefix, wavelength_ranges, source_doi, output_dir='./',
        filename=None, feature_names=None, version="1.0"
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
        feature_names = [feature_names, ]

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
    elapsed_time = (
            (max(exp_end_list) - min(exp_start_list)) * u.d).to(u.s).value
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
    hdu_0.header["CAL_VER"] = (time_series_dict[0]["cal_version"],
                               "HST Calibration Software Version")
    hdu_0.header["PIPELINE"] = ("rocky-worlds-utils",
                                "Pipeline used to reduce the HLSP data")
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
            hdu_1.header["DESCRIP"] = (feature_names[i] + " light curve",
                                       "Description of data")
        else:
            hdu_1.header["DESCRIP"] = ("Light curve", "Description of data")

        hdu_1.header["SRC_DOI"] = (source_doi,
                                   "DOI for the source data taken from MAST")
        hdu_1.header["WAVE_START"] = ((wavelength_range[0]),
                                      "Wavelength integration start value")
        hdu_1.header["WAVE_END"] = ((wavelength_range[1]),
                                      "Wavelength integration end value")
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
        hdu_1.header["RADESYS"] = ("ICRS", "Celestial coordinate reference system")
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
