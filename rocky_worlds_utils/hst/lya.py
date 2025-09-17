#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to analyze Lyman-alpha profiles with HST data.
This code take a few inspirations from the lyapy code authored by Allison
Youngblood: https://github.com/allisony/lyapy.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""


import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.special import wofz, voigt_profile
from rocky_worlds_utils.hst.tools import get_stis_lsf
from astropy.convolution import convolve


__all__ = ["intrinsic_stellar_profile", "ism_profile", "observed_lya_profile"]


# Physical constants
LIGHT_SPEED = const.c.to(u.cm / u.s).value         # speed of light [cm/s]
BOLTZMANN_CONSTANT = const.k_B.to(u.erg / u.K).value   # Boltzmann constant [erg/K]
HYDROGEN_MASS = const.m_p.to(u.g).value       # hydrogen mass [g]
DEUTERIUM_MASS = 2 * HYDROGEN_MASS  # deuterium mass [approximate]
ELECTRON_CHARGE = 4.80320425e-10    # electron charge [esu]
ELECTRON_MASS = const.m_e.to(u.g).value      # electron mass [g]
LYA_OSCILLATOR_STRENGTH = 0.4164    # oscillator strength for Lyman-alpha
LYA_BROADENING = 6.265e8            # natural broadening [s^-1]

# Line center wavelengths [cm]
LAMBDA_HI_REST = 1215.67e-8
LAMBDA_DI_REST = 1215.3394e-8
LAMBDA_OV_REST = 1218.344e-8


# Lyman-alpha intrinsic profile
def intrinsic_stellar_profile(wavelength, star_velocity,
                              log_lorentzian_amplitude, lorentzian_width,
                              gaussian_width,
                              reference_wavelength=LAMBDA_HI_REST,
                              self_absorption_parameter=0):
    """
    Calculates the Lyman-alpha intrinsic profile of a star for a given
    wavelength array.

    Parameters
    ----------
    wavelength : ``numpy.ndarray``
        Wavelength array in unit Angstrom.

    star_velocity : ``float``
        Radial velocity of the star in km/s (reasonable range: -300 to +300 for
        Lya).

    log_lorentzian_amplitude : ``float``
        Logarithm of the amplitude of the Lorentzian (emission line) in
        erg/cm2/s/A (reasonable range: -13 to -9 for Lya).

    lorentzian_width : ``float``
        FWHM of the Lorentzian in km/s (reasonable range: ~5-100 for Lya).

    gaussian_width : ``float``
        FWHM of the Gaussian in km/s (reasonable range: ~10-200 for Lya).

    reference_wavelength : ``float``, optional
        Reference wavelength in cm. Default value is the Lyman-alpha rest
        wavelength.

    self_absorption_parameter : ``float``, optional
        Self-absorption parameter "p" (unitless) - (reasonable range: 0-3 for
        Lya). Default is 0, or no self-absorption.

    Returns
    -------
    profile : ``numpy.ndarray``
        Stellar emission intrinsic profile.
    """
    line_rest_wl = reference_wavelength * 1E8  # Angstrom to cm
    gaussian_width_cm = gaussian_width * 1E5  # km / s to cm / s
    lorentzian_width_cm = lorentzian_width * 1E5  # km / s to cm / s
    star_velocity_cm = star_velocity * 1E5  # km / s to cm / s
    line_center = star_velocity_cm / LIGHT_SPEED * line_rest_wl + line_rest_wl
    sigma_gaussian = gaussian_width_cm / LIGHT_SPEED * line_rest_wl / 2.35482
        # Convert Gaussian FWHM to HWHM
    hwhm_lorentzian = lorentzian_width_cm / LIGHT_SPEED * line_rest_wl / 2
    normalization = 2 / (np.pi * lorentzian_width_cm /
                         LIGHT_SPEED * line_rest_wl)
    amplitude = 10 ** log_lorentzian_amplitude

    emission_profile = (normalization * amplitude *
                        voigt_profile(wavelength - line_center, sigma_gaussian,
                                      hwhm_lorentzian))
    reversal_profile = np.exp(-self_absorption_parameter * emission_profile /
                              np.max(emission_profile))
    profile = emission_profile * reversal_profile
    return profile


# ISM absorption profile
def ism_profile(wavelength, log_h1_column_density, gas_temperature,
                turbulence_velocity=0.0, los_velocity=0.0,
                deuterium_hydrogen_ratio=2e-5):
    """
    Calculates the ISM profile of a star for a given wavelength array and ISM
    properties.

    Parameters
    ----------
    wavelength: ``numpy.ndarray``
        Wavelength array.

    log_h1_column_density : ``float``
        Logarithm 10 of the column density of neutral hydrogen in the line of
        sight.

    gas_temperature : ``float``
        ISM gas isothermal temperature in Kelvin.

    turbulence_velocity : ``float``, optional
        ISM turbulence broadening velocity in km/s. Default is 0.0.

    los_velocity : ``float``, optional
        Line-of-sight velocity of the ISM cloud in km/s. Default is 0.0.

    deuterium_hydrogen_ratio : ``float``, optional
        Deuterium hydrogen ratio of the ISM cloud. Default is 2E-5.

    Returns
    -------
    absorption_profile : ``numpy.ndarray``
        ISM absorption profile in the Lyman-alpha and deuterium lines.
    """
    # User parameters
    n_hi = 10 ** log_h1_column_density  # column density of H I [cm^-2]
    n_di = n_hi * deuterium_hydrogen_ratio  # column density of D I [cm^-2]

    # Convert to Hz
    def _lambda_to_nu(lambda_cm):
        return LIGHT_SPEED / lambda_cm

    # Frequency array
    frequency = LIGHT_SPEED / (wavelength * 1e-8)  # Hz

    # Shifted line centers
    lambda_hi = LAMBDA_HI_REST * (1 + los_velocity * 1e5 / LIGHT_SPEED)
    lambda_di = LAMBDA_DI_REST * (1 + los_velocity * 1e5 / LIGHT_SPEED)
    nu_hi = _lambda_to_nu(lambda_hi)
    nu_di = _lambda_to_nu(lambda_di)

    # Doppler b parameters
    b_hi = np.sqrt(2 * BOLTZMANN_CONSTANT * gas_temperature / HYDROGEN_MASS +
                   turbulence_velocity ** 2)
    b_di = np.sqrt(2 * BOLTZMANN_CONSTANT * gas_temperature / DEUTERIUM_MASS +
                   turbulence_velocity ** 2)
    delta_nu_d_hi = nu_hi * b_hi / LIGHT_SPEED
    delta_nu_d_di = nu_di * b_di / LIGHT_SPEED
    a_hi = LYA_BROADENING / (4 * np.pi * delta_nu_d_hi)
    a_di = LYA_BROADENING / (4 * np.pi * delta_nu_d_di)

    # Voigt profile function
    def _voigt_profile(nu, nu_0, a, delta_nu_d):
        u = (nu - nu_0) / delta_nu_d
        return np.real(wofz(u + 1j * a)) / (delta_nu_d * np.sqrt(np.pi))

    # Cross-section coefficient
    sigma_0 = ((np.pi * ELECTRON_CHARGE ** 2) / (ELECTRON_MASS * LIGHT_SPEED) *
               LYA_OSCILLATOR_STRENGTH)  # [cm² Hz]

    # H I absorption
    phi_hi = _voigt_profile(frequency, nu_hi, a_hi, delta_nu_d_hi)
    tau_hi = sigma_0 * n_hi * phi_hi

    # D I absorption
    phi_di = _voigt_profile(frequency, nu_di, a_di, delta_nu_d_di)
    tau_di = sigma_0 * n_di * phi_di

    # Total optical depth and flux
    tau_total = tau_hi + tau_di
    absorption_profile = np.exp(-tau_total)

    return absorption_profile


# Calculate the Lyman-alpha profile as it would be observed with HST/STIS
def observed_lya_profile(grating, aperture, wavelength,
                         star_log_lorentzian_amplitude, star_lorentzian_width,
                         star_gaussian_width, ism_log_h1_column_density,
                         ism_gas_temperature, star_velocity=0.0,
                         star_self_absorption_parameter=0,
                         star_ov_log_lorentzian_amplitude=-20,
                         star_ov_lorentzian_width=999,
                         star_ov_gaussian_width=999,
                         ism_turbulence_velocity=0.0, ism_los_velocity=0.0,
                         ism_deuterium_hydrogen_ratio=2e-5,
                         return_all_profiles=False):
    """
    Calculates the Lyman-alpha profile of a star given its intrinsic parameters,
    parameters of the interstellar medium and the instrumental parameters.

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

    wavelength: ``numpy.ndarray``
        Wavelength array.

    star_log_lorentzian_amplitude : ``float``
        Logarithm of the amplitude of the Lorentzian (emission line) in
        erg/cm2/s/A (reasonable range: -13 to -9).

    star_lorentzian_width : ``float``
        FWHM of the Lorentzian in km/s (reasonable range: ~5-100).

    star_gaussian_width : ``float``
        FWHM of the Gaussian in km/s (reasonable range: ~10-200).

    ism_log_h1_column_density : ``float``
        Logarithm 10 of the column density of neutral hydrogen in the line of
        sight.

    ism_gas_temperature : ``float``
        ISM gas isothermal temperature in Kelvin.

    star_velocity : ``float``, optional
        Radial velocity of the star in km/s (reasonable range: -300 to +300).
        Default is 0.

    star_self_absorption_parameter : ``float``, optional
        Self-absorption parameter "p" (unitless) - (reasonable range: 0-3).
        Default is 0, or no self-absorption.

    star_ov_log_lorentzian_amplitude : ``float``, optional
        Logarithm of the amplitude of the OV emission line in erg/cm2/s/A.
        Default is -20 (nearly zero flux).

    star_ov_lorentzian_width : ``float``, optional
        FWHM of the OV emission line Lorentzian component in km/s. Default is
        999 (nearly zero flux).

    star_ov_gaussian_width : ``float``, optional
        FWHM of the OV emission line Gaussian component in km/s. Default is 999
        (nearly zero flux).

    ism_turbulence_velocity : ``float``, optional
        ISM turbulence broadening velocity in km/s. Default is 0.0.

    ism_los_velocity : ``float``, optional
        Line-of-sight velocity of the ISM cloud in km/s. Default is 0.0.

    ism_deuterium_hydrogen_ratio : ``float``, optional
        Deuterium hydrogen ratio of the ISM cloud. Default is 2E-5.

    return_all_profiles : ``bool``, optional
        Whether to return all profiles. Default is ``False``.

    Returns
    -------
    observed_lya_flux : ``numpy.ndarray``
        Observed Lya profile model (convolved to instrumental profile).

    observable_lya_profile : ``numpy.ndarray``
        Observable Lya profile model (not convolved to instrumental profile).

    intrinsic_profile : ``numpy.ndarray``
        Stellar intrinsic Lya profile (without ISM absorption).

    ism_absorption_profile : ``numpy.ndarray``
        ISM absorption Lya profile.
    """
    intrinsic_profile = intrinsic_stellar_profile(
        wavelength,
        star_velocity,
        star_log_lorentzian_amplitude,
        star_lorentzian_width,
        star_gaussian_width,
        self_absorption_parameter=star_self_absorption_parameter
    )
    ism_absorption_profile = ism_profile(wavelength,
                                         ism_log_h1_column_density,
                                         ism_gas_temperature,
                                         ism_turbulence_velocity,
                                         ism_los_velocity,
                                         ism_deuterium_hydrogen_ratio)
    ov_emission_profile = intrinsic_stellar_profile(
        wavelength,
        star_velocity,
        star_ov_log_lorentzian_amplitude,
        star_ov_lorentzian_width,
        star_ov_gaussian_width,
        reference_wavelength=LAMBDA_OV_REST
    )
    observable_lya_profile = ((intrinsic_profile + ov_emission_profile) *
                              ism_absorption_profile)

    lsf_wavelength, lsf_profile = get_stis_lsf(grating, aperture)
    lsf_wavelength += np.mean(wavelength)  # Necessary for convolution to work
    lsf_profile_interp = np.interp(wavelength, lsf_wavelength, lsf_profile,
                                   left=0.0, right=0.0)
    observed_lya_flux = convolve(observable_lya_profile, lsf_profile_interp)

    if return_all_profiles:
        return (observed_lya_flux, observable_lya_profile, intrinsic_profile,
                ism_absorption_profile)
    else:
        return observed_lya_flux
