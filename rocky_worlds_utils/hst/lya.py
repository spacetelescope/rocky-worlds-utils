#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to analyze Lyman-alpha profiles with HST data.
This code is inspired by the lyapy code authored by Allison Youngblood:
https://github.com/allisony/lyapy.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""


import numpy as np
from scipy.special import wofz, voigt_profile


__all__ = ["lya_intrinsic_profile", "ism_profile"]


# Physical constants
c = 2.99792458e10        # speed of light [cm/s]
k_B = 1.380649e-16       # Boltzmann constant [erg/K]
m_H = 1.6735575e-24      # hydrogen mass [g]
m_D = 2 * m_H            # deuterium mass [approximate]
e = 4.80320425e-10       # electron charge [esu]
m_e = 9.10938356e-28     # electron mass [g]
f = 0.4164               # oscillator strength for Lyman-alpha
gamma = 6.265e8          # natural broadening [s^-1]

# Line center wavelengths [cm]
lambda_HI_rest = 1215.67e-8
lambda_DI_rest = 1215.3394e-8


# Lyman-alpha intrinsic profile
def lya_intrinsic_profile(wavelength, star_velocity, log_lorentzian_amplitude,
                          lorentzian_width, gaussian_width,
                          self_absorption_parameter=0):
    """
    Calculates the Lyman-alpha intrinsic profile of a star for a given
    wavelength array.

    Parameters
    ----------
    wavelength : ``numpy.ndarray``
        Wavelength array in unit Angstrom.

    star_velocity : ``float``
        Radial velocity of the star in km/s (reasonable range: -300 to +300).

    log_lorentzian_amplitude : ``float``
        Logarithm of the amplitude of the Lorentzian (emission line) in
        erg/cm2/s/A (reasonable range: -13 to -9).

    lorentzian_width : ``float``
        FWHM of the Lorentzian in km/s (reasonable range: ~5-100).

    gaussian_width : ``float``
        FWHM of the Gaussian in km/s (reasonable range: ~10-200).

    self_absorption_parameter : ``float``, optional
        Self-absorption parameter "p" (unitless) - (reasonable range: 0-3).
        Default is 0, or no self-absorption.

    Returns
    -------
    intrinsic_profile : ``numpy.ndarray``
        Stellar Lyman-alpha intrinsic profile.

    """
    lya_rest_wl = lambda_HI_rest * 1E8
    gaussian_width_cm = gaussian_width * 1E5
    lorentzian_width_cm = lorentzian_width * 1E5
    star_velocity_cm = star_velocity * 1E5
    line_center = star_velocity_cm / c * lya_rest_wl + lya_rest_wl
    sigma_gaussian = gaussian_width_cm / c * lya_rest_wl / 2.35482
    hwhm_lorentzian = lorentzian_width_cm / c * lya_rest_wl / 2
    normalization = 2 / (np.pi * lorentzian_width_cm / c * lya_rest_wl)
    amplitude = 10 ** log_lorentzian_amplitude

    emission_profile = (normalization * amplitude *
                        voigt_profile(wavelength - line_center, sigma_gaussian,
                                      hwhm_lorentzian))
    reversal_profile = np.exp(-self_absorption_parameter * emission_profile /
                              np.max(emission_profile))
    intrinsic_profile = emission_profile * reversal_profile
    return intrinsic_profile


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
        return c / lambda_cm

    # Frequency array
    frequency = c / (wavelength * 1e-8)  # Hz

    # Shifted line centers
    lambda_hi = lambda_HI_rest * (1 + los_velocity * 1e5 / c)
    lambda_di = lambda_DI_rest * (1 + los_velocity * 1e5 / c)
    nu_hi = _lambda_to_nu(lambda_hi)
    nu_di = _lambda_to_nu(lambda_di)

    # Doppler b parameters
    b_hi = np.sqrt(2 * k_B * gas_temperature / m_H + turbulence_velocity ** 2)
    b_di = np.sqrt(2 * k_B * gas_temperature / m_D + turbulence_velocity ** 2)
    delta_nu_d_hi = nu_hi * b_hi / c
    delta_nu_d_di = nu_di * b_di / c
    a_hi = gamma / (4 * np.pi * delta_nu_d_hi)
    a_di = gamma / (4 * np.pi * delta_nu_d_di)

    # Voigt profile function
    def _voigt_profile(nu, nu_0, a, delta_nu_d):
        u = (nu - nu_0) / delta_nu_d
        return np.real(wofz(u + 1j * a)) / (delta_nu_d * np.sqrt(np.pi))

    # Cross-section coefficient
    sigma_0 = (np.pi * e ** 2) / (m_e * c) * f  # [cm² Hz]

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