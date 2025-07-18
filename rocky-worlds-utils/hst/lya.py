#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to analyze Lyman-alpha profiles with HST data.
This code is heavily inspired by the lyapy code authored by Allison Youngblood:
https://github.com/allisony/lyapy.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""


import numpy as np
import astropy.units as u
import astropy.constants as c

from astropy.modeling.models import Voigt1D
from scipy.special import voigt_profile


__all__ = ["lya_intrinsic_profile", "ism_profile"]


# Defining global constants
_LYMAN_A_REST_WAVELENGTH = 1215.6702  # Angstrom
_LYMAN_A_OSCILLATOR_STRENGTH = 0.4161
_LYMAN_A_DAMPENING = 6.26E8
_DEUTERIUM_REST_WAVELENGTH = 1215.3394  # Angstrom
_DEUTERIUM_OSCILLATOR_STRENGTH = 0.4161
_DEUTERIUM_DAMPENING = 6.27e8
_DEUTERIUM_TO_H_RATIO = 1.5E-5
_C = c.c.to(u.km / u.s).value
_C_CGS = 2.99792458e10
_PROTON_MASS_CGS = 1.6726231e-24  # Proton mass in grams
_ELECTRON_MASS_CGS = _PROTON_MASS_CGS / 1836  # Electron mass in grams
_ELECTRON_CHARGE = 4.8032e-10  # Electron charge in esu


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
        Wavelength array.

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
    lya_rest_wl = _LYMAN_A_REST_WAVELENGTH
    line_center = star_velocity / _C * lya_rest_wl + lya_rest_wl
    sigma_gaussian = gaussian_width / _C * lya_rest_wl / 2.35482
    hwhm_lorentzian = lorentzian_width / _C * lya_rest_wl / 2
    normalization = 2 / (np.pi * lorentzian_width / _C * lya_rest_wl)
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
                los_velocity=0.0):
    """
    """
    # Some necessary bookkeeping here
    lya_rest_wl = _LYMAN_A_REST_WAVELENGTH
    lya_f = _LYMAN_A_OSCILLATOR_STRENGTH
    ly_a = _LYMAN_A_DAMPENING
    deuterium_rest_wl = _DEUTERIUM_REST_WAVELENGTH
    deuterium_f = _DEUTERIUM_OSCILLATOR_STRENGTH
    deuterium_a = _DEUTERIUM_DAMPENING
    k_b = 1.380649e-23  # Boltzmann's constant in J / K
    lya_nu0 = _C / lya_rest_wl  # Reference frequency in Hz

    frequency = _C_CGS / wavelength / 1E8  # Hz


def absorption_cross_section(frequency, rest_frequency, oscillator_strength,
                             einstein_coefficient, broadening_temperature,
                             particle_mass, velocity_los=0.0):
    """

    Parameters
    ----------
    frequency
    rest_frequency
    oscillator_strength
    einstein_coefficient
    broadening_temperature
    particle_mass
    velocity_los

    Returns
    -------

    """
    # Constants in SI
    c_speed = 2.99792458e+08  # Speed of light in m / s
    k_b = 1.380649e-23  # Boltzmann's constant in J / K

    # Calculate the Lorentzian width of the Voigt profile
    gamma = einstein_coefficient / 4 / np.pi

    # Calculate Doppler width (standard deviation) of the Voigt profile
    alpha_nu = (rest_frequency / c_speed *
                (k_b * broadening_temperature / particle_mass) ** 0.5)

    # Calculate the frequency shifts due to gas velocity
    velocity_los_si = velocity_los * 1E3  # Convert to m / s
    delta_nu_add = velocity_los_si / c_speed * rest_frequency

    # Calculate absorption Voigt profile
    profile = voigt_profile(frequency + delta_nu_add, alpha_nu, gamma)

    # Calculate the absorption cross-section
    cross_section = 2.654008854574474e-06 * oscillator_strength * profile
    return cross_section