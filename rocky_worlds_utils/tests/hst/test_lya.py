"""
Module for testing HST lya.py

Authors
-------
- Leonardo Dos Santos <<ldsantos@stsci.edu>>
"""

import numpy as np

from rocky_worlds_utils.hst.lya import observed_lya_profile


# Stellar parameters
star_velocity = -10.0  # cm/s
log_lorentzian_amplitude = -12  # flam
lorentzian_width = 100  # km/s
gaussian_width = 200  # km/s
self_absorption_parameter = 0
ov_log_lorentzian_amplitude = -15  # flam
ov_lorentzian_width = 10  # km/s
ov_gaussian_width = 20  # km/s

# ISM parameters
n_hi = 18  # cm^-2
temp = 8000  # K

wavelength = np.linspace(1210, 1220, 999)


def test_observed_lya_profile():
    lya_profile = observed_lya_profile(
        grating='G140M',
        aperture='52x0.2',
        wavelength=wavelength,
        star_log_lorentzian_amplitude=log_lorentzian_amplitude,
        star_lorentzian_width=lorentzian_width,
        star_gaussian_width=gaussian_width,
        ism_log_h1_column_density=n_hi,
        ism_gas_temperature=temp,
        star_velocity=0.0,
        star_self_absorption_parameter=0,
        star_ov_log_lorentzian_amplitude=ov_log_lorentzian_amplitude,
        star_ov_lorentzian_width=ov_lorentzian_width,
        star_ov_gaussian_width=ov_gaussian_width,
        ism_turbulence_velocity=0.0,
        ism_los_velocity=0.0,
        return_all_profiles=False
    )
    expected = 5.578568567509619e-13
    result = np.max(lya_profile)

    assert np.isclose(result, expected, atol=1e-3)
