"""
Module for testing HST spectrum.py

Authors
-------
- Mees B Fix <<mfix@stsci.edu>>
"""

import numpy as np
import os
import pytest

from rocky_worlds_utils.hst.spectrum import read_hsla_product, calculate_snr_hsla


@pytest.mark.parametrize(
    "filename",
    [
        (os.path.join(os.getcwd(), "lcil2ajnq_x1d.fits")),
        (os.path.join(os.getcwd(), "oev303010_x1d.fits")),
    ],
)
def test_read_hsla_product(filename):
    """Test that mock HSLA file is readable."""

    _ = read_hsla_product(filename)

    # If file is read with no errors, pass.
    assert True


@pytest.mark.order(
    after=["test_read_hsla_product", "test_stis_timetag_split"]
)  # Ensures product can be read first and that data exists
def test_calculate_snr_hsla():
    """Calculate SNR for STIS testing spectrum."""
    filename = os.path.join(os.getcwd(), "oev303010_x1d.fits")
    wavelength, flux, err = read_hsla_product(filename)
    expected = 46.447159145177686
    result = calculate_snr_hsla(wavelength, flux, err)

    assert np.isclose(expected, result, atol=5)
