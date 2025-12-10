from astropy.io import fits
import numpy as np
import os
import pytest


from rocky_worlds_utils.hst.time_series import integrate_flux, read_fits


@pytest.mark.order(after="test_cos_timetag_split")  # Ensures file to exists first
@pytest.mark.parametrize(
    "exp_flux, exp_flux_err, exp_gross, exp_gross_err, mean_sens, return_integrated_gross",
    [
        (
            6.413406112662675e-11,
            1.0756661559158064e-09,
            None,
            None,
            None,
            False,
        ),
        (
            6.413406112662675e-11,
            1.0756661559158064e-09,
            2364.58349609375,
            49.63469436620062,
            2.0667177e-12,
            True,
        ),
    ],
)
def test_integrate_flux(
    exp_flux,
    exp_flux_err,
    exp_gross,
    exp_gross_err,
    mean_sens,
    return_integrated_gross,
):
    filename = os.path.join(os.getcwd(), "lcil2ajnq_x1d.fits")
    hdu = fits.open(filename)

    wavelength = hdu[1].data["WAVELENGTH"].ravel()
    flux = hdu[1].data["FLUX"].ravel()
    gross = hdu[1].data["GROSS"].ravel()
    net = hdu[1].data["NET"].ravel()
    exptime = hdu[1].header["EXPTIME"]

    if return_integrated_gross:
        result_flux, result_flux_err, result_gross, result_gross_err, result_sens = integrate_flux(
            (1600.0, 1700.0),
            wavelength,
            flux,
            gross,
            net,
            exptime,
            return_integrated_gross=return_integrated_gross,
        )
        assert (
            np.isclose(result_flux, exp_flux)
            & np.isclose(result_flux_err, exp_flux_err)
            & np.isclose(result_gross, exp_gross)
            & np.isclose(result_gross_err, exp_gross_err)
            & np.isclose(result_sens, mean_sens)
        )
    else:
        result_flux, result_flux_err = integrate_flux(
            (1600.0, 1700.0),
            wavelength,
            flux,
            gross,
            net,
            exptime,
            return_integrated_gross=return_integrated_gross,
        )
        assert np.isclose(result_flux, exp_flux) & np.isclose(
            result_flux_err, exp_flux_err
        )


@pytest.mark.order(
    after=["test_cos_timetag_split", "test_stis_timetag_split"]
)  # make sure _ts_x1d.fits products exists in test env.
@pytest.mark.parametrize(
    "dataset, prefix",
    [
        ("lcil2ajnq", os.path.join(os.getcwd(), "cos_analysis_output")),
        ("oev303010", os.path.join(os.getcwd(), "stis_analysis_output")),
    ],
)
def test_read_fits(dataset, prefix):
    result = read_fits(dataset, prefix=prefix)

    if result["instrument"] == "COS":
        assert (
            (result["ra"] == 176.810576708)
            & (result["dec"] == 61.25883333318)
            & (result["proposal_id"] == 13635)
            & (result["cenwave"] == 1600)
            & (result["target"] == "V-KL-UMA")
        )
    elif result["instrument"] == "STIS":
        assert (
            (result["ra"] == 129.6881578724)
            & (result["dec"] == -13.25644837046)
            & (result["proposal_id"] == 17221)
            & (result["cenwave"] == 1222)
            & (result["target"] == "HD-73583")
        )
    else:
        raise ValueError(f"WE DO NOT SUPPORT INSTRUMENT: {result['instrument']}")
