from astropy.table import Table
import astropy.units as u
import numpy as np
import pytest

from rocky_worlds_utils.jwst.archive_utils import (
    check_jwst_event_type,
    query_nexsci_archive,
)


def test_check_jwst_event_type():
    """Test event type for MAST archive file id 187223627. This test
    uses the orbital parameters from Rathcke et al. 2025

    https://ui.adsabs.harvard.edu/abs/2025ApJ...979L..19R/abstract
    """
    jwst_obs = {
        "ArchiveFileID": np.int64(187223627),
        "fileSetName": np.str_("jw01177007001_03101_00001"),
        "targprop": np.str_("TRAPPIST-1B"),
        "targ_ra": np.float64(346.6283008720181),
        "targ_dec": np.float64(-5.044440303727514),
        "instrume": np.str_("MIRI"),
        "exp_type": np.str_("MIR_IMAGE"),
        "opticalElements": np.str_("F1500W"),
        "date_obs": np.str_("2022-11-08T03:24:40.9510000"),
        "duration": np.float64(15690.076),
        "program": np.int64(1177),
        "observtn": np.int64(7),
        "visit": np.int64(1),
        "pi_name": np.str_("Greene, Thomas P."),
        "proposal_type": np.str_("GTO"),
        "proposal_cycle": np.int64(1),
        "targtype": np.str_("FIXED"),
        "access": np.str_("PUBLIC"),
        "s_region": np.str_(
            "POLYGON ICRS  346.619809552 -5.069614860 346.649678383 -5.059969988 346.639993670 -5.030074608 346.609987448 -5.039459713"
        ),
    }

    # check_jwst_event_type expects astropy table
    jwst_obs_table = Table([jwst_obs])

    # When query using NASA Exoplanet DB, astropy quantity table (has units)
    # is returned. Adding units to orbital params.
    orbital_period = 1.5108261 * u.day
    ephemeris = 2460501.405464 * u.day

    event_type = check_jwst_event_type(orbital_period, ephemeris, jwst_obs_table)

    assert event_type[187223627] == "NO EVENT"


@pytest.mark.parametrize(
    "planet_name, ra, dec",
    [
        ("LTT 1445 A b", 45.4624781, -16.5944956),
        ("TRAPPIST-1 g", 346.6263919, -5.0434618),
        ("GJ 357 b", 144.007464, -21.6650634),
        ("K2-415 b", 137.2015437, 11.8622503),
    ],
)
def test_query_nexsci_archive(planet_name, ra, dec):
    all_planet_data, _ = query_nexsci_archive(planet_name)

    assert all(np.isclose(all_planet_data["ra"].value, ra))
    assert all(np.isclose(all_planet_data["dec"].value, dec))
