#! /usr/bin/env python
"""
Tests for the Rocky Worlds DDT JWST archive utility module.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

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
    uses the orbital parameters from Xue et al. 2024

    https://ui.adsabs.harvard.edu/abs/2024arXiv240813340X/abstract
    """
    jwst_obs = {
        "ArchiveFileID": np.int64(185871464),
        "fileSetName": np.str_("jw01274012001_04103_00001"),
        "targprop": np.str_("GJ1132"),
        "targ_ra": np.float64(153.7056262595806),
        "targ_dec": np.float64(-47.15401570103403),
        "instrume": np.str_("MIRI"),
        "exp_type": np.str_("MIR_LRS-SLITLESS"),
        "opticalElements": np.str_("P750L"),
        "date_obs": np.str_("2023-07-01T14:09:11.2610000"),
        "duration": np.float64(16575.944),
        "program": np.int64(1274),
        "observtn": np.int64(12),
        "visit": np.int64(1),
        "pi_name": np.str_("Lunine, Jonathan I."),
        "proposal_type": np.str_("GTO"),
        "proposal_cycle": np.int64(1),
        "targtype": np.str_("FIXED"),
        "access": np.str_("PUBLIC"),
        "s_region": np.str_(
            "POLYGON ICRS  153.694946544 -47.148072635 153.697311445 -47.146570037 153.710253529 -47.155835468 153.707886277 -47.157327651"
        ),
    }

    # check_jwst_event_type expects astropy table
    jwst_obs_table = Table([jwst_obs])
    jwst_obs_table.add_index("ArchiveFileID")

    # When query using NASA Exoplanet DB, astropy quantity table (has units)
    # is returned. Adding units to orbital params.
    orbital_period = 1.62892911 * u.day
    ephemeris = 2459280.98988 * u.day

    event_type = check_jwst_event_type(orbital_period, ephemeris, jwst_obs_table)

    assert event_type.loc[185871464]["event_type"] == "SECONDARY ECLIPSE"


@pytest.mark.parametrize(
    "planet_name, ra, dec",
    [
        ("LTT 1445 A b", 45.4624781, -16.5944956),
        ("GJ 357 b", 144.007464, -21.6650634),
        ("K2-415 b", 137.2015437, 11.8622503),
    ],
)
def test_query_nexsci_archive(planet_name, ra, dec):
    all_planet_data, _ = query_nexsci_archive(planet_name)

    assert all(np.isclose(all_planet_data["ra"].value, ra))
    assert all(np.isclose(all_planet_data["dec"].value, dec))
