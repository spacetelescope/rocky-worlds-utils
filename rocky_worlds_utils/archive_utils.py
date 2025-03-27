#! /usr/bin/env python
"""
Utilities module for the Rocky Worlds DDT project.

Authors
-------
- Mees Fix

Use
---
>>> from rocky_worlds_ddt.utils import check_jwst_observations
"""

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.mast import MastMissions
import numpy as np


def check_jwst_observations(ra, dec, radius=0.1):
    """Check MAST to see if target ra and dec has JWST observations.

    Parameters
    ----------
    ra : float
        Target Right Ascension (units: deg)
    dec : float
        Target Declination (units: deg)
    radius : float
        Search radius around ra and dec [default: 0.1]

    Returns
    -------
    results : astropy.Table.table
        Astropy table of query results
    """
    regionCoords = SkyCoord(ra, dec, unit=("deg", "deg"))
    missions = MastMissions(mission="jwst")
    results = missions.query_region(
        regionCoords,
        radius=radius,
        select_cols=[
            "targprop",
            "targ_ra",
            "targ_dec",
            "instrume",
            "exp_type",
            "opticalElements",
            "date_obs",
            "duration",
            "program",
            "observtn",
            "visit",
            "pi_name",
            "proposal_type",
            "proposal_cycle",
            "targtype",
            "access",
        ],
    )

    return results


def check_jwst_observation_type(
    target_name, period, planet_ephemeris, jwst_observations
):
    """
    This fough code tries to figure out, given some target information and
    an observation start and end time,what exoplanet event is being
    observed: transit, eclipse, phase curve, or nothing.

    This code may be buggy and has not been well-tested.

    WARNING: This only makes any sense if you already know what host star
    is being observing for the observation start/end, and then test the
    planets in that system. Otherwise, you can put in any time range,
    and for a large enough sample of planets, you will probably randomly
    hit on a transit/eclispe/phase curve event. This may be obvious but
    there you are.

    Parameters
    ----------
    target_name : str
        Name of your target
    period : float
        Period of exoplanet
    planet_ephemeris : float
        The reference time for a transit mid-point.
    jwst_observations : astropy.Table.table
        Astropy table of results from astroquery.mast.MastMissions('jwst') query.
        See `query_mast_jwst_archive` function.
    """

    # Create time objects for API data to get them in same units as Exoplanet Archive data.
    obs_start = (
        Time(jwst_observations["date_obs"], format="isot", scale="utc").jd * u.day
    )
    obs_end = obs_start + (jwst_observations["duration"] * u.second).to(u.day)

    for row, (start, end) in enumerate(zip(obs_start, obs_end)):
        n_start = (start - planet_ephemeris) / period
        n_end = (end - planet_ephemeris) / period
        n = int(n_start)

        print(
            f"This observation starts at n={n_start:0.3f} orbits after ephemeris and ends at n={n_end:0.3f} after"
        )

        phase_start = n_start - n
        phase_end = n_end - n

        print(
            f"This observation spans from phase={phase_start:0.3f} to phase={phase_end:0.3f}"
        )
        if phase_end - phase_start > 1:
            print(f"This observation probably contains a phase curve of {target_name}")
        elif phase_start < 0.5 < phase_end:
            print(
                f"This observation probably contains a secondary eclipse of {target_name}"
            )
        elif phase_start < 1.0 < phase_end:
            print(f"This observation probably contains a transit of {target_name}")
        else:
            print(
                f"This observation probably does not include an event for {target_name}; or check your parameters"
            )
        print(jwst_observations[row])


def query_nexsci_archive(target_name):
    """Query NASA NexSci Exoplanet Archive for planetary parameters.

    Parameters
    ----------
    target_name : str
        Name of exoplanet to query database on. The string is case sensitive.
    """
    all_planet_data = NasaExoplanetArchive.query_criteria(
        table="ps",
        select="pl_name, ra, dec, pl_orbper, pl_tranmid, pl_refname, default_flag",
        where=f"pl_name='{target_name}'",
    )

    # See if the preferred data set from NEA
    preferred_data_index = np.where(all_planet_data["default_flag"] == 1)[0]

    return all_planet_data, preferred_data_index
