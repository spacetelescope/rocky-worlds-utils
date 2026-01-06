#! /usr/bin/env python
"""
JWST Archive utilities module for the Rocky Worlds DDT project. Most of the functions
here are used to query for targets included in the RWDDT TUC that are not a part of
our programs.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>

Use
---
>>> from rocky_worlds_utils.jwst.archive_utils import query_nexsci_archive
>>> data, pref_idx = query_nexsci_archive("TRAPPIST-1 b")
"""

from copy import copy
import warnings

from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn
from astropy.time import Time
import astropy.units as u
from astroquery.mast import MastMissions
import numpy as np


from rocky_worlds_utils.constants import JWST_PROGRAMS

__all__ = [
    "check_jwst_observations",
    "check_jwst_event_type",
    "query_nexsci_archive",
]


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

    # drop rows that contain observations from RWDDT programs
    for rwddt_program in JWST_PROGRAMS:
        if rwddt_program in results["program"]:
            idx = np.where(results["program"] == rwddt_program)[0]
            results.remove_rows(idx)
        else:
            continue

    results.add_index("ArchiveFileID")

    return results


def check_jwst_event_type(period, planet_ephemeris, jwst_observations):
    """
    This function tries to figure out, given some target information and
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
    period : float
        Period of exoplanet

    planet_ephemeris : float
        The reference time for a transit mid-point.

    jwst_observations : astropy.Table.table
        Astropy table of results from astroquery.mast.MastMissions('jwst') query.
        See `query_mast_jwst_archive` function.

    Returns
    -------
    event_types : list
        List of the event types as they appear in the jwst_observations table.
    """

    # If table already has column, no need to add it.
    if "event_type" in jwst_observations.columns:
        raise ValueError("event_type column already exists in jwst_observations table!")
    else:
        # Don't overwrite table in memory, lets make a copy of it.
        observations_with_events = copy(jwst_observations)

        # Set max string length in astropy table column
        max_string_length = len("SECONDARY ECLIPSE")

        # Create an empty string column of the desired length
        event_type_col = MaskedColumn(
            name="event_type",
            dtype="U" + str(max_string_length),  # Use 'U' for Unicode strings
            length=len(observations_with_events),
        )

        # Add the empty string column to the table
        observations_with_events.add_column(event_type_col)

    # Create time objects for API data to get them in same units as Exoplanet Archive data.
    obs_start = (
        Time(observations_with_events["date_obs"], format="isot", scale="utc").jd
        * u.day
    )

    obs_end = obs_start + (observations_with_events["duration"] * u.second).to(u.day)

    file_ids = observations_with_events["ArchiveFileID"]

    for _, (start, end, fileid) in enumerate(zip(obs_start, obs_end, file_ids)):
        n_start = (start - planet_ephemeris) / period
        n_end = (end - planet_ephemeris) / period
        n = int(n_start)

        phase_start = n_start - n
        phase_end = n_end - n

        if phase_end - phase_start > 1:
            event_type = "PHASE CURVE"
        elif phase_start < 0.5 < phase_end:
            event_type = "SECONDARY ECLIPSE"
        elif phase_start < 1.0 < phase_end:
            event_type = "TRANSIT"
        else:
            event_type = "NO EVENT"

        # Locate row for file ID in table and assign event type.
        observations_with_events.loc["ArchiveFileID", fileid]["event_type"] = event_type

    return observations_with_events


def query_nexsci_archive(target_name):
    """Query NASA NexSci Exoplanet Archive for planetary parameters.

    Parameters
    ----------
    target_name : str
        Name of exoplanet to query database on. The string is case sensitive.

    Returns
    -------
    all_planet_data : astropy.table.Table
        Astropy table containing planet meta data requested from NexSci

    preferred_data_index : np.array
        Numpy array with row value of community preferred measurements
    """
    all_planet_data = NasaExoplanetArchive.query_criteria(
        table="ps",
        select="pl_name, ra, dec, pl_orbper, pl_tranmid, pl_refname, default_flag",
        where=f"pl_name='{target_name}'",
    )

    # See if the preferred data set from NEA
    preferred_data_index = np.where(all_planet_data["default_flag"] == 1)[0]

    period, ephemeris, refname = (
        all_planet_data[preferred_data_index]["pl_orbper"][0].value,
        all_planet_data[preferred_data_index]["pl_tranmid"][0].value,
        all_planet_data[preferred_data_index]["pl_refname"][0],
    )

    if period or ephemeris is None:
        warning_msg = f"""Returned result from NEA for orbital period and planetary ephemeris are None or NaN. Review results for planetary parameters that have calculated values.
pl_orbper = {period}
pl_tranmid = {ephemeris}
pl_refname = {refname}"""
        warnings.warn(warning_msg)

    return all_planet_data, preferred_data_index
