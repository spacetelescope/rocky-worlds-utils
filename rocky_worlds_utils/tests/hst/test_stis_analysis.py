from astropy.io import fits
from astroquery.mast import Observations
import numpy as np
import os

from rocky_worlds_utils.hst.stis_analysis import timetag_split
from rocky_worlds_utils.tests.hst.utils import download_stis_data

ON_GITHUB_ACTIONS = "/home/runner" in os.path.expanduser(
    "~"
) or "/Users/runner" in os.path.expanduser("~")

if ON_GITHUB_ACTIONS:
    download_stis_data()


def test_stis_timetag_split():
    input_dir = os.path.join(os.getcwd())
    output_dir = os.path.join(os.getcwd(), "stis_analysis_output")

    timetag_split(
        "oev303010", output_dir=output_dir, prefix=input_dir, n_subexposures=1
    )

    time_series_file = os.path.join(output_dir, "oev303010_ts_x1d.fits")

    assert os.path.exists(time_series_file)
