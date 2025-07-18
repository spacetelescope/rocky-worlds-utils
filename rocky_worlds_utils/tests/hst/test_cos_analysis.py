from astropy.io import fits
from astroquery.mast import Observations
import numpy as np
import os

from rocky_worlds_utils.hst.cos_analysis import timetag_split
from rocky_worlds_utils.tests.hst.utils import download_cos_data

ON_GITHUB_ACTIONS = "/home/runner" in os.path.expanduser(
    "~"
) or "/Users/runner" in os.path.expanduser("~")

if ON_GITHUB_ACTIONS:
    download_cos_data()


def test_cos_timetag_split():
    input_dir = os.path.join(os.getcwd())
    output_dir = os.path.join(os.getcwd(), "cos_analysis_output")

    timetag_split(
        "lcil2ajnq", output_dir=output_dir, prefix=input_dir, n_cpus=1, n_subexposures=1
    )

    time_series_file = os.path.join(output_dir, "lcil2ajnq_ts_x1d.fits")

    assert os.path.exists(time_series_file)
