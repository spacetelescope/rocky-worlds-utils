"""Utils for testing HST data analysis tools

Authors
-------
- Mees B Fix <<mfix@stsci.edu>>
"""

from astroquery.mast import MastMissions
from crds import assign_bestrefs
import glob

from astropy.io import fits
import numpy as np


def assign_reffiles(files, task_verbosity=0):
    """Assign and download reference files

    files : list
        List of absolute path for files to pull refs for

    task_verbosity : int
        Level of output from assign_bestrefs [default: 0]
    """
    assign_bestrefs(files, sync_references=True, verbosity=task_verbosity)


def download_cos_data():
    hst = MastMissions(mission="hst")

    query = hst.query_criteria(sci_data_set_name="lcil2a010")
    products = hst.get_product_list(query)
    products = products.filled()
    products = products[
        [
            filename.endswith(".fits")
            and suffix
            in {
                "X1D",
                "CORRTAG",
                "CORRTAG_A",
                "CORRTAG_B",
                "RAWTAG",
                "RAWTAG_A",
                "RAWTAG_B",
            }
            for suffix, filename in zip(products["file_suffix"], products["filename"])
        ]
    ]

    hst.download_products(products, flat=True)

    files = glob.glob("lcil2ajnq*")
    assign_reffiles(files)


def download_stis_data():
    hst = MastMissions(mission="hst")

    query = hst.query_criteria(sci_data_set_name="oev303010")
    products = hst.get_product_list(query)
    products = products.filled()
    products = products[
        [
            filename.endswith(".fits")
            and suffix
            in {
                "X1D",
                "WAV",
                "TAG",
                "FLT",
            }
            for suffix, filename in zip(products["file_suffix"], products["filename"])
        ]
    ]

    hst.download_products(products, flat=True)

    files = glob.glob("oev303010*")
    assign_reffiles(files)
