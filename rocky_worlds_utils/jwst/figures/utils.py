#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains class to collect all of the RWDDT eclipse depths metadata.

Authors
-------
- Mees Fix <<mfix@stsci.edu>>
"""

import fnmatch
import os

from astropy.io import ascii
from astropy.table import Table
import xarray as xr

from rocky_worlds_utils.jwst.constants import ECLIPSE_TABLE_DEFS


class EclipseDepthTable:
    def __init__(self, tablename):
        """Obtain eclipse depth measurements from Rocky Worlds DDT
        High Level Science Products.

        Parameters
        ----------
        tablename : str
            Absolute path of eclipse depth table. If tablename exists
            already, data will be loaded. If not, new table will be
            created.
        """
        self.tablename = tablename
        self.create_eclipse_depth_table()

    def add_eclipse_data_to_table(self, data_root):
        """Add new datasets to eclipse depths data table.

        Parameters
        ----------
        data_root : str
            Top level directory containing `eclipse-cat.h5` files. This
            root directory is searched recursively for datasets.
        """
        previous_datasets = self.eclipse_data["filename"]

        for root, _, files in os.walk(data_root):
            for filename in files:
                if fnmatch.fnmatch(filename, "*eclipse-cat.h5"):
                    if filename in previous_datasets:
                        continue
                    else:
                        print(f"FOUND NEW DATASET: {filename}")
                        abs_filename_path = os.path.join(root, filename)
                        new_entry = self.pull_eclipse_metadata(abs_filename_path)
                        self.eclipse_data.add_row(new_entry)

    def create_eclipse_depth_table(self):
        """Create an astropy table to store eclipse depth metadata.
        If table eclipse depth measurements table exists already, it will
        be loaded in.
        """
        if os.path.isfile(self.tablename):
            print(f"LOCATED EXISTING TABLE: {self.tablename}...LOADING")
            self.eclipse_data = open_eclipse_depths_table(self.tablename)
        else:
            print(
                f"NO EXISTING TABLE {self.tablename}, CREATING NEW EMPTY ECLIPSE TABLE"
            )
            self.eclipse_data = Table(dtype=ECLIPSE_TABLE_DEFS)

    def pull_eclipse_metadata(self, eclipse_dataset):
        """Open and pull attributes and data from datasets.
        This represents a single row in the eclipse depths table
        used to generate the eclipse depth figures.

        Parameters
        ----------
        eclipse_dataset : str
            Absolute path to eclipse depth HLSP.

        Returns
        -------
        new_entry : dict
            A dictionary containing all relevant data to be added
            to table.
        """
        data = xr.load_dataset(eclipse_dataset)
        new_entry = {
            "location": os.path.dirname(eclipse_dataset),
            "filename": os.path.basename(eclipse_dataset),
            "planet_name": data.PLANET,
            "star": data.STAR,
            "telescope": data.TELESCOP,
            "instrument": data.INSTRUME,
            "filter": data.filter.data[0],
            "propid": data.PROPOSID,
            "ra": data.RA_TARG,
            "dec": data.DEC_TARG,
            "mjd_beg": data.MJD_BEG,
            "mjd_mid": data.MJD_MID,
            "mjd_end": data.MJD_END,
            "eclipse_time": data.eclipseTime.data[0],
            "eclipse_time_err": data.eclipseTimeError.data[0],
            "eclipse_time_upper_err": data.eclipseTimeUpperError.data[0],
            "eclipse_time_lower_err": data.eclipseTimeLowerError.data[0],
            "eclipse_depth": data.eclipseDepth.data[0],
            "eclipse_depth_err": data.eclipseDepthError.data[0],
            "eclipse_depth_upper_err": data.eclipseDepthUpperError.data[0],
            "eclipse_depth_lower_err": data.eclipseDepthLowerError.data[0],
        }

        return new_entry

    def write_eclipse_table(self, output_name=None, overwrite=False):
        """Write out eclipse depth data table.

        Paramaters
        ----------
        output_name : str
            Optional parameter if user wants to write table out to new name
            than name provided at instantiation.

        overwrite : bool
            Flag to overwrite existing table. If table exists and set to False,
            table will not overwrite.
        """
        if output_name:
            writename = output_name
        else:
            writename = self.tablename

        ascii.write(
            self.eclipse_data,
            writename,
            format="fixed_width",
            header_rows=["name", "dtype"],
            overwrite=overwrite,
        )


def open_eclipse_depths_table(tablename):
    """Utility function to open tables formatted for RWDDT
    eclipse depth tables.

    Parameters
    ----------
    tablename : str
        Absolute path of eclipse depth table.

    Returns
    -------
    eclipse_data : astropy.table.Table
        Astropy table containing eclipse depth data.
    """
    eclipse_data = ascii.read(
        tablename, format="fixed_width", header_rows=["name", "dtype"]
    )

    return eclipse_data
