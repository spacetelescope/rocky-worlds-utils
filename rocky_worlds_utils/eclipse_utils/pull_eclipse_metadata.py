import datetime
from pathlib import Path

from astropy.io import ascii
from astropy.table import Table
import numpy as np
import xarray as xr

from constants import ECLIPSE_TABLE_DEFS
from utils import open_eclipse_depths_table


class eclipseDepthTable:
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
        self.tablename = Path(tablename)
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

        for dataset in Path(data_root).rglob("*eclipse-cat.h5"):
            if dataset.name in previous_datasets:
                print(f"{dataset.name} EXISTS in {self.tablename.name}, CONTINUING")
                continue
            else:
                print(f"FOUND NEW DATASET: {dataset.name}")
                new_entry = self.pull_eclipse_metadata(dataset)
                self.eclipse_data.add_row(new_entry)

    def create_eclipse_depth_table(self):
        """Create an astropy table to store eclipse depth metadata.
        If table eclipse depth measurements table exists already, it will
        be loaded in.
        """
        if self.tablename.is_file():
            print(f"LOCATED EXISTING TABLE: {self.tablename}")
            self.eclipse_data = open_eclipse_depths_table(self.tablename)
            print("TABLE LOADED")
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
            "location": eclipse_dataset.parent,
            "filename": eclipse_dataset.name,
            "planet_name": data.PLANET,
            "star": data.STAR,
            "telescope": data.TELESCOP,
            "instrument": data.INSTRUME,
            "filter": data.FILTER,
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

    def write(self, output_name=None, overwrite=False):
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

        print(f"WROTE ECLIPSE TABLE TO {writename}")
