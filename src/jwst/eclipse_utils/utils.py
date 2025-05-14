"""Utilities module for eclipse depths."""

from astropy.io import ascii


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
