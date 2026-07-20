"""Tests for the JWST-specific MAST download helper."""

from unittest.mock import Mock

from astropy.table import Table

from ..jwst import download_jwst_files


def test_query_mast_uses_jwst_mission_fields(monkeypatch):
    """The download helper queries the JWST API, not CAOM criteria."""
    mission = Mock()
    datasets = Table({"fileSetName": ["jw01234001001_02101_00001"]})
    products = Table({"filename": ["jw01234001001_02101_00001_nircam_uncal.fits"]})
    mission.query_criteria.return_value = datasets
    mission.get_product_list.return_value = products
    mission.filter_products.return_value = products
    monkeypatch.setattr(download_jwst_files, "jwst_mast", mission)

    result = download_jwst_files.query_MAST("1234", "1", "1", "UNCAL")

    mission.query_criteria.assert_called_once_with(program=1234, observtn=1, visit=1)
    mission.get_product_list.assert_called_once_with(datasets)
    mission.filter_products.assert_called_once_with(
        products, file_suffix="_uncal", extension=".fits"
    )
    assert result["filename"][0].endswith("_uncal.fits")
