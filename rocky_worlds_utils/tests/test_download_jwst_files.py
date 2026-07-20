"""Tests for the JWST-specific MAST download helper."""

from unittest.mock import Mock

from astropy.table import Table
import pytest

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


def test_retry_data_unavailable_retries_only_mast_availability(monkeypatch):
    """Unavailable MAST data retry the complete operation after a delay."""
    operation = Mock(side_effect=[
        ValueError("No data found for proposal 01234, observation 001, visit 001"),
        "complete",
    ])
    sleep = Mock()
    monkeypatch.setattr(download_jwst_files.time, "sleep", sleep)

    result = download_jwst_files.retry_data_unavailable(
        operation, retry_seconds=7, max_attempts=2
    )

    assert result == "complete"
    assert operation.call_count == 2
    sleep.assert_called_once_with(7)


def test_retry_data_unavailable_stops_on_other_errors():
    """Unexpected failures are not retried."""
    operation = Mock(side_effect=RuntimeError("connection failed"))

    with pytest.raises(RuntimeError, match="connection failed"):
        download_jwst_files.retry_data_unavailable(operation, retry_seconds=1)

    operation.assert_called_once()
