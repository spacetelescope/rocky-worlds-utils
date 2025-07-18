import numpy as np
import pytest

from rocky_worlds_utils.jwst.archive_utils import query_nexsci_archive


@pytest.mark.parametrize(
    "planet_name, ra, dec",
    [
        ("LTT 1445 A b", 45.4624781, -16.5944956),
        ("TRAPPIST-1 g", 346.6263919, -5.0434618),
        ("GJ 357 b", 144.007464, -21.6650634),
        ("K2-415 b", 137.2015437, 11.8622503),
    ],
)
def test_query_nexsci_archive(planet_name, ra, dec):
    all_planet_data, _ = query_nexsci_archive(planet_name)

    assert all(np.isclose(all_planet_data["ra"].value, ra))
    assert all(np.isclose(all_planet_data["dec"].value, dec))
