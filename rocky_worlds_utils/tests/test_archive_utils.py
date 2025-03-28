import numpy as np
import pytest

from rocky_worlds_utils.archive_utils import query_nexsci_archive


@pytest.mark.parametrize(
    "planet_name, ra, dec, preferred_idx",
    [
        ("LTT 1445 A b", 45.4624781, -16.5944956, 1),
        ("TRAPPIST-1 g", 346.6263919, -5.0434618, 2),
        ("GJ 357 b", 144.007464, -21.6650634, 2),
        ("K2-415 b", 137.2015437, 11.8622503, 1),
    ],
)
def test_query_nexsci_archive(planet_name, ra, dec, preferred_idx):
    all_planet_data, preferred_data_index = query_nexsci_archive(planet_name)

    assert preferred_data_index[0] == preferred_idx
    assert all(np.isclose(all_planet_data["ra"].value, ra))
    assert all(np.isclose(all_planet_data["dec"].value, dec))
