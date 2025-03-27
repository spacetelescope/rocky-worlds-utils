import numpy as np

from rocky_worlds_utils.archive_utils import query_nexsci_archive


def test_query_nexsci_archive():
    planet_name = "LTT 1445 A b"
    all_planet_data, preferred_data_index = query_nexsci_archive(planet_name)

    assert preferred_data_index[0] == 1
    assert all(np.isclose(all_planet_data["ra"].value, 45.4624781))
    assert all(np.isclose(all_planet_data["dec"].value, -16.5944956))
