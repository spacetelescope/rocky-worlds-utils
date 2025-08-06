#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for `tools` module.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

import numpy as np
from rocky_worlds_utils.hst.tools import nearest_index


# Test the nearest_index function
def test_nearest_index():
    array = np.linspace(1, 10, 10)
    ind = nearest_index(array, np.pi)
    assert ind == 2
