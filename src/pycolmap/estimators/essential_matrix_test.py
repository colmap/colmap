# SPDX-License-Identifier: BSD-3-Clause

import pycolmap


def test_estimate_essential_matrix_is_callable() -> None:
    assert callable(pycolmap.estimate_essential_matrix)
