# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt


def normalize_vec(
    vec: npt.NDArray[np.floating], eps: float = 1e-10
) -> npt.NDArray[np.floating]:
    return vec / max(eps, float(np.linalg.norm(vec)))


def vec_angular_dist_deg(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> float:
    """Return the angle between directions independently of their magnitudes."""
    # Scale before normalization to avoid overflow and the small-norm floor.
    scale1, scale2 = np.max(np.abs(vec1)), np.max(np.abs(vec2))
    if scale1 > 0:
        vec1 = vec1 / scale1
    if scale2 > 0:
        vec2 = vec2 / scale2
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(np.acos(cos_dist))
