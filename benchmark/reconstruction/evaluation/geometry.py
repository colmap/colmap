# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt


def normalize_vec(vec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    scale = float(np.max(np.abs(vec)))
    if scale == 0.0:
        return np.zeros_like(vec)
    scaled = vec / scale
    return scaled / float(np.linalg.norm(scaled))


def vec_angular_dist_deg(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> float:
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(np.acos(cos_dist))
