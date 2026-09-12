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
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(np.acos(cos_dist))
