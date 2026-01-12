import typing

import numpy as np


def fn(
    x: np.ndarray[
        tuple[typing.Literal[4], typing.Literal[1]],
        np.dtype[np.float64],
    ],
):
    print(x)


if __name__ == "__main__":
    arr: np.typing.NDArray[np.float64] = np.array([0.0, 0.0, 0.0, 1.0])
    fn(arr)
