# SPDX-License-Identifier: BSD-3-Clause

"""cusift must give bit-identical output across separate processes.

colmap#1197 reports GPU SIFT returning different results for the same image
in different runs. The C++ tests (cusift_determinism_test.cc) cover repeated
calls within one process; this covers fresh processes, where any dependence
on device or allocator state would show up.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

NUM_PROCESSES = 4

# Runs in a fresh interpreter. Extracts features from the image saved at
# argv[1] and prints a digest of every keypoint float and every descriptor
# byte, in output order.
_EXTRACT_SCRIPT = """
import hashlib
import struct
import sys

import numpy as np

import pycolmap

image = np.load(sys.argv[1])

options = pycolmap.FeatureExtractionOptions(
    type=pycolmap.FeatureExtractorType.CUSIFT
)
extractor = pycolmap.FeatureExtractor.create(
    options=options, device=pycolmap.Device.cuda
)
keypoints, descriptors = extractor.extract_from_uint8_array(image)
digest = hashlib.sha256()
for kp in keypoints:
    digest.update(struct.pack("6f", kp.x, kp.y, kp.a11, kp.a12, kp.a21, kp.a22))
digest.update(np.ascontiguousarray(descriptors.data).tobytes())
print(len(keypoints), digest.hexdigest())
"""


def _run_in_fresh_process(image_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _EXTRACT_SCRIPT, str(image_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.usefixtures("requires_cusift")
def test_cusift_extraction_identical_across_processes(
    overlapping_squares_image: np.ndarray, tmp_path: Path
) -> None:
    image_path = tmp_path / "image.npy"
    np.save(image_path, overlapping_squares_image)
    outputs = [_run_in_fresh_process(image_path) for _ in range(NUM_PROCESSES)]
    num_keypoints = int(outputs[0].split()[0])
    # Guard against a vacuous pass on an image with almost no features.
    assert num_keypoints >= 200
    assert len(set(outputs)) == 1, outputs
