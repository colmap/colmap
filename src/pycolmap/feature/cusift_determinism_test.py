# SPDX-License-Identifier: BSD-3-Clause

"""cusift must give bit-identical output across separate processes.

colmap#1197 reports GPU SIFT returning different results for the same image
in different runs. The C++ tests (cusift_determinism_test.cc) cover repeated
calls within one process; this covers fresh processes, where any dependence
on device or allocator state would show up.
"""

import subprocess
import sys

import pytest

import pycolmap

requires_cusift = pytest.mark.skipif(
    not pycolmap.has_cuda, reason="cusift requires CUDA"
)

NUM_PROCESSES = 4

# Runs in a fresh interpreter. Extracts features from a fixed synthetic image
# (many overlapping squares, so there are several hundred keypoints) and
# prints a digest of every keypoint float and every descriptor byte, in output
# order.
_EXTRACT_SCRIPT = """
import hashlib
import struct

import numpy as np

import pycolmap

rng = np.random.default_rng(42)
image = np.full((480, 640), 40, dtype=np.uint8)
for _ in range(150):
    x, y = int(rng.integers(0, 640)), int(rng.integers(0, 480))
    s = int(rng.integers(4, 29))
    image[y : y + s, x : x + s] = int(rng.integers(70, 256))

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


def _run_in_fresh_process() -> str:
    result = subprocess.run(
        [sys.executable, "-c", _EXTRACT_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout.strip().splitlines()[-1]


@requires_cusift
def test_cusift_extraction_identical_across_processes():
    outputs = [_run_in_fresh_process() for _ in range(NUM_PROCESSES)]
    num_keypoints = int(outputs[0].split()[0])
    # Guard against a vacuous pass on an image with almost no features.
    assert num_keypoints >= 200
    assert len(set(outputs)) == 1, outputs
