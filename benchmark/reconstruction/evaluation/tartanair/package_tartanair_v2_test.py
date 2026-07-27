import io

import numpy as np
import PIL.Image

from .package_tartanair_v2 import (
    DEPTH_SCALE,
    DEPTH_SIZE,
    encode_depth_png,
    encode_image_jpeg,
)


def test_encode_depth_png_min_pools_and_quantizes():
    depth = np.full((1024, 2048), 10.0, dtype="<f4")
    depth[:4, :4] = 2.0
    rgba = depth.view(np.uint8).reshape(1024, 2048, 4)
    source = io.BytesIO()
    PIL.Image.fromarray(rgba, mode="RGBA").save(source, format="PNG")

    decoded = np.asarray(
        PIL.Image.open(io.BytesIO(encode_depth_png(source.getvalue())))
    )
    assert decoded.shape == (DEPTH_SIZE[1], DEPTH_SIZE[0])
    assert decoded.dtype == np.uint16
    assert decoded[0, 0] == round(2.0 * DEPTH_SCALE)
    assert decoded[0, 1] == round(10.0 * DEPTH_SCALE)


def test_encode_image_jpeg():
    source = io.BytesIO()
    PIL.Image.new("RGB", (64, 32), (10, 20, 30)).save(source, format="PNG")
    encoded = encode_image_jpeg(source.getvalue())
    image = PIL.Image.open(io.BytesIO(encoded))
    assert image.format == "JPEG"
    assert image.size == (64, 32)
