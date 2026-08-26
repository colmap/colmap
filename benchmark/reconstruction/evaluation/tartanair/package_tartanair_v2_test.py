import hashlib
import io

import numpy as np
import pytest

PILImage = pytest.importorskip("PIL.Image")

from . import package_tartanair_v2  # noqa: E402
from .package_tartanair_v2 import (  # noqa: E402
    DEPTH_SCALE,
    DEPTH_SIZE,
    encode_depth_png,
    encode_image_jpeg,
)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_download_tartanair_license(monkeypatch):
    license_data = b"license text"
    monkeypatch.setattr(
        package_tartanair_v2,
        "TARTANAIR_LICENSE_SHA256",
        hashlib.sha256(license_data).hexdigest(),
    )
    monkeypatch.setattr(
        package_tartanair_v2.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(license_data),
    )
    assert package_tartanair_v2.download_tartanair_license() == license_data


def test_download_tartanair_license_rejects_unexpected_content(monkeypatch):
    monkeypatch.setattr(
        package_tartanair_v2.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(b"unexpected"),
    )
    with pytest.raises(RuntimeError, match="Unexpected TartanAir license"):
        package_tartanair_v2.download_tartanair_license()


def test_encode_depth_png_min_pools_and_quantizes():
    depth = np.full((1024, 2048), 10.0, dtype="<f4")
    depth[:4, :4] = 2.0
    rgba = depth.view(np.uint8).reshape(1024, 2048, 4)
    source = io.BytesIO()
    PILImage.fromarray(rgba, mode="RGBA").save(source, format="PNG")

    decoded = np.asarray(
        PILImage.open(io.BytesIO(encode_depth_png(source.getvalue())))
    )
    assert decoded.shape == (DEPTH_SIZE[1], DEPTH_SIZE[0])
    assert decoded.dtype == np.uint16
    assert decoded[0, 0] == round(2.0 * DEPTH_SCALE)
    assert decoded[0, 1] == round(10.0 * DEPTH_SCALE)


def test_encode_image_jpeg():
    source = io.BytesIO()
    PILImage.new("RGB", (64, 32), (10, 20, 30)).save(source, format="PNG")
    encoded = encode_image_jpeg(source.getvalue())
    image = PILImage.open(io.BytesIO(encoded))
    assert image.format == "JPEG"
    assert image.size == (64, 32)
