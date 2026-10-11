# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest

import pycolmap

from .panorama import (
    PANO_RENDER_OPTIONS,
    PanoProcessor,
    PanoramaReconstructionOptions,
    PanoRenderType,
    filter_database_by_covisibility,
    get_virtual_rotations,
    run_perspective,
)


@pytest.mark.parametrize("mode", ["P", "PA", "CMYK", "RGB", "RGBA", "L", "LA"])
def test_process_preserves_image_colors(tmp_path: Path, mode: str) -> None:
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (128, 64), (32, 96, 160))
    if mode == "P":
        image = image.quantize(colors=1)
    elif mode == "PA":
        # No dithering, so the solid fixture stays uniform.
        image = image.convert("PA", dither=image_module.Dither.NONE)
    else:
        image = image.convert(mode)
    # PNG cannot store CMYK; TIFF round-trips palette+alpha (PA).
    pano_suffix = {"CMYK": "jpg", "PA": "tiff"}.get(mode, "png")
    pano_name = f"pano.{pano_suffix}"
    image.save(tmp_path / pano_name)
    with image_module.open(tmp_path / pano_name) as source:
        expected_color = np.asarray(source.convert("RGB"))[0, 0]

    # JPEG compression is lossy; the CMYK fixture must round-trip through
    # JPEG (PNG cannot store CMYK), so it needs a wider tolerance.
    atol = 8 if mode == "CMYK" else 1

    processor = PanoProcessor(
        tmp_path,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
    )
    processor.process(pano_name)

    for cam_idx in range(4):
        image_path = tmp_path / "images" / f"pano_camera{cam_idx}" / pano_name
        with image_module.open(image_path) as rendered:
            pixels = np.asarray(rendered.convert("RGB"))
            np.testing.assert_allclose(
                pixels,
                np.broadcast_to(expected_color, pixels.shape),
                atol=atol,
                rtol=0,
            )


def test_process_preserves_palette_transparency(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (128, 64), (32, 96, 160)).quantize(colors=1)
    image.save(tmp_path / "pano.png", transparency=image.getpixel((0, 0)))
    processor = PanoProcessor(
        tmp_path,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
    )
    processor.process("pano.png")

    for cam_idx in range(4):
        image_path = tmp_path / "images" / f"pano_camera{cam_idx}/pano.png"
        with image_module.open(image_path) as rendered:
            assert rendered.mode == "RGBA"
            assert not np.asarray(rendered)[..., 3].any()


@pytest.mark.parametrize(
    ("mode", "color", "expected_mode"),
    [("RGBA", (32, 96, 160, 200), "RGB"), ("LA", (128, 200), "L")],
)
def test_process_drops_alpha_for_jpeg_output(
    tmp_path: Path, mode: str, color: tuple[int, ...], expected_mode: str
) -> None:
    pytest.importorskip("cv2")
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new(mode, (128, 64), color)
    # PNG-encoded bytes under a .jpg name, as with misnamed uploads. JPEG
    # output cannot store alpha; it must be dropped instead of raising
    # `OSError: cannot write mode RGBA as JPEG`.
    image.save(tmp_path / "pano.jpg", format="PNG")
    processor = PanoProcessor(
        tmp_path,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
    )
    processor.process("pano.jpg")

    for cam_idx in range(4):
        image_path = tmp_path / "images" / f"pano_camera{cam_idx}/pano.jpg"
        with image_module.open(image_path) as rendered:
            assert rendered.mode == expected_mode
            pixels = np.asarray(rendered)
            # JPEG output is lossy; the semi-transparent alpha is dropped
            # rather than composited, so only the color channels remain.
            np.testing.assert_allclose(
                pixels,
                np.broadcast_to(np.asarray(color[:-1]), pixels.shape),
                atol=8,
                rtol=0,
            )


def test_get_virtual_rotations() -> None:
    rotations = get_virtual_rotations(4, [-35.0, 0.0, 35.0])

    assert len(rotations) == 12
    np.testing.assert_allclose(
        rotations[4],
        np.eye(3),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        rotations[1],
        [
            [0.0, 0.0, -1.0],
            [-0.573576436351046, 0.8191520442889917, 0.0],
            [0.8191520442889917, 0.573576436351046, 0.0],
        ],
        atol=1e-15,
    )
    for rotation in rotations:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-15)


def test_filter_database_by_covisibility(tmp_path: Path) -> None:
    database_path = tmp_path / "database.db"
    with pycolmap.Database.open(database_path) as database:
        camera = pycolmap.Camera.create_from_model_name(
            1, "SIMPLE_PINHOLE", 100.0, 100, 100
        )
        database.write_camera(camera)
        for image_id, name in enumerate(["a.png", "b.png", "c.png"], start=1):
            database.write_image(
                pycolmap.Image(
                    image_id=image_id, camera_id=camera.camera_id, name=name
                ),
                use_image_id=True,
            )
        geometry = pycolmap.TwoViewGeometry()
        geometry.inlier_matches = np.array([[0, 0]], dtype=np.uint32)
        database.write_two_view_geometry(1, 2, geometry)
        database.write_two_view_geometry(1, 3, geometry)

    covisibility_path = tmp_path / "covisibility.npz"
    np.savez(
        covisibility_path,
        image_names=np.array(["a.png", "b.png", "c.png"]),
        directed_overlap_counts=np.array(
            [[1, 10, 0], [10, 1, 0], [0, 0, 1]], dtype=np.uint32
        ),
    )
    filter_database_by_covisibility(
        database_path, covisibility_path, min_shared_points=1
    )

    with pycolmap.Database.open(database_path) as database:
        assert database.exists_two_view_geometry(1, 2)
        assert not database.exists_two_view_geometry(1, 3)


def test_run_perspective_raises_for_empty_input_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "panoramas"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="No panorama could be rendered"):
        run_perspective(
            input_dir,
            output_dir,
            PanoramaReconstructionOptions(show_progress=False),
            output_dir / "database.db",
            output_dir / "sparse",
        )
