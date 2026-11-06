# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import pycolmap

from . import panorama
from .panorama import (
    PANO_RENDER_OPTIONS,
    PanoProcessor,
    PanoramaReconstructionOptions,
    PanoRenderOptions,
    PanoRenderType,
    reconstruct,
    render_perspective_images,
)


def write_image(path: Path, image: npt.NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert pycolmap.Bitmap.from_array(image).write(path)


def read_image(path: Path, as_rgb: bool = False) -> npt.NDArray[np.uint8]:
    bitmap = pycolmap.Bitmap.read(path, as_rgb=as_rgb)
    assert bitmap is not None
    return bitmap.to_array()


@pytest.fixture
def pano_dir(tmp_path: Path) -> Path:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL.Image")
    image_dir = tmp_path / "panos"
    image = np.random.default_rng(4).integers(
        0, 256, (64, 128, 3), dtype=np.uint8
    )
    write_image(image_dir / "nested/frame.jpg", image)
    return image_dir


@pytest.mark.parametrize(
    "render_options",
    [
        *PANO_RENDER_OPTIONS.values(),
        PanoRenderOptions(
            num_steps_yaw=4,
            pitches_deg=(-90.0, 90.0),
            hfov_deg=90.0,
            vfov_deg=90.0,
        ),
    ],
)
@pytest.mark.parametrize("mask_mode", ["image", "camera", "both", "white"])
def test_project_input_masks(
    tmp_path: Path,
    pano_dir: Path,
    render_options: PanoRenderOptions,
    mask_mode: str,
) -> None:
    baseline = PanoProcessor(
        pano_dir, tmp_path / "images", tmp_path / "baseline", render_options
    )
    baseline.process("nested/frame.jpg")
    # Exercise longitude wrapping, the polar regions, and nonzero grayscale
    # values. Interpolating a binary mask linearly would leak invalid pixels.
    image_mask = np.ones((64, 128), dtype=np.uint8)
    image_mask[:, :11] = 0
    image_mask[:, -7:] = 0
    image_mask[:13] = 0
    camera_mask = np.full((64, 128), 255, dtype=np.uint8)
    camera_mask[42:] = 0
    camera_mask[:, 47:69] = 0
    if mask_mode == "white":
        image_mask[:] = 255
    write_image(tmp_path / "input_masks/nested/frame.jpg.png", image_mask)
    write_image(tmp_path / "camera.png", camera_mask)
    use_image = mask_mode in ("image", "both", "white")
    use_camera = mask_mode in ("camera", "both")
    processor = PanoProcessor(
        pano_dir,
        tmp_path / "masked_images",
        tmp_path / "masked",
        render_options,
        input_mask_path=tmp_path / "input_masks" if use_image else None,
        input_camera_mask_path=tmp_path / "camera.png" if use_camera else None,
    )
    processor.process("nested/frame.jpg")
    valid = np.ones((64, 128), dtype=bool)
    if use_image:
        valid &= image_mask != 0
    if use_camera:
        valid &= camera_mask != 0

    for idx, rotation in enumerate(processor.cams_from_pano_rotation):
        image_name = f"pano_camera{idx}/nested/frame.jpg"
        mask_name = image_name + ".png"
        old_mask = read_image(tmp_path / "baseline" / mask_name)
        actual = read_image(tmp_path / "masked" / mask_name)
        height, width = actual.shape
        # Independent pinhole -> longitude/latitude oracle in pixel-center
        # coordinates, including periodic longitude and clamped latitude.
        yy, xx = np.mgrid[:height, :width]
        focal = width / (2 * np.tan(np.deg2rad(render_options.hfov_deg / 2)))
        rays = (
            np.stack(
                [
                    (xx + 0.5 - width / 2) / focal,
                    (yy + 0.5 - height / 2) / focal,
                    np.ones_like(xx),
                ],
                axis=-1,
            )
            @ rotation
        )
        longitude = np.arctan2(rays[..., 0], rays[..., 2])
        latitude = np.arcsin(rays[..., 1] / np.linalg.norm(rays, axis=-1))
        u = np.rint((longitude / (2 * np.pi) + 0.5) * 128 - 0.5)
        v = np.rint((latitude / np.pi + 0.5) * 64 - 0.5)
        expected = (
            old_mask * valid[np.clip(v.astype(int), 0, 63), u.astype(int) % 128]
        )
        np.testing.assert_array_equal(actual, expected)
        assert set(np.unique(actual)) <= {0, 255}
        np.testing.assert_array_equal(
            read_image(tmp_path / "masked_images" / image_name, as_rgb=True),
            read_image(tmp_path / "images" / image_name, as_rgb=True),
        )


@pytest.mark.parametrize("filename", ["frame.jpg.png", "frame.png"])
def test_input_mask_filename_conventions(
    tmp_path: Path, pano_dir: Path, filename: str
) -> None:
    write_image(
        tmp_path / "input_masks/nested" / filename,
        np.zeros((64, 128), dtype=np.uint8),
    )
    if filename == "frame.jpg.png":
        # Appended-extension masks take precedence when both names exist.
        write_image(
            tmp_path / "input_masks/nested/frame.png",
            np.full((64, 128), 255, dtype=np.uint8),
        )
    processor = PanoProcessor(
        pano_dir,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
        input_mask_path=tmp_path / "input_masks",
    )
    processor.process("nested/frame.jpg")
    masks = list((tmp_path / "masks").rglob("*.png"))
    assert len(masks) == 4
    assert all(not read_image(path).any() for path in masks)


def test_mask_latitude_does_not_wrap_at_poles(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL.Image")
    # Odd rendered dimensions put a pixel center directly on each pole.
    image = np.full((66, 132, 3), 128, dtype=np.uint8)
    write_image(tmp_path / "panos/frame.png", image)
    mask = np.zeros((66, 132), dtype=np.uint8)
    mask[33:] = 255
    write_image(tmp_path / "camera.png", mask)
    processor = PanoProcessor(
        tmp_path / "panos",
        tmp_path / "images",
        tmp_path / "masks",
        PanoRenderOptions(
            num_steps_yaw=1,
            pitches_deg=(-90.0, 90.0),
            hfov_deg=90.0,
            vfov_deg=90.0,
        ),
        input_camera_mask_path=tmp_path / "camera.png",
    )
    processor.process("frame.png")
    for idx, expected in enumerate([255, 0]):
        rendered = read_image(
            tmp_path / "masks" / f"pano_camera{idx}/frame.png.png"
        )
        assert rendered.shape == (33, 33)
        assert rendered[16, 16] == expected


def test_input_masks_preserve_gps(tmp_path: Path, pano_dir: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    exif_tags = pytest.importorskip("PIL.ExifTags")
    gps = {1: "N", 2: (51.0, 30.0, 0.0), 3: "W", 4: (0.0, 7.0, 0.0)}
    exif = image_module.Exif()
    exif[exif_tags.IFD.GPSInfo] = gps
    image = read_image(pano_dir / "nested/frame.jpg", as_rgb=True)
    image_module.fromarray(image).save(pano_dir / "nested/frame.jpg", exif=exif)
    write_image(tmp_path / "camera.png", np.full((64, 128), 255, np.uint8))
    processor = PanoProcessor(
        pano_dir,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
        input_camera_mask_path=tmp_path / "camera.png",
    )
    processor.process("nested/frame.jpg")
    for path in (tmp_path / "images").rglob("*.jpg"):
        with image_module.open(path) as rendered:
            assert rendered.getexif().get_ifd(exif_tags.IFD.GPSInfo) == gps


@pytest.mark.parametrize("mask_kind", ["image", "camera"])
@pytest.mark.parametrize("error", ["missing", "invalid", "dimensions"])
def test_invalid_input_mask(
    tmp_path: Path, pano_dir: Path, mask_kind: str, error: str
) -> None:
    mask_path = tmp_path / "input_masks/nested/frame.jpg.png"
    if error == "invalid":
        mask_path.parent.mkdir(parents=True)
        mask_path.write_text("not an image")
    elif error == "dimensions":
        write_image(mask_path, np.ones((32, 64), dtype=np.uint8))
    processor = PanoProcessor(
        pano_dir,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
        input_mask_path=tmp_path / "input_masks"
        if mask_kind == "image"
        else None,
        input_camera_mask_path=mask_path if mask_kind == "camera" else None,
    )
    exception = ValueError if error == "dimensions" else OSError
    with pytest.raises(exception, match="[Mm]ask"):
        processor.process("nested/frame.jpg")
    assert not list((tmp_path / "images").rglob("*.jpg"))


def test_shared_mask_not_modified_by_per_image_masks(
    tmp_path: Path, pano_dir: Path
) -> None:
    names = ["nested/frame.jpg", "other/frame.jpg"]
    write_image(pano_dir / names[1], read_image(pano_dir / names[0], True))
    for name, value in zip(names, [0, 255], strict=True):
        write_image(
            tmp_path / "input_masks" / f"{name}.png",
            np.full((64, 128), value, dtype=np.uint8),
        )
    write_image(tmp_path / "camera.png", np.full((64, 128), 255, np.uint8))
    render_perspective_images(
        names,
        pano_dir,
        tmp_path / "images",
        tmp_path / "masks",
        PANO_RENDER_OPTIONS[PanoRenderType.PERSPECTIVE_NON_OVERLAPPING],
        show_progress=False,
        input_mask_path=tmp_path / "input_masks",
        input_camera_mask_path=tmp_path / "camera.png",
    )
    for idx in range(4):
        for name, value in zip(names, [0, 255], strict=True):
            mask = read_image(
                tmp_path / "masks" / f"pano_camera{idx}/{name}.png"
            )
            assert np.all(mask == value)


@pytest.mark.parametrize("render_type", list(PanoRenderType))
def test_reconstruction_masks_filter_sift_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, render_type: PanoRenderType
) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL.Image")
    # Real rendering, native CPU SIFT and database IO. Matching and mapping are
    # unrelated to whether the masks reach the feature extractor.
    monkeypatch.setattr(panorama, "run_matcher", lambda *args: None)
    monkeypatch.setattr(
        panorama.pycolmap, "incremental_mapping", lambda *args: {}
    )
    image = np.random.default_rng(12).integers(
        0, 256, (128, 256, 3), dtype=np.uint8
    )
    image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
    write_image(tmp_path / "panos/frame.png", image)
    image_mask = np.full((512, 1024), 255, dtype=np.uint8)
    image_mask[:, :256] = 0
    camera_mask = np.full_like(image_mask, 255)
    camera_mask[256:] = 0
    write_image(tmp_path / "input_masks/frame.png.png", image_mask)
    write_image(tmp_path / "camera.png", camera_mask)

    for masked in (False, True):
        reconstruct(
            tmp_path / "panos",
            tmp_path / ("masked" if masked else "baseline"),
            PanoramaReconstructionOptions(
                render_type=render_type,
                num_threads=1,
                use_gpu=False,
                show_progress=False,
                input_mask_path=tmp_path / "input_masks" if masked else None,
                input_camera_mask_path=tmp_path / "camera.png"
                if masked
                else None,
            ),
        )

    num_removed = num_kept = 0
    with (
        pycolmap.Database.open(tmp_path / "baseline/database.db") as baseline,
        pycolmap.Database.open(tmp_path / "masked/database.db") as masked,
    ):
        masked_images = {
            image.name: image for image in masked.read_all_images()
        }
        for image in baseline.read_all_images():
            keypoints = baseline.read_keypoints(image.image_id)
            actual = masked.read_keypoints(masked_images[image.name].image_id)
            if render_type == PanoRenderType.SPHERICAL:
                mask = image_mask & camera_mask
            else:
                mask = read_image(
                    tmp_path / "masked/masks" / f"{image.name}.png"
                )
            xy = keypoints[:, :2].astype(int)
            keep = mask[xy[:, 1], xy[:, 0]] != 0
            np.testing.assert_array_equal(actual, keypoints[keep])
            np.testing.assert_array_equal(
                masked.read_descriptors(
                    masked_images[image.name].image_id
                ).data,
                baseline.read_descriptors(image.image_id).data[keep],
            )
            num_removed += int(np.count_nonzero(~keep))
            num_kept += len(actual)
    assert num_removed > 100
    assert num_kept > 100
