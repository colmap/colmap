# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np

import pycolmap

from .imc import DatasetIMC2024


def _add_image(
    reconstruction: pycolmap.Reconstruction,
    image_id: int,
    name: str,
    cam_from_world: pycolmap.Rigid3d,
) -> None:
    camera = pycolmap.Camera(
        camera_id=image_id,
        model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        width=640,
        height=480,
        params=[500, 320, 240],
    )
    reconstruction.add_camera_with_trivial_rig(camera)
    image = pycolmap.Image(image_id=image_id, camera_id=image_id, name=name)
    reconstruction.add_image_with_trivial_frame(image, cam_from_world)


def _format_label(
    name: str, scene: str, cam_from_world: pycolmap.Rigid3d, scale: float
) -> str:
    rotation = ";".join(map(str, cam_from_world.rotation.matrix().ravel()))
    translation = ";".join(map(str, scale * cam_from_world.translation))
    return f"{name},{rotation},{translation},,{scene},{scene}\n"


def test_imc2024_matches_renamed_images_by_pose(tmp_path: Path) -> None:
    # The released images and labels use different names than sfm/, and the
    # labels' translations are scaled.
    scene = "scene"
    scene_path = tmp_path / "imc2024" / "train" / "all" / scene
    (scene_path / "sfm").mkdir(parents=True)
    (scene_path / "images").mkdir()

    rng = np.random.default_rng(42)
    cams_from_world = [
        pycolmap.Rigid3d(
            pycolmap.Rotation3d(rng.normal(size=3)), rng.normal(size=3)
        )
        for _ in range(5)
    ]
    sfm_names = ["a.jpg", "b.JPG", "ignored.jpg", "c'1.jpg", "c_1.jpg"]
    # Two original names map to the same released name, which is ambiguous.
    released_names: list[str | None] = [
        "a.png",
        "b.png",
        None,
        "c_1.png",
        "c_1.png",
    ]

    sparse_sfm = pycolmap.Reconstruction()
    for image_id, (name, cam_from_world) in enumerate(
        zip(sfm_names, cams_from_world, strict=True), start=1
    ):
        _add_image(sparse_sfm, image_id, name, cam_from_world)
    sparse_sfm.write_text(scene_path / "sfm")

    labels = "image_name,rotation_matrix,translation_vector,calibration_matrix,"
    labels += "dataset,scene\n"
    for released_name, cam_from_world in zip(
        released_names, cams_from_world, strict=True
    ):
        if released_name is not None:
            (scene_path / "images" / released_name).touch()
            labels += _format_label(
                released_name, scene, cam_from_world, scale=10
            )
    (scene_path.parent / "train_labels.csv").write_text(labels)

    dataset = DatasetIMC2024(
        data_path=tmp_path,
        categories=[],
        scenes=[],
        run_path=tmp_path / "runs",
        run_name="test",
    )
    (scene_info,) = dataset.list_scenes()
    # A stale GT reconstruction without images is regenerated.
    scene_info.sparse_gt_path.mkdir()
    pycolmap.Reconstruction().write(scene_info.sparse_gt_path)
    dataset.prepare_scene(scene_info)

    sparse_gt = pycolmap.Reconstruction(scene_info.sparse_gt_path)
    gt_cams_from_world = {
        image.name: image.cam_from_world().matrix()
        for image in sparse_gt.images.values()
    }
    assert sorted(gt_cams_from_world) == ["a.png", "b.png"]
    np.testing.assert_allclose(
        gt_cams_from_world["a.png"], cams_from_world[0].matrix()
    )
    np.testing.assert_allclose(
        gt_cams_from_world["b.png"], cams_from_world[1].matrix()
    )
