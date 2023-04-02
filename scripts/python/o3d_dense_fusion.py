import os
import argparse

import numpy as np

from skimage.io import imsave

import open3d as o3d
from open3d.io import read_image, write_triangle_mesh
from open3d.camera import (
    PinholeCameraIntrinsic,
    PinholeCameraParameters,
    PinholeCameraTrajectory,
)
from open3d.geometry import RGBDImage
from open3d.pipelines import integration, color_map

from read_write_model import read_model
from read_write_dense import read_array


# COLMAP models are normalized to [-10, 10] in x/y/z. Open3D expects depth maps
# discretized to 16 bits.
DEPTH_SCALE = 1000


def convert_depth_maps(dense_path, dense_type, images):
    for i, image in enumerate(images.values()):
        depth_map_path = os.path.join(dense_path, "stereo/depth_maps", image.name)
        orig_path = f"{depth_map_path}.{dense_type}.bin"
        converted_path = f"{depth_map_path}.{dense_type}.bin.png"
        if os.path.exists(converted_path):
            continue
        print(f"Converting depth map for image: {image.name} [{i+1}/{len(images)}]")
        depth_map = np.round(read_array(orig_path) * DEPTH_SCALE).astype(np.uint16)
        depth_map[depth_map == 0] = np.iinfo(np.uint16).max
        imsave(converted_path, depth_map)


def convert_to_o3d_camera_params(image, camera):
    assert camera.model == "PINHOLE"
    params = PinholeCameraParameters()
    params.intrinsic = PinholeCameraIntrinsic(
        width=camera.width,
        height=camera.height,
        fx=camera.params[0],
        fy=camera.params[1],
        cx=camera.params[2],
        cy=camera.params[3],
    )
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = image.qvec2rotmat()
    extrinsic[:3, 3] = image.tvec
    params.extrinsic = extrinsic
    return params


def run_o3d_texture_mapping(mesh, rgbd_images, camera_trajectory, max_depth):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        option = color_map.NonRigidOptimizerOption(
            maximum_iteration=100, maximum_allowable_depth=max_depth,
        )
        optimized_mesh, _ = color_map.run_non_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory, option
        )
    return optimized_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_path", required=True)
    parser.add_argument(
        "--dense_type", default="photometric", choices=["photometric", "geometric"]
    )
    parser.add_argument("--voxel_length", type=float, default=0.03)
    parser.add_argument("--sdf_trunc", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=np.sqrt(10**3))
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    cameras, images, _ = read_model(path=os.path.join(args.dense_path, "sparse"))
    print(f"Loaded COLMAP dense model with {len(images)} images")

    convert_depth_maps(args.dense_path, args.dense_type, images)

    tsdf = integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=1,
    )
    camera_params_list = []
    rgbd_images = []
    for i, image in enumerate(images.values()):
        print(f"Integrating {image.name} into TSDF volume [{i+1}/{len(images)}]")
        rgb = read_image(os.path.join(args.dense_path, "images", image.name))
        depth = read_image(
            os.path.join(
                args.dense_path,
                "stereo/depth_maps",
                f"{image.name}.{args.dense_type}.bin.png",
            )
        )
        rgbd_image = RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )
        camera_params = convert_to_o3d_camera_params(image, cameras[image.camera_id])

        tsdf.integrate(
            rgbd_image,
            camera_params.intrinsic,
            camera_params.extrinsic,
        )

        camera_params_list.append(camera_params)
        rgbd_images.append(rgbd_image)

        if args.debug:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, camera_params.intrinsic
            )
            o3d.visualization.draw_geometries([pcd])

    camera_trajectory = PinholeCameraTrajectory()
    camera_trajectory.parameters = camera_params_list

    print("Extracting mesh")
    mesh = tsdf.extract_triangle_mesh()
    write_triangle_mesh(os.path.join(args.dense_path, "o3d.tsdf.ply"), mesh)

    print("Running texture mapping")
    print(camera_trajectory.parameters)
    optimized_mesh = run_o3d_texture_mapping(
        mesh, rgbd_images, camera_trajectory, args.max_depth
    )
    write_triangle_mesh(
        os.path.join(args.dense_path, "o3d.optimized.ply"), optimized_mesh
    )


if __name__ == "__main__":
    main()
