# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse

import numpy as np
import numpy.typing as npt
import open3d

import pycolmap


class Model:
    def __init__(self) -> None:
        self.reconstruction: pycolmap.Reconstruction
        self.visualizer: open3d.visualization.Visualizer

    def read_model(self, path: str) -> None:
        self.reconstruction = pycolmap.Reconstruction(path)

    def add_points(
        self, min_track_len: int = 3, remove_statistical_outlier: bool = True
    ) -> None:
        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.reconstruction.points3D.values():
            if point.track.length() < min_track_len:
                continue
            xyz.append(point.xyz)
            rgb.append(point.color / 255)

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

        # open3d.visualization.draw_geometries([pcd])
        self.visualizer.add_geometry(pcd)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def add_cameras(self, scale: float = 1) -> None:
        frustums = []
        for img in self.reconstruction.images.values():
            # extrinsics
            world_from_cam = img.cam_from_world().inverse()
            R = world_from_cam.rotation.matrix()
            t = world_from_cam.translation

            # intrinsics
            cam = img.camera
            if cam.model in (
                pycolmap.CameraModelId.SIMPLE_PINHOLE,
                pycolmap.CameraModelId.SIMPLE_RADIAL,
                pycolmap.CameraModelId.RADIAL,
            ):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in (
                pycolmap.CameraModelId.PINHOLE,
                pycolmap.CameraModelId.OPENCV,
                pycolmap.CameraModelId.OPENCV_FISHEYE,
                pycolmap.CameraModelId.FULL_OPENCV,
            ):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception("Camera model not supported")

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramid geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
            frustums.extend(cam_model)

        # add geometries to visualizer
        for i in frustums:
            self.visualizer.add_geometry(i)

    def create_window(self) -> None:
        self.visualizer = open3d.visualization.Visualizer()
        self.visualizer.create_window()

    def show(self) -> None:
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        self.visualizer.run()
        self.visualizer.destroy_window()


def draw_camera(
    K: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    w: int,
    h: int,
    scale: float = 1,
    color: list[float] = None,
) -> list[open3d.geometry.Geometry]:
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    if color is None:
        color = [0.8, 0.2, 0.8]

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale
    )
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize COLMAP binary and text models"
    )
    parser.add_argument(
        "--input_model", required=True, help="path to input model folder"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # read COLMAP model
    model = Model()
    model.read_model(args.input_model)

    print("num_cameras:", model.reconstruction.num_cameras())
    print("num_images:", model.reconstruction.num_images())
    print("num_points3D:", model.reconstruction.num_points3D())

    # display using Open3D visualization tools
    model.create_window()
    model.add_points()
    model.add_cameras(scale=0.25)
    model.show()


if __name__ == "__main__":
    main()
