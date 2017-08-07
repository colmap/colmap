# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import collections
import numpy as np
import struct

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
camera_models = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=4, model_name="RADIAL", num_params=5),
    CameraModel(model_id=5, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=6, model_name="OPENCV", num_params=8),
    CameraModel(model_id=7, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=8, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=9, model_name="FOV", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
cam_model_id_to_model = dict([(cam_model.model_id, cam_model) for cam_model in camera_models])


def read_next_bytes(binary_file, num_bytes, format_char_sequence, endian_character='<'):
    """
    :param binary_file:
    :param num_bytes: combination of {2,4,8}
    :param format_char_sequence: sequence of {c,e,f,d,h,H,i,I,l,L,q,Q}
    :param endian_character: {@, =, <, >, !}
    :return: tuple of corresponding values
    """
    binary_data = binary_file.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, binary_data)


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as binary_file:
        num_cameras = read_next_bytes(binary_file, 8, 'Q')[0]
        cameras = {}
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                binary_file, num_bytes=24, format_char_sequence='iiQQ')
            camera_id = camera_properties[0]
            camera_model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = cam_model_id_to_model[camera_model_id].num_params
            params = read_next_bytes(
                binary_file, num_bytes=8*num_params, format_char_sequence='d'*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=cam_model_id_to_model[camera_model_id].model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as binary_file:
        num_reg_images = read_next_bytes(binary_file, 8, 'Q')[0]
        images = {}
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                binary_file,
                num_bytes=64,   # 7*8+2*4
                format_char_sequence='idddddddi')

            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = np.array(binary_image_properties[8])

            # Read the image name
            current_char = read_next_bytes(binary_file, 1, 'c')[0]
            image_name = ''
            while current_char != b'\x00':   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(binary_file, 1, 'c')[0]

            num_points2D = read_next_bytes(
                binary_file, num_bytes=8, format_char_sequence='Q')[0]
            x_y_id_s = read_next_bytes(
                binary_file, num_bytes=24*num_points2D, format_char_sequence='ddq'*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                       tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as binary_file:
        num_points = read_next_bytes(binary_file, 8, 'Q')[0]
        binary_point_lines = {}
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                binary_file, num_bytes=43, format_char_sequence='QdddBBBd')

            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])

            track_length = read_next_bytes(
                binary_file, num_bytes=8, format_char_sequence='Q')[0]
            binary_point_track_pairs = read_next_bytes(
                binary_file, num_bytes=8*track_length, format_char_sequence='ii'*track_length)

            image_ids = np.array(tuple(map(int, binary_point_track_pairs[0::2])))
            point2D_idxs = np.array(tuple(map(int, binary_point_track_pairs[1::2])))

            binary_point_lines[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return binary_point_lines


def read_model(path, ext):
    if ext == '.txt':
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def main():
    if len(sys.argv) != 3:
        print("Usage: python read_model.py path/to/model/folder/ ext")
        return

    cameras, images, points3D = read_model(path=sys.argv[1], ext=sys.argv[2])

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))


if __name__ == '__main__':
    main()
