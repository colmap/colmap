// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_io_binary.h"
#include "colmap/scene/reconstruction_io_text.h"

#include <filesystem>

#include <Eigen/Core>

namespace colmap {

// Exports in NVM format http://ccwu.me/vsfm/doc.html#nvm. Only supports
// SIMPLE_RADIAL camera model when exporting distortion parameters. When
// skip_distortion == true it supports all camera models with the caveat that
// it's using the mean focal length which will be inaccurate for camera models
// with two focal lengths and distortion.
bool ExportNVM(const Reconstruction& reconstruction,
               const std::filesystem::path& path,
               bool skip_distortion = false);

// Exports in CAM format which is a simple text file that contains pose
// information and camera intrinsics for each image and exports one file per
// image; it does not include information on the 3D points. The format is as
// follows (2 lines of text with space separated numbers):
// <Tvec; 3 values> <Rotation matrix in row-major format; 9 values>
// <focal_length> <k1> <k2> 1.0 <principal point X> <principal point Y>
// Note that focal length is relative to the image max(width, height),
// and principal points x and y are relative to width and height respectively.
//
// Only supports SIMPLE_RADIAL and RADIAL camera models when exporting
// distortion parameters. When skip_distortion == true it supports all camera
// models with the caveat that it's using the mean focal length which will be
// inaccurate for camera models with two focal lengths and distortion.
bool ExportCam(const Reconstruction& reconstruction,
               const std::filesystem::path& path,
               bool skip_distortion = false);

// Exports in Recon3D format which consists of three text files with the
// following format and content:
// 1) imagemap_0.txt: a list of image numeric IDs with one entry per line.
// 2) urd-images.txt: A list of images with one entry per line as:
//    <image file name> <width> <height>
// 3) synth_0.out: Contains information for image poses, camera intrinsics,
//    and 3D points as:
//    <N; num images> <M; num points>
//    <N lines of image entries>
//    <M lines of point entries>
//
//    Each image entry consists of 5 lines as:
//    <focal length> <k1> <k2>
//    <Rotation matrix; 3x3 array>
//    <Tvec; 3 values>
//    Note that the focal length is scaled by 1 / max(width, height)
//
//    Each point entry consists of 3 lines as:
//    <point x, y, z coordinates>
//    <point RGB color>
//    <K; num track elements> <Track Element 1> ... <Track Element K>
//
//    Each track elemenet is a sequence of 5 values as:
//    <image ID> <2D point ID> -1.0 <X> <Y>
//    Note that the 2D point coordinates are centered around the principal
//    point and scaled by 1 / max(width, height).
//
// When skip_distortion == true it supports all camera models with the
// caveat that it's using the mean focal length which will be inaccurate
// for camera models with two focal lengths and distortion.
bool ExportRecon3D(const Reconstruction& reconstruction,
                   const std::filesystem::path& path,
                   bool skip_distortion = false);

// Exports in Bundler format https://www.cs.cornell.edu/~snavely/bundler/.
// Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL and RADIAL camera models
// when exporting distortion parameters. When skip_distortion == true it
// supports all camera models with the caveat that it's using the mean focal
// length which will be inaccurate for camera models with two focal lengths
// and distortion.
bool ExportBundler(const Reconstruction& reconstruction,
                   const std::filesystem::path& path,
                   const std::filesystem::path& list_path,
                   bool skip_distortion = false);

// Exports 3D points only in PLY format.
void ExportPLY(const Reconstruction& reconstruction,
               const std::filesystem::path& path);

// Exports in VRML format https://en.wikipedia.org/wiki/VRML.
void ExportVRML(const Reconstruction& reconstruction,
                const std::filesystem::path& images_path,
                const std::filesystem::path& points3D_path,
                double image_scale,
                const Eigen::Vector3d& image_rgb);

}  // namespace colmap
