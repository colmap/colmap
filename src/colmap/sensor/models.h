// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Umbrella header for the camera models. The implementation lives in
// `sensor/models/`, split by concern and model family:
//
//   base.h            Model id enum, declaration/registration macros, CRTP
//                     base classes shared by all models.
//   pinhole.h, radial.h, opencv.h, fov.h, division.h, eucm.h,
//   fisheye.h, fisheye_radial.h, thin_prism.h, spherical.h
//                     Per-family model structs and their implementations.
//   runtime.h         Runtime dispatch over CameraModelId (the CameraModel*
//                     free functions). Aggregates the family headers above.
//   jacobian.h        Analytic ImgFromCamWithJac kernels for all models.
//
// Include this umbrella to get everything, or include individual
// `sensor/models/*.h` headers (each is self-contained) to reduce compile time.
// A new family header must be added to the include list in `runtime.h`.
//
// New camera models can be added by the following steps:
//
//  1. Add a new struct in the appropriate `sensor/models/<family>.h` header
//     (or a new family header, included from `runtime.h` as described above),
//     which implements all the necessary methods.
//  2. Define a unique model_name, model_id, and params_info string for the
//     camera model (passed to the model definition macro, which also derives
//     the parameter group index arrays from the group sizes).
//  3. Add camera model to `CAMERA_MODEL_CASES` macro in `sensor/models/base.h`.
//  4. Add new template specialization of test case for camera model to
//     `models_test.cc`.
//  5. Implement the analytic `ImgFromCamWithJac` for the camera model in
//     `sensor/models/jacobian.h` (required by the runtime dispatch).
//
// A camera model can have three different types of camera parameters: focal
// length, principal point, extra parameters (distortion parameters). The
// parameter array is split into different groups, so that we can enable or
// disable the refinement of the individual groups during bundle adjustment. It
// is up to the camera model to access the parameters correctly (it is free to
// do so in an arbitrary manner) - the parameters are not accessed from outside.
// Note that additional metadata parameters (e.g. image width and height) are
// can be stored at the end of the parameter array, but they are not considered
// as optimizable camera parameters.
//
// A camera model must have the following methods:
//
//  - `ImgFromCam`: Projects points in camera frame to pixel coordinates in
//    image plane (the inverse of `CamFromImg`). Assumes that the camera
//    coordinates are given as (u, v, w). Returns false, if projection failed.
//    The `check_cheirality` flag selects whether points behind the camera are
//    rejected, see `HasProjectableDepth`.
//  - `CamFromImg`: lift pixel coordinates in image plane to normalized camera
//    coordinates (the inverse of `ImgFromCam`). Produces camera coordinates
//    as (u, v, 1). Returns false, if lifting failed.
//  - `CamFromImgThreshold`: transform a threshold given in pixels to
//    normalized units (e.g. useful for reprojection error thresholds).
//
// Whenever you specify the camera parameters in a list, they must appear
// exactly in the order as they are accessed in the defined model struct.
//
// The camera models follow the convention that the upper left image corner has
// the coordinate (0, 0), the lower right corner (width, height), i.e. that
// the upper left pixel center has coordinate (0.5, 0.5) and the lower right
// pixel center has the coordinate (width - 0.5, height - 0.5).

#include "colmap/sensor/models/jacobian.h"
#include "colmap/sensor/models/runtime.h"
