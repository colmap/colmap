// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <functional>

namespace colmap {

// This enum can be used as optional input for feature_extractor and
// feature_importer to ensure that the camera flags of ImageReader are set in an
// exclusive and unambiguous way. The table below explains the correspondence of
// each setting with the flags
//
// clang-format off
// -----------------------------------------------------------------------------------
// |            |                         ImageReaderOptions                         |
// | CameraMode | single_camera | single_camera_per_folder | single_camera_per_image |
// |------------|---------------|--------------------------|-------------------------|
// | AUTO       | false         | false                    | false                   |
// | SINGLE     | true          | false                    | false                   |
// | PER_FOLDER | false         | true                     | false                   |
// | PER_IMAGE  | false         | false                    | true                    |
// -----------------------------------------------------------------------------------
// clang-format on
//
// Note: When using AUTO mode a camera model will be uniquely identified by the
// following 5 parameters from EXIF tags:
// 1. Camera Make
// 2. Camera Model
// 3. Focal Length
// 4. Image Width
// 5. Image Height
//
// If any of the tags is missing then a camera model is considered invalid and a
// new camera is created similar to the PER_IMAGE mode.
//
// If these considered fields are not sufficient to uniquely identify a camera
// then using the AUTO mode will lead to incorrect setup for the cameras, e.g.
// the same camera is used with same focal length but different principal point
// between captures. In these cases it is recommended to either use the
// PER_FOLDER or PER_IMAGE settings.
enum class CameraMode { AUTO = 0, SINGLE = 1, PER_FOLDER = 2, PER_IMAGE = 3 };

void UpdateImageReaderOptionsFromCameraMode(ImageReaderOptions& options,
                                            CameraMode mode);

bool VerifySiftGPUParams(bool use_gpu);

bool VerifyCameraParams(const std::string& camera_model,
                        const std::string& params);

void RunGuidedGeometricVerifierImpl(
    const Reconstruction& reconstruction,
    const std::filesystem::path& database_path,
    const ExistingMatchedPairingOptions& pairing_options,
    const TwoViewGeometryOptions& geometry_options,
    int num_threads,
    std::function<bool()> check_if_stopped = {});

int RunFeatureExtractor(int argc, char** argv);
int RunFeatureImporter(int argc, char** argv);
int RunExhaustiveMatcher(int argc, char** argv);
int RunMatchesImporter(int argc, char** argv);
int RunSequentialMatcher(int argc, char** argv);
int RunSpatialMatcher(int argc, char** argv);
int RunTransitiveMatcher(int argc, char** argv);
int RunVocabTreeMatcher(int argc, char** argv);
int RunGeometricVerifier(int argc, char** argv);
int RunGuidedGeometricVerifier(int argc, char** argv);

}  // namespace colmap
