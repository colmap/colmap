// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/single_view_calibrator.h"
#include "colmap/util/threading.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

// Factory for the calibrator backend, injectable for testing. Defaults to
// `SingleViewCalibrator::Create` when empty.
using SingleViewCalibratorFactory =
    std::function<std::unique_ptr<SingleViewCalibrator>(
        const SingleViewCalibrationOptions& options)>;

// Calibrate the selected images in the database with a learned single-view
// calibrator and update the database cameras, replacing e.g. EXIF-based
// initialization. An empty `image_names` selects all database images. Images
// are processed sequentially with a single shared calibrator instance, as
// calibration models are large.
//
// The worker thread never throws: database inconsistencies and calibrator
// failures are logged and either skip the affected image or fail the stage
// without updating any camera.
std::unique_ptr<Thread> CreateSingleViewCalibrationController(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const SingleViewCalibrationOptions& calibration_options,
    const std::vector<std::string>& image_names = {},
    SingleViewCalibratorFactory calibrator_factory = {});

}  // namespace colmap
