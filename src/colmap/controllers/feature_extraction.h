// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/calibrator.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/feature/extractor.h"
#include "colmap/util/threading.h"

#include <filesystem>

namespace colmap {

// Reads images from a folder, extracts features, and writes them to database.
// Pose priors are populated from EXIF and the created cameras are calibrated
// with the configured monocular calibration backend (EXIF focal length
// initialization by default), unless explicit camera parameters are provided.
// The calibrator factory is injectable for testing and defaults to
// `MonocularCalibrator::Create` when empty.
std::unique_ptr<Thread> CreateFeatureExtractorController(
    const std::filesystem::path& database_path,
    const ImageReaderOptions& reader_options,
    const FeatureExtractionOptions& extraction_options,
    const MonocularCalibrationOptions& calibration_options,
    MonocularCalibratorFactory calibrator_factory = {});

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix. Pose priors and
// camera intrinsics are handled as in feature extraction.
std::unique_ptr<Thread> CreateFeatureImporterController(
    const std::filesystem::path& database_path,
    const ImageReaderOptions& reader_options,
    const std::filesystem::path& import_path,
    const MonocularCalibrationOptions& calibration_options,
    MonocularCalibratorFactory calibrator_factory = {});

}  // namespace colmap
