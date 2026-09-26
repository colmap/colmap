// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/calibration/calibrator.h"

#include <memory>

namespace colmap {

// EXIF backend: reads the focal length from the image EXIF tags and writes it
// into the camera, setting the focal length prior flag, and populates the
// pose prior (GPS position and gravity) from the EXIF tags. Images without an
// EXIF focal length fail calibration, keeping the existing (default)
// intrinsics. Useful as an explicit no-network baseline and as a fallback
// when no learned model is available.
std::unique_ptr<MonocularCalibrator> CreateExifCalibrator(
    const MonocularCalibrationOptions& options);

}  // namespace colmap
