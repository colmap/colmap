// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace colmap {

#ifdef COLMAP_DOWNLOAD_ENABLED
// Exported from AnyCalib v1.0.0 (upstream commit 027a8497) at its square
// training resolution using `scripts/anycalib/export_onnx.py`.
inline const std::string kDefaultAnyCalibGenUri =
    "https://github.com/colmap/colmap/releases/download/"
    "models-anycalib-v1.0.0/anycalib_gen_v1.0.0_322x322.onnx;"
    "anycalib_gen_v1.0.0_322x322.onnx;"
    "a3afb0d236bb5021c6c34dda6dcf5377687d39c541d06b779d420aca2e28e0ff";

// Exported from GeoCalib v1.0.0 (distorted weights) using
// `scripts/geocalib/export_onnx.py`.
inline const std::string kDefaultGeoCalibUri =
    "https://github.com/colmap/colmap/releases/download/"
    "models-geocalib-v1.0.0/geocalib_distorted_v1.0.0.onnx;"
    "geocalib_distorted_v1.0.0.onnx;"
    "92c78e8bcb909bb68a0674d0bcf3db503bf9eaf8ae4b5dda74beee34e1d83a9e";
#else
inline const std::string kDefaultAnyCalibGenUri = "";
inline const std::string kDefaultGeoCalibUri = "";
#endif

}  // namespace colmap
