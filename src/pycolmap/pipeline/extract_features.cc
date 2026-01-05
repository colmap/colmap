#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/sfm.h"
#include "colmap/feature/sift.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <filesystem>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void ExtractFeatures(const std::filesystem::path& database_path,
                     const std::filesystem::path& image_path,
                     const std::vector<std::string>& image_names,
                     const CameraMode camera_mode,
                     ImageReaderOptions reader_options,
                     FeatureExtractionOptions extraction_options,
                     const Device device) {
  THROW_CHECK_DIR_EXISTS(image_path.string());
  extraction_options.use_gpu = IsGPU(device);
  THROW_CHECK(extraction_options.Check());

  UpdateImageReaderOptionsFromCameraMode(reader_options, camera_mode);
  reader_options.image_path = image_path;

  if (!image_names.empty()) {
    reader_options.image_names = image_names;
  }

  THROW_CHECK(ExistsCameraModelWithName(reader_options.camera_model));
  THROW_CHECK(VerifyCameraParams(reader_options.camera_model,
                                 reader_options.camera_params))
      << "Invalid camera parameters.";

  py::gil_scoped_release release;
  std::unique_ptr<Thread> extractor = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  extractor->Start();
  PyWait(extractor.get());
}

void BindExtractFeatures(py::module& m) {
  m.def(
      "extract_features",
      &ExtractFeatures,
      "database_path"_a,
      "image_path"_a,
      "image_names"_a = std::vector<std::string>(),
      "camera_mode"_a = CameraMode::AUTO,
      py::arg_v("reader_options", ImageReaderOptions(), "ImageReaderOptions()"),
      py::arg_v("extraction_options",
                FeatureExtractionOptions(),
                "FeatureExtractionOptions()"),
      "device"_a = Device::AUTO,
      "Extract SIFT Features and write them to database");
}
