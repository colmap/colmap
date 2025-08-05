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

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void ExtractFeatures(const std::string& database_path,
                     const std::string& image_path,
                     FeatureExtractionOptions extraction_options,
                     const Device device) {
  THROW_CHECK_DIR_EXISTS(image_path);
  extraction_options.use_gpu = IsGPU(device);
  THROW_CHECK(extraction_options.Check());

  py::gil_scoped_release release;
  std::unique_ptr<Thread> extractor = CreateFeatureExtractorController(
      database_path, image_path, extraction_options);
  extractor->Start();
  PyWait(extractor.get());
}

void BindExtractFeatures(py::module& m) {
  m.def("extract_features",
        &ExtractFeatures,
        "database_path"_a,
        "image_path"_a,
        py::arg_v("extraction_options",
                  FeatureExtractionOptions(),
                  "FeatureExtractionOptions()"),
        "device"_a = Device::AUTO,
        "Extract SIFT Features and write them to database");
}
