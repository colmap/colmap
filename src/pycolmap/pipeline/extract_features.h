#pragma once

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/sfm.h"
#include "colmap/feature/sift.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <memory>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void ExtractFeatures(const std::string& database_path,
                     const std::string& image_path,
                     const std::vector<std::string>& image_list,
                     const CameraMode camera_mode,
                     const std::string& camera_model,
                     ImageReaderOptions reader_options,
                     SiftExtractionOptions sift_options,
                     const Device device) {
  THROW_CHECK_DIR_EXISTS(image_path);
  sift_options.use_gpu = IsGPU(device);
  VerifyGPUParams(sift_options.use_gpu);

  UpdateImageReaderOptionsFromCameraMode(reader_options, camera_mode);
  reader_options.camera_model = camera_model;
  reader_options.database_path = database_path;
  reader_options.image_path = image_path;

  if (!image_list.empty()) {
    reader_options.image_list = image_list;
  }

  THROW_CHECK(ExistsCameraModelWithName(reader_options.camera_model));
  THROW_CHECK(VerifyCameraParams(reader_options.camera_model,
                                 reader_options.camera_params))
      << "Invalid camera parameters.";

  py::gil_scoped_release release;
  std::unique_ptr<Thread> extractor =
      CreateFeatureExtractorController(reader_options, sift_options);
  extractor->Start();
  PyWait(extractor.get());
}

void BindExtractFeatures(py::module& m) {
  using SEOpts = SiftExtractionOptions;
  auto PyNormalization =
      py::enum_<SEOpts::Normalization>(m, "Normalization")
          .value("L1_ROOT",
                 SEOpts::Normalization::L1_ROOT,
                 "L1-normalizes each descriptor followed by element-wise "
                 "square rooting. This normalization is usually better than "
                 "standard "
                 "L2-normalization. See 'Three things everyone should know "
                 "to improve object retrieval', Relja Arandjelovic and "
                 "Andrew Zisserman, CVPR 2012.")
          .value(
              "L2", SEOpts::Normalization::L2, "Each vector is L2-normalized.");
  AddStringToEnumConstructor(PyNormalization);
  auto PySiftExtractionOptions =
      py::class_<SEOpts>(m, "SiftExtractionOptions")
          .def(py::init<>())
          .def_readwrite("num_threads",
                         &SEOpts::num_threads,
                         "Number of threads for feature matching and "
                         "geometric verification.")
          .def_readwrite("gpu_index",
                         &SEOpts::gpu_index,
                         "Index of the GPU used for feature matching. For "
                         "multi-GPU matching, you should separate multiple "
                         "GPU indices by comma, e.g., '0,1,2,3'.")
          .def_readwrite(
              "max_image_size",
              &SEOpts::max_image_size,
              "Maximum image size, otherwise image will be down-scaled.")
          .def_readwrite("max_num_features",
                         &SEOpts::max_num_features,
                         "Maximum number of features to detect, keeping "
                         "larger-scale features.")
          .def_readwrite("first_octave",
                         &SEOpts::first_octave,
                         "First octave in the pyramid, i.e. -1 upsamples the "
                         "image by one level.")
          .def_readwrite("num_octaves", &SEOpts::num_octaves)
          .def_readwrite("octave_resolution",
                         &SEOpts::octave_resolution,
                         "Number of levels per octave.")
          .def_readwrite("peak_threshold",
                         &SEOpts::peak_threshold,
                         "Peak threshold for detection.")
          .def_readwrite("edge_threshold",
                         &SEOpts::edge_threshold,
                         "Edge threshold for detection.")
          .def_readwrite("estimate_affine_shape",
                         &SEOpts::estimate_affine_shape,
                         "Estimate affine shape of SIFT features in the form "
                         "of oriented ellipses as opposed to original SIFT "
                         "which estimates oriented disks.")
          .def_readwrite("max_num_orientations",
                         &SEOpts::max_num_orientations,
                         "Maximum number of orientations per keypoint if not "
                         "estimate_affine_shape.")
          .def_readwrite("upright",
                         &SEOpts::upright,
                         "Fix the orientation to 0 for upright features")
          .def_readwrite("darkness_adaptivity",
                         &SEOpts::darkness_adaptivity,
                         "Whether to adapt the feature detection depending "
                         "on the image darkness. only available on GPU.")
          .def_readwrite(
              "domain_size_pooling",
              &SEOpts::domain_size_pooling,
              "\"Domain-Size Pooling in Local Descriptors and Network"
              "Architectures\", J. Dong and S. Soatto, CVPR 2015")
          .def_readwrite("dsp_min_scale", &SEOpts::dsp_min_scale)
          .def_readwrite("dsp_max_scale", &SEOpts::dsp_max_scale)
          .def_readwrite("dsp_num_scales", &SEOpts::dsp_num_scales)
          .def_readwrite("normalization",
                         &SEOpts::normalization,
                         "L1_ROOT or L2 descriptor normalization");
  MakeDataclass(PySiftExtractionOptions);
  auto sift_extraction_options = PySiftExtractionOptions().cast<SEOpts>();

  /* PIPELINE */
  m.def("extract_features",
        &ExtractFeatures,
        "database_path"_a,
        "image_path"_a,
        "image_list"_a = std::vector<std::string>(),
        "camera_mode"_a = CameraMode::AUTO,
        "camera_model"_a = "SIMPLE_RADIAL",
        "reader_options"_a = ImageReaderOptions(),
        "sift_options"_a = sift_extraction_options,
        "device"_a = Device::AUTO,
        "Extract SIFT Features and write them to database");
}
