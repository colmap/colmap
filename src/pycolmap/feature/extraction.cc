#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/aliked.h"
#endif

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <memory>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

inline static constexpr int kKeypointDim = 4;

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename dtype>
using pyimage_t =
    Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    descriptors_t;
typedef Eigen::Matrix<float, Eigen::Dynamic, kKeypointDim, Eigen::RowMajor>
    keypoints_t;
typedef std::tuple<keypoints_t, descriptors_t> sift_output_t;

static std::map<int, std::unique_ptr<std::mutex>> sift_gpu_mutexes;

// Rescale bitmap if it exceeds max_image_size. Returns the scale factor used.
double MaybeRescaleBitmap(Bitmap& bitmap, int max_image_size) {
  const int width = bitmap.Width();
  const int height = bitmap.Height();
  if (width <= max_image_size && height <= max_image_size) {
    return 1.0;
  }
  const double scale =
      static_cast<double>(max_image_size) / std::max(width, height);
  const int new_width = static_cast<int>(width * scale);
  const int new_height = static_cast<int>(height * scale);
  bitmap.Rescale(new_width, new_height);
  return scale;
}

// Convert FeatureKeypoints to keypoints_t, scaling coordinates by inv_scale.
keypoints_t ConvertKeypoints(const FeatureKeypoints& feature_keypoints,
                             double inv_scale) {
  const size_t num_features = feature_keypoints.size();
  keypoints_t keypoints(num_features, kKeypointDim);
  for (size_t i = 0; i < num_features; ++i) {
    keypoints(i, 0) = feature_keypoints[i].x * inv_scale;
    keypoints(i, 1) = feature_keypoints[i].y * inv_scale;
    keypoints(i, 2) = feature_keypoints[i].ComputeScale() * inv_scale;
    keypoints(i, 3) = feature_keypoints[i].ComputeOrientation();
  }
  return keypoints;
}

class Sift {
 public:
  Sift(std::optional<FeatureExtractionOptions> options, Device device)
      : use_gpu_(IsGPU(device)) {
    if (options) {
      options_ = std::move(*options);
    }
    options_.use_gpu = use_gpu_;
    THROW_CHECK(options_.Check());
    extractor_ = THROW_CHECK_NOTNULL(CreateSiftFeatureExtractor(options_));
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image) {
    Bitmap bitmap(image.cols(), image.rows(), /*as_rgb=*/false);
    std::memcpy(bitmap.RowMajorData().data(), image.data(), bitmap.NumBytes());

    const double scale = MaybeRescaleBitmap(bitmap, options_.EffMaxImageSize());

    FeatureKeypoints feature_keypoints;
    FeatureDescriptors feature_descriptors;
    THROW_CHECK(
        extractor_->Extract(bitmap, &feature_keypoints, &feature_descriptors));

    keypoints_t keypoints = ConvertKeypoints(feature_keypoints, 1.0 / scale);
    descriptors_t descriptors = feature_descriptors.ToFloat().data;
    descriptors /= 512.0f;

    return std::make_tuple(std::move(keypoints), std::move(descriptors));
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<float>>& image) {
    const pyimage_t<uint8_t> image_f = (image * 255.0f).cast<uint8_t>();
    return Extract(image_f);
  }

  const FeatureExtractionOptions& Options() const { return options_; };

  Device GetDevice() const { return (use_gpu_) ? Device::CUDA : Device::CPU; };

 private:
  std::unique_ptr<FeatureExtractor> extractor_;
  FeatureExtractionOptions options_;
  bool use_gpu_ = false;
};

#ifdef COLMAP_ONNX_ENABLED
typedef std::tuple<keypoints_t, descriptors_t> aliked_output_t;

class Aliked {
 public:
  Aliked(std::optional<FeatureExtractionOptions> options, Device device)
      : use_gpu_(IsGPU(device)) {
    if (options) {
      options_ = std::move(*options);
    } else {
      options_.type = FeatureExtractorType::ALIKED_N16ROT;
    }
    options_.use_gpu = use_gpu_;
    THROW_CHECK(options_.Check());
    extractor_ = THROW_CHECK_NOTNULL(CreateAlikedFeatureExtractor(options_));
  }

  aliked_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image) {
    THROW_CHECK(image.rows() > 0 && image.cols() > 0);
    // For RGB images, cols = width * 3
    THROW_CHECK_EQ(image.cols() % 3, 0);
    const int width = image.cols() / 3;

    Bitmap bitmap(width, image.rows(), /*as_rgb=*/true);
    std::memcpy(bitmap.RowMajorData().data(), image.data(), bitmap.NumBytes());

    const double scale = MaybeRescaleBitmap(bitmap, options_.EffMaxImageSize());

    FeatureKeypoints feature_keypoints;
    FeatureDescriptors feature_descriptors;
    THROW_CHECK(
        extractor_->Extract(bitmap, &feature_keypoints, &feature_descriptors));

    keypoints_t keypoints = ConvertKeypoints(feature_keypoints, 1.0 / scale);
    descriptors_t descriptors = feature_descriptors.ToFloat().data;

    return std::make_tuple(std::move(keypoints), std::move(descriptors));
  }

  aliked_output_t Extract(const Eigen::Ref<const pyimage_t<float>>& image) {
    const pyimage_t<uint8_t> image_u8 = (image * 255.0f).cast<uint8_t>();
    return Extract(image_u8);
  }

  const FeatureExtractionOptions& Options() const { return options_; };

  Device GetDevice() const { return (use_gpu_) ? Device::CUDA : Device::CPU; };

 private:
  std::unique_ptr<FeatureExtractor> extractor_;
  FeatureExtractionOptions options_;
  bool use_gpu_ = false;
};
#endif

void BindFeatureExtraction(py::module& m) {
  auto PyNormalization =
      py::enum_<SiftExtractionOptions::Normalization>(m, "Normalization")
          .value("L1_ROOT",
                 SiftExtractionOptions::Normalization::L1_ROOT,
                 "L1-normalizes each descriptor followed by element-wise "
                 "square rooting. This normalization is usually better than "
                 "standard "
                 "L2-normalization. See 'Three things everyone should know "
                 "to improve object retrieval', Relja Arandjelovic and "
                 "Andrew Zisserman, CVPR 2012.")
          .value("L2",
                 SiftExtractionOptions::Normalization::L2,
                 "Each vector is L2-normalized.");
  AddStringToEnumConstructor(PyNormalization);

  auto PySiftExtractionOptions =
      py::classh<SiftExtractionOptions>(m, "SiftExtractionOptions")
          .def(py::init<>())
          .def_readwrite("max_num_features",
                         &SiftExtractionOptions::max_num_features,
                         "Maximum number of features to detect, keeping "
                         "larger-scale features.")
          .def_readwrite("first_octave",
                         &SiftExtractionOptions::first_octave,
                         "First octave in the pyramid, i.e. -1 upsamples the "
                         "image by one level.")
          .def_readwrite("num_octaves", &SiftExtractionOptions::num_octaves)
          .def_readwrite("octave_resolution",
                         &SiftExtractionOptions::octave_resolution,
                         "Number of levels per octave.")
          .def_readwrite("peak_threshold",
                         &SiftExtractionOptions::peak_threshold,
                         "Peak threshold for detection.")
          .def_readwrite("edge_threshold",
                         &SiftExtractionOptions::edge_threshold,
                         "Edge threshold for detection.")
          .def_readwrite("estimate_affine_shape",
                         &SiftExtractionOptions::estimate_affine_shape,
                         "Estimate affine shape of SIFT features in the form "
                         "of oriented ellipses as opposed to original SIFT "
                         "which estimates oriented disks.")
          .def_readwrite("max_num_orientations",
                         &SiftExtractionOptions::max_num_orientations,
                         "Maximum number of orientations per keypoint if not "
                         "estimate_affine_shape.")
          .def_readwrite("upright",
                         &SiftExtractionOptions::upright,
                         "Fix the orientation to 0 for upright features")
          .def_readwrite("darkness_adaptivity",
                         &SiftExtractionOptions::darkness_adaptivity,
                         "Whether to adapt the feature detection depending "
                         "on the image darkness. only available on GPU.")
          .def_readwrite(
              "domain_size_pooling",
              &SiftExtractionOptions::domain_size_pooling,
              "\"Domain-Size Pooling in Local Descriptors and Network"
              "Architectures\", J. Dong and S. Soatto, CVPR 2015")
          .def_readwrite("dsp_min_scale", &SiftExtractionOptions::dsp_min_scale)
          .def_readwrite("dsp_max_scale", &SiftExtractionOptions::dsp_max_scale)
          .def_readwrite("dsp_num_scales",
                         &SiftExtractionOptions::dsp_num_scales)
          .def_readwrite("normalization",
                         &SiftExtractionOptions::normalization,
                         "L1_ROOT or L2 descriptor normalization")
          .def("check", &SiftExtractionOptions::Check);
  MakeDataclass(PySiftExtractionOptions);

#ifdef COLMAP_ONNX_ENABLED
  auto PyAlikedExtractionOptions =
      py::classh<AlikedExtractionOptions>(m, "AlikedExtractionOptions")
          .def(py::init<>())
          .def_readwrite("max_num_features",
                         &AlikedExtractionOptions::max_num_features,
                         "Maximum number of features to detect, keeping "
                         "higher-score features.")
          .def_readwrite("min_score",
                         &AlikedExtractionOptions::min_score,
                         "Minimum score threshold for keypoint detection.")
          .def_readwrite("n16rot_model_path",
                         &AlikedExtractionOptions::n16rot_model_path,
                         "Path to the ONNX model file for the n16rot ALIKED "
                         "extractor.")
          .def_readwrite("n32_model_path",
                         &AlikedExtractionOptions::n32_model_path,
                         "Path to the ONNX model file for the n32 ALIKED "
                         "extractor.")
          .def("check", &AlikedExtractionOptions::Check);
  MakeDataclass(PyAlikedExtractionOptions);
#endif

  auto PyFeatureExtractionOptions =
      py::classh<FeatureExtractionOptions>(m, "FeatureExtractionOptions")
          .def(py::init<>())
          .def_readwrite(
              "max_image_size",
              &FeatureExtractionOptions::max_image_size,
              "Maximum image size, otherwise image will be down-scaled. If "
              "max_image_size is non-positive, the appropriate size is "
              "selected automatically based on the extractor type.")
          .def_readwrite("num_threads",
                         &FeatureExtractionOptions::num_threads,
                         "Number of threads for feature matching and "
                         "geometric verification.")
          .def_readwrite("use_gpu", &FeatureExtractionOptions::use_gpu)
          .def_readwrite("gpu_index",
                         &FeatureExtractionOptions::gpu_index,
                         "Index of the GPU used for feature matching. For "
                         "multi-GPU matching, you should separate multiple "
                         "GPU indices by comma, e.g., '0,1,2,3'.")
          .def_readwrite("sift", &FeatureExtractionOptions::sift)

          .def("requires_rgb", &FeatureExtractionOptions::RequiresRGB)
          .def("requires_opengl", &FeatureExtractionOptions::RequiresOpenGL)
          .def("eff_max_image_size", &FeatureExtractionOptions::EffMaxImageSize)
          .def("check", &FeatureExtractionOptions::Check);
#ifdef COLMAP_ONNX_ENABLED
  PyFeatureExtractionOptions.def_readwrite("aliked",
                                           &FeatureExtractionOptions::aliked);
#endif
  MakeDataclass(PyFeatureExtractionOptions);

  py::classh<Sift>(m, "Sift")
      .def(py::init<std::optional<FeatureExtractionOptions>, Device>(),
           "options"_a = std::nullopt,
           "device"_a = Device::AUTO)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<uint8_t>>&>(
               &Sift::Extract),
           "image"_a.noconvert())
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<float>>&>(
               &Sift::Extract),
           "image"_a.noconvert())
      .def_property_readonly("options", &Sift::Options)
      .def_property_readonly("device", &Sift::GetDevice);

#ifdef COLMAP_ONNX_ENABLED
  py::classh<Aliked>(m, "Aliked")
      .def(py::init<std::optional<FeatureExtractionOptions>, Device>(),
           "options"_a = std::nullopt,
           "device"_a = Device::AUTO)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<uint8_t>>&>(
               &Aliked::Extract),
           "image"_a.noconvert(),
           "Extract ALIKED features from an RGB image. The image should be "
           "passed as a 2D array with shape (height, width * 3) where the "
           "last dimension contains interleaved RGB values.")
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<float>>&>(
               &Aliked::Extract),
           "image"_a.noconvert(),
           "Extract ALIKED features from an RGB image with float values in "
           "[0, 1] range.")
      .def_property_readonly("options", &Aliked::Options)
      .def_property_readonly("device", &Aliked::GetDevice);
#endif
}
