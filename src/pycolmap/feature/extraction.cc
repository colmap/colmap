#include "colmap/feature/extractor.h"
#include "colmap/feature/sift.h"
#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/aliked.h"
#endif

#include "pycolmap/helpers.h"
#include "pycolmap/sensor/bitmap.h"
#include "pycolmap/utils.h"

#include <algorithm>
#include <memory>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename dtype>
using pyimage_t =
    Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    descriptors_t;
typedef std::tuple<FeatureKeypointsMatrix, descriptors_t> sift_output_t;

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

namespace {

class PyFeatureExtractor : public FeatureExtractor,
                           py::trampoline_self_life_support {
 public:
  static std::unique_ptr<FeatureExtractor> CreateOnDevice(
      std::optional<FeatureExtractionOptions> options, Device device) {
    if (options) {
      if (options->use_gpu != IsGPU(device)) {
        LOG(WARNING) << "FeatureExtractionOptions::use_gpu does not match "
                        "device. FeatureExtractionOptions::use_gpu is ignored.";
      }
    } else {
      options = FeatureExtractionOptions();
    }
    options->use_gpu = IsGPU(device);
    return THROW_CHECK_NOTNULL(FeatureExtractor::Create(*options));
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    PYBIND11_OVERRIDE_PURE(
        bool, FeatureExtractor, Extract, bitmap, keypoints, descriptors);
  }
};

}  // namespace

class Sift {
 public:
  Sift(std::optional<FeatureExtractionOptions> options, Device device)
      : use_gpu_(IsGPU(device)) {
    PyErr_WarnEx(PyExc_DeprecationWarning,
                 "pycolmap.Sift is deprecated, use "
                 "pycolmap.FeatureExtractor.create() instead.",
                 1);
    if (options) {
      options_ = std::move(*options);
    }
    options_.use_gpu = use_gpu_;
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

    FeatureKeypointsMatrix keypoints = KeypointsToMatrix(feature_keypoints);
    const double inv_scale = 1.0 / scale;
    keypoints.col(0) *= inv_scale;
    keypoints.col(1) *= inv_scale;
    keypoints.col(2) *= inv_scale;
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
          .def(py::init<FeatureExtractorType>(),
               "type"_a = FeatureExtractorType::SIFT)
          .def_readwrite("type", &FeatureExtractionOptions::type)
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

  py::classh<FeatureExtractor, PyFeatureExtractor>(m, "FeatureExtractor")
      .def_static("create",
                  &PyFeatureExtractor::CreateOnDevice,
                  "options"_a = std::nullopt,
                  "device"_a = Device::AUTO)
      .def(
          "extract",
          [](FeatureExtractor& self, const Bitmap& bitmap) {
            FeatureKeypoints keypoints;
            FeatureDescriptors descriptors;
            THROW_CHECK(self.Extract(bitmap, &keypoints, &descriptors));
            return py::make_tuple(std::move(keypoints), std::move(descriptors));
          },
          "bitmap"_a,
          "Extract features from a Bitmap. Returns (FeatureKeypoints, "
          "FeatureDescriptors).")
      .def(
          "extract_from_uint8_array",
          [](FeatureExtractor& self,
             py::array_t<uint8_t, py::array::c_style> image) {
            const Bitmap bitmap = BitmapFromArray(image);
            FeatureKeypoints keypoints;
            FeatureDescriptors descriptors;
            THROW_CHECK(self.Extract(bitmap, &keypoints, &descriptors));
            return py::make_tuple(std::move(keypoints), std::move(descriptors));
          },
          "image"_a,
          "Extract features from a uint8 numpy array with shape (H, W) or "
          "(H, W, 3). Returns (FeatureKeypoints, FeatureDescriptors).")
      .def(
          "extract_from_float32_array",
          [](FeatureExtractor& self,
             py::array_t<float, py::array::c_style> image) {
            auto buf = image.request();
            std::vector<size_t> shape(buf.shape.begin(), buf.shape.end());
            py::array_t<uint8_t> image_u8(shape);
            const float* src = static_cast<const float*>(buf.ptr);
            uint8_t* dst = static_cast<uint8_t*>(image_u8.request().ptr);
            const size_t num_elements = buf.size;
            for (size_t i = 0; i < num_elements; ++i) {
              dst[i] = static_cast<uint8_t>(
                  std::clamp(src[i] * 255.0f, 0.0f, 255.0f));
            }
            const Bitmap bitmap = BitmapFromArray(image_u8);
            FeatureKeypoints keypoints;
            FeatureDescriptors descriptors;
            THROW_CHECK(self.Extract(bitmap, &keypoints, &descriptors));
            return py::make_tuple(std::move(keypoints), std::move(descriptors));
          },
          "image"_a,
          "Extract features from a float32 numpy array with values in "
          "[0, 1] and shape (H, W) or (H, W, 3). Returns "
          "(FeatureKeypoints, FeatureDescriptors).");

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
}
