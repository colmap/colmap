#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"

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

class Sift {
 public:
  Sift(std::optional<FeatureExtractionOptions> options, Device device)
      : use_gpu_(IsGPU(device)) {
    VerifyGPUParams(use_gpu_);
    if (options) {
      options_ = std::move(*options);
    } else {
      // For backwards compatibility.
      PyErr_WarnEx(PyExc_DeprecationWarning,
                   "No SIFT extraction options specified. Setting them to "
                   "peak_threshold=0.01, first_octave=0, max_image_size=7000 "
                   "for backwards compatibility. If you want to keep the "
                   "settings, explicitly specify them, because the defaults "
                   "will change in the next major release.",
                   1);
      options_.sift->peak_threshold = 0.01;
      options_.sift->first_octave = 0;
      options_.sift->max_image_size = 7000;
    }
    options_.use_gpu = use_gpu_;
    extractor_ = THROW_CHECK_NOTNULL(CreateSiftFeatureExtractor(options_));
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image) {
    THROW_CHECK_LE(image.rows(), options_.sift->max_image_size);
    THROW_CHECK_LE(image.cols(), options_.sift->max_image_size);

    const Bitmap bitmap =
        Bitmap::ConvertFromRawBits(const_cast<uint8_t*>(image.data()),
                                   /*pitch=*/image.cols(),
                                   /*width=*/image.cols(),
                                   /*height=*/image.rows(),
                                   /*rgb=*/false);

    FeatureKeypoints keypoints_;
    FeatureDescriptors descriptors_;
    THROW_CHECK(extractor_->Extract(bitmap, &keypoints_, &descriptors_));
    const size_t num_features = keypoints_.size();

    keypoints_t keypoints(num_features, kKeypointDim);
    for (size_t i = 0; i < num_features; ++i) {
      keypoints(i, 0) = keypoints_[i].x;
      keypoints(i, 1) = keypoints_[i].y;
      keypoints(i, 2) = keypoints_[i].ComputeScale();
      keypoints(i, 3) = keypoints_[i].ComputeOrientation();
    }

    descriptors_t descriptors = descriptors_.cast<float>();
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
      py::class_<SiftExtractionOptions, std::shared_ptr<SiftExtractionOptions>>(
          m, "SiftExtractionOptions")
          .def(py::init<>())
          .def_readwrite(
              "max_image_size",
              &SiftExtractionOptions::max_image_size,
              "Maximum image size, otherwise image will be down-scaled.")
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

  auto PyFeatureExtractionOptions =
      py::class_<FeatureExtractionOptions,
                 std::shared_ptr<FeatureExtractionOptions>>(
          m, "FeatureExtractionOptions")
          .def(py::init<>())
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
          .def("check", &FeatureExtractionOptions::Check);
  MakeDataclass(PyFeatureExtractionOptions);

  py::class_<Sift>(m, "Sift")
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
