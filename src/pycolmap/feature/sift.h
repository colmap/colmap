#pragma once

#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <memory>

#include <Eigen/Core>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define kdim 4

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename dtype>
using pyimage_t =
    Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    descriptors_t;
typedef Eigen::Matrix<float, Eigen::Dynamic, kdim, Eigen::RowMajor> keypoints_t;
typedef std::tuple<keypoints_t, descriptors_t> sift_output_t;

static std::map<int, std::unique_ptr<std::mutex>> sift_gpu_mutexes;

class Sift {
 public:
  Sift(SiftExtractionOptions options, Device device)
      : options_(std::move(options)), use_gpu_(IsGPU(device)) {
    VerifyGPUParams(use_gpu_);
    options_.use_gpu = use_gpu_;
    extractor_ = CreateSiftFeatureExtractor(options_);
    THROW_CHECK(extractor_ != nullptr);
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image) {
    THROW_CHECK_LE(image.rows(), options_.max_image_size);
    THROW_CHECK_LE(image.cols(), options_.max_image_size);

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

    keypoints_t keypoints(num_features, kdim);
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

  const SiftExtractionOptions& Options() const { return options_; };

  Device GetDevice() const { return (use_gpu_) ? Device::CUDA : Device::CPU; };

 private:
  std::unique_ptr<FeatureExtractor> extractor_;
  SiftExtractionOptions options_;
  bool use_gpu_ = false;
};

void BindSift(py::module& m) {
  // For backwards consistency
  py::dict sift_options;
  sift_options["peak_threshold"] = 0.01;
  sift_options["first_octave"] = 0;
  sift_options["max_image_size"] = 7000;

  py::class_<Sift>(m, "Sift")
      .def(py::init<SiftExtractionOptions, Device>(),
           "options"_a = sift_options,
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
