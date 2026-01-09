#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"

#include "pycolmap/feature/types.h"
#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

inline static constexpr int kDescDim = 128;

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

class SiftMatcher {
 public:
  SiftMatcher(std::optional<FeatureMatchingOptions> options, Device device)
      : use_gpu_(IsGPU(device)), next_image_id_(1) {
    if (options) {
      options_ = std::move(*options);
    }

    if (!options_.sift) {
      options_.sift = std::make_shared<SiftMatchingOptions>();
    }

    options_.use_gpu = use_gpu_;

    if (!use_gpu_ && !options_.sift->cpu_brute_force_matcher) {
      THROW_CHECK(false)
          << "FAISS CPU matching (cpu_brute_force_matcher=False) is not "
             "supported in this interface. Please use "
             "cpu_brute_force_matcher=True.";
    }

    THROW_CHECK(options_.Check());
    matcher_ = THROW_CHECK_NOTNULL(FeatureMatcher::Create(options_));
  }

  PyFeatureMatches Match(
      const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors1,
      const Eigen::Ref<const FeatureDescriptorsFloat>& descriptors2,
      std::optional<image_t> image_id1 = std::nullopt,
      std::optional<image_t> image_id2 = std::nullopt) {
    THROW_CHECK_EQ(descriptors1.cols(), kDescDim);
    THROW_CHECK_EQ(descriptors2.cols(), kDescDim);

    // If ids are not provided, generate unique ids so the matcher cannot reuse
    // cached descriptors across calls.
    const image_t id1 = image_id1.value_or(next_image_id_.fetch_add(1));
    const image_t id2 = image_id2.value_or(next_image_id_.fetch_add(1));

    auto desc1 = std::make_shared<FeatureDescriptors>(
        FeatureDescriptorsToUnsignedByte(descriptors1));
    auto desc2 = std::make_shared<FeatureDescriptors>(
        FeatureDescriptorsToUnsignedByte(descriptors2));

    FeatureMatcher::Image im1, im2;
    im1.image_id = id1;
    im2.image_id = id2;
    im1.descriptors = std::move(desc1);
    im2.descriptors = std::move(desc2);

    FeatureMatches matches;
    {
      py::gil_scoped_release release;
      matcher_->Match(im1, im2, &matches);
    }
    return FeatureMatchesToMatrix(matches);
  }

  const FeatureMatchingOptions& Options() const { return options_; }
  Device GetDevice() const { return use_gpu_ ? Device::CUDA : Device::CPU; }

 private:
  std::unique_ptr<FeatureMatcher> matcher_;
  FeatureMatchingOptions options_;
  bool use_gpu_ = false;

  std::atomic<image_t> next_image_id_{1};
};

void BindFeatureMatching(py::module& m) {
  auto PySiftMatchingOptions =
      py::classh<SiftMatchingOptions>(m, "SiftMatchingOptions")
          .def(py::init<>())
          .def_readwrite(
              "max_ratio",
              &SiftMatchingOptions::max_ratio,
              "Maximum distance ratio between first and second best match.")
          .def_readwrite("max_distance",
                         &SiftMatchingOptions::max_distance,
                         "Maximum distance to best match.")
          .def_readwrite("cross_check",
                         &SiftMatchingOptions::cross_check,
                         "Whether to enable cross checking in matching.")
          .def_readwrite(
              "cpu_brute_force_matcher",
              &SiftMatchingOptions::cpu_brute_force_matcher,
              "Whether to use brute-force instead of faiss based CPU matching.")
          .def("check", &SiftMatchingOptions::Check);
  MakeDataclass(PySiftMatchingOptions);

  auto PyFeatureMatchingOptions =
      py::classh<FeatureMatchingOptions>(m, "FeatureMatchingOptions")
          .def(py::init<>())
          .def_readwrite("num_threads", &FeatureMatchingOptions::num_threads)
          .def_readwrite("use_gpu", &FeatureMatchingOptions::use_gpu)
          .def_readwrite("gpu_index",
                         &FeatureMatchingOptions::gpu_index,
                         "Index of the GPU used for feature matching. For "
                         "multi-GPU matching, "
                         "you should separate multiple GPU indices by comma, "
                         "e.g., \"0,1,2,3\".")
          .def_readwrite("max_num_matches",
                         &FeatureMatchingOptions::max_num_matches,
                         "Maximum number of matches.")
          .def_readwrite("guided_matching",
                         &FeatureMatchingOptions::guided_matching,
                         "Whether to perform guided matching, if geometric "
                         "verification succeeds.")
          .def_readwrite(
              "skip_geometric_verification",
              &FeatureMatchingOptions::skip_geometric_verification,
              "Skips the geometric verification stage and forwards matches "
              "unchanged. Ignored when guided matching is enabled, because "
              "guided matching depends on the two-view geometry produced by "
              "geometric verification.")
          .def_readwrite("rig_verification",
                         &FeatureMatchingOptions::rig_verification,
                         "Whether to perform geometric verification using rig "
                         "constraints between pairs of non-trivial frames. If "
                         "disabled, performs geometric two-view verification "
                         "for non-trivial frames without rig constraints. "
                         "Ignored when skip_geometric_verification is enabled.")
          .def_readwrite(
              "skip_image_pairs_in_same_frame",
              &FeatureMatchingOptions::skip_image_pairs_in_same_frame,
              "Whether to skip matching images within the same frame. This is "
              "useful for the case of non-overlapping cameras in a rig.")
          .def_readwrite("sift", &FeatureMatchingOptions::sift)
          .def("check", &FeatureMatchingOptions::Check);
  MakeDataclass(PyFeatureMatchingOptions);

  py::classh<SiftMatcher>(m, "SiftMatcher")
      .def(py::init<std::optional<FeatureMatchingOptions>, Device>(),
           "options"_a = std::nullopt,
           "device"_a = Device::AUTO)
      .def("match",
           &SiftMatcher::Match,
           "descriptors1"_a.noconvert(),
           "descriptors2"_a.noconvert(),
           "image_id1"_a = std::nullopt,
           "image_id2"_a = std::nullopt,
           "If you repeatedly match pairs that share an image (e.g., (A,B), "
           "(A,C), ...), providing stable image_id1/image_id2 can speed up GPU "
           "matching by allowing reuse of cached descriptors. If omitted, "
           "reuse will not occur."
           "Warning: image_id values must consistently identify the same "
           "descriptors; reusing an image_id for different descriptors can "
           "lead to incorrect matches when caching is enabled.")
      .def_property_readonly("options", &SiftMatcher::Options)
      .def_property_readonly("device", &SiftMatcher::GetDevice);
}
