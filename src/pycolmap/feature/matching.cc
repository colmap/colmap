#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/aliked.h"
#include "colmap/feature/onnx_matchers.h"
#endif

#include "pycolmap/feature/types.h"
#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

class PyFeatureMatcher : public FeatureMatcher,
                         py::trampoline_self_life_support {
 public:
  using MatcherImage = FeatureMatcher::Image;

  static std::unique_ptr<FeatureMatcher> CreateOnDevice(
      std::optional<FeatureMatchingOptions> options, Device device) {
    if (options) {
      if (options->use_gpu != IsGPU(device)) {
        LOG(WARNING) << "FeatureMatchingOptions::use_gpu does not match "
                        "device. FeatureMatchingOptions::use_gpu is ignored.";
      }
    } else {
      options = FeatureMatchingOptions();
    }
    if (options->sift->cpu_descriptor_index_cache == nullptr) {
      options->sift->cpu_brute_force_matcher = true;
    }
    options->use_gpu = IsGPU(device);
    return THROW_CHECK_NOTNULL(FeatureMatcher::Create(*options));
  }

  void Match(const MatcherImage& image1,
             const MatcherImage& image2,
             FeatureMatches* matches) override {
    PYBIND11_OVERRIDE_PURE(
        void, FeatureMatcher, Match, image1, image2, matches);
  }

  void MatchGuided(double max_error,
                   const MatcherImage& image1,
                   const MatcherImage& image2,
                   TwoViewGeometry* two_view_geometry) override {
    PYBIND11_OVERRIDE_PURE(void,
                           FeatureMatcher,
                           MatchGuided,
                           max_error,
                           image1,
                           image2,
                           two_view_geometry);
  }
};

}  // namespace

void BindFeatureMatching(py::module& m) {
  py::enum_<FeatureMatcherType>(m, "FeatureMatcherType")
      .value("UNDEFINED", FeatureMatcherType::UNDEFINED)
      .value("SIFT_BRUTEFORCE", FeatureMatcherType::SIFT_BRUTEFORCE)
      .value("SIFT_LIGHTGLUE", FeatureMatcherType::SIFT_LIGHTGLUE)
      .value("ALIKED_BRUTEFORCE", FeatureMatcherType::ALIKED_BRUTEFORCE)
      .value("ALIKED_LIGHTGLUE", FeatureMatcherType::ALIKED_LIGHTGLUE);

#ifdef COLMAP_ONNX_ENABLED
  auto PyBruteForceONNXMatchingOptions =
      py::classh<BruteForceONNXMatchingOptions>(m,
                                                "BruteForceONNXMatchingOptions")
          .def(py::init<>())
          .def_readwrite("min_cossim",
                         &BruteForceONNXMatchingOptions::min_cossim,
                         "Minimum cosine similarity for a match to be "
                         "considered valid.")
          .def_readwrite("max_ratio",
                         &BruteForceONNXMatchingOptions::max_ratio,
                         "Maximum ratio for Lowe's ratio test.")
          .def_readwrite("cross_check",
                         &BruteForceONNXMatchingOptions::cross_check,
                         "Enable cross-checking (mutual nearest neighbor).")
          .def_readwrite("model_path",
                         &BruteForceONNXMatchingOptions::model_path,
                         "Path to the ONNX model file.")
          .def("check", &BruteForceONNXMatchingOptions::Check);
  MakeDataclass(PyBruteForceONNXMatchingOptions);

  auto PyLightGlueONNXMatchingOptions =
      py::classh<LightGlueONNXMatchingOptions>(m,
                                               "LightGlueONNXMatchingOptions")
          .def(py::init<>())
          .def_readwrite("min_score",
                         &LightGlueONNXMatchingOptions::min_score,
                         "Minimum match score threshold.")
          .def_readwrite("model_path",
                         &LightGlueONNXMatchingOptions::model_path,
                         "Path to the LightGlue ONNX model file.")
          .def("check", &LightGlueONNXMatchingOptions::Check);
  MakeDataclass(PyLightGlueONNXMatchingOptions);

  auto PyAlikedMatchingOptions =
      py::classh<AlikedMatchingOptions>(m, "AlikedMatchingOptions")
          .def(py::init<>())
          .def_readwrite("brute_force",
                         &AlikedMatchingOptions::brute_force,
                         "Brute-force matching options.")
          .def_readwrite("lightglue",
                         &AlikedMatchingOptions::lightglue,
                         "LightGlue matching options.")
          .def("check", &AlikedMatchingOptions::Check);
  MakeDataclass(PyAlikedMatchingOptions);
#endif

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
#ifdef COLMAP_ONNX_ENABLED
          .def_readwrite("lightglue",
                         &SiftMatchingOptions::lightglue,
                         "LightGlue matching options.")
#endif
          .def("check", &SiftMatchingOptions::Check);
  MakeDataclass(PySiftMatchingOptions);

  auto PyFeatureMatchingOptions =
      py::classh<FeatureMatchingOptions>(m, "FeatureMatchingOptions")
          .def(py::init<FeatureMatcherType>(),
               "type"_a = FeatureMatcherType::SIFT_BRUTEFORCE)
          .def_readwrite("type", &FeatureMatchingOptions::type)
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
#ifdef COLMAP_ONNX_ENABLED
  PyFeatureMatchingOptions.def_readwrite("aliked",
                                         &FeatureMatchingOptions::aliked);
#endif
  MakeDataclass(PyFeatureMatchingOptions);

  auto PyFeatureMatcherCls =
      py::classh<FeatureMatcher, PyFeatureMatcher>(m, "FeatureMatcher");

  PyFeatureMatcherCls
      .def_static("create",
                  &PyFeatureMatcher::CreateOnDevice,
                  "options"_a = std::nullopt,
                  "device"_a = Device::AUTO)
      .def(
          "match",
          [](FeatureMatcher& self,
             const FeatureKeypoints* keypoints1,
             std::shared_ptr<const FeatureDescriptors> descriptors1,
             const FeatureKeypoints* keypoints2,
             std::shared_ptr<const FeatureDescriptors> descriptors2) {
            FeatureMatcher::Image image1;
            if (keypoints1) {
              image1.keypoints = std::shared_ptr<const FeatureKeypoints>(
                  std::shared_ptr<void>(), keypoints1);
            }
            image1.descriptors = std::move(descriptors1);
            FeatureMatcher::Image image2;
            if (keypoints2) {
              image2.keypoints = std::shared_ptr<const FeatureKeypoints>(
                  std::shared_ptr<void>(), keypoints2);
            }
            image2.descriptors = std::move(descriptors2);
            FeatureMatches matches;
            self.Match(image1, image2, &matches);
            return MatchesToMatrix(matches);
          },
          "keypoints1"_a,
          "descriptors1"_a,
          "keypoints2"_a,
          "descriptors2"_a,
          "Match features between two images. Keypoints are optional. "
          "Returns an Nx2 matrix of point2D indices.")
      .def(
          "match_guided",
          [](FeatureMatcher& self,
             double max_error,
             const FeatureKeypoints& keypoints1,
             std::shared_ptr<const FeatureDescriptors> descriptors1,
             const Camera& camera1,
             const FeatureKeypoints& keypoints2,
             std::shared_ptr<const FeatureDescriptors> descriptors2,
             const Camera& camera2,
             TwoViewGeometry& two_view_geometry) {
            FeatureMatcher::Image image1;
            image1.camera = &camera1;
            image1.keypoints = std::shared_ptr<const FeatureKeypoints>(
                std::shared_ptr<void>(), &keypoints1);
            image1.descriptors = std::move(descriptors1);
            FeatureMatcher::Image image2;
            image2.camera = &camera2;
            image2.keypoints = std::shared_ptr<const FeatureKeypoints>(
                std::shared_ptr<void>(), &keypoints2);
            image2.descriptors = std::move(descriptors2);
            self.MatchGuided(max_error, image1, image2, &two_view_geometry);
          },
          "max_error"_a,
          "keypoints1"_a,
          "descriptors1"_a,
          "camera1"_a,
          "keypoints2"_a,
          "descriptors2"_a,
          "camera2"_a,
          "two_view_geometry"_a,
          "Perform guided matching using existing two-view geometry. "
          "Updates the two_view_geometry in-place.");
}
