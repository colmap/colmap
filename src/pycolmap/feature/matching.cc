// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/aliked.h"
#include "colmap/feature/flexglue.h"
#include "colmap/feature/lightglue.h"
#include "colmap/feature/loma.h"
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
      .value("ALIKED_LIGHTGLUE", FeatureMatcherType::ALIKED_LIGHTGLUE)
      .value("LOMA_BRUTEFORCE", FeatureMatcherType::LOMA_BRUTEFORCE)
      .value("LOMA_B", FeatureMatcherType::LOMA_B)
      .value("LOMA_B128", FeatureMatcherType::LOMA_B128)
      .value("LOMA_R", FeatureMatcherType::LOMA_R)
      .value("LOMA_L", FeatureMatcherType::LOMA_L)
      .value("LOMA_G", FeatureMatcherType::LOMA_G)
      .value("FLEXGLUE", FeatureMatcherType::FLEXGLUE);

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

  auto PyLomaVariantMatcherOptions =
      py::classh<LomaMatchingOptions::Variant>(m, "LomaVariantMatcherOptions")
          .def(py::init<>())
          .def_readwrite("model_path",
                         &LomaMatchingOptions::Variant::model_path,
                         "Path to the fp32 ONNX model file for this variant.")
          .def_readwrite("model_path_bf16",
                         &LomaMatchingOptions::Variant::model_path_bf16,
                         "Path to the bf16 ONNX model file for this variant.");
  MakeDataclass(PyLomaVariantMatcherOptions);

  auto PyLomaMatchingOptions =
      py::classh<LomaMatchingOptions>(m, "LomaMatchingOptions")
          .def(py::init<>())
          .def_readwrite("min_score",
                         &LomaMatchingOptions::min_score,
                         "Matching filter, matches LG.")
          .def_readwrite("use_bf16",
                         &LomaMatchingOptions::use_bf16,
                         "Whether to use the bf16 matcher variant when "
                         "supported by the selected ONNX execution provider "
                         "(falls back to fp32 otherwise).")
          .def_readwrite("b", &LomaMatchingOptions::b, "LOMA_B model paths.")
          .def_readwrite(
              "b128", &LomaMatchingOptions::b128, "LOMA_B128 model paths.")
          .def_readwrite("r", &LomaMatchingOptions::r, "LOMA_R model paths.")
          .def_readwrite("l", &LomaMatchingOptions::l, "LOMA_L model paths.")
          .def_readwrite("g", &LomaMatchingOptions::g, "LOMA_G model paths.")
          .def_readwrite("brute_force",
                         &LomaMatchingOptions::brute_force,
                         "Brute-force matching options.")
          .def("check", &LomaMatchingOptions::Check);
  MakeDataclass(PyLomaMatchingOptions);

  auto PyFlexGlueMatchingOptions =
      py::classh<FlexGlueMatchingOptions>(m, "FlexGlueMatchingOptions")
          .def(py::init<>())
          .def_readwrite("image_path",
                         &FlexGlueMatchingOptions::image_path,
                         "Root folder of the images (required), which are "
                         "read during matching to compute the dense "
                         "features.")
          .def_readwrite("filter_threshold",
                         &FlexGlueMatchingOptions::filter_threshold,
                         "Minimum match confidence (geometric mean of the "
                         "confidences of both matching directions).")
          .def_readwrite(
              "max_distance_px",
              &FlexGlueMatchingOptions::max_distance_px,
              "Maximum distance between a keypoint and the predicted "
              "correspondence of its mutual nearest neighbor, averaged "
              "over both matching directions, in pixels at the "
              "descriptor resolution (512x512): in the original images, "
              "the threshold is scaled by the long image side / 512, "
              "averaged over both images. Non-positive values disable "
              "the check.")
          .def_readwrite("min_covisibility",
                         &FlexGlueMatchingOptions::min_covisibility,
                         "Reject all matches of a pair whose covisibility "
                         "score is below this value (0 disables it).")
          .def_readwrite("feature_cache_size",
                         &FlexGlueMatchingOptions::feature_cache_size,
                         "Maximum number of images whose dense features are "
                         "cached in memory.")
          .def_readwrite("descriptor_model_path",
                         &FlexGlueMatchingOptions::descriptor_model_path,
                         "Path to the ONNX model of the dense descriptor.")
          .def_readwrite("matcher_model_path",
                         &FlexGlueMatchingOptions::matcher_model_path,
                         "Path to the ONNX model of the matcher.")
          .def_readwrite("matcher_model_path_fp16",
                         &FlexGlueMatchingOptions::matcher_model_path_fp16,
                         "Path to the ONNX model of the matcher with float16 "
                         "transformer layers, used on CUDA GPUs.")
          .def_readwrite("use_fp16",
                         &FlexGlueMatchingOptions::use_fp16,
                         "Whether to use the float16 matcher on CUDA GPUs "
                         "(falls back to float32 otherwise).")
          .def("check", &FlexGlueMatchingOptions::Check);
  MakeDataclass(PyFlexGlueMatchingOptions);
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
  PyFeatureMatchingOptions.def_readwrite("loma", &FeatureMatchingOptions::loma);
  PyFeatureMatchingOptions.def_readwrite("flexglue",
                                         &FeatureMatchingOptions::flexglue);
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
             std::shared_ptr<const FeatureDescriptors> descriptors2,
             const std::string& name1,
             const std::string& name2) {
            FeatureMatcher::Image image1;
            if (keypoints1) {
              image1.keypoints = std::shared_ptr<const FeatureKeypoints>(
                  std::shared_ptr<void>(), keypoints1);
            }
            image1.descriptors = std::move(descriptors1);
            image1.name = name1;
            FeatureMatcher::Image image2;
            if (keypoints2) {
              image2.keypoints = std::shared_ptr<const FeatureKeypoints>(
                  std::shared_ptr<void>(), keypoints2);
            }
            image2.descriptors = std::move(descriptors2);
            image2.name = name2;
            FeatureMatches matches;
            self.Match(image1, image2, &matches);
            return MatchesToMatrix(matches);
          },
          "keypoints1"_a,
          "descriptors1"_a,
          "keypoints2"_a,
          "descriptors2"_a,
          "name1"_a = "",
          "name2"_a = "",
          "Match features between two images. Keypoints are optional. "
          "Image names (relative to the image path of the matcher options) "
          "are only required by matchers that read the image pixels, such as "
          "FlexGlue. Returns an Nx2 matrix of point2D indices.")
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
