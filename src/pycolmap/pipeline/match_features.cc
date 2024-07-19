#include "colmap/controllers/feature_matching.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/sfm.h"
#include "colmap/feature/pairing.h"
#include "colmap/feature/sift.h"
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

template <typename Opts,
          std::unique_ptr<Thread> MatcherFactory(const Opts&,
                                                 const SiftMatchingOptions&,
                                                 const TwoViewGeometryOptions&,
                                                 const std::string&)>
void MatchFeatures(const std::string& database_path,
                   SiftMatchingOptions sift_options,
                   const Opts& matching_options,
                   const TwoViewGeometryOptions& verification_options,
                   const Device device) {
  THROW_CHECK_FILE_EXISTS(database_path);
  try {
    py::cast(matching_options).attr("check").attr("__call__")();
  } catch (py::error_already_set& ex) {
    // Allow pass if no check function defined.
    if (!ex.matches(PyExc_AttributeError)) {
      throw ex;
    }
  }

  sift_options.use_gpu = IsGPU(device);
  VerifyGPUParams(sift_options.use_gpu);
  py::gil_scoped_release release;
  std::unique_ptr<Thread> matcher = MatcherFactory(
      matching_options, sift_options, verification_options, database_path);
  matcher->Start();
  PyWait(matcher.get());
}

void verify_matches(const std::string& database_path,
                    const std::string& pairs_path,
                    const TwoViewGeometryOptions& verification_options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_FILE_EXISTS(pairs_path);
  py::gil_scoped_release release;  // verification is multi-threaded

  SiftMatchingOptions sift_options;
  sift_options.use_gpu = false;

  ImagePairsMatchingOptions matcher_options;
  matcher_options.match_list_path = pairs_path;

  std::unique_ptr<Thread> matcher = CreateImagePairsFeatureMatcher(
      matcher_options, sift_options, verification_options, database_path);
  matcher->Start();
  PyWait(matcher.get());
}

void BindMatchFeatures(py::module& m) {
  using SMOpts = SiftMatchingOptions;
  auto PySiftMatchingOptions =
      py::class_<SMOpts>(m, "SiftMatchingOptions")
          .def(py::init<>())
          .def_readwrite("num_threads", &SMOpts::num_threads)
          .def_readwrite("gpu_index",
                         &SMOpts::gpu_index,
                         "Index of the GPU used for feature matching. For "
                         "multi-GPU matching, "
                         "you should separate multiple GPU indices by comma, "
                         "e.g., \"0,1,2,3\".")
          .def_readwrite(
              "max_ratio",
              &SMOpts::max_ratio,
              "Maximum distance ratio between first and second best match.")
          .def_readwrite("max_distance",
                         &SMOpts::max_distance,
                         "Maximum distance to best match.")
          .def_readwrite("cross_check",
                         &SMOpts::cross_check,
                         "Whether to enable cross checking in matching.")
          .def_readwrite("max_num_matches",
                         &SMOpts::max_num_matches,
                         "Maximum number of matches.")
          .def_readwrite("guided_matching",
                         &SMOpts::guided_matching,
                         "Whether to perform guided matching, if geometric "
                         "verification succeeds.");
  MakeDataclass(PySiftMatchingOptions);
  auto sift_matching_options = PySiftMatchingOptions().cast<SMOpts>();

  using EMOpts = ExhaustiveMatchingOptions;
  auto PyExhaustiveMatchingOptions =
      py::class_<ExhaustiveMatchingOptions>(m, "ExhaustiveMatchingOptions")
          .def(py::init<>())
          .def_readwrite("block_size", &EMOpts::block_size);
  MakeDataclass(PyExhaustiveMatchingOptions);
  auto exhaustive_options = PyExhaustiveMatchingOptions().cast<EMOpts>();

  using SeqMOpts = SequentialMatchingOptions;
  auto PySequentialMatchingOptions =
      py::class_<SeqMOpts>(m, "SequentialMatchingOptions")
          .def(py::init<>())
          .def_readwrite("overlap",
                         &SeqMOpts::overlap,
                         "Number of overlapping image pairs.")
          .def_readwrite(
              "quadratic_overlap",
              &SeqMOpts::quadratic_overlap,
              "Whether to match images against their quadratic neighbors.")
          .def_readwrite("loop_detection",
                         &SeqMOpts::loop_detection,
                         "Loop detection is invoked every "
                         "`loop_detection_period` images.")
          .def_readwrite("loop_detection_num_images",
                         &SeqMOpts::loop_detection_num_images,
                         "The number of images to retrieve in loop "
                         "detection. This number should be significantly "
                         "bigger than the sequential matching overlap.")
          .def_readwrite(
              "loop_detection_num_nearest_neighbors",
              &SeqMOpts::loop_detection_num_nearest_neighbors,
              "Number of nearest neighbors to retrieve per query feature.")
          .def_readwrite(
              "loop_detection_num_checks",
              &SeqMOpts::loop_detection_num_checks,
              "Number of nearest-neighbor checks to use in retrieval.")
          .def_readwrite(
              "loop_detection_num_images_after_verification",
              &SeqMOpts::loop_detection_num_images_after_verification,
              "How many images to return after spatial verification. Set to "
              "0 to turn off spatial verification.")
          .def_readwrite("loop_detection_max_num_features",
                         &SeqMOpts::loop_detection_max_num_features,
                         "The maximum number of features to use for indexing "
                         "an image. If an image has more features, only the "
                         "largest-scale features will be indexed.")
          .def_readwrite("vocab_tree_path",
                         &SeqMOpts::vocab_tree_path,
                         "Path to the vocabulary tree.")
          .def("vocab_tree_options", &SeqMOpts::VocabTreeOptions);
  MakeDataclass(PySequentialMatchingOptions);
  auto sequential_options = PySequentialMatchingOptions().cast<SeqMOpts>();

  using SpMOpts = SpatialMatchingOptions;
  auto PySpatialMatchingOptions =
      py::class_<SpMOpts>(m, "SpatialMatchingOptions")
          .def(py::init<>())
          .def_readwrite(
              "ignore_z",
              &SpMOpts::ignore_z,
              "Whether to ignore the Z-component of the location prior.")
          .def_readwrite("max_num_neighbors",
                         &SpMOpts::max_num_neighbors,
                         "The maximum number of nearest neighbors to match.")
          .def_readwrite("max_distance",
                         &SpMOpts::max_distance,
                         "The maximum distance between the query and nearest "
                         "neighbor [meters].")
          .def_readwrite("num_threads", &SpMOpts::num_threads);
  MakeDataclass(PySpatialMatchingOptions);
  auto spatial_options = PySpatialMatchingOptions().cast<SpMOpts>();

  using VTMOpts = VocabTreeMatchingOptions;
  auto PyVocabTreeMatchingOptions =
      py::class_<VTMOpts>(m, "VocabTreeMatchingOptions")
          .def(py::init<>())
          .def_readwrite("num_images",
                         &VTMOpts::num_images,
                         "Number of images to retrieve for each query image.")
          .def_readwrite(
              "num_nearest_neighbors",
              &VTMOpts::num_nearest_neighbors,
              "Number of nearest neighbors to retrieve per query feature.")
          .def_readwrite(
              "num_checks",
              &VTMOpts::num_checks,
              "Number of nearest-neighbor checks to use in retrieval.")
          .def_readwrite(
              "num_images_after_verification",
              &VTMOpts::num_images_after_verification,
              "How many images to return after spatial verification. Set to "
              "0 to turn off spatial verification.")
          .def_readwrite(
              "max_num_features",
              &VTMOpts::max_num_features,
              "The maximum number of features to use for indexing an image.")
          .def_readwrite("vocab_tree_path",
                         &VTMOpts::vocab_tree_path,
                         "Path to the vocabulary tree.")
          .def_readwrite(
              "match_list_path",
              &VTMOpts::match_list_path,
              "Optional path to file with specific image names to match.")
          .def_readwrite("num_threads", &VTMOpts::num_threads)
          .def("check", [](VTMOpts& self) {
            THROW_CHECK(!self.vocab_tree_path.empty())
                << "vocab_tree_path required.";
            THROW_CHECK_FILE_EXISTS(self.vocab_tree_path);
          });
  MakeDataclass(PyVocabTreeMatchingOptions);
  auto vocabtree_options = PyVocabTreeMatchingOptions().cast<VTMOpts>();

  auto verification_options =
      m.attr("TwoViewGeometryOptions")().cast<TwoViewGeometryOptions>();

  m.def("match_exhaustive",
        &MatchFeatures<EMOpts, CreateExhaustiveFeatureMatcher>,
        "database_path"_a,
        "sift_options"_a = sift_matching_options,
        "matching_options"_a = exhaustive_options,
        "verification_options"_a = verification_options,
        "device"_a = Device::AUTO,
        "Exhaustive feature matching");

  m.def("match_sequential",
        &MatchFeatures<SeqMOpts, CreateSequentialFeatureMatcher>,
        "database_path"_a,
        "sift_options"_a = sift_matching_options,
        "matching_options"_a = sequential_options,
        "verification_options"_a = verification_options,
        "device"_a = Device::AUTO,
        "Sequential feature matching");

  m.def("match_spatial",
        &MatchFeatures<SpMOpts, CreateSpatialFeatureMatcher>,
        "database_path"_a,
        "sift_options"_a = sift_matching_options,
        "matching_options"_a = spatial_options,
        "verification_options"_a = verification_options,
        "device"_a = Device::AUTO,
        "Spatial feature matching");

  m.def("match_vocabtree",
        &MatchFeatures<VTMOpts, CreateVocabTreeFeatureMatcher>,
        "database_path"_a,
        "sift_options"_a = sift_matching_options,
        "matching_options"_a = vocabtree_options,
        "verification_options"_a = verification_options,
        "device"_a = Device::AUTO,
        "Vocab tree feature matching");

  m.def("verify_matches",
        &verify_matches,
        "database_path"_a,
        "pairs_path"_a,
        "options"_a = verification_options,
        "Run geometric verification of the matches");

  py::class_<PairGenerator>(m, "PairGenerator")
      .def("reset", &PairGenerator::Reset)
      .def("has_finished", &PairGenerator::HasFinished)
      .def("next", &PairGenerator::Next)
      .def("all_pairs", &PairGenerator::AllPairs);
  py::class_<ExhaustivePairGenerator, PairGenerator>(m,
                                                     "ExhaustivePairGenerator")
      .def(py::init<const ExhaustiveMatchingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<VocabTreePairGenerator, PairGenerator>(m, "VocabTreePairGenerator")
      .def(py::init<const VocabTreeMatchingOptions&,
                    const std::shared_ptr<Database>&,
                    const std::vector<image_t>&>(),
           "options"_a,
           "database"_a,
           "query_image_ids"_a = std::vector<image_t>());
  py::class_<SequentialPairGenerator, PairGenerator>(m,
                                                     "SequentialPairGenerator")
      .def(py::init<const SequentialMatchingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<SpatialPairGenerator, PairGenerator>(m, "SpatialPairGenerator")
      .def(py::init<const SpatialMatchingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<ImportedPairGenerator, PairGenerator>(m, "ImportedPairGenerator")
      .def(py::init<const ImagePairsMatchingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
}
