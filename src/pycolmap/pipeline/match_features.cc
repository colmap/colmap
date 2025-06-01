#include "colmap/controllers/feature_matching.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/sfm.h"
#include "colmap/feature/pairing.h"
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

void VerifyMatches(const std::string& database_path,
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
  using EMOpts = ExhaustiveMatchingOptions;
  auto PyExhaustiveMatchingOptions =
      py::class_<ExhaustiveMatchingOptions>(m, "ExhaustiveMatchingOptions")
          .def(py::init<>())
          .def_readwrite("block_size", &EMOpts::block_size)
          .def("check", &EMOpts::Check);
  MakeDataclass(PyExhaustiveMatchingOptions);

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
          .def_readwrite("num_threads", &SpMOpts::num_threads)
          .def("check", &SpMOpts::Check);
  MakeDataclass(PySpatialMatchingOptions);

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
          .def("check", &VTMOpts::Check);
  MakeDataclass(PyVocabTreeMatchingOptions);

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
          .def_readwrite("expand_rig_images",
                         &SeqMOpts::expand_rig_images,
                         "Whether to match an image against all images in "
                         "neighboring rig frames. If no rigs/frames are "
                         "configured in the database, this option is ignored.")
          .def_readwrite("loop_detection",
                         &SeqMOpts::loop_detection,
                         "Loop detection is invoked every "
                         "`loop_detection_period` images.")
          .def_readwrite("loop_detection_period",
                         &SeqMOpts::loop_detection_period,
                         "The number of images to retrieve in loop detection. "
                         "This number should be significantly bigger than the "
                         "sequential matching overlap.")
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
          .def_readwrite(
              "num_threads",
              &SeqMOpts::num_threads,
              "Number of threads for loop detection indexing and retrieval.")
          .def("vocab_tree_options", &SeqMOpts::VocabTreeOptions)
          .def("check", &SeqMOpts::Check);
  MakeDataclass(PySequentialMatchingOptions);

  using IPMOpts = ImagePairsMatchingOptions;
  auto PyImagePairsMatchingOptions =
      py::class_<IPMOpts>(m, "ImagePairsMatchingOptions")
          .def(py::init<>())
          .def_readwrite("block_size",
                         &IPMOpts::block_size,
                         "Number of image pairs to match in one batch.")
          .def_readwrite("match_list_path",
                         &IPMOpts::match_list_path,
                         "Path to the file with the matches.")
          .def("check", &IPMOpts::Check);
  MakeDataclass(PyImagePairsMatchingOptions);

  m.def(
      "match_exhaustive",
      &MatchFeatures<EMOpts, CreateExhaustiveFeatureMatcher>,
      "database_path"_a,
      py::arg_v("sift_options", SiftMatchingOptions(), "SiftMatchingOptions()"),
      py::arg_v("matching_options",
                ExhaustiveMatchingOptions(),
                "ExhaustiveMatchingOptions()"),
      py::arg_v("verification_options",
                TwoViewGeometryOptions(),
                "TwoViewGeometryOptions()"),
      "device"_a = Device::AUTO,
      "Exhaustive feature matching");

  m.def(
      "match_spatial",
      &MatchFeatures<SpMOpts, CreateSpatialFeatureMatcher>,
      "database_path"_a,
      py::arg_v("sift_options", SiftMatchingOptions(), "SiftMatchingOptions()"),
      py::arg_v("matching_options",
                SpatialMatchingOptions(),
                "SpatialMatchingOptions()"),
      py::arg_v("verification_options",
                TwoViewGeometryOptions(),
                "TwoViewGeometryOptions()"),
      "device"_a = Device::AUTO,
      "Spatial feature matching");

  m.def(
      "match_vocabtree",
      &MatchFeatures<VTMOpts, CreateVocabTreeFeatureMatcher>,
      "database_path"_a,
      py::arg_v("sift_options", SiftMatchingOptions(), "SiftMatchingOptions()"),
      py::arg_v("matching_options",
                VocabTreeMatchingOptions(),
                "VocabTreeMatchingOptions()"),
      py::arg_v("verification_options",
                TwoViewGeometryOptions(),
                "TwoViewGeometryOptions()"),
      "device"_a = Device::AUTO,
      "Vocab tree feature matching");

  m.def(
      "match_sequential",
      &MatchFeatures<SeqMOpts, CreateSequentialFeatureMatcher>,
      "database_path"_a,
      py::arg_v("sift_options", SiftMatchingOptions(), "SiftMatchingOptions()"),
      py::arg_v("matching_options",
                SequentialMatchingOptions(),
                "SequentialMatchingOptions()"),
      py::arg_v("verification_options",
                TwoViewGeometryOptions(),
                "TwoViewGeometryOptions()"),
      "device"_a = Device::AUTO,
      "Sequential feature matching");

  m.def("verify_matches",
        &VerifyMatches,
        "database_path"_a,
        "pairs_path"_a,
        py::arg_v(
            "options", TwoViewGeometryOptions(), "TwoViewGeometryOptions()"),
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
