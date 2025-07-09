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

template <typename PairingOptions,
          std::unique_ptr<Thread> MatcherFactory(const PairingOptions&,
                                                 const FeatureMatchingOptions&,
                                                 const TwoViewGeometryOptions&,
                                                 const std::string&)>
void MatchFeatures(const std::string& database_path,
                   FeatureMatchingOptions matching_options,
                   const PairingOptions& pairing_options,
                   const TwoViewGeometryOptions& verification_options,
                   const Device device) {
  THROW_CHECK_FILE_EXISTS(database_path);
  try {
    py::cast(pairing_options).attr("check").attr("__call__")();
  } catch (py::error_already_set& ex) {
    // Allow pass if no check function defined.
    if (!ex.matches(PyExc_AttributeError)) {
      throw ex;
    }
  }

  matching_options.use_gpu = IsGPU(device);
  VerifyGPUParams(matching_options.use_gpu);
  py::gil_scoped_release release;
  std::unique_ptr<Thread> matcher = MatcherFactory(
      pairing_options, matching_options, verification_options, database_path);
  matcher->Start();
  PyWait(matcher.get());
}

void VerifyMatches(const std::string& database_path,
                   const std::string& pairs_path,
                   const TwoViewGeometryOptions& verification_options) {
  THROW_CHECK_FILE_EXISTS(database_path);
  THROW_CHECK_FILE_EXISTS(pairs_path);
  py::gil_scoped_release release;  // verification is multi-threaded

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;

  ImportedPairingOptions matcher_options;
  matcher_options.match_list_path = pairs_path;

  std::unique_ptr<Thread> matcher = CreateImagePairsFeatureMatcher(
      matcher_options, matching_options, verification_options, database_path);
  matcher->Start();
  PyWait(matcher.get());
}

void BindMatchFeatures(py::module& m) {
  auto PyExhaustivePairingOptions =
      py::class_<ExhaustivePairingOptions>(m, "ExhaustivePairingOptions")
          .def(py::init<>())
          .def_readwrite("block_size", &ExhaustivePairingOptions::block_size)
          .def("check", &ExhaustivePairingOptions::Check);
  MakeDataclass(PyExhaustivePairingOptions);

  auto PySpatialPairingOptions =
      py::class_<SpatialPairingOptions>(m, "SpatialPairingOptions")
          .def(py::init<>())
          .def_readwrite(
              "ignore_z",
              &SpatialPairingOptions::ignore_z,
              "Whether to ignore the Z-component of the location prior.")
          .def_readwrite("max_num_neighbors",
                         &SpatialPairingOptions::max_num_neighbors,
                         "The maximum number of nearest neighbors to match.")
          .def_readwrite(
              "min_num_neighbors",
              &SpatialPairingOptions::min_num_neighbors,
              "The minimum number of nearest neighbors to match. Neighbors "
              "include those within max_distance or to satisfy "
              "min_num_neighbors.")
          .def_readwrite("max_distance",
                         &SpatialPairingOptions::max_distance,
                         "The maximum distance between the query and nearest "
                         "neighbor [meters].")
          .def_readwrite("num_threads", &SpatialPairingOptions::num_threads)
          .def("check", &SpatialPairingOptions::Check);
  MakeDataclass(PySpatialPairingOptions);

  auto PyVocabTreePairingOptions =
      py::class_<VocabTreePairingOptions>(m, "VocabTreePairingOptions")
          .def(py::init<>())
          .def_readwrite("num_images",
                         &VocabTreePairingOptions::num_images,
                         "Number of images to retrieve for each query image.")
          .def_readwrite(
              "num_nearest_neighbors",
              &VocabTreePairingOptions::num_nearest_neighbors,
              "Number of nearest neighbors to retrieve per query feature.")
          .def_readwrite(
              "num_checks",
              &VocabTreePairingOptions::num_checks,
              "Number of nearest-neighbor checks to use in retrieval.")
          .def_readwrite(
              "num_images_after_verification",
              &VocabTreePairingOptions::num_images_after_verification,
              "How many images to return after spatial verification. Set to "
              "0 to turn off spatial verification.")
          .def_readwrite(
              "max_num_features",
              &VocabTreePairingOptions::max_num_features,
              "The maximum number of features to use for indexing an image.")
          .def_readwrite("vocab_tree_path",
                         &VocabTreePairingOptions::vocab_tree_path,
                         "Path to the vocabulary tree.")
          .def_readwrite(
              "match_list_path",
              &VocabTreePairingOptions::match_list_path,
              "Optional path to file with specific image names to match.")
          .def_readwrite("num_threads", &VocabTreePairingOptions::num_threads)
          .def("check", &VocabTreePairingOptions::Check);
  MakeDataclass(PyVocabTreePairingOptions);

  auto PySequentialPairingOptions =
      py::class_<SequentialPairingOptions>(m, "SequentialPairingOptions")
          .def(py::init<>())
          .def_readwrite("overlap",
                         &SequentialPairingOptions::overlap,
                         "Number of overlapping image pairs.")
          .def_readwrite(
              "quadratic_overlap",
              &SequentialPairingOptions::quadratic_overlap,
              "Whether to match images against their quadratic neighbors.")
          .def_readwrite("expand_rig_images",
                         &SequentialPairingOptions::expand_rig_images,
                         "Whether to match an image against all images in "
                         "neighboring rig frames. If no rigs/frames are "
                         "configured in the database, this option is ignored.")
          .def_readwrite("loop_detection",
                         &SequentialPairingOptions::loop_detection,
                         "Loop detection is invoked every "
                         "`loop_detection_period` images.")
          .def_readwrite("loop_detection_period",
                         &SequentialPairingOptions::loop_detection_period,
                         "The number of images to retrieve in loop detection. "
                         "This number should be significantly bigger than the "
                         "sequential matching overlap.")
          .def_readwrite("loop_detection_num_images",
                         &SequentialPairingOptions::loop_detection_num_images,
                         "The number of images to retrieve in loop "
                         "detection. This number should be significantly "
                         "bigger than the sequential matching overlap.")
          .def_readwrite(
              "loop_detection_num_nearest_neighbors",
              &SequentialPairingOptions::loop_detection_num_nearest_neighbors,
              "Number of nearest neighbors to retrieve per query feature.")
          .def_readwrite(
              "loop_detection_num_checks",
              &SequentialPairingOptions::loop_detection_num_checks,
              "Number of nearest-neighbor checks to use in retrieval.")
          .def_readwrite(
              "loop_detection_num_images_after_verification",
              &SequentialPairingOptions::
                  loop_detection_num_images_after_verification,
              "How many images to return after spatial verification. Set to "
              "0 to turn off spatial verification.")
          .def_readwrite(
              "loop_detection_max_num_features",
              &SequentialPairingOptions::loop_detection_max_num_features,
              "The maximum number of features to use for indexing "
              "an image. If an image has more features, only the "
              "largest-scale features will be indexed.")
          .def_readwrite("vocab_tree_path",
                         &SequentialPairingOptions::vocab_tree_path,
                         "Path to the vocabulary tree.")
          .def_readwrite(
              "num_threads",
              &SequentialPairingOptions::num_threads,
              "Number of threads for loop detection indexing and retrieval.")
          .def("vocab_tree_options",
               &SequentialPairingOptions::VocabTreeOptions)
          .def("check", &SequentialPairingOptions::Check);
  MakeDataclass(PySequentialPairingOptions);

  auto PyImportedPairingOptions =
      py::class_<ImportedPairingOptions>(m, "ImportedPairingOptions")
          .def(py::init<>())
          .def_readwrite("block_size",
                         &ImportedPairingOptions::block_size,
                         "Number of image pairs to match in one batch.")
          .def_readwrite("match_list_path",
                         &ImportedPairingOptions::match_list_path,
                         "Path to the file with the matches.")
          .def("check", &ImportedPairingOptions::Check);
  MakeDataclass(PyImportedPairingOptions);

  m.def(
      "match_exhaustive",
      &MatchFeatures<ExhaustivePairingOptions, CreateExhaustiveFeatureMatcher>,
      "database_path"_a,
      py::arg_v("matching_options",
                FeatureMatchingOptions(),
                "FeatureMatchingOptions()"),
      py::arg_v("pairing_options",
                ExhaustivePairingOptions(),
                "ExhaustivePairingOptions()"),
      py::arg_v("verification_options",
                TwoViewGeometryOptions(),
                "TwoViewGeometryOptions()"),
      "device"_a = Device::AUTO,
      "Exhaustive feature matching");

  m.def("match_spatial",
        &MatchFeatures<SpatialPairingOptions, CreateSpatialFeatureMatcher>,
        "database_path"_a,
        py::arg_v("matching_options",
                  FeatureMatchingOptions(),
                  "FeatureMatchingOptions()"),
        py::arg_v("pairing_options",
                  SpatialPairingOptions(),
                  "SpatialPairingOptions()"),
        py::arg_v("verification_options",
                  TwoViewGeometryOptions(),
                  "TwoViewGeometryOptions()"),
        "device"_a = Device::AUTO,
        "Spatial feature matching");

  m.def("match_vocabtree",
        &MatchFeatures<VocabTreePairingOptions, CreateVocabTreeFeatureMatcher>,
        "database_path"_a,
        py::arg_v("matching_options",
                  FeatureMatchingOptions(),
                  "FeatureMatchingOptions()"),
        py::arg_v("pairing_options",
                  VocabTreePairingOptions(),
                  "VocabTreePairingOptions()"),
        py::arg_v("verification_options",
                  TwoViewGeometryOptions(),
                  "TwoViewGeometryOptions()"),
        "device"_a = Device::AUTO,
        "Vocab tree feature matching");

  m.def(
      "match_sequential",
      &MatchFeatures<SequentialPairingOptions, CreateSequentialFeatureMatcher>,
      "database_path"_a,
      py::arg_v("matching_options",
                FeatureMatchingOptions(),
                "FeatureMatchingOptions()"),
      py::arg_v("pairing_options",
                SequentialPairingOptions(),
                "SequentialPairingOptions()"),
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
      .def(py::init<const ExhaustivePairingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<VocabTreePairGenerator, PairGenerator>(m, "VocabTreePairGenerator")
      .def(py::init<const VocabTreePairingOptions&,
                    const std::shared_ptr<Database>&,
                    const std::vector<image_t>&>(),
           "options"_a,
           "database"_a,
           "query_image_ids"_a = std::vector<image_t>());
  py::class_<SequentialPairGenerator, PairGenerator>(m,
                                                     "SequentialPairGenerator")
      .def(py::init<const SequentialPairingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<SpatialPairGenerator, PairGenerator>(m, "SpatialPairGenerator")
      .def(py::init<const SpatialPairingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
  py::class_<ImportedPairGenerator, PairGenerator>(m, "ImportedPairGenerator")
      .def(py::init<const ImportedPairingOptions&,
                    const std::shared_ptr<Database>&>(),
           "options"_a,
           "database"_a);
}
