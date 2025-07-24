#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindFeatureMatching(py::module& m) {
  auto PySiftMatchingOptions =
      py::class_<SiftMatchingOptions, std::shared_ptr<SiftMatchingOptions>>(
          m, "SiftMatchingOptions")
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
      py::class_<FeatureMatchingOptions,
                 std::shared_ptr<FeatureMatchingOptions>>(
          m, "FeatureMatchingOptions")
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
          .def_readwrite("sift", &FeatureMatchingOptions::sift)
          .def("check", &FeatureMatchingOptions::Check);
  MakeDataclass(PyFeatureMatchingOptions);
}
