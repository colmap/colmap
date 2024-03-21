#include "colmap/optim/ransac.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>

using namespace colmap;
namespace py = pybind11;

void BindOptim(py::module& m) {
  auto PyRANSACOptions =
      py::class_<RANSACOptions>(m, "RANSACOptions")
          .def(py::init<>([]() {
            RANSACOptions options;
            options.max_error = 4.0;
            options.min_inlier_ratio = 0.01;
            options.confidence = 0.9999;
            options.min_num_trials = 1000;
            options.max_num_trials = 100000;
            return options;
          }))
          .def_readwrite("max_error", &RANSACOptions::max_error)
          .def_readwrite("min_inlier_ratio", &RANSACOptions::min_inlier_ratio)
          .def_readwrite("confidence", &RANSACOptions::confidence)
          .def_readwrite("dyn_num_trials_multiplier",
                         &RANSACOptions::dyn_num_trials_multiplier)
          .def_readwrite("min_num_trials", &RANSACOptions::min_num_trials)
          .def_readwrite("max_num_trials", &RANSACOptions::max_num_trials);
  MakeDataclass(PyRANSACOptions);
}
