#include "colmap/util/timestamp.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTimestamp(py::module& m) {
  m.def("seconds_from_timestamp", &SecondsFromTimestamp, "t"_a);
  m.def("timestamp_from_seconds", &TimestampFromSeconds, "s"_a);
  m.def("timestamp_diff_seconds", &TimestampDiffSeconds, "t1"_a, "t0"_a);
}
