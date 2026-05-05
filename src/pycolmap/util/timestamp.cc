#include "colmap/util/timestamp.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindTimestamp(py::module& m) {
  m.def("timestamp_to_seconds", &TimestampToSeconds, "timestamp_ns"_a);
  m.def("seconds_to_timestamp", &SecondsToTimestamp, "seconds"_a);
  m.def("timestamp_diff_seconds", &TimestampDiffSeconds, "t1"_a, "t0"_a);
}
