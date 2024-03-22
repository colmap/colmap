#include "colmap/math/random.h"
#include "colmap/util/version.h"

#include "pycolmap/helpers.h"
#include "pycolmap/logging.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/timer.h"
#include "pycolmap/utils.h"

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
namespace py = pybind11;

void BindEstimators(py::module& m);
void BindGeometry(py::module& m);
void BindOptim(py::module& m);
void BindPipeline(py::module& m);
void BindScene(py::module& m);
void BindSfMObjects(py::module& m);
void BindSift(py::module& m);

PYBIND11_MODULE(pycolmap, m) {
  m.doc() = "COLMAP plugin";
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("dev");
#endif
  m.attr("has_cuda") = IsGPU(Device::AUTO);
  m.attr("COLMAP_version") = py::str(GetVersionInfo());
  m.attr("COLMAP_build") = py::str(GetBuildInfo());

  auto PyDevice = py::enum_<Device>(m, "Device")
                      .value("auto", Device::AUTO)
                      .value("cpu", Device::CPU)
                      .value("cuda", Device::CUDA);
  AddStringToEnumConstructor(PyDevice);

  BindLogging(m);
  BindTimer(m);
  BindGeometry(m);
  BindOptim(m);
  BindScene(m);
  BindEstimators(m);
  BindSfMObjects(m);
  BindSift(m);
  BindPipeline(m);

  m.def("set_random_seed",
        &SetPRNGSeed,
        "Initialize the PRNG with the given seed.");

  py::add_ostream_redirect(m, "ostream");
}
