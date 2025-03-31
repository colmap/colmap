#include "colmap/math/random.h"
#include "colmap/util/version.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/utils.h"

#include <ceres/version.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindEstimators(py::module& m);
void BindFeature(py::module& m);
void BindGeometry(py::module& m);
void BindImage(py::module& m);
void BindOptim(py::module& m);
void BindPipeline(py::module& m);
void BindFeature(py::module& m);
void BindRetrieval(py::module& m);
void BindScene(py::module& m);
void BindSensor(py::module& m);
void BindSfm(py::module& m);
void BindUtil(py::module& m);

PYBIND11_MODULE(_core, m) {
  m.doc() = "COLMAP plugin";
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("dev");
#endif
  m.attr("__ceres_version__") = py::str(CERES_VERSION_STRING);
  m.attr("has_cuda") = IsGPU(Device::AUTO);
  m.attr("COLMAP_version") = py::str(GetVersionInfo());
  m.attr("COLMAP_build") = py::str(GetBuildInfo());

  auto PyDevice = py::enum_<Device>(m, "Device")
                      .value("auto", Device::AUTO)
                      .value("cpu", Device::CPU)
                      .value("cuda", Device::CUDA);
  AddStringToEnumConstructor(PyDevice);

  BindUtil(m);
  BindGeometry(m);
  BindOptim(m);
  BindSensor(m);
  BindScene(m);
  BindImage(m);
  BindEstimators(m);
  BindFeature(m);
  BindRetrieval(m);
  BindSfm(m);
  BindPipeline(m);

  m.def("set_random_seed",
        &SetPRNGSeed,
        "seed"_a,
        "Initialize the PRNG with the given seed.");

  py::add_ostream_redirect(m, "ostream");
}
