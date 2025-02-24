#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindEigenGeometry(py::module& m);
void BindRigid3(py::module& m);
void BindSim3(py::module& m);
void BindPosePrior(py::module& m);
void BindHomographyMatrixGeometry(py::module& m);
void BindEssentialMatrixGeometry(py::module& m);
void BindTriangulation(py::module& m);

void BindGeometry(py::module& m) {
  BindEigenGeometry(m);
  BindRigid3(m);
  BindSim3(m);
  BindPosePrior(m);
  BindHomographyMatrixGeometry(m);
  BindEssentialMatrixGeometry(m);
  BindTriangulation(m);
}
