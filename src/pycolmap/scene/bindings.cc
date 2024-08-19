#include "pycolmap/scene/types.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindCamera(py::module& m);
void BindCorrespondenceGraph(py::module& m);
void BindDatabase(py::module& m);
void BindDatabaseCache(py::module& m);
void BindImage(py::module& m);
void BindPoint2D(py::module& m);
void BindPoint3D(py::module& m);
void BindReconstruction(py::module& m);
void BindReconstructionManager(py::module& m);
void BindTrack(py::module& m);
void BindTwoViewGeometryScene(py::module& m);

void BindScene(py::module& m) {
  BindPoint2D(m);
  BindImage(m);
  BindCamera(m);
  BindTrack(m);
  BindPoint3D(m);
  BindCorrespondenceGraph(m);
  BindReconstruction(m);
  BindReconstructionManager(m);
  BindTwoViewGeometryScene(m);
  BindDatabase(m);
  BindDatabaseCache(m);

  py::implicitly_convertible<py::iterable, Point2DVector>();
}
