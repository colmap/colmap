#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindCamera(py::module& m);
void BindCorrespondenceGraph(py::module& m);
void BindDatabase(py::module& m);
void BindImage(py::module& m);
void BindPoint2D(py::module& m);
void BindPoint3D(py::module& m);
void BindReconstruction(py::module& m);
void BindTrack(py::module& m);

void BindScene(py::module& m) {
  BindImage(m);
  BindCamera(m);
  BindPoint2D(m);
  BindTrack(m);
  BindPoint3D(m);
  BindCorrespondenceGraph(m);
  BindReconstruction(m);
  BindDatabase(m);
}
