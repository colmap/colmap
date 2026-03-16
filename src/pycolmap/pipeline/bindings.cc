#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindExtractFeatures(py::module& m);
void BindImages(py::module& m);
void BindMatchFeatures(py::module& m);
#if defined(COLMAP_MVS_ENABLED)
void BindMeshing(py::module& m);
void BindMVS(py::module& m);
#endif
void BindSfM(py::module& m);

void BindPipeline(py::module& m) {
  BindImages(m);
  BindExtractFeatures(m);
  BindMatchFeatures(m);
  BindSfM(m);
#if defined(COLMAP_MVS_ENABLED)
  BindMVS(m);
  BindMeshing(m);
#endif
}
